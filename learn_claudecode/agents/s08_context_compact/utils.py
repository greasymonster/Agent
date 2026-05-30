try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
import time
import json
import os

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"): os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
CONTEXT_LIMIT = 50000
KEEP_RECENT = 3
PERSIST_THRESHOLD = 30000
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# =====================================================================================================
# agent行为规范hook定义部分
# =====================================================================================================
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []} 

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result

    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"] 

def permission_hook(block):
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input(" Allow?[Y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"    
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input(" Allow?[Y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    return None

def log_hook(block):
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None   

def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ Large output from {block.name}: {len(str(output))} chars\033[0m")
    return None

def context_inject_hook(query:str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages:list):
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

# =====================================================================================================
# agent skills 加载实现部分
# =====================================================================================================
def _parse_frontmatter(text: str) ->tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    
    return meta, parts[2].strip()

SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()

def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

# =====================================================================================================
# 上下文压缩部分
# =====================================================================================================
def estimate_size(msgs):
    return len(str(msgs))

# 第一级压缩， 保留前三条和后几条
def snip_compact(messages, max_messages=50):
    if len(messages) < max_messages:
        return messages
    keep_head, keep_tail = 3, max_messages - 3
    snipped = len(messages) - keep_head - keep_tail
    return messages[:keep_head] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[-keep_tail:]

# 第二级压缩，将旧的工具调用信息隐藏掉
def collect_tool_results(messages):
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))

    return blocks

def micro_compact(messages):
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT:
        return messages
    for _, _, block in tool_results[:-KEEP_RECENT]:
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages

# 第三级压缩，把工具调用信息存储在本地
def persist_large_output(tool_use_id, output):
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"
    
def tool_result_budget(messages, max_bytes=200_000):
    last = messages[-1] if messages else None
    if not last or last.get("role") != "user" or isinstance(last.get("content"), list):
        return messages
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages
    
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for _, block in ranked:
        if total <= max_bytes:
            break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD:
            continue
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    
    return messages

# 第四级压缩，调用大模型对上下文进行压缩提取摘要
def write_transcript(messages):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path

def summary_history(messages):
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
        "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation
    )
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    return "\n".join(getattr(block, "text", "") for block in response.content 
                     if getattr(block, "type", None) == "text").strip() or "(empty summary)"

def compact_history(messages):
    transcript_path = write_transcript(messages)
    print(f"[transprict saved: {transcript_path}]")
    summary = summary_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]

def reactive_compact(messages):
    transcript = write_transcript(messages)
    summary = summary_history(messages)
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[-5:]]

