"""
s05: TodoWrite — add a planning tool on top of s04 hooks.

  +---------+      +-------+      +------------------+
  |  User   | ---> |  LLM  | ---> | TOOL_HANDLERS    |
  | prompt  |      |       |      |  bash            |
  +---------+      +---+---+      |  read_file       |
                        ^         |  write_file      |
                        | result  |  edit_file       |
                        +---------+  glob            |
                                      todo_write ← NEW
                                   +------------------+
                                        |
                         in-memory current_todos
                                        |
                        if rounds_since_todo >= 3:
                          inject <reminder>

Changes from s04:
  + todo_write tool + run_todo_write() implementation
  + Nag reminder (inject reminder after 3 rounds without todo update)
  + SYSTEM prompt includes "plan before execute" guidance
  + rounds_since_todo counter in agent_loop
  Loop unchanged: new tool auto-dispatches via TOOL_HANDLERS.

Run: python s05_todo_write/code.py
Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""
import subprocess
import os

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

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).reslove()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path:str, limit: int=None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"error {e}"

def run_write(path:str, content:str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"error {e}"

def run_edit(path:str, old_text:str, new_text:str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    
    except Exception as e:
        return f"Error: {e}"

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:5000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

def run_todo_write(todos: list) ->str:
    global CURRENT_TODOS

    for i, t in enumerate(todos):
        if "content" not in t or "status" not in t:
            return f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] has invalid status '{t['status']}'"
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f" [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Update {len(CURRENT_TODOS)} tasks"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "glob": lambda **kw : run_glob(kw["pattern"]),
    "todo_write": lambda **kw: run_todo_write(kw["todos"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
     {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
]

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

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


rounds_since_todo = 0


def agnet_loop(messages: list):
    global rounds_since_todo
    while True:
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user", "content": "<reminder>Updata your todos.<reminder>"})
            rounds_since_todo = 0

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, 
            max_tokens=5000,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return
        
        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                continue
            
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            trigger_hooks("PostToolUse", block, output)

            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agnet_loop(history)
        response_content = history[-1]["content"]

        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        
        print()
