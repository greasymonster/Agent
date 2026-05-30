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

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "glob": lambda **kw : run_glob(kw["pattern"]),
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
]


# Gate-1 Hard deny list — always forbidden
Deny_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

def check_deny_list(command: str) -> str | None:
    for pattern in Deny_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"

    return None

# Gate-2 Rule matching - context-dependent checks
PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     "check": lambda args: not(WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "messages": "Writing outside workspace"},
    {"tools": ["bash"],
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm", "> /etc/", "chmod 777"]),
     "messages": "Potentially destructive command"},
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["messages"]

# Gate-3 User approval - wait for confirmation after rule match
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("Allow? [Y/N]").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# Pipeline: all three gates chained
def check_permission(block) -> bool:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
        
        return True

def agnet_loop(message: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=message, tools=TOOLS, 
            max_tokens=5000,
        )

        message.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return
        
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if not check_permission(block):
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": "permission denied"})
                continue
            
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            print(f">{block.name}:")
            print(output[:200])
            results.append(
                {"type": "tool_result", 
                    "tool_use_id": block.id,
                    "content": output}
            )
        message.append({"role": "user", "content": results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
