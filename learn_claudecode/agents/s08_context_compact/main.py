"""
s08_context_compact.py - Context Compact

Four-layer compaction pipeline inserted before LLM calls:

    L1: snip_compact      — trim middle messages when count > 50
    L2: micro_compact     — replace old tool_results with placeholders
    L3: tool_result_budget — persist large results to disk
    L4: compact_history   — LLM full summary (1 API call)

    Emergency: reactive_compact — when API still returns prompt_too_long

    ┌─────────────────────────────────────────────────────────────┐
    │  messages[]                                                 │
    │    ↓                                                        │
    │  L3 budget ─→ L1 snip ─→ L2 micro ─→ [token > threshold?]  │
    │                                      ├─ No  → LLM          │
    │                                      └─ Yes → L4 summary   │
    │                                              ↓              │
    │                                          LLM call           │
    │                                    [prompt_too_long?]        │
    │                                      └─ Yes → reactive      │
    └─────────────────────────────────────────────────────────────┘

Core principle: cheap first, expensive last.
Execution order matches CC source: budget → snip → micro → auto.

Builds on s07 (skill loading). Usage:

    python s08_context_compact/code.py
    Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""
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
from utils import * 
from tools import *

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SYSTEM = build_system()
MAX_REACTIVE_RETRIES = 1
# =====================================================================================================
# 工具调用配置部分
# =====================================================================================================

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "glob": lambda **kw : run_glob(kw["pattern"]),
    "todo_write": lambda **kw: run_todo_write(kw["todos"]),
    "task": lambda **kw: spawn_subagent(kw["description"]),
    "load_skill": lambda **kw: load_skill(kw["name"]),
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
    {"name": "task", "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)

rounds_since_todo = 0

def agnet_loop(messages: list):
    global rounds_since_todo
    while True:
        messages[:] = tool_result_budget(messages) # L3 存储大的工具调用结果
        messages[:] = snip_compact(messages) # L1 将中间过程隐藏
        messages[:] = micro_compact(messages) # L2 旧的工具调用结果就抛弃了
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user", "content": "<reminder>Updata your todos.<reminder>"})
            rounds_since_todo = 0
        
        try:
            response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
            reactive_retries = 0
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

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
            
            if block.name == "compact":
                messages[:] = compact_history(messages)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break

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
        
        else:
            messages.append({"role": "user", "content": results})
            continue
        continue

if __name__ == "__main__":
    print("s08: Context Compact — four-layer compaction pipeline")
    print("Type a question, press Enter. Type q to quit.\n")
    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
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
