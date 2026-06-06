"""
s13: Background Tasks — thread-based async execution + notification injection.

Run:  python s13_background_tasks/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s12:
  - threading.Thread for background execution
  - background_tasks dict for lifecycle tracking (bg_id, command, status)
  - background_results dict + threading.Lock for thread-safe storage
  - should_run_background: model explicit request via run_in_background param
  - is_slow_operation: fallback heuristic when model doesn't specify
  - start_background_task: dispatch to daemon thread, return bg task id
  - collect_background_results: gather completed, return as notifications
  - agent_loop: slow ops → background + placeholder, inject notifications
  - Notifications use <task_notification> format, not reused tool_use_id

Note: Teaching code keeps a basic agent loop to stay focused on background
tasks. S11's full error recovery (RecoveryState, backoff, escalation,
reactive compact, fallback model) is omitted.
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
MAX_REACTIVE_RETRIES = 1
DEFAULT_MAX_TOKENS = 8000
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
    "create_task": lambda **kw: run_create_task(kw["subject"], kw.get("description"), kw.get("blockedBy")), 
    "list_tasks": lambda **kw: run_list_tasks(),
    "get_task": lambda **kw: run_get_task(kw["task_id"]), 
    "claim_task": lambda **kw: run_claim_task(kw["task_id"]),
    "complete_task": lambda **kw: run_complete_task(kw["task_id"]),
}

TOOLS = [
    {"name": "bash", 
     "description": "Run a shell command.",
     "input_schema": {"type": "object", 
                      "properties": {"command": {"type": "string"}}, 
                      "required": ["command"]}},
    {"name": "read_file", 
     "description": "Read file contents.",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}, ""
                      "limit": {"type": "integer"}}, 
                      "required": ["path"]}},
    {"name": "write_file", 
     "description": "Write content to file.",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}, 
                        "content": {"type": "string"}}, 
                        "required": ["path", "content"]}},
    {"name": "edit_file", 
     "description": "Replace exact text in file.",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}, 
                                     "old_text": {"type": "string"}, 
                                     "new_text": {"type": "string"}}, 
                                     "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", 
     "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", 
                      "properties": {"pattern": {"type": "string"}}, 
                      "required": ["pattern"]}},
    {"name": "todo_write", 
     "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", 
                      "properties": {"todos": {"type": "array", 
                                               "items": {"type": "object", 
                                                         "properties": {"content": {"type": "string"}, 
                                                                        "status": {"type": "string", 
                                                                                   "enum": ["pending", "in_progress", "completed"]}}, 
                                                                                   "required": ["content", "status"]}}}, 
                      "required": ["todos"]}},
    {"name": "task", 
     "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", 
                     "properties": {"description": {"type": "string"}}, 
                                    "required": ["description"]}},
    {"name": "load_skill", 
     "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", 
                      "properties": {"name": {"type": "string"}}, 
                                    "required": ["name"]}},
      {"name": "create_task",
     "description": "Create a new task with optional blockedBy dependencies.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task. Sets owner, changes status to in_progress.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
]

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def update_context(context: dict, messages: list)-> dict:
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }

def execute_tool(block) -> str:
    handler = TOOL_HANDLERS.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"

def start_background_task(block) -> str:
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result
    
    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running"
        }
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id

# =====================================================================================================
# 主循环部分
# =====================================================================================================
rounds_since_todo = 0
def agent_loop(messages: list, context: dict):
    global rounds_since_todo

    SYSTEM = get_system_prompt(context=context)
    reactive_retries = 0
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS
    memories_content = load_memories(messages)
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None

    while True:
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role", ""), 
                                                       "content": str(m.get("content", ""))} for m in messages]
        
        messages[:] = tool_result_budget(messages) # L3 存储大的工具调用结果
        messages[:] = snip_compact(messages) # L1 将中间过程隐藏
        messages[:] = micro_compact(messages) # L2 旧的工具调用结果就抛弃了
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user", "content": "<reminder>Update your todos.<reminder>"})
            rounds_since_todo = 0
        
        try:
            request_messages = messages
            if memories_content and memory_turn is not None and memory_turn < len(messages):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }
            response = with_retry(
                lambda mt=max_tokens, mdl=state.current_model:
                    client.messages.create(model=mdl, 
                                           system=SYSTEM, 
                                           messages=request_messages, 
                                           tools=TOOLS, 
                                           max_tokens=mt), state
            )
            reactive_retries = 0
        except Exception as e:
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    messages[:] = reactive_compact_error(messages)
                    state.has_attempted_reactive_compact = True
                    continue
                print(" \033[31m[unrecoverable] still too long after compact\033[0m")
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "[Error] Context too large, cannot continue."
                    }]
                })
                return
            
            name = type(e).__name__
            print(f" \033[31m [unrecoverable] {name}: {str(e)[:100]}\033[0m")
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"[Error] {name}: {str(e)[:200]}"
                }]
            })
            return
        
        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f" \033[33m[max_tokens] escalating"
                      f"{DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            messages.append({"role": "assistant", "content": response.content})
            
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f" \033[33m[max_tokens] continuation"
                      f" {state.recovery_count} / {MAX_RECOVERY_RETRIES}\033[0m")
                continue
            print(" \033[31m[max_tokens] recovery limit reached\033[0m")
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            extract_memories(pre_compress)
            consolidate_memories()
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
            
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content":  f"[Background task {bg_id} started] "
                                f"Command: {block.input.get('command', '')}. "
                                f"Result will be available when complete."
                })
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
        
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                results.append({"type": "text", "text": notif})
            print(f"  \033[32m[inject] {len(bg_notifications)} background "
                f"notification(s)\033[0m")
            
        messages.append({"role": "user", "content": results})
        context = update_context(context, messages)
        SYSTEM = get_system_prompt(context)
        continue

    

if __name__ == "__main__":
    print("s13: background task")
    print("Type a question, press Enter. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms013 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        response_content = history[-1]["content"]

        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        
        print()
