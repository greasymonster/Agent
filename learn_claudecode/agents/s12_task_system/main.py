"""
s12: Task System — file-persisted task graph with blockedBy dependencies.

Run:  python s12_task_system/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s11:
  - Task dataclass (id, subject, description, status, owner, blockedBy)
  - TASKS_DIR = .tasks/ for persistent JSON storage
  - create_task / save_task / load_task / list_tasks / get_task
  - can_start: checks blockedBy all completed (missing deps = blocked)
  - claim_task: set owner + pending -> in_progress
  - complete_task: set completed + report unblocked downstream
  - 5 new tools: create_task, list_tasks, get_task, claim_task, complete_task

Note: Teaching code keeps a basic agent loop to stay focused on the task
system. S11's full error recovery (RecoveryState, backoff, escalation,
reactive compact, fallback model) is omitted — in real CC, tasks.ts and
withRetry are independent layers that compose naturally.
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

# =====================================================================================================
# 主循环部分
# =====================================================================================================
rounds_since_todo = 0
def agnet_loop(messages: list, context: dict):
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
            context = update_context(context, messages)
            SYSTEM = get_system_prompt(context)
            continue
        continue
    

if __name__ == "__main__":
    print("s12: task system")
    print("Type a question, press Enter. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms012 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agnet_loop(history, context)
        response_content = history[-1]["content"]

        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        
        print()
