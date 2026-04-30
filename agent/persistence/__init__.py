"""agent/persistence — Self-contained trajectory storage

Every agent trajectory is saved as a single ``dialog.json`` file
holding the full record of the run: system prompt, available tools,
every turn (user / assistant thought / tool call / tool response), and
final outcome. Open one file → see everything.

See ``docs/UNIFIED_TRAJECTORY_FORMAT.md`` for the schema.
"""
from agent.persistence.dialog_format import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    DIALOG_FILENAME,
    CONTEXT_SPLIT_ROLE,
    DEFAULT_SERVER_MAP,
    DialogBuilder,
    Message,
    ToolCall,
    ValidationIssue,
    save_dialog,
    load_dialog,
    validate_dialog,
    messages_of,
    meta_of,
    result_of,
    search_tree_of,
    solved_path_of,
    split_dialog_at_markers,
    is_tool_response_user_msg,
    strip_tool_response_wrapper,
)
from agent.persistence.dialog_adapters import (
    from_loop_messages,
    from_trajectory,
    from_proof_trace,
    from_session_messages,
    to_openai_messages,
)
from agent.persistence.sft_export import (
    ChatTemplate,
    QWEN3_PRESET,
    AGENTCPM_PRESET,
    OPENAI_PRESET,
    PRESETS,
    dialog_to_sft_sample,
    dialogs_to_sft_jsonl,
    write_sft_jsonl,
)
from agent.persistence.unified_storage import (
    save_task,
    load_task,
    save_task_outputs,    # back-compat alias
    load_task_outputs,    # back-compat alias
    collect_dialogs,
    build_meta,
    build_result,
)

__all__ = [
    # dialog_format
    "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
    "DIALOG_FILENAME", "CONTEXT_SPLIT_ROLE",
    "DEFAULT_SERVER_MAP",
    "DialogBuilder", "Message", "ToolCall", "ValidationIssue",
    "save_dialog", "load_dialog", "validate_dialog",
    "messages_of", "meta_of", "result_of",
    "search_tree_of", "solved_path_of",
    "split_dialog_at_markers",
    "is_tool_response_user_msg", "strip_tool_response_wrapper",
    # adapters
    "from_loop_messages", "from_trajectory", "from_proof_trace",
    "from_session_messages", "to_openai_messages",
    # SFT export
    "ChatTemplate", "QWEN3_PRESET", "AGENTCPM_PRESET", "OPENAI_PRESET",
    "PRESETS",
    "dialog_to_sft_sample", "dialogs_to_sft_jsonl", "write_sft_jsonl",
    # unified storage (single file)
    "save_task", "load_task",
    "save_task_outputs", "load_task_outputs",
    "collect_dialogs",
    "build_meta", "build_result",
]
