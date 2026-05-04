"""agent.persistence — 自包含的 trajectory 落盘

每次 agent 运行产出一个 ``dialog.json``: system prompt, available tools,
每一轮 (user / assistant thought / tool call / tool response), 最终结果。
打开一个文件看到一切。

Schema 定义见 docs/UNIFIED_TRAJECTORY_FORMAT.md。
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
    collect_dialogs,
    build_meta,
    build_result,
)

__all__ = [
    "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
    "DIALOG_FILENAME", "CONTEXT_SPLIT_ROLE",
    "DEFAULT_SERVER_MAP",
    "DialogBuilder", "Message", "ToolCall", "ValidationIssue",
    "save_dialog", "load_dialog", "validate_dialog",
    "messages_of", "meta_of", "result_of",
    "search_tree_of", "solved_path_of",
    "split_dialog_at_markers",
    "is_tool_response_user_msg", "strip_tool_response_wrapper",
    "from_loop_messages", "from_trajectory", "from_proof_trace",
    "to_openai_messages",
    "ChatTemplate", "QWEN3_PRESET", "AGENTCPM_PRESET", "OPENAI_PRESET",
    "PRESETS",
    "dialog_to_sft_sample", "dialogs_to_sft_jsonl", "write_sft_jsonl",
    "save_task", "load_task",
    "collect_dialogs",
    "build_meta", "build_result",
]
