"""colab_core.agent_tools —— 同进程 AI 副驾核心协议、安全护栏、双引擎驱动与 MCP Stdio 兼容门面。

所有实际实现已拆解至 `colab_core.agent` 子包（`specs` / `safety` / `protocols` / `provider_probe` /
`drivers` / `mcp_server`）；本模块作为薄兼容门面保持 100% 公开与测试依赖符号向后兼容。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from colab_core.agent import (  # noqa: E402
    AGENT_TOOL_MAP,
    AGENT_TOOL_SPECS,
    PROTOCOL_ADAPTERS,
    SUPPORTED_API_FORMATS,
    AnthropicAdapter,
    BaseProtocolAdapter,
    ClaudeCodeDriver,
    DEFAULT_SYSTEM_PROMPT,
    GeminiAdapter,
    NativeLightDriver,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ToolExecRecord,
    TurnStreamResult,
    _LLMHttpError,
    _READONLY_CMD_VALIDATORS,
    _ThinkTagSplitter,
    _build_probe_auth_headers,
    _default_mcp_executor,
    _extract_models_and_capabilities,
    _extract_models_from_payload,
    _is_mcp_mutations_allowed,
    _read_json_line,
    _redact_obj,
    _safe_json_dumps,
    _summarize_tool_result,
    _write_json_line,
    check_remote_code_safety,
    detect_provider_protocol,
    fetch_provider_models,
    get_protocol_adapter,
    get_tool_spec,
    is_loopback_base_url,
    is_readonly_safe_remote_cmd,
    probe_provider_models,
    run_mcp_stdio_server,
    url_origin_key,
)

__all__ = [
    "AGENT_TOOL_MAP",
    "AGENT_TOOL_SPECS",
    "ClaudeCodeDriver",
    "DEFAULT_SYSTEM_PROMPT",
    "NativeLightDriver",
    "SUPPORTED_API_FORMATS",
    "_default_mcp_executor",
    "check_remote_code_safety",
    "detect_provider_protocol",
    "fetch_provider_models",
    "get_tool_spec",
    "is_loopback_base_url",
    "is_readonly_safe_remote_cmd",
    "probe_provider_models",
    "run_mcp_stdio_server",
    "subprocess",
    "url_origin_key",
]
