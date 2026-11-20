"""colab_core.agent —— 同进程 AI 副驾子包（协议清单、安全护栏、四协议适配、探针、双引擎驱动与 MCP 服务）。"""

from __future__ import annotations

from colab_core.agent.drivers import ClaudeCodeDriver, NativeLightDriver
from colab_core.agent.mcp_server import (
    _default_mcp_executor,
    _is_mcp_mutations_allowed,
    _read_json_line,
    _write_json_line,
    run_mcp_stdio_server,
)
from colab_core.agent.protocols import (
    PROTOCOL_ADAPTERS,
    SUPPORTED_API_FORMATS,
    AnthropicAdapter,
    BaseProtocolAdapter,
    GeminiAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ToolExecRecord,
    TurnStreamResult,
    _LLMHttpError,
    _ThinkTagSplitter,
    get_protocol_adapter,
    is_loopback_base_url,
)
from colab_core.agent.provider_probe import (
    _build_probe_auth_headers,
    _extract_models_and_capabilities,
    _extract_models_from_payload,
    detect_provider_protocol,
    fetch_provider_models,
    probe_provider_models,
)
from colab_core.agent.safety import (
    _READONLY_CMD_VALIDATORS,
    _redact_obj,
    _safe_json_dumps,
    _summarize_tool_result,
    check_remote_code_safety,
    is_readonly_safe_remote_cmd,
)
from colab_core.agent.specs import AGENT_TOOL_MAP, AGENT_TOOL_SPECS, DEFAULT_SYSTEM_PROMPT, get_tool_spec
from colab_security import url_origin_key

__all__ = [
    "AGENT_TOOL_MAP",
    "AGENT_TOOL_SPECS",
    "AnthropicAdapter",
    "BaseProtocolAdapter",
    "ClaudeCodeDriver",
    "DEFAULT_SYSTEM_PROMPT",
    "GeminiAdapter",
    "NativeLightDriver",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "PROTOCOL_ADAPTERS",
    "SUPPORTED_API_FORMATS",
    "ToolExecRecord",
    "TurnStreamResult",
    "_LLMHttpError",
    "_READONLY_CMD_VALIDATORS",
    "_ThinkTagSplitter",
    "_build_probe_auth_headers",
    "_default_mcp_executor",
    "_extract_models_and_capabilities",
    "_extract_models_from_payload",
    "_is_mcp_mutations_allowed",
    "_read_json_line",
    "_redact_obj",
    "_safe_json_dumps",
    "_summarize_tool_result",
    "_write_json_line",
    "check_remote_code_safety",
    "detect_provider_protocol",
    "fetch_provider_models",
    "get_protocol_adapter",
    "get_tool_spec",
    "is_loopback_base_url",
    "is_readonly_safe_remote_cmd",
    "probe_provider_models",
    "run_mcp_stdio_server",
    "url_origin_key",
]
