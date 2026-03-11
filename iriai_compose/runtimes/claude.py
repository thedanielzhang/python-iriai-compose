from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from iriai_compose.runner import AgentRuntime
from iriai_compose.storage import AgentSession, SessionStore

if TYPE_CHECKING:
    from iriai_compose.actors import Role
    from iriai_compose.workflow import Workspace


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref`` references by inlining ``$defs``.

    Pydantic generates JSON schemas with ``$defs`` + ``$ref`` for nested
    models.  The Claude API's constrained decoding does not support
    ``$ref``, so we inline all definitions to make the full structure
    visible at every nesting level.
    """
    defs = schema.pop("$defs", None)
    if not defs:
        return schema

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if ref and isinstance(ref, str):
                name = ref.rsplit("/", 1)[-1]
                if name in defs:
                    return _resolve(defs[name])
                return obj
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


class ClaudeAgentRuntime(AgentRuntime):
    """Agent runtime backed by the Claude Agent SDK.

    Uses deferred import — the module is importable, but instantiation
    raises a clear error if the SDK is not installed.
    """

    name = "claude"

    def __init__(
        self,
        session_store: SessionStore | None = None,
        on_message: Callable[[Any], None] | None = None,
    ) -> None:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            raise ImportError(
                "ClaudeAgentRuntime requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            )
        self.session_store = session_store
        self.on_message = on_message

    async def invoke(
        self,
        role: Role,
        prompt: str,
        *,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | None = None,
        session_key: str | None = None,
    ) -> str | BaseModel:
        from claude_agent_sdk.types import ResultMessage

        options = self._build_options(role, workspace)

        # Session resumption
        if session_key and self.session_store:
            session = await self.session_store.load(session_key)
            if session and session.session_id:
                options.resume = session.session_id

        if output_type:
            inlined_schema = _inline_defs(output_type.model_json_schema())
            options.output_format = {
                "type": "json_schema",
                "schema": inlined_schema,
            }

        result_msg = await self._query(prompt, options)

        # Persist session for resumption
        if session_key and self.session_store and self._last_session_id:
            await self.session_store.save(
                AgentSession(
                    session_key=session_key,
                    session_id=self._last_session_id,
                )
            )

        if not output_type:
            return result_msg.result

        # Structured output — the SDK uses constrained decoding to
        # guarantee the result matches the schema.
        subtype = getattr(result_msg, "subtype", None)
        if subtype == "error_max_structured_output_retries":
            raise RuntimeError(
                f"Claude could not produce valid {output_type.__name__} "
                f"after multiple attempts. Last result: {result_msg.result}"
            )

        structured = getattr(result_msg, "structured_output", None)
        if structured is None:
            raise RuntimeError(
                f"Expected structured output for {output_type.__name__} "
                f"but received None. Result text: {result_msg.result}"
            )

        return output_type.model_validate(structured)

    def _build_options(self, role: Role, workspace: Workspace | None) -> Any:
        """Construct ClaudeAgentOptions from a role."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            system_prompt=role.prompt,
            allowed_tools=role.tools,
            model=role.model or "claude-sonnet-4-6",
            cwd=str(workspace.path) if workspace else None,
        )

        if "setting_sources" in role.metadata:
            options.setting_sources = role.metadata["setting_sources"]

        if "mcp_servers" in role.metadata:
            options.mcp_servers = role.metadata["mcp_servers"]

        return options

    async def _query(self, prompt: str, options: Any) -> Any:
        """Run a single Claude Agent SDK query, returning the ResultMessage."""
        from claude_agent_sdk import query
        from claude_agent_sdk.types import ResultMessage

        result_msg: ResultMessage | None = None
        self._last_session_id: str | None = None

        async for msg in query(prompt=prompt, options=options):
            if self.on_message is not None:
                self.on_message(msg)
            if isinstance(msg, ResultMessage):
                result_msg = msg

        if result_msg is None:
            raise RuntimeError("Claude query completed without a result message")

        self._last_session_id = getattr(result_msg, "session_id", None)
        return result_msg
