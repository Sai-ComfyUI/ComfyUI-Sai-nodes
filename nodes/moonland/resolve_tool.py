"""Resolve a Moonland Lab or generation URL into a stable tool contract."""

from __future__ import annotations

import json
import time
from typing import Any

from comfy_api.latest import io

from ...moonland_bridge_protocol import (
    MoonlandContractError,
    canonicalize_moonland_url,
    normalize_resolved_contract,
)
from ...server.moonland_bridge import request_bridge_action


MoonlandToolContractType = io.Custom("MOONLAND_TOOL_CONTRACT")


async def resolve_tool_contract(url: str, preferred_step_id: str = "") -> dict[str, Any]:
    parsed_url = canonicalize_moonland_url(url)
    result = await request_bridge_action(
        "tool.resolveContract",
        parsed_url.action_payload(preferred_step_id),
    )
    return normalize_resolved_contract(result, expected_url=parsed_url)


class MoonlandResolveToolSai(io.ComfyNode):
    """Read the active Moonland tool contract without submitting a generation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MoonlandResolveTool_Sai",
            display_name="Moonland Resolve Tool Ψ",
            category="Sai/Moonland",
            description=(
                "Resolves a Moonland Lab or generation URL through the signed-in browser tab. "
                "This node is read-only and never submits a generation."
            ),
            search_aliases=["Moonland URL", "Moonland contract", "Moonland tool"],
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "url",
                    default="https://moonland.ai/w/example/lab?t=topic-id",
                    tooltip="A Moonland Lab topic URL or generation URL.",
                ),
                io.String.Input(
                    "preferred_step_id",
                    default="",
                    advanced=True,
                    tooltip="Optional exact Lab step ID. Leave empty to resolve the active or latest complete step.",
                ),
            ],
            outputs=[
                MoonlandToolContractType.Output(display_name="tool_contract"),
                io.String.Output(display_name="target_id"),
                io.Int.Output(display_name="contract_revision"),
                io.String.Output(display_name="output_modality"),
                io.String.Output(display_name="summary_json"),
            ],
        )

    @classmethod
    def validate_inputs(cls, url: str, preferred_step_id: str = "") -> bool | str:
        try:
            parsed = canonicalize_moonland_url(url)
            parsed.action_payload(preferred_step_id)
        except MoonlandContractError as error:
            return f"{error.code}: {error}"
        return True

    @classmethod
    def fingerprint_inputs(cls, url: str, preferred_step_id: str = "") -> int:
        # Remote page state and private contracts can change without node inputs changing.
        return time.time_ns()

    @classmethod
    async def execute(cls, url: str, preferred_step_id: str = "") -> io.NodeOutput:
        contract = await resolve_tool_contract(url, preferred_step_id)
        target = contract["target"]
        summary = json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2)
        return io.NodeOutput(
            contract,
            target["id"],
            target["contractRevision"],
            target["outputModality"],
            summary,
        )


__all__ = ["MoonlandResolveToolSai", "MoonlandToolContractType", "resolve_tool_contract"]
