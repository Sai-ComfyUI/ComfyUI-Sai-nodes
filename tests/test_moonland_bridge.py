"""Tests for the credential-free Moonland browser bridge protocol."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from moonland_bridge_protocol import (
    BridgeProtocolError,
    BridgeRegistry,
    MoonlandContractError,
    PROTOCOL_VERSION,
    canonicalize_moonland_url,
    normalize_resolved_contract,
    parse_hello,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def hello_message(**overrides):
    value = {
        "type": "hello",
        "protocolVersion": PROTOCOL_VERSION,
        "bridgeId": "bridge-12345678",
        "extensionVersion": "0.1.0",
        "pairingCode": "123456",
        "moonland": {
            "authenticated": True,
            "pageUrl": "https://moonland.ai/w/example/lab?t=topic",
            "workspaceId": "workspace-id",
            "workspaceSlug": "example",
        },
    }
    value.update(overrides)
    return value


class BridgeProtocolTests(unittest.TestCase):
    def test_pairing_code_is_single_use(self) -> None:
        registry = BridgeRegistry()
        code, expires_in = registry.issue_pairing_code()
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())
        self.assertEqual(expires_in, 300)
        self.assertTrue(registry.consume_pairing_code(code))
        self.assertFalse(registry.consume_pairing_code(code))

    def test_pairing_code_expires(self) -> None:
        now = [100.0]
        registry = BridgeRegistry(clock=lambda: now[0], pairing_ttl_seconds=5)
        code, _ = registry.issue_pairing_code()
        now[0] = 106.0
        self.assertFalse(registry.consume_pairing_code(code))

    def test_hello_is_sanitized_to_allowlisted_fields(self) -> None:
        message = hello_message()
        message["cookie"] = "must-not-be-retained"
        message["moonland"]["token"] = "must-not-be-retained"
        parsed = parse_hello(message)
        registry = BridgeRegistry()
        registry.connect(parsed)
        status = registry.status()
        encoded = json.dumps(status)
        self.assertTrue(status["connected"])
        self.assertNotIn("cookie", encoded)
        self.assertNotIn("token", encoded)
        self.assertNotIn("pairingCode", encoded)
        self.assertEqual(status["connections"][0]["moonland"]["workspace_slug"], "example")

    def test_rejects_unknown_protocol_version(self) -> None:
        with self.assertRaises(BridgeProtocolError):
            parse_hello(hello_message(protocolVersion=99))

    def test_extension_manifest_and_protocol_schema_are_valid_json(self) -> None:
        manifest = json.loads((PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "manifest.json").read_text(encoding="utf-8"))
        schema = json.loads((PACKAGE_ROOT / "moonland_bridge_protocol" / "v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertEqual(schema["$defs"]["hello"]["properties"]["protocolVersion"]["const"], 1)
        self.assertNotIn("cookies", manifest["permissions"])
        self.assertIn("scripting", manifest["permissions"])
        self.assertEqual(manifest["version"], "0.7.6")
        self.assertEqual(
            schema["$defs"]["actionRequest"]["properties"]["action"]["enum"],
            ["resource.ensureImage", "tool.resolveContract"],
        )
        observer, resolver = manifest["content_scripts"]
        self.assertEqual(observer["js"], ["page_query_observer.js"])
        self.assertEqual(observer["run_at"], "document_start")
        self.assertEqual(observer["world"], "MAIN")
        self.assertEqual(resolver["js"], ["contract_resolver.js", "content.js"])

    def test_contract_observer_is_read_only_and_redacts_content(self) -> None:
        observer_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "page_query_observer.js").read_text(encoding="utf-8")
        self.assertIn("MUTATION_PREFIXES", observer_source)
        self.assertIn("READ_PREFIXES", observer_source)
        self.assertIn("prompt|negativePrompt|text|content|value|defaultValue", observer_source)
        self.assertIn("signedUrl|uploadUrl|token|accessToken|refreshToken|authorization|cookie", observer_source)
        self.assertIn("safeToCapture", observer_source)
        self.assertNotIn("originalFetch(...args,", observer_source)

    def test_extension_uses_dev_port_and_portless_host_permission_pattern(self) -> None:
        popup_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "popup.js").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_COMFY_URL = "http://127.0.0.1:9527"', popup_source)
        self.assertIn("`${url.protocol}//${url.hostname}/*`", popup_source)
        self.assertNotIn("`${new URL(comfyUrl).origin}/*`", popup_source)
        self.assertNotIn("targetAddressSpace", popup_source)

    def test_extension_handles_invalidated_content_script_context(self) -> None:
        content_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "content.js").read_text(encoding="utf-8")
        background_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "background.js").read_text(encoding="utf-8")
        self.assertIn("handleDeliveryError", content_source)
        self.assertIn("Extension context invalidated", content_source)
        self.assertIn("Reload the Moonland tab after reloading the extension", background_source)

    def test_resolver_queries_are_fixed_read_only_allowlist(self) -> None:
        content_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "content.js").read_text(encoding="utf-8")
        self.assertIn('new Set(["lab.listStepsByTopic"])', content_source)
        self.assertIn('method: "GET"', content_source)
        self.assertIn("Resolver query is not allowlisted", content_source)

    def test_pairing_command_copies_code_before_falling_back_to_dialog(self) -> None:
        frontend_source = (PACKAGE_ROOT / "web" / "moonland_bridge.js").read_text(encoding="utf-8")
        self.assertIn("navigator.clipboard?.writeText", frontend_source)
        self.assertIn('document.execCommand("copy")', frontend_source)
        self.assertIn("Pairing code copied", frontend_source)
        self.assertIn("Clipboard access was unavailable", frontend_source)

    def test_websocket_runs_in_comfyui_page_origin(self) -> None:
        background_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "background.js").read_text(encoding="utf-8")
        page_bridge_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "comfy_bridge_content.js").read_text(encoding="utf-8")
        self.assertIn('files: ["comfy_bridge_content.js"]', background_source)
        self.assertNotIn("new WebSocket", background_source)
        self.assertIn("window.location.origin", page_bridge_source)
        self.assertIn("new WebSocket(targetUrl)", page_bridge_source)

    def test_extension_resource_action_is_allowlisted_and_credential_free(self) -> None:
        manifest = json.loads((PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "manifest.json").read_text(encoding="utf-8"))
        content_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "content.js").read_text(encoding="utf-8")
        background_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "background.js").read_text(encoding="utf-8")
        page_bridge_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "comfy_bridge_content.js").read_text(encoding="utf-8")
        self.assertIn('action === "resource.ensureImage"', content_source)
        self.assertIn('trpcMutation("resource.requestUpload"', content_source)
        self.assertIn('allocation?.status === "REUSED"', content_source)
        self.assertIn('trpcMutation("resource.postUpload"', content_source)
        self.assertIn("0.92", content_source)
        self.assertIn('type: "bridge.action"', page_bridge_source)
        self.assertIn('type: "moonland.action"', background_source)
        self.assertNotIn("cookies", manifest["permissions"])
        self.assertNotIn("chrome.cookies", content_source + background_source + page_bridge_source)

    def test_background_aggregates_status_across_moonland_tabs(self) -> None:
        background_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "background.js").read_text(encoding="utf-8")
        self.assertIn("moonlandStatusByTab", background_source)
        self.assertIn("refreshCurrentMoonlandStatus", background_source)
        self.assertIn("sender.tab?.id", background_source)
        self.assertIn("chrome.tabs.onRemoved", background_source)

    def test_content_script_retains_workspace_status_during_nextjs_lifecycle(self) -> None:
        content_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "content.js").read_text(encoding="utf-8")
        self.assertIn("lastKnownWorkspaceStatus", content_source)
        self.assertIn("window.location.pathname.startsWith(workspacePrefix)", content_source)

    def test_contract_url_is_canonical_and_rejects_proxy_shaped_inputs(self) -> None:
        parsed = canonicalize_moonland_url(
            "https://moonland.ai/w/purple-bright-coyote/lab?g=generation-id&t=topic-id#ignored"
        )
        self.assertEqual(
            parsed.canonical_url,
            "https://moonland.ai/w/purple-bright-coyote/lab?t=topic-id&g=generation-id",
        )
        self.assertEqual(parsed.action_payload("step-id")["preferredStepId"], "step-id")
        generation = canonicalize_moonland_url(
            "https://moonland.ai/w/purple-bright-coyote/gen?g=cmuau8th800es01s63xsq2k7b#ignored"
        )
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.workspace_slug, "purple-bright-coyote")
        self.assertEqual(generation.generation_id, "cmuau8th800es01s63xsq2k7b")
        self.assertEqual(
            generation.canonical_url,
            "https://moonland.ai/w/purple-bright-coyote/gen?g=cmuau8th800es01s63xsq2k7b",
        )
        for value in (
            "http://moonland.ai/w/example/lab?t=topic-id",
            "https://example.com/w/example/lab?t=topic-id",
            "https://moonland.ai/w/example/lab?t=topic-id&procedure=lab.submit",
            "https://moonland.ai/w/example/gen",
            "https://moonland.ai/w/example/gen?g=generation-id&procedure=generation.create",
            "https://user:secret@moonland.ai/w/example/lab?t=topic-id",
        ):
            with self.subTest(value=value), self.assertRaises(MoonlandContractError):
                canonicalize_moonland_url(value)

    def test_normalized_contract_is_strict_and_hashes_semantics(self) -> None:
        requested = canonicalize_moonland_url("https://moonland.ai/w/example/lab?t=topic-id")
        raw = {
            "schemaVersion": 1,
            "source": {
                "canonicalUrl": requested.canonical_url,
                "resolvedAt": "2026-09-22T00:00:00.000Z",
            },
            "workspace": {"id": "workspace-id", "slug": "example"},
            "lab": {"topicId": "topic-id", "stepId": "step-id", "generationId": None},
            "target": {
                "id": "minimax-h3",
                "contractRevision": 6,
                "displayName": "MiniMax H3",
                "outputModality": "VIDEO",
            },
            "fields": [
                {"id": "prompt", "type": "string", "required": True, "maxLength": 6000},
                {"id": "referenceAudio", "type": "resource", "required": False},
            ],
            "resourceSlots": [
                {
                    "id": "referenceAudio",
                    "mediaType": "AUDIO",
                    "purposes": [],
                    "minCount": 0,
                    "maxCount": 1,
                    "acceptedContentTypes": ["audio/opus"],
                    "maxBytes": 10485760,
                    "normalizationProfile": "moonland-audio-opus-128k-v1",
                    "uploadMode": "single",
                }
            ],
            "pricing": {"estimatedPoints": 626, "currency": "POINT"},
        }
        normalized = normalize_resolved_contract(raw, expected_url=requested)
        self.assertRegex(normalized["source"]["contractHash"], r"^[0-9a-f]{64}$")
        self.assertEqual(normalized["target"]["contractRevision"], 6)
        changed_timestamp = json.loads(json.dumps(raw))
        changed_timestamp["source"]["resolvedAt"] = "2026-09-22T01:00:00.000Z"
        self.assertEqual(
            normalize_resolved_contract(changed_timestamp, expected_url=requested)["source"]["contractHash"],
            normalized["source"]["contractHash"],
        )
        with self.assertRaises(MoonlandContractError):
            normalize_resolved_contract({**raw, "cookie": "must-not-pass"}, expected_url=requested)

    def test_resolver_source_is_read_only_and_tab_scoped(self) -> None:
        content_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "content.js").read_text(encoding="utf-8")
        resolver_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "contract_resolver.js").read_text(encoding="utf-8")
        background_source = (PACKAGE_ROOT / "tools" / "moonland-bridge-extension" / "background.js").read_text(encoding="utf-8")
        self.assertIn('action === "tool.resolveContract"', content_source)
        self.assertIn("CONTRACT_AMBIGUOUS", resolver_source)
        self.assertIn("TAB_NOT_FOUND", background_source)
        self.assertIn('open.searchParams.get("g") === requested.searchParams.get("g")', background_source)
        self.assertNotIn("lab.submit", content_source + resolver_source)
        self.assertNotIn("generation.create", content_source + resolver_source)

    def test_javascript_and_python_share_the_normalized_hash_contract(self) -> None:
        snapshot = json.loads(
            (PACKAGE_ROOT / "tests" / "fixtures" / "moonland_krea2_normalized.json").read_text(encoding="utf-8")
        )
        expected = canonicalize_moonland_url(snapshot["source"]["canonicalUrl"])
        normalized = normalize_resolved_contract(snapshot, expected_url=expected)
        self.assertEqual(normalized, snapshot)


if __name__ == "__main__":
    unittest.main()
