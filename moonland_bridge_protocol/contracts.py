"""Stable, credential-free contract primitives for Moonland tool resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit


MAX_CONTRACT_BYTES = 512 * 1024
CONTRACT_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,128}$")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_LAB_PATH_PATTERN = re.compile(r"^/w/([^/]+)/lab/?$")
_WORKSPACE_GENERATION_PATH_PATTERN = re.compile(r"^/w/([^/]+)/gen/?$")
_GENERATION_PATH_PATTERN = re.compile(r"^/generation/([^/]+)/?$")
_MODALITIES = {"IMAGE", "VIDEO", "AUDIO", "MASK", "UNKNOWN"}
_FIELD_TYPES = {"string", "integer", "number", "boolean", "enum", "resource", "object", "list"}
_MEDIA_TYPES = {"IMAGE", "VIDEO", "AUDIO", "MASK"}
_UPLOAD_MODES = {"single", "chunked"}


class MoonlandContractError(ValueError):
    """Raised when a URL or normalized Moonland contract is unsafe or incomplete."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MoonlandUrl:
    kind: str
    canonical_url: str
    workspace_slug: str | None = None
    topic_id: str | None = None
    generation_id: str | None = None

    def action_payload(self, preferred_step_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.canonical_url}
        resolved_step_id = preferred_step_id.strip()
        if resolved_step_id:
            _require_id(resolved_step_id, "preferred_step_id")
            payload["preferredStepId"] = resolved_step_id
        return payload


def _require_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise MoonlandContractError("INVALID_CONTRACT", f"{field_name} has an invalid format.")
    return value


def canonicalize_moonland_url(value: str) -> MoonlandUrl:
    """Validate and canonicalize one supported Moonland Lab or generation URL."""

    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 2048:
        raise MoonlandContractError("INVALID_URL", "Moonland URL must be between 1 and 2048 characters.")
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or parsed.hostname != "moonland.ai" or parsed.port is not None:
        raise MoonlandContractError("INVALID_URL", "Only https://moonland.ai URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise MoonlandContractError("INVALID_URL", "Moonland URLs must not contain credentials.")

    lab_match = _LAB_PATH_PATTERN.fullmatch(parsed.path)
    if lab_match:
        workspace_slug = lab_match.group(1)
        if not _SLUG_PATTERN.fullmatch(workspace_slug):
            raise MoonlandContractError("INVALID_URL", "Moonland workspace slug has an invalid format.")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
        if set(query) - {"t", "g"}:
            raise MoonlandContractError("INVALID_URL", "Moonland Lab URL contains unsupported query parameters.")
        if len(query.get("t", [])) != 1:
            raise MoonlandContractError("INVALID_URL", "Moonland Lab URL must contain exactly one topic ID.")
        topic_id = _require_id(query["t"][0], "topic_id")
        generation_values = query.get("g", [])
        if len(generation_values) > 1:
            raise MoonlandContractError("INVALID_URL", "Moonland Lab URL contains multiple generation IDs.")
        generation_id = _require_id(generation_values[0], "generation_id") if generation_values else None
        canonical_query: list[tuple[str, str]] = [("t", topic_id)]
        if generation_id:
            canonical_query.append(("g", generation_id))
        canonical = urlunsplit(("https", "moonland.ai", f"/w/{workspace_slug}/lab", urlencode(canonical_query), ""))
        return MoonlandUrl("lab", canonical, workspace_slug, topic_id, generation_id)

    workspace_generation_match = _WORKSPACE_GENERATION_PATH_PATTERN.fullmatch(parsed.path)
    if workspace_generation_match:
        workspace_slug = workspace_generation_match.group(1)
        if not _SLUG_PATTERN.fullmatch(workspace_slug):
            raise MoonlandContractError("INVALID_URL", "Moonland workspace slug has an invalid format.")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
        if set(query) != {"g"} or len(query["g"]) != 1:
            raise MoonlandContractError(
                "INVALID_URL", "Moonland generation detail URL must contain exactly one generation ID."
            )
        generation_id = _require_id(query["g"][0], "generation_id")
        canonical = urlunsplit(
            ("https", "moonland.ai", f"/w/{workspace_slug}/gen", urlencode({"g": generation_id}), "")
        )
        return MoonlandUrl("generation", canonical, workspace_slug, generation_id=generation_id)

    generation_match = _GENERATION_PATH_PATTERN.fullmatch(parsed.path)
    if generation_match:
        if parsed.query:
            raise MoonlandContractError("INVALID_URL", "Moonland generation URL must not contain query parameters.")
        generation_id = _require_id(generation_match.group(1), "generation_id")
        canonical = f"https://moonland.ai/generation/{generation_id}"
        return MoonlandUrl("generation", canonical, generation_id=generation_id)

    raise MoonlandContractError(
        "INVALID_URL",
        "Supported paths are /w/<workspace>/lab?t=<topic>, /w/<workspace>/gen?g=<generation>, and /generation/<id>.",
    )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MoonlandContractError("INVALID_CONTRACT", f"{field_name} must be an object.")
    return value


def _optional_string(value: Any, field_name: str, maximum: int = 256) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise MoonlandContractError("INVALID_CONTRACT", f"{field_name} has an invalid format.")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise MoonlandContractError("INVALID_CONTRACT", f"{field_name} contains unsupported fields: {names}.")


def _normalize_field(value: Any) -> dict[str, Any]:
    field = _require_mapping(value, "fields[]")
    allowed = {"id", "type", "label", "required", "default", "minimum", "maximum", "step", "maxLength", "options", "itemType"}
    _strict_keys(field, allowed, "fields[]")
    field_id = _require_id(field.get("id"), "fields[].id")
    field_type = field.get("type")
    if field_type not in _FIELD_TYPES:
        raise MoonlandContractError("INVALID_CONTRACT", f"Unsupported field type for {field_id}.")
    normalized: dict[str, Any] = {"id": field_id, "type": field_type, "required": bool(field.get("required", False))}
    label = _optional_string(field.get("label"), "fields[].label")
    if label is not None:
        normalized["label"] = label
    for key in ("default", "minimum", "maximum", "step", "maxLength", "options", "itemType"):
        if key in field:
            normalized[key] = field[key]
    if field_type == "enum":
        options = normalized.get("options")
        if not isinstance(options, list) or not options or any(not isinstance(item, (str, int, float, bool)) for item in options):
            raise MoonlandContractError("INVALID_CONTRACT", f"Enum field {field_id} requires scalar options.")
    return normalized


def _normalize_resource_slot(value: Any) -> dict[str, Any]:
    slot = _require_mapping(value, "resourceSlots[]")
    allowed = {
        "id", "label", "mediaType", "purposes", "minCount", "maxCount", "acceptedContentTypes",
        "maxBytes", "normalizationProfile", "uploadMode",
    }
    _strict_keys(slot, allowed, "resourceSlots[]")
    slot_id = _require_id(slot.get("id"), "resourceSlots[].id")
    media_type = slot.get("mediaType")
    if media_type not in _MEDIA_TYPES:
        raise MoonlandContractError("INVALID_CONTRACT", f"Unsupported media type for {slot_id}.")
    min_count = slot.get("minCount", 0)
    max_count = slot.get("maxCount", 1)
    max_bytes = slot.get("maxBytes")
    if not isinstance(min_count, int) or not isinstance(max_count, int) or min_count < 0 or max_count < max(1, min_count):
        raise MoonlandContractError("INVALID_CONTRACT", f"Invalid resource count bounds for {slot_id}.")
    if max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes <= 0):
        raise MoonlandContractError("INVALID_CONTRACT", f"Invalid maxBytes for {slot_id}.")
    content_types = slot.get("acceptedContentTypes", [])
    if not isinstance(content_types, list) or any(not isinstance(item, str) or "/" not in item for item in content_types):
        raise MoonlandContractError("INVALID_CONTRACT", f"Invalid acceptedContentTypes for {slot_id}.")
    purposes = slot.get("purposes", [])
    if not isinstance(purposes, list) or any(not isinstance(item, str) or not item for item in purposes):
        raise MoonlandContractError("INVALID_CONTRACT", f"Invalid purposes for {slot_id}.")
    upload_mode = slot.get("uploadMode", "single")
    if upload_mode not in _UPLOAD_MODES:
        raise MoonlandContractError("INVALID_CONTRACT", f"Invalid uploadMode for {slot_id}.")
    normalized: dict[str, Any] = {
        "id": slot_id,
        "mediaType": media_type,
        "purposes": purposes,
        "minCount": min_count,
        "maxCount": max_count,
        "acceptedContentTypes": content_types,
        "uploadMode": upload_mode,
    }
    for key in ("label", "maxBytes", "normalizationProfile"):
        item = slot.get(key)
        if item is not None:
            if key in {"label", "normalizationProfile"}:
                item = _optional_string(item, f"resourceSlots[].{key}")
            normalized[key] = item
    return normalized


def normalize_resolved_contract(value: Mapping[str, Any], *, expected_url: MoonlandUrl | None = None) -> dict[str, Any]:
    """Validate an extension result and return a canonical semantic contract."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_CONTRACT_BYTES:
        raise MoonlandContractError("CONTRACT_TOO_LARGE", "Moonland contract exceeds the 512 KiB bridge limit.")
    root = _require_mapping(value, "contract")
    _strict_keys(root, {"schemaVersion", "source", "workspace", "lab", "target", "fields", "resourceSlots", "pricing"}, "contract")
    if root.get("schemaVersion") != CONTRACT_SCHEMA_VERSION:
        raise MoonlandContractError("INVALID_CONTRACT", "Unsupported local contract schema version.")

    source = _require_mapping(root.get("source"), "source")
    _strict_keys(source, {"canonicalUrl", "resolvedAt", "contractHash"}, "source")
    parsed_url = canonicalize_moonland_url(source.get("canonicalUrl"))
    if expected_url is not None and parsed_url.canonical_url != expected_url.canonical_url:
        raise MoonlandContractError("URL_MISMATCH", "Resolved contract URL does not match the requested URL.")
    resolved_at = _optional_string(source.get("resolvedAt"), "source.resolvedAt", 64)
    if resolved_at:
        try:
            datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise MoonlandContractError("INVALID_CONTRACT", "source.resolvedAt must be ISO-8601.") from error

    workspace = _require_mapping(root.get("workspace"), "workspace")
    _strict_keys(workspace, {"id", "slug"}, "workspace")
    workspace_id = _require_id(workspace.get("id"), "workspace.id")
    workspace_slug = workspace.get("slug")
    if not isinstance(workspace_slug, str) or not _SLUG_PATTERN.fullmatch(workspace_slug):
        raise MoonlandContractError("INVALID_CONTRACT", "workspace.slug has an invalid format.")
    if parsed_url.workspace_slug and parsed_url.workspace_slug != workspace_slug:
        raise MoonlandContractError("URL_MISMATCH", "Resolved workspace does not match the requested URL.")

    lab = _require_mapping(root.get("lab"), "lab")
    _strict_keys(lab, {"topicId", "stepId", "generationId"}, "lab")
    normalized_lab = {
        "topicId": _optional_string(lab.get("topicId"), "lab.topicId", 128),
        "stepId": _optional_string(lab.get("stepId"), "lab.stepId", 128),
        "generationId": _optional_string(lab.get("generationId"), "lab.generationId", 128),
    }
    for key, item in normalized_lab.items():
        if item is not None:
            _require_id(item, f"lab.{key}")
    if parsed_url.topic_id and normalized_lab["topicId"] != parsed_url.topic_id:
        raise MoonlandContractError("URL_MISMATCH", "Resolved topic does not match the requested URL.")
    if parsed_url.generation_id and normalized_lab["generationId"] != parsed_url.generation_id:
        raise MoonlandContractError("URL_MISMATCH", "Resolved generation does not match the requested URL.")

    target = _require_mapping(root.get("target"), "target")
    _strict_keys(target, {"id", "contractRevision", "displayName", "outputModality"}, "target")
    target_id = _require_id(target.get("id"), "target.id")
    revision = target.get("contractRevision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise MoonlandContractError("INVALID_CONTRACT", "target.contractRevision must be a positive integer.")
    display_name = _optional_string(target.get("displayName"), "target.displayName")
    modality = target.get("outputModality", "UNKNOWN")
    if modality not in _MODALITIES:
        raise MoonlandContractError("INVALID_CONTRACT", "target.outputModality is unsupported.")

    fields = root.get("fields")
    resource_slots = root.get("resourceSlots")
    if not isinstance(fields, list) or not isinstance(resource_slots, list):
        raise MoonlandContractError("INVALID_CONTRACT", "fields and resourceSlots must be arrays.")
    normalized_fields = [_normalize_field(field) for field in fields]
    normalized_slots = [_normalize_resource_slot(slot) for slot in resource_slots]
    field_ids = [item["id"] for item in normalized_fields]
    slot_ids = [item["id"] for item in normalized_slots]
    if len(field_ids) != len(set(field_ids)) or len(slot_ids) != len(set(slot_ids)):
        raise MoonlandContractError("INVALID_CONTRACT", "Field IDs and resource slot IDs must each be unique.")
    resource_field_ids = {item["id"] for item in normalized_fields if item["type"] == "resource"}
    if resource_field_ids != set(slot_ids):
        raise MoonlandContractError("INVALID_CONTRACT", "Every resource field must have exactly one matching resource slot.")

    pricing_value = root.get("pricing")
    pricing: dict[str, Any] | None = None
    if pricing_value is not None:
        pricing_map = _require_mapping(pricing_value, "pricing")
        _strict_keys(pricing_map, {"estimatedPoints", "currency"}, "pricing")
        points = pricing_map.get("estimatedPoints")
        if points is not None and (not isinstance(points, (int, float)) or isinstance(points, bool) or points < 0):
            raise MoonlandContractError("INVALID_CONTRACT", "pricing.estimatedPoints must be non-negative.")
        if pricing_map.get("currency") != "POINT":
            raise MoonlandContractError("INVALID_CONTRACT", "pricing.currency must be POINT.")
        pricing = {"estimatedPoints": points, "currency": "POINT"}

    semantic: dict[str, Any] = {
        "schemaVersion": CONTRACT_SCHEMA_VERSION,
        "workspace": {"id": workspace_id, "slug": workspace_slug},
        "lab": normalized_lab,
        "target": {
            "id": target_id,
            "contractRevision": revision,
            "displayName": display_name,
            "outputModality": modality,
        },
        "fields": normalized_fields,
        "resourceSlots": normalized_slots,
        "pricing": pricing,
    }
    semantic_json = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract_hash = hashlib.sha256(semantic_json.encode("utf-8")).hexdigest()
    supplied_hash = source.get("contractHash")
    if supplied_hash is not None and supplied_hash != contract_hash:
        raise MoonlandContractError("CONTRACT_HASH_MISMATCH", "Moonland contract semantic hash is invalid.")
    return {
        "schemaVersion": CONTRACT_SCHEMA_VERSION,
        "source": {
            "canonicalUrl": parsed_url.canonical_url,
            "resolvedAt": resolved_at,
            "contractHash": contract_hash,
        },
        **{key: semantic[key] for key in ("workspace", "lab", "target", "fields", "resourceSlots", "pricing")},
    }


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "MAX_CONTRACT_BYTES",
    "MoonlandContractError",
    "MoonlandUrl",
    "canonicalize_moonland_url",
    "normalize_resolved_contract",
]
