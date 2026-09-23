(() => {
    const MAX_VISITED_OBJECTS = 20000;
    const MAX_DEPTH = 14;
    const IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]{3,128}$/;
    const OUTPUT_MODALITIES = new Set(["IMAGE", "VIDEO", "AUDIO", "MASK", "UNKNOWN"]);
    const KNOWN_TARGET_METADATA = Object.freeze({
        "aio-krea2": { displayName: "Krea 2", outputModality: "IMAGE" },
        "aio-flux2-klein-9b": { displayName: "FLUX.2 [klein]", outputModality: "IMAGE" },
        "minimax-h3": { displayName: "MiniMax H3", outputModality: "VIDEO" },
    });

    function resolverError(code, message) {
        const error = new Error(`${code}: ${message}`);
        error.code = code;
        return error;
    }

    function requireIdentifier(value, name) {
        if (typeof value !== "string" || !IDENTIFIER_PATTERN.test(value)) {
            throw resolverError("INVALID_URL", `${name} has an invalid format.`);
        }
        return value;
    }

    function canonicalizeMoonlandUrl(value) {
        if (typeof value !== "string" || value.length === 0 || value.length > 2048) {
            throw resolverError("INVALID_URL", "Moonland URL must be between 1 and 2048 characters.");
        }
        let url;
        try {
            url = new URL(value);
        } catch {
            throw resolverError("INVALID_URL", "Moonland URL is invalid.");
        }
        if (url.protocol !== "https:" || url.hostname !== "moonland.ai" || url.port || url.username || url.password) {
            throw resolverError("INVALID_URL", "Only credential-free https://moonland.ai URLs are supported.");
        }
        const labMatch = url.pathname.match(/^\/w\/([^/]+)\/lab\/?$/);
        if (labMatch) {
            const allowed = new Set(["t", "g"]);
            for (const key of url.searchParams.keys()) {
                if (!allowed.has(key)) throw resolverError("INVALID_URL", `Unsupported query parameter: ${key}.`);
            }
            if (url.searchParams.getAll("t").length !== 1) {
                throw resolverError("INVALID_URL", "Moonland Lab URL must contain exactly one topic ID.");
            }
            if (url.searchParams.getAll("g").length > 1) {
                throw resolverError("INVALID_URL", "Moonland Lab URL contains multiple generation IDs.");
            }
            const workspaceSlug = requireIdentifier(labMatch[1], "workspace slug");
            const topicId = requireIdentifier(url.searchParams.get("t"), "topic ID");
            const generationId = url.searchParams.has("g") ? requireIdentifier(url.searchParams.get("g"), "generation ID") : null;
            const canonical = new URL(`https://moonland.ai/w/${workspaceSlug}/lab`);
            canonical.searchParams.set("t", topicId);
            if (generationId) canonical.searchParams.set("g", generationId);
            return { kind: "lab", canonicalUrl: canonical.toString(), workspaceSlug, topicId, generationId };
        }
        const workspaceGenerationMatch = url.pathname.match(/^\/w\/([^/]+)\/gen\/?$/);
        if (workspaceGenerationMatch) {
            const keys = Array.from(url.searchParams.keys());
            if (keys.length !== 1 || keys[0] !== "g" || url.searchParams.getAll("g").length !== 1) {
                throw resolverError("INVALID_URL", "Moonland generation detail URL must contain exactly one generation ID.");
            }
            const workspaceSlug = requireIdentifier(workspaceGenerationMatch[1], "workspace slug");
            const generationId = requireIdentifier(url.searchParams.get("g"), "generation ID");
            const canonical = new URL(`https://moonland.ai/w/${workspaceSlug}/gen`);
            canonical.searchParams.set("g", generationId);
            return { kind: "generation", canonicalUrl: canonical.toString(), workspaceSlug, topicId: null, generationId };
        }
        const generationMatch = url.pathname.match(/^\/generation\/([^/]+)\/?$/);
        if (generationMatch && !url.search) {
            const generationId = requireIdentifier(generationMatch[1], "generation ID");
            return { kind: "generation", canonicalUrl: `https://moonland.ai/generation/${generationId}`, workspaceSlug: null, topicId: null, generationId };
        }
        throw resolverError("INVALID_URL", "Supported paths are Moonland Lab topics and generation detail pages.");
    }

    function valueAt(object, path) {
        let value = object;
        for (const key of path) value = value?.[key];
        return value;
    }

    function firstValue(object, paths, predicate = (value) => value !== undefined && value !== null) {
        for (const path of paths) {
            const value = valueAt(object, path);
            if (predicate(value)) return value;
        }
        return null;
    }

    function targetIdentity(record) {
        const targetId = firstValue(record, [
            ["targetId"], ["target", "id"], ["request", "targetId"], ["requestedSpec", "targetId"],
            ["resolvedSpec", "targetId"], ["descriptor", "targetId"], ["descriptor", "target", "id"],
        ], (value) => typeof value === "string" && IDENTIFIER_PATTERN.test(value));
        const contractRevision = firstValue(record, [
            ["contractRevision"], ["target", "contractRevision"], ["request", "contractRevision"],
            ["requestedSpec", "contractRevision"], ["resolvedSpec", "contractRevision"],
            ["descriptor", "contractRevision"], ["descriptor", "contract", "revision"], ["contract", "revision"],
        ], (value) => Number.isInteger(value) && value > 0);
        return targetId && contractRevision ? { targetId, contractRevision } : null;
    }

    function containsString(root, wanted) {
        if (!wanted) return false;
        const pending = [{ value: root, depth: 0 }];
        let visited = 0;
        while (pending.length && visited < 2000) {
            const { value, depth } = pending.pop();
            visited += 1;
            if (value === wanted) return true;
            if (!value || typeof value !== "object" || depth >= 6) continue;
            for (const child of Array.isArray(value) ? value : Object.values(value)) {
                pending.push({ value: child, depth: depth + 1 });
            }
        }
        return false;
    }

    function directRequestScore(value, requested) {
        if (!value || typeof value !== "object" || Array.isArray(value)) return 0;
        const directValues = [
            value.id,
            value.topicId,
            value.stepId,
            value.generationId,
            value.labContext?.topicId,
            value.labContext?.stepId,
            value.owner?.generationId,
        ];
        let score = 0;
        if (requested.topicId && directValues.includes(requested.topicId)) score += 80;
        if (requested.generationId && directValues.includes(requested.generationId)) score += 140;
        if (requested.preferredStepId && directValues.includes(requested.preferredStepId)) score += 180;
        return score;
    }

    function collectCandidates(pageProps, requested) {
        const candidates = [];
        const pending = [{ value: pageProps, depth: 0, path: "pageProps", inheritedScore: 0, inheritedStepId: null, inheritedTimestamp: null }];
        const seen = new Set();
        let visited = 0;
        while (pending.length) {
            const { value, depth, path, inheritedScore, inheritedStepId, inheritedTimestamp } = pending.pop();
            if (!value || typeof value !== "object" || seen.has(value)) continue;
            seen.add(value);
            visited += 1;
            if (visited > MAX_VISITED_OBJECTS) throw resolverError("CONTRACT_TOO_LARGE", "Moonland page state exceeded the resolver traversal limit.");
            const identity = targetIdentity(value);
            const contextScore = inheritedScore + directRequestScore(value, requested);
            const directStepId = firstValue(value, [["stepId"], ["labContext", "stepId"]], (item) => typeof item === "string" && IDENTIFIER_PATTERN.test(item));
            const containerStepId = !Array.isArray(value) && value.descriptor && typeof value.id === "string" && IDENTIFIER_PATTERN.test(value.id)
                ? value.id
                : null;
            const contextStepId = directStepId ?? containerStepId ?? inheritedStepId;
            const directTimestamp = firstValue(value, [["updatedAt"], ["createdAt"]], (item) => typeof item === "string" && Number.isFinite(Date.parse(item)));
            const contextTimestamp = directTimestamp ? Date.parse(directTimestamp) : inheritedTimestamp;
            if (identity) {
                let score = 10 + contextScore;
                if (containsString(value, requested.topicId)) score += 50;
                if (containsString(value, requested.generationId)) score += 80;
                if (containsString(value, requested.preferredStepId)) score += 60;
                if (value.descriptor || value.contract || value.inputSchema || value.fields) score += 20;
                if (value.requestedSpec || value.resolvedSpec) score += 15;
                candidates.push({ record: value, path, score, contextStepId, contextTimestamp, ...identity });
            }
            if (depth >= MAX_DEPTH) continue;
            const entries = Array.isArray(value) ? value.entries() : Object.entries(value);
            for (const [key, child] of entries) {
                pending.push({
                    value: child,
                    depth: depth + 1,
                    path: `${path}.${String(key)}`,
                    inheritedScore: contextScore,
                    inheritedStepId: contextStepId,
                    inheritedTimestamp: contextTimestamp,
                });
            }
        }
        return candidates;
    }

    function rawFieldArray(record) {
        const candidates = [
            record.fields,
            record.inputs,
            record.inputFields,
            record.inputSchema?.fields,
            record.contract?.fields,
            record.descriptor?.fields,
            record.descriptor?.inputs,
            record.descriptor?.inputSchema?.fields,
            record.descriptor?.contract?.fields,
        ];
        return candidates.find((value) => Array.isArray(value) && value.length > 0) ?? null;
    }

    function candidateShape(candidate) {
        const record = candidate.record;
        const nested = {};
        for (const key of ["descriptor", "contract", "target", "request", "requestedSpec", "resolvedSpec", "inputSchema"]) {
            const value = record[key];
            if (value && typeof value === "object" && !Array.isArray(value)) nested[key] = Object.keys(value).slice(0, 30).sort();
        }
        const arrays = Object.entries(record)
            .filter(([, value]) => Array.isArray(value))
            .slice(0, 20)
            .map(([key, value]) => `${key}[${value.length}]`);
        return {
            target: `${candidate.targetId}@${candidate.contractRevision}`,
            path: candidate.path.slice(0, 240),
            keys: Object.keys(record).slice(0, 40).sort(),
            nested,
            arrays,
        };
    }

    function keyShape(value, depth = 0) {
        if (!value || typeof value !== "object" || depth > 2) return null;
        if (Array.isArray(value)) {
            const firstObject = value.find((item) => item && typeof item === "object");
            return {
                length: value.length,
                ...(firstObject ? { item: keyShape(firstObject, depth + 1) } : {}),
            };
        }
        const result = { keys: Object.keys(value).slice(0, 40).sort() };
        if (depth < 2) {
            const nested = {};
            for (const [key, child] of Object.entries(value)) {
                if (child && typeof child === "object") nested[key] = keyShape(child, depth + 1);
                if (Object.keys(nested).length >= 12) break;
            }
            if (Object.keys(nested).length) result.nested = nested;
        }
        return result;
    }

    function collectDiagnosticShapes(pageProps, requested) {
        const pending = [{ value: pageProps, depth: 0, path: "pageProps" }];
        const seen = new Set();
        const matches = [];
        let visited = 0;
        while (pending.length && visited < MAX_VISITED_OBJECTS) {
            const { value, depth, path } = pending.pop();
            if (!value || typeof value !== "object" || seen.has(value)) continue;
            seen.add(value);
            visited += 1;
            const keys = Array.isArray(value) ? [] : Object.keys(value);
            const interestingKeys = keys.filter((key) => /field|input|schema|contract|parameter|resource|target|revision|step|topic|generation/i.test(key));
            let score = interestingKeys.length * 10;
            if (containsString(value, requested.topicId)) score += 80;
            if (containsString(value, requested.generationId)) score += 100;
            if (containsString(value, requested.preferredStepId)) score += 90;
            if (path.includes("catalogPriceEstimates")) score -= 100;
            if (depth > 2) score += Math.min(depth, 10);
            if (score > 0 && !Array.isArray(value)) {
                matches.push({ path, score, shape: keyShape(value) });
            }
            if (depth >= MAX_DEPTH) continue;
            const entries = Array.isArray(value) ? value.entries() : Object.entries(value);
            for (const [key, child] of entries) pending.push({ value: child, depth: depth + 1, path: `${path}.${String(key)}` });
        }
        return matches
            .sort((left, right) => right.score - left.score || right.path.length - left.path.length)
            .slice(0, 5)
            .map(({ path, shape }) => ({ path: path.slice(0, 240), ...shape }));
    }

    function normalizeRawType(field) {
        const raw = String(field.type ?? field.kind ?? field.valueType ?? field.inputType ?? "").toLowerCase();
        if (raw === "variant") return { type: "enum" };
        const constrainedResourceTypes = Array.isArray(field.constraints?.resourceTypes)
            ? field.constraints.resourceTypes.filter((item) => typeof item === "string").map((item) => item.toUpperCase())
            : [];
        if (raw === "resource") {
            const fieldName = String(field.fieldId ?? field.id ?? "").toLowerCase();
            if (fieldName.includes("mask")) return { type: "resource", mediaType: "MASK" };
            if (fieldName.includes("image") || fieldName.includes("frame")) return { type: "resource", mediaType: "IMAGE" };
            if (fieldName.includes("video")) return { type: "resource", mediaType: "VIDEO" };
            if (fieldName.includes("audio")) return { type: "resource", mediaType: "AUDIO" };
            for (const candidate of ["VIDEO", "AUDIO", "IMAGE", "MASK"]) {
                if (constrainedResourceTypes.some((item) => item.includes(candidate))) return { type: "resource", mediaType: candidate };
            }
            return null;
        }
        if (raw === "model") return { type: "string" };
        const media = String(field.mediaType ?? field.resourceType ?? raw).toUpperCase();
        for (const candidate of ["IMAGE", "VIDEO", "AUDIO", "MASK"]) {
            if (media.includes(candidate)) return { type: "resource", mediaType: candidate };
        }
        if (Array.isArray(field.options) || Array.isArray(field.values) || raw.includes("enum") || raw.includes("select")) return { type: "enum" };
        if (raw.includes("bool")) return { type: "boolean" };
        if (raw.includes("int")) return { type: "integer" };
        if (raw.includes("number") || raw.includes("float") || raw.includes("slider")) return { type: "number" };
        if (raw.includes("array") || raw.includes("list")) return { type: "list" };
        if (raw.includes("object")) return { type: "object" };
        if (raw.includes("string") || raw.includes("text") || raw.includes("prompt")) return { type: "string" };
        return null;
    }

    function scalarOptions(field) {
        const options = field.options ?? field.values ?? field.enum ?? field.constraints?.values ?? field.constraints?.branchIds;
        if (!Array.isArray(options)) return null;
        const values = options.map((item) => item && typeof item === "object" ? (item.value ?? item.id ?? item.key ?? item.name) : item);
        return values.every((item) => ["string", "number", "boolean"].includes(typeof item)) ? values : null;
    }

    function mediaDefaults(mediaType) {
        if (mediaType === "IMAGE") return { maxBytes: 2621440, acceptedContentTypes: ["image/jpeg", "image/png", "image/webp"] };
        if (mediaType === "VIDEO") return { maxBytes: 31457280, acceptedContentTypes: ["video/mp4"] };
        if (mediaType === "AUDIO") return { maxBytes: 10485760, acceptedContentTypes: ["audio/wav", "audio/mpeg", "audio/ogg", "audio/aac", "audio/mp4", "audio/x-m4a", "audio/opus"] };
        return { maxBytes: 2621440, acceptedContentTypes: ["image/png", "image/webp"] };
    }

    function unsupportedFieldShape(rawField, id) {
        const descriptors = {};
        for (const key of ["type", "kind", "valueType", "inputType", "mediaType", "resourceType", "component", "widget"]) {
            const value = rawField[key];
            if (["string", "number", "boolean"].includes(typeof value)) descriptors[key] = value;
        }
        const nested = {};
        for (const key of ["schema", "input", "spec", "definition", "config", "constraints", "metadata", "ui", "type"]) {
            const value = rawField[key];
            if (value && typeof value === "object" && !Array.isArray(value)) nested[key] = Object.keys(value).slice(0, 30).sort();
        }
        const arrays = Object.entries(rawField)
            .filter(([, value]) => Array.isArray(value))
            .slice(0, 12)
            .map(([key, value]) => ({
                key,
                length: value.length,
                itemKeys: value.find((item) => item && typeof item === "object")
                    ? Object.keys(value.find((item) => item && typeof item === "object")).slice(0, 20).sort()
                    : [],
            }));
        return { id, keys: Object.keys(rawField).slice(0, 40).sort(), descriptors, nested, arrays };
    }

    function mergeUniqueScalars(left = [], right = []) {
        return Array.from(new Set([...left, ...right]));
    }

    function mergeNormalizedFields(fields, resourceSlots) {
        const mergedFields = new Map();
        for (const field of fields) {
            const existing = mergedFields.get(field.id);
            if (!existing) {
                mergedFields.set(field.id, { ...field, ...(field.options ? { options: [...field.options] } : {}) });
                continue;
            }
            if (existing.type !== field.type) {
                throw resolverError("CONTRACT_INCOMPLETE", `Moonland field ${field.id} has conflicting branch types.`);
            }
            existing.required = existing.required && field.required;
            if (field.type === "enum") existing.options = mergeUniqueScalars(existing.options, field.options);
            if (existing.label === undefined && field.label !== undefined) existing.label = field.label;
            if (existing.default !== field.default) delete existing.default;
            if (typeof field.minimum === "number") existing.minimum = typeof existing.minimum === "number" ? Math.min(existing.minimum, field.minimum) : field.minimum;
            if (typeof field.maximum === "number") existing.maximum = typeof existing.maximum === "number" ? Math.max(existing.maximum, field.maximum) : field.maximum;
            if (typeof field.maxLength === "number") existing.maxLength = typeof existing.maxLength === "number" ? Math.max(existing.maxLength, field.maxLength) : field.maxLength;
            if (existing.step !== field.step) delete existing.step;
        }

        const mergedSlots = new Map();
        for (const slot of resourceSlots) {
            const existing = mergedSlots.get(slot.id);
            if (!existing) {
                mergedSlots.set(slot.id, {
                    ...slot,
                    purposes: [...slot.purposes],
                    acceptedContentTypes: [...slot.acceptedContentTypes],
                });
                continue;
            }
            if (existing.mediaType !== slot.mediaType || existing.uploadMode !== slot.uploadMode) {
                throw resolverError("CONTRACT_INCOMPLETE", `Moonland resource ${slot.id} has conflicting branch media types.`);
            }
            existing.minCount = Math.min(existing.minCount, slot.minCount);
            existing.maxCount = Math.max(existing.maxCount, slot.maxCount);
            existing.maxBytes = Math.max(existing.maxBytes ?? 0, slot.maxBytes ?? 0) || undefined;
            existing.purposes = mergeUniqueScalars(existing.purposes, slot.purposes);
            existing.acceptedContentTypes = mergeUniqueScalars(existing.acceptedContentTypes, slot.acceptedContentTypes);
            if (existing.label === undefined && slot.label !== undefined) existing.label = slot.label;
            if (existing.normalizationProfile !== slot.normalizationProfile) {
                throw resolverError("CONTRACT_INCOMPLETE", `Moonland resource ${slot.id} has conflicting normalization profiles.`);
            }
        }
        return { fields: Array.from(mergedFields.values()), resourceSlots: Array.from(mergedSlots.values()) };
    }

    function normalizeFields(rawFields) {
        const fields = [];
        const resourceSlots = [];
        const unsupportedFields = [];
        const unsupportedFieldShapes = [];
        for (const rawField of rawFields) {
            if (!rawField || typeof rawField !== "object") continue;
            const id = rawField.id ?? rawField.fieldId ?? rawField.key ?? rawField.name;
            const typeInfo = normalizeRawType(rawField);
            if (typeof id !== "string" || !IDENTIFIER_PATTERN.test(id) || !typeInfo) {
                unsupportedFields.push(typeof id === "string" ? id : "<unknown>");
                unsupportedFieldShapes.push(unsupportedFieldShape(rawField, typeof id === "string" ? id : "<unknown>"));
                continue;
            }
            const required = rawField.required !== undefined
                ? Boolean(rawField.required)
                : String(rawField.presence ?? "").toLowerCase() === "required";
            const field = { id, type: typeInfo.type, required };
            const label = rawField.label ?? rawField.displayName ?? rawField.title;
            if (typeof label === "string" && label) field.label = label.slice(0, 256);
            const mappings = [
                ["default", rawField.default ?? rawField.defaultValue],
                ["minimum", rawField.minimum ?? rawField.min],
                ["maximum", rawField.maximum ?? rawField.max],
                ["step", rawField.step],
                ["maxLength", rawField.maxLength],
            ];
            for (const [key, value] of mappings) if (value !== undefined && value !== null) field[key] = value;
            if (typeInfo.type === "enum") {
                const options = scalarOptions(rawField);
                if (!options?.length) {
                    unsupportedFields.push(id);
                    unsupportedFieldShapes.push(unsupportedFieldShape(rawField, id));
                    continue;
                }
                field.options = options;
            }
            if (typeInfo.type === "resource") {
                const defaults = mediaDefaults(typeInfo.mediaType);
                const minCount = Number.isInteger(rawField.minCount) ? rawField.minCount : (required ? 1 : 0);
                const maxCount = Number.isInteger(rawField.maxCount)
                    ? rawField.maxCount
                    : (Number.isInteger(rawField.constraints?.maxCount) ? rawField.constraints.maxCount : (rawField.multiple ? 10 : 1));
                const purposesSource = rawField.purposes ?? rawField.constraints?.selectionKinds;
                const purposes = Array.isArray(purposesSource) ? purposesSource.filter((item) => typeof item === "string") : [];
                resourceSlots.push({
                    id,
                    ...(field.label ? { label: field.label } : {}),
                    mediaType: typeInfo.mediaType,
                    purposes,
                    minCount,
                    maxCount,
                    acceptedContentTypes: Array.isArray(rawField.acceptedContentTypes ?? rawField.constraints?.acceptedContentTypes)
                        ? (rawField.acceptedContentTypes ?? rawField.constraints.acceptedContentTypes)
                        : defaults.acceptedContentTypes,
                    maxBytes: Number.isInteger(rawField.maxBytes ?? rawField.constraints?.maxBytes)
                        ? (rawField.maxBytes ?? rawField.constraints.maxBytes)
                        : defaults.maxBytes,
                    normalizationProfile: typeInfo.mediaType === "AUDIO" ? "moonland-audio-opus-128k-v1" : "moonland-original-v1",
                    uploadMode: "single",
                });
            }
            fields.push(field);
        }
        const merged = mergeNormalizedFields(fields, resourceSlots);
        return { ...merged, unsupportedFields, unsupportedFieldShapes };
    }

    function inferOutputModality(record) {
        const raw = firstValue(record, [
            ["outputModality"], ["outputType"], ["target", "outputModality"], ["target", "outputType"],
            ["primaryOutputModality"], ["outputModalities", 0],
            ["descriptor", "outputModality"], ["descriptor", "outputType"], ["descriptor", "primaryOutputModality"],
            ["descriptor", "outputModalities", 0], ["descriptor", "target", "outputType"],
        ], (value) => typeof value === "string");
        if (!raw) {
            const identity = targetIdentity(record);
            return KNOWN_TARGET_METADATA[identity?.targetId]?.outputModality ?? "UNKNOWN";
        }
        const upper = raw.toUpperCase();
        return OUTPUT_MODALITIES.has(upper) ? upper : (upper.includes("VIDEO") ? "VIDEO" : upper.includes("IMAGE") ? "IMAGE" : upper.includes("AUDIO") ? "AUDIO" : "UNKNOWN");
    }

    function stableStringify(value) {
        if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
        if (value && typeof value === "object") {
            return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
        }
        return JSON.stringify(value);
    }

    async function sha256Text(value) {
        const bytes = new TextEncoder().encode(value);
        const digest = await crypto.subtle.digest("SHA-256", bytes);
        return Array.from(new Uint8Array(digest), (item) => item.toString(16).padStart(2, "0")).join("");
    }

    async function resolveContractFromPageProps(pageProps, pageUrl, payload) {
        if (!pageProps || typeof pageProps !== "object") throw resolverError("CONTRACT_INCOMPLETE", "Moonland page initial state is unavailable.");
        const requested = { ...canonicalizeMoonlandUrl(payload?.url), preferredStepId: payload?.preferredStepId ?? null };
        const current = new URL(pageUrl);
        if (current.hostname !== "moonland.ai") throw resolverError("TAB_NOT_FOUND", "Resolver is not running in a Moonland tab.");
        const routeWorkspace = pageProps.routeWorkspace ?? null;
        const workspaceId = routeWorkspace?.id;
        const workspaceSlug = routeWorkspace?.slug ?? requested.workspaceSlug;
        if (typeof workspaceId !== "string" || !IDENTIFIER_PATTERN.test(workspaceId) || typeof workspaceSlug !== "string") {
            throw resolverError("CONTRACT_INCOMPLETE", "Moonland workspace identity is unavailable.");
        }
        if (requested.workspaceSlug && requested.workspaceSlug !== workspaceSlug) {
            throw resolverError("TAB_NOT_FOUND", "The open Moonland tab belongs to another workspace.");
        }

        const candidates = collectCandidates(pageProps, requested);
        if (!candidates.length) throw resolverError("CONTRACT_INCOMPLETE", "No complete target identity was found in Moonland page state.");
        const contractCandidates = candidates.filter((item) => rawFieldArray(item.record));
        if (!contractCandidates.length) {
            const shapes = candidates
                .sort((left, right) => right.score - left.score)
                .slice(0, 4)
                .map(candidateShape);
            const nearbyShapes = collectDiagnosticShapes(pageProps, requested);
            const queryEntries = [
                ...(Array.isArray(pageProps.observedQueries) ? pageProps.observedQueries : []),
                ...(Array.isArray(pageProps.onDemandQueries) ? pageProps.onDemandQueries : []),
            ];
            const observedProcedures = queryEntries.slice(-12).map((entry) => ({
                path: entry?.path,
                method: entry?.method,
                captured: entry?.captured,
                resultShapes: Array.isArray(entry?.payload)
                    ? entry.payload.slice(0, 10).map((item) => keyShape(item?.result?.data?.json ?? item?.result?.data ?? item))
                    : [],
            }));
            throw resolverError(
                "CONTRACT_INCOMPLETE",
                `No candidate exposed a supported Moonland field schema. Observed procedures: ${JSON.stringify(observedProcedures).slice(0, 1800)}. Nearby shapes: ${JSON.stringify(nearbyShapes).slice(0, 2600)}. Identity shapes: ${JSON.stringify(shapes).slice(0, 1200)}`,
            );
        }
        contractCandidates.sort((left, right) => right.score - left.score);
        const topScore = contractCandidates[0].score;
        let top = contractCandidates.filter((item) => item.score === topScore);
        if (!requested.preferredStepId && !requested.generationId) {
            const timestamps = top.map((item) => item.contextTimestamp).filter(Number.isFinite);
            if (timestamps.length) {
                const newestTimestamp = Math.max(...timestamps);
                top = top.filter((item) => item.contextTimestamp === newestTimestamp);
            }
        }
        const identities = new Set(top.map((item) => `${item.targetId}:${item.contractRevision}`));
        if (identities.size > 1) {
            const shapes = top.slice(0, 6).map((item) => ({
                target: `${item.targetId}@${item.contractRevision}`,
                path: item.path.slice(0, 240),
                score: item.score,
                stepId: item.contextStepId,
                updatedAt: Number.isFinite(item.contextTimestamp) ? new Date(item.contextTimestamp).toISOString() : null,
                keys: Object.keys(item.record).slice(0, 30).sort(),
            }));
            throw resolverError("CONTRACT_AMBIGUOUS", `Multiple Moonland targets match the requested URL: ${JSON.stringify(shapes).slice(0, 2200)}`);
        }
        const candidate = top.sort((left, right) => (rawFieldArray(right.record)?.length ?? 0) - (rawFieldArray(left.record)?.length ?? 0))[0];
        const rawFields = rawFieldArray(candidate.record);
        if (!rawFields) throw resolverError("CONTRACT_INCOMPLETE", "The selected Moonland target does not expose a supported field schema.");
        const normalized = normalizeFields(rawFields);
        if (normalized.unsupportedFields.length) {
            throw resolverError(
                "CONTRACT_INCOMPLETE",
                `Unsupported Moonland fields: ${normalized.unsupportedFields.join(", ")}. Shapes: ${JSON.stringify(normalized.unsupportedFieldShapes).slice(0, 3600)}`,
            );
        }

        const stepId = firstValue(candidate.record, [["stepId"], ["labContext", "stepId"], ["owner", "labContext", "stepId"]], (value) => typeof value === "string")
            ?? requested.preferredStepId
            ?? (requested.topicId ? candidate.contextStepId : null);
        const generationId = firstValue(candidate.record, [["generationId"], ["owner", "generationId"]], (value) => typeof value === "string" && IDENTIFIER_PATTERN.test(value)) ?? requested.generationId;
        const displayName = firstValue(candidate.record, [["displayName"], ["name"], ["target", "displayName"], ["target", "name"], ["descriptor", "displayName"], ["descriptor", "name"]], (value) => typeof value === "string")
            ?? KNOWN_TARGET_METADATA[candidate.targetId]?.displayName
            ?? candidate.targetId;
        const points = firstValue(candidate.record, [["estimatedPoints"], ["price"], ["acceptedAmount"], ["owner", "acceptedAmount"]], (value) => typeof value === "number" && value >= 0);
        const semantic = {
            schemaVersion: 1,
            workspace: { id: workspaceId, slug: workspaceSlug },
            lab: { topicId: requested.topicId, stepId: stepId ?? null, generationId: generationId ?? null },
            target: {
                id: candidate.targetId,
                contractRevision: candidate.contractRevision,
                displayName: displayName.slice(0, 256),
                outputModality: inferOutputModality(candidate.record),
            },
            fields: normalized.fields,
            resourceSlots: normalized.resourceSlots,
            pricing: { estimatedPoints: points ?? null, currency: "POINT" },
        };
        const contractHash = await sha256Text(stableStringify(semantic));
        return {
            schemaVersion: 1,
            source: { canonicalUrl: requested.canonicalUrl, resolvedAt: new Date().toISOString(), contractHash },
            workspace: semantic.workspace,
            lab: semantic.lab,
            target: semantic.target,
            fields: semantic.fields,
            resourceSlots: semantic.resourceSlots,
            pricing: semantic.pricing,
        };
    }

    globalThis.SaiMoonlandContractResolver = Object.freeze({
        canonicalizeMoonlandUrl,
        normalizeFields,
        resolveContractFromPageProps,
        stableStringify,
    });
})();
