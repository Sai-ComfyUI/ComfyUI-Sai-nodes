const STATUS_INTERVAL_MS = 5000;
const REQUEST_UPLOAD_DELAYS_MS = [500, 1000, 2000, 4000];
const REQUEST_UPLOAD_TIMEOUT_MS = 30000;
const RESOLVER_QUERY_ACTIONS = new Set(["lab.listStepsByTopic"]);
let statusTimer = null;
let extensionContextInvalidated = false;
let lastKnownWorkspaceStatus = null;

function readPageProps() {
    const initialStateScript = Array.from(document.scripts).find((script) => (script.textContent || "").startsWith('{"props"'));
    if (initialStateScript) {
        try {
            return JSON.parse(initialStateScript.textContent).props?.pageProps ?? null;
        } catch {
            return null;
        }
    }
    return null;
}

function readObservedQueries(timeoutMs = 1000) {
    const requestId = crypto.randomUUID();
    return new Promise((resolve) => {
        let settled = false;
        const finish = (entries) => {
            if (settled) return;
            settled = true;
            window.removeEventListener("message", onMessage);
            resolve(Array.isArray(entries) ? entries : []);
        };
        const onMessage = (event) => {
            if (event.source !== window || event.data?.type !== "sai.moonland.readCache.response.v1" || event.data?.requestId !== requestId) return;
            finish(event.data.entries);
        };
        window.addEventListener("message", onMessage);
        window.postMessage({ type: "sai.moonland.readCache.request.v1", requestId }, window.location.origin);
        window.setTimeout(() => finish([]), timeoutMs);
    });
}

function sanitizeResolverPayload(value, key = "", depth = 0, seen = new WeakSet(), schemaContext = false) {
    const contentKey = /^(?:prompt|negativePrompt|text|content|value|defaultValue)$/i;
    const privateKey = /^(?:url|uri|signedUrl|uploadUrl|token|accessToken|refreshToken|authorization|cookie|headers|messages?|resources?|assets?|images?|videos?|audios?)$/i;
    if (privateKey.test(key) || depth > 18) return undefined;
    if (contentKey.test(key) && !schemaContext && (!value || typeof value !== "object")) return undefined;
    if (value === null || ["string", "number", "boolean"].includes(typeof value)) return value;
    if (!value || typeof value !== "object" || seen.has(value)) return undefined;
    seen.add(value);
    if (Array.isArray(value)) {
        return value.slice(0, 500).map((item) => sanitizeResolverPayload(item, "", depth + 1, seen, schemaContext)).filter((item) => item !== undefined);
    }
    const output = {};
    for (const [childKey, child] of Object.entries(value).slice(0, 500)) {
        const childSchemaContext = schemaContext || ["constraints", "schema", "definition", "inputSchema"].includes(childKey);
        const clean = sanitizeResolverPayload(child, childKey, depth + 1, seen, childSchemaContext);
        if (clean !== undefined) output[childKey] = clean;
    }
    return output;
}

async function resolverTrpcQuery(path, input) {
    if (!RESOLVER_QUERY_ACTIONS.has(path)) throw new Error(`Resolver query is not allowlisted: ${path}`);
    const query = new URLSearchParams({ batch: "1", input: JSON.stringify({ "0": { json: input } }) });
    const response = await fetch(`/api/trpc/${path}?${query}`, { method: "GET", credentials: "include", cache: "no-store" });
    let payload;
    try {
        payload = await response.json();
    } catch {
        throw new Error(`${path} returned ${response.status} ${response.statusText}.`);
    }
    if (!response.ok || (Array.isArray(payload) ? payload[0]?.error : payload?.error)) {
        throw new Error(trpcErrorMessage(payload, `${path} failed with HTTP ${response.status}.`));
    }
    return { path: `/api/trpc/${path}`, method: "GET", captured: true, payload: sanitizeResolverPayload(payload) };
}

function readMoonlandStatus() {
    const pageProps = readPageProps();
    const queries = pageProps?.trpcState?.json?.queries ?? [];
    const workspaceQuery = queries.find((query) => query?.queryKey?.[0]?.[0] === "workspace" && query?.queryKey?.[0]?.[1] === "list" && query?.state?.status === "success");
    const routeWorkspace = pageProps?.routeWorkspace ?? null;
    const current = {
        authenticated: Boolean(workspaceQuery),
        pageUrl: window.location.href,
        workspaceId: routeWorkspace?.id ?? null,
        workspaceSlug: routeWorkspace?.slug ?? null,
    };
    if (current.authenticated && current.workspaceId) {
        lastKnownWorkspaceStatus = current;
        return current;
    }
    if (lastKnownWorkspaceStatus?.authenticated && lastKnownWorkspaceStatus.workspaceSlug) {
        const workspacePrefix = `/w/${lastKnownWorkspaceStatus.workspaceSlug}/`;
        if (window.location.pathname.startsWith(workspacePrefix)) {
            return { ...lastKnownWorkspaceStatus, pageUrl: window.location.href };
        }
    }
    return current;
}

function publishStatus() {
    if (extensionContextInvalidated) return;
    try {
        const delivery = chrome.runtime.sendMessage({ type: "moonland.status", status: readMoonlandStatus() });
        delivery?.catch(handleDeliveryError);
    } catch (error) {
        handleDeliveryError(error);
    }
}

function handleDeliveryError(error) {
    if (String(error?.message ?? error).includes("Extension context invalidated")) {
        extensionContextInvalidated = true;
        if (statusTimer !== null) window.clearInterval(statusTimer);
    }
}

function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function decodeBase64Blob(base64, contentType) {
    if (typeof base64 !== "string" || base64.length === 0) {
        throw new Error("Bridge image payload is empty.");
    }
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return new Blob([bytes], { type: contentType });
}

async function blobToImage(blob) {
    const objectUrl = URL.createObjectURL(blob);
    try {
        const image = new Image();
        image.decoding = "async";
        image.src = objectUrl;
        await image.decode();
        return image;
    } finally {
        URL.revokeObjectURL(objectUrl);
    }
}

function isSafari() {
    return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

async function normalizeImage(blob) {
    if (blob.type !== "image/png") return blob;
    const image = await blobToImage(blob);
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Unable to create an image canvas.");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, 0, 0);
    const outputType = isSafari() ? "image/jpeg" : "image/webp";
    const normalized = await new Promise((resolve, reject) => {
        canvas.toBlob(
            (value) => value ? resolve(value) : reject(new Error("Image normalization failed.")),
            outputType,
            0.92,
        );
    });
    return normalized;
}

async function sha256(blob) {
    const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function trpcErrorMessage(payload, fallback) {
    const item = Array.isArray(payload) ? payload[0] : payload;
    return item?.error?.json?.message || item?.error?.message || fallback;
}

async function trpcMutation(path, input) {
    const response = await fetch(`/api/trpc/${path}?batch=1`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "0": { json: input } }),
    });
    let payload;
    try {
        payload = await response.json();
    } catch {
        throw new Error(`${path} returned ${response.status} ${response.statusText}.`);
    }
    if (!response.ok || (Array.isArray(payload) ? payload[0]?.error : payload?.error)) {
        throw new Error(trpcErrorMessage(payload, `${path} failed with HTTP ${response.status}.`));
    }
    const item = Array.isArray(payload) ? payload[0] : payload;
    const data = item?.result?.data;
    return data?.json ?? data;
}

async function requestUpload(input) {
    const startedAt = Date.now();
    let retryIndex = 0;
    for (;;) {
        const response = await trpcMutation("resource.requestUpload", input);
        if (response?.status !== "PENDING") return response;
        const delay = REQUEST_UPLOAD_DELAYS_MS[retryIndex] ?? 5000;
        retryIndex += 1;
        if (Date.now() - startedAt + delay > REQUEST_UPLOAD_TIMEOUT_MS) {
            throw new Error("Moonland resource request remained pending for more than 30 seconds.");
        }
        await sleep(delay);
    }
}

async function uploadSignedBlob(uploadUrl, blob) {
    if (typeof uploadUrl !== "string" || !uploadUrl) throw new Error("Moonland did not return an upload URL.");
    const options = {
        method: "POST",
        headers: { "Content-Type": blob.type },
        body: new File([blob], "resource"),
    };
    let response = await fetch(uploadUrl, options);
    for (let attempt = 0; response.status === 503 && attempt < 3; attempt += 1) {
        response = await fetch(uploadUrl, options);
    }
    if (!response.ok) throw new Error(`Moonland storage upload failed: ${response.status} ${response.statusText}.`);
}

function resourceSummary(resource, details) {
    if (!resource || typeof resource.id !== "string") throw new Error("Moonland returned an invalid resource.");
    return {
        resourceId: resource.id,
        type: resource.type ?? "IMAGE",
        filename: resource.filename ?? details.filename ?? null,
        url: resource.url ?? null,
        width: resource.width ?? null,
        height: resource.height ?? null,
        sha256: details.sha256,
        normalizedSize: details.normalizedSize,
        contentType: details.contentType,
        disposition: details.disposition,
        workspaceId: details.workspaceId,
    };
}

async function ensureImageResource(payload) {
    const status = readMoonlandStatus();
    if (!status.authenticated) throw new Error("The open Moonland tab is not signed in.");
    const workspaceId = payload?.workspaceId || status.workspaceId;
    if (!workspaceId) throw new Error("Moonland workspace ID is unavailable. Open a workspace page and try again.");
    const source = decodeBase64Blob(payload?.base64, payload?.contentType || "image/png");
    const blob = await normalizeImage(source);
    const hash = await sha256(blob);
    const filename = typeof payload?.filename === "string" ? payload.filename.trim().slice(0, 200) : "comfyui-image.png";
    const uploadRequest = {
        filename: filename || "comfyui-image.png",
        workspaceId,
        size: blob.size,
        sha256: hash,
        source: "UPLOAD",
        type: "IMAGE",
        contentType: blob.type,
    };
    const allocation = await requestUpload(uploadRequest);
    const details = {
        filename: uploadRequest.filename,
        sha256: hash,
        normalizedSize: blob.size,
        contentType: blob.type,
        workspaceId,
    };
    if (allocation?.status === "REUSED") {
        return resourceSummary(allocation.resource, { ...details, disposition: "REUSED" });
    }
    await uploadSignedBlob(allocation?.uploadUrl, blob);
    const resource = await trpcMutation("resource.postUpload", { id: allocation?.id, markAsGuaranteed: false });
    return resourceSummary(resource, { ...details, disposition: "UPLOADED" });
}

async function resolveToolContract(payload) {
    const status = readMoonlandStatus();
    if (!status.authenticated) throw new Error("AUTH_REQUIRED: The open Moonland tab is not signed in.");
    const resolver = globalThis.SaiMoonlandContractResolver;
    if (!resolver) throw new Error("CONTRACT_INCOMPLETE: Moonland contract resolver is unavailable.");
    const pageProps = readPageProps();
    const observedQueries = await readObservedQueries();
    const requested = resolver.canonicalizeMoonlandUrl(payload?.url);
    const onDemandQueries = requested.topicId
        ? [await resolverTrpcQuery("lab.listStepsByTopic", { topicId: requested.topicId })]
        : [];
    const resolverState = { ...pageProps, observedQueries, onDemandQueries };
    return resolver.resolveContractFromPageProps(resolverState, window.location.href, payload);
}

async function runAction(action, payload) {
    if (action === "resource.ensureImage") return ensureImageResource(payload);
    if (action === "tool.resolveContract") return resolveToolContract(payload);
    throw new Error(`Unsupported Moonland action: ${String(action)}`);
}

publishStatus();
statusTimer = window.setInterval(publishStatus, STATUS_INTERVAL_MS);
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "moonland.refresh") publishStatus();
    if (message.type === "moonland.action") {
        runAction(message.action, message.payload)
            .then((result) => sendResponse({ ok: true, result }))
            .catch((error) => sendResponse({ ok: false, error: String(error?.message || error) }));
        return true;
    }
    return false;
});
