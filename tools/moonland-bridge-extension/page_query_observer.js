(() => {
    const REQUEST_TYPE = "sai.moonland.readCache.request.v1";
    const RESPONSE_TYPE = "sai.moonland.readCache.response.v1";
    const MAX_ENTRIES = 24;
    const MAX_SERIALIZED_BYTES = 512 * 1024;
    const CONTENT_KEYS = /^(?:prompt|negativePrompt|text|content|value|defaultValue)$/i;
    const ALWAYS_REDACTED_KEYS = /^(?:url|uri|signedUrl|uploadUrl|token|accessToken|refreshToken|authorization|cookie|headers|messages?|resources?|assets?|images?|videos?|audios?)$/i;
    const MUTATION_PREFIXES = ["create", "submit", "generate", "run", "execute", "mutate", "update", "delete", "remove", "upload", "requestupload", "postupload", "publish", "unpublish", "cancel", "retry", "favorite", "set", "edit", "archive", "restore", "duplicate", "move", "rename", "invite", "revoke", "login", "signout"];
    const READ_PREFIXES = ["get", "list", "find", "search", "read", "load", "fetch", "resolve", "describe", "contract", "history", "status", "detail", "topic", "step"];
    const cache = [];

    function sanitize(value, key = "", depth = 0, seen = new WeakSet(), schemaContext = false) {
        if (ALWAYS_REDACTED_KEYS.test(key) || depth > 18) return undefined;
        if (CONTENT_KEYS.test(key) && !schemaContext && (!value || typeof value !== "object")) return undefined;
        if (value === null || ["string", "number", "boolean"].includes(typeof value)) return value;
        if (!value || typeof value !== "object" || seen.has(value)) return undefined;
        seen.add(value);
        if (Array.isArray(value)) {
            return value.slice(0, 500).map((item) => sanitize(item, "", depth + 1, seen, schemaContext)).filter((item) => item !== undefined);
        }
        const output = {};
        for (const [childKey, child] of Object.entries(value).slice(0, 500)) {
            const childSchemaContext = schemaContext || ["constraints", "schema", "definition", "inputSchema"].includes(childKey);
            const clean = sanitize(child, childKey, depth + 1, seen, childSchemaContext);
            if (clean !== undefined) output[childKey] = clean;
        }
        return output;
    }

    function requestDetails(url, method) {
        try {
            const parsed = new URL(url, window.location.href);
            if (parsed.origin !== window.location.origin || !parsed.pathname.startsWith("/api/trpc/")) return null;
            const procedurePath = decodeURIComponent(parsed.pathname.slice("/api/trpc/".length));
            const procedures = procedurePath.split(",").filter(Boolean);
            const actions = procedures.map((name) => name.split(".").at(-1).toLowerCase());
            const safeToCapture = procedures.length > 0
                && actions.every((action) => READ_PREFIXES.some((prefix) => action.startsWith(prefix)))
                && actions.every((action) => MUTATION_PREFIXES.every((prefix) => !action.startsWith(prefix)));
            return { path: parsed.pathname, method: String(method || "GET").toUpperCase(), safeToCapture };
        } catch {
            return null;
        }
    }

    async function capture(url, method, response) {
        const details = requestDetails(url, method);
        if (!details) return;
        if (!details.safeToCapture || !response?.ok) {
            cache.push({ path: details.path, method: details.method, captured: false });
            if (cache.length > MAX_ENTRIES) cache.splice(0, cache.length - MAX_ENTRIES);
            return;
        }
        try {
            const payload = sanitize(await response.clone().json());
            const serialized = JSON.stringify(payload);
            if (!serialized || serialized.length > MAX_SERIALIZED_BYTES) return;
            cache.push({ path: details.path, method: details.method, captured: true, payload });
            if (cache.length > MAX_ENTRIES) cache.splice(0, cache.length - MAX_ENTRIES);
        } catch {
            // Non-JSON and malformed responses are irrelevant to contract discovery.
        }
    }

    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
        const request = args[0];
        const init = args[1] || {};
        const url = typeof request === "string" || request instanceof URL ? String(request) : request?.url;
        const method = init.method || request?.method || "GET";
        const response = await originalFetch(...args);
        void capture(url, method, response);
        return response;
    };

    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        this.__saiMoonlandReadRequest = { method, url };
        return originalOpen.call(this, method, url, ...rest);
    };
    XMLHttpRequest.prototype.send = function(...args) {
        const request = this.__saiMoonlandReadRequest;
        const details = request ? requestDetails(request.url, request.method) : null;
        if (details) {
            this.addEventListener("load", () => {
                if (!details.safeToCapture || this.status < 200 || this.status >= 300) {
                    cache.push({ path: details.path, method: details.method, captured: false });
                    if (cache.length > MAX_ENTRIES) cache.splice(0, cache.length - MAX_ENTRIES);
                    return;
                }
                try {
                    const payload = sanitize(JSON.parse(this.responseText));
                    const serialized = JSON.stringify(payload);
                    if (!serialized || serialized.length > MAX_SERIALIZED_BYTES) return;
                    cache.push({ path: details.path, method: details.method, captured: true, payload });
                    if (cache.length > MAX_ENTRIES) cache.splice(0, cache.length - MAX_ENTRIES);
                } catch {
                    // Ignore non-JSON responses.
                }
            }, { once: true });
        }
        return originalSend.apply(this, args);
    };

    window.addEventListener("message", (event) => {
        if (event.source !== window || event.data?.type !== REQUEST_TYPE || typeof event.data?.requestId !== "string") return;
        window.postMessage({ type: RESPONSE_TYPE, requestId: event.data.requestId, entries: cache }, window.location.origin);
    });
})();
