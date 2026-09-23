import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(root, "tools", "moonland-bridge-extension", "contract_resolver.js"), "utf8");
const context = { URL, TextEncoder, crypto, Date, console };
context.globalThis = context;
vm.runInNewContext(source, context, { filename: "contract_resolver.js" });
const resolver = context.SaiMoonlandContractResolver;
assert.ok(resolver, "resolver API was not installed");

const fixtures = JSON.parse(fs.readFileSync(path.join(root, "tests", "fixtures", "moonland_contracts.json"), "utf8"));
const kreaSnapshot = JSON.parse(fs.readFileSync(path.join(root, "tests", "fixtures", "moonland_krea2_normalized.json"), "utf8"));
for (const fixture of fixtures) {
    const contract = await resolver.resolveContractFromPageProps(fixture.pageProps, fixture.pageUrl, fixture.request);
    assert.equal(contract.target.id, fixture.expected.targetId, fixture.name);
    assert.equal(contract.target.contractRevision, fixture.expected.revision, fixture.name);
    assert.equal(contract.target.outputModality, fixture.expected.modality, fixture.name);
    assert.deepEqual(Array.from(contract.resourceSlots, (slot) => slot.mediaType), fixture.expected.resourceTypes, fixture.name);
    assert.match(contract.source.contractHash, /^[0-9a-f]{64}$/);
    assert.equal(contract.source.canonicalUrl, resolver.canonicalizeMoonlandUrl(fixture.request.url).canonicalUrl);
}

const kreaContract = await resolver.resolveContractFromPageProps(fixtures[0].pageProps, fixtures[0].pageUrl, fixtures[0].request);
assert.equal(kreaContract.source.contractHash, kreaSnapshot.source.contractHash, "cross-runtime semantic hash snapshot");
assert.deepEqual(
    JSON.parse(JSON.stringify({ ...kreaContract, source: { ...kreaContract.source, resolvedAt: kreaSnapshot.source.resolvedAt } })),
    kreaSnapshot,
    "normalized Krea snapshot",
);

assert.throws(
    () => resolver.canonicalizeMoonlandUrl("https://example.com/w/example/lab?t=topic-id"),
    /INVALID_URL/,
);
assert.throws(
    () => resolver.canonicalizeMoonlandUrl("https://moonland.ai/w/example/lab?t=topic-id&procedure=lab.submit"),
    /INVALID_URL/,
);
assert.deepEqual(
    JSON.parse(JSON.stringify(resolver.canonicalizeMoonlandUrl("https://moonland.ai/w/purple-bright-coyote/gen?g=cmuau8th800es01s63xsq2k7b"))),
    {
        kind: "generation",
        canonicalUrl: "https://moonland.ai/w/purple-bright-coyote/gen?g=cmuau8th800es01s63xsq2k7b",
        workspaceSlug: "purple-bright-coyote",
        topicId: null,
        generationId: "cmuau8th800es01s63xsq2k7b",
    },
);
assert.throws(
    () => resolver.canonicalizeMoonlandUrl("https://moonland.ai/w/example/gen?g=generation-id&procedure=generation.create"),
    /INVALID_URL/,
);

const generationId = "cmuau8th800es01s63xsq2k7b";
const generationContract = await resolver.resolveContractFromPageProps(
    {
        routeWorkspace: { id: "workspace-id", slug: "example" },
        generation: {
            id: generationId,
            descriptor: {
                targetId: "aio-krea2",
                contractRevision: 7,
                outputModality: "IMAGE",
                fields: [{ id: "prompt", kind: "string" }],
            },
        },
    },
    `https://moonland.ai/w/example/gen?g=${generationId}`,
    { url: `https://moonland.ai/w/example/gen?g=${generationId}` },
);
assert.equal(generationContract.lab.generationId, generationId, "generation detail preserves the requested generation ID");
assert.equal(generationContract.lab.stepId, null, "generation container IDs must not be inferred as Lab step IDs");

const catalogNoise = JSON.parse(JSON.stringify(fixtures[0]));
catalogNoise.pageProps.catalog = {
    topicId: catalogNoise.pageProps.trpcState.json.queries[0].state.data.labContext.topicId,
    targetId: "unrelated-catalog-target",
    contractRevision: 99,
};
const resolvedWithNoise = await resolver.resolveContractFromPageProps(
    catalogNoise.pageProps,
    catalogNoise.pageUrl,
    catalogNoise.request,
);
assert.equal(resolvedWithNoise.target.id, "aio-krea2", "catalog records without fields must not create ambiguity");

const preferredStepState = {
    routeWorkspace: { id: "workspace-id", slug: "example" },
    steps: [
        {
            id: "other-step-id",
            topicId: "topic-id",
            nested: {
                requestedSpec: { targetId: "other-target", contractRevision: 2 },
                descriptor: { outputType: "IMAGE", fields: [{ id: "otherPrompt", type: "string" }] },
            },
        },
        {
            id: "preferred-step-id",
            topicId: "topic-id",
            nested: {
                requestedSpec: { targetId: "preferred-target", contractRevision: 3 },
                descriptor: { outputType: "VIDEO", fields: [{ id: "preferredPrompt", type: "string" }] },
            },
        },
    ],
};
const preferredContract = await resolver.resolveContractFromPageProps(
    preferredStepState,
    "https://moonland.ai/w/example/lab?t=topic-id",
    { url: "https://moonland.ai/w/example/lab?t=topic-id", preferredStepId: "preferred-step-id" },
);
assert.equal(preferredContract.target.id, "preferred-target", "preferred step context must propagate to its nested contract");

const latestStepContract = await resolver.resolveContractFromPageProps(
    {
        routeWorkspace: { id: "workspace-id", slug: "example" },
        steps: [
            {
                id: "newer-step-id",
                topicId: "topic-id",
                createdAt: "2026-09-23T08:00:00.000Z",
                descriptor: {
                    targetId: "newer-target",
                    contractRevision: 7,
                    primaryOutputModality: "IMAGE",
                    inputFields: [{ fieldId: "prompt", kind: "string" }],
                },
            },
            {
                id: "older-step-id",
                topicId: "topic-id",
                createdAt: "2026-09-22T08:00:00.000Z",
                descriptor: {
                    targetId: "older-target",
                    contractRevision: 9,
                    primaryOutputModality: "VIDEO",
                    inputFields: [{ fieldId: "prompt", kind: "string" }],
                },
            },
        ],
    },
    "https://moonland.ai/w/example/lab?t=topic-id",
    { url: "https://moonland.ai/w/example/lab?t=topic-id" },
);
assert.equal(latestStepContract.target.id, "newer-target", "topic URLs default to the newest complete step, not the highest revision");
assert.equal(latestStepContract.lab.stepId, "newer-step-id");
assert.equal(latestStepContract.target.outputModality, "IMAGE");

const unsupported = resolver.normalizeFields([
    { id: "customField", type: "moonland-special", schema: { schemaKind: "secret-descriptor-value" }, config: { controlKind: "secret-control-value" } },
]);
const unsupportedDiagnostic = JSON.stringify(unsupported.unsupportedFieldShapes);
assert.match(unsupportedDiagnostic, /schemaKind/);
assert.match(unsupportedDiagnostic, /controlKind/);
assert.doesNotMatch(unsupportedDiagnostic, /secret-descriptor-value|secret-control-value/);

const liveFieldShapes = resolver.normalizeFields([
    { fieldId: "aspectRatio", kind: "enum", presence: "required", constraints: { values: ["1:1", "16:9"] } },
    { fieldId: "checkpoint", kind: "model", presence: "required", constraints: { baselines: ["FLUX"], modelTypes: ["CHECKPOINT"] } },
    { fieldId: "inputImage", kind: "resource", presence: "optional", constraints: { resourceTypes: ["VIDEO", "IMAGE"], selectionKinds: ["video-frame"] } },
    { fieldId: "maskImage", kind: "resource", nullable: true, constraints: { resourceTypes: ["IMAGE"], selectionKinds: ["MASK"] } },
]);
assert.deepEqual(Array.from(liveFieldShapes.fields, (field) => field.type), ["enum", "string", "resource", "resource"]);
assert.deepEqual(Array.from(liveFieldShapes.fields[0].options), ["1:1", "16:9"]);
assert.equal(liveFieldShapes.fields[0].required, true);
assert.deepEqual(Array.from(liveFieldShapes.resourceSlots, (slot) => slot.mediaType), ["IMAGE", "MASK"]);
assert.deepEqual(Array.from(liveFieldShapes.resourceSlots[0].purposes), ["video-frame"]);
assert.deepEqual(Array.from(liveFieldShapes.unsupportedFields), []);

const variantField = resolver.normalizeFields([
    { fieldId: "inputMode", kind: "variant", presence: "required", constraints: { branchIds: ["TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"], discriminator: "inputMode" } },
]);
assert.equal(variantField.fields[0].type, "enum");
assert.deepEqual(Array.from(variantField.fields[0].options), ["TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"]);

const branchMerged = resolver.normalizeFields([
    { fieldId: "mode", kind: "enum", presence: "required", constraints: { values: ["text", "image"] } },
    { fieldId: "mode", kind: "enum", presence: "optional", constraints: { values: ["image", "mask"] } },
    { fieldId: "inputImage", kind: "resource", presence: "required", constraints: { resourceTypes: ["IMAGE"], selectionKinds: ["REFERENCE"] } },
    { fieldId: "inputImage", kind: "resource", presence: "optional", constraints: { resourceTypes: ["IMAGE"], selectionKinds: ["START_FRAME"], maxCount: 2 } },
]);
assert.equal(branchMerged.fields.length, 2);
assert.equal(branchMerged.resourceSlots.length, 1);
assert.equal(branchMerged.fields.find((field) => field.id === "mode").required, false);
assert.deepEqual(Array.from(branchMerged.fields.find((field) => field.id === "mode").options), ["text", "image", "mask"]);
assert.deepEqual(Array.from(branchMerged.resourceSlots[0].purposes), ["REFERENCE", "START_FRAME"]);
assert.equal(branchMerged.resourceSlots[0].minCount, 0);
assert.equal(branchMerged.resourceSlots[0].maxCount, 2);

assert.throws(
    () => resolver.normalizeFields([
        { fieldId: "conflict", kind: "string" },
        { fieldId: "conflict", kind: "resource", constraints: { resourceTypes: ["IMAGE"] } },
    ]),
    /conflicting branch types/,
);

const knownTargetFallback = await resolver.resolveContractFromPageProps(
    {
        routeWorkspace: { id: "workspace-id", slug: "example" },
        active: {
            topicId: "topic-id",
            stepId: "step-id",
            requestedSpec: { targetId: "aio-krea2", contractRevision: 7 },
            descriptor: { fields: [{ fieldId: "prompt", kind: "string" }] },
        },
    },
    "https://moonland.ai/w/example/lab?t=topic-id",
    { url: "https://moonland.ai/w/example/lab?t=topic-id", preferredStepId: "step-id" },
);
assert.equal(knownTargetFallback.target.displayName, "Krea 2");
assert.equal(knownTargetFallback.target.outputModality, "IMAGE");

const shapeOnly = {
    routeWorkspace: { id: "workspace-id", slug: "example" },
    active: { topicId: "topic-id", targetId: "shape-target", contractRevision: 3, privateSchema: { widgets: [] } },
};
await assert.rejects(
    resolver.resolveContractFromPageProps(shapeOnly, "https://moonland.ai/w/example/lab?t=topic-id", { url: "https://moonland.ai/w/example/lab?t=topic-id" }),
    (error) => /CONTRACT_INCOMPLETE/.test(error.message) && /privateSchema/.test(error.message) && !/workspace-id/.test(error.message),
    "shape diagnostics expose keys, not unrelated values",
);

console.log(JSON.stringify({ passed: fixtures.length, names: fixtures.map((item) => item.name) }));
