# Moonland Resolve Tool Ψ

Resolves an existing Moonland Lab topic or generation URL through the paired,
signed-in browser tab. The node is read-only: it cannot upload media, submit a
generation, or spend points.

## Inputs

- `url`: either `https://moonland.ai/w/<workspace>/lab?t=<topic>` or
  `https://moonland.ai/w/<workspace>/gen?g=<generation-id>`. The legacy
  `https://moonland.ai/generation/<generation-id>` shape is also accepted. A Lab URL may include one
  `g` generation ID. Other hosts, schemes, credentials, paths, and query
  parameters are rejected.
- `preferred_step_id`: optional exact Lab step ID when a topic contains more
  than one usable step.

The exact URL must already be open in a signed-in Moonland tab. This prevents
the bridge from resolving a similarly named topic from another workspace or
browser tab.

## Outputs

- `tool_contract`: versioned `MOONLAND_TOOL_CONTRACT` containing the workspace,
  Lab identifiers, target ID, contract revision, normalized fields, media
  resource slots, and a semantic SHA-256 hash.
- `target_id`, `contract_revision`, and `output_modality`: convenient scalar
  outputs for inspection and routing.
- `summary_json`: readable form of the normalized contract. It does not contain
  cookies, request headers, signed upload URLs, or the raw Moonland page state.

The resolver reruns whenever its downstream path executes because Moonland can
change a private contract without changing the workflow inputs. `resolvedAt`
does not affect `contractHash`; only semantic contract changes do.

## Failure behavior

The resolver fails closed when the page is signed out, the exact tab is not
open, multiple targets match, required schema data is absent, an unsupported
field appears, or the contract exceeds the bridge size limit. It never guesses
a new revision or silently removes an unsupported field.
