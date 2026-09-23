# Moonland Bridge Extension

This unpacked Manifest V3 extension reports the authenticated state of an open
Moonland tab and performs explicitly allowlisted resource actions for ComfyUI.
It never reads, exports, or stores Moonland cookies.

## Usage

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable developer mode.
3. Choose **Load unpacked** and select this directory.
4. Keep a signed-in `https://moonland.ai/` tab open.
5. In ComfyUI, run **Sai → Moonland Bridge → Pair Moonland Bridge**.
6. In ComfyUI choose **Sai > Moonland Bridge > Pair Moonland Bridge**. The
   one-time code is copied automatically; open the extension, paste it, and connect.
7. In ComfyUI, run **Check Moonland Bridge**.

For this repository's development launcher, the ComfyUI URL is normally
`http://127.0.0.1:9527`. After extension files change, click **Reload** for the
unpacked extension on `chrome://extensions` or `edge://extensions` before
testing again.

Reloading an unpacked extension invalidates content scripts already running in
open pages. Refresh every open Moonland tab after clicking extension **Reload**.

The extension also requires the target ComfyUI page to be open in the same
browser. Its in-page bridge opens the WebSocket from the ComfyUI origin, keeping
ComfyUI's cross-origin protection enabled.

Only the ComfyUI URL is stored by the extension. The pairing code is single-use
and is never persisted.

Version 0.6.2 gives a requested Lab step priority over other complete contracts in
the same topic, including when the contract is nested below the step object.

Version 0.6.1 resolves the latest step list through a fixed, read-only
`lab.listStepsByTopic` GET query, so browser query caching cannot hide the contract.

Version 0.5.1 recognizes read-only tRPC procedure names independently of their HTTP
transport method. It can therefore inspect a sanitized response delivered over POST,
while mutation-like procedure names (create, submit, generate, upload, update, delete,
and similar) are metadata-only and never inspected. The observer never replays a request.

Version 0.5.0 added a page-start observer for Moonland's already completed queries.
It strips prompt/text values, resource URLs, credentials, and similar content before
exposing the bounded cache to `tool.resolveContract`.

Version 0.4.0 added the read-only `tool.resolveContract` action alongside
`resource.ensureImage`. The resolver only accepts canonical Moonland Lab topic
or generation URLs, selects the exact open Moonland tab, and returns a bounded,
allowlisted contract summary. It does not expose raw page state or an arbitrary
HTTP/tRPC proxy.

Image upload still hashes the same browser-normalized bytes used by Moonland's
uploader, returns an existing resource when possible, and uploads only when the
resource is absent. The extension cannot submit generations.
