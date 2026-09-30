# Local development tools

These Windows scripts are for this repository's isolated ComfyUI and MCP
development workflow. They are not required by end users.

1. Copy `dev_environment.example.bat` to `dev_environment.local.bat`.
2. Replace the placeholder paths with the local ComfyUI workspace and Python
   interpreter. The local file is ignored by Git.
3. Run `setup_mcp.bat` to create the agent-side environment.
4. Use `start_updated_dev_comfyui.bat` for the normal development launch. It
   updates the portable ComfyUI Git checkout, updates KJNodes and
   this repository with fast-forward-only Git operations, synchronizes all
   three requirement files, runs an isolated import smoke test, and launches
   only KJNodes plus ComfyUI-Sai-nodes.
5. The default ComfyUI channel is the latest stable tag. Use
   `start_updated_dev_comfyui.bat --edge` to test against the tip of ComfyUI's
   `master` branch, `--check` to synchronize and stop after the import test, or
   `--sync-only` to synchronize without launching.
6. `start_dev_comfyui.bat` remains the no-update launcher for offline work or
   for quickly restarting the already-synchronized environment.
7. Copy `../../.codex/config.example.toml` to `.codex/config.toml`, replace
   `<PROJECT_ROOT>` with the absolute repository path, and adjust its script
   path to `tools\\dev\\start_comfy_mcp.bat`.

`requirements-mcp.txt` belongs to the agent-side environment and must not be
installed into ComfyUI's Python environment.

The updater never stashes or overwrites local changes in ComfyUI-Sai-nodes. A
dirty Sai-nodes working tree is treated as the active development version and
its remote update is skipped. Local changes in third-party repositories stop
the update. The first run preserves a Manager-installed, non-Git KJNodes folder
under `.dev/backups/` before cloning the official repository.

ComfyUI-Manager is intentionally not part of this environment. It may remain
installed on disk, but the launcher's custom-node allowlist prevents it from
loading. This keeps the development runtime limited to ComfyUI core, KJNodes,
and ComfyUI-Sai-nodes.

If the machine requires a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` in
`dev_environment.local.bat`. The values are inherited by Git and pip. System
Git is used for repository updates because the portable updater's `pygit2`
transport does not reliably honor this proxy setup.

Both launchers check whether the configured port is already occupied before
starting. Close the existing ComfyUI process before updating this environment.
When startup or synchronization fails, a double-clicked command window pauses
so the error remains visible.
