# Local development tools

These Windows scripts are for this repository's isolated ComfyUI and MCP
development workflow. They are not required by end users.

1. Copy `dev_environment.example.bat` to `dev_environment.local.bat`.
2. Replace the placeholder paths with the local ComfyUI workspace and Python
   interpreter. The local file is ignored by Git.
3. Run `setup_mcp.bat` to create the agent-side environment.
4. Use `start_dev_comfyui.bat` to launch only this custom-node package, or
   `start_dev_comfyui.bat --check` for ComfyUI's quick import test.
5. Copy `../../.codex/config.example.toml` to `.codex/config.toml`, replace
   `<PROJECT_ROOT>` with the absolute repository path, and adjust its script
   path to `tools\\dev\\start_comfy_mcp.bat`.

`requirements-mcp.txt` belongs to the agent-side environment and must not be
installed into ComfyUI's Python environment.
