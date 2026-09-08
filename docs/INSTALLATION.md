# Installation and updates

## Requirements

- A recent ComfyUI installation that provides the V3 `comfy_api.latest` API.
- Python 3.10 or newer, using the interpreter that runs ComfyUI.
- Git for clone-based installation.
- Enough disk space for optional model files. The complete MyTimeMachine set is
  approximately 5.65 GB.

## Install with Git

From the ComfyUI directory:

```bash
cd custom_nodes
git clone https://github.com/Sai-ComfyUI/ComfyUI-Sai-nodes.git
python -m pip install -r ComfyUI-Sai-nodes/requirements.txt
```

If `python` is not the interpreter used by ComfyUI, replace it with the exact
interpreter path used to launch ComfyUI.

### Windows portable build

From the portable installation root:

```bat
git clone https://github.com/Sai-ComfyUI/ComfyUI-Sai-nodes.git ComfyUI\custom_nodes\ComfyUI-Sai-nodes
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Sai-nodes\requirements.txt
```

Do not install the requirements into a separate system Python environment.

## Install the models

Most nodes do not require additional weights. LUA-FLUX can download its own
checkpoint. MyTimeMachine uses manually downloaded files; follow
[Model downloads](MODELS.md) and place them in the documented directory.

## Start and verify

Restart ComfyUI and search for `Ψ` or browse the `Sai` category. The server log
should not report an import failure for `ComfyUI-Sai-nodes`.

If the package does not load:

1. Confirm the repository directory is directly under `custom_nodes` and
   contains this file's sibling `../__init__.py`.
2. Re-run the dependency command with ComfyUI's Python interpreter.
3. Update ComfyUI if `comfy_api.latest` cannot be imported.
4. For an OpenCV conflict, keep only one OpenCV wheel family (`opencv-python`,
   `opencv-contrib-python`, or a headless equivalent) in the environment. The
   nodes require an OpenCV build that provides `FaceDetectorYN`.

## Update

```bash
cd ComfyUI/custom_nodes/ComfyUI-Sai-nodes
git pull --ff-only
python -m pip install -r requirements.txt
```

Restart ComfyUI after updating. Model weights are separate from Git and are not
removed by a normal update.

## Uninstall

Stop ComfyUI, remove the `ComfyUI-Sai-nodes` directory from `custom_nodes`, then
start ComfyUI again. Model weights stored inside the package directory should be
backed up first if you want to keep them.
