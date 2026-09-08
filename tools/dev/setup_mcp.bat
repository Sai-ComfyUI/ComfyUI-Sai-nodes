@echo off
setlocal

set "TOOLS_ROOT=%~dp0"
for %%I in ("%TOOLS_ROOT%..\..") do set "PROJECT_ROOT=%%~fI\"
set "MCP_ENV=%PROJECT_ROOT%.venv-mcp"
set "LOCAL_CONFIG=%TOOLS_ROOT%dev_environment.local.bat"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv was not found on PATH.
    exit /b 1
)

if not exist "%MCP_ENV%\Scripts\python.exe" (
    echo [INFO] Creating MCP environment...
    uv venv "%MCP_ENV%" --python 3.12
    if errorlevel 1 exit /b 1
)

echo [INFO] Installing MCP tool requirements...
uv pip install --python "%MCP_ENV%\Scripts\python.exe" -r "%TOOLS_ROOT%requirements-mcp.txt"
if errorlevel 1 exit /b 1

if not exist "%LOCAL_CONFIG%" (
    echo [ERROR] Missing %LOCAL_CONFIG%
    echo Copy "%TOOLS_ROOT%dev_environment.example.bat" to "%LOCAL_CONFIG%" first.
    exit /b 1
)

call "%LOCAL_CONFIG%"
if not defined COMFYUI_WORKSPACE (
    echo [ERROR] COMFYUI_WORKSPACE is not defined.
    exit /b 1
)
if not exist "%COMFYUI_WORKSPACE%\main.py" (
    echo [ERROR] Invalid ComfyUI workspace: %COMFYUI_WORKSPACE%
    exit /b 1
)

echo [INFO] Registering the existing ComfyUI workspace with comfy-cli...
"%MCP_ENV%\Scripts\comfy.exe" set-default "%COMFYUI_WORKSPACE%" --where local
if errorlevel 1 exit /b 1

echo [OK] MCP environment is ready.
"%MCP_ENV%\Scripts\comfy.exe" --version
exit /b %ERRORLEVEL%
