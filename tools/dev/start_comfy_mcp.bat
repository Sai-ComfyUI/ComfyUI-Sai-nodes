@echo off
setlocal

set "TOOLS_ROOT=%~dp0"
for %%I in ("%TOOLS_ROOT%..\..") do set "PROJECT_ROOT=%%~fI\"
set "LOCAL_CONFIG=%TOOLS_ROOT%dev_environment.local.bat"

if not exist "%LOCAL_CONFIG%" (
    echo [ERROR] Missing local configuration: "%LOCAL_CONFIG%" 1>&2
    echo [ERROR] Copy "%TOOLS_ROOT%dev_environment.example.bat" and configure local paths first. 1>&2
    exit /b 1
)

call "%LOCAL_CONFIG%"
if errorlevel 1 exit /b %errorlevel%

if not defined COMFYUI_HOST set "COMFYUI_HOST=127.0.0.1"
if not defined COMFYUI_PORT set "COMFYUI_PORT=9527"

set "COMFY_LOCAL_URL=http://%COMFYUI_HOST%:%COMFYUI_PORT%"
set "NO_PROXY=127.0.0.1,localhost,::1"
set "no_proxy=127.0.0.1,localhost,::1"
set "COMFY_BIN=%PROJECT_ROOT%.venv-mcp\Scripts\comfy.exe"
set "COMFY_PROJECT=%PROJECT_ROOT:~0,-1%"
set "COMFY_MCP_DEBUG_LOG=%PROJECT_ROOT%.dev\comfy-mcp\failures.jsonl"

if not exist "%COMFY_BIN%" (
    echo [ERROR] Missing comfy-cli: "%COMFY_BIN%" 1>&2
    echo [ERROR] Run setup_mcp.bat first. 1>&2
    exit /b 1
)

if not exist "%PROJECT_ROOT%.venv-mcp\Scripts\python.exe" (
    echo [ERROR] Missing MCP Python environment. Run setup_mcp.bat first. 1>&2
    exit /b 1
)

if /i "%~1"=="--check" (
    "%COMFY_BIN%" --where local --json env
    exit /b %errorlevel%
)

"%PROJECT_ROOT%.venv-mcp\Scripts\python.exe" -c "from comfy_mcp.server import main; main()"
exit /b %errorlevel%
