@echo off
title ComfyUI Dev 9527
setlocal

set "TOOLS_ROOT=%~dp0"
for %%I in ("%TOOLS_ROOT%..\..") do set "PROJECT_ROOT=%%~fI\"
set "LOCAL_CONFIG=%TOOLS_ROOT%dev_environment.local.bat"
set "DEV_ROOT=%PROJECT_ROOT%.dev"
set "COMFYUI_HOST=127.0.0.1"
set "COMFYUI_PORT=9527"
set "CHECK_ARGS="
set "DATABASE_URL="
set "LAUNCH_MODE=run"

if not exist "%LOCAL_CONFIG%" (
    echo [ERROR] Missing %LOCAL_CONFIG%
    echo Copy "%TOOLS_ROOT%dev_environment.example.bat" to "%LOCAL_CONFIG%" first.
    exit /b 1
)

call "%LOCAL_CONFIG%"

if /I "%~1"=="--check" (
    set "LAUNCH_MODE=check"
    set "CHECK_ARGS=--quick-test-for-ci"
    set "DATABASE_URL=sqlite:///:memory:"
)
if /I "%~1"=="--preflight" set "LAUNCH_MODE=preflight"

if not defined COMFYUI_ROOT (
    echo [ERROR] COMFYUI_ROOT is not defined.
    exit /b 1
)
if not defined COMFYUI_PYTHON (
    echo [ERROR] COMFYUI_PYTHON is not defined.
    exit /b 1
)
if not exist "%COMFYUI_ROOT%\main.py" (
    echo [ERROR] ComfyUI main.py was not found under %COMFYUI_ROOT%
    exit /b 1
)
if not exist "%COMFYUI_PYTHON%" (
    echo [ERROR] ComfyUI Python was not found: %COMFYUI_PYTHON%
    exit /b 1
)
if not exist "%COMFYUI_ROOT%\custom_nodes\ComfyUI-Sai-nodes\AGENTS.md" (
    echo [ERROR] The ComfyUI-Sai-nodes development link is missing or points elsewhere.
    exit /b 1
)

if /I not "%LAUNCH_MODE%"=="check" (
    powershell.exe -NoProfile -Command "$client=[Net.Sockets.TcpClient]::new(); $inUse=$false; try { $client.Connect('%COMFYUI_HOST%', %COMFYUI_PORT%); $inUse=$true } catch {} finally { $client.Dispose() }; if ($inUse) { exit 1 } else { exit 0 }" >nul 2>&1
    if errorlevel 1 goto port_in_use
)
if /I "%LAUNCH_MODE%"=="preflight" exit /b 0

for %%D in ("%DEV_ROOT%" "%DEV_ROOT%\user" "%DEV_ROOT%\input" "%DEV_ROOT%\output" "%DEV_ROOT%\temp") do (
    if not exist "%%~D" mkdir "%%~D"
    if errorlevel 1 exit /b 1
)

set "DEV_DB_PATH=%DEV_ROOT:\=/%/user/comfyui.db"
if not defined DATABASE_URL set "DATABASE_URL=sqlite:///%DEV_DB_PATH%"

echo [INFO] ComfyUI root: %COMFYUI_ROOT%
echo [INFO] Python: %COMFYUI_PYTHON%
echo [INFO] URL: http://%COMFYUI_HOST%:%COMFYUI_PORT%
echo [INFO] Custom nodes: comfyui-kjnodes and ComfyUI-Sai-nodes

:run_comfyui
pushd "%COMFYUI_ROOT%"
"%COMFYUI_PYTHON%" -s main.py ^
    --windows-standalone-build ^
    --listen "%COMFYUI_HOST%" ^
    --port "%COMFYUI_PORT%" ^
    --disable-all-custom-nodes ^
    --whitelist-custom-nodes comfyui-kjnodes ComfyUI-Sai-nodes ^
    --user-directory "%DEV_ROOT%\user" ^
    --input-directory "%DEV_ROOT%\input" ^
    --output-directory "%DEV_ROOT%\output" ^
    --temp-directory "%DEV_ROOT%\temp" ^
    --models-directory "%COMFYUI_ROOT%\models" ^
    --database-url "%DATABASE_URL%" ^
    %CHECK_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd

if "%EXIT_CODE%"=="75" (
    echo [INFO] Restart requested by ComfyUI-Sai-nodes.
    timeout /t 1 /nobreak >nul
    goto run_comfyui
)

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] ComfyUI stopped with exit code %EXIT_CODE%.
    echo Review the messages above before closing this window.
    if not defined COMFYUI_NO_PAUSE pause
)

exit /b %EXIT_CODE%

:port_in_use
echo.
echo [ERROR] Port %COMFYUI_PORT% is already in use at %COMFYUI_HOST%.
echo Another ComfyUI instance is probably already running at:
echo http://%COMFYUI_HOST%:%COMFYUI_PORT%
echo Close that instance before starting or updating this development environment.
if not defined COMFYUI_NO_PAUSE pause
exit /b 10
