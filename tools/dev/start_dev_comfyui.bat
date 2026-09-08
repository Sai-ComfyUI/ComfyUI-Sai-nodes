@echo off
setlocal

set "TOOLS_ROOT=%~dp0"
for %%I in ("%TOOLS_ROOT%..\..") do set "PROJECT_ROOT=%%~fI\"
set "LOCAL_CONFIG=%TOOLS_ROOT%dev_environment.local.bat"
set "DEV_ROOT=%PROJECT_ROOT%.dev"
set "COMFYUI_HOST=127.0.0.1"
set "COMFYUI_PORT=9527"
set "CHECK_ARGS="

if not exist "%LOCAL_CONFIG%" (
    echo [ERROR] Missing %LOCAL_CONFIG%
    echo Copy "%TOOLS_ROOT%dev_environment.example.bat" to "%LOCAL_CONFIG%" first.
    exit /b 1
)

call "%LOCAL_CONFIG%"

if /I "%~1"=="--check" set "CHECK_ARGS=--quick-test-for-ci"

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

for %%D in ("%DEV_ROOT%" "%DEV_ROOT%\user" "%DEV_ROOT%\input" "%DEV_ROOT%\output" "%DEV_ROOT%\temp") do (
    if not exist "%%~D" mkdir "%%~D"
    if errorlevel 1 exit /b 1
)

set "DEV_DB_PATH=%DEV_ROOT:\=/%/user/comfyui.db"

echo [INFO] ComfyUI root: %COMFYUI_ROOT%
echo [INFO] Python: %COMFYUI_PYTHON%
echo [INFO] URL: http://%COMFYUI_HOST%:%COMFYUI_PORT%
echo [INFO] Custom nodes: ComfyUI-Sai-nodes only

:run_comfyui
pushd "%COMFYUI_ROOT%"
"%COMFYUI_PYTHON%" -s main.py ^
    --windows-standalone-build ^
    --listen "%COMFYUI_HOST%" ^
    --port "%COMFYUI_PORT%" ^
    --disable-all-custom-nodes ^
    --whitelist-custom-nodes ComfyUI-Sai-nodes ^
    --user-directory "%DEV_ROOT%\user" ^
    --input-directory "%DEV_ROOT%\input" ^
    --output-directory "%DEV_ROOT%\output" ^
    --temp-directory "%DEV_ROOT%\temp" ^
    --models-directory "%COMFYUI_ROOT%\models" ^
    --database-url "sqlite:///%DEV_DB_PATH%" ^
    %CHECK_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd

if "%EXIT_CODE%"=="75" (
    echo [INFO] Restart requested by ComfyUI-Sai-nodes.
    timeout /t 1 /nobreak >nul
    goto run_comfyui
)

exit /b %EXIT_CODE%
