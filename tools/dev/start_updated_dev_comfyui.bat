@echo off
setlocal

set "TOOLS_ROOT=%~dp0"
set "LOCAL_CONFIG=%TOOLS_ROOT%dev_environment.local.bat"
set "CORE_CHANNEL=stable"
set "LAUNCH_MODE=run"
set "UPDATER_NO_PAUSE=%COMFYUI_NO_PAUSE%"

if not exist "%LOCAL_CONFIG%" (
    echo [ERROR] Missing %LOCAL_CONFIG%
    echo Copy "%TOOLS_ROOT%dev_environment.example.bat" to "%LOCAL_CONFIG%" first.
    exit /b 1
)

call "%LOCAL_CONFIG%"

:parse_args
if "%~1"=="" goto sync
if /I "%~1"=="--edge" (
    set "CORE_CHANNEL=latest"
    shift
    goto parse_args
)
if /I "%~1"=="--check" (
    set "LAUNCH_MODE=check"
    shift
    goto parse_args
)
if /I "%~1"=="--sync-only" (
    set "LAUNCH_MODE=sync-only"
    shift
    goto parse_args
)
echo [ERROR] Unknown argument: %~1
echo Usage: %~nx0 [--edge] [--check^|--sync-only]
if not defined UPDATER_NO_PAUSE pause
exit /b 2

:sync
set "COMFYUI_NO_PAUSE=1"
call "%TOOLS_ROOT%start_dev_comfyui.bat" --preflight
set "PREFLIGHT_EXIT=%ERRORLEVEL%"
set "COMFYUI_NO_PAUSE=%UPDATER_NO_PAUSE%"
if not "%PREFLIGHT_EXIT%"=="0" goto environment_in_use

set "SYNC_ARGS=-CoreChannel %CORE_CHANNEL%"
if /I "%LAUNCH_MODE%"=="check" set "SYNC_ARGS=%SYNC_ARGS% -SkipSmokeTest"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TOOLS_ROOT%sync_dev_environment.ps1" %SYNC_ARGS%
set "SYNC_EXIT=%ERRORLEVEL%"
if not "%SYNC_EXIT%"=="0" goto sync_failed

if /I "%LAUNCH_MODE%"=="sync-only" exit /b 0
if /I "%LAUNCH_MODE%"=="check" call "%TOOLS_ROOT%start_dev_comfyui.bat" --check
if /I "%LAUNCH_MODE%"=="run" call "%TOOLS_ROOT%start_dev_comfyui.bat"
exit /b %ERRORLEVEL%

:environment_in_use
echo.
echo [ERROR] Update cancelled because the development port is already in use.
echo Close the running ComfyUI instance, then run this updater again.
if not defined UPDATER_NO_PAUSE pause
exit /b %PREFLIGHT_EXIT%

:sync_failed
echo.
echo [ERROR] Environment synchronization failed with exit code %SYNC_EXIT%.
echo Review the messages above before closing this window.
if not defined UPDATER_NO_PAUSE pause
exit /b %SYNC_EXIT%
