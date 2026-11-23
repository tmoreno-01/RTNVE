@echo off
cd /d "%~dp0"
echo RTNVE - real neural WebGPU portability probe
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\04_Build_Probe.ps1"
echo.
if errorlevel 1 (
  echo EXPORT FAILED. Copy the terminal error into your RTNVE conversation.
) else (
  echo MODEL EXPORTED. Open the RTNVE WebGPU student probe in Chrome or mobile.
)
pause
