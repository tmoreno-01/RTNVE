@echo off
cd /d "%~dp0"
echo RTNVE - Windows GPU setup
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\01_Setup.ps1"
echo.
if errorlevel 1 (
  echo SETUP FAILED. Copy the error above into your RTNVE conversation.
) else (
  echo SETUP COMPLETE. Next double-click 02_BENCHMARK_V3.bat.
)
pause
