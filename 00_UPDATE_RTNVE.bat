@echo off
cd /d "%~dp0"
echo RTNVE - update local experiment scripts
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\00_Update.ps1"
echo.
if errorlevel 1 (
  echo UPDATE FAILED. Check the message above and try again.
) else (
  echo UPDATE COMPLETE. You can now run 03_BENCHMARK_TEMPORAL.bat.
)
pause
