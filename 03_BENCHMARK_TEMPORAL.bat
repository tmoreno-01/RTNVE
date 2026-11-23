@echo off
cd /d "%~dp0"
echo RTNVE - A1111-style long temporal context test
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0scripts\03_Run_Temporal.ps1"
echo.
if errorlevel 1 (
  echo TEMPORAL TEST FAILED. Send the full error to the RTNVE conversation.
) else (
  echo TEMPORAL TEST COMPLETE. Send the MP4 plus temporal_results.json.
)
pause
