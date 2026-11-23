@echo off
cd /d "%~dp0"
echo RTNVE - Original AnimateDiff V3 benchmark
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0scripts\02_Run_Benchmark.ps1"
echo.
if errorlevel 1 (
  echo BENCHMARK FAILED. Copy the error above into your RTNVE conversation.
) else (
  echo BENCHMARK COMPLETE. Share results.json, results.csv and previews.
)
pause
