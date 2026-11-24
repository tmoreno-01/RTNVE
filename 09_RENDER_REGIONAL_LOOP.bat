@echo off
cd /d "%~dp0"
echo RTNVE - 60 FRAME REGIONAL CCTV LOOP - 720p CFG7
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\09_Run_Regional_Loop.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\regional60_cfg7)
pause
