@echo off
cd /d "%~dp0"
echo RTNVE - 150 FRAME REGIONAL CCTV LOOP - 720p CFG7 R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\10_Run_Regional_Loop_150.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\regional150_rp_cfg7)
pause
