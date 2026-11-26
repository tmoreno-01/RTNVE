@echo off
cd /d "%~dp0"
echo RTNVE - HIGHERCAR 300 FRAME TRIPLET - 720p R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\14_Run_Highercar_300_Triplet.ps1"
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See benchmarks\results\highercar_300_triplet)
pause
