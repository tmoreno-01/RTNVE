@echo off
cd /d "%~dp0"
echo RTNVE - ORIGINAL REFERENCE TRIPLET - 720p R-P - NO LORAS
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\13_Run_Reference_Triplet.ps1"
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See benchmarks\results\reference_triplet_1280x720)
pause
