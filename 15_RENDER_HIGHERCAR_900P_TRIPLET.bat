@echo off
cd /d "%~dp0"
echo RTNVE - HIGHERCAR 900p TRIPLET - 3 NEW SEEDS - R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\14_Run_Highercar_300_Triplet.ps1" -HighresTriplet
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See benchmarks\results\highercar_900p_triplet)
pause
