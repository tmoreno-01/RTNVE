@echo off
cd /d "%~dp0"
echo RTNVE - HIGHERCAR 900p - THREE 70-FRAME TESTS - R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\14_Run_Highercar_300_Triplet.ps1" -Highres70Triplet
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See benchmarks\results\highercar_900p_70f_triplet)
pause
