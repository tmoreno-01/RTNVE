@echo off
cd /d "%~dp0"
echo RTNVE - THREE NATIVE 720p 300-FRAME TESTS - R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\14_Run_Highercar_300_Triplet.ps1" -Return720pTriplet
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See benchmarks\results\highercar_720p_300f_return)
pause
