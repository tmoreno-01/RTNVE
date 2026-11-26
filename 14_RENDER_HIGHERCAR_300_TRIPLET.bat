@echo off
cd /d "%~dp0"
echo RTNVE - HIGHERCAR - 3 x 720p + 1 x 900p - R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\14_Run_Highercar_300_Triplet.ps1" %*
if errorlevel 1 (echo RENDER FAILED. Completed earlier clips are retained. Copy the console error.) else (echo COMPLETE. See results path above)
pause
