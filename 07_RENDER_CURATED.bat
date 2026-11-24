@echo off
cd /d "%~dp0"
echo RTNVE - 360 FRAME CURATED URBAN TEST - CFG 7
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\07_Run_Curated.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\curated360_cfg7)
pause
