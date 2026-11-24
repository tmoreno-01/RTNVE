@echo off
cd /d "%~dp0"
echo RTNVE - 200 FRAME CURATED URBAN TEST - CFG 7
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\08_Run_Curated_200.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\curated200_cfg7)
pause
