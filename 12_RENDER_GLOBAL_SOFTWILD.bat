@echo off
cd /d "%~dp0"
echo RTNVE - 150 FRAME GLOBAL SOFTWILD - 720p CFG6.5 N
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\12_Run_Global_Softwild.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\global150_cfg65_softwild)
pause
