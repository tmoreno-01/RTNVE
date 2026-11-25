@echo off
cd /d "%~dp0"
echo RTNVE - 150 FRAME REGIONAL SOFTWILD LOOP - 720p CFG6 R-P
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\11_Run_Regional_Softwild.ps1"
if errorlevel 1 (echo RENDER FAILED. Copy the console error.) else (echo COMPLETE. See benchmarks\results\regional150_rp_cfg6_softwild)
pause
