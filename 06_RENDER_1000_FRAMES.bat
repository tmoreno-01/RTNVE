@echo off
cd /d "%~dp0"
echo RTNVE - 1000 FRAME 1280x720 NEURAL WORLD
echo 10 FPS, nonloop N, 16/8 context, 20 steps, CFG6
echo Random surreal prompts with random 20-60-frame neural conditioning transitions
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\06_Run_Overnight.ps1"
echo.
if errorlevel 1 (
  echo OVERNIGHT RUN DID NOT COMPLETE. See console and status.json.
) else (
  echo RENDER COMPLETE. Find MP4 in benchmarksesultsovernight1000
)
pause
