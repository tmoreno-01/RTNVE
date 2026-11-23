@echo off
cd /d "%~dp0"
echo RTNVE - real neural codec quality check
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0scripts\05_Test_VAE_Quality.ps1"
echo.
if errorlevel 1 (
  echo VAE QUALITY TEST FAILED. Please share the error.
) else (
  echo QUALITY TEST COMPLETE. Share the comparison JPG and results JSON.
)
pause
