@echo off
cd /d "%~dp0"
echo RTNVE - update local experiment scripts
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $p=Join-Path (Get-Location) 'scripts\00_Update.ps1'; $tmp=$p+'.download'; try { Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/tmoreno-01/RTNVE/main/scripts/00_Update.ps1' -OutFile $tmp; if ((Get-Item $tmp).Length -lt 100) {throw 'Incomplete updater'}; Move-Item -LiteralPath $tmp -Destination $p -Force } finally {if (Test-Path $tmp) {Remove-Item -LiteralPath $tmp -Force}}; & $p"
if errorlevel 1 (echo UPDATE FAILED. Check the message above.) else (echo UPDATE COMPLETE. Run your chosen render BAT.)
pause
