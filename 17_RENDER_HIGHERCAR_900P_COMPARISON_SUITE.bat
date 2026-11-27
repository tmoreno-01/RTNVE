@echo off
cd /d "%~dp0"
echo RTNVE - 900p NATIVE VS MULTI-RES - SIX TESTS
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\17_Run_Highercar_900P_Comparison.ps1"
if errorlevel 1 (echo SOME TESTS FAILED. Completed clips are retained. See console and suite manifest.) else (echo COMPLETE. See benchmarks\results\highercar_900p_comparison_suite)
pause
