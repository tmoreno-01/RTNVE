# One-click six-test comparison suite; keep Windows awake.
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe=Join-Path $root ".venv\Scripts\python.exe"
$script=Join-Path $root "benchmarks\highercar_900p_comparison.py"
if (-not (Test-Path $pythonExe -PathType Leaf)) { throw "RTNVE .venv missing." }
if (-not (Test-Path $script -PathType Leaf)) { throw "Global renderer missing; run updater." }
$checkpoint="Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\models\Stable-diffusion\cyberrealistic_final.safetensors"
$motion="Z:\ART\AI\2026\AI\SD\NEWTEST\sandboxnew\extensions\sd-webui-animatediff\model\v3_sd15_mm.ckpt"
if (-not (Test-Path $checkpoint -PathType Leaf)) {
    throw "CyberRealistic checkpoint not found: $checkpoint"
}
if (-not (Test-Path $motion -PathType Leaf)) {
    throw "AnimateDiff V3 motion model not found: $motion"
}
$out=Join-Path $root "benchmarks\results\highercar_900p_comparison_suite"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Host "RTNVE: SIX 70-FRAME 900p COMPARISON TESTS" -ForegroundColor Cyan
Write-Host "A1-A3: native 900p; B1-B3: 720p base + 900p latent refinement."
Write-Host "All sequential, CFG5.4, R-P, context16/4, 10 FPS. No LoRAs."
Write-Host "Results: $out"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RTNVEHighercarPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
[void][RTNVEHighercarPower]::SetThreadExecutionState([uint32]2147483649)
$argsList=@("--checkpoint",$checkpoint,"--motion",$motion,"--out",$out)
try {
    & $pythonExe $script @argsList
    if ($LASTEXITCODE -ne 0) { throw "Comparison suite reported an error. Completed earlier clips remain saved. See suite_manifest.json." }
} finally {
    [void][RTNVEHighercarPower]::SetThreadExecutionState([uint32]2147483648)
}
Write-Host "SUCCESS: Upload six MP4s and suite_manifest.json." -ForegroundColor Green
