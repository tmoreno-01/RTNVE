# Three sequential 300-frame Highercar variants, R-P, V3/CyberRealistic.
# Prevents Windows system sleep while running; monitor may sleep.
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe=Join-Path $root ".venv\Scripts\python.exe"
$script=Join-Path $root "benchmarks\highercar_300_triplet.py"
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
$out=Join-Path $root "benchmarks\results\highercar_300_triplet"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host ""
Write-Host "RTNVE HIGHERCAR TRIPLET: THREE 300-FRAME MP4S AT 1280x720, 10 FPS" -ForegroundColor Cyan
Write-Host "All three: seed 144, 300 frames, 20 steps, R-P loop."
Write-Host "Baseline CFG5 16/4; dark CFG5 16/4; soft CFG4.5 16/8. No LoRAs."
Write-Host "Three independent whole-frame renders, no prompt travel or masks."
Write-Host "Full-resolution GPU preflight. No lower-resolution fallback."
Write-Host "One whole-frame conditioning stream; no regional attention."
Write-Host "It CANNOT resume unfinished global denoising after power loss."
Write-Host "Progress: $out\status.json"
Write-Host ""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RTNVEHighercarPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
[void][RTNVEHighercarPower]::SetThreadExecutionState([uint32]2147483649)
$argsList=@(
    "--checkpoint", $checkpoint,
    "--motion", $motion,
    "--out", $out
)
try {
    & $pythonExe $script @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Highercar triplet stopped on an error. Completed earlier clips remain saved. Copy the error from the Python console."
    }
} finally {
    [void][RTNVEHighercarPower]::SetThreadExecutionState([uint32]2147483648)
}
Write-Host ""
Write-Host "SUCCESS. Upload the three MP4s and triplet_manifest.json." -ForegroundColor Green
Write-Host "Results: $out"
