# Three sequential 300-frame Highercar variants plus a high-resolution test, R-P, V3/CyberRealistic.
# Prevents Windows system sleep while running; monitor may sleep.
param([switch]$OnlyHighres, [switch]$HighresTriplet, [switch]$Highres70Triplet, [switch]$Return720pTriplet)
$ErrorActionPreference="Stop"
if (([int][bool]$OnlyHighres + [int][bool]$HighresTriplet + [int][bool]$Highres70Triplet + [int][bool]$Return720pTriplet) -gt 1) { throw "Select only one render mode." }
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
if ($OnlyHighres) { $out=Join-Path $root "benchmarks\results\highercar_50_highres" }
if ($HighresTriplet) { $out=Join-Path $root "benchmarks\results\highercar_900p_triplet" }
if ($Highres70Triplet) { $out=Join-Path $root "benchmarks\results\highercar_900p_70f_triplet" }
if ($Return720pTriplet) { $out=Join-Path $root "benchmarks\results\highercar_720p_300f_return" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host ""
if ($Return720pTriplet) {
    Write-Host "RTNVE: THREE NEW 300-FRAME NATIVE 1280x720 TESTS" -ForegroundColor Cyan
    Write-Host "01 Original seed 1223; 02 Original seed 731; 03 Subtle dark seed 1889."
    Write-Host "All CFG5, context16/4, original negative prompt. No second pass or upscaling."
} elseif ($HighresTriplet -or $Highres70Triplet) {
    $frameCount=50
    if ($Highres70Triplet) { $frameCount=70 }
    Write-Host "RTNVE HIGHERCAR: THREE $frameCount-FRAME 900p TESTS" -ForegroundColor Cyan
    Write-Host "01 Original: seed 1223, CFG5, 16/4; 02 Subtle dark: seed 731, CFG5, 16/4."
    Write-Host "03 Hybrid: seed 2849, CFG4.8, 16/6. Native 1600x904 cropped to 1600x900."
} elseif ($OnlyHighres) {
    Write-Host "RTNVE HIGHERCAR: HIGH-RESOLUTION TEST ONLY" -ForegroundColor Cyan
} else {
    Write-Host "RTNVE HIGHERCAR: THREE 300-FRAME 720p TESTS PLUS ONE 50-FRAME 900p TEST" -ForegroundColor Cyan
    Write-Host "Baseline CFG5 16/4; dark CFG5 16/4; soft CFG4.5 16/8."
}
if (-not ($HighresTriplet -or $Highres70Triplet -or $Return720pTriplet)) {
    Write-Host "High-res: baseline prompt, CFG5 16/4, native 1600x904 cropped to 1600x900."
    Write-Host "All: seed 144."
}
Write-Host "All: 20 steps, R-P, 10 FPS, no LoRAs, prompt travel or masks."
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
if ($OnlyHighres) { $argsList += "--only-highres" }
if ($HighresTriplet) { $argsList += "--highres-triplet" }
if ($Highres70Triplet) { $argsList += "--highres-70-triplet" }
if ($Return720pTriplet) { $argsList += "--return-720p-triplet" }
try {
    & $pythonExe $script @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Highercar tests stopped on an error. Completed earlier clips remain saved. Copy the error from the Python console."
    }
} finally {
    [void][RTNVEHighercarPower]::SetThreadExecutionState([uint32]2147483648)
}
Write-Host ""
Write-Host "SUCCESS. Upload the MP4s and triplet_manifest.json." -ForegroundColor Green
Write-Host "Results: $out"
