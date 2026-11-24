# 360-frame single-global-latent overnight render, original V3/CyberRealistic.
# Prevents Windows system sleep while running; monitor may sleep.
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe=Join-Path $root ".venv\Scripts\python.exe"
$script=Join-Path $root "benchmarks\overnight_1000.py"
$promptModule=Join-Path $root "benchmarks\curated_prompt_travel.py"
if (-not (Test-Path $pythonExe -PathType Leaf)) { throw "RTNVE .venv missing." }
if (-not (Test-Path $script -PathType Leaf)) { throw "Overnight script missing; run updater." }
if (-not (Test-Path $promptModule -PathType Leaf)) { throw "Curated prompts missing; run updater." }
$checkpoint="Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\models\Stable-diffusion\cyberrealistic_final.safetensors"
$motion="Z:\ART\AI\2026\AI\SD\NEWTEST\sandboxnew\extensions\sd-webui-animatediff\model\v3_sd15_mm.ckpt"
if (-not (Test-Path $checkpoint -PathType Leaf)) {
    throw "CyberRealistic checkpoint not found: $checkpoint"
}
if (-not (Test-Path $motion -PathType Leaf)) {
    throw "AnimateDiff V3 motion model not found: $motion"
}
$out=Join-Path $root "benchmarks\results\curated360_cfg7"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host ""
Write-Host "RTNVE CURATED: 360 FRAMES AT 1280x720, 10 FPS" -ForegroundColor Cyan
Write-Host "20 steps, CFG 7, context16 overlap8, N nonloop."
Write-Host "Five curated urban families; 80-140 fully conditioned frames plus 24-48-frame morphs."
Write-Host "One global latent video, no LoRA tags, no stitching or clips."
Write-Host "Preflight tests 1280x720 VRAM. If OOM, script logs fallback 896x504."
Write-Host "WARNING: experimental high-VRAM job, shorter than the 1000-frame run; full GPU render is still experimental."
Write-Host "It CANNOT resume unfinished global denoising after power loss."
Write-Host "Progress: $out\status.json"
Write-Host ""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RTNVECuratedPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
[void][RTNVECuratedPower]::SetThreadExecutionState([uint32]2147483649)
$argsList=@(
    "--checkpoint", $checkpoint,
    "--motion", $motion,
    "--frames", "360",
    "--width", "1280", "--height", "720",
    "--fps", "10", "--steps", "20", "--guidance", "7", "--seed", "611",
    "--travel-seed", "9191",
    "--travel-min", "80", "--travel-max", "140",
    "--travel-transition-min", "24", "--travel-transition-max", "48",
    "--window", "16", "--overlap", "8",
    "--curated",
    "--fallback-896",
    "--out", $out
)
try {
    & $pythonExe $script @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "360-frame render failed. Copy the error from the Python console."
    }
} finally {
    [void][RTNVECuratedPower]::SetThreadExecutionState([uint32]2147483648)
}
Write-Host ""
Write-Host "SUCCESS. Upload the MP4, completed_results.json, and curated_prompt_plan.json." -ForegroundColor Green
Write-Host "Results: $out"
