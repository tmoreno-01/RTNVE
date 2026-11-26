# 150-frame N global single-global-latent render, original V3/CyberRealistic.
# Prevents Windows system sleep while running; monitor may sleep.
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe=Join-Path $root ".venv\Scripts\python.exe"
$script=Join-Path $root "benchmarks\global_softwild_150.py"
$promptModule=Join-Path $root "benchmarks\global_softwild_prompt_travel.py"
if (-not (Test-Path $pythonExe -PathType Leaf)) { throw "RTNVE .venv missing." }
if (-not (Test-Path $script -PathType Leaf)) { throw "Global renderer missing; run updater." }
if (-not (Test-Path $promptModule -PathType Leaf)) { throw "Global prompts missing; run updater." }
$checkpoint="Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\models\Stable-diffusion\cyberrealistic_final.safetensors"
$motion="Z:\ART\AI\2026\AI\SD\NEWTEST\sandboxnew\extensions\sd-webui-animatediff\model\v3_sd15_mm.ckpt"
if (-not (Test-Path $checkpoint -PathType Leaf)) {
    throw "CyberRealistic checkpoint not found: $checkpoint"
}
if (-not (Test-Path $motion -PathType Leaf)) {
    throw "AnimateDiff V3 motion model not found: $motion"
}
$out=Join-Path $root "benchmarks\results\global150_cfg65_softwild"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host ""
Write-Host "RTNVE GLOBAL SOFTWILD: 150 FRAMES AT 1280x720, 10 FPS" -ForegroundColor Cyan
Write-Host "20 steps, CFG 6.5, context16 overlap8, N non-loop."
Write-Host "SOFTWILD: high-angle crowds, faces, cars, screens, independently drifting global modifiers. No masks."
Write-Host "One global latent video, no LoRA tags, no stitching or clips."
Write-Host "Full-resolution GPU preflight. No lower-resolution fallback."
Write-Host "One whole-frame conditioning stream; no regional attention."
Write-Host "It CANNOT resume unfinished global denoising after power loss."
Write-Host "Progress: $out\status.json"
Write-Host ""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RTNVEGlobalSoftwildPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
[void][RTNVEGlobalSoftwildPower]::SetThreadExecutionState([uint32]2147483649)
$argsList=@(
    "--checkpoint", $checkpoint,
    "--motion", $motion,
    "--seed", "611", "--travel-seed", "9191",
    "--out", $out
)
try {
    & $pythonExe $script @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "150-frame render failed. Copy the error from the Python console."
    }
} finally {
    [void][RTNVEGlobalSoftwildPower]::SetThreadExecutionState([uint32]2147483648)
}
Write-Host ""
Write-Host "SUCCESS. Upload the MP4, completed_results.json, and global_prompt_plan.json." -ForegroundColor Green
Write-Host "Results: $out"
