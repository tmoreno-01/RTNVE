# RTNVE A1111-style 100-frame overlapping temporal-context benchmark.
# Uses the original A1111 checkpoint and V3 motion module, but never edits A1111.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe -PathType Leaf)) {
    throw "RTNVE Python environment missing. Run 01_SETUP_WINDOWS.bat first."
}
$experiment = Join-Path $root "benchmarks\animatediff_long_context.py"
if (-not (Test-Path $experiment -PathType Leaf)) {
    throw "Missing animatediff_long_context.py. Download the updated RTNVE source."
}
# Fail BEFORE loading multi-GB models if an old ZIP/script is still in use.
$requiredMarker = 'RTNVE_CONTEXT_PATCH = "2026-11-23-cfg-window-v2"'
$code = Get-Content -LiteralPath $experiment -Raw -Encoding UTF8
if (-not $code.Contains($requiredMarker)) {
    throw "Outdated RTNVE temporal Python code detected. Run 00_UPDATE_RTNVE.bat to synchronize the fixes. No GPU test was started."
}
Write-Host "Source version OK: 2026-11-23-cfg-window-v2" -ForegroundColor Green

$checkpoint = "Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\models\Stable-diffusion\cyberrealistic_final.safetensors"
$motion = "Z:\ART\AI\2026\AI\SD\NEWTEST\sandboxnew\extensions\sd-webui-animatediff\model\v3_sd15_mm.ckpt"

function Select-File([string]$caption) {
    Add-Type -AssemblyName System.Windows.Forms
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = $caption
    $dlg.Filter = "Model file (*.ckpt;*.safetensors)|*.ckpt;*.safetensors"
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "File selection cancelled."
    }
    return $dlg.FileName
}
if (-not (Test-Path $checkpoint -PathType Leaf)) {
    $checkpoint = Select-File "Select your CyberRealistic SD1.5 checkpoint"
}
if (-not (Test-Path $motion -PathType Leaf)) {
    $motion = Select-File "Select v3_sd15_mm.ckpt"
}

Write-Host ""
Write-Host "RTNVE - temporal context reconstruction" -ForegroundColor Cyan
Write-Host "This is NOT live streaming or realtime inference."
Write-Host "It processes overlapping video windows at EVERY denoising step."
Write-Host ""
Write-Host "Choose the prompt:"
Write-Host "[1] NYC CCTV: cars / people / destruction / plants (your requested default)"
Write-Host "[2] Apartment window grid from your A1111 screenshot"
$presetSelection = Read-Host "Preset 1 or 2 (Enter for 1)"
$preset = if ($presetSelection -eq "2") { "windows" } else { "city" }
Write-Host ""
Write-Host "Choose the first test:"
Write-Host "[1] 48 frames / 640x360  -- first run (recommended, validate code)"
Write-Host "[2] 100 frames / 640x360 -- full 10-second motion test"
Write-Host "[3] 100 frames / 896x504 -- better detail, heavier"
Write-Host "[4] 100 frames / 1280x720 -- final target orientation; HIGH VRAM / slow; do not start here"
$choice = Read-Host "Option 1, 2, 3 or 4 (Enter for 1)"
$frames = 100; $width = 640; $height = 360
switch ($choice) {
    "1" { $frames = 48 }
    "3" { $width = 896; $height = 504 }
    "4" { $width = 1280; $height = 720 }
    default { if ($choice -notin @("2", "3", "4")) { $frames = 48 } }
}
$out = Join-Path $root ("benchmarks\results\animatediff-context\" + $preset + "_" + $width + "x" + $height + "_" + $frames + "f")
New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Host ""
Write-Host "Checkpoint: $checkpoint"
Write-Host "Motion V3:  $motion"
Write-Host "Video:      $frames frames / $width x $height / 10 playback FPS"
Write-Host ""
Write-Host "Choose the temporal window settings:"
Write-Host "[1] Context 13 / overlap 6 (successful reference)"
Write-Host "[2] Context 16 / overlap 8 (smoother motion hypothesis - A/B comparison)"
$contextChoice = Read-Host "Option 1 or 2 (Enter for 2)"
$window = if ($contextChoice -eq "1") { 13 } else { 16 }
$overlap = if ($contextChoice -eq "1") { 6 } else { 8 }
Write-Host "Temporal context batch $window, overlap $overlap, stride 1, loop N (no wrap)"

Write-Host ""
Write-Host "Choose the prompt behaviour:"
Write-Host "[1] Single static prompt (for controlled baseline comparisons)"
Write-Host "[2] Autonomous prompt travel (default; gradual random semantic evolution)"
$travelChoice = Read-Host "Option 1 or 2 (Enter for 2)"
$travel = ($travelChoice -ne "1")
$travelArgs = @()
if ($travel) {
    # We deliberately avoid the old 6-frame hard-feeling morphs.
    # At 48 frames: one 28-frame gradual transformation beginning after
    # 12 frames, followed by 8 frames of the new scene. Not three abrupt
    # changes. With 100 frames: one ~28-frame transition after frame 22.
    # For the eventual causal engine choose 50/100/200-frame dwell lengths.
    if ($frames -le 48) {
        $travelMin = 40; $travelMax = 40; $travelTransition = 28
    } elseif ($frames -le 100) {
        $travelMin = 50; $travelMax = 50; $travelTransition = 28
    } else {
        $travelMin = 50; $travelMax = 200; $travelTransition = 28
    }
    $travelArgs = @(
        "--prompt-travel",
        "--travel-seed", "9191",
        "--travel-min", [string]$travelMin,
        "--travel-max", [string]$travelMax,
        "--travel-transition", [string]$travelTransition
    )
    Write-Host "Prompt travel preview: interval $travelMin-$travelMax frames; transition $travelTransition frames."
}
Write-Host "20 steps, CFG 6, seed 611"
Write-Host "Outputs: $out"
Write-Host "NOTE: Weighted prompt syntax and scheduler are still not identical to A1111."
Write-Host "This full-video fidelity test is substantially more computationally expensive"
Write-Host "than the earlier 16-frame batch. It may use significant GPU memory."
Write-Host ""

$argList = @(
    $experiment,
    "--checkpoint", $checkpoint,
    "--motion", $motion,
    "--preset", $preset,
    "--frames", [string]$frames,
    "--width", [string]$width,
    "--height", [string]$height,
    "--steps", "20",
    "--guidance", "6",
    "--seed", "611",
    "--window", [string]$window,
    "--overlap", [string]$overlap,
    "--stride", "1",
    "--loop", "N",
    "--fps", "10",
    "--out", $out
)
$argList += $travelArgs
& $pythonExe @argList
if ($LASTEXITCODE -ne 0) {
    throw "RTNVE temporal test failed with exit code $LASTEXITCODE. Send complete error."
}
Write-Host ""
Write-Host "SUCCESS. Please upload the MP4 and its matching *_results.json from:"
Write-Host "$out"
