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
Write-Host "A1111-style context batch 13, overlap 6, stride 1, loop R+P"
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
    "--window", "13",
    "--overlap", "6",
    "--stride", "1",
    "--loop", "R+P",
    "--fps", "10",
    "--out", $out
)
& $pythonExe @argList
if ($LASTEXITCODE -ne 0) {
    throw "RTNVE temporal test failed with exit code $LASTEXITCODE. Send complete error."
}
Write-Host ""
Write-Host "SUCCESS. Please upload the MP4 and temporal_results.json from:"
Write-Host "$out"
