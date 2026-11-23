# First run: choose checkpoint + motion file in Windows file pickers.
# Original A1111 installations are NOT edited.
param(
    [string]$Checkpoint = "",
    [string]$Motion = "",
    [ValidateSet("512x288", "768x432", "1280x720")]
    [string]$Resolution = "",
    [switch]$NoWarmup
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Missing .venv: run 01_SETUP_WINDOWS.bat first."
}
function Pick-File([string]$title, [string]$filter, [string]$initialDirectory) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $title
    $dialog.Filter = $filter
    $dialog.Multiselect = $false
    $dialog.CheckFileExists = $true
    if (Test-Path $initialDirectory -PathType Container) {
        $dialog.InitialDirectory = $initialDirectory
    }
    $result = $dialog.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "File selection cancelled. Rerun this script to choose your files."
    }
    return $dialog.FileName
}

$stableModelDir = "Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\models\Stable-diffusion"
$oldAnimatediffDir = "Z:\ART\AI\2026\AI\SD\STABLE\stable-diffusion-webui\extensions\sd-webui-animatediff\model"
$newAnimatediffDir = "Z:\ART\AI\2026\AI\SD\NEWTEST\sandboxnew\extensions\sd-webui-animatediff\model"
Write-Host ""
Write-Host "RTNVE - AnimateDiff V3 baseline measurement" -ForegroundColor Cyan
Write-Host "First run at 512x288 checks that the pipeline loads."
Write-Host "This is a speed benchmark, NOT the final visual resolution."
Write-Host ""

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Pick-File "Choose your original CyberRealistic SD1.5 checkpoint" "Stable Diffusion checkpoint (*.safetensors;*.ckpt)|*.safetensors;*.ckpt|All files|*.*" $stableModelDir
}
if ([string]::IsNullOrWhiteSpace($Motion)) {
    $motionFolder = if (Test-Path $newAnimatediffDir) { $newAnimatediffDir } else { $oldAnimatediffDir }
    $Motion = Pick-File "Choose the V3 file named v3_sd15_mm.ckpt" "AnimateDiff motion file (*.ckpt;*.safetensors)|*.ckpt;*.safetensors|All files|*.*" $motionFolder
}
if (-not (Test-Path $Checkpoint -PathType Leaf)) { throw "Checkpoint not found: $Checkpoint" }
if (-not (Test-Path $Motion -PathType Leaf)) { throw "Motion module not found: $Motion" }
if ((Split-Path $Motion -Leaf) -notmatch "v3") {
    Write-Warning "The selected motion filename does not contain V3. Target is v3_sd15_mm.ckpt."
    $answer = Read-Host "Continue anyway? [y/N]"
    if ($answer -ne "y") { throw "Choose the V3 motion module and try again." }
}

if ([string]::IsNullOrEmpty($Resolution)) {
    Write-Host ""
    Write-Host "[1] 512x288  / initial compatibility + speed test (recommended)"
    Write-Host "[2] 768x432  / intermediate detail test"
    Write-Host "[3] 1280x720 / high-resolution test (may run out of GPU memory)"
    $choice = Read-Host "Select 1, 2 or 3 (press Enter for 1)"
    $Resolution = switch ($choice) {
        "2" { "768x432" }
        "3" { "1280x720" }
        default { "512x288" }
    }
}
$parts = $Resolution.Split("x")
$w = [int]$parts[0]
$h = [int]$parts[1]
$target = Join-Path $root ("benchmarks\results\animatediff-v3\" + $Resolution)
New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Host ""
Write-Host "CHECKPOINT: $Checkpoint"
Write-Host "MOTION V3:  $Motion"
Write-Host "TEST SIZE:  $Resolution / 16 frames / 24, 12, 8, 4 steps"
Write-Host "RESULTS:    $target"
Write-Host "WARNING: A1111 stays untouched; results may differ from the extension."
Write-Host ""
$cmd = @(
    (Join-Path $root "benchmarks\animatediff_v3.py"),
    "--checkpoint", $Checkpoint,
    "--motion", $Motion,
    "--width", "$w", "--height", "$h",
    "--frames", "16",
    "--steps", "24", "12", "8", "4",
    "--out", $target
)
if (-not $NoWarmup) { $cmd += "--warmup" }
& $pythonExe @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark failed (exit code $LASTEXITCODE). Copy full terminal error here."
}
Write-Host ""
Write-Host "Completed. Share results.json and results.csv from $target" -ForegroundColor Green
Write-Host "Also share the 24-step and 8-step MP4s if the render worked."
