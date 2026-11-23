# Export a tiny untrained recurrent CNN from installed PyTorch to ONNX.
# Does NOT touch A1111, install GPU drivers, or upload data/model anywhere.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
$source = Join-Path $root "training\export_webgpu_student_probe.py"
if (-not (Test-Path $pythonExe -PathType Leaf)) {
    throw "Missing RTNVE .venv. Use 01_SETUP_WINDOWS.bat first."
}
if (-not (Test-Path $source -PathType Leaf)) {
    throw "Missing the exporter. Use 00_UPDATE_RTNVE.bat to update this project's scripts first."
}
Write-Host ""
Write-Host "RTNVE - export UNTRAINED recurrent neural inference probe" -ForegroundColor Cyan
Write-Host "This will NOT create an artwork or change the AnimateDiff model."
Write-Host "Purpose: test whether real PyTorch neural inference can run on desktop / phone WebGPU."
Write-Host ""
Write-Host "[1] 128x72 internal neural state (recommended)"
Write-Host "[2] 256x144 internal neural state (heavier)"
Write-Host "[3] 384x216 internal neural state (heavier; test later)"
$choice = Read-Host "Choose 1, 2 or 3 (Enter for 1)"
$w=128; $h=72
if ($choice -eq "2") {$w=256; $h=144}
if ($choice -eq "3") {$w=384; $h=216}
$out = Join-Path $root ("model_probes\recurrent_probe_" + $w + "x" + $h + ".onnx")

Write-Host "Installing missing ONNX exporter package inside RTNVE .venv only..."
& $pythonExe -m pip install --disable-pip-version-check "onnx>=1.17,<2"
if ($LASTEXITCODE -ne 0) {
    throw "ONNX pip installation failed. Copy the full error into the conversation."
}
Write-Host "Exporting real, random-weight recurrent CNN (NOT a trained video model)..."
& $pythonExe $source --width $w --height $h --output $out
if ($LASTEXITCODE -ne 0) {
    throw "ONNX model export failed. Copy the full error into the conversation."
}
if (-not (Test-Path $out -PathType Leaf)) {
    throw "Export reported success but ONNX file is missing: $out"
}
Write-Host ""
Write-Host "Model created: $out" -ForegroundColor Green
Write-Host "Next: https://tmoreno-01.github.io/RTNVE/webgpu-student-probe/"
Write-Host "Select this .onnx file from the browser file picker, then Run."
Write-Host "Copy the JSON results back into our RTNVE conversation."
Write-Host "For a phone test, transfer this harmless tiny ONNX file to your phone and"
Write-Host "open the same page in supported Safari / Chrome. It is not uploaded to a server."
