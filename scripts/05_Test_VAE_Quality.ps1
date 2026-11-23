# Quality gate: encode/decode your ACTUAL RTNVE video with a pretrained SD1.5 VAE.
# No new video synthesis, training, server GPU, A1111 modification or driver changes.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
$source = Join-Path $root "training\test_pretrained_video_vae.py"
if (-not (Test-Path $pythonExe -PathType Leaf)) { throw "Missing RTNVE .venv; run setup first." }
if (-not (Test-Path $source -PathType Leaf)) { throw "Missing test script; update RTNVE source first." }

$video = Join-Path $root "benchmarks\results\animatediff-context\city_640x360_48f\context_city_640x360_48f_20s_seed611_w16_o8_loopN_travel_seed9191.mp4"
if (-not (Test-Path $video -PathType Leaf)) {
    Add-Type -AssemblyName System.Windows.Forms
    $picker = New-Object System.Windows.Forms.OpenFileDialog
    $picker.Title = "Select a GOOD RTNVE teacher MP4 to check neural compression fidelity"
    $picker.Filter = "Video (*.mp4)|*.mp4"
    if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "No MP4 selected."
    }
    $video = $picker.FileName
}
Write-Host ""
Write-Host "RTNVE pretrained neural codec roundtrip - art quality first" -ForegroundColor Cyan
Write-Host "Using video: $video"
Write-Host "This test measures whether the detailed 640x360 morphing frames survive compression."
Write-Host "It is NOT the future causal video engine and NOT a WebGPU FPS test."
Write-Host "The SD1.5 VAE will download once from Hugging Face, if not already cached."
Write-Host ""
$out=Join-Path $root "benchmarks\results\vae_quality"
& $pythonExe $source --video $video --frames 8 --out $out
if ($LASTEXITCODE -ne 0) {
    throw "VAE reconstruction quality test failed. Copy its full error."
}
Write-Host ""
Write-Host "Quality comparison (original left, neural decoder right):"
Write-Host (Join-Path $out "reconstruction_comparison.jpg")
Write-Host "Result metrics:"
Write-Host (Join-Path $out "reconstruction_results.json")
Write-Host "Share these two files with the RTNVE conversation."
