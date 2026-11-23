# RTNVE Windows / NVIDIA RTX 3090 environment installer.
# Installs only within .venv folder of this repository.
# Does not alter NVIDIA drivers, A1111, or system Python.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

function Assert-OK([string]$description) {
    if ($LASTEXITCODE -ne 0) {
        throw "$description failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "RTNVE setup - independent Python environment" -ForegroundColor Cyan
Write-Host "Location: $root"
Write-Host "Your existing Automatic1111 installs will not be modified."
Write-Host ""

if (-not (Test-Path $pythonExe)) {
    Write-Host "Checking Python 3.11..."
    & py -3.11 --version
    Assert-OK "Python 3.11 availability. Install Python 3.11 from python.org, then rerun"
    & py -3.11 -m venv (Join-Path $root ".venv")
    Assert-OK "Creating RTNVE .venv"
}
Write-Host "Installing base Python dependencies..."
& $pythonExe -m pip install --upgrade pip
Assert-OK "Updating pip"

# CUDA wheel is essential on Windows. cu126 supports RTX 3090 with a
# sufficiently recent NVIDIA driver. Never silently install CPU-only PyTorch.
& $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
Assert-OK "Installing CUDA-enabled PyTorch (cu126)"

& $pythonExe -m pip install "diffusers>=0.30,<0.43" "transformers>=4.37,<5" "accelerate>=0.31" safetensors imageio imageio-ffmpeg pillow omegaconf
Assert-OK "Installing Diffusers and video tools"

Write-Host ""
Write-Host "Checking CUDA and neural inference..."
& $pythonExe -c "import sys,torch,diffusers; print('Python',sys.version.split()[0]); print('Torch',torch.__version__); print('Diffusers',diffusers.__version__); print('CUDA available',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE: check NVIDIA driver and CUDA wheel'; print('GPU',torch.cuda.get_device_name(0)); print('GPU VRAM GiB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))"
Assert-OK "Verifying RTX 3090 / CUDA"

Write-Host ""
Write-Host "PASS - RTNVE is ready for its first AnimateDiff V3 benchmark." -ForegroundColor Green
