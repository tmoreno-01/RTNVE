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

# Windows doesn't always install the "py" launcher (especially Store installs).
# Detect existing Python 3.11 through either the launcher, executable on PATH,
# or the usual per-user/system install folders. Do not install Python globally.
function Find-CompatiblePython {
    # Run actual commands rather than judging a Python installation by its
    # location. WindowsApps often contains a *working* Microsoft Store Python
    # alias, and rejecting that path caused the previous false missing error.
    $candidates = @(
        @{ exe = "python"; args = @(); label = "python on PATH" },
        @{ exe = "python3"; args = @(); label = "python3 on PATH" },
        @{ exe = "python3.12"; args = @(); label = "python3.12 on PATH" },
        @{ exe = "python3.11"; args = @(); label = "python3.11 on PATH" },
        @{ exe = "py"; args = @("-3.12"); label = "py -3.12" },
        @{ exe = "py"; args = @("-3.11"); label = "py -3.11" }
    )

    $known = @()
    if ($env:LOCALAPPDATA) {
        $known += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe")
        $known += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe")
    }
    if ($env:ProgramFiles) {
        $known += (Join-Path $env:ProgramFiles "Python312\python.exe")
        $known += (Join-Path $env:ProgramFiles "Python311\python.exe")
    }
    $known += "C:\Python312\python.exe"
    $known += "C:\Python311\python.exe"

    foreach ($exe in $known) {
        if (Test-Path $exe -PathType Leaf) {
            $candidates += @{ exe = $exe; args = @(); label = $exe }
        }
    }

    foreach ($candidate in $candidates) {
        $exe = [string]$candidate.exe
        $prefix = @($candidate.args)
        try {
            # The script below prints only the version number. This works
            # with the WindowsApps/Store Python alias if it launches Python.
            $version = (& $exe @prefix -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>$null | Select-Object -Last 1)
            if ($LASTEXITCODE -eq 0 -and "$version" -match '^3\.(11|12)\.\d+') {
                Write-Host "Found compatible Python $version via $($candidate.label)" -ForegroundColor Green
                return $candidate
            }
        } catch {
            # Command not installed or not working. Try next.
        }
    }
    return $null
}
if (-not (Test-Path $pythonExe)) {
    Write-Host "Checking available Python commands directly (including Windows Store aliases)..."
    $found = Find-CompatiblePython
    if ($null -eq $found) {
        Write-Host ""
        Write-Host "Could not execute compatible Python 3.11 or 3.12. A1111 and your RTX 3090 are unaffected." -ForegroundColor Yellow
        Write-Host "Install Python 3.12 for Windows (64-bit) from:"
        Write-Host "https://www.python.org/downloads/windows/"
        Write-Host "Choose the Windows installer (64-bit), not the embeddable package."
        Write-Host "In the installer, enable 'Add python.exe to PATH' if offered."
        Write-Host "Then run 01_SETUP_WINDOWS.bat again. No need to reinstall A1111."
        throw "Python 3.11/3.12 missing; no packages have been installed."
    }
    $basePython = $found.exe
    $baseArgs = $found.args
    Write-Host "Creating independent RTNVE Python environment..."
    & $basePython @baseArgs -m venv (Join-Path $root ".venv")
    Assert-OK "Creating RTNVE .venv"
    if (-not (Test-Path $pythonExe -PathType Leaf)) {
        throw "The Python environment was not created correctly. Check whether Python includes the venv module."
    }
}
Write-Host "Installing base Python dependencies..."
& $pythonExe -m pip install --upgrade pip
Assert-OK "Updating pip"

# CUDA wheel is essential on Windows. cu126 supports RTX 3090 with a
# sufficiently recent NVIDIA driver. Never silently install CPU-only PyTorch.
& $pythonExe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
Assert-OK "Installing CUDA-enabled PyTorch (cu126)"

& $pythonExe -m pip install "diffusers>=0.30,<0.43" "transformers>=4.37,<5" "accelerate>=0.31" safetensors imageio imageio-ffmpeg pillow omegaconf
Assert-OK "Installing Diffusers and video tools"

Write-Host ""
Write-Host "Checking CUDA and neural inference..."
& $pythonExe -c "import sys,torch,diffusers; print('Python',sys.version.split()[0]); print('Torch',torch.__version__); print('Diffusers',diffusers.__version__); print('CUDA available',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE: check NVIDIA driver and CUDA wheel'; print('GPU',torch.cuda.get_device_name(0)); print('GPU VRAM GiB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))"
Assert-OK "Verifying RTX 3090 / CUDA"

Write-Host ""
Write-Host "PASS - RTNVE is ready for its first AnimateDiff V3 benchmark." -ForegroundColor Green
