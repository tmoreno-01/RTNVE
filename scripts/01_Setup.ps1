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
    # Don't rely on "py", installation paths, or complex quoting in -c.
    # Windows PowerShell 5 can strip embedded double quotes from native
    # command arguments; the old version test returned invalid Python code.
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
            # Only numbers, no string literals, no double quotes to be lost.
            # "312" or "311" is returned by any usable Python installation.
            $versionCode = & $exe @prefix -c 'import sys; print(sys.version_info.major * 100 + sys.version_info.minor)' 2>$null
            $status = $LASTEXITCODE
            if ($status -eq 0 -and ("$versionCode").Trim() -in @("311", "312")) {
                Write-Host "Found Python $versionCode through $($candidate.label)" -ForegroundColor Green
                return $candidate
            }
            Write-Host "Tried $($candidate.label): exit $status, version code '$versionCode'"
        } catch {
            Write-Host "Tried $($candidate.label): $($_.Exception.Message)"
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
