# Explicit user-invoked sync of RTNVE source scripts from its GitHub main branch.
# Never edits the .venv, PyTorch, CUDA, A1111 or locally generated results.
# Save this installer in scripts\00_Update.ps1 and run 00_UPDATE_RTNVE.bat.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$base = "https://raw.githubusercontent.com/tmoreno-01/RTNVE/main"
$files = @(
    "benchmarks/animatediff_long_context.py",
    "benchmarks/animatediff_v3.py",
    "scripts/03_Run_Temporal.ps1",
    "03_BENCHMARK_TEMPORAL.bat"
)
Write-Host ""
Write-Host "RTNVE - download updated source files from GitHub" -ForegroundColor Cyan
Write-Host "Local project: $root"
Write-Host "Your GPU environment, existing A1111 installation and results stay untouched."
Write-Host ""
foreach ($relative in $files) {
    $dest = Join-Path $root ($relative.Replace("/", "\"))
    $pending = "$dest.rtnve-download"
    $parent = Split-Path -Parent $dest
    if (-not (Test-Path $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $url = "$base/$relative"
    Write-Host "Fetching $relative"
    try {
        Invoke-WebRequest -Uri $url -OutFile $pending -UseBasicParsing -ErrorAction Stop
        if (-not (Test-Path $pending -PathType Leaf) -or (Get-Item $pending).Length -lt 100) {
            throw "File download incomplete: $relative"
        }
        if ($relative -eq "benchmarks/animatediff_long_context.py") {
            $contents = Get-Content -LiteralPath $pending -Raw -Encoding UTF8
            if (-not $contents.Contains('RTNVE_CONTEXT_PATCH = "2026-11-23-cfg-window-v2"')) {
                throw "Downloaded temporal script is the wrong version. Keeping your old file untouched."
            }
        }
        # Overwrite only after the full download has passed checks.
        Move-Item -LiteralPath $pending -Destination $dest -Force
        Write-Host "Updated: $relative" -ForegroundColor Green
    } finally {
        if (Test-Path $pending -PathType Leaf) {
            Remove-Item -LiteralPath $pending -Force
        }
    }
}
Write-Host ""
Write-Host "All RTNVE scripts updated. No model downloads or Python setup needed." -ForegroundColor Green
