# Explicit user-invoked sync of RTNVE source scripts from its GitHub main branch.
# Never edits the .venv, PyTorch, CUDA, A1111 or locally generated results.
# Save this installer in scripts\00_Update.ps1 and run 00_UPDATE_RTNVE.bat.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$base = "https://raw.githubusercontent.com/tmoreno-01/RTNVE/main"
$files = @(
    "benchmarks/animatediff_long_context.py",
    "benchmarks/prompt_travel.py",
    "benchmarks/animatediff_v3.py",
    "scripts/03_Run_Temporal.ps1",
    "03_BENCHMARK_TEMPORAL.bat",
    "training/export_webgpu_student_probe.py",
    "webgpu-student-probe/index.html",
    "scripts/04_Build_Probe.ps1",
    "04_BUILD_WEBGPU_PROBE.bat",
    "training/test_pretrained_video_vae.py",
    "scripts/05_Test_VAE_Quality.ps1",
    "05_TEST_NEURAL_CODEC.bat",
    "benchmarks/wild_prompt_travel.py",
    "benchmarks/overnight_1000.py",
    "scripts/06_Run_Overnight.ps1",
    "06_RENDER_1000_FRAMES.bat",
    "benchmarks/curated_prompt_travel.py",
    "scripts/07_Run_Curated.ps1",
    "07_RENDER_CURATED.bat",
    "scripts/08_Run_Curated_200.ps1",
    "08_RENDER_CURATED_200.bat",
    "benchmarks/regional_prompt_travel.py",
    "benchmarks/regional_loop_60.py",
    "scripts/09_Run_Regional_Loop.ps1",
    "09_RENDER_REGIONAL_LOOP.bat",
    "benchmarks/regional_loop_150.py",
    "scripts/10_Run_Regional_Loop_150.ps1",
    "10_RENDER_REGIONAL_LOOP_150.bat",
    "benchmarks/regional_loop_150_softwild.py",
    "scripts/11_Run_Regional_Softwild.ps1",
    "11_RENDER_REGIONAL_SOFTWILD.bat",
    "benchmarks/global_softwild_prompt_travel.py",
    "benchmarks/global_softwild_150.py",
    "scripts/12_Run_Global_Softwild.ps1",
    "12_RENDER_GLOBAL_SOFTWILD.bat",
    "benchmarks/reference_match_triplet.py",
    "scripts/13_Run_Reference_Triplet.ps1",
    "13_RENDER_REFERENCE_TRIPLET.bat",
    "benchmarks/highercar_300_triplet.py",
    "scripts/14_Run_Highercar_300_Triplet.ps1",
    "14_RENDER_HIGHERCAR_300_TRIPLET.bat",
    "15_RENDER_HIGHERCAR_900P_TRIPLET.bat",
    "16_RENDER_HIGHERCAR_900P_70F_TRIPLET.bat",
    "benchmarks/highercar_900p_comparison.py",
    "scripts/17_Run_Highercar_900P_Comparison.ps1",
    "17_RENDER_HIGHERCAR_900P_COMPARISON_SUITE.bat",
    "18_RENDER_HIGHERCAR_720P_300F_TRIPLET.bat",
    "00_UPDATE_RTNVE.bat"
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
            if (-not $contents.Contains('RTNVE_CONTEXT_PATCH = "2026-11-23-nonloop-travel-v3"')) {
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
