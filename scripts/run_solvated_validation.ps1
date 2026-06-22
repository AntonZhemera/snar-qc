<#
.SYNOPSIS
  Single-command launcher (Windows) for the solvated cross-leaving-group ΔG‡ re-validation.

.DESCRIPTION
  One command runs the whole campaign on the extended Lu slice:
    1. the resumable parallel batch (run_poc_batch.sh) with PCM solvation on, then
    2. the validation join + per-leaving-group stats + scatter (validate_poc.py).
  The batch step is a bash script (round-robin workers), so it is invoked through `bash`
  from Git Bash / WSL; the Python steps run with the active snar-qc conda env's python.
  Each substrate writes a resumable per-substrate sidecar, so a crash/restart just
  re-runs the unfinished ones. The transition-state Hessian + PCM dominate the multi-hour
  per-substrate cost; budget generously.

.EXAMPLE
  # from the repo root, with the snar-qc conda env active:
  conda activate snar-qc
  pwsh scripts/run_solvated_validation.ps1

.NOTES
  Override any of the parameters below on the command line, e.g.
    pwsh scripts/run_solvated_validation.ps1 -Solvent DMSO -NWorkers 4 -Threads 4 -MemGb 5
#>
param(
    [string]$Slice      = "data/external/lu74_solv_slice.csv",
    [string]$OutDir     = "data/processed/solv_run",
    [string]$Solvent    = "DMSO",          # matches Lu's reaction solvent
    [ValidateSet("concerted", "addition")]
    [string]$Coordinate = "concerted",
    [int]$NWorkers      = 4,
    [int]$Threads       = 4,
    [int]$MemGb         = 5
)

$ErrorActionPreference = "Stop"
# Repo root = parent of this script's directory.
$Here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Here

# run_poc_batch.sh reads these as environment variables.
$env:SOLVENT    = $Solvent
$env:COORDINATE = $Coordinate
$env:N_WORKERS  = $NWorkers
$env:THREADS    = $Threads
$env:MEM_GB     = $MemGb

Write-Host "=== solvated ΔG‡ re-validation ==="
Write-Host "slice=$Slice outdir=$OutDir solvent=$Solvent coordinate=$Coordinate"
Write-Host "workers=$NWorkers threads=$Threads mem_gb=$MemGb"

# 1. compute (resumable; PCM solvation on via SOLVENT). The batch is a shell script,
#    so route it through Git Bash.
#
#    NB: a bare `bash` on Windows usually resolves to WSL's C:\Windows\System32\bash.exe,
#    which runs a *separate* Linux environment: it neither sees the conda env's `python`
#    nor inherits the SOLVENT/N_WORKERS/THREADS/MEM_GB vars set above. Git Bash inherits
#    both, so we locate it explicitly (relative to git on PATH, then well-known paths).
$bash = $null
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
    $gitRoot = Split-Path -Parent (Split-Path -Parent $gitCmd.Source)
    foreach ($c in @("$gitRoot\bin\bash.exe", "$gitRoot\usr\bin\bash.exe")) {
        if (Test-Path $c) { $bash = $c; break }
    }
}
if (-not $bash) {
    foreach ($c in @("C:\Program Files\Git\bin\bash.exe", "C:\Program Files (x86)\Git\bin\bash.exe")) {
        if (Test-Path $c) { $bash = $c; break }
    }
}
if (-not $bash) {
    throw "Git Bash not found. Install Git for Windows, or run scripts/run_poc_batch.sh from a bash shell."
}

& $bash scripts/run_poc_batch.sh $Slice $OutDir
if ($LASTEXITCODE -ne 0) { throw "run_poc_batch.sh failed (exit $LASTEXITCODE)" }

# 2. validate (correlation, per-leaving-group breakdown, scatter).
python scripts/validate_poc.py --slice $Slice --run $OutDir --outdir notes/assets
if ($LASTEXITCODE -ne 0) { throw "validate_poc.py failed (exit $LASTEXITCODE)" }

Write-Host "Done. Stats/scatter under notes/assets; sidecars under $OutDir."
