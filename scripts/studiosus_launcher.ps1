# Studiosus launcher - stand the whole local stack up correctly, then hand
# the window to Hermes.
#
# Double-click Studiosus.bat (repo root) or run this directly:
#   powershell -ExecutionPolicy Bypass -File scripts\studiosus_launcher.ps1
#   ... -NoLaunch      configure + verify only, do not start the CLI
#   ... -Model <tag>   a different flame (default qwen3.6:latest)
#   ... -NumCtx <n>    a different context (default: the dyno profile's choice)
#
# Why this exists: Ollama's OpenAI-compatible /v1 endpoint silently discards
# num_ctx (measured 2026-07-19 - /api/generate honors it, /v1 does not), so no
# value Hermes computes can reach the server, and every /v1 turn reloads the
# model at the server's own default. The ONLY reliable fix is server-side:
# OLLAMA_CONTEXT_LENGTH in the service environment. This launcher writes that
# as a systemd drop-in inside WSL (idempotent - it restarts the service only
# when the value actually changes), tends the flame, and only then starts
# Hermes. Someone on different hardware edits $Model/$NumCtx or measures with
# `python -m agent.dyno` and the launcher follows their profile.

param(
    [string]$Model = "qwen3.6:latest",
    [int]$NumCtx = 0,          # 0 = resolve from the dyno profile
    [string]$Distro = "Ubuntu",
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# Prefer the repo venv; fall back to whatever python is on PATH.
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

# 1. Resolve the operating context from the dyno profile (a human's choice
#    wins over the bench's recommendation) unless -NumCtx pinned it.
if ($NumCtx -le 0) {
    try {
        $NumCtx = [int](& $Python -c "from agent.dyno import load_profile, operating_num_ctx; print(operating_num_ctx(load_profile('$Model')) or 0)")
    } catch { $NumCtx = 0 }
    if ($NumCtx -le 0) {
        Write-Host "no dyno profile for $Model - measure one with: python -m agent.dyno --model $Model" -ForegroundColor Yellow
        Write-Host "falling back to num_ctx=32768 (the Hermes minimum)" -ForegroundColor Yellow
        $NumCtx = 32768
    }
}
Step "flame: $Model at num_ctx=$NumCtx"

# 2. Server-side context. /v1 discards per-request num_ctx, so the server
#    itself must be told. Idempotent: touch nothing when already right.
Step "checking the WSL Ollama service environment"
$Conf = "/etc/systemd/system/ollama.service.d/studiosus-context.conf"
$Current = (wsl -d $Distro -u root -- sh -c "grep -o 'OLLAMA_CONTEXT_LENGTH=[0-9]*' $Conf 2>/dev/null | head -1") -replace '[^0-9]', ''
if ($Current -eq "$NumCtx") {
    Step "server already carries OLLAMA_CONTEXT_LENGTH=$NumCtx"
} else {
    Step "writing OLLAMA_CONTEXT_LENGTH=$NumCtx and restarting ollama"
    wsl -d $Distro -u root -- sh -c "mkdir -p /etc/systemd/system/ollama.service.d && printf '[Service]\nEnvironment=OLLAMA_CONTEXT_LENGTH=$NumCtx\n' > $Conf && systemctl daemon-reload && systemctl restart ollama"
    if ($LASTEXITCODE -ne 0) { throw "could not configure the ollama service (is WSL distro '$Distro' running?)" }
    # The service takes a moment to answer again.
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3
            break
        } catch { Start-Sleep -Seconds 2 }
    }
}

# 3. Tend the flame: release anything competing for the card, load the model
#    at the chosen context, pinned. agent/flame.py refuses the unsafe fixes
#    (it will not pull an uninstalled model or start a dead server).
Step "tending the flame"
$env:OLLAMA_HOST = ""   # the Windows-side 0.0.0.0 bind-address trap
& $Python -m agent.flame --model $Model --num-ctx $NumCtx --ensure --clear-conflicts
if ($LASTEXITCODE -ne 0) {
    Write-Host "flame not ready - see the report above. Hermes will still start; the first turn will be slow." -ForegroundColor Yellow
}

# 4. The Studiosus heart, on for this window: episodes, reflection, tending.
$env:HERMES_SOUL = "1"
$env:HERMES_CHRONICLE = "1"
$env:HERMES_FLAME = "1"

if ($NoLaunch) {
    Step "-NoLaunch: stack is up; not starting the CLI"
    exit 0
}

# 5. Hand the window to Hermes.
Step "starting Hermes"
& $Python -m hermes_cli.main
