param([switch]$UseWsl)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if ($UseWsl) {
    wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /home/gaosong/Code/Project/AgentGS && bash scripts/tools/install_all.sh'
    if ($LASTEXITCODE -ne 0) { throw 'WSL installation failed' }
} else {
    $python = Join-Path $root '.venv/Scripts/python.exe'
    & $python (Join-Path $PSScriptRoot 'install_node.py')
    if ($LASTEXITCODE -ne 0) { throw 'Node installation failed' }
    Write-Host 'Linux sandbox tools are installed through -UseWsl.'
}
