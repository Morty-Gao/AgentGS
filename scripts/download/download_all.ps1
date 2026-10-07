# Run with: pwsh -File scripts/download/download_all.ps1 --verify-only
$ErrorActionPreference = 'Stop'
$forwardArgs = @($args)
$repoRoot = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'pyproject.toml') -PathType Leaf)) {
    $parent = Split-Path -Parent $repoRoot
    if (-not $parent -or $parent -eq $repoRoot) { throw 'pyproject.toml not found' }
    $repoRoot = $parent
}
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $repoRoot '.venv/bin/python'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python not found; run uv sync first.' }
$failed = 0
foreach ($script in (Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'download_*.py' -File | Sort-Object Name)) {
    & $python -X utf8 $script.FullName @forwardArgs
    if ($LASTEXITCODE -ne 0) { $failed++ }
}
if ($failed -gt 0) {
    [Console]::Error.WriteLine("$failed dataset commands failed; see errors above.")
    exit 1
}
exit 0
