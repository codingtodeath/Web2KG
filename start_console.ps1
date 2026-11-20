param(
    [int]$Port = 8787,
    [string]$Graph = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$gitCommon = (git rev-parse --path-format=absolute --git-common-dir).Trim()
$projectPython = Join-Path (Split-Path $gitCommon -Parent) '.venv-research/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Project Python missing: $projectPython"
}
$consoleArgs = @('scripts/start_control.py', '--port', "$Port")
if ($Graph) { $consoleArgs += @('--graph', $Graph) }
& $projectPython @consoleArgs
exit $LASTEXITCODE
