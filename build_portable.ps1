$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv is required for building.' }
$oldPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $projectRoot
try {
    & $python -m pip install --upgrade --target build/tools -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw 'Build tool installation failed.' }
    $env:PYTHONPATH = Join-Path $projectRoot 'build/tools'
    & $python -m PyInstaller --noconfirm --workpath build/portable portable.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    $output = Join-Path $projectRoot 'dist/LocalTranscriber-portable'
    $weights = Join-Path $output 'models/whisper-russian'
    New-Item -ItemType Directory -Path $weights -Force | Out-Null
    foreach ($name in @('config.json','model.bin','preprocessor_config.json','tokenizer.json','vocabulary.json')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot ('models/whisper-russian/' + $name)) -Destination (Join-Path $weights $name)
    }
    [IO.File]::WriteAllText((Join-Path $output 'portable.flag'), 'LocalTranscriber portable')
    Copy-Item -LiteralPath (Join-Path $projectRoot 'PORTABLE.md') -Destination (Join-Path $output 'README.md')
    Copy-Item -LiteralPath (Join-Path $projectRoot 'components.json') -Destination $output
    $size = (Get-ChildItem -LiteralPath $output -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Host ('Portable folder: ' + $output)
    Write-Host ('Size GB: {0:N3}' -f ($size / 1000000000))
    if ($size -gt 5000000000) { Write-Warning 'The 5 GB target has been exceeded.' }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
