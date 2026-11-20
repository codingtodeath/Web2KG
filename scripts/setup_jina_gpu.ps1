param([string]$Conda = 'D:/Anaconda/Scripts/conda.exe')
$ErrorActionPreference = 'Stop'
$jinaPreviousConfig = $env:PIP_CONFIG_FILE
$jinaPreviousExtra = $env:PIP_EXTRA_INDEX_URL
$jinaProxyNames = @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'PIP_PROXY')
$jinaPreviousProxies = @{}
foreach ($name in $jinaProxyNames) {
    $jinaPreviousProxies[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    foreach ($name in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'PIP_PROXY')) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:NO_PROXY = '*'
    $env:PIP_CONFIG_FILE = 'NUL'
    $env:PIP_EXTRA_INDEX_URL = ''
    $jinaBase = (& $Conda info --base).Trim()
    if (-not (Test-Path -LiteralPath "$jinaBase/envs/web2kg-jina-gpu/python.exe")) {
        & $Conda create -n web2kg-jina-gpu python=3.12.15 pip -y --override-channels -c conda-forge
        if ($LASTEXITCODE) { throw 'Conda environment creation failed' }
    }
    & $Conda run -n web2kg-jina-gpu python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
    & $Conda run -n web2kg-jina-gpu python -m pip install -r "$PSScriptRoot/../environments/jina-gpu-lock.txt" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE) { throw 'Jina dependencies installation failed' }
} finally {
    $env:PIP_CONFIG_FILE = $jinaPreviousConfig
    $env:PIP_EXTRA_INDEX_URL = $jinaPreviousExtra
    foreach ($name in $jinaProxyNames) {
        [Environment]::SetEnvironmentVariable($name, $jinaPreviousProxies[$name], 'Process')
    }
}
