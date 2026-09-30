[CmdletBinding()]
param(
    [ValidateSet("stable", "latest")]
    [string]$CoreChannel = "stable",
    [switch]$SkipDependencySync,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host "[SYNC] $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(ValueFromRemainingArguments)] [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory)] [string]$Repository,
        [Parameter(ValueFromRemainingArguments)] [string[]]$Arguments
    )

    $safePath = $Repository.Replace("\", "/")
    $gitArguments = @("-c", "safe.directory=$safePath")
    if ($env:HTTP_PROXY) {
        $gitArguments += @("-c", "http.proxy=$env:HTTP_PROXY")
    }
    if ($env:HTTPS_PROXY) {
        $gitArguments += @("-c", "https.proxy=$env:HTTPS_PROXY")
    }
    $gitArguments += @("-C", $Repository)
    $gitArguments += $Arguments
    Invoke-External git @gitArguments
}

function Invoke-GitClone {
    param(
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Destination
    )

    $gitArguments = @()
    if ($env:HTTP_PROXY) {
        $gitArguments += @("-c", "http.proxy=$env:HTTP_PROXY")
    }
    if ($env:HTTPS_PROXY) {
        $gitArguments += @("-c", "https.proxy=$env:HTTPS_PROXY")
    }
    $gitArguments += @("clone", "--origin", "origin", $Url, $Destination)
    Invoke-External git @gitArguments
}

function Test-GitRepository([string]$Repository) {
    if (-not (Test-Path -LiteralPath $Repository -PathType Container)) {
        return $false
    }
    return Test-Path -LiteralPath (Join-Path $Repository ".git")
}

function Test-CleanRepository([string]$Repository) {
    $safePath = $Repository.Replace("\", "/")
    $status = & git -c "safe.directory=$safePath" -C $Repository status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Git repository: $Repository"
    }
    return -not [bool]($status)
}

function Update-TrackedRepository {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Repository,
        [switch]$AllowDirtyDevelopmentTree
    )

    if (-not (Test-GitRepository $Repository)) {
        throw "$Name is not a Git repository: $Repository"
    }

    if (-not (Test-CleanRepository $Repository)) {
        if ($AllowDirtyDevelopmentTree) {
            Write-Warning "$Name has local development changes. Remote update skipped; the working tree remains authoritative."
            return
        }
        throw "$Name has local changes. Commit or stash them before updating."
    }

    Write-Step "Updating $Name with fast-forward only"
    Invoke-Git $Repository fetch --prune origin

    $safePath = $Repository.Replace("\", "/")
    $branch = (& git -c "safe.directory=$safePath" -C $Repository branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $branch) {
        throw "$Name is in detached HEAD state; refusing to guess which branch to update."
    }

    $upstream = (& git -c "safe.directory=$safePath" -C $Repository rev-parse --abbrev-ref --symbolic-full-name "@{upstream}" 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $upstream) {
        throw "$Name branch '$branch' has no upstream tracking branch."
    }

    Invoke-Git $Repository merge --ff-only $upstream
}

function Update-ComfyUIRepository {
    param(
        [Parameter(Mandatory)] [string]$Repository,
        [Parameter(Mandatory)] [ValidateSet("stable", "latest")] [string]$Channel
    )

    if (-not (Test-GitRepository $Repository)) {
        throw "ComfyUI is not a Git repository: $Repository"
    }
    if (-not (Test-CleanRepository $Repository)) {
        throw "ComfyUI has local changes. Refusing to overwrite them during update."
    }

    Write-Step "Fetching ComfyUI branches and tags"
    Invoke-Git $Repository fetch --prune --tags origin

    if ($Channel -eq "stable") {
        $safePath = $Repository.Replace("\", "/")
        $tags = @(& git -c "safe.directory=$safePath" -C $Repository tag --list "v*" --sort=-version:refname)
        $tag = $tags | Where-Object { $_ -match '^v\d+\.\d+\.\d+$' } | Select-Object -First 1
        if (-not $tag) {
            throw "No ComfyUI stable version tag was found."
        }
        Write-Step "Checking out ComfyUI stable tag $tag"
        Invoke-Git $Repository checkout --detach $tag
        return
    }

    Write-Step "Fast-forwarding ComfyUI master"
    Invoke-Git $Repository checkout master
    Invoke-Git $Repository merge --ff-only origin/master
}

function Get-RepositoryCommit([string]$Repository) {
    $safePath = $Repository.Replace("\", "/")
    $commit = (& git -c "safe.directory=$safePath" -C $Repository rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read Git commit: $Repository"
    }
    return $commit
}

if (-not $env:COMFYUI_ROOT) {
    throw "COMFYUI_ROOT is not defined. Load dev_environment.local.bat first."
}
if (-not $env:COMFYUI_PYTHON) {
    throw "COMFYUI_PYTHON is not defined. Load dev_environment.local.bat first."
}

$comfyRoot = [System.IO.Path]::GetFullPath($env:COMFYUI_ROOT)
$pythonPath = [System.IO.Path]::GetFullPath($env:COMFYUI_PYTHON)
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$kjNodesRoot = Join-Path $comfyRoot "custom_nodes\comfyui-kjnodes"
$saiLink = Join-Path $comfyRoot "custom_nodes\ComfyUI-Sai-nodes"
$stateDirectory = Join-Path $projectRoot ".dev"
$statePath = Join-Path $stateDirectory "environment-state.json"

foreach ($requiredPath in @(
    (Join-Path $comfyRoot "main.py"),
    $pythonPath,
    (Join-Path $projectRoot "AGENTS.md")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

$resolvedSaiLink = (Get-Item -LiteralPath $saiLink -ErrorAction Stop).Target
if (-not $resolvedSaiLink) {
    $resolvedSaiLink = (Get-Item -LiteralPath $saiLink -ErrorAction Stop).FullName
}
$resolvedSaiLink = [System.IO.Path]::GetFullPath([string]$resolvedSaiLink)
if (-not $resolvedSaiLink.TrimEnd("\").Equals($projectRoot.TrimEnd("\"), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ComfyUI-Sai-nodes must point to the development workspace. Found: $resolvedSaiLink"
}

Write-Step "Updating ComfyUI ($CoreChannel channel)"
Update-ComfyUIRepository -Repository $comfyRoot -Channel $CoreChannel

if (-not (Test-GitRepository $kjNodesRoot)) {
    if (Test-Path -LiteralPath $kjNodesRoot) {
        $backupDirectory = Join-Path $stateDirectory "backups"
        if (-not (Test-Path -LiteralPath $backupDirectory)) {
            New-Item -ItemType Directory -Path $backupDirectory | Out-Null
        }
        $backupPath = Join-Path $backupDirectory "comfyui-kjnodes.manager-snapshot-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Write-Step "Preserving the non-Git KJNodes snapshot at $backupPath"
        Move-Item -LiteralPath $kjNodesRoot -Destination $backupPath
    }
    Write-Step "Cloning the official KJNodes repository"
    Invoke-GitClone -Url "https://github.com/kijai/ComfyUI-KJNodes.git" -Destination $kjNodesRoot
}

Update-TrackedRepository -Name "ComfyUI-KJNodes" -Repository $kjNodesRoot
Update-TrackedRepository -Name "ComfyUI-Sai-nodes" -Repository $projectRoot -AllowDirtyDevelopmentTree

if (-not $SkipDependencySync) {
    Write-Step "Synchronizing runtime dependencies"
    foreach ($requirementsPath in @(
        (Join-Path $comfyRoot "requirements.txt"),
        (Join-Path $kjNodesRoot "requirements.txt"),
        (Join-Path $projectRoot "requirements.txt")
    )) {
        Invoke-External $pythonPath -s -m pip install --disable-pip-version-check -r $requirementsPath
    }
    Invoke-External $pythonPath -s -m pip check
}

if (-not (Test-Path -LiteralPath $stateDirectory)) {
    New-Item -ItemType Directory -Path $stateDirectory | Out-Null
}

$state = [ordered]@{
    synchronized_at = (Get-Date).ToString("o")
    core_channel = $CoreChannel
    comfyui_commit = Get-RepositoryCommit $comfyRoot
    kjnodes_commit = Get-RepositoryCommit $kjNodesRoot
    sai_nodes_commit = Get-RepositoryCommit $projectRoot
    sai_nodes_has_local_changes = -not (Test-CleanRepository $projectRoot)
    python = $pythonPath
}
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8

if (-not $SkipSmokeTest) {
    Write-Step "Running isolated ComfyUI import smoke test"
    $smokeLauncher = Join-Path $PSScriptRoot "start_dev_comfyui.bat"
    Invoke-External "$env:SystemRoot\System32\cmd.exe" /d /c $smokeLauncher --check
}

Write-Host "[OK] Development environment synchronized." -ForegroundColor Green
Write-Host "[OK] State: $statePath" -ForegroundColor Green
