[CmdletBinding()]
param(
    [string]$QuartusBin = "C:\altera_lite\25.1std\quartus\bin64",
    [string]$Python = "python",
    [switch]$SkipModelArtifacts,
    [switch]$SkipArmRuntime,
    [switch]$SkipQuartus
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )

    Write-Host "==> $Description"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

$repoPath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sofPath = Join-Path $repoPath "synthesis\build\lpu_lite_de1_soc_hps\lpu_lite_de1_soc_hps.sof"

Push-Location $repoPath
try {
    if (-not $SkipModelArtifacts) {
        Invoke-Checked -FilePath $Python `
            -Arguments @("model/tools/compile_microgpt_lpu.py") `
            -Description "Compile the INT8 checkpoint into MEM1 and VLIW images"
        Invoke-Checked -FilePath $Python `
            -Arguments @("synthesis/scripts/export_microgpt_hps_headers.py") `
            -Description "Export model and microcode images into the ARM C header"
    }

    if (-not $SkipArmRuntime) {
        $wslRepoPath = (& wsl.exe wslpath -a $repoPath).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRepoPath)) {
            throw "Could not translate the repository path with WSL."
        }
        $linuxPath = "$wslRepoPath/synthesis/linux"
        $buildCommand = "cd -- '$linuxPath' && make clean && make CC=arm-linux-gnueabihf-gcc LDFLAGS=-static"
        Invoke-Checked -FilePath "wsl.exe" `
            -Arguments @("bash", "-lc", $buildCommand) `
            -Description "Cross-compile the static ARM Cortex-A9 runtime in WSL"
    }

    if (-not $SkipQuartus) {
        $quartusSh = Join-Path $QuartusBin "quartus_sh.exe"
        if (-not (Test-Path -LiteralPath $quartusSh -PathType Leaf)) {
            throw "Quartus was not found at $quartusSh. Pass -QuartusBin with the correct bin64 directory."
        }

        $mappingCreated = $false
        $existingMapping = (& subst.exe) | Where-Object { $_ -match '^T:\\: => ' }
        if ($existingMapping) {
            $mappedPath = ($existingMapping -replace '^T:\\: => ', '').TrimEnd('\')
            if (-not $mappedPath.Equals($repoPath.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "T: is already mapped to '$mappedPath'. Remove it or choose the expected repository mapping."
            }
        } else {
            Invoke-Checked -FilePath "subst.exe" `
                -Arguments @("T:", $repoPath) `
                -Description "Map the repository to T: for Quartus-safe paths"
            $mappingCreated = $true
        }

        try {
            Invoke-Checked -FilePath $quartusSh `
                -Arguments @(
                    "--flow",
                    "compile",
                    "T:\synthesis\build\lpu_lite_de1_soc_hps\lpu_lite_de1_soc_hps"
                ) `
                -Description "Compile the existing LPULite HPS Quartus project"
        } finally {
            if ($mappingCreated) {
                & subst.exe T: /D
            }
        }

        if (-not (Test-Path -LiteralPath $sofPath -PathType Leaf)) {
            throw "Quartus completed without producing $sofPath"
        }
    }

    if (Test-Path -LiteralPath $sofPath -PathType Leaf) {
        $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $sofPath
        Write-Host "SOF:    $sofPath"
        Write-Host "SHA256: $($hash.Hash)"
    }

    Write-Host "Release build complete."
} finally {
    Pop-Location
}
