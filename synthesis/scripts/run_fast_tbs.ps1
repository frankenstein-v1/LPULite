param(
    [string]$Iverilog = "C:\iverilog\bin\iverilog.exe",
    [string]$Vvp = "C:\iverilog\bin\vvp.exe"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$simDir = Join-Path $repo ".sim"
New-Item -ItemType Directory -Force -Path $simDir | Out-Null

function Run-Tb {
    param(
        [string]$Name,
        [string[]]$Sources
    )

    $out = Join-Path $simDir "$Name.vvp"
    $sourcePaths = @()
    foreach ($src in $Sources) {
        $sourcePaths += Join-Path $repo $src
    }

    & $Iverilog -g2012 -I (Join-Path $repo "src") -s $Name -o $out @sourcePaths
    if ($LASTEXITCODE -ne 0) {
        throw "iverilog failed for $Name"
    }

    & $Vvp $out
    if ($LASTEXITCODE -ne 0) {
        throw "simulation failed for $Name"
    }
}

Run-Tb -Name "rmsnorm_chunk_tb" -Sources @(
    "tb\rmsnorm_chunk_tb.sv",
    "src\rmsnorm.sv"
)

Run-Tb -Name "quant_q8_8_tb" -Sources @(
    "tb\quant_q8_8_tb.sv",
    "src\quant.sv"
)

Run-Tb -Name "mac_mixed_scale_tb" -Sources @(
    "tb\mac_mixed_scale_tb.sv",
    "src\mac.sv"
)

Run-Tb -Name "softmax_lut_tb" -Sources @(
    "tb\softmax_lut_tb.sv",
    "src\lut_softmax_exp.sv",
    "src\lut_softmax_div.sv",
    "src\softmax.sv"
)

Write-Host "FAST_TB_ALL_PASS"
