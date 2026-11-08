# Ask for the Gemini key here and run SynthEx end to end with it.
#
# The key is typed into this prompt, never into a command line, so it does
# not land in PowerShell history. It lives in the environment of this run
# only and is cleared at the end.
#
#     & ".\evaluation\scripts\run_synthex_probe.ps1"
#     & ".\evaluation\scripts\run_synthex_probe.ps1" "CCO"

param([string]$Target = "")

$ErrorActionPreference = "Stop"
$prompt = "Gemini API key (paste and press Enter)"

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $key = Read-Host $prompt -MaskInput
} else {
    $secure = Read-Host $prompt -AsSecureString
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

if ([string]::IsNullOrWhiteSpace($key)) {
    Write-Output "키가 비었습니다. 다시 실행하세요."
    exit 2
}

$env:GEMINI_API_KEY = $key
try {
    $python = "$PSScriptRoot\..\..\.venv\Scripts\python.exe"
    $script = "$PSScriptRoot\run_synthex_probe.py"
    if ([string]::IsNullOrWhiteSpace($Target)) {
        & $python $script
    } else {
        & $python $script $Target
    }
} finally {
    $env:GEMINI_API_KEY = ""
    $key = $null
}
