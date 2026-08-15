# deploy-test.ps1 — ProsperoAI deploy loop
#
# Builds the PS5 payload, injects it into the console, waits for the
# payload to finish and pulls the on-console log via FTP.
#
# Usage:
#   pwsh toolchain/deploy-test.ps1 [-Ps5Ip 192.168.1.7] [-Ps5Port 9021]
#                                  [-FtpPort 2120] [-TimeoutSec 120]
#
# Exit code: 0 = run captured and summarized; 1 = no log captured.

param(
  [string]$Ps5Ip = "192.168.1.7",
  [int]$Ps5Port = 9021,
  [int]$FtpPort = 2120,
  [string]$FtpUser = "anonymous",
  [string]$FtpPass = "test",
  [int]$TimeoutSec = 150
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$root = (Resolve-Path $root).Path

Write-Output "== build =="
cmake --preset ps5-debug 2>&1 | Out-Null
cmake --build --preset ps5-debug
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$elf = Join-Path $root "build\ps5-debug\payload\prosperoai.elf"
$plink = Join-Path $root "ps5-payload-sdk\win\plink.exe"
$runDir = Join-Path $root "build\ps5-runs"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$localLog = Join-Path $runDir "run-$stamp.log"

Write-Output "== deploy to ${Ps5Ip}:${Ps5Port} =="
cmd /c "type `"$elf`" | `"$plink`" -raw -P $Ps5Port $Ps5Ip" | Out-Null

Write-Output "== waiting for console log (FTP :${FtpPort}) =="
$logUrl = "ftp://${Ps5Ip}:${FtpPort}/data/prosperoai/prosperoai.log"
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$prevSize = -1
$stableCount = 0
$found = $false

while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 2
  curl.exe -s --connect-timeout 5 "$logUrl" --user "${FtpUser}:${FtpPass}" -o $localLog 2>$null
  if (($LASTEXITCODE -eq 0) -and (Test-Path $localLog)) {
    $size = (Get-Item $localLog).Length
    $found = $true
    if ($size -eq $prevSize) {
      $stableCount++
    } else {
      $stableCount = 0
    }
    $prevSize = $size
    if ($stableCount -ge 3) { break }
  }
}

if (-not $found) {
  Write-Output "NO LOG CAPTURED (console unresponsive or log path missing)"
  exit 1
}

Write-Output "== summary ($((Get-Item $localLog).Length) bytes) =="
Get-Content $localLog | Where-Object { $_ -match "M0-|PAI-M0:|ERROR|WARN" } |
  Select-Object -Last 50
Write-Output "== full log: $localLog =="
