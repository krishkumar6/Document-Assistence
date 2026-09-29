<#
.SYNOPSIS
  Copy the committed project + deploy/oracle/.env to the Oracle server and (re)start it.

.EXAMPLE
  .\deploy\oracle\upload.ps1 -Ip 129.146.10.20 -Key $HOME\.ssh\oracle.key

  Run from the project folder. Uses the Windows built-in ssh/scp. Only committed
  files are sent (git archive), plus the git-ignored deploy/oracle/.env with your keys.
  Re-run it after every change you commit to deploy the update; uploaded PDFs are kept.
#>
param(
    [Parameter(Mandatory = $true)][string]$Ip,
    [string]$Key = "$HOME\.ssh\oracle.key",
    [string]$User = "ubuntu"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { throw "Create deploy\oracle\.env first (copy deploy\oracle\env.example and fill it in)." }
if (-not (Test-Path $Key)) { throw "SSH key not found: $Key (the private key you downloaded when creating the Oracle instance)." }
if (git -C $root status --porcelain) { Write-Warning "You have uncommitted changes; only committed files are deployed." }

# Windows OpenSSH rejects private keys other users can read ("UNPROTECTED PRIVATE KEY FILE").
# Restrict the downloaded key to the current user.
icacls $Key /inheritance:r /grant:r "$($env:USERNAME):R" | Out-Null

$tar = Join-Path $env:TEMP "document-assistant.tar"
git -C $root archive --format=tar -o $tar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }

$target = "$User@$Ip"
$sshOpts = @("-i", $Key, "-o", "StrictHostKeyChecking=accept-new")
Write-Host "==> Uploading to $target"
scp @sshOpts $tar "${target}:/tmp/document-assistant.tar"
scp @sshOpts $envFile "${target}:/tmp/document-assistant.env"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

# Unpack over the old code (deploy/oracle/data with the PDFs and index is not in the archive, so it's kept).
$remote = "mkdir -p ~/document-assistant && tar -xf /tmp/document-assistant.tar -C ~/document-assistant " +
          "&& mv /tmp/document-assistant.env ~/document-assistant/deploy/oracle/.env && chmod 600 ~/document-assistant/deploy/oracle/.env " +
          "&& rm /tmp/document-assistant.tar && bash ~/document-assistant/deploy/oracle/setup.sh"
ssh @sshOpts $target $remote
