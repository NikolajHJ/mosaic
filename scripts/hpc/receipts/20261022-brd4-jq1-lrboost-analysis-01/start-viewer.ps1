$ErrorActionPreference = 'Stop'
$mosaicRoot = 'C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic'
$viewerRoot = Join-Path $mosaicRoot 'results\hpc\20261022-brd4-jq1-lrboost-analysis-01\viewer'
$outputRoot = Split-Path $viewerRoot
$pythonPath = 'C:\Users\nikol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$receiptPath = Join-Path $mosaicRoot 'scripts\hpc\receipts\20261022-brd4-jq1-lrboost-analysis-01\viewer-server.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Existing server receipt requires inspection before any recovery' }
if (-not (Test-Path -LiteralPath (Join-Path $viewerRoot 'latest.json'))) { throw 'Publish a verified edition before starting its viewer' }
$listener = Get-NetTCPConnection -LocalPort 8778 -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw 'Port8778 already has a listener; inspect it before recovery, never duplicate or replace blindly' }
$serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'http.server', '8778', '--bind', '127.0.0.1', '--directory', $viewerRoot) -WorkingDirectory $mosaicRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputRoot 'server-8778-stdout.txt') -RedirectStandardError (Join-Path $outputRoot 'server-8778-stderr.txt') -PassThru
@{ pid = $serverProcess.Id; port = 8778; host = '127.0.0.1'; directory = $viewerRoot; url = 'http://127.0.0.1:8778/report.html'; started_utc = [DateTime]::UtcNow.ToString('o'); scope = 'Separate saved-only presentation; historical8776/8777 unchanged' } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
Get-Content -LiteralPath $receiptPath
