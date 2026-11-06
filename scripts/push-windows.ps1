[CmdletBinding()]
param(
    [string]$Video = "",
    [string]$RtmpUrl = "rtmp://localhost:1935/drone1",
    [int]$Width = 1920,
    [int]$Height = 1080,
    [int]$Fps = 25,
    [string]$Bitrate = "4M",
    [string]$BufSize = "8M",
    [string]$Preset = "veryfast",
    [string]$FfmpegPath = "ffmpeg",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Video)) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $Video = Join-Path $repoRoot "assets\drone.mp4"
}

if (-not (Test-Path -LiteralPath $Video)) {
    throw "Video not found: $Video"
}

$cmd = Get-Command $FfmpegPath -ErrorAction SilentlyContinue
if (-not $cmd) {
    throw "ffmpeg not found: $FfmpegPath. Install FFmpeg or pass -FfmpegPath."
}
$ffmpeg = $cmd.Path

$filter = "scale=${Width}:${Height}:force_original_aspect_ratio=decrease,pad=${Width}:${Height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"

$inputArgs = @("-re")
if (-not $Once) {
    $inputArgs += @("-stream_loop", "-1")
}
$inputArgs += @("-i", $Video)

$ffmpegArgs = @(
    "-nostdin", "-hide_banner", "-loglevel", "warning"
) + $inputArgs + @(
    "-vf", $filter,
    "-r", "$Fps",
    "-c:v", "libx264", "-preset", $Preset, "-tune", "zerolatency",
    "-profile:v", "baseline", "-bf", "0", "-g", "$Fps",
    "-b:v", $Bitrate, "-maxrate", $Bitrate, "-bufsize", $BufSize,
    "-an", "-f", "flv", $RtmpUrl
)

Write-Host "ffmpeg : $ffmpeg"
Write-Host "video  : $Video"
Write-Host "stream : $RtmpUrl"
Write-Host "output : ${Width}x${Height}@${Fps}, ${Bitrate}, preset=${Preset}"

if ($Once) {
    Write-Host "mode   : once"
} else {
    Write-Host "mode   : loop (Ctrl+C to stop)"
}

while ($true) {
    & $ffmpeg @ffmpegArgs
    $code = $LASTEXITCODE

    if ($Once) {
        exit $code
    }

    Write-Warning "ffmpeg exited (code $code); retrying in 2s..."
    Start-Sleep -Seconds 2
}
