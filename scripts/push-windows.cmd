@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push-windows.ps1" %*
if errorlevel 1 pause
