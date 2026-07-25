@echo off
title QuantTrader - Claude Code
cd /d "%~dp0"

:: Self-register for Windows auto-start (P8, owner-approved 2026-07-25):
:: first run registers this same script to launch again at every logon,
:: so after today's run "no manual launch needed" is actually true.
schtasks /query /tn "QuantTrader" >nul 2>nul
if errorlevel 1 (
    schtasks /create /tn "QuantTrader" /tr "\"%~dpnx0\"" /sc onlogon /f >nul
    if not errorlevel 1 echo Registered QuantTrader to auto-start at Windows logon.
)

where claude >nul 2>nul
if errorlevel 1 (
    echo Claude Code CLI not found on PATH.
    echo Open a normal terminal in this folder and run: claude
    pause
    exit /b 1
)
claude
