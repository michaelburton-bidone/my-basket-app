@echo off
echo 🛑 Stopping all microservices...

REM Kill processes on specific ports
for %%p in (3000,3001,3002,3003,3004,9002) do (
    echo    🔄 Stopping service on port %%p...
    for /f "tokens=5" %%a in ('netstat -aon ^| find ":%%p" ^| find "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        if !errorlevel! == 0 (
            echo       ✅ Stopped process on port %%p
        ) else (
            echo       ⚠️  No process found on port %%p
        )
    )
)

echo ✅ All microservices stopped!