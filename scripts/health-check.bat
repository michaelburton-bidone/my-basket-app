@echo off
echo 🏥 Testing microservice health...

set "all_healthy=true"

echo    Testing API Gateway... 
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% == 0 (
    echo    ✅ Healthy
) else (
    echo    ❌ Unhealthy
    set "all_healthy=false"
)

echo    Testing Product Service... 
curl -s http://localhost:3001/api/health >nul 2>&1
if %errorlevel% == 0 (
    echo    ✅ Healthy
) else (
    echo    ❌ Unhealthy
    set "all_healthy=false"
)

echo    Testing Cart Service... 
curl -s http://localhost:3002/api/health >nul 2>&1
if %errorlevel% == 0 (
    echo    ✅ Healthy
) else (
    echo    ❌ Unhealthy
    set "all_healthy=false"
)

echo    Testing Order Service... 
curl -s http://localhost:3003/api/health >nul 2>&1
if %errorlevel% == 0 (
    echo    ✅ Healthy
) else (
    echo    ❌ Unhealthy
    set "all_healthy=false"
)

echo    Testing AI Service... 
curl -s http://localhost:3004/api/health >nul 2>&1
if %errorlevel% == 0 (
    echo    ✅ Healthy
) else (
    echo    ❌ Unhealthy
    set "all_healthy=false"
)

echo.
if "%all_healthy%"=="true" (
    echo 🎉 All microservices are healthy!
) else (
    echo ⚠️  Some microservices are not responding
)