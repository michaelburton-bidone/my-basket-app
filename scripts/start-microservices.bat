@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting microservices in development mode...

REM Kill any existing processes on the ports
echo 🔄 Killing existing processes...
for %%p in (3000,3001,3002,3003,3004) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| find ":%%p" ^| find "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

REM Start all services
echo 📦 Starting Product Service on port 3001...
start "Product Service" cmd /k "cd microservices\product-service && npm install && npm run dev"

echo 📦 Starting Cart Service on port 3002...
start "Cart Service" cmd /k "cd microservices\cart-service && npm install && npm run dev"

echo 📦 Starting Order Service on port 3003...
start "Order Service" cmd /k "cd microservices\order-service && npm install && npm run dev"

echo 📦 Starting AI Service on port 3004...
start "AI Service" cmd /k "cd microservices\ai-service && npm install && npm run dev"

echo 📦 Starting API Gateway on port 3000...
start "API Gateway" cmd /k "cd microservices\api-gateway && npm install && npm run dev"

echo ⏳ Waiting for services to start up...
timeout /t 10 >nul

echo ✅ All microservices are starting up!
echo.
echo 📋 Service URLs:
echo    🌐 API Gateway:     http://localhost:3000
echo    📦 Product Service: http://localhost:3001
echo    🛒 Cart Service:    http://localhost:3002
echo    📋 Order Service:   http://localhost:3003
echo    🤖 AI Service:      http://localhost:3004
echo.
echo 🔍 Health checks:
echo    curl http://localhost:3000/health
echo.
echo ⚠️  Note: Run 'npm run dev' in the main directory to start the Next.js frontend
echo    Frontend will be available at: http://localhost:9002

pause