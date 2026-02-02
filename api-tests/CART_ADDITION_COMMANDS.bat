@echo off
REM Cart Addition Test - Quick Commands for Windows
REM Run this file or use individual commands below

echo 🧪 Cart Addition Test - Command Reference
echo ==========================================
echo.

echo 1. Run all cart addition tests:
echo    npx playwright test tests/cart-addition.spec.ts
echo.

echo 2. Run specific test groups:
echo    Successful additions:
echo    npx playwright test tests/cart-addition.spec.ts -g "Successful Item Addition"
echo.
echo    Error handling:
echo    npx playwright test tests/cart-addition.spec.ts -g "Error Handling"
echo.
echo    Business logic:
echo    npx playwright test tests/cart-addition.spec.ts -g "Business Logic"
echo.
echo    Edge cases:
echo    npx playwright test tests/cart-addition.spec.ts -g "Edge Cases"
echo.

echo 3. Run with different modes:
echo    Headed mode (see browser):
echo    npx playwright test tests/cart-addition.spec.ts --headed
echo.
echo    Debug mode (with inspector):
echo    npx playwright test tests/cart-addition.spec.ts --debug
echo.
echo    UI mode (interactive):
echo    npm run test:ui
echo.

echo 4. Run on specific browser:
echo    Chrome/Chromium:
echo    npx playwright test tests/cart-addition.spec.ts --project=chromium
echo.
echo    Firefox:
echo    npx playwright test tests/cart-addition.spec.ts --project=firefox
echo.
echo    Safari/WebKit:
echo    npx playwright test tests/cart-addition.spec.ts --project=webkit
echo.

echo 5. Run with different reporters:
echo    List reporter (default):
echo    npx playwright test tests/cart-addition.spec.ts
echo.
echo    Verbose reporter:
echo    npx playwright test tests/cart-addition.spec.ts --reporter=verbose
echo.
echo    JSON report:
echo    npx playwright test tests/cart-addition.spec.ts --reporter=json
echo.

echo 6. View test reports:
echo    HTML report:
echo    npm run test:report
echo.

echo 7. Other useful commands:
echo    Run all framework tests:
echo    npm test
echo.
echo    Type check:
echo    npm run type-check
echo.
echo    Lint:
echo    npm run lint
echo.
echo    Format:
echo    npm run format
echo.

echo ==========================================
echo ✅ For more details, see CART_ADDITION_TEST.md
echo.
