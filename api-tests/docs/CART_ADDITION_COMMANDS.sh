#!/usr/bin/env bash
# Cart Addition Test - Quick Commands
# Run this file or use individual commands below

echo "🧪 Cart Addition Test - Command Reference"
echo "=========================================="
echo ""

# Run all cart addition tests
echo "1. Run all cart addition tests:"
echo "   npx playwright test tests/cart-addition.spec.ts"
echo ""

# Run specific test groups
echo "2. Run specific test groups:"
echo "   Successful additions:"
echo "   npx playwright test tests/cart-addition.spec.ts -g 'Successful Item Addition'"
echo ""
echo "   Error handling:"
echo "   npx playwright test tests/cart-addition.spec.ts -g 'Error Handling'"
echo ""
echo "   Business logic:"
echo "   npx playwright test tests/cart-addition.spec.ts -g 'Business Logic'"
echo ""
echo "   Edge cases:"
echo "   npx playwright test tests/cart-addition.spec.ts -g 'Edge Cases'"
echo ""

# Run with different modes
echo "3. Run with different modes:"
echo "   Headed mode (see browser):"
echo "   npx playwright test tests/cart-addition.spec.ts --headed"
echo ""
echo "   Debug mode (with inspector):"
echo "   npx playwright test tests/cart-addition.spec.ts --debug"
echo ""
echo "   UI mode (interactive):"
echo "   npm run test:ui"
echo ""

# Run on specific browser
echo "4. Run on specific browser:"
echo "   Chrome/Chromium:"
echo "   npx playwright test tests/cart-addition.spec.ts --project=chromium"
echo ""
echo "   Firefox:"
echo "   npx playwright test tests/cart-addition.spec.ts --project=firefox"
echo ""
echo "   Safari/WebKit:"
echo "   npx playwright test tests/cart-addition.spec.ts --project=webkit"
echo ""

# Run with different reporters
echo "5. Run with different reporters:"
echo "   List reporter (default):"
echo "   npx playwright test tests/cart-addition.spec.ts"
echo ""
echo "   Verbose reporter:"
echo "   npx playwright test tests/cart-addition.spec.ts --reporter=verbose"
echo ""
echo "   JSON report:"
echo "   npx playwright test tests/cart-addition.spec.ts --reporter=json"
echo ""

# View reports
echo "6. View test reports:"
echo "   HTML report:"
echo "   npm run test:report"
echo ""

# Other useful commands
echo "7. Other useful commands:"
echo "   Run all framework tests:"
echo "   npm test"
echo ""
echo "   Type check:"
echo "   npm run type-check"
echo ""
echo "   Lint:"
echo "   npm run lint"
echo ""
echo "   Format:"
echo "   npm run format"
echo ""

echo "=========================================="
echo "✅ For more details, see CART_ADDITION_TEST.md"
