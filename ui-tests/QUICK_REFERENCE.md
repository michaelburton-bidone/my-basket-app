# UI Testing Framework - Quick Reference

## 🚀 Getting Started (2 minutes)

```bash
# 1. Install dependencies
npm install && npx playwright install

# 2. Run tests
npm test

# 3. View report
npm run report
```

## 📋 Common Commands

```bash
# Run all tests
npm test

# Interactive UI mode
npm run test:ui

# Run specific tag
npm run test:smoke    # @smoke tests
npm run test:cart     # @cart tests

# Debug mode
npm run test:debug

# Generate code from browser interactions
npm run codegen

# View HTML report
npm run report

# Run specific file
npx playwright test tests/cart-addition.spec.ts

# Run specific test
npx playwright test cart-addition.spec.ts -g "should add single product"
```

## 🎯 Page Objects

### ProductPage
```typescript
await productPage.navigateTo();
await productPage.addProductToCartByIndex(0);
await productPage.searchProduct('headphones');
const count = await productPage.getProductCount();
const prices = await productPage.getAllProductPrices();
```

### CartPage
```typescript
await cartPage.navigateTo();
const items = await cartPage.getAllCartItems();
const totals = await cartPage.getCartTotals();
await cartPage.updateQuantity(0, 3);
await cartPage.removeItem(0);
await cartPage.proceedToCheckout();
```

### CheckoutPage
```typescript
const checkoutData = generateCheckoutData();
await checkoutPage.fillCheckoutForm(checkoutData);
await checkoutPage.selectShippingMethod('express');
await checkoutPage.fillPaymentInfo(card, name, expiry, cvv);
await checkoutPage.placeOrder();
```

## 🧪 Writing Tests

### Basic Test Template
```typescript
import { test, expect } from '@fixtures/page-fixtures';

test('should do something', async ({ productPage, cartPage }) => {
  // Arrange
  await productPage.navigateTo();
  
  // Act
  await productPage.addProductToCartByIndex(0);
  
  // Assert
  await cartPage.navigateTo();
  const items = await cartPage.getAllCartItems();
  expect(items).toHaveLength(1);
});
```

### With Test Tags
```typescript
test('test name @smoke @cart', async ({ productPage }) => {
  // Test code
});

// Run: npm run test:smoke
// Run: npm run test:cart
```

## 📦 Utilities

### Waiters
```typescript
import { waitForElementVisible, waitForText, waitForElementCount } from '@utils/waiters';

await waitForElementVisible(locator);
await waitForText(locator, 'Expected text');
await waitForElementCount(locator, 5);
```

### Assertions
```typescript
import {
  assertElementVisible,
  assertElementEnabled,
  assertElementContainsText,
  assertUrlContains
} from '@utils/assertions';

await assertElementVisible(button);
await assertElementEnabled(submitBtn);
await assertElementContainsText(title, 'Cart');
await assertUrlContains(page, '/checkout');
```

### Helpers
```typescript
import {
  formatCurrency,
  parseCurrency,
  generateUniqueId,
  isValidEmail,
  retry
} from '@utils/helpers';

const price = formatCurrency(49.99);              // "$49.99"
const amount = parseCurrency('$49.99');           // 49.99
const id = generateUniqueId('user_');             // "user_1707234567890_abc123"
isValidEmail('test@example.com');                 // true
await retry(() => productPage.navigateTo(), 3);  // Retry 3 times
```

## 🔧 Configuration

### Switch Environment
```bash
TEST_ENV=dev npm test        # Development
TEST_ENV=staging npm test    # Staging
TEST_ENV=prod npm test       # Production
```

### Set Base URL
```bash
BASE_URL=http://localhost:3000 npm test
```

### Run Single Browser
```bash
npm run test:chromium
```

## 📊 Test Data

### Use Sample Products
```typescript
import { sampleProducts } from '@config/test-data';

const firstProduct = sampleProducts[0];  // { id: 1, name: 'Wireless Headphones', price: 79.99 }
```

### Generate Random Data
```typescript
import {
  generateTestUser,
  generateCheckoutData,
  generateRandomCheckoutData,
  getTestCreditCard
} from '@config/test-data';

const user = generateTestUser();
const checkoutData = generateCheckoutData();
const randomCheckout = generateRandomCheckoutData();
const card = getTestCreditCard('visa');  // { number: '4111...', name: 'John Doe', expiry: '12/25', cvv: '123' }
```

## 🐛 Debugging

### Debug Tests
```bash
npm run test:debug
```

### View Trace
```bash
npx playwright show-trace test-results/trace.zip
```

### See Browser (Headed Mode)
```bash
npm run test:headed
```

### Generate Code
```bash
npm run codegen
# Interact with browser, see code generated
```

### Run Single Test with Details
```bash
npx playwright test tests/cart-addition.spec.ts -g "should add single" --headed --debug
```

## 📝 BasePage Methods

```typescript
// Navigation
await page.navigateTo('/path');
await page.navigateToHome();

// Clicks & Input
await page.click(locator);
await page.fill(locator, 'text');
await page.typeText(locator, 'text');
await page.clearInput(locator);

// Getting Values
const text = await page.getText(locator);
const value = await page.getInputValue(locator);
const attr = await page.getAttribute(locator, 'href');

// State Checks
const visible = await page.isVisible(locator);
const enabled = await page.isEnabled(locator);
const checked = await page.isChecked(locator);

// Waits
await page.waitForElement(locator);
await page.waitForUrlChange();
await page.waitForNavigation();

// Other
await page.hover(locator);
await page.pressKey('Enter');
await page.takeScreenshot('name');
await page.reload();
```

## 🔍 Locator Best Practices

### Preferred: Role-based
```typescript
this.page.getByRole('button', { name: /add to cart/i })
this.page.getByRole('textbox', { name: /email/i })
this.page.getByRole('link', { name: /checkout/i })
```

### Preferred: Test ID
```typescript
this.page.getByTestId('add-to-cart-btn')
this.page.getByTestId('cart-item')
this.page.getByTestId('total-price')
```

### Preferred: Label (Forms)
```typescript
this.page.getByLabel(/first name/i)
this.page.getByLabel(/password/i)
```

### Avoid: CSS/XPath (Brittle)
```typescript
// ❌ Don't use
this.page.locator('.btn.btn-primary.add-btn')
this.page.locator('xpath=//button[@class="add"]')
```

## 📊 Test Structure

```
tests/
├── cart-addition.spec.ts      # Happy path workflows
├── cart-crud.spec.ts          # Create, read, update, delete
├── checkout.spec.ts           # Purchase process
├── form-validation.spec.ts    # Input validation
└── ui-validation.spec.ts      # Component states
```

### Test File Template
```typescript
/**
 * Test File Description
 * 
 * What this test file covers
 * @tag @cart
 * @tag @smoke
 */

import { test, expect } from '@fixtures/page-fixtures';
import { sampleProducts } from '@config/test-data';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ productPage }) => {
    // Setup before each test
    await productPage.navigateTo();
  });

  test('should do something specific', async ({ page, productPage, cartPage }) => {
    // Test implementation
  });

  test.afterEach(async ({ page }) => {
    // Cleanup after each test (optional)
  });
});
```

## 🎯 Common Test Patterns

### Pattern: Add to Cart and Verify
```typescript
test('should add product to cart', async ({ productPage, cartPage }) => {
  await productPage.navigateTo();
  await productPage.addProductToCartByIndex(0);
  await cartPage.navigateTo();
  
  const items = await cartPage.getAllCartItems();
  expect(items.length).toBeGreaterThan(0);
});
```

### Pattern: Form Submission
```typescript
test('should submit checkout form', async ({ cartPage, checkoutPage }) => {
  const data = generateCheckoutData();
  
  await cartPage.proceedToCheckout();
  await checkoutPage.fillCheckoutForm(data);
  await checkoutPage.continueToBilling();
});
```

### Pattern: State Verification
```typescript
test('should show disabled button when cart empty', async ({ cartPage }) => {
  await cartPage.navigateTo();
  
  const isEmpty = await cartPage.isCartEmpty();
  expect(isEmpty).toBe(true);
  
  const isEnabled = await cartPage.isCheckoutButtonEnabled();
  expect(isEnabled).toBe(false);
});
```

### Pattern: Error Handling
```typescript
test('should show error for invalid email', async ({ checkoutPage }) => {
  await checkoutPage.fill(checkoutPage.emailInput, 'invalid');
  
  const error = await checkoutPage.getFieldError('email');
  expect(error).toBeTruthy();
});
```

## 📈 Performance Tips

1. **Use page objects** - Reduces duplication
2. **Reuse fixtures** - Avoid recreating pages
3. **Parallel execution** - Run multiple tests
4. **Efficient selectors** - Use role/testid
5. **Smart waits** - Let Playwright auto-wait
6. **Batch setup** - Do common setup once

## 🔗 File Reference

```
Configuration
├── playwright.config.ts      # Main config
├── tsconfig.json             # TypeScript config
└── .eslintrc.json            # ESLint config

Source Code
├── src/pages/                # Page objects
├── src/utils/                # Utilities
├── src/types/                # Interfaces
└── src/fixtures/             # Fixtures

Configuration
├── config/environments.ts    # URLs by env
└── config/test-data.ts       # Test data

Tests
└── tests/*.spec.ts           # Test files

Documentation
├── README.md                 # Start here
├── GUIDE.md                  # Detailed guide
├── FRAMEWORK_SUMMARY.md      # Summary
├── FILE_TREE.md              # File structure
└── QUICK_REFERENCE.md        # This file
```

## ✅ Checklist Before Committing

- [ ] All tests pass locally
- [ ] No console errors
- [ ] Code follows ESLint rules
- [ ] New tests have proper tags
- [ ] Page objects used (no direct selectors)
- [ ] TypeScript types defined
- [ ] Comments added for complex logic
- [ ] Test data uses fixtures
- [ ] No hardcoded URLs

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests timeout | Increase timeout in config or check base URL |
| Flaky tests | Use explicit waits, check for race conditions |
| Port conflict | Change BASE_URL env var |
| Element not found | Verify test IDs exist in app, use codegen |
| Auth issues | Check storageState in global-setup.ts |
| CI failures | Run locally first, check env vars |

## 📚 Resources

- [Playwright Docs](https://playwright.dev)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [POM Guide](https://playwright.dev/docs/pom)
- [Debugging](https://playwright.dev/docs/debug)
- [API Reference](https://playwright.dev/docs/api/class-test)

---

**Last Updated**: February 3, 2026
**Framework Version**: 1.0.0
