# My-Basket-App UI Testing Framework

## Overview

This is a comprehensive UI testing framework built with **Playwright** and **TypeScript** for testing the My-Basket-App e-commerce platform. The framework focuses on the shopping cart functionality and follows industry best practices.

## Quick Start

### 1. Install Dependencies

```bash
npm install
npx playwright install
```

### 2. Run Tests

```bash
# Run all tests
npm test

# Run with UI
npm run test:ui

# Run specific suite
npm run test:cart
```

### 3. View Reports

```bash
npm run report
```

## Framework Architecture

### File Structure

```
ui-tests/
├── config/                    # Configuration files
│   ├── environments.ts       # Environment URLs
│   └── test-data.ts          # Test data and fixtures
├── src/
│   ├── pages/                # Page Object Models
│   │   ├── BasePage.ts       # Base class with common methods
│   │   ├── ProductPage.ts    # Product listing page
│   │   ├── CartPage.ts       # Shopping cart page
│   │   └── CheckoutPage.ts   # Checkout page
│   ├── utils/                # Utilities
│   │   ├── waiters.ts        # Wait helpers
│   │   ├── assertions.ts     # Custom assertions
│   │   └── helpers.ts        # General utilities
│   ├── types/                # TypeScript interfaces
│   │   ├── page.types.ts     # Page-related types
│   │   └── test.types.ts     # Test data types
│   └── fixtures/             # Playwright fixtures
│       └── page-fixtures.ts  # Page object injection
├── tests/                    # Test specifications
│   ├── cart-addition.spec.ts      # Happy path tests
│   ├── cart-crud.spec.ts          # CRUD operations
│   ├── checkout.spec.ts           # Checkout flow
│   ├── form-validation.spec.ts    # Form validation
│   └── ui-validation.spec.ts      # UI validation
├── playwright.config.ts      # Playwright configuration
├── tsconfig.json             # TypeScript configuration
├── package.json              # Dependencies
└── README.md                 # Documentation
```

## Key Features

### 1. **Page Object Model (POM)**
Every page has a dedicated class with:
- **Locators**: Using strict selectors (getByRole, getByTestId)
- **Methods**: Encapsulating user interactions
- **State queries**: Checking element states

Example:
```typescript
// ProductPage methods
await productPage.addProductToCartByIndex(0);
await productPage.searchProduct('headphones');
```

### 2. **Custom Fixtures**
Tests receive pre-initialized page objects:
```typescript
test('should add product to cart', async ({ productPage, cartPage }) => {
  // Page objects are ready to use
});
```

### 3. **Utility Functions**
Reusable helpers for common operations:
```typescript
// Waiters
await waitForElementVisible(locator);

// Assertions
await assertElementContainsText(locator, 'Add to Cart');

// Helpers
const price = parseCurrency('$49.99'); // Returns 49.99
```

### 4. **Type Safety**
All data structures have TypeScript interfaces:
```typescript
interface Product {
  id: number;
  name: string;
  price: number;
}

interface Cart {
  items: CartItem[];
  subtotal: number;
  total: number;
}
```

### 5. **Environment Configuration**
Easy switching between environments:
```bash
TEST_ENV=dev npm test    # Local dev
TEST_ENV=staging npm test # Staging
TEST_ENV=prod npm test   # Production
```

### 6. **Test Data Management**
Sample data and utilities:
```typescript
const testUser = generateTestUser();
const checkoutData = generateCheckoutData();
const creditCard = getTestCreditCard('visa');
```

## Test Organization

### Test Tags for Running Specific Suites

```bash
npm run test:smoke    # @smoke tests
npm run test:cart     # @cart tests
npm run test:ui       # @ui tests
```

### Test Categories

1. **Happy Path Tests** (`cart-addition.spec.ts`)
   - Basic workflows
   - Successfully adding products to cart
   - Cart calculations

2. **CRUD Operations** (`cart-crud.spec.ts`)
   - Create (add items)
   - Read (verify cart contents)
   - Update (change quantities)
   - Delete (remove items)

3. **Checkout Flow** (`checkout.spec.ts`)
   - Complete end-to-end checkout
   - Shipping method selection
   - Payment processing
   - Order confirmation

4. **UI Validation** (`ui-validation.spec.ts`)
   - Component visibility
   - Button states
   - Loading indicators
   - Empty states

5. **Form Validation** (`form-validation.spec.ts`)
   - Required field validation
   - Format validation
   - Error messages
   - Field interactions

## Writing Tests

### Basic Test Example

```typescript
import { test, expect } from '@fixtures/page-fixtures';

test('should add product to cart', async ({ productPage, cartPage }) => {
  // Arrange
  await productPage.navigateTo();

  // Act
  await productPage.addProductToCartByIndex(0);
  await cartPage.navigateTo();

  // Assert
  const items = await cartPage.getAllCartItems();
  expect(items).toHaveLength(1);
});
```

### Best Practices

#### 1. Use Page Objects
```typescript
// ✅ Good - encapsulated
await productPage.addProductToCartByIndex(0);

// ❌ Avoid - brittle selectors
await page.click('.product-item:nth-child(1) .add-to-cart-btn');
```

#### 2. Strict Locators
```typescript
// ✅ Good - maintainable
readonly addToCartButton = this.page.getByRole('button', { name: /add to cart/i });

// ⚠️ Avoid - brittle
readonly addToCartButton = this.page.locator('button.btn.btn-primary.add-btn');
```

#### 3. Wait for Elements
```typescript
// ✅ Good - implicit wait
await this.click(button);

// ✅ Good - explicit wait if needed
await this.waitForElement(button, { state: 'visible' });
```

#### 4. Type-Safe Data
```typescript
// ✅ Good - typed interface
const checkoutData: CheckoutData = generateCheckoutData();

// ⚠️ Avoid - any type
const checkoutData: any = { firstName: 'John' };
```

## Configuration

### Environment URLs

Edit `config/environments.ts`:

```typescript
export const environments = {
  dev: {
    baseUrl: 'http://localhost:9002',
    apiUrl: 'http://localhost:3001/api',
  },
  staging: {
    baseUrl: 'https://staging.basket.app',
    apiUrl: 'https://staging.basket.app/api',
  },
};
```

### Playwright Configuration

Key settings in `playwright.config.ts`:

```typescript
// Parallel execution
fullyParallel: true

// Retries (CI only)
retries: isCI ? 2 : 0

// Trace recording
trace: 'on-first-retry'

// Screenshot on failure
screenshot: 'only-on-failure'

// Video on failure
video: 'retain-on-failure'
```

## Debugging

### Run in Debug Mode
```bash
npm run test:debug
```

### Use UI Mode
```bash
npm run test:ui
```

### Generate Code from Actions
```bash
npm run codegen
```

### View Trace
```bash
npx playwright show-trace test-results/trace.zip
```

## CI/CD Integration

The framework auto-detects CI environments and:
- Enables retries (2 attempts)
- Sets workers to 1 (sequential)
- Records traces on first retry
- Captures screenshots on failure
- Generates JUnit XML reports

Example GitHub Actions setup:
```yaml
- name: Install dependencies
  run: npm install && npx playwright install

- name: Run tests
  run: npm test
  env:
    TEST_ENV: staging

- name: Upload report
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: playwright-report/
```

## Reporting

### Reporters Configured

1. **HTML Report** - Visual test results
2. **JSON Report** - Machine-readable results
3. **JUnit Report** - CI/CD integration
4. **List Report** - Console output

View HTML report:
```bash
npm run report
```

## Extending the Framework

### Adding a New Page Object

```typescript
// src/pages/LoginPage.ts
import { BasePage } from './BasePage';

export class LoginPage extends BasePage {
  readonly usernameInput = this.page.getByLabel(/username/i);
  readonly passwordInput = this.page.getByLabel(/password/i);
  readonly loginButton = this.page.getByRole('button', { name: /login/i });

  async login(username: string, password: string): Promise<void> {
    await this.fill(this.usernameInput, username);
    await this.fill(this.passwordInput, password);
    await this.click(this.loginButton);
  }
}
```

### Adding a New Fixture

```typescript
// Update src/fixtures/page-fixtures.ts
import { LoginPage } from '@pages/LoginPage';

export const test = base.extend<PageFixtures>({
  loginPage: async ({ page }, use) => {
    const loginPage = new LoginPage(page);
    await use(loginPage);
  },
});
```

### Adding Test Data

```typescript
// Update config/test-data.ts
export const newTestData = {
  // Your test data
};
```

## Troubleshooting

### Tests timeout
- Increase timeout in `playwright.config.ts`
- Check if dev server is running
- Verify base URL is correct

### Flaky tests
- Use explicit waits instead of fixed delays
- Check for race conditions
- Increase retry attempts in CI

### Port conflicts
```bash
BASE_URL=http://localhost:3001 npm test
```

### Element not found errors
- Check test IDs in HTML: `[data-testid="..."]`
- Verify selectors with `npm run codegen`
- Check element visibility

## Performance Tips

1. **Parallel execution** - Run multiple tests simultaneously
2. **Shared setup** - Use fixtures to initialize page objects
3. **Efficient waits** - Let Playwright auto-wait instead of fixed delays
4. **Minimal isolation** - Share browser context when possible

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [POM Guide](https://playwright.dev/docs/pom)
- [Debugging Guide](https://playwright.dev/docs/debug)
- [Configuration](https://playwright.dev/docs/test-configuration)

## Contributing

1. Follow existing code style
2. Use TypeScript for type safety
3. Add tests for new features
4. Update documentation
5. Ensure all tests pass

## License

MIT
