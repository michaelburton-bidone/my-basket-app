# My-Basket-App UI Testing Framework

A comprehensive Playwright-based UI testing framework for the My-Basket-App e-commerce application. This framework provides automated testing for the shopping cart functionality with a focus on maintainability, scalability, and best practices.

## Features

- ✅ Page Object Model (POM) architecture for maintainable tests
- ✅ TypeScript for type-safe test development
- ✅ Custom Playwright fixtures for dependency injection
- ✅ Environment-specific configuration (dev, staging, prod)
- ✅ Comprehensive utilities for common testing operations
- ✅ HTML, JSON, and JUnit reporting
- ✅ Screenshot and video capture on failures
- ✅ Parallel test execution
- ✅ Auto-retry on failure
- ✅ Trace recording for debugging

## Installation

### Prerequisites

- Node.js 18+ and npm/yarn
- My-Basket-App running locally at `http://localhost:9002` (default)

### Setup

1. Install dependencies:

```bash
npm install
```

2. Install Playwright browsers:

```bash
npx playwright install
```

## Running Tests

### Run all tests

```bash
npm test
```

### Run tests in UI mode (interactive)

```bash
npm run test:ui
```

### Run tests in headed mode (see browser)

```bash
npm run test:headed
```

### Run specific test suite

```bash
npm run test:cart        # Run cart-related tests
npm run test:smoke       # Run smoke tests
```

### Run with debugging

```bash
npm run test:debug
```

### Generate code from user actions

```bash
npm run codegen
```

### View test report

```bash
npm run report
```

## Project Structure

```
ui-tests/
├── src/
│   ├── pages/              # Page Object Models
│   │   ├── BasePage.ts     # Base class for all pages
│   │   ├── ProductPage.ts  # Product listing page
│   │   ├── CartPage.ts     # Shopping cart page
│   │   └── CheckoutPage.ts # Checkout page
│   ├── utils/              # Utility functions
│   │   ├── waiters.ts      # Wait helpers
│   │   ├── assertions.ts   # Custom assertions
│   │   └── helpers.ts      # General helpers
│   ├── types/              # TypeScript interfaces
│   │   ├── page.types.ts   # Page object types
│   │   └── test.types.ts   # Test data types
│   └── fixtures/           # Playwright fixtures
│       └── page-fixtures.ts # Page object fixtures
├── tests/                  # Test specifications
│   ├── cart-ui.spec.ts     # Cart UI validation
│   ├── cart-crud.spec.ts   # Cart operations
│   └── checkout.spec.ts    # Checkout flow
├── config/                 # Configuration
│   └── environments.ts     # Environment URLs & credentials
├── playwright.config.ts    # Playwright configuration
├── tsconfig.json           # TypeScript configuration
├── package.json            # Dependencies
└── README.md               # This file
```

## Configuration

### Environment URLs

Edit `config/environments.ts` to configure URLs for different environments:

```typescript
export const config = {
  dev: { baseUrl: 'http://localhost:9002' },
  staging: { baseUrl: 'https://staging.basket.app' },
  prod: { baseUrl: 'https://basket.app' },
};
```

Set the environment with:

```bash
BASE_URL=http://localhost:9002 npm test
```

## Writing Tests

### Basic Test Example

```typescript
import { test } from '@playwright/test';
import { fixtures } from '@fixtures/page-fixtures';

test.use(fixtures);

test('should add product to cart', async ({ page, productPage, cartPage }) => {
  // Navigate and interact with pages
  await productPage.navigateTo();
  await productPage.addProductToCart(1);
  
  // Assert using page methods
  await cartPage.navigateTo();
  await cartPage.assertProductInCart('Product Name');
});
```

### Page Object Model Example

```typescript
import { BasePage } from './BasePage';

export class ProductPage extends BasePage {
  readonly addToCartButton = this.page.getByRole('button', { name: /add to cart/i });

  async addProductToCart(productId: number) {
    await this.addToCartButton.click();
  }
}
```

## Best Practices

1. **Use Locators Strictly**
   - Prioritize `getByRole` and `getByTestId`
   - Avoid brittle CSS/XPath selectors

2. **Follow DRY Principle**
   - Reuse methods in BasePage and utilities
   - Create helper functions for common operations

3. **Explicit Waits**
   - Use Playwright's auto-waiting capabilities
   - Add explicit waits for complex interactions

4. **Type Safety**
   - Define TypeScript interfaces for all data structures
   - Use strict TypeScript settings

5. **Test Organization**
   - Group related tests with `describe` blocks
   - Use meaningful test names
   - Add `@tags` for test categorization

6. **Error Handling**
   - Include timeout configurations
   - Capture screenshots on failures
   - Use meaningful assertion messages

## Debugging

### Visual Debugging

```bash
npm run test:debug
```

### Trace Viewer

Traces are saved automatically on failures. View them:

```bash
npx playwright show-trace test-results/trace.zip
```

### Screenshots and Videos

Found in `test-results/` folder on failure.

## CI/CD Integration

The framework automatically detects CI environments and:

- Enables retry (2 attempts)
- Sets workers to 1 (sequential execution)
- Records traces on first retry
- Captures screenshots on failure
- Generates JUnit XML reports

## Troubleshooting

### Tests timeout at localhost:9002

Ensure the dev server is running:

```bash
cd .. && npm run dev
```

### Flaky tests

- Increase timeout in `playwright.config.ts`
- Use explicit waits instead of fixed delays
- Check for race conditions in test logic

### Port conflicts

Change the base URL:

```bash
BASE_URL=http://localhost:3000 npm test
```

## Additional Resources

- [Playwright Documentation](https://playwright.dev)
- [Page Object Model Guide](https://playwright.dev/docs/pom)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [Debugging Guide](https://playwright.dev/docs/debug)

## Contributing

1. Follow existing code style and patterns
2. Write tests for new features
3. Ensure all tests pass before submitting
4. Update documentation as needed

## License

MIT
