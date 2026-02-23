# API Testing Framework - Quick Reference

## 🚀 Quick Start

```bash
# 1. Install
cd api-tests && npm install

# 2. Configure
cp .env.example .env
# Edit .env with your settings

# 3. Run tests
npm test

# 4. View report
npm run test:report
```

---

## 📋 File Structure Quick Reference

```
api-tests/
├── src/pages/
│   ├── BaseAPI.ts      → Base HTTP client (GET, POST, PUT, DELETE)
│   └── CartAPI.ts      → Cart-specific API methods
├── src/utils/
│   ├── logger.ts       → Logging utility
│   ├── config.ts       → Configuration management
│   ├── auth.ts         → Authentication handler
│   ├── response-validator.ts → Response validation
│   └── error-handler.ts → Error handling & retry
├── src/types/
│   ├── api.types.ts    → API response types
│   └── config.types.ts → Configuration types
├── src/fixtures/
│   ├── test-data.ts    → Test data & mocks
│   └── fixtures.ts     → Playwright fixtures
├── tests/
│   ├── cart-crud.spec.ts       → Create/Read/Update/Delete
│   ├── cart-errors.spec.ts     → Error scenarios
│   ├── cart-auth.spec.ts       → Authentication
│   └── cart-integration.spec.ts → Workflows
├── config/
│   └── environments.ts → Environment configs
├── playwright.config.ts
├── tsconfig.json
├── package.json
├── README.md → Full documentation
├── SETUP.md → Setup guide
├── BEST_PRACTICES.md → Best practices
└── .env → Configuration (create from .env.example)
```

---

## 🧪 Test Commands

```bash
# Run all tests
npm test

# Run with browser visible
npm run test:headed

# Debug mode with inspector
npm run test:debug

# Interactive UI mode
npm run test:ui

# View HTML report
npm run test:report

# Run specific test file
npx playwright test tests/cart-crud.spec.ts

# Run tests matching pattern
npx playwright test -g "add.*item"

# Run on specific browser
npm run test:chrome      # Chromium
npm run test:firefox     # Firefox
npm run test:webkit      # WebKit

# Code quality
npm run lint             # Check code quality
npm run format           # Auto-format code
npm run type-check       # Check TypeScript
```

---

## 🔧 API Methods (CartAPI)

```typescript
// Get all items in cart
await cartAPI.getCartItems(userId)

// Add item to cart
await cartAPI.addItemToCart(userId, {
  productId: 'prod-123',
  quantity: 2,
  name: 'Product Name',
  price: 19.99
})

// Update item quantity
await cartAPI.updateCartItem(userId, itemId, newQuantity)

// Remove item from cart
await cartAPI.removeItemFromCart(userId, itemId)

// Clear entire cart
await cartAPI.clearCart(userId)

// Get cart summary
await cartAPI.getCartSummary(userId)

// Check if item exists
await cartAPI.itemExistsInCart(userId, productId)
```

---

## 📝 Writing a Test

```typescript
import { test, expect } from '@fixtures/index';
import { testUsers, testProducts } from '@fixtures/test-data';

test('should add item to cart', async ({ cartAPI }) => {
  const userId = testUsers.user1.id;

  const response = await cartAPI.addItemToCart(userId, {
    productId: testProducts.apple.productId,
    quantity: 2,
  });

  expect(response.success).toBe(true);
  expect(response.data?.quantity).toBe(2);
});
```

---

## 🔐 Authentication Configuration

```env
# No Authentication (default)
AUTH_TYPE=none

# Bearer Token
AUTH_TYPE=bearer
BEARER_TOKEN=your_token_here

# API Key
AUTH_TYPE=api-key
API_KEY=your_api_key
API_KEY_HEADER=X-API-Key

# Basic Auth
AUTH_TYPE=basic
BASIC_AUTH_USERNAME=username
BASIC_AUTH_PASSWORD=password
```

---

## 📊 Response Validation

```typescript
import { ResponseValidator } from '@utils/index';

// Validate status code
ResponseValidator.validateStatus(200, [200, 201]);

// Validate structure
ResponseValidator.validateResponseStructure(response, ['userId', 'items']);

// Validate types
ResponseValidator.validateFieldTypes(response, {
  userId: 'string',
  items: 'object',
});

// Assert validation
ResponseValidator.assertValid(result, 'Validation failed');
```

---

## 🚨 Error Handling

```typescript
import { ErrorHandler } from '@utils/index';

// Retry with exponential backoff
await ErrorHandler.retry(
  () => cartAPI.getCartItems(userId),
  3,    // max retries
  1000  // initial delay ms
);

// Check if error is retryable
if (ErrorHandler.isRetryable(error)) {
  // Retry logic
}

// Handle specific error types
try {
  await cartAPI.addItemToCart(userId, item);
} catch (error) {
  if (error instanceof ApiError) {
    console.log(`API Error: ${error.statusCode}`);
  }
}
```

---

## 📝 Logging

```typescript
import { logger, LogLevel } from '@utils/index';

logger.debug('Debug info', data);
logger.info('Information', data);
logger.warn('Warning message', data);
logger.error('Error occurred', error);

// Set log level
logger.setLevel(LogLevel.DEBUG);
```

---

## 🧩 Test Data

```typescript
import {
  testUsers,
  testProducts,
  validAddToCartRequests,
  invalidAddToCartRequests,
  mockCartResponses,
  errorMessages,
} from '@fixtures/test-data';

// Users
testUsers.user1.id      // 'user-001'
testUsers.user2.id      // 'user-002'

// Products
testProducts.apple.productId    // 'prod-001'
testProducts.banana.productId   // 'prod-002'
testProducts.milk.price         // 3.5

// Valid requests
validAddToCartRequests.singleItem
validAddToCartRequests.largeQuantity
validAddToCartRequests.minimalData

// Invalid requests
invalidAddToCartRequests.missingProductId
invalidAddToCartRequests.zeroQuantity
invalidAddToCartRequests.negativePrice

// Mock responses
mockCartResponses.emptyCart
mockCartResponses.cartWithOneItem
mockCartResponses.cartWithMultipleItems
```

---

## 🔍 Debugging

```bash
# Debug UI with Inspector
npm run test:debug

# Headed mode (see browser)
npm run test:headed

# View trace files
npx playwright show-trace test-results/trace.zip

# Check for type errors
npm run type-check

# View HTML report
npm run test:report
```

---

## ⚙️ Configuration

```env
# Base URL
BASE_URL=http://localhost:9002

# Environment
ENVIRONMENT=dev|staging|prod

# Timeouts (milliseconds)
REQUEST_TIMEOUT=30000
API_TIMEOUT=10000

# Debug
DEBUG=false
LOG_LEVEL=debug|info|warn|error
```

---

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| Connection refused | Start My Basket App: `npm run microservices:start:win` |
| Type errors | Run `npm run type-check` |
| Import errors | Verify `tsconfig.json` paths |
| Timeout | Increase `REQUEST_TIMEOUT` in `.env` |
| Auth failures | Check `.env` authentication settings |
| Module not found | Run `npm install` |

---

## 📚 Documentation Files

- **README.md** - Complete documentation
- **SETUP.md** - Detailed setup instructions
- **BEST_PRACTICES.md** - Testing best practices
- **Quick Reference** - This file

---

## 🔗 Key Exports

```typescript
// API Classes
import { BaseAPI, CartAPI } from '@pages/index';

// Utilities
import {
  logger,
  configManager,
  AuthHandler,
  ResponseValidator,
  ErrorHandler,
} from '@utils/index';

// Types
import {
  Cart,
  CartItem,
  CartResponse,
  AddToCartRequest,
  UpdateCartItemRequest,
} from '@types/index';

// Test Fixtures
import {
  test,
  expect,
  testUsers,
  testProducts,
  validAddToCartRequests,
} from '@fixtures/index';
```

---

## 🎯 Test Suites Overview

| Suite | Purpose | Command |
|-------|---------|---------|
| cart-crud.spec.ts | Create/Read/Update/Delete | `npx playwright test tests/cart-crud.spec.ts` |
| cart-errors.spec.ts | Error handling | `npx playwright test tests/cart-errors.spec.ts` |
| cart-auth.spec.ts | Authentication | `npx playwright test tests/cart-auth.spec.ts` |
| cart-integration.spec.ts | Workflows | `npx playwright test tests/cart-integration.spec.ts` |

---

## 💡 Tips

1. **Use fixtures** - Avoid creating new API context in tests
2. **Leverage test data** - Use predefined data instead of hardcoding
3. **Group assertions** - Related assertions should be together
4. **Descriptive names** - Test names should describe what they test
5. **Parallel execution** - Most tests run in parallel by default
6. **Error details** - Check logs when tests fail
7. **Type safety** - Let TypeScript catch errors before runtime
8. **DRY principle** - Use utilities and helpers to avoid duplication

---

## 🚀 Next Steps

1. Review [README.md](./README.md) for complete documentation
2. Check [SETUP.md](./SETUP.md) for setup details
3. Read [BEST_PRACTICES.md](./BEST_PRACTICES.md) for patterns
4. Run tests: `npm test`
5. View results: `npm run test:report`
6. Write custom tests
7. Integrate with CI/CD

---

**Framework**: Playwright 1.40+
**Language**: TypeScript 5.3+
**Node**: 16+ (LTS recommended)

For detailed information, see [README.md](./README.md)
