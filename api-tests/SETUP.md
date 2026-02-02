# API Testing Setup Guide

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
cd api-tests
npm install
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
BASE_URL=http://localhost:9002
AUTH_TYPE=none
```

### 3. Run Tests
```bash
npm test
```

---

## Detailed Setup Guide

### Step 1: Verify Prerequisites

```bash
# Node.js (required v16+)
node --version

# npm (required v7+)
npm --version
```

If missing, install from [nodejs.org](https://nodejs.org/)

### Step 2: Project Setup

```bash
# Navigate to api-tests directory
cd api-tests

# Install all dependencies
npm install

# This installs:
# - @playwright/test (testing framework)
# - typescript (type checking)
# - dotenv (environment variables)
# - eslint & prettier (code quality)
```

### Step 3: Environment Configuration

```bash
# Create .env file from template
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Required
BASE_URL=http://localhost:9002
ENVIRONMENT=dev

# Authentication (leave as 'none' for no auth)
AUTH_TYPE=none

# Optional - uncomment for Bearer Token Auth
# AUTH_TYPE=bearer
# BEARER_TOKEN=your_token_here

# Optional - uncomment for API Key Auth
# AUTH_TYPE=api-key
# API_KEY=your_api_key_here
# API_KEY_HEADER=X-API-Key

# Optional - uncomment for Basic Auth
# AUTH_TYPE=basic
# BASIC_AUTH_USERNAME=username
# BASIC_AUTH_PASSWORD=password

# Performance
REQUEST_TIMEOUT=30000
API_TIMEOUT=10000

# Debug (set to true to see detailed logs)
DEBUG=false
```

### Step 4: Verify Installation

```bash
# Type check (verify no TypeScript errors)
npm run type-check

# This should complete without errors
```

### Step 5: Start Your App

Before running tests, ensure the My Basket App is running:

```bash
# In root directory of my-basket-app
npm run microservices:start:win  # Windows
# OR
npm run microservices:start:unix # Linux/Mac
```

Verify it's running:
```bash
curl http://localhost:9002/api/cart/user-1/items
```

### Step 6: Run Tests

```bash
# Run all tests
npm test

# Run with browser visible
npm run test:headed

# Run with debug UI
npm run test:ui

# Run specific test file
npx playwright test tests/cart-crud.spec.ts
```

---

## Configuration Options

### Authentication Types

#### No Authentication (Default)
```env
AUTH_TYPE=none
```

#### Bearer Token
```env
AUTH_TYPE=bearer
BEARER_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### API Key
```env
AUTH_TYPE=api-key
API_KEY=sk_test_51234567890
API_KEY_HEADER=X-API-Key
```

#### Basic Auth
```env
AUTH_TYPE=basic
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=secretpassword
```

### Environment Profiles

#### Development (Default)
```env
ENVIRONMENT=dev
BASE_URL=http://localhost:9002
```

#### Staging
```env
ENVIRONMENT=staging
BASE_URL=https://staging.basketapp.com
REQUEST_TIMEOUT=45000
```

#### Production
```env
ENVIRONMENT=prod
BASE_URL=https://api.basketapp.com
REQUEST_TIMEOUT=60000
DEBUG=false
```

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `npm test` | Run all tests |
| `npm run test:headed` | Run with visible browser |
| `npm run test:debug` | Open debug UI |
| `npm run test:ui` | Interactive test runner |
| `npm run test:report` | View HTML report |
| `npm run test:chrome` | Run on Chromium only |
| `npm run lint` | Check code quality |
| `npm run format` | Auto-format code |
| `npm run build` | Compile TypeScript |
| `npm run type-check` | Check for type errors |

---

## Project Structure

```
api-tests/
├── src/
│   ├── pages/           # API Page Objects (CartAPI, etc.)
│   ├── utils/           # Utilities (auth, logging, validation)
│   ├── types/           # TypeScript types
│   └── fixtures/        # Test data and Playwright fixtures
├── tests/               # Test files
│   ├── cart-crud.spec.ts
│   ├── cart-errors.spec.ts
│   ├── cart-auth.spec.ts
│   └── cart-integration.spec.ts
├── config/              # Environment configs
├── playwright.config.ts # Playwright configuration
├── tsconfig.json        # TypeScript configuration
├── package.json         # Dependencies
└── README.md            # Full documentation
```

---

## Troubleshooting

### Tests Won't Run

**Error**: `Error: connect ECONNREFUSED`

**Solution**: Start the My Basket App:
```bash
npm run microservices:start:win
```

### Import Errors

**Error**: `TS2307: Cannot find module '@pages/CartAPI'`

**Solution**: Run type check and verify paths:
```bash
npm run type-check
```

### Timeout Issues

**Error**: `Timeout waiting for response`

**Solution**: Increase timeout in `.env`:
```env
REQUEST_TIMEOUT=60000
```

### Authentication Failures

**Error**: `401 Unauthorized`

**Solution**: Verify `.env` settings match your API requirements

---

## Running Specific Tests

### Test by Feature
```bash
# CRUD operations
npx playwright test tests/cart-crud.spec.ts

# Error handling
npx playwright test tests/cart-errors.spec.ts

# Authentication
npx playwright test tests/cart-auth.spec.ts

# Integration workflows
npx playwright test tests/cart-integration.spec.ts
```

### Test by Pattern
```bash
# Tests with "add" in the name
npx playwright test -g "add"

# Tests with "remove" in the name
npx playwright test -g "remove"
```

### Test by Browser
```bash
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

---

## Writing Your First Test

1. Create a new file: `tests/my-test.spec.ts`

```typescript
import { test, expect } from '@fixtures/index';
import { testUsers, testProducts } from '@fixtures/test-data';

test.describe('My First Test Suite', () => {
  test('should add item to cart', async ({ cartAPI }) => {
    const userId = testUsers.user1.id;
    
    const response = await cartAPI.addItemToCart(userId, {
      productId: testProducts.apple.productId,
      quantity: 2,
    });
    
    expect(response.success).toBe(true);
  });
});
```

2. Run it:
```bash
npx playwright test tests/my-test.spec.ts
```

---

## Next Steps

1. ✅ Install dependencies (`npm install`)
2. ✅ Create `.env` file
3. ✅ Start My Basket App
4. ✅ Run tests (`npm test`)
5. ✅ View report (`npm run test:report`)
6. ✅ Write custom tests
7. ✅ Integrate with CI/CD

---

## Need Help?

- **Type Errors**: Run `npm run type-check`
- **Linting**: Run `npm run lint`
- **Formatting**: Run `npm run format`
- **Documentation**: See `README.md`
- **Examples**: Check existing tests in `tests/`

---

For complete documentation, see [README.md](./README.md)
