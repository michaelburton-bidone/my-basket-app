# My Basket App - Playwright API Testing Framework

A comprehensive Playwright TypeScript API testing framework using Page Object Model (POM) architecture for testing the My Basket App cart endpoints.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Test Organization](#test-organization)
- [Architecture](#architecture)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

### What This Is

This project provides a complete API testing framework for the My Basket App cart service using:
- **Playwright Test**: Modern testing framework with excellent API testing support
- **TypeScript**: Type-safe test development
- **Page Object Model**: Maintainable and scalable test architecture
- **Fixtures**: Reusable test data and API context

### What Gets Tested

- ✅ **Cart CRUD Operations**: Create, read, update, and delete cart items
- ✅ **Error Handling**: Invalid requests, edge cases, and error scenarios
- ✅ **Authentication**: Multiple authentication methods and validation
- ✅ **Integration Workflows**: Complete shopping workflows and business logic
- ✅ **Data Consistency**: Cross-request validation and state management

### Base URL

```
http://localhost:9002
```

### Tested API Endpoint

```
/api/cart/:userId/items
```

## 📁 Project Structure

```
api-tests/
├── src/
│   ├── pages/                    # Page Object Models
│   │   ├── BaseAPI.ts           # Base API client with common HTTP methods
│   │   ├── CartAPI.ts           # Cart-specific API operations
│   │   └── index.ts             # Exports
│   ├── utils/                    # Utility functions
│   │   ├── logger.ts            # Logging utility
│   │   ├── config.ts            # Configuration management
│   │   ├── auth.ts              # Authentication handler
│   │   ├── response-validator.ts # Response validation
│   │   ├── error-handler.ts     # Error handling and retry logic
│   │   └── index.ts             # Exports
│   ├── types/                    # TypeScript types and interfaces
│   │   ├── api.types.ts         # API response types
│   │   ├── config.types.ts      # Configuration types
│   │   └── index.ts             # Exports
│   └── fixtures/                 # Test fixtures and data
│       ├── test-data.ts         # Test data and mock responses
│       ├── fixtures.ts          # Playwright fixtures
│       └── index.ts             # Exports
├── tests/                        # Test files
│   ├── cart-crud.spec.ts        # CRUD operation tests
│   ├── cart-errors.spec.ts      # Error scenario tests
│   ├── cart-auth.spec.ts        # Authentication tests
│   └── cart-integration.spec.ts # Integration workflow tests
├── config/                       # Environment configuration
├── playwright.config.ts          # Playwright configuration
├── tsconfig.json                 # TypeScript configuration
├── package.json                  # Project dependencies
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## ✅ Prerequisites

- **Node.js**: Version 16+ (LTS recommended)
- **npm**: Version 7+
- **My Basket App**: Running on `http://localhost:9002`

### Verify Prerequisites

```bash
# Check Node version
node --version

# Check npm version
npm --version
```

## 🚀 Installation

### 1. Clone/Navigate to Project

```bash
cd api-tests
```

### 2. Install Dependencies

```bash
npm install
```

This will install:
- `@playwright/test` - Testing framework
- `typescript` - TypeScript compiler
- `dotenv` - Environment variable management
- `eslint` & `prettier` - Code quality tools

### 3. Create Environment File

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` with your configuration:

```env
BASE_URL=http://localhost:9002
ENVIRONMENT=dev
AUTH_TYPE=none
REQUEST_TIMEOUT=30000
API_TIMEOUT=10000
DEBUG=false
```

### 4. Verify Installation

```bash
npm run type-check
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
BASE_URL=http://localhost:9002
ENVIRONMENT=dev

# Authentication Configuration
AUTH_TYPE=none|bearer|api-key|basic

# For Bearer Token Authentication
BEARER_TOKEN=your_token_here

# For API Key Authentication
API_KEY=your_api_key_here
API_KEY_HEADER=X-API-Key

# For Basic Authentication
BASIC_AUTH_USERNAME=username
BASIC_AUTH_PASSWORD=password

# Timeouts (milliseconds)
REQUEST_TIMEOUT=30000
API_TIMEOUT=10000

# Debug and Logging
LOG_LEVEL=info|debug|warn|error
DEBUG=false
```

### Playwright Configuration

Edit `playwright.config.ts` to customize:

```typescript
// Change browser configurations
projects: [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  { name: 'webkit', use: { ...devices['Desktop Safari'] } },
]

// Adjust timeouts
use: {
  baseURL: 'http://localhost:9002',
  trace: 'on-first-retry',
}
```

## 📖 Usage

### Run All Tests

```bash
npm test
```

### Run Tests in Headed Mode

```bash
npm run test:headed
```

Shows browser interactions in real-time.

### Run Tests in Debug Mode

```bash
npm run test:debug
```

Opens Playwright Inspector for step-by-step debugging.

### Run Tests with UI

```bash
npm run test:ui
```

Interactive test runner with live UI.

### Run Specific Test Suite

```bash
# CRUD operations only
npx playwright test tests/cart-crud.spec.ts

# Error scenarios only
npx playwright test tests/cart-errors.spec.ts

# Authentication tests only
npx playwright test tests/cart-auth.spec.ts

# Integration tests only
npx playwright test tests/cart-integration.spec.ts
```

### Run Tests by Browser

```bash
# Chromium only
npm run test:chrome

# Firefox only
npm run test:firefox

# WebKit only
npm run test:webkit
```

### View Test Report

```bash
npm run test:report
```

Opens the HTML test report in your default browser.

## 🧪 Test Organization

### Test Suites

#### 1. **CRUD Operations** (`cart-crud.spec.ts`)

Tests Create, Read, Update, Delete operations:
- ✅ Add single/multiple items
- ✅ Retrieve cart items
- ✅ Update item quantities
- ✅ Remove items and clear cart
- ✅ Cart response validation

**Run**: `npx playwright test tests/cart-crud.spec.ts`

#### 2. **Error Scenarios** (`cart-errors.spec.ts`)

Tests error handling and edge cases:
- ✅ Invalid user IDs
- ✅ Invalid request data
- ✅ Invalid operations
- ✅ Concurrent operations
- ✅ Large data handling
- ✅ HTTP status code validation

**Run**: `npx playwright test tests/cart-errors.spec.ts`

#### 3. **Authentication** (`cart-auth.spec.ts`)

Tests authentication mechanisms:
- ✅ Auth configuration validation
- ✅ Auth headers generation
- ✅ Bearer token, API key, basic auth
- ✅ Current configuration loading
- ✅ Authorization checks

**Run**: `npx playwright test tests/cart-auth.spec.ts`

#### 4. **Integration Workflows** (`cart-integration.spec.ts`)

Tests complete business workflows:
- ✅ Full shopping flow
- ✅ Multi-user cart management
- ✅ Business logic validation
- ✅ Price calculations
- ✅ Cart state transitions
- ✅ Data consistency

**Run**: `npx playwright test tests/cart-integration.spec.ts`

## 🏗️ Architecture

### Page Object Model (POM)

Separates test logic from API interaction code:

```typescript
// In tests
const response = await cartAPI.addItemToCart(userId, item);

// In CartAPI (POM)
async addItemToCart(userId: string, item: AddToCartRequest) {
  return await this.post<CartItem>(`/api/cart/${userId}/items`, item);
}

// In BaseAPI (Base POM)
async post<T>(endpoint: string, data?: Record<string, any>) {
  // Handles HTTP request, auth, retries, error handling
}
```

**Benefits**:
- Maintainable: Changes to API are made in one place
- Reusable: APIs shared across tests
- Readable: Tests focus on business logic, not HTTP details

### Directory Organization

```
src/
├── pages/           # POM classes for API operations
├── utils/           # Shared utilities (auth, validation, errors)
├── types/           # TypeScript type definitions
└── fixtures/        # Test data and Playwright fixtures

tests/               # Test files organized by feature
```

### Request Flow

```
Test
  ↓
CartAPI (Page Object)
  ↓
BaseAPI.post/get/put/patch/delete()
  ↓
ErrorHandler.retry()
  ↓
APIRequestContext.fetch()
  ↓
HTTP Response
  ↓
Response Validation
  ↓
Test Assertion
```

## 🔧 Advanced Features

### Authentication Handler

```typescript
// Automatically generates correct auth headers
const authHeaders = AuthHandler.getAuthHeaders({
  type: 'bearer',
  token: 'your-token',
});

// Validates auth configuration
const validation = AuthHandler.validateAuthConfig(config);
```

### Response Validation

```typescript
// Validate status codes
ResponseValidator.validateStatus(200, [200, 201]);

// Validate response structure
ResponseValidator.validateResponseStructure(response, ['userId', 'items']);

// Validate field types
ResponseValidator.validateFieldTypes(response, {
  userId: 'string',
  items: 'object',
});

// Assert validation
ResponseValidator.assertValid(result, 'Custom message');
```

### Error Handling & Retry

```typescript
// Automatic retry with exponential backoff
await ErrorHandler.retry(
  async () => await cartAPI.getCartItems(userId),
  3,      // max retries
  1000    // initial delay ms
);

// Check if error is retryable
if (ErrorHandler.isRetryable(error)) {
  // Retry logic
}
```

### Configuration Management

```typescript
const config = configManager.getConfig();
const baseURL = configManager.getBaseURL();
const authConfig = configManager.getAuthConfig();
const timeout = configManager.getTimeout();
```

### Logging

```typescript
import { logger, LogLevel } from '@utils/index';

logger.debug('Debug message', data);
logger.info('Information', data);
logger.warn('Warning', data);
logger.error('Error message', error);

// Set log level
logger.setLevel(LogLevel.DEBUG);
```

## 📝 Writing New Tests

### Basic Test Template

```typescript
import { test, expect } from '@fixtures/index';
import { testUsers, testProducts } from '@fixtures/test-data';

test.describe('New Feature Tests', () => {
  test('should do something', async ({ cartAPI }) => {
    const userId = testUsers.user1.id;

    const response = await cartAPI.addItemToCart(userId, {
      productId: testProducts.apple.productId,
      quantity: 2,
    });

    expect(response.success).toBe(true);
    expect(response.data?.quantity).toBe(2);
  });
});
```

### Using Test Data

```typescript
import {
  testUsers,
  testProducts,
  validAddToCartRequests,
  invalidAddToCartRequests,
  mockCartResponses,
} from '@fixtures/test-data';
```

### Creating New Page Objects

```typescript
import { BaseAPI } from './BaseAPI';
import { APIRequestContext } from '@playwright/test';

export class ProductAPI extends BaseAPI {
  constructor(apiContext: APIRequestContext) {
    super(apiContext);
  }

  async getProducts() {
    return await this.get('/api/products');
  }

  async getProduct(productId: string) {
    return await this.get(`/api/products/${productId}`);
  }
}
```

## 🔍 Debugging

### Enable Debug Mode

```bash
npm run test:debug
```

### Add Debug Logs

```typescript
import { logger } from '@utils/index';

logger.debug('Request URL:', url);
logger.debug('Response:', response);
```

### View Trace Files

Test traces are saved in `test-results/` and can be viewed with:

```bash
npx playwright show-trace test-results/trace.zip
```

### Use Headed Mode

```bash
npm run test:headed
```

### Add Manual Breakpoints

```typescript
test('debug test', async ({ cartAPI }) => {
  const response = await cartAPI.getCartItems('user-1');
  
  // Execution pauses here - use console in browser/debugger
  debugger;
  
  expect(response.success).toBe(true);
});
```

## 🐛 Troubleshooting

### Connection Refused

**Problem**: `Error: connect ECONNREFUSED`

**Solution**: Ensure My Basket App is running:
```bash
npm run microservices:start:win
```

### Invalid Base URL

**Problem**: `Error: Invalid URL`

**Solution**: Check `.env` file:
```env
BASE_URL=http://localhost:9002
```

### Authentication Failures

**Problem**: `401 Unauthorized`

**Solution**: Verify auth configuration in `.env`:
```env
AUTH_TYPE=none  # or bearer/api-key/basic
BEARER_TOKEN=your_token  # if using bearer
```

### Timeout Errors

**Problem**: `Timeout waiting for request`

**Solution**: Increase timeout in `.env`:
```env
REQUEST_TIMEOUT=60000
API_TIMEOUT=20000
```

### Type Errors

**Problem**: `TS2307: Cannot find module`

**Solution**: Run type check:
```bash
npm run type-check
```

### Path Resolution Issues

**Problem**: Import paths not resolving

**Solution**: Verify `tsconfig.json` has correct paths:
```json
{
  "compilerOptions": {
    "paths": {
      "@pages/*": ["src/pages/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"],
      "@fixtures/*": ["src/fixtures/*"]
    }
  }
}
```

## 📊 Test Reports

### HTML Report

```bash
npm run test:report
```

Generates detailed HTML report in `playwright-report/`

### JSON Report

Located in `test-results/results.json`

### JUnit XML Report

Located in `test-results/junit.xml`

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: API Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install
      - run: npm test
```

## 📚 Additional Resources

- [Playwright Documentation](https://playwright.dev)
- [Playwright API Testing Guide](https://playwright.dev/docs/api-testing)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Page Object Model Pattern](https://www.selenium.dev/documentation/test_practices/encouraged/page_object_models/)

## 🤝 Contributing

1. Create feature branches for new tests
2. Follow existing naming conventions
3. Add tests to appropriate `describe` blocks
4. Update test data in `fixtures/test-data.ts`
5. Run `npm run lint` before committing

## 📄 License

MIT

## 👤 Support

For issues or questions, check:
- Existing test examples in `tests/`
- Type definitions in `src/types/`
- Test data in `src/fixtures/test-data.ts`

---

**Last Updated**: February 2026
**Framework**: Playwright 1.40+
**TypeScript**: 5.3+
