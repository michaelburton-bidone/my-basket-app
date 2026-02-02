# Cart Addition Test Suite - cart-addition.spec.ts

## Overview

Complete Playwright API test suite for testing the **Cart Addition** functionality (`POST /api/cart/:userId/items` endpoint).

## Test Statistics

- **Total Tests**: 30+
- **Test File**: `tests/cart-addition.spec.ts`
- **Test Target**: `POST http://localhost:9002/api/cart/:userId/items`
- **Port**: 9002 (Cart Service)

## Test Organization

### 1. Successful Item Addition (6 tests)
Tests basic cart addition functionality with valid inputs:
- ✅ Add item with valid productId and quantity
- ✅ Verify 200 OK response
- ✅ Verify cart updates with new item
- ✅ Verify total price calculation
- ✅ Verify item appears in response
- ✅ Verify itemId is assigned

### 2. Response Structure Validation (3 tests)
Validates JSON response structure and field types:
- ✅ Cart response contains required fields
- ✅ Response fields have correct types
- ✅ Cart items have valid structure

### 3. Add Multiple Items (3 tests)
Tests adding multiple items and price accumulation:
- ✅ Add multiple different items
- ✅ Prices accumulate correctly
- ✅ Handle duplicate items (increase quantity)

### 4. Error Handling - Invalid Inputs (6 tests)
Tests error scenarios and validation:
- ✅ Missing productId → 400 Bad Request
- ✅ Missing quantity (should use default 1)
- ✅ Zero quantity → 400 Bad Request
- ✅ Negative quantity → 400 Bad Request
- ✅ Empty productId → 400 Bad Request
- ✅ Non-existent product → 404 Not Found
- ✅ Invalid userId → 400 Bad Request

### 5. Business Logic Validation (4 tests)
Verifies business logic correctness:
- ✅ Item quantity matches request
- ✅ Product details preserved in cart
- ✅ Item count accuracy
- ✅ Total price calculation with multiple quantities

### 6. Edge Cases (3 tests)
Tests boundary conditions:
- ✅ Handle large quantities (1000+)
- ✅ Handle decimal quantities
- ✅ Handle rapid consecutive additions

### 7. Response JSON Body Assertions (5 tests)
Comprehensive JSON response validation:
- ✅ Valid JSON response
- ✅ Required fields present
- ✅ Items array contains cart items
- ✅ Numeric fields have correct types
- ✅ Field values are valid

## Key Features

### Authentication
- Uses existing CartAPI POM (no authentication required)
- Direct API context for advanced testing

### Data
- Uses real products from `test-data.ts`
- Real user IDs for testing
- Valid and invalid request examples

### Error Handling
- Try-catch blocks for expected failures
- Status code validation
- Error message verification

### Assertions
- HTTP status codes
- JSON response structure
- Field types and values
- Business logic validation
- Price calculations

## Running the Tests

### Run all cart addition tests
```bash
npx playwright test tests/cart-addition.spec.ts
```

### Run specific test group
```bash
# Successful additions only
npx playwright test tests/cart-addition.spec.ts -g "Successful Item Addition"

# Error scenarios only
npx playwright test tests/cart-addition.spec.ts -g "Error Handling"

# Business logic validation
npx playwright test tests/cart-addition.spec.ts -g "Business Logic"
```

### Run with options
```bash
# Headed mode (see browser)
npx playwright test tests/cart-addition.spec.ts --headed

# Debug mode
npx playwright test tests/cart-addition.spec.ts --debug

# Verbose output
npx playwright test tests/cart-addition.spec.ts --reporter=verbose
```

## Example Test

```typescript
test('should add item to cart with valid productId and quantity', async ({
  cartAPI,
}) => {
  const userId = testUsers.user1.id;
  const item = validAddToCartRequests.singleItem;

  const response = await cartAPI.addItemToCart(userId, item);

  // Assertions
  expect(response.success).toBe(true);
  expect(response.data).toBeDefined();
  expect(response.data?.items).toBeDefined();
  expect(Array.isArray(response.data?.items)).toBe(true);
});
```

## Request Format

```typescript
POST http://localhost:9002/api/cart/:userId/items

Request Body:
{
  productId: "1",      // Required: string
  quantity: 2          // Optional: positive number (default: 1)
}
```

## Response Format (Success - 200)

```json
{
  "userId": "user-001",
  "items": [
    {
      "itemId": "item-123",
      "productId": "1",
      "name": "Organic Apples",
      "price": 3.99,
      "quantity": 2
    }
  ],
  "itemCount": 1,
  "totalPrice": 7.98
}
```

## Response Format (Error - 400)

```json
{
  "error": "Invalid request data",
  "details": [
    {
      "code": "too_small",
      "minimum": 1,
      "type": "string",
      "path": ["productId"],
      "message": "String must contain at least 1 character(s)"
    }
  ]
}
```

## Test Data Used

### Products
- Apples (ID: 1, $3.99)
- Whole Wheat Bread (ID: 2, $4.49)
- Free-Range Eggs (ID: 3, $5.99)
- Organic Spinach (ID: 4, $2.99)
- Almond Milk (ID: 5, $3.79)
- Chicken Breast (ID: 6, $9.99)
- Brown Rice (ID: 7, $2.49)
- Greek Yogurt (ID: 8, $4.99)

### Users
- user-001, user-002, user-003

## Validation Patterns

### Status Code Validation
```typescript
expect(response.status()).toBe(200);
expect(response.ok()).toBe(true);
```

### JSON Structure Validation
```typescript
const validation = ResponseValidator.validateResponseStructure(
  response.data,
  ['userId', 'items', 'itemCount', 'totalPrice']
);
expect(validation.isValid).toBe(true);
```

### Field Type Validation
```typescript
const typeValidation = ResponseValidator.validateFieldTypes(response.data, {
  userId: 'string',
  items: 'object',
  itemCount: 'number',
  totalPrice: 'number',
});
expect(typeValidation.isValid).toBe(true);
```

### Value Assertion
```typescript
expect(response.data?.totalPrice).toBe(7.98);
expect(response.data?.items?.length).toBeGreaterThan(0);
```

## Expected Outcomes

### Success Case
```
✅ Status: 200 OK
✅ Item added to cart
✅ Cart items array updated
✅ Total price recalculated
✅ Item has unique itemId
✅ All response fields present
```

### Error Cases
```
✅ Invalid productId: 400 Bad Request
✅ Zero/Negative quantity: 400 Bad Request
✅ Non-existent product: 404 Not Found
✅ Missing required fields: 400 Bad Request
✅ Invalid userId: 400 Bad Request
```

## Integration Points

- **CartAPI POM**: Uses existing `CartAPI` class
- **Test Data**: Uses `testProducts` and `testUsers`
- **Validation**: Uses `ResponseValidator` utility
- **Fixtures**: Uses `@fixtures/index` Playwright fixtures

## Dependencies

- @playwright/test
- CartAPI (Page Object Model)
- ResponseValidator (Utility)
- Test data fixtures

## Notes

- Tests use unique user IDs where needed to avoid state conflicts
- Port 9002 is the Cart Service port
- Quantity defaults to 1 if not provided
- Price is calculated as: quantity × product price
- All tests follow Arrange-Act-Assert pattern

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests fail | Ensure cart service is running on port 9002 |
| 404 Not Found | Verify product IDs match actual products |
| Flaky tests | Use unique userIds to avoid state issues |
| Type errors | Run `npm run type-check` |

## References

- [Playwright Documentation](https://playwright.dev)
- [Cart Service Routes](../../../microservices/cart-service/src/routes.ts)
- [Test Data](../src/fixtures/test-data.ts)
- [CartAPI POM](../src/pages/CartAPI.ts)
- [Response Validator](../src/utils/response-validator.ts)

---

**Created**: February 2026  
**Framework**: Playwright 1.40+  
**TypeScript**: 5.3+
