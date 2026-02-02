# API Testing Best Practices

## Test Structure

### Test Organization

```typescript
test.describe('Feature Name', () => {
  test.describe('Specific Scenario', () => {
    test('should behave like X', async ({ cartAPI }) => {
      // Arrange
      const userId = testUsers.user1.id;
      
      // Act
      const response = await cartAPI.addItemToCart(userId, item);
      
      // Assert
      expect(response.success).toBe(true);
    });
  });
});
```

### Use Descriptive Test Names

✅ **Good**:
```typescript
test('should add item to cart and update total price')
test('should fail when adding item with negative quantity')
test('should maintain cart consistency after concurrent updates')
```

❌ **Bad**:
```typescript
test('test 1')
test('add item')
test('update')
```

---

## Fixture and Data Management

### Using Test Data

```typescript
import { testUsers, testProducts, validAddToCartRequests } from '@fixtures/test-data';

// Use predefined data
const userId = testUsers.user1.id;
const product = testProducts.apple;
const request = validAddToCartRequests.singleItem;
```

### Creating Test Data

```typescript
// In tests/cart-crud.spec.ts
const customItem = {
  productId: 'custom-prod-123',
  quantity: 5,
  name: 'Custom Product',
  price: 99.99,
};

const response = await cartAPI.addItemToCart(userId, customItem);
```

---

## Error Handling

### Handling Expected Errors

```typescript
test('should handle invalid user ID', async ({ cartAPI }) => {
  try {
    await cartAPI.getCartItems('invalid-id');
    // If no error, test fails
    expect.fail('Should have thrown error');
  } catch (error: any) {
    expect(error).toBeDefined();
    expect(error.statusCode).toBe(404);
  }
});
```

### Using Error Handling Utilities

```typescript
import { ErrorHandler } from '@utils/index';

// Retry logic
const response = await ErrorHandler.retry(
  () => cartAPI.getCartItems(userId),
  3,      // max retries
  1000    // delay ms
);

// Check if retryable
if (ErrorHandler.isRetryable(error)) {
  // Handle retry
}
```

---

## Assertions and Validation

### Response Validation

```typescript
import { ResponseValidator } from '@utils/index';

// Validate structure
const structureValidation = ResponseValidator.validateResponseStructure(
  response.data,
  ['userId', 'items', 'totalPrice']
);
expect(structureValidation.isValid).toBe(true);

// Validate types
const typeValidation = ResponseValidator.validateFieldTypes(response.data, {
  userId: 'string',
  items: 'object',
  totalPrice: 'number',
});
expect(typeValidation.isValid).toBe(true);

// Assert valid
ResponseValidator.assertValid(typeValidation, 'Response structure is invalid');
```

### Assertion Patterns

```typescript
// Check existence
expect(response.data).toBeDefined();

// Check types
expect(typeof response.status).toBe('number');

// Check values
expect(response.data?.quantity).toBe(5);

// Check ranges
expect(response.data?.totalPrice).toBeGreaterThan(0);

// Check arrays
expect(Array.isArray(response.data?.items)).toBe(true);
expect(response.data?.items).toHaveLength(2);

// Check properties
expect(response.data).toHaveProperty('userId');
```

---

## Debugging Tests

### Using Logging

```typescript
import { logger, LogLevel } from '@utils/index';

logger.setLevel(LogLevel.DEBUG);
logger.debug('Adding item:', item);
logger.info('Cart response:', response);
logger.warn('Unusual behavior detected');
logger.error('Operation failed', error);
```

### Debug Mode

```bash
# Open debug UI
npm run test:debug

# Or use headed mode with logs
npm run test:headed
```

### Adding Breakpoints

```typescript
test('debug test', async ({ cartAPI }) => {
  const response = await cartAPI.getCartItems('user-1');
  
  // Pause execution here
  debugger;
  
  expect(response.success).toBe(true);
});
```

---

## Performance Considerations

### Parallel Execution

```typescript
// Tests run in parallel by default
test.describe.parallel('Parallel Tests', () => {
  test('test 1', async ({ cartAPI }) => { /* ... */ });
  test('test 2', async ({ cartAPI }) => { /* ... */ });
  test('test 3', async ({ cartAPI }) => { /* ... */ });
});
```

### Sequential Execution (When Needed)

```typescript
// Force sequential execution
test.describe.serial('Sequential Tests', () => {
  test('test 1', async ({ cartAPI }) => { /* ... */ });
  test('test 2', async ({ cartAPI }) => { /* ... */ });
});
```

### Slow Test

Mark slow tests:
```typescript
test('slow operation', async ({ cartAPI }) => {
  test.slow(); // Multiplies timeout by 3
  // Long operation here
});
```

---

## Authentication Testing

### Test with Different Auth Types

```typescript
// No Auth
test('should work without auth', async ({ cartAPI }) => {
  const response = await cartAPI.getCartItems('user-1');
  expect(response.success).toBe(true);
});

// Bearer Token
test('should validate bearer token', () => {
  const validation = AuthHandler.validateAuthConfig({
    type: 'bearer',
    token: 'test-token',
  });
  expect(validation.valid).toBe(true);
});

// API Key
test('should generate API key header', () => {
  const headers = AuthHandler.getAuthHeaders({
    type: 'api-key',
    apiKey: 'test-key',
  });
  expect(headers['X-API-Key']).toBe('test-key');
});
```

---

## Test Data Isolation

### Use Unique Identifiers

```typescript
// Good - unique per test
const userId = `user-${Date.now()}-${Math.random()}`;

// Avoid - shared state
const userId = testUsers.user1.id; // (only if intentional)
```

### Clean Up After Tests

```typescript
test.afterEach(async ({ cartAPI }) => {
  // Clean up test data
  // Remove items, clear carts, etc.
});
```

---

## Common Patterns

### Test Setup and Cleanup

```typescript
test.beforeEach(async ({ cartAPI }) => {
  // Setup before each test
  // Clear cart, initialize data, etc.
});

test.afterEach(async ({ cartAPI }) => {
  // Cleanup after each test
  // Remove test data, reset state, etc.
});

test('test operation', async ({ cartAPI }) => {
  // Test code
});
```

### Multiple Assertions in One Test

```typescript
test('should validate cart item completely', async ({ cartAPI }) => {
  const userId = testUsers.user1.id;
  
  // Add item
  const addResponse = await cartAPI.addItemToCart(userId, {
    productId: 'test-prod',
    quantity: 2,
  });
  
  // Multiple assertions
  expect(addResponse.success).toBe(true);
  expect(addResponse.data?.productId).toBe('test-prod');
  expect(addResponse.data?.quantity).toBe(2);
  expect(addResponse.data?.itemId).toBeDefined();
  
  // Get cart
  const cartResponse = await cartAPI.getCartItems(userId);
  expect(cartResponse.data?.itemCount).toBeGreaterThan(0);
});
```

### Testing State Transitions

```typescript
test('should track state transitions', async ({ cartAPI }) => {
  const userId = testUsers.user1.id;
  
  // Initial state (empty)
  let cart = await cartAPI.getCartItems(userId);
  expect(cart.data?.itemCount).toBe(0);
  
  // Transition (add item)
  await cartAPI.addItemToCart(userId, { productId: 'prod-1', quantity: 1 });
  
  // New state (has items)
  cart = await cartAPI.getCartItems(userId);
  expect(cart.data?.itemCount).toBeGreaterThan(0);
  
  // Transition (remove item)
  const itemId = cart.data?.items?.[0]?.itemId;
  if (itemId) {
    await cartAPI.removeItemFromCart(userId, itemId);
  }
  
  // Back to initial state
  cart = await cartAPI.getCartItems(userId);
  expect(cart.data?.itemCount).toBe(0);
});
```

---

## Performance Tips

1. **Use test data fixtures** instead of creating new data each time
2. **Parallel execution** by default - only serialize when necessary
3. **Reuse API context** through fixtures instead of creating new ones
4. **Set appropriate timeouts** - don't make them too large
5. **Mock when possible** - use test data instead of real APIs
6. **Batch assertions** - group related assertions together

---

## Code Quality

### Run Linting

```bash
npm run lint
```

### Format Code

```bash
npm run format
```

### Type Checking

```bash
npm run type-check
```

---

## Continuous Integration

### Run Tests in CI

```bash
npm test -- --reporter=json --reporter=junit
```

### Generate Reports

```bash
# HTML report
npm run test:report

# JSON report
cat test-results/results.json

# JUnit XML
cat test-results/junit.xml
```

---

For more information, see [README.md](./README.md)
