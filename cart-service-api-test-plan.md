# Cart Service API Test Plan

**API Base URL:** `http://localhost:3002/api`  
**Service:** Cart Microservice  
**Last Updated:** February 23, 2026

---

## Table of Contents

1. [API Overview](#api-overview)
2. [Endpoints Summary](#endpoints-summary)
3. [Test Scenarios by Endpoint](#test-scenarios-by-endpoint)
4. [Error Handling Test Cases](#error-handling-test-cases)
5. [Security & Validation Tests](#security--validation-tests)
6. [Integration Test Scenarios](#integration-test-scenarios)
7. [Edge Case Coverage](#edge-case-coverage)

---

## API Overview

### Service Information
- **Microservice:** Cart Service
- **Framework:** Express.js + TypeScript
- **Database:** In-Memory Map (per service instance)
- **Validation:** Zod Schema Validation
- **Total Endpoints:** 6
- **Total Test Scenarios:** 50+

### Authentication
Currently no authentication required (future: JWT tokens)

### Response Format
All responses are JSON with appropriate HTTP status codes.

---

## Endpoints Summary

| Method | Endpoint | Description | Status Codes |
|--------|----------|-------------|--------------|
| GET | `/cart/{userId}` | Get user's cart | 200, 400, 500 |
| POST | `/cart/{userId}/items` | Add item to cart | 200, 400, 404, 500 |
| PUT | `/cart/{userId}/items/{productId}` | Update item quantity | 200, 400, 404, 500 |
| DELETE | `/cart/{userId}/items/{productId}` | Remove item from cart | 200, 500 |
| DELETE | `/cart/{userId}` | Clear entire cart | 200, 500 |
| GET | `/cart/{userId}/summary` | Get cart summary | 200, 400, 500 |
| GET | `/health` | Health check | 200 |

---

## Test Scenarios by Endpoint

### 1. GET /cart/{userId} - Retrieve Cart

#### 1.1 Happy Path - Get Existing Cart
**Scenario:** Retrieve cart for user with existing items
```
Request:
  GET /cart/user-123

Expected Response:
  Status: 200 OK
  Body: {
    id: "cart-uuid",
    userId: "user-123",
    items: [
      {
        id: "product-1",
        name: "Product Name",
        price: 19.99,
        quantity: 2,
        addedAt: "2026-02-23T10:30:00Z"
      }
    ],
    totalAmount: 39.98,
    totalItems: 2,
    createdAt: "2026-02-23T09:00:00Z",
    updatedAt: "2026-02-23T10:30:00Z"
  }
```

#### 1.2 Happy Path - Get Empty Cart
**Scenario:** Retrieve cart for new user (empty cart)
```
Request:
  GET /cart/user-new

Expected Response:
  Status: 200 OK
  Body: {
    id: "new-cart-uuid",
    userId: "user-new",
    items: [],
    totalAmount: 0,
    totalItems: 0,
    createdAt: "2026-02-23T10:45:00Z",
    updatedAt: "2026-02-23T10:45:00Z"
  }
```

#### 1.3 Validation Error - Missing User ID
**Scenario:** Request without userId parameter
```
Request:
  GET /cart/

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid user ID",
    details: [
      {
        code: "too_small",
        minimum: 1,
        type: "string",
        path: ["userId"],
        message: "String must contain at least 1 character(s)"
      }
    ]
  }
```

#### 1.4 Validation Error - Empty User ID
**Scenario:** Request with empty userId
```
Request:
  GET /cart/""

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid user ID",
    details: [...]
  }
```

#### 1.5 Validation Error - Invalid User ID Type
**Scenario:** Request with non-string userId
```
Request:
  GET /cart/123  (if system expects string)

Expected Response:
  Status: 200 OK (coerced to string)
  OR
  Status: 400 Bad Request
```

---

### 2. POST /cart/{userId}/items - Add Item to Cart

#### 2.1 Happy Path - Add New Item with Default Quantity
**Scenario:** Add product to empty cart
```
Request:
  POST /cart/user-123/items
  Content-Type: application/json
  Body: {
    "productId": "product-1"
  }

Expected Response:
  Status: 200 OK
  Body: {
    id: "cart-uuid",
    userId: "user-123",
    items: [
      {
        id: "product-1",
        name: "Laptop",
        price: 999.99,
        quantity: 1,
        addedAt: "2026-02-23T10:50:00Z"
      }
    ],
    totalAmount: 999.99,
    totalItems: 1,
    createdAt: "2026-02-23T10:50:00Z",
    updatedAt: "2026-02-23T10:50:00Z"
  }
```

#### 2.2 Happy Path - Add Item with Specific Quantity
**Scenario:** Add product with quantity > 1
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-2",
    "quantity": 5
  }

Expected Response:
  Status: 200 OK
  Body: {
    ...cart with item quantity: 5...
  }
```

#### 2.3 Happy Path - Increment Existing Item
**Scenario:** Add same product again (should increment)
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": 3
  }

Expected Response:
  Status: 200 OK
  Body: {
    items: [
      {
        id: "product-1",
        quantity: 4  // 1 + 3
      }
    ]
  }
```

#### 2.4 Validation Error - Missing productId
**Scenario:** Request without productId
```
Request:
  POST /cart/user-123/items
  Body: {
    "quantity": 1
  }

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid request data",
    details: [
      {
        code: "missing",
        path: ["productId"],
        message: "Required"
      }
    ]
  }
```

#### 2.5 Validation Error - Empty productId
**Scenario:** Request with empty string productId
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": ""
  }

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid request data"
  }
```

#### 2.6 Validation Error - Invalid Quantity (Zero)
**Scenario:** Add with quantity = 0
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": 0
  }

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid request data",
    details: [
      {
        code: "too_small",
        minimum: 0.0000001,
        type: "number",
        path: ["quantity"],
        message: "Number must be greater than 0"
      }
    ]
  }
```

#### 2.7 Validation Error - Invalid Quantity (Negative)
**Scenario:** Add with negative quantity
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": -5
  }

Expected Response:
  Status: 400 Bad Request
```

#### 2.8 Validation Error - Invalid Quantity Type
**Scenario:** Add with non-numeric quantity
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": "five"
  }

Expected Response:
  Status: 400 Bad Request
```

#### 2.9 Error - Product Not Found
**Scenario:** Add product that doesn't exist
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "non-existent-product"
  }

Expected Response:
  Status: 404 Not Found
  Body: {
    error: "Product not found"
  }
```

#### 2.10 Error - Invalid JSON
**Scenario:** Send malformed JSON
```
Request:
  POST /cart/user-123/items
  Content-Type: application/json
  Body: {
    "productId": "product-1"
    MISSING CLOSING BRACE

Expected Response:
  Status: 400 Bad Request
  Body: {
    error: "Invalid JSON"
  }
```

#### 2.11 Edge Case - Very Large Quantity
**Scenario:** Add with quantity = 999999
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": 999999
  }

Expected Response:
  Status: 200 OK
  Body: {
    items: [{quantity: 999999}]
  }
```

#### 2.12 Edge Case - Decimal Quantity
**Scenario:** Add with decimal quantity
```
Request:
  POST /cart/user-123/items
  Body: {
    "productId": "product-1",
    "quantity": 2.5
  }

Expected Response:
  Status: 200 OK (or 400 if integers only)
```

---

### 3. PUT /cart/{userId}/items/{productId} - Update Item Quantity

#### 3.1 Happy Path - Update Quantity Up
**Scenario:** Increase quantity of existing item
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": 5
  }

Expected Response:
  Status: 200 OK
  Body: {
    items: [{
      id: "product-1",
      quantity: 5
    }]
  }
```

#### 3.2 Happy Path - Update Quantity Down
**Scenario:** Decrease quantity of existing item
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": 1
  }

Expected Response:
  Status: 200 OK
```

#### 3.3 Happy Path - Update to Minimum (1)
**Scenario:** Set quantity to 1
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": 1
  }

Expected Response:
  Status: 200 OK
```

#### 3.4 Happy Path - Remove Item (Quantity 0)
**Scenario:** Update quantity to 0 should remove item
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": 0
  }

Expected Response:
  Status: 200 OK
  Body: {
    items: [] (product-1 removed)
  }
```

#### 3.5 Validation Error - Missing Quantity
**Scenario:** Update without quantity field
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {}

Expected Response:
  Status: 400 Bad Request
```

#### 3.6 Validation Error - Invalid Quantity Type
**Scenario:** Update with string quantity
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": "ten"
  }

Expected Response:
  Status: 400 Bad Request
```

#### 3.7 Validation Error - Negative Quantity
**Scenario:** Update with negative quantity
```
Request:
  PUT /cart/user-123/items/product-1
  Body: {
    "quantity": -1
  }

Expected Response:
  Status: 200 OK (removes item) OR 400 Bad Request
```

#### 3.8 Error - Item Not Found
**Scenario:** Update item that doesn't exist in cart
```
Request:
  PUT /cart/user-123/items/non-existent-product
  Body: {
    "quantity": 5
  }

Expected Response:
  Status: 404 Not Found
  Body: {
    error: "Item not found in cart"
  }
```

---

### 4. DELETE /cart/{userId}/items/{productId} - Remove Item

#### 4.1 Happy Path - Remove Existing Item
**Scenario:** Remove item from cart with multiple items
```
Request:
  DELETE /cart/user-123/items/product-1

Expected Response:
  Status: 200 OK
  Body: {
    items: [/* remaining items */]
  }
```

#### 4.2 Happy Path - Remove Last Item
**Scenario:** Remove only item in cart
```
Request:
  DELETE /cart/user-123/items/product-1

Expected Response:
  Status: 200 OK
  Body: {
    items: []
  }
```

#### 4.3 Edge Case - Remove Non-existent Item
**Scenario:** Delete item not in cart
```
Request:
  DELETE /cart/user-123/items/non-existent

Expected Response:
  Status: 200 OK (no error - idempotent)
  Body: {
    items: [/* unchanged */]
  }
```

#### 4.4 Edge Case - Remove from Empty Cart
**Scenario:** Delete from cart with no items
```
Request:
  DELETE /cart/user-123/items/product-1

Expected Response:
  Status: 200 OK
```

---

### 5. DELETE /cart/{userId} - Clear Cart

#### 5.1 Happy Path - Clear Cart with Items
**Scenario:** Clear cart containing multiple items
```
Request:
  DELETE /cart/user-123

Expected Response:
  Status: 200 OK
  Body: {
    id: "cart-uuid",
    userId: "user-123",
    items: [],
    totalAmount: 0,
    totalItems: 0,
    updatedAt: "2026-02-23T11:00:00Z"
  }
```

#### 5.2 Happy Path - Clear Empty Cart
**Scenario:** Clear cart that's already empty
```
Request:
  DELETE /cart/user-123

Expected Response:
  Status: 200 OK
  Body: {
    items: []
  }
```

#### 5.3 Edge Case - Clear New User Cart
**Scenario:** Clear cart for user with no prior cart
```
Request:
  DELETE /cart/user-new

Expected Response:
  Status: 200 OK
  Body: {
    items: []
  }
```

---

### 6. GET /cart/{userId}/summary - Cart Summary

#### 6.1 Happy Path - Get Summary with Items
**Scenario:** Get summary of cart with multiple items
```
Request:
  GET /cart/user-123/summary

Expected Response:
  Status: 200 OK
  Body: {
    totalItems: 5,
    totalAmount: 149.95,
    itemCount: 2
  }
```

#### 6.2 Happy Path - Get Summary Empty Cart
**Scenario:** Get summary of empty cart
```
Request:
  GET /cart/user-123/summary

Expected Response:
  Status: 200 OK
  Body: {
    totalItems: 0,
    totalAmount: 0,
    itemCount: 0
  }
```

#### 6.3 Validation Error - Invalid userId
**Scenario:** Get summary with empty userId
```
Request:
  GET /cart//summary

Expected Response:
  Status: 400 Bad Request
```

---

### 7. GET /health - Health Check

#### 7.1 Happy Path - Health Check
**Scenario:** Service health verification
```
Request:
  GET /health

Expected Response:
  Status: 200 OK
  Body: {
    status: "healthy",
    service: "cart-service",
    timestamp: "2026-02-23T11:05:00Z"
  }
```

---

## Error Handling Test Cases

### HTTP Status Code Validation

| Scenario | Expected Status | Test Case |
|----------|-----------------|-----------|
| Valid request | 200 | All happy paths |
| Invalid/missing parameters | 400 | Invalid userId, empty fields |
| Resource not found | 404 | Product not found, item not in cart |
| Malformed request | 400 | Invalid JSON, missing required fields |
| Server errors | 500 | Handle gracefully |

### Error Response Format
All error responses should follow format:
```json
{
  "error": "Error message",
  "details": [/* Optional validation errors */]
}
```

---

## Security & Validation Tests

### Input Validation

#### String Fields (userId, productId)
- ✅ Minimum length: 1 character required
- ✅ Empty strings rejected
- ✅ XSS payload handling (e.g., `<script>alert('xss')</script>`)
- ✅ SQL injection attempts (e.g., `'; DROP TABLE carts;--`)

#### Numeric Fields (quantity)
- ✅ Positive numbers only (> 0)
- ✅ Zero rejected for POST quantity
- ✅ Zero accepted for PUT quantity (removes item)
- ✅ Negative numbers rejected
- ✅ Type validation (number, not string)
- ✅ Decimal precision (if applicable)

### Request Body Validation

#### Missing Fields
- ✅ POST /items: `productId` required
- ✅ PUT /items: `quantity` required
- ✅ Extra fields ignored (forward compatibility)

#### Type Coercion
- ✅ String "5" → number 5 (if applicable)
- ✅ Number 123 → string "123" (if applicable)
- ✅ Boolean values rejected where invalid

---

## Integration Test Scenarios

### Scenario 1: Complete User Shopping Journey
```
1. GET /cart/user-new → Empty cart created
2. POST /items → Add laptop (qty: 1, price: $999.99)
3. POST /items → Add mouse (qty: 2, price: $29.99 each)
4. GET /cart → Verify 2 items, total: $1059.97
5. GET /summary → Verify itemCount: 2, totalItems: 3
6. PUT /items/{mouse} → Update qty to 1
7. DELETE /items/{laptop} → Remove laptop
8. GET /cart → Verify only mouse remains
9. DELETE /cart → Clear cart
10. GET /cart → Verify empty cart
```

### Scenario 2: Quantity Management Workflow
```
1. POST /items → Add product-1, qty: 5
2. PUT /items → Update to qty: 3
3. PUT /items → Update to qty: 0 (remove)
4. POST /items → Re-add product-1, qty: 2
5. GET /cart → Verify qty: 2 (not 3 or 5)
```

### Scenario 3: Concurrent Operations (if applicable)
```
1. User A: POST /items → Add product-1
2. User B: POST /items → Add product-1
3. User A: GET /cart → Verify only their items
4. User B: GET /cart → Verify only their items
```

### Scenario 4: Data Persistence & Consistency
```
1. POST /items → Add item
2. GET /cart → Verify item present
3. GET /summary → Verify totals match
4. DELETE /items → Remove item
5. GET /cart → Verify item removed
6. GET /summary → Verify totals updated
```

---

## Edge Case Coverage

### Boundary Testing

#### Quantity Boundaries
| Test Case | Input | Expected |
|-----------|-------|----------|
| Min quantity | 0 (DELETE operation) | Item removed |
| Max quantity | 999999 | Accepted |
| Decimal quantity | 2.5 | Accept or reject consistently |
| Very large number | 1e10 | Handle appropriately |

#### Price Precision
| Scenario | Input | Expected Calculation |
|----------|-------|----------------------|
| Single item | price: 10.99, qty: 3 | total: 32.97 |
| Multiple items | prices: 10.99, 20.50 | total: 31.49 |
| Decimal precision | Multiple decimals | Round to 2 places |

### Field Length Testing
- ✅ userId: Test with 1, 50, 100, 500 characters
- ✅ productId: Test with 1, 50, 100, 500 characters
- ✅ Verify field length limits enforced

### Unicode & Special Characters
- ✅ userId with emoji: `user-🛒`
- ✅ userId with spaces: `user name`
- ✅ userId with special chars: `user!@#$%`
- ✅ productId with hyphens: `product-12-34`

### Request Format Testing
- ✅ Content-Type: application/json (required)
- ✅ Missing Content-Type header
- ✅ Charset specification (UTF-8)
- ✅ Gzip compression

---

## Performance Test Scenarios (Optional)

### Load Testing
- Add 100+ items to single cart
- Execute 1000 requests in quick succession
- Verify response times remain < 100ms
- Monitor memory usage

### Stress Testing
- Create carts for 10,000 unique users
- Parallel requests (10+ concurrent users)
- Verify data isolation between users

---

## Test Execution Checklist

- [ ] All 50+ test cases documented
- [ ] Happy path scenarios: 15+ cases
- [ ] Error handling: 15+ cases
- [ ] Edge cases: 15+ cases
- [ ] Integration scenarios: 5+ cases
- [ ] Status code validation: Complete
- [ ] Error response format validation: Complete
- [ ] Input validation: String, numeric, type coercion
- [ ] Security: XSS, SQL injection, input sanitization
- [ ] Data consistency: Pre/post verification
- [ ] Performance: Response time < 100ms
- [ ] Concurrent users: No cross-contamination

---

## Tools & Frameworks for Testing

### Recommended Testing Tools
- **API Testing:** Postman, Thunder Client, REST Client (VS Code)
- **Automated Testing:** Playwright, Jest, Supertest
- **Load Testing:** Apache JMeter, K6, Artillery
- **API Monitoring:** Postman Collections, Newman

### Example Jest Test Structure
```typescript
describe('Cart Service API', () => {
  describe('GET /cart/{userId}', () => {
    test('should retrieve cart successfully', async () => {
      const response = await request(app)
        .get('/cart/user-123')
        .expect(200);
      
      expect(response.body).toHaveProperty('userId', 'user-123');
      expect(response.body).toHaveProperty('items');
    });

    test('should return 400 for invalid userId', async () => {
      await request(app)
        .get('/cart/')
        .expect(400);
    });
  });
});
```

---

## Summary

This comprehensive test plan covers:
- ✅ **6 API Endpoints** with complete CRUD operations
- ✅ **50+ Test Scenarios** covering happy paths, errors, and edge cases
- ✅ **Input Validation** for all parameters
- ✅ **Security Testing** for common vulnerabilities
- ✅ **Integration Workflows** for realistic user journeys
- ✅ **Error Response** format validation
- ✅ **Data Consistency** verification
- ✅ **Performance** baseline expectations

All test scenarios should be executed before deployment to production to ensure API reliability and stability.
