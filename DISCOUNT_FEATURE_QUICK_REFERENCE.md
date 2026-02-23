# Discount Feature - Quick Reference Guide

## What Was Added?

The Product Service now supports **discount information** on products. Each product can have an optional discount with a percentage and expiry date.

## Quick Start

### Running Tests
```bash
# Run all E2E tests
npx playwright test

# Run product service tests only
npx playwright test e2e-tests/api/product/product-service.spec.ts

# Run with HTML report
npx playwright test && npx playwright show-report
```

### API Examples

#### Get Products with Discounts
```bash
curl http://localhost:3001/api/products
```

**Response includes**:
```json
{
  "products": [
    {
      "id": "1",
      "name": "Organic Apples",
      "price": 3.99,
      "discount": {
        "percentage": 10,
        "endsAt": "2026-02-16T00:00:00Z"
      }
      // ... other fields
    }
  ]
}
```

#### Create Product with Discount
```bash
curl -X POST http://localhost:3001/api/products \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Organic Bananas",
    "price": 2.99,
    "description": "Fresh bananas",
    "image": "https://example.com/banana.jpg",
    "dataAiHint": "bananas fruit",
    "category": "fruits",
    "inStock": true,
    "discount": {
      "percentage": 12,
      "endsAt": "2026-02-23T00:00:00Z"
    }
  }'
```

#### Update Product with Discount
```bash
curl -X PUT http://localhost:3001/api/products/1 \
  -H "Content-Type: application/json" \
  -d '{
    "discount": {
      "percentage": 25,
      "endsAt": "2026-02-20T00:00:00Z"
    }
  }'
```

## Discount Specification

### Discount Object
```typescript
discount?: {
  percentage: number;  // 0-100 (inclusive)
  endsAt: Date;        // ISO 8601 format
}
```

### Rules
- **Percentage**: Must be between 0 and 100
- **End Date**: Must be valid ISO 8601 datetime
- **Optional**: Discount can be omitted when creating/updating products
- **Validation**: Zod schema validates all discount data

## Test Coverage

**24 E2E Test Cases** covering:
- ✅ List products with discounts
- ✅ Get single product (with/without discount)
- ✅ Create product (with/without discount)
- ✅ Update product (add/modify/remove discount)
- ✅ Delete product
- ✅ Error handling (invalid percentage, missing fields)
- ✅ Filtering and pagination
- ✅ Categories and health checks

**Test File**: `e2e-tests/api/product/product-service.spec.ts`

## Implementation Details

### Files Modified
1. `microservices/product-service/src/types.ts` - Added discount interface
2. `microservices/product-service/src/data.ts` - Added sample discount data
3. `microservices/product-service/src/routes.ts` - Added discount validation
4. `microservices/product-service/src/swagger.ts` - Updated API docs

### Files Created
1. `e2e-tests/api/product/product-service.spec.ts` - 24 E2E tests
2. `api-test-reports/product-service-test-plan.md` - Test plan
3. `playwright.config.ts` - Playwright configuration

## Sample Discounts in Data

- **Product 1** (Organic Apples): 10% off, expires in 7 days
- **Product 2** (Whole Wheat Bread): 15% off, expires in 14 days
- **Product 4** (Organic Spinach): 20% off, expires in 3 days
- **Product 8** (Greek Yogurt): 5% off, expires in 10 days

## Price Calculation

**Discounted Price** = Original Price × (1 - Percentage / 100)

Examples:
- Apples: $3.99 × (1 - 10/100) = $3.59
- Bread: $4.49 × (1 - 15/100) = $3.82
- Spinach: $2.99 × (1 - 20/100) = $2.39

## Error Responses

### Invalid Discount Percentage
```json
{
  "error": "Invalid product data",
  "details": [
    {
      "path": ["discount", "percentage"],
      "message": "Number must be less than or equal to 100"
    }
  ]
}
```

### Missing Required Fields
```json
{
  "error": "Invalid product data",
  "details": [
    {
      "path": ["name"],
      "message": "String must contain at least 1 character(s)"
    }
  ]
}
```

## API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/products` | List all products |
| GET | `/api/products/:id` | Get single product |
| POST | `/api/products` | Create product |
| PUT | `/api/products/:id` | Update product |
| DELETE | `/api/products/:id` | Delete product |
| GET | `/api/categories` | List categories |
| GET | `/api/health` | Health check |

## Development Tips

### Local Testing
```bash
# Start the service
npm run dev

# In another terminal, run tests
cd api-tests
npm install
npm test

# Or from root
npx playwright test
```

### Debugging Tests
```bash
# Run tests in debug mode
npx playwright test --debug

# Run in headed mode (see browser)
npx playwright test --headed

# Run specific test
npx playwright test -g "Create product with discount"
```

### Check API Documentation
```
http://localhost:3001/api-docs
```

## Integration Checklist

- [ ] Verify discount field appears in API responses
- [ ] Test creating products with discounts
- [ ] Test updating product discounts
- [ ] Verify validation of discount percentage
- [ ] Check discount dates are valid
- [ ] Confirm backward compatibility (products without discount)
- [ ] Run full E2E test suite
- [ ] Review test report
- [ ] Update frontend to display discounts
- [ ] Update cart service for discount calculations

## Common Issues & Solutions

### Issue: Tests fail with "Address already in use"
**Solution**: Stop existing service on port 3001
```bash
# Find and kill process on port 3001
netstat -ano | findstr :3001
taskkill /PID <PID> /F
```

### Issue: Invalid percentage error on valid input
**Solution**: Ensure percentage is a number, not string
```json
// ✅ Correct
{"percentage": 15}

// ❌ Wrong
{"percentage": "15"}
```

### Issue: Date validation errors
**Solution**: Use ISO 8601 format with 'Z' suffix
```json
// ✅ Correct
{"endsAt": "2026-02-23T00:00:00Z"}

// ❌ Wrong
{"endsAt": "02/23/2026"}
```

## Additional Resources

- **Full Implementation Guide**: `DISCOUNT_FEATURE_IMPLEMENTATION.md`
- **Test Plan**: `api-test-reports/product-service-test-plan.md`
- **Test File**: `e2e-tests/api/product/product-service.spec.ts`
- **API Docs**: `http://localhost:3001/api-docs` (when running)

---

**Status**: ✅ Complete and Ready to Use
**Last Updated**: 2026-02-09
