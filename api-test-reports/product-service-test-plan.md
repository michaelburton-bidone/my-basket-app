# Product Service API Test Plan - Discount Feature

## Service Information
- **Service Name**: Product Service
- **Base URL**: http://localhost:3001
- **Schema File**: microservices/product-service/src/swagger.ts
- **Testing Scope**: Discount feature implementation

## Test Scenarios

### 1. GET /products - List Products with Discount Information
**Purpose**: Verify products are returned with discount data
**Expected Status**: 200
**Expected Response**: Array of products with discount objects
- Product 1 (Organic Apples): 10% discount
- Product 2 (Whole Wheat Bread): 15% discount
- Product 4 (Organic Spinach): 20% discount
- Product 8 (Greek Yogurt): 5% discount

### 2. GET /products/:id - Get Single Product with Discount
**Purpose**: Verify individual product retrieves discount information
**Test Cases**:
- Get product ID 1 (with discount)
- Get product ID 3 (without discount)
**Expected Status**: 200
**Validation**: Discount object present/absent based on product

### 3. POST /products - Create Product with Discount
**Purpose**: Create a new product with discount information
**Request Body**:
```json
{
  "name": "Organic Bananas",
  "price": 2.99,
  "description": "Fresh organic bananas, rich in potassium",
  "image": "https://placehold.co/300x200.png",
  "dataAiHint": "bananas fruit",
  "category": "fruits",
  "inStock": true,
  "discount": {
    "percentage": 12,
    "endsAt": "2026-02-23T00:00:00Z"
  }
}
```
**Expected Status**: 201
**Validation**: Response includes created product with discount

### 4. POST /products - Create Product without Discount
**Purpose**: Create product without discount (optional field)
**Request Body**:
```json
{
  "name": "Cheddar Cheese",
  "price": 6.99,
  "description": "Sharp cheddar cheese block",
  "image": "https://placehold.co/300x200.png",
  "dataAiHint": "cheese dairy",
  "category": "dairy",
  "inStock": true
}
```
**Expected Status**: 201
**Validation**: Response includes product without discount field

### 5. PUT /products/:id - Update Product with Discount
**Purpose**: Update existing product to add/modify discount
**Product ID**: 3 (Free-Range Eggs - currently no discount)
**Request Body**:
```json
{
  "discount": {
    "percentage": 8,
    "endsAt": "2026-02-20T00:00:00Z"
  }
}
```
**Expected Status**: 200
**Validation**: Product updated with new discount

### 6. PUT /products/:id - Remove Discount from Product
**Purpose**: Update product to remove discount
**Product ID**: 1 (Organic Apples - has 10% discount)
**Request Body**:
```json
{
  "discount": null
}
```
**Expected Status**: 200
**Validation**: Discount field removed/null

### 7. Discount Percentage Validation - Invalid Percentage
**Purpose**: Verify validation rejects percentages outside 0-100
**Request Body**:
```json
{
  "name": "Test Product",
  "price": 4.99,
  "description": "Test",
  "image": "https://placehold.co/300x200.png",
  "dataAiHint": "test",
  "discount": {
    "percentage": 150,
    "endsAt": "2026-02-20T00:00:00Z"
  }
}
```
**Expected Status**: 400
**Validation**: Error message about invalid percentage

### 8. Product Filters - List with Discounts (Future Enhancement)
**Purpose**: Verify products can be filtered by discount availability
**Query Parameters**: discountOnly=true
**Expected Status**: 200
**Validation**: Only products with active discounts returned

### 9. Price Calculation with Discount (Informational)
**Purpose**: Document expected discounted prices
**Examples**:
- Organic Apples: $3.99 → $3.59 (10% off)
- Whole Wheat Bread: $4.49 → $3.82 (15% off)
- Organic Spinach: $2.99 → $2.39 (20% off)

## Test Categories

### Functional Tests
- ✅ Retrieve products with discount information
- ✅ Retrieve single product with discount
- ✅ Create product with discount
- ✅ Create product without discount
- ✅ Update product to add discount
- ✅ Update product to remove discount

### Edge Case Tests
- ✅ Invalid discount percentage (>100%)
- ✅ Invalid discount percentage (<0%)
- ✅ Invalid discount date format
- ✅ Discount date in the past

### Data Validation Tests
- ✅ Required fields validation
- ✅ Discount object structure validation
- ✅ Percentage range validation

## Success Criteria
- All happy path scenarios return expected status codes
- Response data includes complete discount information when present
- Discount fields are properly validated
- Products without discounts work correctly
- All endpoints maintain backward compatibility

## Notes
- Discount percentage is 0-100 (inclusive)
- Discount end date uses ISO 8601 format
- Discount is optional on product creation/update
- Sample products include realistic discount data
