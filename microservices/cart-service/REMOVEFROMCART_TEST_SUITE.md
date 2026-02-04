# removeFromCart Test Suite Documentation

## Overview
A comprehensive Jest test suite for the `removeFromCart` method in the CartService, validating state integrity and recalculation logic (totals/amounts) following item removal.

**Test File**: [src/service.test.ts](src/service.test.ts)  
**Method Under Test**: `CartService.removeFromCart(userId: string, productId: string)`

---

## Test Coverage Summary

### ✅ All 20 Tests Pass

**Test Execution**:
```
Test Suites: 1 passed, 1 total
Tests:       20 passed, 20 total
```

---

## Test Suite Structure

### 1️⃣ **State Integrity & Item Removal** (2 tests)

#### Test: "should completely remove the item from the items array"
- **Purpose**: Verify item is fully purged, not set to zero quantity
- **Setup**: Cart seeded with 2 items (Product 1: qty=2, Product 2: qty=1)
- **Action**: Remove Product 1
- **Assertions**:
  - Items array length decremented from 2 → 1
  - Removed product ID not in array
  - Remaining product ID preserved with all properties

#### Test: "should preserve remaining items when one is removed"
- **Purpose**: Ensure non-removed items maintain their state unchanged
- **Setup**: Cart with 2 items (Product 1: qty=2, Product 2: qty=3)
- **Action**: Remove Product 1
- **Assertions**:
  - Product 2 retains quantity=3
  - Product 2 price and name unchanged
  - Item count = 1

---

### 2️⃣ **Recalculation Logic - Total Items** (2 tests)

#### Test: "should update totalItems to reflect only remaining item quantities"
- **Purpose**: Validate `totalItems` reflects count of remaining items
- **Setup**: Cart with Item 1 (qty=5) and Item 2 (qty=3) → `totalItems=8`
- **Action**: Remove Item 1
- **Assertions**:
  - `totalItems` = 3 (only Item 2's quantity)
  - Verification: `totalItems = Σ(quantity)`

#### Test: "should set totalItems to 0 when last item is removed"
- **Purpose**: Edge case where cart becomes empty
- **Setup**: Single-item cart (qty=5)
- **Action**: Remove only item
- **Assertions**:
  - `totalItems = 0`

---

### 3️⃣ **Recalculation Logic - Total Amount** (4 tests)

#### Test: "should recalculate totalAmount correctly after removal"
- **Purpose**: Validate amount updates after item removal
- **Setup**: 
  - Item 1: $10.99 × 2 = $21.98
  - Item 2: $25.50 × 1 = $25.50
  - Pre-removal: `totalAmount = $47.48`
- **Action**: Remove Item 1
- **Assertions**:
  - Post-removal: `totalAmount = $25.50` (Item 2 only)

#### Test: "should apply formula: totalAmount = Σ(price × quantity) for remaining items"
- **Purpose**: Verify the mathematical formula with multi-item scenario
- **Setup**: 3-item cart with varying prices/quantities
  - Item 1: $10.99 × 1 = $10.99
  - Item 2: $25.50 × 2 = $51.00
  - Item 3: $99.99 × 1 = $99.99
  - Pre-removal: `totalAmount = $161.98`
- **Action**: Remove Item 2 ($51.00)
- **Assertions**:
  - Post-removal: `totalAmount = $110.98`
  - Formula verification: `totalAmount = Σ(price × quantity)` with `toBeCloseTo()` for floating-point safety

#### Test: "should set totalAmount to 0 when last item is removed"
- **Purpose**: Edge case - empty cart
- **Setup**: Cart with 3 units at $10.99 → `totalAmount = $32.97`
- **Action**: Remove the only item
- **Assertions**:
  - `totalAmount = 0`

#### Test: "should maintain 2 decimal precision after removal"
- **Purpose**: Ensure rounding consistency (banker's rounding: round to 2 decimals)
- **Setup**: 
  - Item 1: $10.99 × 3 = $32.97
  - Item 2: $25.50 × 2 = $51.00
  - Pre-removal: `totalAmount = $83.97`
- **Action**: Remove Item 1
- **Assertions**:
  - Post-removal: `totalAmount = $51.00`
  - Decimal places ≤ 2

---

### 4️⃣ **Array Structure & Length** (2 tests)

#### Test: "should decrement items array length by exactly one"
- **Purpose**: Verify precise array manipulation
- **Setup**: 2-item cart
- **Action**: Remove 1 item
- **Assertions**:
  - `items.length` = 1 (initial 2 → final 1)
  - Decrement exactly 1

#### Test: "should maintain item order and data integrity for remaining items"
- **Purpose**: Ensure no unintended data mutation or reordering
- **Setup**: 3-item cart [Item 1, Item 2, Item 3]
- **Action**: Remove Item 2 (middle item)
- **Assertions**:
  - Remaining IDs: [Item 1, Item 3]
  - Item 1 properties intact (name, price, quantity)
  - Item 3 properties intact (name, price, quantity)

---

### 5️⃣ **Edge Cases & State Management** (3 tests)

#### Test: "should handle removal of non-existent item gracefully"
- **Purpose**: Graceful degradation - no errors on invalid input
- **Setup**: 1-item cart
- **Action**: 
  1. Remove the existing item
  2. Try to remove non-existent item ID (999)
- **Assertions**:
  - Cart remains empty after both operations
  - `totalItems = 0`, `totalAmount = 0`
  - No exceptions thrown

#### Test: "should update the updatedAt timestamp on removal"
- **Purpose**: State mutation flag - track when cart changes
- **Setup**: 1-item cart with initial `updatedAt` timestamp
- **Action**: Remove item with 10ms delay
- **Assertions**:
  - `updatedAt` timestamp ≥ previous timestamp
  - Timestamp reflects recent mutation

#### Test: "should persist cart state after removal"
- **Purpose**: Verify state durability - cart remains consistent across retrievals
- **Setup**: Multi-item cart (Item 1 qty=2, Item 2 qty=1)
- **Action**: Remove Item 1
- **Assertions** (after removal):
  - First retrieval: 1 item, `totalItems=1`, `totalAmount=$25.50`
  - Second retrieval: Same values preserved
  - Consistent state across multiple `getCart()` calls

---

## Mocking Strategy

### ProductServiceClient Mock
```typescript
(productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
  if (productId === '1') return mockProduct;      // $10.99
  if (productId === '2') return mockProduct2;     // $25.50
  return mockProduct3;                            // $99.99
});
```

**Benefits**:
- ✅ Avoids external network calls
- ✅ Deterministic test behavior
- ✅ Realistic CartItem/Product objects
- ✅ Different price points for multi-scenario testing

---

## Key Testing Principles Applied

| Principle | Implementation |
|-----------|-----------------|
| **State Integrity** | Filter removes item, not zero quantity |
| **Recalculation** | `totalItems = Σ(qty)`, `totalAmount = Σ(price×qty)` |
| **Floating-Point Safety** | `toBeCloseTo(value, 2)` for decimal comparisons |
| **Edge Cases** | Non-existent items, empty cart, single item |
| **Persistence** | State verified across multiple retrievals |
| **Timestamp Tracking** | `updatedAt` reflects recent mutations |
| **Mocking** | ProductServiceClient mocked; zero external dependencies |

---

## Running the Tests

### Via npm
```bash
cd microservices/cart-service
npm test                      # Run all tests
npm test -- service.test.ts   # Run only service tests
npm test -- --watch          # Watch mode
```

### Coverage (Optional)
```bash
npm test -- --coverage
```

---

## Notes for Maintainers

1. **Floating-Point Precision**: The test uses `toBeCloseTo(expected, 2)` for the formula verification to handle IEEE 754 rounding edge cases.

2. **Mock Strategy**: ProductServiceClient is mocked at the constructor level via `jest.mock('./product-client')`.

3. **Realistic Data**: Product prices ($10.99, $25.50, $99.99) are chosen to test multi-scenario calculations without artificial edge cases.

4. **Backwards Compatible**: All tests follow existing patterns in `service.test.ts` for consistency and maintainability.

---

## Assertion Matrix Quick Reference

| Category | Assertion | Method |
|----------|-----------|--------|
| **Removed ID** | `not in items` | `toContainEqual()` |
| **Remaining ID** | `in items` | `toContainEqual()` |
| **Quantity (Total)** | `totalItems = Σ(qty)` | `toBe()` |
| **Amount (Total)** | `totalAmount = Σ(price×qty)` | `toBe()` or `toBeCloseTo()` |
| **Array Length** | `--1` exactly | `toHaveLength()`, `toBe()` |
| **Decimals** | `≤ 2` places | String split + length check |
| **Timestamp** | `≥ previous` | `toBeGreaterThanOrEqual()` |
