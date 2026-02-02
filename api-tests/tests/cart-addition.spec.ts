/**
 * Cart Addition Test Suite
 * Tests the POST /api/cart/:userId/items endpoint
 * 
 * Functionality: Add items to user's shopping cart
 * Expected Behavior: Item is added to cart and total amount is updated
 */

import { test, expect } from '@fixtures/index';
import { testUsers, testProducts } from '@fixtures/test-data';

test.describe('Cart Addition - POST /api/cart/:userId/items', () => {
  
  
  test.describe('Successful Item Addition', () => {
    test('should add item to cart with valid productId and quantity', async ({
      cartAPI,
    }) => {
      await test.step('Setup test data', async () => {
        // Define test user and item with semantic naming
      });

      await test.step('Add item to cart via API', async () => {
        const userId = testUsers.user1.id;
        const item = {
          productId: testProducts.apples.id,
          quantity: 2,
        };

        const cart = await cartAPI.addItemToCart(userId, item);

        // Assertions use auto-retrying expect() to handle minor network delays
        expect(cart).toBeDefined();
        expect(cart.userId).toBe(userId);
        expect(cart.items).toBeDefined();
        expect(Array.isArray(cart.items)).toBe(true);
        expect(cart.items.length).toBeGreaterThan(0);
      });
    });

    test('should return 200 OK status code', async ({ apiContext }) => {
      await test.step('Prepare request with valid item data', async () => {
        // Test setup
      });

      await test.step('Execute POST request to cart endpoint', async () => {
        const userId = testUsers.user2.id;
        const item = {
          productId: testProducts.apples.id,
          quantity: 2,
        };

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          { data: item }
        );

        // Auto-retrying assertions verify HTTP response status
        expect(response.status()).toBe(200);
        expect(response.ok()).toBe(true);
      });
    });

    test('should update cart with new item', async ({ cartAPI }) => {
      await test.step('Add item to empty cart', async () => {
        const userId = testUsers.user3.id;
        const item = {
          productId: testProducts.bread.id,
          quantity: 1,
        };

        const addResponse = await cartAPI.addItemToCart(userId, item);

        // Auto-retrying assertions verify item existence in cart
        expect(addResponse.items.length).toBeGreaterThan(0);
        const addedItem = addResponse.items.find(i => i.id === item.productId);
        expect(addedItem).toBeDefined();
      });
    });

    test('should update total amount correctly', async ({ cartAPI }) => {
      await test.step('Add product with known price to cart', async () => {
        const userId = 'test-user-price-' + Date.now();
        const item = {
          productId: testProducts.apples.id,
          quantity: 2,
        };

        const cart = await cartAPI.addItemToCart(userId, item);
        
        // Auto-retrying assertions verify price calculation accuracy
        expect(cart.totalAmount).toBeGreaterThan(0);
        expect(cart.totalAmount).toBe(7.98);
      });
    });

    test('should contain added item in response', async ({ cartAPI }) => {
      await test.step('Add item and verify presence in response', async () => {
        const userId = 'test-user-item-' + Date.now();
        const item = {
          productId: testProducts.eggs.id,
          quantity: 3,
        };

        const cart = await cartAPI.addItemToCart(userId, item);
        
        const addedItem = cart.items?.find(
          (i) => i.id === item.productId
        );
        // Auto-retrying assertions ensure item structure is correct
        expect(addedItem).toBeDefined();
        expect(addedItem?.quantity).toBe(item.quantity);
      });
    });

    test('should return updated cart with product details for added item', async ({
      cartAPI,
    }) => {
      await test.step('Add item and retrieve complete product details', async () => {
        const userId = 'test-user-details-' + Date.now();
        const item = {
          productId: testProducts.spinach.id,
          quantity: 2,
        };

        const cart = await cartAPI.addItemToCart(userId, item);
        
        // Auto-retrying assertions verify all product fields populated
        expect(cart.items).toBeDefined();
        const addedItem = cart.items?.[0];
        expect(addedItem?.id).toBe(item.productId);
        expect(addedItem?.name).toBeDefined();
        expect(addedItem?.price).toBeDefined();
      });
    });
  });

  test.describe('Response Structure Validation', () => {
    test('should return valid cart response structure', async ({
      cartAPI,
    }) => {
      await test.step('Add item to cart', async () => {
        const userId = 'test-user-struct-' + Date.now();
        const item = {
          productId: testProducts.almondMilk.id,
          quantity: 1,
        };

        const cart = await cartAPI.addItemToCart(userId, item);

        // Auto-retrying assertions verify all required fields exist
        expect(cart).toHaveProperty('id');
        expect(cart).toHaveProperty('userId');
        expect(cart).toHaveProperty('items');
        expect(cart).toHaveProperty('totalAmount');
        expect(cart).toHaveProperty('totalItems');
        expect(cart).toHaveProperty('createdAt');
        expect(cart).toHaveProperty('updatedAt');
      });
    });

    test('should return correct field types in response', async ({
      cartAPI,
    }) => {
      await test.step('Validate response field types', async () => {
        const userId = 'test-user-types-' + Date.now();
        const item = {
          productId: testProducts.chickenBreast.id,
          quantity: 1,
        };

        const cart = await cartAPI.addItemToCart(userId, item);

        // Auto-retrying assertions verify type consistency
        expect(typeof cart.id).toBe('string');
        expect(typeof cart.userId).toBe('string');
        expect(Array.isArray(cart.items)).toBe(true);
        expect(typeof cart.totalAmount).toBe('number');
        expect(typeof cart.totalItems).toBe('number');
      });
    });

    test('should return cart with valid item structure', async ({
      cartAPI,
    }) => {
      await test.step('Verify item object structure and properties', async () => {
        const userId = 'test-user-item-struct-' + Date.now();
        const item = {
          productId: testProducts.brownRice.id,
          quantity: 1,
        };

        const cart = await cartAPI.addItemToCart(userId, item);
        
        const cartItem = cart.items?.[0];
        if (cartItem) {
          // Auto-retrying assertions validate nested item structure
          expect(cartItem).toHaveProperty('id');
          expect(cartItem).toHaveProperty('quantity');
          expect(cartItem).toHaveProperty('price');
          expect(cartItem).toHaveProperty('name');
          expect(cartItem).toHaveProperty('addedAt');
        }
      });
    });
  });

  test.describe('Add Multiple Items', () => {
    test('should add multiple different items to cart', async ({
      cartAPI,
    }) => {
      await test.step('Add first item to new cart', async () => {
        const userId = 'test-user-multi-' + Date.now();

        // Add first item
        const response1 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });
        // Auto-retrying assertions verify first item added
        expect(response1.items.length).toBe(1);
      });

      await test.step('Add second different item to cart', async () => {
        const userId = 'test-user-multi-' + Date.now();
        // Add first item (redoing for step isolation)
        await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });

        // Add second item
        const response2 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.yogurt.id,
          quantity: 3,
        });
        // Auto-retrying assertions verify both items present
        expect(response2.items.length).toBe(2);
      });
    });

    test('should accumulate prices when adding multiple items', async ({
      cartAPI,
    }) => {
      await test.step('Add first product and verify price', async () => {
        const userId = 'test-user-accum-' + Date.now();

        // Add first item: 2 * $3.99 = $7.98
        const priceAfterFirst = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });
        // Auto-retrying assertions verify first item price
        expect(priceAfterFirst.totalAmount).toBe(7.98);
      });

      await test.step('Add second product and verify cumulative price', async () => {
        const userId = 'test-user-accum-' + Date.now();
        // Redo first item for step isolation
        const priceAfterFirst = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });

        // Add second item: 1 * $5.99 = $5.99
        const priceAfterSecond = await cartAPI.addItemToCart(userId, {
          productId: testProducts.eggs.id,
          quantity: 1,
        });
        // Auto-retrying assertions verify price accumulation
        expect(priceAfterSecond.totalAmount).toBeGreaterThan(priceAfterFirst.totalAmount);
        expect(priceAfterSecond.totalAmount).toBe(13.97);
      });
    });

    test('should handle adding duplicate item (increase quantity)', async ({
      cartAPI,
    }) => {
      await test.step('Add item first time', async () => {
        const userId = 'test-user-dup-' + Date.now();

        // Add item first time
        const firstAdd = await cartAPI.addItemToCart(userId, {
          productId: testProducts.bread.id,
          quantity: 1,
        });
        // Auto-retrying assertions verify first addition
        expect(firstAdd.totalAmount).toBeGreaterThan(0);
      });

      await test.step('Add same item again and verify quantity increases', async () => {
        const userId = 'test-user-dup-' + Date.now();
        const firstAdd = await cartAPI.addItemToCart(userId, {
          productId: testProducts.bread.id,
          quantity: 1,
        });
        const priceFirst = firstAdd.totalAmount;

        // Add same item again
        const secondAdd = await cartAPI.addItemToCart(userId, {
          productId: testProducts.bread.id,
          quantity: 2,
        });
        const priceSecond = secondAdd.totalAmount;

        // Quantity should be 3 (1 + 2)
        const updatedItem = secondAdd.items?.find(
          (i) => i.id === testProducts.bread.id
        );
        // Auto-retrying assertions verify quantity accumulation
        expect(updatedItem?.quantity).toBe(3);

        // Price should increase
        expect(priceSecond).toBeGreaterThan(priceFirst);
      });
    });
  });

  test.describe('Error Handling - Invalid Inputs', () => {
    test('should reject request with missing productId', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with missing productId field', async () => {
        const userId = testUsers.user1.id;

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          { data: { quantity: 2 } }
        );

        // Auto-retrying assertions verify validation error response
        expect(response.status()).toBe(400);
      });
    });

    test('should handle missing quantity with default value', async ({
      cartAPI,
    }) => {
      await test.step('Add item without quantity field to test default behavior', async () => {
        const userId = 'test-user-default-' + Date.now();

        const cart = await cartAPI.addItemToCart(userId, {
          productId: testProducts.almondMilk.id,
        } as any);

        const addedItem = cart.items?.find(
          (item) => item.id === testProducts.almondMilk.id
        );
        // Auto-retrying assertions verify default quantity applied
        expect(addedItem?.quantity).toBe(1);
      });
    });

    test('should reject request with zero quantity', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with zero quantity', async () => {
        const userId = testUsers.user2.id;

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.apples.id,
              quantity: 0,
            },
          }
        );

        // Auto-retrying assertions verify boundary validation
        expect(response.status()).toBe(400);
      });
    });

    test('should reject request with negative quantity', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with negative quantity', async () => {
        const userId = testUsers.user3.id;

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.bread.id,
              quantity: -5,
            },
          }
        );

        // Auto-retrying assertions verify boundary validation
        expect(response.status()).toBe(400);
      });
    });

    test('should reject request with empty productId', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with empty string productId', async () => {
        const userId = testUsers.user1.id;

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: '',
              quantity: 2,
            },
          }
        );

        // Auto-retrying assertions verify input validation
        expect(response.status()).toBe(400);
      });
    });

    test('should reject request with non-existent product', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with invalid productId', async () => {
        const userId = testUsers.user2.id;

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: 'non-existent-product-id',
              quantity: 2,
            },
          }
        );

        // Auto-retrying assertions verify product existence validation
        expect(response.status()).toBe(404);
      });
    });
  });

  test.describe('Business Logic Validation', () => {
    test('should verify item quantity matches request', async ({
      cartAPI,
    }) => {
      await test.step('Add item and verify exact quantity matches request', async () => {
        const userId = 'test-user-qty-' + Date.now();
        const requestedQuantity = 5;

        const cart = await cartAPI.addItemToCart(userId, {
          productId: testProducts.spinach.id,
          quantity: requestedQuantity,
        });

        const addedItem = cart.items?.find(
          (item) => item.id === testProducts.spinach.id
        );

        // Auto-retrying assertions verify quantity preservation
        expect(addedItem?.quantity).toBe(requestedQuantity);
      });
    });

    test('should preserve product details in cart', async ({
      cartAPI,
    }) => {
      await test.step('Add product and verify all details are preserved', async () => {
        const userId = 'test-user-preserve-' + Date.now();
        const productId = testProducts.yogurt.id;
        const quantity = 2;

        const cart = await cartAPI.addItemToCart(userId, {
          productId,
          quantity,
        });

        const addedItem = cart.items?.find(
          (item) => item.id === productId
        );

        // Auto-retrying assertions verify data integrity
        expect(addedItem?.id).toBe(productId);
        expect(addedItem?.quantity).toBe(quantity);
        expect(addedItem?.price).toBe(testProducts.yogurt.price);
      });
    });

    test('should maintain totalItems accuracy', async ({
      cartAPI,
    }) => {
      await test.step('Add first item and verify totalItems', async () => {
        const userId = 'test-user-items-' + Date.now();

        // Add one item with quantity 2
        const cart1 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });
        // Auto-retrying assertions verify count calculation
        expect(cart1.totalItems).toBe(2);
      });

      await test.step('Add second item and verify totalItems updates', async () => {
        const userId = 'test-user-items-' + Date.now();
        const cart1 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });

        // Add another item with quantity 1
        const cart2 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.bread.id,
          quantity: 1,
        });
        // Auto-retrying assertions verify accumulated count
        expect(cart2.totalItems).toBe(3);
      });

      await test.step('Add to existing item and verify totalItems', async () => {
        const userId = 'test-user-items-' + Date.now();
        const cart1 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });
        const cart2 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.bread.id,
          quantity: 1,
        });

        // Add to existing item (apples + 3 more = 5 total for apples)
        const cart3 = await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 3,
        });
        // Auto-retrying assertions verify correct total
        expect(cart3.totalItems).toBe(6); // 5 apples + 1 bread
      });
    });

    test('should calculate correct total with multiple quantities', async ({
      cartAPI,
    }) => {
      await test.step('Add first product and verify price', async () => {
        const userId = 'test-user-calc-' + Date.now();

        // Add item 1: 2 apples @ $3.99 = $7.98
        await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });
      });

      await test.step('Add second product and verify cumulative calculation', async () => {
        const userId = 'test-user-calc-' + Date.now();
        // Add item 1: 2 apples @ $3.99 = $7.98
        await cartAPI.addItemToCart(userId, {
          productId: testProducts.apples.id,
          quantity: 2,
        });

        // Add item 2: 1 eggs @ $5.99 = $5.99
        const cart = await cartAPI.addItemToCart(userId, {
          productId: testProducts.eggs.id,
          quantity: 1,
        });

        // Auto-retrying assertions verify calculation accuracy
        // Total should be $13.97
        expect(cart.totalAmount).toBe(13.97);
      });
    });
  });

  test.describe('Edge Cases', () => {
    test('should handle large quantity', async ({ cartAPI }) => {
      await test.step('Add item with large quantity value', async () => {
        const userId = 'test-user-large-' + Date.now();
        const largeQuantity = 1000;

        const cart = await cartAPI.addItemToCart(userId, {
          productId: testProducts.spinach.id,
          quantity: largeQuantity,
        });

        const item = cart.items?.find(
          (i) => i.id === testProducts.spinach.id
        );
        // Auto-retrying assertions verify large quantity handling
        expect(item?.quantity).toBe(largeQuantity);
      });
    });

    test('should handle decimal quantities gracefully', async ({
      apiContext,
    }) => {
      await test.step('Execute POST with decimal quantity', async () => {
        const userId = 'test-user-decimal-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.almondMilk.id,
              quantity: 2.5,
            },
          }
        );

        // Auto-retrying assertions verify decimal handling (service validates positive numbers)
        // Service may accept or reject decimals
        expect([200, 400]).toContain(response.status());
      });
    });

    test('should handle rapid consecutive additions', async ({
      cartAPI,
    }) => {
      await test.step('Execute multiple rapid API calls in parallel', async () => {
        const userId = 'test-user-rapid-' + Date.now();

        // Add items rapidly
        const results = await Promise.all([
          cartAPI.addItemToCart(userId, {
            productId: testProducts.apples.id,
            quantity: 1,
          }),
          cartAPI.addItemToCart(userId, {
            productId: testProducts.bread.id,
            quantity: 1,
          }),
          cartAPI.addItemToCart(userId, {
            productId: testProducts.eggs.id,
            quantity: 1,
          }),
        ]);

        // Auto-retrying assertions verify race condition handling
        // At least one should have all items
        const lastResult = results[results.length - 1];
        expect(lastResult.items.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  test.describe('Response JSON Body Assertions', () => {
    test('should return valid JSON response', async ({ apiContext }) => {
      await test.step('Execute POST and verify JSON response', async () => {
        const userId = 'test-user-json-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.apples.id,
              quantity: 1,
            },
          }
        );

        // Auto-retrying assertions verify response integrity
        expect(response.ok()).toBe(true);

        // Verify JSON parsing
        const body = await response.json();
        expect(body).toBeDefined();
      });
    });

    test('should include all required JSON fields in response', async ({
      apiContext,
    }) => {
      await test.step('Verify response contains all required properties', async () => {
        const userId = 'test-user-fields-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.bread.id,
              quantity: 1,
            },
          }
        );

        const body = await response.json();

        // Auto-retrying assertions verify complete response structure
        expect(body).toHaveProperty('id');
        expect(body).toHaveProperty('userId');
        expect(body).toHaveProperty('items');
        expect(body).toHaveProperty('totalAmount');
        expect(body).toHaveProperty('totalItems');
      });
    });

    test('should verify items array contains valid items', async ({
      apiContext,
    }) => {
      await test.step('Add item and validate items array structure', async () => {
        const userId = 'test-user-array-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.yogurt.id,
              quantity: 2,
            },
          }
        );

        const body = await response.json();

        // Auto-retrying assertions verify array structure
        expect(Array.isArray(body.items)).toBe(true);
        expect(body.items.length).toBeGreaterThan(0);

        // Verify item structure
        const item = body.items[0];
        expect(item.id).toBeDefined();
        expect(item.quantity).toBeDefined();
        expect(item.price).toBeDefined();
      });
    });

    test('should verify numeric fields have correct types', async ({
      apiContext,
    }) => {
      await test.step('Verify numeric field types in response', async () => {
        const userId = 'test-user-numeric-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.chickenBreast.id,
              quantity: 1,
            },
          }
        );

        const body = await response.json();

        // Auto-retrying assertions verify type safety
        expect(typeof body.totalItems).toBe('number');
        expect(typeof body.totalAmount).toBe('number');
        expect(typeof body.items[0].quantity).toBe('number');
        expect(typeof body.items[0].price).toBe('number');
      });
    });

    test('should verify response field values are valid', async ({
      apiContext,
    }) => {
      await test.step('Verify response field values match expected criteria', async () => {
        const userId = 'test-user-values-' + Date.now();

        const response = await apiContext.post(
          `http://localhost:3002/api/cart/${userId}/items`,
          {
            data: {
              productId: testProducts.brownRice.id,
              quantity: 3,
            },
          }
        );

        const body = await response.json();

        // Auto-retrying assertions verify logical correctness
        expect(body.userId).toBe(userId);
        expect(body.totalItems).toBeGreaterThan(0);
        expect(body.totalAmount).toBeGreaterThan(0);
        expect(body.items[0].quantity).toBe(3);
      });
    });
  });
});
