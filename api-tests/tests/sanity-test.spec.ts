/**
 * Sanity Test Suite - Happy Path Scenarios
 * 
 * Tests all happy path scenarios from the Cart Service API Test Plan
 * Session: sanity-test
 * 
 * Coverage:
 * - GET /cart/{userId} - Happy paths
 * - POST /cart/{userId}/items - Happy paths
 * - PUT /cart/{userId}/items/{productId} - Happy paths
 * - DELETE /cart/{userId}/items/{productId} - Happy paths
 * - DELETE /cart/{userId} - Happy paths
 * - GET /cart/{userId}/summary - Happy paths
 * - GET /health - Health check
 */

import { test, expect } from '@fixtures/index';

// Product ID mapping - using real product IDs from the product service
const PRODUCTS = {
  laptop: '1',      // Organic Apples - $3.99
  mouse: '2',       // Whole Wheat Bread - $4.49
  keyboard: '3',    // Free-Range Eggs - $5.99
  monitor: '4',     // Organic Spinach - $2.99
  headphones: '5',  // Almond Milk - $3.79
  cable: '6',       // Chicken Breast - $9.99
  usb: '7',         // Brown Rice - $2.49
  adapter: '1',
  charger: '2',
  stand: '3',
  dock: '4',
  case: '5',
  glass: '6',
  screen: '7',
  item1: '1',
  item2: '2',
  x: '3',
  y: '4',
  a: '5',
  b: '6',
};

test.describe('Cart Service API - Sanity Test (Happy Paths)', () => {
  
  test.describe('1. Health Check Endpoint', () => {
    test('1.1: GET /health - Service is healthy', async ({ apiContext }) => {
      const response = await apiContext.get('/api/health');
      
      expect(response.status()).toBe(200);
      const body = await response.json();
      expect(body).toHaveProperty('status', 'healthy');
      expect(body).toHaveProperty('service', 'cart-service');
      expect(body).toHaveProperty('timestamp');
    });
  });

  test.describe('2. GET /cart/{userId} - Retrieve Cart', () => {
    test('2.1: Get empty cart for new user', async ({ cartAPI }) => {
      const userId = `sanity-test-user-empty-${Date.now()}`;
      
      const response = await cartAPI.getCartItems(userId);
      
      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(response.data?.userId).toBe(userId);
      expect(response.data?.items).toEqual([]);
      expect(response.data?.totalItems).toBe(0);
      expect(response.data?.totalAmount).toBe(0);
      expect(response.data?.id).toBeDefined();
      expect(response.data?.createdAt).toBeDefined();
      expect(response.data?.updatedAt).toBeDefined();
    });

    test('2.2: Get cart after adding items', async ({ cartAPI }) => {
      const userId = `sanity-test-user-with-items-${Date.now()}`;
      
      // Add an item first
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.laptop,
        quantity: 1,
      });
      expect(addResponse.success).toBe(true);
      
      // Now retrieve the cart
      const getResponse = await cartAPI.getCartItems(userId);
      
      expect(getResponse.success).toBe(true);
      expect(getResponse.data?.items).toHaveLength(1);
      expect(getResponse.data?.items?.[0].id).toBe(PRODUCTS.laptop);
      expect(getResponse.data?.items?.[0].quantity).toBe(1);
      expect(getResponse.data?.totalItems).toBe(1);
      expect(getResponse.data?.totalAmount).toBeGreaterThan(0);
    });
  });

  test.describe('3. POST /cart/{userId}/items - Add Item to Cart', () => {
    test('3.1: Add new item with default quantity (1)', async ({ cartAPI }) => {
      const userId = `sanity-test-add-default-qty-${Date.now()}`;
      
      const response = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.mouse,
      });
      
      expect(response.success).toBe(true);
      expect(response.data?.items).toHaveLength(1);
      expect(response.data?.items?.[0].id).toBe(PRODUCTS.mouse);
      expect(response.data?.items?.[0].quantity).toBe(1);
      expect(response.data?.totalItems).toBe(1);
      expect(response.data?.userId).toBe(userId);
    });

    test('3.2: Add new item with specific quantity', async ({ cartAPI }) => {
      const userId = `sanity-test-add-specific-qty-${Date.now()}`;
      
      const response = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.keyboard,
        quantity: 5,
      });
      
      expect(response.success).toBe(true);
      expect(response.data?.items).toHaveLength(1);
      expect(response.data?.items?.[0].id).toBe(PRODUCTS.keyboard);
      expect(response.data?.items?.[0].quantity).toBe(5);
      expect(response.data?.totalItems).toBe(5);
    });

    test('3.3: Add multiple different items to cart', async ({ cartAPI }) => {
      const userId = `sanity-test-add-multiple-${Date.now()}`;
      
      // Add first item
      const response1 = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.monitor,
        quantity: 1,
      });
      expect(response1.success).toBe(true);
      
      // Add second item
      const response2 = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.headphones,
        quantity: 2,
      });
      expect(response2.success).toBe(true);
      
      // Verify both items are in cart
      expect(response2.data?.items).toHaveLength(2);
      expect(response2.data?.totalItems).toBe(3); // 1 + 2
      
      const monitor = response2.data?.items?.find(i => i.id === PRODUCTS.monitor);
      const headphones = response2.data?.items?.find(i => i.id === PRODUCTS.headphones);
      
      expect(monitor?.quantity).toBe(1);
      expect(headphones?.quantity).toBe(2);
    });

    test('3.4: Increment existing item when adding same product again', async ({ cartAPI }) => {
      const userId = `sanity-test-increment-qty-${Date.now()}`;
      
      // Add item first time with qty 2
      const response1 = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.cable,
        quantity: 2,
      });
      expect(response1.success).toBe(true);
      const firstItem = response1.data?.items?.find(i => i.id === PRODUCTS.cable);
      expect(firstItem?.quantity).toBe(2);
      
      // Add same product again with qty 3
      const response2 = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.cable,
        quantity: 3,
      });
      expect(response2.success).toBe(true);
      
      // Verify quantity is incremented (2 + 3 = 5)
      expect(response2.data?.items).toHaveLength(1);
      const updatedItem = response2.data?.items?.find(i => i.id === PRODUCTS.cable);
      expect(updatedItem?.quantity).toBe(5);
      expect(response2.data?.totalItems).toBe(5);
    });
  });

  test.describe('4. PUT /cart/{userId}/items/{productId} - Update Item Quantity', () => {
    test('4.1: Update item quantity - increase', async ({ cartAPI }) => {
      const userId = `sanity-test-update-qty-increase-${Date.now()}`;
      
      // Add item with qty 2
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.usb,
        quantity: 2,
      });
      expect(addResponse.success).toBe(true);
      
      // Update quantity to 5
      const updateResponse = await cartAPI.updateCartItem(userId, PRODUCTS.usb, 5);
      
      expect(updateResponse.success).toBe(true);
      expect(updateResponse.data?.items?.[0].quantity).toBe(5);
      expect(updateResponse.data?.totalItems).toBe(5);
    });

    test('4.2: Update item quantity - decrease', async ({ cartAPI }) => {
      const userId = `sanity-test-update-qty-decrease-${Date.now()}`;
      
      // Add item with qty 10
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.adapter,
        quantity: 10,
      });
      expect(addResponse.success).toBe(true);
      
      // Update quantity to 3
      const updateResponse = await cartAPI.updateCartItem(userId, PRODUCTS.adapter, 3);
      
      expect(updateResponse.success).toBe(true);
      expect(updateResponse.data?.items?.[0].quantity).toBe(3);
      expect(updateResponse.data?.totalItems).toBe(3);
    });

    test('4.3: Update item quantity to minimum (1)', async ({ cartAPI }) => {
      const userId = `sanity-test-update-qty-min-${Date.now()}`;
      
      // Add item with qty 5
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.charger,
        quantity: 5,
      });
      expect(addResponse.success).toBe(true);
      
      // Update quantity to 1
      const updateResponse = await cartAPI.updateCartItem(userId, PRODUCTS.charger, 1);
      
      expect(updateResponse.success).toBe(true);
      expect(updateResponse.data?.items?.[0].quantity).toBe(1);
      expect(updateResponse.data?.totalItems).toBe(1);
    });

    test('4.4: Remove item by updating quantity to 0', async ({ cartAPI }) => {
      const userId = `sanity-test-update-qty-remove-${Date.now()}`;
      
      // Add items
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.stand,
        quantity: 2,
      });
      expect(addResponse.success).toBe(true);
      expect(addResponse.data?.items).toHaveLength(1);
      
      // Add another item
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.dock,
        quantity: 1,
      });
      
      // Remove first item by setting qty to 0
      const updateResponse = await cartAPI.updateCartItem(userId, PRODUCTS.stand, 0);
      
      expect(updateResponse.success).toBe(true);
      expect(updateResponse.data?.items).toHaveLength(1);
      expect(updateResponse.data?.items?.[0].id).toBe(PRODUCTS.dock);
      expect(updateResponse.data?.totalItems).toBe(1);
    });
  });

  test.describe('5. DELETE /cart/{userId}/items/{productId} - Remove Item', () => {
    test('5.1: Remove item from cart with multiple items', async ({ cartAPI }) => {
      const userId = `sanity-test-remove-multi-${Date.now()}`;
      
      // Add two items
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.case,
        quantity: 1,
      });
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.glass,
        quantity: 1,
      });
      
      // Remove first item
      const response = await cartAPI.removeItemFromCart(userId, PRODUCTS.case);
      
      expect(response.success).toBe(true);
      expect(response.data?.items).toHaveLength(1);
      expect(response.data?.items?.[0].id).toBe(PRODUCTS.glass);
      expect(response.data?.totalItems).toBe(1);
    });

    test('5.2: Remove last item from cart', async ({ cartAPI }) => {
      const userId = `sanity-test-remove-last-${Date.now()}`;
      
      // Add single item
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.screen,
        quantity: 1,
      });
      
      // Remove it
      const response = await cartAPI.removeItemFromCart(userId, PRODUCTS.screen);
      
      expect(response.success).toBe(true);
      expect(response.data?.items).toHaveLength(0);
      expect(response.data?.totalItems).toBe(0);
      expect(response.data?.totalAmount).toBe(0);
    });

    test('5.3: Remove item from empty cart (idempotent)', async ({ cartAPI }) => {
      const userId = `sanity-test-remove-empty-${Date.now()}`;
      
      // Get empty cart first
      const getResponse = await cartAPI.getCartItems(userId);
      expect(getResponse.success).toBe(true);
      expect(getResponse.data?.items).toHaveLength(0);
      
      // Try to remove non-existent item
      const removeResponse = await cartAPI.removeItemFromCart(userId, PRODUCTS.item1);
      
      expect(removeResponse.success).toBe(true);
      expect(removeResponse.data?.items).toHaveLength(0);
    });
  });

  test.describe('6. DELETE /cart/{userId} - Clear Cart', () => {
    test('6.1: Clear cart with multiple items', async ({ cartAPI }) => {
      const userId = `sanity-test-clear-multi-${Date.now()}`;
      
      // Add multiple items
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.item1,
        quantity: 2,
      });
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.item2,
        quantity: 3,
      });
      
      // Verify items are added
      let getResponse = await cartAPI.getCartItems(userId);
      expect(getResponse.data?.items).toHaveLength(2);
      expect(getResponse.data?.totalItems).toBe(5);
      
      // Clear cart
      const clearResponse = await cartAPI.clearCart(userId);
      
      expect(clearResponse.success).toBe(true);
      expect(clearResponse.data?.items).toHaveLength(0);
      expect(clearResponse.data?.totalItems).toBe(0);
      expect(clearResponse.data?.totalAmount).toBe(0);
    });

    test('6.2: Clear already empty cart', async ({ cartAPI }) => {
      const userId = `sanity-test-clear-empty-${Date.now()}`;
      
      // Get empty cart
      const getResponse = await cartAPI.getCartItems(userId);
      expect(getResponse.data?.items).toHaveLength(0);
      
      // Clear empty cart
      const clearResponse = await cartAPI.clearCart(userId);
      
      expect(clearResponse.success).toBe(true);
      expect(clearResponse.data?.items).toHaveLength(0);
      expect(clearResponse.data?.totalAmount).toBe(0);
    });
  });

  test.describe('7. GET /cart/{userId}/summary - Cart Summary', () => {
    test('7.1: Get summary of cart with items', async ({ cartAPI }) => {
      const userId = `sanity-test-summary-with-items-${Date.now()}`;
      
      // Add items to cart
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.a,
        quantity: 2,
      });
      await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.b,
        quantity: 3,
      });
      
      // Get summary
      const response = await cartAPI.getCartSummary(userId);
      
      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(response.data?.totalItems).toBe(5); // 2 + 3
      expect(response.data?.itemCount).toBe(2); // 2 different products
      expect(response.data?.totalAmount).toBeGreaterThan(0);
    });

    test('7.2: Get summary of empty cart', async ({ cartAPI }) => {
      const userId = `sanity-test-summary-empty-${Date.now()}`;
      
      // Get summary of empty cart
      const response = await cartAPI.getCartSummary(userId);
      
      expect(response.success).toBe(true);
      expect(response.data?.totalItems).toBe(0);
      expect(response.data?.itemCount).toBe(0);
      expect(response.data?.totalAmount).toBe(0);
    });
  });

  test.describe('8. Integration - Complete Shopping Journey', () => {
    test('8.1: Complete workflow - Add, Update, Remove, Clear', async ({ cartAPI }) => {
      const userId = `sanity-test-complete-workflow-${Date.now()}`;
      
      // Step 1: Get empty cart
      let cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.items).toHaveLength(0);
      
      // Step 2: Add first item (qty: 1)
      cartResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.laptop,
        quantity: 1,
      });
      expect(cartResponse.data?.items).toHaveLength(1);
      expect(cartResponse.data?.totalItems).toBe(1);
      
      // Step 3: Add second item (qty: 2)
      cartResponse = await cartAPI.addItemToCart(userId, {
        productId: PRODUCTS.mouse,
        quantity: 2,
      });
      expect(cartResponse.data?.items).toHaveLength(2);
      expect(cartResponse.data?.totalItems).toBe(3);
      
      // Step 4: Get summary - verify 2 items, 3 total qty
      let summaryResponse = await cartAPI.getCartSummary(userId);
      expect(summaryResponse.data?.itemCount).toBe(2);
      expect(summaryResponse.data?.totalItems).toBe(3);
      
      // Step 5: Update first item quantity (1 → 3)
      cartResponse = await cartAPI.updateCartItem(userId, PRODUCTS.laptop, 3);
      expect(cartResponse.data?.totalItems).toBe(5); // 3 + 2
      
      // Step 6: Remove second item
      cartResponse = await cartAPI.removeItemFromCart(userId, PRODUCTS.mouse);
      expect(cartResponse.data?.items).toHaveLength(1);
      expect(cartResponse.data?.totalItems).toBe(3);
      
      // Step 7: Get summary - verify update
      summaryResponse = await cartAPI.getCartSummary(userId);
      expect(summaryResponse.data?.itemCount).toBe(1);
      expect(summaryResponse.data?.totalItems).toBe(3);
      
      // Step 8: Clear cart
      cartResponse = await cartAPI.clearCart(userId);
      expect(cartResponse.data?.items).toHaveLength(0);
      expect(cartResponse.data?.totalItems).toBe(0);
      
      // Step 9: Verify cart is empty
      cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.items).toHaveLength(0);
    });

    test('8.2: Multiple users - isolated carts', async ({ cartAPI }) => {
      const userId1 = `sanity-test-user1-${Date.now()}`;
      const userId2 = `sanity-test-user2-${Date.now()}`;
      
      // User 1 adds items
      await cartAPI.addItemToCart(userId1, {
        productId: PRODUCTS.x,
        quantity: 2,
      });
      
      // User 2 adds different items
      await cartAPI.addItemToCart(userId2, {
        productId: PRODUCTS.y,
        quantity: 3,
      });
      
      // Verify User 1's cart
      let cart1 = await cartAPI.getCartItems(userId1);
      expect(cart1.data?.items).toHaveLength(1);
      expect(cart1.data?.items?.[0].id).toBe(PRODUCTS.x);
      expect(cart1.data?.items?.[0].quantity).toBe(2);
      expect(cart1.data?.totalItems).toBe(2);
      
      // Verify User 2's cart
      let cart2 = await cartAPI.getCartItems(userId2);
      expect(cart2.data?.items).toHaveLength(1);
      expect(cart2.data?.items?.[0].id).toBe(PRODUCTS.y);
      expect(cart2.data?.items?.[0].quantity).toBe(3);
      expect(cart2.data?.totalItems).toBe(3);
      
      // User 1 clears cart
      await cartAPI.clearCart(userId1);
      
      // Verify User 1's cart is empty but User 2's is not
      cart1 = await cartAPI.getCartItems(userId1);
      expect(cart1.data?.items).toHaveLength(0);
      
      cart2 = await cartAPI.getCartItems(userId2);
      expect(cart2.data?.items).toHaveLength(1);
    });
  });

});

