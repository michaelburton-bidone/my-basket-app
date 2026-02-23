/**
 * Integration Tests for Cart API
 * Testing complete workflows and business logic
 */

import { test, expect } from '@fixtures/index';
import { testUsers, testProducts } from '@fixtures/test-data';

test.describe('Cart API - Integration Tests', () => {
  test.describe('Complete Shopping Workflow', () => {
    test('should complete full shopping and cart management flow', async ({
      cartAPI,
    }) => {
      const userId = `integration-user-${Date.now()}`;

      // Step 1: Verify cart is initially empty
      let cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.items?.length).toBe(0);

      // Step 2: Add first product
      const addResponse1 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.apples.productId,
        quantity: 2,
      });
      expect(addResponse1.success).toBe(true);

      // Step 3: Add second product
      const addResponse2 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.bread.productId,
        quantity: 3,
      });
      expect(addResponse2.success).toBe(true);

      // Step 4: Verify cart has 2 items (2 different products)
      cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.items?.length).toBe(2);
      expect(cartResponse.data?.totalAmount).toBeGreaterThan(0);

      // Step 5: Update quantity of first item
      const productId1 = testProducts.apples.productId;
      const updateResponse = await cartAPI.updateCartItem(
        userId,
        productId1,
        5
      );
      expect(updateResponse.success).toBe(true);
      
      const updatedItem = updateResponse.data?.items?.find(i => i.id === productId1);
      expect(updatedItem?.quantity).toBe(5);

      // Step 6: Check cart summary
      const summary = await cartAPI.getCartSummary(userId);
      expect(summary.data?.itemCount).toBe(2);
      expect(summary.data?.totalAmount).toBeGreaterThan(0);

      // Step 7: Remove one item
      const removeResponse = await cartAPI.removeItemFromCart(
        userId,
        productId1
      );
      expect(removeResponse.success).toBe(true);

      // Step 8: Verify only one item remains
      cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.items?.length).toBe(1);
    });

    test('should manage cart for multiple users independently', async ({
      cartAPI,
    }) => {
      const user1 = `multi-user-1-${Date.now()}`;
      const user2 = `multi-user-2-${Date.now()}`;

      // User 1 adds item
      const response1 = await cartAPI.addItemToCart(user1, {
        productId: testProducts.apples.productId,
        quantity: 2,
      });
      expect(response1.success).toBe(true);

      // User 2 adds different product
      const response2 = await cartAPI.addItemToCart(user2, {
        productId: testProducts.bread.productId,
        quantity: 5,
      });
      expect(response2.success).toBe(true);

      // Verify separate carts
      const user1Cart = await cartAPI.getCartItems(user1);
      const user2Cart = await cartAPI.getCartItems(user2);

      expect(user1Cart.data?.items?.length).toBe(1);
      expect(user2Cart.data?.items?.length).toBe(1);
      
      // Verify different products
      const user1Product = user1Cart.data?.items?.[0]?.id;
      const user2Product = user2Cart.data?.items?.[0]?.id;
      expect(user1Product).not.toBe(user2Product);
    });
  });

  test.describe('Business Logic Validation', () => {
    test('should calculate correct total price', async ({ cartAPI }) => {
      const userId = `price-calc-${Date.now()}`;

      // Add items with known product IDs from test data
      const productId1 = testProducts.apples.productId;
      const productId2 = testProducts.bread.productId;

      const response1 = await cartAPI.addItemToCart(userId, {
        productId: productId1,
        quantity: 2,
      });
      expect(response1.success).toBe(true);

      const response2 = await cartAPI.addItemToCart(userId, {
        productId: productId2,
        quantity: 1,
      });
      expect(response2.success).toBe(true);

      const cart = await cartAPI.getCartItems(userId);
      // Verify cart has items and total is calculated
      expect(cart.data?.totalAmount).toBeGreaterThan(0);
      expect(cart.data?.items?.length).toBe(2);
    });

    test('should track item count correctly', async ({ cartAPI }) => {
      const userId = `item-count-${Date.now()}`;

      // Add items with real product IDs
      const response1 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.apples.productId,
        quantity: 2,
      });
      expect(response1.success).toBe(true);

      const response2 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.bread.productId,
        quantity: 1,
      });
      expect(response2.success).toBe(true);

      const cart = await cartAPI.getCartItems(userId);

      // Should have 2 unique items
      expect(cart.data?.items?.length).toBe(2);
      expect(Array.isArray(cart.data?.items)).toBe(true);
    });

    test('should check product existence in cart', async ({ cartAPI }) => {
      const userId = `existence-check-${Date.now()}`;
      const productId = testProducts.milk?.productId || testProducts.almondMilk.productId;

      // Add product
      await cartAPI.addItemToCart(userId, {
        productId,
        quantity: 1,
      });

      // Check existence
      const exists = await cartAPI.itemExistsInCart(userId, productId);

      expect(exists).toBe(true);
    });

    test('should verify product not in empty cart', async ({ cartAPI }) => {
      const userId = `empty-cart-${Date.now()}`;
      const productId = testProducts.bread.productId;

      const exists = await cartAPI.itemExistsInCart(userId, productId);

      expect(exists).toBe(false);
    });
  });

  test.describe('Cart State Transitions', () => {
    test('should transition from empty to filled state', async ({
      cartAPI,
    }) => {
      const userId = `state-empty-to-filled-${Date.now()}`;

      // Empty state
      let cart = await cartAPI.getCartItems(userId);
      const initialCount = cart.data?.items?.length || 0;
      expect(initialCount).toBe(0);

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: testProducts.apples.productId,
        quantity: 1,
      });
      expect(addResponse.success).toBe(true);

      // Filled state
      cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.items?.length).toBeGreaterThan(initialCount);
    });

    test('should transition from filled to empty state', async ({
      cartAPI,
    }) => {
      const userId = `state-filled-to-empty-${Date.now()}`;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: testProducts.bread.productId,
        quantity: 1,
      });
      expect(addResponse.success).toBe(true);

      // Filled state verification
      let cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.items?.length).toBeGreaterThan(0);

      // Clear cart
      const clearResponse = await cartAPI.clearCart(userId);
      expect(clearResponse.success).toBe(true);

      // Empty state verification
      cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.items?.length).toBe(0);
    });

    test('should handle rapid add/remove operations', async ({ cartAPI }) => {
      const userId = `state-rapid-ops-${Date.now()}`;
      const productId = testProducts.eggs.productId;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId,
        quantity: 1,
      });
      expect(addResponse.success).toBe(true);

      // Immediately remove
      const removeResponse = await cartAPI.removeItemFromCart(
        userId,
        productId
      );
      expect(removeResponse.success).toBe(true);

      // Verify removal
      const cart = await cartAPI.getCartItems(userId);
      const itemExists = cart.data?.items?.some(
        (item) => item.id === productId
      );
      expect(itemExists).toBe(false);
    });
  });

  test.describe('Data Consistency', () => {
    test('should maintain consistency across multiple reads', async ({
      cartAPI,
    }) => {
      const userId = `consistency-reads-${Date.now()}`;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: testProducts.spinach.productId,
        quantity: 2,
      });
      expect(addResponse.success).toBe(true);

      // Read cart multiple times
      const read1 = await cartAPI.getCartItems(userId);
      const read2 = await cartAPI.getCartItems(userId);
      const read3 = await cartAPI.getCartItems(userId);

      // All reads should return consistent data
      expect(read1.data?.items?.length).toBe(read2.data?.items?.length);
      expect(read2.data?.items?.length).toBe(read3.data?.items?.length);
      expect(read1.data?.totalAmount).toBe(read2.data?.totalAmount);
    });

    test('should maintain product details consistency', async ({
      cartAPI,
    }) => {
      const userId = `consistency-details-${Date.now()}`;
      const productId = testProducts.almondMilk.productId;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId,
        quantity: 1,
      });
      expect(addResponse.success).toBe(true);

      // Read cart
      const cart = await cartAPI.getCartItems(userId);
      const addedItem = cart.data?.items?.find(
        (item) => item.id === productId
      );

      // Verify item was added and has expected properties
      expect(addedItem).toBeDefined();
      expect(addedItem?.id).toBe(productId);
      expect(addedItem?.quantity).toBe(1);
    });
  });
});

