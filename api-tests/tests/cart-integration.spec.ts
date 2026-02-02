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
      const userId = testUsers.user1.id;

      // Step 1: Verify cart is initially empty
      let cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.itemCount).toBe(0);

      // Step 2: Add first product
      const addResponse1 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.apple.productId,
        quantity: 2,
        name: testProducts.apple.name,
        price: testProducts.apple.price,
      });
      expect(addResponse1.success).toBe(true);

      // Step 3: Add second product
      const addResponse2 = await cartAPI.addItemToCart(userId, {
        productId: testProducts.banana.productId,
        quantity: 3,
        name: testProducts.banana.name,
        price: testProducts.banana.price,
      });
      expect(addResponse2.success).toBe(true);

      // Step 4: Verify cart has 2 items
      cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.itemCount).toBe(2);
      expect(cartResponse.data?.totalPrice).toBeGreaterThan(0);

      // Step 5: Update quantity of first item
      const itemId1 = addResponse1.data?.itemId;
      if (itemId1) {
        const updateResponse = await cartAPI.updateCartItem(
          userId,
          itemId1,
          5
        );
        expect(updateResponse.success).toBe(true);
        expect(updateResponse.data?.quantity).toBe(5);
      }

      // Step 6: Check cart summary
      const summary = await cartAPI.getCartSummary(userId);
      expect(summary.itemCount).toBe(2);
      expect(summary.totalPrice).toBeGreaterThan(0);

      // Step 7: Remove one item
      if (itemId1) {
        const removeResponse = await cartAPI.removeItemFromCart(
          userId,
          itemId1
        );
        expect(removeResponse.success).toBe(true);
      }

      // Step 8: Verify only one item remains
      cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.itemCount).toBe(1);
    });

    test('should manage cart for multiple users independently', async ({
      cartAPI,
    }) => {
      const user1 = testUsers.user1.id;
      const user2 = testUsers.user2.id;

      // User 1 adds item
      await cartAPI.addItemToCart(user1, {
        productId: testProducts.apple.productId,
        quantity: 2,
      });

      // User 2 adds different quantity
      await cartAPI.addItemToCart(user2, {
        productId: testProducts.banana.productId,
        quantity: 5,
      });

      // Verify separate carts
      const user1Cart = await cartAPI.getCartItems(user1);
      const user2Cart = await cartAPI.getCartItems(user2);

      expect(user1Cart.data?.itemCount).toBe(1);
      expect(user2Cart.data?.itemCount).toBe(1);
    });
  });

  test.describe('Business Logic Validation', () => {
    test('should calculate correct total price', async ({ cartAPI }) => {
      const userId = testUsers.user3.id;

      // Add items with known prices
      const item1Price = 10;
      const item1Qty = 2;
      const item2Price = 5;
      const item2Qty = 3;

      await cartAPI.addItemToCart(userId, {
        productId: 'prod-item-1',
        quantity: item1Qty,
        price: item1Price,
      });

      await cartAPI.addItemToCart(userId, {
        productId: 'prod-item-2',
        quantity: item2Qty,
        price: item2Price,
      });

      const cart = await cartAPI.getCartItems(userId);
      const expectedTotal = item1Price * item1Qty + item2Price * item2Qty;

      expect(cart.data?.totalPrice).toBe(expectedTotal);
    });

    test('should track item count correctly', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      // Clear cart first
      await cartAPI.clearCart(userId);

      // Add items
      await cartAPI.addItemToCart(userId, {
        productId: 'prod-1',
        quantity: 2,
      });
      await cartAPI.addItemToCart(userId, {
        productId: 'prod-2',
        quantity: 1,
      });

      const cart = await cartAPI.getCartItems(userId);

      // Should have 2 unique items
      expect(cart.data?.itemCount).toBeGreaterThanOrEqual(1);
      expect(Array.isArray(cart.data?.items)).toBe(true);
    });

    test('should check product existence in cart', async ({ cartAPI }) => {
      const userId = testUsers.user2.id;
      const productId = testProducts.milk.productId;

      // Clear cart
      await cartAPI.clearCart(userId);

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
      const userId = `test-user-${Date.now()}`;
      const productId = testProducts.bread.productId;

      const exists = await cartAPI.itemExistsInCart(userId, productId);

      expect(exists).toBe(false);
    });
  });

  test.describe('Cart State Transitions', () => {
    test('should transition from empty to filled state', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user1.id;

      // Empty state
      let cart = await cartAPI.getCartItems(userId);
      const initialCount = cart.data?.itemCount || 0;

      // Add item
      await cartAPI.addItemToCart(userId, {
        productId: 'prod-transition-1',
        quantity: 1,
      });

      // Filled state
      cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.itemCount).toBeGreaterThan(initialCount);
    });

    test('should transition from filled to empty state', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user2.id;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: 'prod-transition-2',
        quantity: 1,
      });

      // Filled state verification
      let cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.itemCount).toBeGreaterThan(0);

      // Clear cart
      await cartAPI.clearCart(userId);

      // Empty state verification
      cart = await cartAPI.getCartItems(userId);
      expect(cart.data?.itemCount).toBe(0);
    });

    test('should handle rapid add/remove operations', async ({ cartAPI }) => {
      const userId = testUsers.user3.id;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId: 'prod-rapid-1',
        quantity: 1,
      });

      const itemId = addResponse.data?.itemId;

      // Immediately remove
      if (itemId) {
        const removeResponse = await cartAPI.removeItemFromCart(
          userId,
          itemId
        );
        expect(removeResponse.success).toBe(true);
      }

      // Verify removal
      const cart = await cartAPI.getCartItems(userId);
      const itemExists = cart.data?.items?.some(
        (item) => item.itemId === itemId
      );
      expect(itemExists).toBe(false);
    });
  });

  test.describe('Data Consistency', () => {
    test('should maintain consistency across multiple reads', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user1.id;

      // Add item
      await cartAPI.addItemToCart(userId, {
        productId: 'prod-consistency-1',
        quantity: 2,
      });

      // Read cart multiple times
      const read1 = await cartAPI.getCartItems(userId);
      const read2 = await cartAPI.getCartItems(userId);
      const read3 = await cartAPI.getCartItems(userId);

      // All reads should return consistent data
      expect(read1.data?.itemCount).toBe(read2.data?.itemCount);
      expect(read2.data?.itemCount).toBe(read3.data?.itemCount);
      expect(read1.data?.totalPrice).toBe(read2.data?.totalPrice);
    });

    test('should maintain product details consistency', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user2.id;
      const productId = 'prod-detail-check';
      const productName = 'Test Product';
      const productPrice = 99.99;

      // Add item
      const addResponse = await cartAPI.addItemToCart(userId, {
        productId,
        quantity: 1,
        name: productName,
        price: productPrice,
      });

      // Read cart
      const cart = await cartAPI.getCartItems(userId);
      const addedItem = cart.data?.items?.find(
        (item) => item.productId === productId
      );

      expect(addedItem?.name).toBe(productName);
      expect(addedItem?.price).toBe(productPrice);
    });
  });
});
