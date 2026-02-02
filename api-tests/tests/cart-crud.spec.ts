/**
 * CRUD Operations Tests for Cart API
 */

import { test, expect } from '@fixtures/index';
import {
  testUsers,
  testProducts,
  validAddToCartRequests,
  mockCartResponses,
} from '@fixtures/test-data';
import { ResponseValidator } from '@utils/index';

test.describe('Cart API - CRUD Operations', () => {
  test.describe('CREATE - Add items to cart', () => {
    test('should add a single item to cart successfully', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const item = validAddToCartRequests.singleItem;

      const response = await cartAPI.addItemToCart(userId, item);

      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(response.data?.productId).toBe(item.productId);
      expect(response.data?.quantity).toBe(item.quantity);
      expect(response.data?.name).toBe(item.name);
      expect(response.data?.price).toBe(item.price);
    });

    test('should add multiple different items to cart', async ({ cartAPI }) => {
      const userId = testUsers.user2.id;

      // Add first item
      const response1 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      expect(response1.success).toBe(true);

      // Add second item
      const response2 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.minimalData
      );
      expect(response2.success).toBe(true);

      // Verify both items are in cart
      const cartSummary = await cartAPI.getCartSummary(userId);
      expect(cartSummary.itemCount).toBe(2);
    });

    test('should add item with minimal data', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      const response = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.minimalData
      );

      expect(response.success).toBe(true);
      expect(response.data?.productId).toBe(
        validAddToCartRequests.minimalData.productId
      );
      expect(response.data?.quantity).toBe(
        validAddToCartRequests.minimalData.quantity
      );
    });

    test('should increase quantity when adding duplicate product', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user3.id;
      const item = validAddToCartRequests.singleItem;

      // Add item first time
      await cartAPI.addItemToCart(userId, item);

      // Add same item again
      const response2 = await cartAPI.addItemToCart(userId, item);

      expect(response2.success).toBe(true);
      expect(response2.data?.quantity).toBeGreaterThanOrEqual(item.quantity);
    });
  });

  test.describe('READ - Retrieve cart items', () => {
    test('should get empty cart for new user', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}`;

      const response = await cartAPI.getCartItems(userId);

      expect(response.success).toBe(true);
      expect(response.data?.items).toBeDefined();
      expect(Array.isArray(response.data?.items)).toBe(true);
      expect(response.data?.itemCount).toBe(0);
      expect(response.data?.totalPrice).toBe(0);
    });

    test('should get cart with added items', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      // Add items first
      await cartAPI.addItemToCart(userId, validAddToCartRequests.singleItem);
      await cartAPI.addItemToCart(userId, validAddToCartRequests.minimalData);

      // Get cart
      const response = await cartAPI.getCartItems(userId);

      expect(response.success).toBe(true);
      expect(response.data?.items?.length).toBeGreaterThan(0);
      expect(response.data?.itemCount).toBeGreaterThan(0);
      expect(response.data?.totalPrice).toBeGreaterThan(0);
    });

    test('should validate cart response structure', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      const response = await cartAPI.getCartItems(userId);

      const validation = ResponseValidator.validateResponseStructure(
        response.data,
        ['userId', 'items', 'itemCount', 'totalPrice']
      );

      expect(validation.isValid).toBe(true);
    });

    test('should get correct cart summary', async ({ cartAPI }) => {
      const userId = testUsers.user2.id;

      const summary = await cartAPI.getCartSummary(userId);

      expect(summary.itemCount).toBeGreaterThanOrEqual(0);
      expect(summary.totalPrice).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('UPDATE - Modify cart items', () => {
    test('should update item quantity successfully', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const newQuantity = 5;

      // Add item
      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      const itemId = addResponse.data?.itemId;

      // Update quantity
      if (itemId) {
        const updateResponse = await cartAPI.updateCartItem(
          userId,
          itemId,
          newQuantity
        );

        expect(updateResponse.success).toBe(true);
        expect(updateResponse.data?.quantity).toBe(newQuantity);
      }
    });

    test('should update quantity to 1', async ({ cartAPI }) => {
      const userId = testUsers.user2.id;

      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.largeQuantity
      );
      const itemId = addResponse.data?.itemId;

      if (itemId) {
        const updateResponse = await cartAPI.updateCartItem(userId, itemId, 1);

        expect(updateResponse.success).toBe(true);
        expect(updateResponse.data?.quantity).toBe(1);
      }
    });

    test('should maintain product details after update', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user3.id;
      const newQuantity = 3;

      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      const itemId = addResponse.data?.itemId;
      const originalProductId = addResponse.data?.productId;

      if (itemId) {
        const updateResponse = await cartAPI.updateCartItem(
          userId,
          itemId,
          newQuantity
        );

        expect(updateResponse.data?.productId).toBe(originalProductId);
        expect(updateResponse.data?.name).toBe(
          validAddToCartRequests.singleItem.name
        );
      }
    });
  });

  test.describe('DELETE - Remove items from cart', () => {
    test('should remove single item from cart', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      // Add item
      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      const itemId = addResponse.data?.itemId;

      // Remove item
      if (itemId) {
        const deleteResponse = await cartAPI.removeItemFromCart(userId, itemId);
        expect(deleteResponse.success).toBe(true);
      }
    });

    test('should clear entire cart', async ({ cartAPI }) => {
      const userId = testUsers.user2.id;

      // Add multiple items
      await cartAPI.addItemToCart(userId, validAddToCartRequests.singleItem);
      await cartAPI.addItemToCart(userId, validAddToCartRequests.minimalData);

      // Clear cart
      const clearResponse = await cartAPI.clearCart(userId);
      expect(clearResponse.success).toBe(true);

      // Verify cart is empty
      const cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.itemCount).toBe(0);
    });

    test('should maintain other items after removing one', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user3.id;

      // Add two items
      const response1 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      const itemId1 = response1.data?.itemId;

      const response2 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.minimalData
      );
      const itemId2 = response2.data?.itemId;

      // Remove first item
      if (itemId1) {
        await cartAPI.removeItemFromCart(userId, itemId1);
      }

      // Verify second item still exists
      const cartResponse = await cartAPI.getCartItems(userId);
      const hasSecondItem = cartResponse.data?.items?.some(
        (item) => item.itemId === itemId2
      );

      expect(hasSecondItem).toBe(true);
    });
  });

  test.describe('Validation', () => {
    test('should validate cart response data types', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      const response = await cartAPI.getCartItems(userId);

      const validation = ResponseValidator.validateFieldTypes(response.data, {
        userId: 'string',
        items: 'object',
        itemCount: 'number',
        totalPrice: 'number',
      });

      expect(validation.isValid).toBe(true);
    });
  });
});
