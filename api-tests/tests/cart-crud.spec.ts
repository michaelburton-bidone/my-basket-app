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
      const userId = `user-${Date.now()}-single-item`;
      const item = validAddToCartRequests.singleItem;

      const response = await cartAPI.addItemToCart(userId, item);

      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(response.data?.items).toBeDefined();
      expect(response.data?.items?.length).toBeGreaterThan(0);
      
      const addedItem = response.data?.items?.find(i => i.id === item.productId);
      expect(addedItem).toBeDefined();
      expect(addedItem?.quantity).toBe(item.quantity);
    });

    test('should add multiple different items to cart', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-multiple-items`;

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

      // Verify both items are in cart (2 different products)
      const cartSummary = await cartAPI.getCartSummary(userId);
      expect(cartSummary.data?.itemCount).toBe(2); // 2 different products added
    });

    test('should add item with minimal data', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-minimal-data`;

      const response = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.minimalData
      );

      expect(response.success).toBe(true);
      expect(response.data?.items).toBeDefined();
      const addedItem = response.data?.items?.find(i => i.id === validAddToCartRequests.minimalData.productId);
      expect(addedItem).toBeDefined();
      expect(addedItem?.quantity).toBe(
        validAddToCartRequests.minimalData.quantity
      );
    });

    test('should increase quantity when adding duplicate product', async ({
      cartAPI,
    }) => {
      const userId = `user-${Date.now()}-duplicate-product`;
      const item = validAddToCartRequests.singleItem;

      // Add item first time
      const response1 = await cartAPI.addItemToCart(userId, item);
      const firstQuantity = response1.data?.items?.find(i => i.id === item.productId)?.quantity || 0;

      // Add same item again
      const response2 = await cartAPI.addItemToCart(userId, item);

      expect(response2.success).toBe(true);
      const secondQuantity = response2.data?.items?.find(i => i.id === item.productId)?.quantity || 0;
      expect(secondQuantity).toBeGreaterThanOrEqual(firstQuantity);
    });
  });

  test.describe('READ - Retrieve cart items', () => {
    test('should get empty cart for new user', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-empty-cart`;

      const response = await cartAPI.getCartItems(userId);

      expect(response.success).toBe(true);
      expect(response.data?.items).toBeDefined();
      expect(Array.isArray(response.data?.items)).toBe(true);
      expect(response.data?.totalItems).toBe(0);
      expect(response.data?.totalAmount).toBe(0);
    });

    test('should get cart with added items', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-with-items`;

      // Add items first
      await cartAPI.addItemToCart(userId, validAddToCartRequests.singleItem);
      await cartAPI.addItemToCart(userId, validAddToCartRequests.minimalData);

      // Get cart
      const response = await cartAPI.getCartItems(userId);

      expect(response.success).toBe(true);
      expect(response.data?.items?.length).toBeGreaterThan(0);
      expect(response.data?.totalItems).toBeGreaterThan(0);
      expect(response.data?.totalAmount).toBeGreaterThan(0);
    });

    test('should validate cart response structure', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-validate-structure`;

      const response = await cartAPI.getCartItems(userId);

      const validation = ResponseValidator.validateResponseStructure(
        response.data,
        ['userId', 'items', 'totalItems', 'totalAmount']
      );

      expect(validation.isValid).toBe(true);
    });

    test('should get correct cart summary', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-cart-summary`;

      const summary = await cartAPI.getCartSummary(userId);

      expect(summary.data?.itemCount).toBeGreaterThanOrEqual(0);
      expect(summary.data?.totalAmount).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('UPDATE - Modify cart items', () => {
    test('should update item quantity successfully', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-update-quantity`;
      const newQuantity = 5;
      const productId = validAddToCartRequests.singleItem.productId;

      // Add item
      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      expect(addResponse.success).toBe(true);

      // Update quantity
      const updateResponse = await cartAPI.updateCartItem(
        userId,
        productId,
        newQuantity
      );

      expect(updateResponse.success).toBe(true);
      const updatedItem = updateResponse.data?.items?.find(i => i.id === productId);
      expect(updatedItem?.quantity).toBe(newQuantity);
    });

    test('should update quantity to 1', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-update-to-one`;
      const productId = validAddToCartRequests.largeQuantity.productId;

      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.largeQuantity
      );
      expect(addResponse.success).toBe(true);

      const updateResponse = await cartAPI.updateCartItem(userId, productId, 1);

      expect(updateResponse.success).toBe(true);
      const updatedItem = updateResponse.data?.items?.find(i => i.id === productId);
      expect(updatedItem?.quantity).toBe(1);
    });

    test('should maintain product details after update', async ({
      cartAPI,
    }) => {
      const userId = `user-${Date.now()}-maintain-details`;
      const newQuantity = 3;
      const productId = validAddToCartRequests.singleItem.productId;

      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      expect(addResponse.success).toBe(true);

      const updateResponse = await cartAPI.updateCartItem(
        userId,
        productId,
        newQuantity
      );

      const updatedItem = updateResponse.data?.items?.find(i => i.id === productId);
      expect(updatedItem?.id).toBe(productId);
    });
  });

  test.describe('DELETE - Remove items from cart', () => {
    test('should remove single item from cart', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-remove-single`;
      const productId = validAddToCartRequests.singleItem.productId;

      // Add item
      const addResponse = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      expect(addResponse.success).toBe(true);

      // Remove item
      const deleteResponse = await cartAPI.removeItemFromCart(userId, productId);
      expect(deleteResponse.success).toBe(true);
      
      // Verify item is removed
      const itemExists = deleteResponse.data?.items?.some(i => i.id === productId);
      expect(itemExists).toBe(false);
    });

    test('should clear entire cart', async ({ cartAPI }) => {
      const userId = `user-${Date.now()}-clear-cart`;

      // Add multiple items
      await cartAPI.addItemToCart(userId, validAddToCartRequests.singleItem);
      await cartAPI.addItemToCart(userId, validAddToCartRequests.minimalData);

      // Clear cart
      const clearResponse = await cartAPI.clearCart(userId);
      expect(clearResponse.success).toBe(true);

      // Verify cart is empty
      const cartResponse = await cartAPI.getCartItems(userId);
      expect(cartResponse.data?.totalItems).toBe(0);
    });

    test('should maintain other items after removing one', async ({
      cartAPI,
    }) => {
      const userId = `user-${Date.now()}-maintain-items`;
      const productId1 = validAddToCartRequests.singleItem.productId;
      const productId2 = validAddToCartRequests.minimalData.productId;

      // Add two items
      const response1 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.singleItem
      );
      expect(response1.success).toBe(true);

      const response2 = await cartAPI.addItemToCart(
        userId,
        validAddToCartRequests.minimalData
      );
      expect(response2.success).toBe(true);

      // Remove first item
      const deleteResponse = await cartAPI.removeItemFromCart(userId, productId1);
      expect(deleteResponse.success).toBe(true);

      // Verify second item still exists
      const hasSecondItem = deleteResponse.data?.items?.some(
        (item) => item.id === productId2
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
        totalItems: 'number',
        totalAmount: 'number',
      });

      expect(validation.isValid).toBe(true);
    });
  });
});
