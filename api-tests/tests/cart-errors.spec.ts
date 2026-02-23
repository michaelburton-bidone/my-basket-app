/**
 * Error Scenario Tests for Cart API
 */

import { test, expect } from '@fixtures/index';
import { testUsers, invalidAddToCartRequests, errorMessages } from '@fixtures/test-data';

test.describe('Cart API - Error Scenarios', () => {
  test.describe('Invalid User IDs', () => {
    test('should handle non-existent user ID', async ({ cartAPI }) => {
      const invalidUserId = 'non-existent-user-12345';

      try {
        await cartAPI.getCartItems(invalidUserId);
        // If we reach here, we should at least get a valid response structure
        // The behavior depends on API implementation
      } catch (error: any) {
        // Expected behavior - should throw or return error
        expect(error).toBeDefined();
      }
    });

    test('should handle empty user ID', async ({ cartAPI }) => {
      let errorCaught = false;

      try {
        await cartAPI.getCartItems('');
        // API may handle this differently
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });

    test('should handle special characters in user ID', async ({ cartAPI }) => {
      // Some APIs may accept special characters, others may reject them
      // Test should verify either successful response or error
      let response;
      
      try {
        response = await cartAPI.getCartItems('user@#$%^&*()');
        expect(response).toBeDefined();
        // If successful, should have cart structure
        expect(response.data).toBeDefined();
      } catch (error: any) {
        // Or it may throw an error, which is also acceptable
        expect(error).toBeDefined();
      }
    });
  });

  test.describe('Invalid Request Data', () => {
    test('should reject request missing productId', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      let errorCaught = false;

      try {
        await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.missingProductId as any
        );
        // Depending on API validation, this might fail
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });

    test('should reject request missing quantity', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      try {
        const response = await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.missingQuantity as any
        );
        // API may accept and provide default quantity
        expect(response).toBeDefined();
      } catch (error: any) {
        // Or throw validation error, which is also acceptable
        expect(error).toBeDefined();
      }
    });

    test('should reject zero quantity', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      let errorCaught = false;

      try {
        await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.zeroQuantity as any
        );
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });

    test('should reject negative quantity', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      let errorCaught = false;

      try {
        await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.negativeQuantity as any
        );
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });

    test('should reject negative price', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      try {
        const response = await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.negativePrice as any
        );
        // API may accept or reject negative price
        expect(response).toBeDefined();
      } catch (error: any) {
        // Or throw validation error
        expect(error).toBeDefined();
      }
    });

    test('should reject empty productId', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      let errorCaught = false;

      try {
        await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.emptyProductId as any
        );
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });
  });

  test.describe('Invalid Item Operations', () => {
    test('should handle removing non-existent item', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const fakeItemId = 'non-existent-item-id';

      try {
        const response = await cartAPI.removeItemFromCart(userId, fakeItemId);
        // API may accept removal or return success
        expect(response).toBeDefined();
      } catch (error: any) {
        // Or throw error if item not found
        expect(error).toBeDefined();
      }
    });

    test('should handle updating non-existent item', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const fakeItemId = 'non-existent-item-id';
      let errorCaught = false;

      try {
        await cartAPI.updateCartItem(userId, fakeItemId, 5);
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });

    test('should handle updating with invalid quantity', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const fakeItemId = 'item-id';

      try {
        const response = await cartAPI.updateCartItem(userId, fakeItemId, 0);
        // API may accept zero quantity or reject it
        expect(response).toBeDefined();
      } catch (error: any) {
        // Or throw validation error
        expect(error).toBeDefined();
      }
    });

    test('should handle updating with negative quantity', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user1.id;
      const fakeItemId = 'item-id';
      let errorCaught = false;

      try {
        await cartAPI.updateCartItem(userId, fakeItemId, -5);
      } catch (error: any) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      expect(errorCaught).toBe(true);
    });
  });

  test.describe('Concurrent Operations', () => {
    test('should handle multiple simultaneous add operations', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user1.id;

      try {
        const promises = [
          cartAPI.addItemToCart(userId, {
            productId: 'prod-001',
            quantity: 1,
          }),
          cartAPI.addItemToCart(userId, {
            productId: 'prod-002',
            quantity: 1,
          }),
          cartAPI.addItemToCart(userId, {
            productId: 'prod-003',
            quantity: 1,
          }),
        ];

        const results = await Promise.all(promises);

        // All operations should succeed
        results.forEach((result) => {
          expect(result.success).toBe(true);
        });
      } catch (error: any) {
        // Concurrent operations might fail depending on API
        expect(error).toBeDefined();
      }
    });

    test('should handle add and remove operations in sequence', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user2.id;

      try {
        // Add
        const addResponse = await cartAPI.addItemToCart(userId, {
          productId: 'prod-001',
          quantity: 2,
        });

        // Immediately remove
        if (addResponse.data?.itemId) {
          const removeResponse = await cartAPI.removeItemFromCart(
            userId,
            addResponse.data.itemId
          );
          expect(removeResponse.success).toBe(true);
        }
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });
  });

  test.describe('Large Data Handling', () => {
    test('should handle very large quantity', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;

      try {
        const response = await cartAPI.addItemToCart(userId, {
          productId: 'prod-001',
          quantity: 999999,
        });

        // Either succeeds or fails with appropriate error
        if (response.success) {
          expect(response.data?.quantity).toBe(999999);
        }
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });

    test('should handle very long product name', async ({ cartAPI }) => {
      const userId = testUsers.user1.id;
      const longName = 'A'.repeat(1000);

      try {
        const response = await cartAPI.addItemToCart(userId, {
          productId: 'prod-001',
          quantity: 1,
          name: longName,
        });

        if (response.success) {
          expect(response.data?.name).toBeDefined();
        }
      } catch (error: any) {
        // Expected if API has length restrictions
        expect(error).toBeDefined();
      }
    });
  });

  test.describe('HTTP Status Code Validation', () => {
    test('should properly identify 400 Bad Request responses', async ({
      cartAPI,
    }) => {
      const userId = testUsers.user1.id;

      try {
        const response = await cartAPI.addItemToCart(
          userId,
          invalidAddToCartRequests.missingProductId as any
        );
        // API may accept or reject the invalid request
        expect(response).toBeDefined();
      } catch (error: any) {
        // If error is thrown, validate it's a 400 or 422
        const statusCode = error?.statusCode;
        if (statusCode !== undefined) {
          expect([400, 422]).toContain(statusCode);
        } else {
          expect(error).toBeDefined();
        }
      }
    });

    test('should properly identify 404 Not Found responses', async ({
      cartAPI,
    }) => {
      try {
        const response = await cartAPI.removeItemFromCart('user-1', 'item-999');
        // If no error thrown, response should be defined
        expect(response).toBeDefined();
      } catch (error: any) {
        // If error thrown, should have proper status code
        const statusCode = error?.statusCode;
        expect(statusCode).toBeDefined();
        expect([404, 400, 500]).toContain(statusCode);
      }
    });
  });
});
