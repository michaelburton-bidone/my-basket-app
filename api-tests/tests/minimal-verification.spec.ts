/**
 * Minimal Verification Test
 * 
 * Quick sanity check to verify:
 * 1. Basic API connectivity
 * 2. Response structure
 * 3. Error handling
 */

import { test, expect } from '@fixtures/index';

test.describe('Minimal API Verification', () => {
  
  test('1. Health endpoint returns 200', async ({ apiContext }) => {
    const response = await apiContext.get('/api/health');
    expect(response.status()).toBe(200);
  });

  test('2. Create empty cart and retrieve', async ({ cartAPI }) => {
    const userId = `verify-${Date.now()}`;
    const response = await cartAPI.getCartItems(userId);
    
    expect(response.success).toBe(true);
    expect(response.data?.userId).toBe(userId);
    expect(response.data?.items).toEqual([]);
  });

  test('3. Get cart summary for empty cart', async ({ cartAPI }) => {
    const userId = `verify-summary-${Date.now()}`;
    
    // First get cart to initialize
    await cartAPI.getCartItems(userId);
    
    // Then get summary
    const response = await cartAPI.getCartSummary(userId);
    
    expect(response.success).toBe(true);
    expect(response.data?.itemCount).toBe(0);
    expect(response.data?.totalAmount).toBe(0);
  });

  test('4. Add item and verify cart', async ({ cartAPI }) => {
    const userId = `verify-add-${Date.now()}`;
    const item = {
      productId: '1', // Real product ID from product service
      quantity: 1
    };
    
    const response = await cartAPI.addItemToCart(userId, item);
    
    expect(response.success).toBe(true);
    expect(response.data?.items?.length).toBeGreaterThan(0);
  });

  test('5. Update item quantity', async ({ cartAPI }) => {
    const userId = `verify-update-${Date.now()}`;
    const item = {
      productId: '2', // Real product ID from product service
      quantity: 2
    };
    
    await cartAPI.addItemToCart(userId, item);
    const updated = await cartAPI.updateCartItem(userId, '2', 5);
    
    expect(updated.success).toBe(true);
    const foundItem = updated.data?.items?.find(i => i.id === '2');
    expect(foundItem?.quantity).toBe(5);
  });

  test('6. Remove item from cart', async ({ cartAPI }) => {
    const userId = `verify-remove-${Date.now()}`;
    const item = {
      productId: '3', // Real product ID from product service
      quantity: 1
    };
    
    await cartAPI.addItemToCart(userId, item);
    const removed = await cartAPI.removeItemFromCart(userId, '3');
    
    expect(removed.success).toBe(true);
    expect(removed.data?.items?.some(i => i.id === '3')).toBe(false);
  });

  test('7. Clear cart', async ({ cartAPI }) => {
    const userId = `verify-clear-${Date.now()}`;
    
    // Add item
    await cartAPI.addItemToCart(userId, {
      productId: '4', // Real product ID from product service
      quantity: 3
    });
    
    // Clear cart
    const cleared = await cartAPI.clearCart(userId);
    
    expect(cleared.success).toBe(true);
    expect(cleared.data?.items).toEqual([]);
    expect(cleared.data?.totalItems).toBe(0);
  });
});
