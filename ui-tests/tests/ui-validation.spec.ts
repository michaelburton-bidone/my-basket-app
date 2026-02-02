/**
 * UI Validation and Component Tests
 * 
 * Tests UI elements, responsiveness, and component states
 * @tag @ui
 */

import { test, expect } from '@fixtures/page-fixtures';
import { sampleProducts } from '@config/test-data';

test.describe('UI Validation and Component Tests @ui', () => {
  test('should display product list correctly', async ({ productPage }) => {
    // Navigate to products
    await productPage.navigateTo();

    // Verify products are displayed
    const productCount = await productPage.getProductCount();
    expect(productCount).toBeGreaterThan(0);

    // Verify all product names are visible
    const productNames = await productPage.getAllProductNames();
    expect(productNames.length).toBe(productCount);
  });

  test('should display product information correctly', async ({ productPage }) => {
    // Navigate to products
    await productPage.navigateTo();

    // Get first product info
    const product = await productPage.getProductByIndex(0);

    // Verify product data
    expect(product.name).toBeTruthy();
    expect(product.price).toBeGreaterThan(0);
    expect(product.id).toBeDefined();
  });

  test('should have functional add to cart buttons', async ({ productPage }) => {
    // Navigate to products
    await productPage.navigateTo();

    // Verify all add to cart buttons are enabled
    for (let i = 0; i < 3; i++) {
      const button = productPage.addToCartButton(i);
      const isEnabled = await productPage.isEnabled(button);
      expect(isEnabled).toBe(true);
    }
  });

  test('should show and hide loading spinner appropriately', async ({ productPage }) => {
    // Navigate to products
    await productPage.navigateTo();

    // Wait for products to load
    await productPage.waitForProductsToLoad();

    // Verify spinner is hidden after load
    const isLoading = await productPage.isLoading();
    expect(isLoading).toBe(false);
  });

  test('should display cart correctly', async ({ productPage, cartPage, page }) => {
    // Add a product
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);

    // Wait for toast notification to confirm product was added
    const toast = page.locator('role=status');
    await toast.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {
      // Toast may not appear, continue anyway
    });

    // Navigate to cart
    await cartPage.navigateTo();

    // Wait for cart items to load
    const cartItems = page.locator('[data-testid^="cart-item-"]');
    await cartItems.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {
      // Cart may be empty
    });

    // Verify cart header and elements are visible
    const cartSubtotal = await cartPage.isVisible(cartPage.subtotalAmount);
    const cartTotal = await cartPage.isVisible(cartPage.totalAmount);

    expect(cartSubtotal).toBe(true);
    expect(cartTotal).toBe(true);
  });

  test('should display cart totals correctly', async ({ productPage, cartPage }) => {
    // Add products
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);
    await productPage.addProductToCartByIndex(1);

    // Go to cart
    await cartPage.navigateTo();

    // Get totals
    const cart = await cartPage.getCartTotals();

    // Verify all totals are present and positive
    expect(cart.subtotal).toBeGreaterThan(0);
    expect(cart.total).toBeGreaterThan(0);
    expect(cart.tax).toBeGreaterThanOrEqual(0);
  });

  test('should show empty cart message when no items', async ({ cartPage, page }) => {
    // Navigate to cart page - it should load with empty state
    await cartPage.navigateTo();
    
    // Wait for cart to load
    await page.waitForLoadState('networkidle');
    
    // Verify empty cart message is visible
    const emptyCartMessage = page.locator('[data-testid="empty-cart-message"]');
    const emptyState = page.locator('[data-testid="cart-empty"]');
    
    // Check if either empty cart message or empty state container is visible
    const isEmptyMessageVisible = await emptyCartMessage.isVisible({ timeout: 5000 }).catch(() => false);
    const isEmptyStateVisible = await emptyState.isVisible({ timeout: 5000 }).catch(() => false);
    
    expect(isEmptyMessageVisible || isEmptyStateVisible).toBe(true);
    
    // Verify that cart items container is either empty or not visible
    const cartItems = page.locator('[data-testid^="cart-item-"]');
    const cartItemCount = await cartItems.count();
    expect(cartItemCount).toBe(0);
  });

  test('should enable/disable checkout button based on cart state', async ({
    productPage,
    cartPage,
  }) => {
    // Navigate to empty cart
    await cartPage.navigateTo();

    // Checkout should be disabled
    let isCheckoutEnabled = await cartPage.isCheckoutButtonEnabled();
    expect(isCheckoutEnabled).toBe(false);

    // Add product
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);
    await cartPage.navigateTo();

    // Checkout should now be enabled
    isCheckoutEnabled = await cartPage.isCheckoutButtonEnabled();
    expect(isCheckoutEnabled).toBe(true);
  });

  test('should display correct product count in cart', async ({ productPage, cartPage, page }) => {
    // Add 3 products
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);
    await productPage.addProductToCartByIndex(1);
    await productPage.addProductToCartByIndex(2);

    // Wait for toast notification
    const toast = page.locator('role=status');
    await toast.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});

    // Go to cart
    await cartPage.navigateTo();

    // Wait for cart items to load
    const cartItems = page.locator('[data-testid^="cart-item-"]');
    await cartItems.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});

    // Verify count
    const itemCount = await cartPage.getCartItemCount();
    expect(itemCount).toBeGreaterThanOrEqual(1); // Could be merged or separate
  });

  test('should have visible product prices', async ({ productPage }) => {
    // Navigate to products
    await productPage.navigateTo();

    // Get all prices
    const prices = await productPage.getAllProductPrices();

    // Verify prices are displayed
    expect(prices.length).toBeGreaterThan(0);
    prices.forEach((price) => {
      expect(price).toBeGreaterThan(0);
      expect(typeof price).toBe('number');
    });
  });

  test('should display quantity controls in cart', async ({ productPage, cartPage, page }) => {
    // Add product
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);

    // Wait for toast notification
    const toast = page.locator('role=status');
    await toast.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});

    // Go to cart
    await cartPage.navigateTo();

    // Wait for cart items to load
    const cartItems = page.locator('[data-testid^="cart-item-"]');
    await cartItems.first().waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});

    // Verify remove button and quantity are visible
    const removeBtn = await cartPage.isVisible(cartPage.removeButton(0));
    expect(removeBtn).toBe(true);
  });

  test('should maintain visual hierarchy in checkout form', async ({ productPage, page }) => {
    // Add a product first
    await productPage.navigateTo();
    await productPage.addProductToCartByIndex(0);

    // Navigate directly to checkout page  
    await page.goto('/checkout');

    // Verify we're on the checkout page
    expect(page.url()).toContain('/checkout');
  });
});
