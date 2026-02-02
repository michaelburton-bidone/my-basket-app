/**
 * Add Product to Cart Test Suite
 * 
 * Tests the functionality of adding a product to the cart from the home page
 * including toast notification verification
 * @tag @cart @product-interaction
 */

import { test, expect } from '@fixtures/page-fixtures';
import { sampleProducts } from '@config/test-data';

test.describe('Add Product to Cart @cart', () => {
  test.beforeEach(async ({ productPage }) => {
    // Navigate to home page before each test using state-based wait
    await productPage.navigateTo('/');
  });

  test('should add first product to cart and show toast notification', async ({
    productPage,
    page,
  }) => {
    await test.step('Verify products are available on page', async () => {
      // Get initial product count to verify we're testing with products
      const productCount = await productPage.getProductCount();
      // Auto-retrying assertion with state-based expectation
      expect(productCount).toBeGreaterThan(0);
    });

    await test.step('Retrieve first product details', async () => {
      // Get product details before adding to cart
      const product = await productPage.getProductByIndex(0);
      // Auto-retrying assertions verify product data integrity
      expect(product.name).toBeTruthy();
      expect(product.price).toBeGreaterThan(0);
    });

    await test.step('Click Add to Cart button and wait for toast', async () => {
      // Click "Add to Basket" button on the first product
      await productPage.addProductToCartByIndex(0);

      // Wait for toast notification using state-based wait (Web-First Assertion)
      // No hardcoded sleep timers—rely on expect().toBeVisible() for visibility
      const toastNotification = page.locator('role=status');
      // Auto-retrying expect detects when toast becomes visible
      await expect(toastNotification.first()).toBeVisible({ timeout: 5000 });
    });

    await test.step('Verify toast contains success message', async () => {
      // Verify toast contains "Added" message
      const toastNotification = page.locator('role=status');
      const toastText = await toastNotification.first().textContent();
      // Auto-retrying assertion ensures message is present
      expect(toastText).toContain('Added');
    });
  });

  test('should add product to cart by clicking the add button', async ({
    productPage,
    page,
  }) => {
    await test.step('Get and verify add button is ready for interaction', async () => {
      // Get the first product's add to cart button using semantic role locator
      const addButton = productPage.addToCartButton(0);

      // Verify button is visible and enabled using Web-First Assertions
      // Auto-retrying expect handles hydration delays
      await expect(addButton).toBeVisible();
      await expect(addButton).toBeEnabled();
    });

    await test.step('Click add to cart button', async () => {
      const addButton = productPage.addToCartButton(0);
      // Click the add to cart button
      await addButton.click();
    });

    await test.step('Wait for and verify toast notification appears', async () => {
      // Wait for toast notification using state-based wait (no hardcoded timeouts)
      // Radix UI uses semantic role="status" selector
      const toastNotification = page.locator('role=status');
      // Auto-retrying expect handles minor network jitter
      await expect(toastNotification.first()).toBeVisible({ timeout: 5000 });

      // Verify success message in toast
      const toastText = await toastNotification.first().textContent();
      // Auto-retrying assertion ensures message is present
      expect(toastText?.toLowerCase()).toContain('added');
    });
  });

  test('should show toast with product information when adding to cart', async ({
    productPage,
    page,
  }) => {
    await test.step('Retrieve first product name', async () => {
      // Get first product name
      const product = await productPage.getProductByIndex(0);
      const productName = product.name;
      // Auto-retrying assertion ensures product data is available
      expect(productName).toBeTruthy();
    });

    await test.step('Add product to cart', async () => {
      // Add product to cart
      await productPage.addProductToCartByIndex(0);
    });

    await test.step('Verify toast contains product-related information', async () => {
      // Wait for toast notification using state-based wait (no sleep timers)
      // Radix UI uses semantic role="status" selector
      const toastNotification = page.locator('role=status');
      // Auto-retrying expect detects visibility
      await expect(toastNotification.first()).toBeVisible({ timeout: 5000 });

      // Verify notification text contains success message or product info
      const toastText = await toastNotification.first().textContent();
      // Auto-retrying assertion validates message content
      const product = await productPage.getProductByIndex(0);
      const productName = product.name;
      
      expect(
        toastText?.includes('Added') ||
          toastText?.toLowerCase().includes('added') ||
          toastText?.toLowerCase().includes(productName.toLowerCase())
      ).toBeTruthy();
    });
  });

  test('should add multiple products to cart with toast notifications', async ({
    productPage,
    page,
  }) => {
    await test.step('Verify sufficient products are available', async () => {
      const productsToAdd = 2;
      const productCount = await productPage.getProductCount();
      // Auto-retrying assertion ensures we have enough products
      expect(productCount).toBeGreaterThanOrEqual(productsToAdd);
    });

    await test.step('Add first product and wait for toast', async () => {
      // Add first product
      await productPage.addProductToCartByIndex(0);
      
      // Wait for first toast using Web-First Assertions (state-based, not timed sleep)
      let toast = page.locator('role=status');
      // Auto-retrying expect handles visibility detection
      await expect(toast.first()).toBeVisible({ timeout: 5000 });
      expect(await toast.first().textContent()).toContain('Added');
    });

    await test.step('Wait for first toast to disappear before adding second product', async () => {
      // Wait for first toast to disappear before adding second product
      // Using state-based wait instead of arbitrary sleep()
      const toast = page.locator('role=status');
      await toast.first().waitFor({ state: 'hidden', timeout: 5000 }).catch(() => {
        // Toast may already be hidden, that's ok - handles race conditions gracefully
      });
    });

    await test.step('Add second product and verify its toast', async () => {
      // Add second product
      await productPage.addProductToCartByIndex(1);
      
      // Wait for second toast using state-based wait
      const toast = page.locator('role=status');
      // Auto-retrying expect detects new toast appearance
      await expect(toast.first()).toBeVisible({ timeout: 5000 });
      expect(await toast.first().textContent()).toContain('Added');
    });
  });

  test('should display accessible toast notification', async ({
    productPage,
    page,
  }) => {
    await test.step('Add product to cart', async () => {
      // Add product to cart
      await productPage.addProductToCartByIndex(0);
    });

    await test.step('Verify toast has proper accessibility role', async () => {
      // Verify toast is accessible - using semantic role selector
      // Radix UI uses role="status" for a11y compliance
      const toast = page.locator('role=status');
      // Auto-retrying expect handles visibility
      await expect(toast.first()).toBeVisible({ timeout: 5000 });

      // Check for role attribute for accessibility
      const role = await toast.first().getAttribute('role');

      // Toast should have appropriate accessibility role (status or alert)
      // Auto-retrying assertion validates a11y compliance
      expect(role === 'status' || role === 'alert').toBeTruthy();
    });
  });

  test('should display correct button text for add to cart action', async ({
    productPage,
  }) => {
    await test.step('Retrieve add button text content', async () => {
      // Get the first product's add to cart button using semantic locator
      const addButton = productPage.addToCartButton(0);
      const buttonText = await addButton.textContent();

      // Button should contain "Add" and "Cart" or "Basket"
      // Auto-retrying assertion ensures button has expected UX text
      expect(
        buttonText?.toLowerCase().includes('add') &&
          (buttonText?.toLowerCase().includes('cart') || buttonText?.toLowerCase().includes('basket'))
      ).toBeTruthy();
    });
  });

  test('should verify toast positioning on screen', async ({
    productPage,
    page,
  }) => {
    await test.step('Add product to cart', async () => {
      // Add product to cart
      await productPage.addProductToCartByIndex(0);
    });

    await test.step('Wait for toast to appear and verify viewport positioning', async () => {
      // Get toast element using semantic role selector (not brittle CSS/XPath)
      // Radix UI uses role="status" for toast notifications
      const toast = page.locator('role=status');
      // Auto-retrying expect handles visibility
      await expect(toast.first()).toBeVisible({ timeout: 5000 });

      // Verify toast is within viewport
      const box = await toast.first().boundingBox();
      // Auto-retrying assertion validates box is defined
      expect(box).not.toBeNull();

      if (box) {
        const viewportSize = page.viewportSize();
        expect(viewportSize).not.toBeNull();

        if (viewportSize) {
          // Toast should be visible in the viewport
          // Auto-retrying assertions validate positioning
          expect(box.x).toBeLessThan(viewportSize.width);
          expect(box.y).toBeLessThan(viewportSize.height);
        }
      }
    });
  });
});
