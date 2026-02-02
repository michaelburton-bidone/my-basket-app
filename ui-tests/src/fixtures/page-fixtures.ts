/**
 * Playwright Fixtures for Page Objects
 * 
 * Provides dependency injection of page objects and utilities into tests
 */

import { test as base, Page } from '@playwright/test';
import { ProductPage } from '@pages/ProductPage';
import { CartPage } from '@pages/CartPage';
import { CheckoutPage } from '@pages/CheckoutPage';

type PageFixtures = {
  page: Page;
  productPage: ProductPage;
  cartPage: CartPage;
  checkoutPage: CheckoutPage;
};

/**
 * Extend base test with custom fixtures
 */
export const test = base.extend<PageFixtures>({
  productPage: async ({ page }, use) => {
    const productPage = new ProductPage(page);
    await use(productPage);
  },

  cartPage: async ({ page }, use) => {
    const cartPage = new CartPage(page);
    await use(cartPage);
  },

  checkoutPage: async ({ page }, use) => {
    const checkoutPage = new CheckoutPage(page);
    await use(checkoutPage);
  },
});

export { expect } from '@playwright/test';
