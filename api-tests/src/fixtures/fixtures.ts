/**
 * Playwright fixtures for API testing
 */

import { test as base, APIRequestContext } from '@playwright/test';
import { CartAPI } from '@pages/index';
import { configManager } from '@utils/index';

/**
 * Fixture providing API context
 */
export const test = base.extend<{
  apiContext: APIRequestContext;
  cartAPI: CartAPI;
}>({
  apiContext: async ({ playwright }, use) => {
    const context = await playwright.request.newContext({
      baseURL: configManager.getBaseURL(),
      extraHTTPHeaders: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
    });

    await use(context);
    await context.dispose();
  },

  cartAPI: async ({ apiContext }, use) => {
    const cartAPI = new CartAPI(apiContext);
    await use(cartAPI);
  },
});

export { expect } from '@playwright/test';
