/**
 * Utility functions for common wait operations
 */

import { Page, Locator } from '@playwright/test';

/**
 * Wait for an element to be visible with a maximum timeout
 */
export async function waitForElementVisible(
  locator: Locator,
  timeout: number = 10000
): Promise<void> {
  await locator.waitFor({ state: 'visible', timeout });
}

/**
 * Wait for an element to be hidden
 */
export async function waitForElementHidden(
  locator: Locator,
  timeout: number = 10000
): Promise<void> {
  await locator.waitFor({ state: 'hidden', timeout });
}

/**
 * Wait for multiple elements to appear
 */
export async function waitForElementsVisible(
  locators: Locator[],
  timeout: number = 10000
): Promise<void> {
  await Promise.all(
    locators.map((locator) =>
      locator.waitFor({ state: 'visible', timeout })
    )
  );
}

/**
 * Wait for element to be stable (not animating)
 */
export async function waitForElementStable(
  locator: Locator,
  timeout: number = 10000
): Promise<void> {
  await locator.waitFor({ state: 'stable', timeout });
}

/**
 * Wait for a condition to be true
 */
export async function waitForCondition(
  page: Page,
  condition: () => Promise<boolean>,
  timeout: number = 10000
): Promise<void> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await page.waitForTimeout(100);
  }
  throw new Error(`Condition not met within ${timeout}ms`);
}

/**
 * Wait for element count to match expected
 */
export async function waitForElementCount(
  locator: Locator,
  expectedCount: number,
  timeout: number = 10000
): Promise<void> {
  await locator.page().waitForFunction(
    ({ loc, count }) => loc.count() === count,
    { loc: locator, count: expectedCount },
    { timeout }
  );
}

/**
 * Wait for text to appear in element
 */
export async function waitForText(
  locator: Locator,
  text: string,
  timeout: number = 10000
): Promise<void> {
  await locator.page().waitForFunction(
    ({ loc, txt }) => loc.textContent().includes(txt),
    { loc: locator, txt: text },
    { timeout }
  );
}

/**
 * Wait for page load
 */
export async function waitForPageLoad(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle');
}

/**
 * Wait for API response
 */
export async function waitForResponse(
  page: Page,
  urlPattern: string | RegExp,
  timeout: number = 10000
): Promise<void> {
  await page.waitForResponse(urlPattern, { timeout });
}

/**
 * Wait for specific number of elements
 */
export async function waitForElements(
  locator: Locator,
  count: number,
  timeout: number = 10000
): Promise<Locator[]> {
  const elements: Locator[] = [];
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const currentCount = await locator.count();
    if (currentCount === count) {
      for (let i = 0; i < count; i++) {
        elements.push(locator.nth(i));
      }
      return elements;
    }
    await locator.page().waitForTimeout(100);
  }

  throw new Error(`Expected ${count} elements, but operation timed out after ${timeout}ms`);
}
