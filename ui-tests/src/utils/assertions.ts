/**
 * Custom assertion utilities
 */

import { expect, Locator } from '@playwright/test';

/**
 * Assert element is visible
 */
export async function assertElementVisible(locator: Locator, message?: string): Promise<void> {
  await expect(locator).toBeVisible();
}

/**
 * Assert element is hidden
 */
export async function assertElementHidden(locator: Locator, message?: string): Promise<void> {
  await expect(locator).toBeHidden();
}

/**
 * Assert element is enabled
 */
export async function assertElementEnabled(locator: Locator, message?: string): Promise<void> {
  await expect(locator).toBeEnabled();
}

/**
 * Assert element is disabled
 */
export async function assertElementDisabled(locator: Locator, message?: string): Promise<void> {
  await expect(locator).toBeDisabled();
}

/**
 * Assert element contains text
 */
export async function assertElementContainsText(
  locator: Locator,
  text: string,
  message?: string
): Promise<void> {
  await expect(locator).toContainText(text);
}

/**
 * Assert element has text (exact match)
 */
export async function assertElementHasText(
  locator: Locator,
  text: string,
  message?: string
): Promise<void> {
  await expect(locator).toHaveText(text);
}

/**
 * Assert element attribute value
 */
export async function assertElementHasAttribute(
  locator: Locator,
  attribute: string,
  value: string,
  message?: string
): Promise<void> {
  await expect(locator).toHaveAttribute(attribute, value);
}

/**
 * Assert element is checked (checkbox/radio)
 */
export async function assertElementChecked(locator: Locator, message?: string): Promise<void> {
  await expect(locator).toBeChecked();
}

/**
 * Assert element is not checked
 */
export async function assertElementNotChecked(locator: Locator, message?: string): Promise<void> {
  await expect(locator).not.toBeChecked();
}

/**
 * Assert element count
 */
export async function assertElementCount(
  locator: Locator,
  expectedCount: number,
  message?: string
): Promise<void> {
  await expect(locator).toHaveCount(expectedCount);
}

/**
 * Assert URL contains string
 */
export async function assertUrlContains(
  page: import('@playwright/test').Page,
  text: string,
  message?: string
): Promise<void> {
  await expect(page).toHaveURL(new RegExp(text));
}

/**
 * Assert URL equals
 */
export async function assertUrlEquals(
  page: import('@playwright/test').Page,
  url: string,
  message?: string
): Promise<void> {
  await expect(page).toHaveURL(url);
}

/**
 * Assert page title
 */
export async function assertPageTitle(
  page: import('@playwright/test').Page,
  title: string,
  message?: string
): Promise<void> {
  await expect(page).toHaveTitle(title);
}

/**
 * Assert page title contains
 */
export async function assertPageTitleContains(
  page: import('@playwright/test').Page,
  title: string,
  message?: string
): Promise<void> {
  await expect(page).toHaveTitle(new RegExp(title));
}

/**
 * Assert element is in viewport
 */
export async function assertElementInViewport(
  locator: Locator,
  message?: string
): Promise<void> {
  const box = await locator.boundingBox();
  const expected = box !== null;
  expect(expected).toBeTruthy();
}

/**
 * Assert element class
 */
export async function assertElementHasClass(
  locator: Locator,
  className: string,
  message?: string
): Promise<void> {
  await expect(locator).toHaveClass(new RegExp(className));
}

/**
 * Assert text appears in page
 */
export async function assertTextInPage(
  page: import('@playwright/test').Page,
  text: string
): Promise<void> {
  const locator = page.getByText(text);
  await expect(locator).toBeVisible();
}

/**
 * Assert value of input
 */
export async function assertInputValue(
  locator: Locator,
  expectedValue: string,
  message?: string
): Promise<void> {
  await expect(locator).toHaveValue(expectedValue);
}
