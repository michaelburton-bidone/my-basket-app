/**
 * Base Page Object
 * 
 * Provides common functionality for all page objects including:
 * - Navigation
 * - Element interaction wrappers
 * - Common UI element handling
 * - Screenshots and debugging
 */

import { Page, Locator, expect } from '@playwright/test';
import { NavigationOptions, ClickOptions, TextInputOptions, WaitOptions } from '@types/page.types';

export class BasePage {
  readonly page: Page;
  readonly baseUrl: string;

  constructor(page: Page, baseUrl: string = '') {
    this.page = page;
    this.baseUrl = baseUrl || page.context().baseURL || '';
  }

  /**
   * Navigate to a specific URL
   */
  async navigateTo(path: string = '/', options?: NavigationOptions): Promise<void> {
    const url = this.baseUrl ? `${this.baseUrl}${path}` : path;
    await this.page.goto(url, options);
  }

  /**
   * Navigate to home page
   */
  async navigateToHome(): Promise<void> {
    await this.navigateTo('/');
  }

  /**
   * Click an element with error handling
   */
  async click(locator: Locator, options?: ClickOptions): Promise<void> {
    await this.waitForElement(locator, { state: 'visible' });
    await locator.click(options);
  }

  /**
   * Fill input field with text
   */
  async fill(locator: Locator, text: string, options?: TextInputOptions): Promise<void> {
    await this.waitForElement(locator, { state: 'visible' });
    await locator.fill(text, options);
  }

  /**
   * Type text character by character
   */
  async typeText(locator: Locator, text: string, options?: TextInputOptions): Promise<void> {
    await this.waitForElement(locator, { state: 'visible' });
    await locator.type(text, options);
  }

  /**
   * Get text content from an element
   */
  async getText(locator: Locator): Promise<string> {
    await this.waitForElement(locator);
    return locator.textContent() as Promise<string>;
  }

  /**
   * Get input value
   */
  async getInputValue(locator: Locator): Promise<string | null> {
    await this.waitForElement(locator);
    return locator.inputValue();
  }

  /**
   * Check if element is visible
   */
  async isVisible(locator: Locator): Promise<boolean> {
    try {
      return await locator.isVisible({ timeout: 5000 });
    } catch {
      return false;
    }
  }

  /**
   * Check if element is enabled
   */
  async isEnabled(locator: Locator): Promise<boolean> {
    try {
      return await locator.isEnabled({ timeout: 5000 });
    } catch {
      return false;
    }
  }

  /**
   * Check if element is checked (for checkboxes/radio)
   */
  async isChecked(locator: Locator): Promise<boolean> {
    try {
      return await locator.isChecked({ timeout: 5000 });
    } catch {
      return false;
    }
  }

  /**
   * Wait for element to be in a specific state
   */
  async waitForElement(locator: Locator, options?: WaitOptions): Promise<void> {
    const state = options?.state || 'attached';
    const timeout = options?.timeout || 10000;
    await locator.waitFor({ state, timeout });
  }

  /**
   * Wait for elements count
   */
  async waitForElementCount(locator: Locator, count: number, timeout: number = 10000): Promise<void> {
    await this.page.waitForFunction(
      async ({ elementCount, expectedCount }) => {
        const actualCount = await elementCount;
        return actualCount === expectedCount;
      },
      { elementCount: locator.count(), expectedCount: count },
      { timeout }
    );
  }

  /**
   * Get all text values from a list of elements
   */
  async getAllTextValues(locator: Locator): Promise<string[]> {
    return locator.allTextContents();
  }

  /**
   * Hover over an element
   */
  async hover(locator: Locator, options?: { force?: boolean; timeout?: number }): Promise<void> {
    await this.waitForElement(locator, { state: 'visible' });
    await locator.hover(options);
  }

  /**
   * Take a screenshot
   */
  async takeScreenshot(name: string): Promise<Buffer> {
    return this.page.screenshot({ path: `./test-results/${name}.png` });
  }

  /**
   * Get current URL
   */
  async getCurrentUrl(): Promise<string> {
    return this.page.url();
  }

  /**
   * Get page title
   */
  async getPageTitle(): Promise<string> {
    return this.page.title();
  }

  /**
   * Wait for URL to change
   */
  async waitForUrlChange(timeout: number = 10000): Promise<void> {
    const currentUrl = this.page.url();
    await this.page.waitForFunction(
      () => {
        return window.location.href !== currentUrl;
      },
      { timeout }
    );
  }

  /**
   * Wait for navigation
   */
  async waitForNavigation(): Promise<void> {
    await this.page.waitForLoadState('networkidle');
  }

  /**
   * Reload the page
   */
  async reload(): Promise<void> {
    await this.page.reload();
  }

  /**
   * Get element count
   */
  async getElementCount(locator: Locator): Promise<number> {
    return locator.count();
  }

  /**
   * Scroll element into view
   */
  async scrollIntoView(locator: Locator): Promise<void> {
    await locator.scrollIntoViewIfNeeded();
  }

  /**
   * Clear input field
   */
  async clearInput(locator: Locator): Promise<void> {
    await this.fill(locator, '');
  }

  /**
   * Press a key
   */
  async pressKey(key: string): Promise<void> {
    await this.page.keyboard.press(key);
  }

  /**
   * Get attribute value
   */
  async getAttribute(locator: Locator, attributeName: string): Promise<string | null> {
    return locator.getAttribute(attributeName);
  }

  /**
   * Close browser context (cleanup)
   */
  async cleanup(): Promise<void> {
    await this.page.context().close();
  }
}
