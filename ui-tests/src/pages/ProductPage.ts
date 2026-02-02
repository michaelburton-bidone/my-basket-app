/**
 * Product Page Object Model
 * 
 * Handles interactions with the product listing page
 */

import { Page } from '@playwright/test';
import { BasePage } from './BasePage';
import { Product } from '@types/test.types';

export class ProductPage extends BasePage {
  // Product listing locators
  readonly productContainer = this.page.locator('[data-testid="product-container"]');
  readonly productItem = this.page.locator('[data-testid="product-item"]');
  
  // Get product name by index - ProductCard uses CardTitle which renders as h3
  getProductNameByIndex = (index: number) => 
    this.productItem.nth(index).locator('[class*="font-semibold"], [class*="title"]').first();
  
  // Get product price by index - matches the price display in CardContent
  getProductPriceByIndex = (index: number) => 
    this.productItem.nth(index).locator('p:has-text("$")').first();
  
  // Get product image by index
  getProductImageByIndex = (index: number) => 
    this.productItem.nth(index).locator('img').first();
  
  // Action buttons - use nth selector to get specific product's button
  readonly addToCartButton = (index: number) =>
    this.productItem.nth(index).getByRole('button', { name: /add.*cart/i });
  readonly viewDetailsButton = (productId: number) =>
    this.page.getByRole('button', { name: new RegExp(`details.*${productId}`, 'i') });
  
  // Filter and sort
  readonly filterButton = this.page.getByRole('button', { name: /filter/i });
  readonly sortDropdown = this.page.locator('[data-testid="sort-dropdown"]');
  readonly sortOption = (option: string) =>
    this.page.locator(`[data-testid="sort-${option.toLowerCase()}"]`);
  
  // Search
  readonly searchInput = this.page.getByPlaceholder(/search.*product/i);
  readonly searchButton = this.page.getByRole('button', { name: /search/i });
  
  // Pagination
  readonly paginationContainer = this.page.locator('[data-testid="pagination"]');
  readonly nextPageButton = this.page.getByRole('button', { name: /next/i });
  readonly previousPageButton = this.page.getByRole('button', { name: /previous/i });
  
  // Messages
  readonly emptyStateMessage = this.page.locator('[data-testid="empty-state"]');
  readonly loadingSpinner = this.page.locator('[data-testid="loading-spinner"]');

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to products page
   */
  async navigateTo(path: string = '/'): Promise<void> {
    await super.navigateTo(path);
    await this.waitForProductsToLoad();
  }

  /**
   * Wait for products to load
   */
  async waitForProductsToLoad(): Promise<void> {
    await this.page.waitForSelector('[data-testid="product-item"]', { state: 'visible' });
  }

  /**
   * Get total number of products displayed
   */
  async getProductCount(): Promise<number> {
    return this.productItem.count();
  }

  /**
   * Get product by index
   */
  async getProductByIndex(index: number): Promise<Product> {
    const item = this.productItem.nth(index);
    const name = await this.getText(this.getProductNameByIndex(index));
    const priceText = await this.getText(this.getProductPriceByIndex(index));
    const price = parseFloat(priceText.replace(/[^0-9.]/g, ''));

    return {
      id: index,
      name,
      price,
    };
  }

  /**
   * Add product to cart by index
   */
  async addProductToCartByIndex(index: number): Promise<void> {
    const button = this.addToCartButton(index);
    await this.click(button);
  }

  /**
   * Add product to cart by name
   */
  async addProductToCartByName(productName: string): Promise<void> {
    const productIndex = await this.getProductIndexByName(productName);
    await this.addProductToCartByIndex(productIndex);
  }

  /**
   * Get product index by name
   */
  async getProductIndexByName(productName: string): Promise<number> {
    const count = await this.getProductCount();
    for (let i = 0; i < count; i++) {
      const name = await this.getText(this.getProductNameByIndex(i));
      if (name.toLowerCase().includes(productName.toLowerCase())) {
        return i;
      }
    }
    throw new Error(`Product "${productName}" not found`);
  }

  /**
   * View product details
   */
  async viewProductDetails(index: number): Promise<void> {
    const button = this.viewDetailsButton(index);
    await this.click(button);
  }

  /**
   * Search for product
   */
  async searchProduct(searchTerm: string): Promise<void> {
    await this.fill(this.searchInput, searchTerm);
    await this.click(this.searchButton);
    await this.waitForProductsToLoad();
  }

  /**
   * Sort products
   */
  async sortProducts(sortOption: string): Promise<void> {
    await this.click(this.sortDropdown);
    await this.click(this.sortOption(sortOption));
    await this.waitForProductsToLoad();
  }

  /**
   * Apply filter
   */
  async applyFilter(filterName: string): Promise<void> {
    await this.click(this.filterButton);
    const filterOption = this.page.locator(`[data-testid="filter-${filterName.toLowerCase()}"]`);
    await this.click(filterOption);
    await this.waitForProductsToLoad();
  }

  /**
   * Go to next page
   */
  async goToNextPage(): Promise<void> {
    const isEnabled = await this.isEnabled(this.nextPageButton);
    if (isEnabled) {
      await this.click(this.nextPageButton);
      await this.waitForProductsToLoad();
    }
  }

  /**
   * Go to previous page
   */
  async goToPreviousPage(): Promise<void> {
    const isEnabled = await this.isEnabled(this.previousPageButton);
    if (isEnabled) {
      await this.click(this.previousPageButton);
      await this.waitForProductsToLoad();
    }
  }

  /**
   * Check if empty state is shown
   */
  async isEmptyStateShown(): Promise<boolean> {
    return this.isVisible(this.emptyStateMessage);
  }

  /**
   * Check if products are loading
   */
  async isLoading(): Promise<boolean> {
    return this.isVisible(this.loadingSpinner);
  }

  /**
   * Get all product names
   */
  async getAllProductNames(): Promise<string[]> {
    const count = await this.getProductCount();
    const names: string[] = [];
    for (let i = 0; i < count; i++) {
      const name = await this.getText(this.getProductNameByIndex(i));
      names.push(name);
    }
    return names;
  }

  /**
   * Get all product prices
   */
  async getAllProductPrices(): Promise<number[]> {
    const count = await this.getProductCount();
    const prices: number[] = [];
    for (let i = 0; i < count; i++) {
      const priceText = await this.getText(this.getProductPriceByIndex(i));
      const price = parseFloat(priceText.replace(/[^0-9.]/g, ''));
      prices.push(price);
    }
    return prices;
  }
}
