/**
 * Cart Page Object Model
 * 
 * Handles interactions with the shopping cart page
 */

import { Page } from '@playwright/test';
import { BasePage } from './BasePage';
import { CartItem, Cart } from '@types/test.types';

export class CartPage extends BasePage {
  // Cart items locators - items are dynamically created with item.id
  readonly cartItemContainer = this.page.locator('[data-testid^="cart-item-"]');

  constructor(page: Page) {
    super(page);
  }

  /**
   * Get cart item element at index
   */
  private getCartItemElement(index: number) {
    return this.cartItemContainer.nth(index);
  }

  /**
   * Get cart item name locator
   */
  private getCartItemName(index: number) {
    return this.getCartItemElement(index).locator('[data-testid^="cart-item-name-"]');
  }

  /**
   * Get cart item price locator
   */
  private getCartItemPrice(index: number) {
    return this.getCartItemElement(index).locator('[data-testid^="cart-item-price-"]');
  }

  /**
   * Get cart item quantity locator
   */
  private getCartItemQuantityLabel(index: number) {
    return this.getCartItemElement(index).locator('[data-testid^="cart-item-quantity-"]');
  }

  /**
   * Get quantity input for cart item
   */
  private getQuantityInput(index: number) {
    return this.getCartItemElement(index).locator('input[type="number"]');
  }

  /**
   * Get increase quantity button
   */
  private getIncreaseButton(index: number) {
    return this.getCartItemElement(index).getByRole('button', { name: /increase|\+/i });
  }

  /**
   * Get decrease quantity button
   */
  private getDecreaseButton(index: number) {
    return this.getCartItemElement(index).getByRole('button', { name: /decrease|\-/i });
  }

  /**
   * Get remove item button
   */
  private getRemoveItemButton(index: number) {
    return this.getCartItemElement(index).locator('[data-testid^="remove-item-"]');
  }

  /**
   * Get remove item button - for testing visibility
   */
  removeButton(index: number) {
    return this.getRemoveItemButton(index);
  }

  /**
   * Get increase quantity button - for testing visibility
   */
  increaseQuantityButton(index: number) {
    return this.getIncreaseButton(index);
  }

  /**
   * Get decrease quantity button - for testing visibility
   */
  decreaseQuantityButton(index: number) {
    return this.getDecreaseButton(index);
  }

  // Cart summary
  readonly subtotalAmount = this.page.locator('[data-testid="subtotal-amount"]');
  readonly taxAmount = this.page.locator('[data-testid="shipping-row"]');
  readonly totalAmount = this.page.locator('[data-testid="total-amount"]');
  readonly discountAmount = this.page.locator('[data-testid="cart-discount"]');

  // Action buttons
  readonly continueShoppingButton = this.page.locator('[data-testid="continue-shopping-button"]');
  readonly checkoutButton = this.page.locator('[data-testid="checkout-button"]');
  readonly clearCartButton = this.page.getByRole('button', { name: /clear cart/i });

  // Empty cart state
  readonly emptyCartMessage = this.page.locator('[data-testid="empty-cart-message"]');
  readonly emptyCartContainer = this.page.locator('[data-testid="cart-empty"]');

  // Promo code
  readonly promoCodeInput = this.page.locator('[data-testid="promo-code-input"]');
  readonly applyPromoButton = this.page.getByRole('button', { name: /apply/i });
  readonly promoError = this.page.locator('[data-testid="promo-error"]');

  // Loading state
  readonly loadingSpinner = this.page.locator('[data-testid="loading-spinner"]');

  /**
   * Navigate to cart page
   */
  async navigateTo(path: string = '/cart'): Promise<void> {
    await super.navigateTo(path);
    // Wait for cart to be loaded
    try {
      await this.waitForCartToLoad();
    } catch (e) {
      // If wait fails, cart might be in an error state - continue
      console.log('Cart load wait timed out, continuing...');
    }
  }

  /**
   * Get cart item count
   */
  async getCartItemCount(): Promise<number> {
    return this.cartItemContainer.count();
  }

  /**
   * Check if cart is empty
   */
  async isCartEmpty(): Promise<boolean> {
    return this.isVisible(this.emptyCartMessage);
  }

  /**
   * Get cart item by index
   */
  async getCartItemByIndex(index: number): Promise<CartItem> {
    const name = await this.getText(this.getCartItemName(index));
    const priceText = await this.getText(this.getCartItemPrice(index));
    const quantityText = await this.getText(this.getCartItemQuantityLabel(index));
    const subtotalText = await this.getText(this.getCartItemPrice(index));

    const price = parseFloat(priceText.replace(/[^0-9.]/g, ''));
    const quantity = parseInt(quantityText.replace(/[^0-9]/g, ''), 10) || 1;
    const subtotal = parseFloat(subtotalText.replace(/[^0-9.]/g, ''));

    return {
      productId: index,
      name,
      price,
      quantity,
      subtotal,
    };
  }

  /**
   * Get all cart items
   */
  async getAllCartItems(): Promise<CartItem[]> {
    const count = await this.getCartItemCount();
    const items: CartItem[] = [];
    for (let i = 0; i < count; i++) {
      try {
        const item = await this.getCartItemByIndex(i);
        items.push(item);
      } catch (e) {
        // If we can't get item at index, cart might have fewer items than count
        break;
      }
    }
    return items;
  }

  /**
   * Get cart totals
   */
  async getCartTotals(): Promise<Cart> {
    // Don't iterate through all items - just get the totals
    // (The cart display only shows totals, not a full list for this method)
    const subtotalText = await this.getText(this.subtotalAmount);
    const totalText = await this.getText(this.totalAmount);
    const count = await this.getCartItemCount();
    const items: CartItem[] = [];

    const subtotal = parseFloat(subtotalText.replace(/[^0-9.]/g, ''));
    const total = parseFloat(totalText.replace(/[^0-9.]/g, ''));
    const tax = 0; // No tax in this store, shipping is free

    return {
      items,
      subtotal,
      tax,
      total,
      isEmpty: await this.isCartEmpty(),
    };
  }

  /**
   * Update quantity for item
   */
  async updateQuantity(index: number, quantity: number): Promise<void> {
    const input = this.getQuantityInput(index);
    await this.fill(input, String(quantity));
    await this.page.keyboard.press('Enter');
  }

  /**
   * Increase quantity for item
   */
  async increaseQuantity(index: number): Promise<void> {
    // Get current quantity and increase by 1
    const item = await this.getCartItemByIndex(index);
    await this.updateQuantity(index, item.quantity + 1);
  }

  /**
   * Decrease quantity for item
   */
  async decreaseQuantity(index: number): Promise<void> {
    const button = this.getDecreaseButton(index);
    await this.click(button);
  }

  /**
   * Remove item from cart
   */
  async removeItem(index: number): Promise<void> {
    const button = this.getRemoveItemButton(index);
    await this.click(button);
  }

  /**
   * Remove item by name
   */
  async removeItemByName(itemName: string): Promise<void> {
    const items = await this.getAllCartItems();
    const index = items.findIndex((item) =>
      item.name.toLowerCase().includes(itemName.toLowerCase())
    );
    if (index >= 0) {
      await this.removeItem(index);
    }
  }

  /**
   * Clear all items from cart by removing them one by one
   */
  async clearCart(): Promise<void> {
    const count = await this.getCartItemCount();
    // Remove items starting from the end to avoid index shifting
    for (let i = count - 1; i >= 0; i--) {
      try {
        await this.removeItem(i);
      } catch (e) {
        // Item already removed or not found
      }
    }
  }

  /**
   * Apply promo code
   */
  async applyPromoCode(code: string): Promise<void> {
    await this.fill(this.promoCodeInput, code);
    await this.click(this.applyPromoButton);
  }

  /**
   * Get promo error message
   */
  async getPromoErrorMessage(): Promise<string | null> {
    const isVisible = await this.isVisible(this.promoError);
    if (isVisible) {
      return this.getText(this.promoError);
    }
    return null;
  }

  /**
   * Proceed to checkout
   */
  async proceedToCheckout(): Promise<void> {
    await this.click(this.checkoutButton);
  }

  /**
   * Continue shopping
   */
  async continueShopping(): Promise<void> {
    await this.click(this.continueShoppingButton);
  }

  /**
   * Assert product in cart
   */
  async assertProductInCart(productName: string): Promise<void> {
    const items = await this.getAllCartItems();
    const found = items.some((item) =>
      item.name.toLowerCase().includes(productName.toLowerCase())
    );
    if (!found) {
      throw new Error(`Product "${productName}" not found in cart`);
    }
  }

  /**
   * Assert cart total is correct
   */
  async assertCartTotal(expectedTotal: number): Promise<void> {
    const cart = await this.getCartTotals();
    if (Math.abs(cart.total - expectedTotal) > 0.01) {
      throw new Error(`Cart total mismatch. Expected: ${expectedTotal}, Got: ${cart.total}`);
    }
  }

  /**
   * Check if checkout button is enabled
   */
  async isCheckoutButtonEnabled(): Promise<boolean> {
    return this.isEnabled(this.checkoutButton);
  }

  /**
   * Get discount amount
   */
  async getDiscountAmount(): Promise<number> {
    const discountText = await this.getText(this.discountAmount);
    return parseFloat(discountText.replace(/[^0-9.]/g, ''));
  }

  /**
   * Wait for cart to load
   */
  async waitForCartToLoad(): Promise<void> {
    // Wait for either cart items or empty state
    const itemsVisible = this.cartItemContainer.first().isVisible({ timeout: 5000 }).catch(() => false);
    const emptyVisible = this.emptyCartMessage.isVisible({ timeout: 5000 }).catch(() => false);
    await Promise.race([itemsVisible, emptyVisible]);
  }
}
