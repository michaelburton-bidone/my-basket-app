/**
 * Cart API Page Object Model
 * Handles all cart-related API operations
 */

import { APIRequestContext } from '@playwright/test';
import { BaseAPI } from './BaseAPI';
import {
  CartResponse,
  CartItemResponse,
  AddToCartRequest,
  UpdateCartItemRequest,
  Cart,
  CartItem,
  EmptyCartResponse,
} from '@types/index';
import { logger } from '@utils/index';

export class CartAPI extends BaseAPI {
  constructor(apiContext: APIRequestContext) {
    super(apiContext);
  }

  /**
   * Get cart items for a user
   */
  async getCartItems(userId: string): Promise<Cart> {
    logger.info(`Getting cart items for user: ${userId}`);

    const response = await this.get<Cart>(
      `/api/cart/${userId}`
    );

    return response.data!;
  }

  /**
   * Add item to cart
   */
  async addItemToCart(
    userId: string,
    item: AddToCartRequest
  ): Promise<Cart> {
    logger.info(`Adding item to cart for user: ${userId}`, item);

    const response = await this.post<Cart>(
      `/api/cart/${userId}/items`,
      item
    );

    return response.data!;
  }

  /**
   * Update cart item quantity
   */
  async updateCartItem(
    userId: string,
    productId: string,
    quantity: number
  ): Promise<Cart> {
    logger.info(
      `Updating cart item: ${productId} for user: ${userId} with quantity: ${quantity}`
    );

    const response = await this.put<Cart>(
      `/api/cart/${userId}/items/${productId}`,
      { quantity }
    );

    return response.data!;
  }

  /**
   * Remove item from cart
   */
  async removeItemFromCart(
    userId: string,
    productId: string
  ): Promise<Cart> {
    logger.info(
      `Removing item: ${productId} from cart for user: ${userId}`
    );

    const response = await this.delete<Cart>(
      `/api/cart/${userId}/items/${productId}`
    );

    return response.data!;
  }

  /**
   * Clear entire cart
   */
  async clearCart(userId: string): Promise<Cart> {
    logger.info(`Clearing cart for user: ${userId}`);

    const response = await this.delete<Cart>(
      `/api/cart/${userId}/items`
    );

    return response.data!;
  }

  /**
   * Get cart summary/stats
   */
  async getCartSummary(userId: string): Promise<{ itemCount: number; totalAmount: number }> {
    logger.info(`Getting cart summary for user: ${userId}`);

    const cart = await this.getCartItems(userId);

    return {
      itemCount: cart.totalItems || 0,
      totalAmount: cart.totalAmount || 0,
    };
  }

  /**
   * Check if item exists in cart
   */
  async itemExistsInCart(userId: string, productId: string): Promise<boolean> {
    logger.info(`Checking if product ${productId} exists in cart for user: ${userId}`);

    const cart = await this.getCartItems(userId);
    return cart.items?.some(item => item.id === productId) || false;
  }
}
