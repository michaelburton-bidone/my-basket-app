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
  async getCartItems(userId: string): Promise<CartResponse> {
    logger.info(`Getting cart items for user: ${userId}`);

    const response = await this.get<Cart>(
      `/api/cart/${userId}`
    );

    return {
      success: response.status === 200,
      data: response.data!,
    };
  }

  /**
   * Add item to cart
   */
  async addItemToCart(
    userId: string,
    item: AddToCartRequest
  ): Promise<CartResponse> {
    logger.info(`Adding item to cart for user: ${userId}`, item);

    const response = await this.post<Cart>(
      `/api/cart/${userId}/items`,
      item
    );

    return {
      success: response.status === 200,
      data: response.data!,
    };
  }

  /**
   * Update cart item quantity
   */
  async updateCartItem(
    userId: string,
    productId: string,
    quantity: number
  ): Promise<CartResponse> {
    logger.info(
      `Updating cart item: ${productId} for user: ${userId} with quantity: ${quantity}`
    );

    const response = await this.put<Cart>(
      `/api/cart/${userId}/items/${productId}`,
      { quantity }
    );

    return {
      success: response.status === 200,
      data: response.data!,
    };
  }

  /**
   * Remove item from cart
   */
  async removeItemFromCart(
    userId: string,
    productId: string
  ): Promise<CartResponse> {
    logger.info(
      `Removing item: ${productId} from cart for user: ${userId}`
    );

    const response = await this.delete<Cart>(
      `/api/cart/${userId}/items/${productId}`
    );

    return {
      success: response.status === 200,
      data: response.data!,
    };
  }

  /**
   * Clear entire cart
   */
  async clearCart(userId: string): Promise<CartResponse> {
    logger.info(`Clearing cart for user: ${userId}`);

    const response = await this.delete<Cart>(
      `/api/cart/${userId}`
    );

    return {
      success: response.status === 200,
      data: response.data!,
    };
  }

  /**
   * Get cart summary/stats
   */
  async getCartSummary(userId: string): Promise<CartResponse> {
    logger.info(`Getting cart summary for user: ${userId}`);

    interface CartSummary {
      itemCount: number;
      totalItems: number;
      totalAmount: number;
    }

    const response = await this.get<CartSummary>(
      `/api/cart/${userId}/summary`
    );

    return {
      success: response.status === 200,
      data: response.data as any || { itemCount: 0, totalItems: 0, totalAmount: 0 },
    };
  }

  /**
   * Check if item exists in cart
   */
  async itemExistsInCart(userId: string, productId: string): Promise<boolean> {
    logger.info(`Checking if product ${productId} exists in cart for user: ${userId}`);

    const response = await this.getCartItems(userId);
    return response.data.items?.some(item => item.id === productId) || false;
  }
}

