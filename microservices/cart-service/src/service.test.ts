import { CartService } from './service';
import { ProductServiceClient } from './product-client';
import { Product } from './types';

// Mock the ProductServiceClient module
jest.mock('./product-client');

// Mock uuid
jest.mock('uuid', () => ({
  v4: jest.fn(() => 'mock-uuid-12345'),
}));

describe('CartService', () => {
  let cartService: CartService;
  let mockGetProduct: jest.Mock;

  // Test data factory
  const mockProduct1: Product = {
    id: 'product-1',
    name: 'Product 1',
    price: 10.99,
    description: 'First test product',
    image: 'img1.jpg',
    dataAiHint: 'hint1',
  };

  const mockProduct2: Product = {
    id: 'product-2',
    name: 'Product 2',
    price: 25.50,
    description: 'Second test product',
    image: 'img2.jpg',
    dataAiHint: 'hint2',
  };

  const mockProduct3: Product = {
    id: 'product-3',
    name: 'Product 3',
    price: 99.99,
    description: 'Premium test product',
    image: 'img3.jpg',
    dataAiHint: 'hint3',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    cartService = new CartService();
    const MockedProductServiceClient = ProductServiceClient as jest.MockedClass<
      typeof ProductServiceClient
    >;
    mockGetProduct = MockedProductServiceClient.prototype.getProduct as jest.Mock;
  });

  describe('getOrCreateCart', () => {
    test('should create a new cart when user cart does not exist', async () => {
      // Arrange
      const userId = 'user-1';

      // Act
      const cart = await cartService.getOrCreateCart(userId);

      // Assert
      expect(cart).toBeDefined();
      expect(cart.userId).toBe(userId);
      expect(cart.items).toEqual([]);
      expect(cart.totalAmount).toBe(0);
      expect(cart.totalItems).toBe(0);
      expect(cart.id).toBe('mock-uuid-12345');
      expect(cart.createdAt).toBeInstanceOf(Date);
      expect(cart.updatedAt).toBeInstanceOf(Date);
    });

    test('should return existing cart when user cart already exists', async () => {
      // Arrange
      const userId = 'user-1';
      const firstCart = await cartService.getOrCreateCart(userId);
      const cartIdBeforeSecondCall = firstCart.id;

      // Act
      const secondCart = await cartService.getOrCreateCart(userId);

      // Assert
      expect(secondCart.id).toBe(cartIdBeforeSecondCall);
      expect(secondCart.userId).toBe(userId);
    });

    test('should maintain separate carts for different users', async () => {
      // Arrange
      const userId1 = 'user-1';
      const userId2 = 'user-2';

      // Act
      const cart1 = await cartService.getOrCreateCart(userId1);
      const cart2 = await cartService.getOrCreateCart(userId2);

      // Assert
      expect(cart1.userId).toBe(userId1);
      expect(cart2.userId).toBe(userId2);
      expect(cart1.userId).not.toBe(cart2.userId);
    });
  });

  describe('addToCart', () => {
    test('should add a new item to an empty cart with default quantity of 1', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      // Act
      const cart = await cartService.addToCart(userId, mockProduct1.id);

      // Assert
      expect(cart.items).toHaveLength(1);
      expect(cart.items[0]).toMatchObject({
        id: mockProduct1.id,
        name: mockProduct1.name,
        price: mockProduct1.price,
        quantity: 1,
      });
      expect(cart.totalItems).toBe(1);
      expect(cart.totalAmount).toBe(mockProduct1.price);
    });

    test('should add a new item to cart with specified quantity', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      const quantity = 5;
      mockGetProduct.mockResolvedValue(product);

      // Act
      const cart = await cartService.addToCart(userId, product.id, quantity);

      // Assert
      expect(cart.items).toHaveLength(1);
      expect(cart.items[0].quantity).toBe(quantity);
      expect(cart.totalItems).toBe(quantity);
      expect(cart.totalAmount).toBe(50);
    });

    test('should increment quantity when adding same product multiple times', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      mockGetProduct.mockResolvedValue(product);

      // Act
      await cartService.addToCart(userId, product.id, 2);
      const cart = await cartService.addToCart(userId, product.id, 3);

      // Assert
      expect(cart.items).toHaveLength(1);
      expect(cart.items[0].quantity).toBe(5);
      expect(cart.totalItems).toBe(5);
      expect(cart.totalAmount).toBe(50);
    });

    test('should throw error when product does not exist', async () => {
      // Arrange
      const userId = 'user-1';
      const productId = 'non-existent';
      mockGetProduct.mockResolvedValue(null);

      // Act & Assert
      await expect(cartService.addToCart(userId, productId)).rejects.toThrow('Product not found');
    });

    test('should add multiple different products to cart', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      // Act
      await cartService.addToCart(userId, mockProduct1.id, 2);
      const cart = await cartService.addToCart(userId, mockProduct2.id, 1);

      // Assert
      expect(cart.items).toHaveLength(2);
      expect(cart.totalItems).toBe(3);
      expect(cart.totalAmount).toBe(40);
    });

    test('should update cart updatedAt timestamp when adding item', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);
      const initialCart = await cartService.getOrCreateCart(userId);
      const initialUpdatedAt = initialCart.updatedAt.getTime();

      // Act
      await new Promise(resolve => setTimeout(resolve, 10));
      const updatedCart = await cartService.addToCart(userId, mockProduct1.id);

      // Assert
      expect(updatedCart.updatedAt.getTime()).toBeGreaterThan(initialUpdatedAt);
    });

    test('should call ProductServiceClient getProduct with correct productId', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      // Act
      await cartService.addToCart(userId, mockProduct1.id);

      // Assert
      expect(mockGetProduct).toHaveBeenCalledWith(mockProduct1.id);
    });

    test('should handle decimal prices correctly with rounding', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 19.99 };
      mockGetProduct.mockResolvedValue(product);

      // Act
      const cart = await cartService.addToCart(userId, product.id, 3);

      // Assert
      expect(cart.totalAmount).toBe(59.97);
    });
  });

  describe('removeFromCart', () => {
    test('should remove item from cart with multiple items', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      await cartService.addToCart(userId, mockProduct1.id, 1);
      await cartService.addToCart(userId, mockProduct2.id, 1);

      // Act
      const cartAfter = await cartService.removeFromCart(userId, mockProduct1.id);

      // Assert
      expect(cartAfter.items).toHaveLength(1);
      expect(cartAfter.items[0].id).toBe(mockProduct2.id);
      expect(cartAfter.totalItems).toBe(1);
      expect(cartAfter.totalAmount).toBe(20);
    });

    test('should remove all items when last item is removed', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      await cartService.addToCart(userId, mockProduct1.id, 5);

      // Act
      const cart = await cartService.removeFromCart(userId, mockProduct1.id);

      // Assert
      expect(cart.items).toHaveLength(0);
      expect(cart.totalItems).toBe(0);
      expect(cart.totalAmount).toBe(0);
    });

    test('should not throw error when removing non-existent product from cart', async () => {
      // Arrange
      const userId = 'user-1';
      await cartService.getOrCreateCart(userId);

      // Act & Assert
      expect(async () => {
        await cartService.removeFromCart(userId, 'non-existent-product-id');
      }).not.toThrow();
    });

    test('should update cart updatedAt timestamp when removing item', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      await cartService.addToCart(userId, mockProduct1.id);
      const cartBefore = await cartService.getCart(userId);
      const updatedAtBefore = cartBefore.updatedAt.getTime();

      // Act
      await new Promise(resolve => setTimeout(resolve, 10));
      const cartAfter = await cartService.removeFromCart(userId, mockProduct1.id);

      // Assert
      expect(cartAfter.updatedAt.getTime()).toBeGreaterThan(updatedAtBefore);
    });

    test('should recalculate totals correctly after item removal', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 15 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 25 });

      await cartService.addToCart(userId, mockProduct1.id, 2);
      await cartService.addToCart(userId, mockProduct2.id, 1);

      // Act
      const cartAfter = await cartService.removeFromCart(userId, mockProduct1.id);

      // Assert
      expect(cartAfter.totalItems).toBe(1);
      expect(cartAfter.totalAmount).toBe(25);
    });
  });

  describe('updateCartItem', () => {
    test('should update quantity of existing cart item', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 2);

      // Act
      const cart = await cartService.updateCartItem(userId, product.id, 5);

      // Assert
      expect(cart.items[0].quantity).toBe(5);
      expect(cart.totalItems).toBe(5);
      expect(cart.totalAmount).toBe(50);
    });

    test('should remove item when quantity is set to zero', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 3);

      // Act
      const cart = await cartService.updateCartItem(userId, product.id, 0);

      // Assert
      expect(cart.items).toHaveLength(0);
      expect(cart.totalItems).toBe(0);
      expect(cart.totalAmount).toBe(0);
    });

    test('should remove item when quantity is set to negative number', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 3);

      // Act
      const cart = await cartService.updateCartItem(userId, product.id, -5);

      // Assert
      expect(cart.items).toHaveLength(0);
      expect(cart.totalItems).toBe(0);
      expect(cart.totalAmount).toBe(0);
    });

    test('should throw error when updating quantity of non-existent item', async () => {
      // Arrange
      const userId = 'user-1';
      const productId = 'non-existent';
      await cartService.getOrCreateCart(userId);

      // Act & Assert
      await expect(cartService.updateCartItem(userId, productId, 5)).rejects.toThrow(
        'Item not found in cart'
      );
    });

    test('should update cart updatedAt timestamp when updating item quantity', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      await cartService.addToCart(userId, mockProduct1.id, 1);
      const cartBefore = await cartService.getCart(userId);
      const updatedAtBefore = cartBefore.updatedAt.getTime();

      // Act
      await new Promise(resolve => setTimeout(resolve, 10));
      const cartAfter = await cartService.updateCartItem(userId, mockProduct1.id, 3);

      // Assert
      expect(cartAfter.updatedAt.getTime()).toBeGreaterThan(updatedAtBefore);
    });

    test('should recalculate totals correctly when updating quantity', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 12.50 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 2);

      // Act
      const cart = await cartService.updateCartItem(userId, product.id, 4);

      // Assert
      expect(cart.totalItems).toBe(4);
      expect(cart.totalAmount).toBe(50.0);
    });

    test('should handle updating quantity of one item with multiple items in cart', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      await cartService.addToCart(userId, mockProduct1.id, 2);
      await cartService.addToCart(userId, mockProduct2.id, 1);

      // Act
      const cart = await cartService.updateCartItem(userId, mockProduct1.id, 5);

      // Assert
      expect(cart.items).toHaveLength(2);
      expect(cart.totalItems).toBe(6);
      expect(cart.totalAmount).toBe(70);
    });

    test('should update to quantity 1 when specified', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 10);

      // Act
      const cart = await cartService.updateCartItem(userId, product.id, 1);

      // Assert
      expect(cart.items[0].quantity).toBe(1);
      expect(cart.totalItems).toBe(1);
      expect(cart.totalAmount).toBe(10);
    });
  });

  describe('clearCart', () => {
    test('should clear all items from cart', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      await cartService.addToCart(userId, mockProduct1.id, 2);
      await cartService.addToCart(userId, mockProduct2.id, 3);

      // Act
      const cart = await cartService.clearCart(userId);

      // Assert
      expect(cart.items).toHaveLength(0);
      expect(cart.totalAmount).toBe(0);
      expect(cart.totalItems).toBe(0);
    });

    test('should clear empty cart without errors', async () => {
      // Arrange
      const userId = 'user-1';
      await cartService.getOrCreateCart(userId);

      // Act
      const cart = await cartService.clearCart(userId);

      // Assert
      expect(cart.items).toHaveLength(0);
      expect(cart.totalAmount).toBe(0);
      expect(cart.totalItems).toBe(0);
    });

    test('should update cart updatedAt timestamp when clearing', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      await cartService.addToCart(userId, mockProduct1.id, 1);
      const cartBefore = await cartService.getCart(userId);
      const updatedAtBefore = cartBefore.updatedAt.getTime();

      // Act
      await new Promise(resolve => setTimeout(resolve, 10));
      const cartAfter = await cartService.clearCart(userId);

      // Assert
      expect(cartAfter.updatedAt.getTime()).toBeGreaterThan(updatedAtBefore);
    });

    test('should preserve cart userId and id after clearing', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      const cartBeforeClear = await cartService.addToCart(userId, mockProduct1.id, 1);
      const cartId = cartBeforeClear.id;

      // Act
      const cartAfterClear = await cartService.clearCart(userId);

      // Assert
      expect(cartAfterClear.userId).toBe(userId);
      expect(cartAfterClear.id).toBe(cartId);
    });
  });

  describe('getCart', () => {
    test('should return existing cart for user', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct.mockResolvedValue(mockProduct1);

      const cartAdded = await cartService.addToCart(userId, mockProduct1.id, 2);

      // Act
      const retrievedCart = await cartService.getCart(userId);

      // Assert
      expect(retrievedCart.userId).toBe(userId);
      expect(retrievedCart.items).toHaveLength(1);
      expect(retrievedCart.totalItems).toBe(2);
    });

    test('should create and return new cart if user has no cart', async () => {
      // Arrange
      const userId = 'user-1';

      // Act
      const cart = await cartService.getCart(userId);

      // Assert
      expect(cart.userId).toBe(userId);
      expect(cart.items).toHaveLength(0);
      expect(cart.totalItems).toBe(0);
      expect(cart.totalAmount).toBe(0);
    });
  });

  describe('getCartSummary', () => {
    test('should return correct summary for empty cart', async () => {
      // Arrange
      const userId = 'user-1';

      // Act
      const summary = await cartService.getCartSummary(userId);

      // Assert
      expect(summary.totalItems).toBe(0);
      expect(summary.totalAmount).toBe(0);
      expect(summary.itemCount).toBe(0);
    });

    test('should return correct summary for cart with single item', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 25 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 3);

      // Act
      const summary = await cartService.getCartSummary(userId);

      // Assert
      expect(summary.totalItems).toBe(3);
      expect(summary.totalAmount).toBe(75);
      expect(summary.itemCount).toBe(1);
    });

    test('should return correct summary for cart with multiple items', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 })
        .mockResolvedValueOnce({ ...mockProduct3, price: 30 });

      await cartService.addToCart(userId, mockProduct1.id, 2);
      await cartService.addToCart(userId, mockProduct2.id, 1);
      await cartService.addToCart(userId, mockProduct3.id, 3);

      // Act
      const summary = await cartService.getCartSummary(userId);

      // Assert
      expect(summary.totalItems).toBe(6);
      expect(summary.totalAmount).toBe(130);
      expect(summary.itemCount).toBe(3);
    });

    test('should have itemCount matching number of distinct products', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      await cartService.addToCart(userId, mockProduct1.id, 5);
      await cartService.addToCart(userId, mockProduct2.id, 10);

      // Act
      const summary = await cartService.getCartSummary(userId);

      // Assert
      expect(summary.itemCount).toBe(2);
      expect(summary.totalItems).toBe(15);
    });
  });

  describe('updateCartTotals (private method via public methods)', () => {
    test('should round totalAmount to 2 decimal places', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 10.33 };
      mockGetProduct.mockResolvedValue(product);

      // Act
      const cart = await cartService.addToCart(userId, product.id, 3);

      // Assert
      expect(cart.totalAmount).toBe(30.99);
    });

    test('should calculate correct totalItems from all quantities', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 15 });

      // Act
      await cartService.addToCart(userId, mockProduct1.id, 7);
      const cart = await cartService.addToCart(userId, mockProduct2.id, 3);

      // Assert
      expect(cart.totalItems).toBe(10);
    });

    test('should calculate totalAmount as sum of all item prices multiplied by quantities', async () => {
      // Arrange
      const userId = 'user-1';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 5.5 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 10.25 });

      // Act
      await cartService.addToCart(userId, mockProduct1.id, 2);
      const cart = await cartService.addToCart(userId, mockProduct2.id, 3);

      // Assert
      expect(cart.totalAmount).toBe(41.75);
    });

    test('should handle very small prices and round correctly', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 0.01 };
      mockGetProduct.mockResolvedValue(product);

      // Act
      const cart = await cartService.addToCart(userId, product.id, 333);

      // Assert
      expect(cart.totalAmount).toBe(3.33);
    });

    test('should reset totals to zero when cart is cleared', async () => {
      // Arrange
      const userId = 'user-1';
      const product = { ...mockProduct1, price: 100 };
      mockGetProduct.mockResolvedValue(product);

      await cartService.addToCart(userId, product.id, 5);

      // Act
      const cart = await cartService.clearCart(userId);

      // Assert
      expect(cart.totalItems).toBe(0);
      expect(cart.totalAmount).toBe(0);
    });
  });

  describe('Integration scenarios', () => {
    test('should handle complete workflow: add, update, remove, and clear', async () => {
      // Arrange
      const userId = 'user-workflow';
      mockGetProduct
        .mockResolvedValueOnce({ ...mockProduct1, price: 10 })
        .mockResolvedValueOnce({ ...mockProduct2, price: 20 });

      // Act - Add items
      let cart = await cartService.addToCart(userId, mockProduct1.id, 2);
      expect(cart.totalAmount).toBe(20);

      // Act - Add another item
      cart = await cartService.addToCart(userId, mockProduct2.id, 1);
      expect(cart.totalAmount).toBe(40);

      // Act - Update quantity
      cart = await cartService.updateCartItem(userId, mockProduct1.id, 3);
      expect(cart.totalAmount).toBe(50);

      // Act - Remove item
      cart = await cartService.removeFromCart(userId, mockProduct2.id);
      expect(cart.totalAmount).toBe(30);

      // Act - Clear cart
      cart = await cartService.clearCart(userId);
      expect(cart.totalAmount).toBe(0);

      // Assert
      expect(cart.items).toHaveLength(0);
    });
  });
});
