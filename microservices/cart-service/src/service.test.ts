import { CartService } from './service';
import { ProductServiceClient } from './product-client';
import { Product, CartItem } from './types';

jest.mock('./product-client');

describe('CartService', () => {
  let cartService: CartService;
  let productClientMock: jest.Mocked<typeof ProductServiceClient>;

  const mockProduct: Product = {
    id: '1',
    name: 'Test Product',
    price: 10.99,
    description: 'A test product',
    image: 'test-image.jpg',
    dataAiHint: 'test hint',
  };

  const mockProduct2: Product = {
    id: '2',
    name: 'Another Product',
    price: 25.50,
    description: 'Another test product',
    image: 'test-image-2.jpg',
    dataAiHint: 'test hint 2',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    productClientMock = ProductServiceClient as jest.Mocked<typeof ProductServiceClient>;
    (productClientMock.prototype.getProduct as jest.Mock) = jest.fn();
    (productClientMock.prototype.getProducts as jest.Mock) = jest.fn();

    cartService = new CartService();
  });

  describe('addToCart', () => {
    it('should add a valid product to cart', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

      await cartService.addToCart('user1', '1', 1);

      const cart = await cartService.getCart('user1');
      expect(cart.items).toContainEqual(
        expect.objectContaining({ id: '1', quantity: 1 })
      );
      expect(cart.totalAmount).toBe(10.99);
    });

    it('should update quantity if product already in cart', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

      await cartService.addToCart('user1', '1', 1);
      await cartService.addToCart('user1', '1', 2);

      const cart = await cartService.getCart('user1');
      expect(cart.items[0].quantity).toBe(3);
      expect(cart.totalAmount).toBe(32.97);
    });
  });

  describe('updateCartItem', () => {
    it('should update cart item quantity and recalculate total', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

      await cartService.addToCart('user1', '1', 2);
      await cartService.updateCartItem('user1', '1', 5);

      const cart = await cartService.getCart('user1');
      expect(cart.items[0].quantity).toBe(5);
      expect(cart.totalAmount).toBe(54.95);
    });

    it('should remove item if quantity is 0', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

      await cartService.addToCart('user1', '1', 2);
      await cartService.updateCartItem('user1', '1', 0);

      const cart = await cartService.getCart('user1');
      expect(cart.items).toHaveLength(0);
      expect(cart.totalAmount).toBe(0.00);
    });
  });

  describe('floating point precision', () => {
    it('should handle 3 items at 10.99 with correct rounding', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

      await cartService.addToCart('user1', '1', 3);

      const cart = await cartService.getCart('user1');
      const total = cart.totalAmount;
      expect(total).toBe(32.97);
      expect(Number(total.toFixed(2))).toBe(32.97);
    });

    it('should round total to 2 decimal places for multiple items', async () => {
      (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct2);

      await cartService.addToCart('user1', '2', 3);

      const cart = await cartService.getCart('user1');
      const total = cart.totalAmount;
      expect(total).toBe(76.50);
      const decimalPlaces = total.toString().split('.')[1]?.length || 0;
      expect(decimalPlaces).toBeLessThanOrEqual(2);
    });

    it('should maintain decimal precision across multiple operations', async () => {
      // Setup mock to return different products based on productId
      (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
        return productId === '1' ? mockProduct : mockProduct2;
      });

      await cartService.addToCart('user1', '1', 1);
      await cartService.addToCart('user1', '2', 1);

      const cart = await cartService.getCart('user1');
      const total = cart.totalAmount;
      // 10.99 + 25.50 = 36.49
      expect(parseFloat(total.toFixed(2))).toBe(36.49);
    });
  });

  describe('removeFromCart', () => {
    describe('State Integrity & Item Removal', () => {
      it('should completely remove the item from the items array', async () => {
        // Setup: Seed cart with two distinct items
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        await cartService.addToCart('user1', '1', 2);
        await cartService.addToCart('user1', '2', 1);

        // Pre-removal state
        let cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(2);
        const removedItemIndex = cart.items.findIndex(item => item.id === '1');
        expect(removedItemIndex).toBeGreaterThanOrEqual(0);

        // Action: Remove one item
        await cartService.removeFromCart('user1', '1');

        // Assertion: Item is completely purged, not set to zero quantity
        cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(1);
        expect(cart.items).not.toContainEqual(
          expect.objectContaining({ id: '1' })
        );
        expect(cart.items).toContainEqual(
          expect.objectContaining({ id: '2' })
        );
      });

      it('should preserve remaining items when one is removed', async () => {
        // Setup: Seed cart with two distinct items
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        await cartService.addToCart('user1', '1', 2);
        await cartService.addToCart('user1', '2', 3);

        // Pre-removal state assertions
        let cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(2);
        const product2Item = cart.items.find(item => item.id === '2');
        expect(product2Item?.quantity).toBe(3);

        // Action: Remove product 1
        await cartService.removeFromCart('user1', '1');

        // Assertion: Product 2 remains unchanged
        cart = await cartService.getCart('user1');
        const remainingItem = cart.items.find(item => item.id === '2');
        expect(remainingItem).toBeDefined();
        expect(remainingItem?.quantity).toBe(3);
        expect(remainingItem?.price).toBe(25.50);
        expect(remainingItem?.name).toBe('Another Product');
      });
    });

    describe('Recalculation Logic - Total Items', () => {
      it('should update totalItems to reflect only remaining item quantities', async () => {
        // Setup: Seed cart with two items with different quantities
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        // Pre-removal: Item 1 qty=5, Item 2 qty=3 → totalItems=8
        await cartService.addToCart('user1', '1', 5);
        await cartService.addToCart('user1', '2', 3);

        let cart = await cartService.getCart('user1');
        expect(cart.totalItems).toBe(8);

        // Action: Remove Item 1 (quantity 5)
        await cartService.removeFromCart('user1', '1');

        // Assertion: totalItems should be 3 (only Item 2's quantity)
        cart = await cartService.getCart('user1');
        expect(cart.totalItems).toBe(3);
        expect(cart.items.reduce((sum, item) => sum + item.quantity, 0)).toBe(3);
      });

      it('should set totalItems to 0 when last item is removed', async () => {
        // Setup: Single item cart
        (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

        await cartService.addToCart('user1', '1', 5);

        let cart = await cartService.getCart('user1');
        expect(cart.totalItems).toBe(5);

        // Action: Remove the only item
        await cartService.removeFromCart('user1', '1');

        // Assertion: totalItems should be 0
        cart = await cartService.getCart('user1');
        expect(cart.totalItems).toBe(0);
      });
    });

    describe('Recalculation Logic - Total Amount', () => {
      it('should recalculate totalAmount correctly after removal', async () => {
        // Setup: Seed with two items
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        // Pre-removal: (10.99 × 2) + (25.50 × 1) = 21.98 + 25.50 = 47.48
        await cartService.addToCart('user1', '1', 2);
        await cartService.addToCart('user1', '2', 1);

        let cart = await cartService.getCart('user1');
        const expectedPreRemovalTotal = 47.48;
        expect(cart.totalAmount).toBe(expectedPreRemovalTotal);

        // Action: Remove Item 1
        await cartService.removeFromCart('user1', '1');

        // Assertion: totalAmount should equal only Item 2's price (25.50 × 1)
        cart = await cartService.getCart('user1');
        const expectedPostRemovalTotal = 25.50;
        expect(cart.totalAmount).toBe(expectedPostRemovalTotal);
      });

      it('should apply formula: totalAmount = Σ(price × quantity) for remaining items', async () => {
        // Setup: Three items with varying prices and quantities
        const mockProduct3: Product = {
          id: '3',
          name: 'Premium Product',
          price: 99.99,
          description: 'A premium product',
          image: 'premium.jpg',
          dataAiHint: 'premium hint',
        };

        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          if (productId === '1') return mockProduct;      // 10.99
          if (productId === '2') return mockProduct2;     // 25.50
          return mockProduct3;                            // 99.99
        });

        // Pre-removal: (10.99×1) + (25.50×2) + (99.99×1) = 10.99 + 51.00 + 99.99 = 161.98
        await cartService.addToCart('user1', '1', 1);
        await cartService.addToCart('user1', '2', 2);
        await cartService.addToCart('user1', '3', 1);

        let cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(161.98);

        // Action: Remove Item 2 (25.50 × 2 = 51.00)
        await cartService.removeFromCart('user1', '2');

        // Assertion: totalAmount = (10.99×1) + (99.99×1) = 110.98
        cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(110.98);

        // Verify formula (using toBeCloseTo for floating-point precision)
        const calculatedTotal = cart.items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
        expect(cart.totalAmount).toBeCloseTo(calculatedTotal, 2);
      });

      it('should set totalAmount to 0 when last item is removed', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

        await cartService.addToCart('user1', '1', 3);

        let cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(32.97);

        // Action: Remove the only item
        await cartService.removeFromCart('user1', '1');

        // Assertion: totalAmount should be 0
        cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(0);
      });

      it('should maintain 2 decimal precision after removal', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        // Add items with precision-sensitive amounts
        await cartService.addToCart('user1', '1', 3);  // 10.99 × 3 = 32.97
        await cartService.addToCart('user1', '2', 2);  // 25.50 × 2 = 51.00
        // Total: 83.97

        let cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(83.97);

        // Action: Remove first item
        await cartService.removeFromCart('user1', '1');

        // Assertion: totalAmount should be precisely 51.00
        cart = await cartService.getCart('user1');
        expect(cart.totalAmount).toBe(51.00);
        const decimalPlaces = cart.totalAmount.toString().split('.')[1]?.length || 0;
        expect(decimalPlaces).toBeLessThanOrEqual(2);
      });
    });

    describe('Array Structure & Length', () => {
      it('should decrement items array length by exactly one', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        // Setup: Two items
        await cartService.addToCart('user1', '1', 1);
        await cartService.addToCart('user1', '2', 1);

        let cart = await cartService.getCart('user1');
        const initialLength = cart.items.length;
        expect(initialLength).toBe(2);

        // Action: Remove one item
        await cartService.removeFromCart('user1', '1');

        // Assertion: Length decreased by exactly 1
        cart = await cartService.getCart('user1');
        expect(cart.items.length).toBe(initialLength - 1);
        expect(cart.items.length).toBe(1);
      });

      it('should maintain item order and data integrity for remaining items', async () => {
        const mockProduct3: Product = {
          id: '3',
          name: 'Third Product',
          price: 15.75,
          description: 'Third test product',
          image: 'test-image-3.jpg',
          dataAiHint: 'test hint 3',
        };

        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          if (productId === '1') return mockProduct;
          if (productId === '2') return mockProduct2;
          return mockProduct3;
        });

        // Setup: Three items in order
        await cartService.addToCart('user1', '1', 2);
        await cartService.addToCart('user1', '2', 3);
        await cartService.addToCart('user1', '3', 1);

        let cart = await cartService.getCart('user1');
        const allIds = cart.items.map(item => item.id);
        expect(allIds).toEqual(['1', '2', '3']);

        // Action: Remove middle item
        await cartService.removeFromCart('user1', '2');

        // Assertion: Remaining items are '1' and '3' with all properties intact
        cart = await cartService.getCart('user1');
        const remainingIds = cart.items.map(item => item.id);
        expect(remainingIds).toEqual(['1', '3']);

        const item1 = cart.items.find(item => item.id === '1');
        expect(item1).toMatchObject({
          id: '1',
          name: 'Test Product',
          price: 10.99,
          quantity: 2,
        });

        const item3 = cart.items.find(item => item.id === '3');
        expect(item3).toMatchObject({
          id: '3',
          name: 'Third Product',
          price: 15.75,
          quantity: 1,
        });
      });
    });

    describe('Edge Cases & State Management', () => {
      it('should handle removal of non-existent item gracefully', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

        // Setup: Single item
        await cartService.addToCart('user1', '1', 2);

        let cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(1);

        // Action: Remove non-existent item
        await cartService.removeFromCart('user1', '1');

        // Assertion: Cart is empty after removal
        cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(0);

        // Action: Try to remove non-existent item again
        await cartService.removeFromCart('user1', '999');

        // Assertion: Cart remains empty (no error, graceful handling)
        cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(0);
        expect(cart.totalItems).toBe(0);
        expect(cart.totalAmount).toBe(0);
      });

      it('should update the updatedAt timestamp on removal', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockResolvedValue(mockProduct);

        await cartService.addToCart('user1', '1', 1);

        let cart = await cartService.getCart('user1');
        const beforeRemoval = cart.updatedAt;

        // Small delay to ensure timestamp difference
        await new Promise(resolve => setTimeout(resolve, 10));

        await cartService.removeFromCart('user1', '1');

        cart = await cartService.getCart('user1');
        const afterRemoval = cart.updatedAt;

        expect(afterRemoval.getTime()).toBeGreaterThanOrEqual(beforeRemoval.getTime());
      });

      it('should persist cart state after removal', async () => {
        (productClientMock.prototype.getProduct as jest.Mock).mockImplementation((productId: string) => {
          return productId === '1' ? mockProduct : mockProduct2;
        });

        // Setup: Multi-item cart
        await cartService.addToCart('user1', '1', 2);
        await cartService.addToCart('user1', '2', 1);

        // Action: Remove item
        await cartService.removeFromCart('user1', '1');

        // Assertion 1: Verify state after removal
        let cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(1);
        expect(cart.totalItems).toBe(1);
        expect(cart.totalAmount).toBe(25.50);

        // Assertion 2: Retrieve same cart again to verify persistence
        cart = await cartService.getCart('user1');
        expect(cart.items).toHaveLength(1);
        expect(cart.items[0].id).toBe('2');
        expect(cart.totalItems).toBe(1);
        expect(cart.totalAmount).toBe(25.50);
      });
    });
  });
});
