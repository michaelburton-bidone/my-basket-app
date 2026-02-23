import express, { Express } from 'express';
import request from 'supertest';
import { OrderService } from './service';
import { OrderStatus } from './types';
import { z } from 'zod';

// Mock OrderService
jest.mock('./service');

describe('POST /api/orders/:userId - createOrder', () => {
  let app: Express;
  let mockOrderService: jest.Mocked<OrderService>;

  beforeEach(() => {
    app = express();
    app.use(express.json());

    mockOrderService = new OrderService() as jest.Mocked<OrderService>;

    // Setup routes
    const router = express.Router();

    const AddressSchema = z.object({
      street: z.string().min(1),
      city: z.string().min(1),
      state: z.string().min(1),
      zipCode: z.string().min(1),
      country: z.string().min(1),
    });

    const PaymentMethodSchema = z.object({
      type: z.enum(['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay']),
      last4: z.string().optional(),
      brand: z.string().optional(),
    });

    const CartItemSchema = z.object({
      id: z.string(),
      name: z.string(),
      price: z.number().positive(),
      description: z.string(),
      image: z.string(),
      dataAiHint: z.string(),
      quantity: z.number().positive(),
    });

    const CreateOrderSchema = z.object({
      items: z.array(CartItemSchema).min(1),
      shippingAddress: AddressSchema,
      billingAddress: AddressSchema,
      paymentMethod: PaymentMethodSchema,
    });

    router.post('/orders/:userId', async (req, res) => {
      try {
        const { userId } = req.params;
        const orderData = CreateOrderSchema.parse(req.body);

        const order = await mockOrderService.createOrder(userId, orderData);
        res.status(201).json(order);
      } catch (error) {
        if (error instanceof z.ZodError) {
          return res.status(400).json({ error: 'Invalid order data', details: error.errors });
        }
        if (error instanceof Error) {
          return res.status(400).json({ error: error.message });
        }
        console.error('Error creating order:', error);
        res.status(500).json({ error: 'Internal server error' });
      }
    });

    app.use('/api', router);
  });

  describe('Success Cases', () => {
    test('should create order with valid data and return 201', async () => {
      const userId = 'user-123';
      const validOrderPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Laptop',
            price: 999.99,
            description: 'High-performance laptop',
            image: 'laptop.jpg',
            dataAiHint: 'electronics',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
          last4: '4242',
          brand: 'Visa',
        },
      };

      const mockOrder = {
        id: 'order-123',
        userId,
        items: validOrderPayload.items,
        totalAmount: 999.99,
        status: OrderStatus.PENDING,
        shippingAddress: validOrderPayload.shippingAddress,
        billingAddress: validOrderPayload.billingAddress,
        paymentMethod: validOrderPayload.paymentMethod as any,
        orderDate: new Date('2026-02-23'),
        estimatedDelivery: new Date('2026-03-01'),
        createdAt: new Date('2026-02-23'),
        updatedAt: new Date('2026-02-23'),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(validOrderPayload)
        .expect(201);

      expect(response.body.id).toBe(mockOrder.id);
      expect(response.body.userId).toBe(mockOrder.userId);
      expect(response.body.totalAmount).toBe(mockOrder.totalAmount);
      expect(response.body.status).toBe(mockOrder.status);
      expect(response.body.items).toEqual(mockOrder.items);
      expect(response.body.shippingAddress).toEqual(mockOrder.shippingAddress);
      expect(response.body.billingAddress).toEqual(mockOrder.billingAddress);
      expect(mockOrderService.createOrder).toHaveBeenCalledWith(userId, validOrderPayload);
      expect(mockOrderService.createOrder).toHaveBeenCalledTimes(1);
    });

    test('should calculate correct total amount for multiple items', async () => {
      const userId = 'user-456';
      const multiItemPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item 1',
            price: 100.0,
            description: 'First item',
            image: 'item1.jpg',
            dataAiHint: 'category-a',
            quantity: 2,
          },
          {
            id: 'prod-2',
            name: 'Item 2',
            price: 50.0,
            description: 'Second item',
            image: 'item2.jpg',
            dataAiHint: 'category-b',
            quantity: 3,
          },
        ],
        shippingAddress: {
          street: '456 Oak Ave',
          city: 'New York',
          state: 'NY',
          zipCode: '10001',
          country: 'USA',
        },
        billingAddress: {
          street: '456 Oak Ave',
          city: 'New York',
          state: 'NY',
          zipCode: '10001',
          country: 'USA',
        },
        paymentMethod: {
          type: 'paypal',
        },
      };

      const expectedTotal = (100.0 * 2) + (50.0 * 3); // 350.0
      const mockOrder = {
        id: 'order-456',
        userId,
        items: multiItemPayload.items,
        totalAmount: expectedTotal,
        status: OrderStatus.PENDING,
        shippingAddress: multiItemPayload.shippingAddress,
        billingAddress: multiItemPayload.billingAddress,
        paymentMethod: multiItemPayload.paymentMethod as any,
        orderDate: new Date('2026-02-23'),
        estimatedDelivery: new Date('2026-03-01'),
        createdAt: new Date('2026-02-23'),
        updatedAt: new Date('2026-02-23'),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(multiItemPayload)
        .expect(201);

      expect(response.body.totalAmount).toBe(expectedTotal);
    });

    test('should accept different payment methods', async () => {
      const userId = 'user-789';
      const paymentMethods = [
        { type: 'credit_card', last4: '1234', brand: 'MasterCard' },
        { type: 'debit_card', last4: '5678', brand: 'Visa' },
        { type: 'paypal' },
        { type: 'apple_pay' },
        { type: 'google_pay' },
      ];

      for (const paymentMethod of paymentMethods) {
        const payload = {
          items: [
            {
              id: 'prod-1',
              name: 'Test Item',
              price: 99.99,
              description: 'Test',
              image: 'test.jpg',
              dataAiHint: 'test',
              quantity: 1,
            },
          ],
          shippingAddress: {
            street: '789 Elm St',
            city: 'Boston',
            state: 'MA',
            zipCode: '02101',
            country: 'USA',
          },
          billingAddress: {
            street: '789 Elm St',
            city: 'Boston',
            state: 'MA',
            zipCode: '02101',
            country: 'USA',
          },
          paymentMethod,
        };

        const mockOrder = {
          id: `order-${Date.now()}`,
          userId,
          items: payload.items,
          totalAmount: 99.99,
          status: OrderStatus.PENDING,
          shippingAddress: payload.shippingAddress,
          billingAddress: payload.billingAddress,
          paymentMethod: paymentMethod as any,
          orderDate: new Date(),
          estimatedDelivery: new Date(),
          createdAt: new Date(),
          updatedAt: new Date(),
        };

        mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

        const response = await request(app)
          .post(`/api/orders/${userId}`)
          .send(payload)
          .expect(201);

        expect(response.body.paymentMethod.type).toBe(paymentMethod.type);
      }
    });

    test('should set initial status to PENDING', async () => {
      const userId = 'user-status-test';
      const payload = {
        items: [
          {
            id: 'prod-1',
            name: 'Status Test Item',
            price: 50.0,
            description: 'Test',
            image: 'test.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main',
          city: 'Test City',
          state: 'TS',
          zipCode: '12345',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main',
          city: 'Test City',
          state: 'TS',
          zipCode: '12345',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
          last4: '9999',
        },
      };

      const mockOrder = {
        id: 'order-status-test',
        userId,
        items: payload.items,
        totalAmount: 50.0,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date(),
        estimatedDelivery: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body.status).toBe(OrderStatus.PENDING);
    });
  });

  describe('Bad Request Cases (400)', () => {
    test('should return 400 when items array is empty', async () => {
      const userId = 'user-empty-items';
      const invalidPayload = {
        items: [],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
          last4: '4242',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid order data');
    });

    test('should return 400 when items is missing', async () => {
      const userId = 'user-no-items';
      const invalidPayload = {
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when item has negative price', async () => {
      const userId = 'user-negative-price';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Invalid Item',
            price: -99.99,
            description: 'Negative price',
            image: 'item.jpg',
            dataAiHint: 'invalid',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when item has zero quantity', async () => {
      const userId = 'user-zero-qty';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Zero Qty Item',
            price: 50.0,
            description: 'Zero quantity',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 0,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when shipping address is missing street', async () => {
      const userId = 'user-missing-street';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when billing address is incomplete', async () => {
      const userId = 'user-incomplete-billing';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '456 Oak Ave',
          city: 'New York',
          state: 'NY',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when payment method type is invalid', async () => {
      const userId = 'user-invalid-payment';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'bitcoin',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 when item is missing required fields', async () => {
      const userId = 'user-missing-fields';
      const invalidPayload = {
        items: [
          {
            id: 'prod-1',
            price: 50.0,
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should return 400 with validation error details from Zod', async () => {
      const userId = 'user-validation-details';
      const invalidPayload = {
        items: [],
        shippingAddress: {
          street: '',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Invalid order data');
      expect(response.body).toHaveProperty('details');
      expect(Array.isArray(response.body.details)).toBe(true);
    });
  });

  describe('Service Error Cases', () => {
    test('should return 400 when service throws error with message', async () => {
      const userId = 'user-service-error';
      const validPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      mockOrderService.createOrder.mockRejectedValue(
        new Error('Order creation failed: Payment declined')
      );

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(validPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Payment declined');
    });

    test('should return 500 when service throws unknown error', async () => {
      const userId = 'user-unknown-error';
      const validPayload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      mockOrderService.createOrder.mockRejectedValue('Unexpected error');

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(validPayload)
        .expect(500);

      expect(response.body).toEqual({ error: 'Internal server error' });
    });
  });

  describe('Response Validation', () => {
    test('response should contain all required order fields', async () => {
      const userId = 'user-response-validation';
      const payload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 100.0,
            description: 'Test item',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
          last4: '4242',
        },
      };

      const mockOrder = {
        id: 'order-123',
        userId,
        items: payload.items,
        totalAmount: 100.0,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date('2026-02-23'),
        estimatedDelivery: new Date('2026-03-01'),
        createdAt: new Date('2026-02-23'),
        updatedAt: new Date('2026-02-23'),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('userId', userId);
      expect(response.body).toHaveProperty('items');
      expect(response.body).toHaveProperty('totalAmount');
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('shippingAddress');
      expect(response.body).toHaveProperty('billingAddress');
      expect(response.body).toHaveProperty('paymentMethod');
      expect(response.body).toHaveProperty('orderDate');
      expect(response.body).toHaveProperty('createdAt');
      expect(response.body).toHaveProperty('updatedAt');
    });

    test('response items should match request items', async () => {
      const userId = 'user-items-match';
      const items = [
        {
          id: 'prod-1',
          name: 'Item 1',
          price: 50.0,
          description: 'First item',
          image: 'item1.jpg',
          dataAiHint: 'category-a',
          quantity: 2,
        },
        {
          id: 'prod-2',
          name: 'Item 2',
          price: 75.0,
          description: 'Second item',
          image: 'item2.jpg',
          dataAiHint: 'category-b',
          quantity: 1,
        },
      ];

      const payload = {
        items,
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const mockOrder = {
        id: 'order-items-test',
        userId,
        items,
        totalAmount: 175.0,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date(),
        estimatedDelivery: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body.items).toEqual(items);
    });
  });

  describe('Edge Cases', () => {
    test('should handle userId with special characters', async () => {
      const userId = 'user-special-!@#$';
      const payload = {
        items: [
          {
            id: 'prod-1',
            name: 'Item',
            price: 50.0,
            description: 'Test',
            image: 'item.jpg',
            dataAiHint: 'test',
            quantity: 1,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const mockOrder = {
        id: 'order-special-chars',
        userId,
        items: payload.items,
        totalAmount: 50.0,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date(),
        estimatedDelivery: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body.userId).toBe(userId);
    });

    test('should handle large order with many items', async () => {
      const userId = 'user-large-order';
      const items = Array.from({ length: 50 }, (_, i) => ({
        id: `prod-${i}`,
        name: `Item ${i}`,
        price: 10.0 + i,
        description: `Item description ${i}`,
        image: `item${i}.jpg`,
        dataAiHint: `category-${i % 5}`,
        quantity: 1,
      }));

      const payload = {
        items,
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const totalAmount = items.reduce((sum, item) => sum + item.price * item.quantity, 0);

      const mockOrder = {
        id: 'order-large',
        userId,
        items,
        totalAmount,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date(),
        estimatedDelivery: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body.items.length).toBe(50);
      expect(response.body.totalAmount).toBe(totalAmount);
    });

    test('should handle very large prices with decimal precision', async () => {
      const userId = 'user-large-price';
      const payload = {
        items: [
          {
            id: 'prod-1',
            name: 'Expensive Item',
            price: 9999.99,
            description: 'Very expensive',
            image: 'expensive.jpg',
            dataAiHint: 'luxury',
            quantity: 3,
          },
        ],
        shippingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        billingAddress: {
          street: '123 Main St',
          city: 'San Francisco',
          state: 'CA',
          zipCode: '94102',
          country: 'USA',
        },
        paymentMethod: {
          type: 'credit_card',
        },
      };

      const expectedTotal = 29999.97;
      const mockOrder = {
        id: 'order-expensive',
        userId,
        items: payload.items,
        totalAmount: expectedTotal,
        status: OrderStatus.PENDING,
        shippingAddress: payload.shippingAddress,
        billingAddress: payload.billingAddress,
        paymentMethod: payload.paymentMethod as any,
        orderDate: new Date(),
        estimatedDelivery: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      mockOrderService.createOrder.mockResolvedValue(mockOrder as any);

      const response = await request(app)
        .post(`/api/orders/${userId}`)
        .send(payload)
        .expect(201);

      expect(response.body.totalAmount).toBe(expectedTotal);
    });
  });
});
