/**
 * Test data for Cart API tests
 */

import { AddToCartRequest, CartItem, Cart } from '@types/index';

export const testUsers = {
  user1: {
    id: 'user-001',
    name: 'Test User 1',
  },
  user2: {
    id: 'user-002',
    name: 'Test User 2',
  },
  user3: {
    id: 'user-003',
    name: 'Test User 3',
  },
};

export const testProducts = {
  apples: {
    id: '1',
    productId: '1',
    name: 'Organic Apples',
    price: 3.99,
    quantity: 10,
  },
  bread: {
    id: '2',
    productId: '2',
    name: 'Whole Wheat Bread',
    price: 4.49,
    quantity: 8,
  },
  eggs: {
    id: '3',
    productId: '3',
    name: 'Free-Range Eggs',
    price: 5.99,
    quantity: 5,
  },
  spinach: {
    id: '4',
    productId: '4',
    name: 'Organic Spinach',
    price: 2.99,
    quantity: 3,
  },
  almondMilk: {
    id: '5',
    productId: '5',
    name: 'Almond Milk',
    price: 3.79,
    quantity: 2,
  },
  chickenBreast: {
    id: '6',
    productId: '6',
    name: 'Chicken Breast',
    price: 9.99,
    quantity: 4,
  },
  brownRice: {
    id: '7',
    productId: '7',
    name: 'Brown Rice',
    price: 2.49,
    quantity: 6,
  },
  yogurt: {
    id: '8',
    productId: '8',
    name: 'Greek Yogurt',
    price: 4.99,
    quantity: 5,
  },
};

export const validAddToCartRequests = {
  singleItem: {
    productId: testProducts.apples.productId,
    quantity: 2,
    name: testProducts.apples.name,
    price: testProducts.apples.price,
  } as AddToCartRequest,

  largeQuantity: {
    productId: testProducts.eggs.productId,
    quantity: 100,
    name: testProducts.eggs.name,
    price: testProducts.eggs.price,
  } as AddToCartRequest,

  minimalData: {
    productId: testProducts.spinach.productId,
    quantity: 1,
  } as AddToCartRequest,
};

export const invalidAddToCartRequests = {
  missingProductId: {
    quantity: 2,
    name: 'Test',
    price: 10,
  },

  missingQuantity: {
    productId: testProducts.apples.productId,
    name: 'Organic Apples',
    price: 3.99,
  },

  zeroQuantity: {
    productId: testProducts.apples.productId,
    quantity: 0,
    name: 'Organic Apples',
    price: 3.99,
  },

  negativeQuantity: {
    productId: testProducts.apples.productId,
    quantity: -5,
    name: 'Organic Apples',
    price: 3.99,
  },

  negativePrice: {
    productId: testProducts.apples.productId,
    quantity: 2,
    name: 'Organic Apples',
    price: -10,
  },

  emptyProductId: {
    productId: '',
    quantity: 2,
    name: 'Test',
    price: 10,
  },
};

export const mockCartResponses = {
  emptyCart: {
    id: 'cart-001',
    userId: testUsers.user1.id,
    items: [],
    totalAmount: 0,
    totalItems: 0,
  } as Cart,

  cartWithOneItem: {
    id: 'cart-002',
    userId: testUsers.user1.id,
    items: [
      {
        id: testProducts.apples.id,
        itemId: 'item-001',
        productId: testProducts.apples.productId,
        name: testProducts.apples.name,
        price: testProducts.apples.price,
        quantity: 2,
      },
    ],
    totalAmount: 7.98,
    totalItems: 2,
  } as Cart,

  cartWithMultipleItems: {
    id: 'cart-003',
    userId: testUsers.user1.id,
    items: [
      {
        id: testProducts.apples.id,
        itemId: 'item-001',
        productId: testProducts.apples.productId,
        name: testProducts.apples.name,
        price: testProducts.apples.price,
        quantity: 2,
      },
      {
        id: testProducts.eggs.id,
        itemId: 'item-002',
        productId: testProducts.eggs.productId,
        name: testProducts.eggs.name,
        price: testProducts.eggs.price,
        quantity: 3,
      },
      {
        id: testProducts.almondMilk.id,
        itemId: 'item-003',
        productId: testProducts.almondMilk.productId,
        name: testProducts.almondMilk.name,
        price: testProducts.almondMilk.price,
        quantity: 1,
      },
    ],
    totalAmount: 40.76,
    totalItems: 6,
  } as Cart,
};

export const errorMessages = {
  cartNotFound: 'Cart not found',
  itemNotFound: 'Item not found in cart',
  invalidUserId: 'Invalid user ID',
  invalidProductId: 'Invalid product ID',
  invalidQuantity: 'Quantity must be a positive number',
  invalidData: 'Invalid request data',
  serverError: 'Internal server error',
  unauthorized: 'Unauthorized access',
  forbidden: 'Forbidden',
  methodNotAllowed: 'Method not allowed',
};
