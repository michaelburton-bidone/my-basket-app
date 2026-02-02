/**
 * Test Data Manager
 * 
 * Provides test data fixtures and utilities for populating forms and testing
 */

import { Product, CartItem, CheckoutData, User } from '@types/test.types';
import { generateUniqueId } from '@utils/helpers';

/**
 * Sample products
 */
export const sampleProducts: Product[] = [
  {
    id: 1,
    name: 'Wireless Headphones',
    price: 79.99,
    description: 'High-quality wireless headphones with noise cancellation',
  },
  {
    id: 2,
    name: 'USB-C Cable',
    price: 12.99,
    description: 'Durable USB-C charging cable, 6ft length',
  },
  {
    id: 3,
    name: 'Phone Case',
    price: 24.99,
    description: 'Protective phone case with shock absorption',
  },
  {
    id: 4,
    name: 'Screen Protector',
    price: 9.99,
    description: 'Tempered glass screen protector',
  },
  {
    id: 5,
    name: 'Portable Charger',
    price: 49.99,
    description: '20000mAh portable power bank',
  },
];

/**
 * Generate test user
 */
export function generateTestUser(): User {
  const uniqueId = generateUniqueId();
  return {
    username: `testuser_${uniqueId}`,
    email: `testuser_${uniqueId}@test.com`,
    password: 'TestPassword123!',
  };
}

/**
 * Generate checkout data
 */
export function generateCheckoutData(): CheckoutData {
  return {
    firstName: 'John',
    lastName: 'Doe',
    email: 'john.doe@test.com',
    phone: '+1-555-123-4567',
    address: '123 Main Street',
    city: 'New York',
    state: 'NY',
    zipCode: '10001',
    country: 'United States',
  };
}

/**
 * Generate random checkout data
 */
export function generateRandomCheckoutData(): CheckoutData {
  const uniqueId = generateUniqueId();
  return {
    firstName: `First_${uniqueId}`,
    lastName: `Last_${uniqueId}`,
    email: `test_${uniqueId}@test.com`,
    phone: `+1-555-${Math.floor(Math.random() * 9000) + 1000}-${Math.floor(Math.random() * 9000) + 1000}`,
    address: `${Math.floor(Math.random() * 999) + 1} Test Street`,
    city: 'Test City',
    state: 'TS',
    zipCode: String(Math.floor(Math.random() * 90000) + 10000),
    country: 'United States',
  };
}

/**
 * Get sample product by name
 */
export function getSampleProductByName(name: string): Product | undefined {
  return sampleProducts.find((product) =>
    product.name.toLowerCase().includes(name.toLowerCase())
  );
}

/**
 * Get sample product by index
 */
export function getSampleProductByIndex(index: number): Product | undefined {
  return sampleProducts[index];
}

/**
 * Create cart item from product
 */
export function createCartItem(product: Product, quantity: number = 1): CartItem {
  return {
    productId: product.id,
    name: product.name,
    price: product.price,
    quantity,
    subtotal: product.price * quantity,
  };
}

/**
 * Calculate cart total
 */
export function calculateCartTotal(items: CartItem[], tax: number = 0, discount: number = 0): number {
  const subtotal = items.reduce((sum, item) => sum + item.subtotal, 0);
  return subtotal + tax - discount;
}

/**
 * Valid test credit card numbers
 */
export const testCreditCards = {
  visa: {
    number: '4111111111111111',
    name: 'John Doe',
    expiry: '12/25',
    cvv: '123',
  },
  mastercard: {
    number: '5555555555554444',
    name: 'John Doe',
    expiry: '12/25',
    cvv: '123',
  },
  amex: {
    number: '378282246310005',
    name: 'John Doe',
    expiry: '12/25',
    cvv: '1234',
  },
  invalid: {
    number: '1234567890123456',
    name: 'John Doe',
    expiry: '12/25',
    cvv: '123',
  },
};

/**
 * Get valid test credit card
 */
export function getTestCreditCard(type: 'visa' | 'mastercard' | 'amex' = 'visa') {
  return testCreditCards[type];
}

/**
 * Test promo codes
 */
export const testPromoCodes = {
  valid10: {
    code: 'SAVE10',
    discountPercentage: 10,
    minOrderAmount: 50,
  },
  valid20: {
    code: 'SAVE20',
    discountPercentage: 20,
    minOrderAmount: 100,
  },
  expired: {
    code: 'EXPIRED2023',
    discountPercentage: 15,
    minOrderAmount: 0,
  },
  invalid: {
    code: 'INVALIDCODE123',
    discountPercentage: 0,
    minOrderAmount: 0,
  },
};

/**
 * Get test promo code
 */
export function getTestPromoCode(type: 'valid10' | 'valid20' | 'expired' | 'invalid' = 'valid10') {
  return testPromoCodes[type];
}
