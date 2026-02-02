/**
 * Test data and fixture types
 */

export interface Product {
  id: number;
  name: string;
  price: number;
  description?: string;
  imageUrl?: string;
  quantity?: number;
}

export interface CartItem {
  productId: number;
  name: string;
  price: number;
  quantity: number;
  subtotal: number;
}

export interface Cart {
  items: CartItem[];
  subtotal: number;
  tax: number;
  total: number;
  isEmpty: boolean;
}

export interface CheckoutData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

export interface User {
  username: string;
  email: string;
  password: string;
}

export interface AuthCredentials {
  username?: string;
  email?: string;
  password: string;
}

export interface TestResult {
  passed: boolean;
  message: string;
  timestamp: Date;
}

export interface ValidationError {
  field: string;
  message: string;
  type: 'error' | 'warning';
}
