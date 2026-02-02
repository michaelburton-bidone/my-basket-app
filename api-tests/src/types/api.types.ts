/**
 * Common API response types
 */

export interface ApiResponse<T = any> {
  status: number;
  data?: T;
  error?: string;
  message?: string;
  headers: Record<string, any>;
}

export interface ErrorResponse {
  error?: string;
  message?: string;
  details?: any;
  statusCode?: number;
}

export interface PageInfo {
  page: number;
  limit: number;
  total: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: PageInfo;
}

/**
 * Cart API specific types
 */

export interface Product {
  productId: string;
  name: string;
  price: number;
  quantity: number;
  category?: string;
  description?: string;
}

export interface CartItem extends Product {
  itemId?: string;
  addedAt?: string;
  updatedAt?: string;
}

export interface Cart {
  userId: string;
  items: CartItem[];
  totalPrice: number;
  itemCount: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface AddToCartRequest {
  productId: string;
  quantity: number;
  name?: string;
  price?: number;
}

export interface UpdateCartItemRequest {
  quantity: number;
}

export interface RemoveFromCartRequest {
  itemId: string;
}

export interface CartResponse {
  success: boolean;
  data: Cart;
  message?: string;
}

export interface CartItemResponse {
  success: boolean;
  data: CartItem;
  message?: string;
}

export interface EmptyCartResponse {
  success: boolean;
  message: string;
}

/**
 * Request/Response validation
 */

export interface ValidationRule {
  field: string;
  rule: string;
  value?: any;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}
