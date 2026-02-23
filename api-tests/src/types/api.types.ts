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
  id: string;
  productId?: string; // For backward compatibility
  name: string;
  price: number;
  description?: string;
  image?: string;
  dataAiHint?: string;
  category?: string;
}

export interface CartItem extends Product {
  quantity: number;
  addedAt?: string;
  updatedAt?: string;
}

export interface Cart {
  id: string;
  userId: string;
  items: CartItem[];
  totalAmount: number;
  totalItems: number;
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
