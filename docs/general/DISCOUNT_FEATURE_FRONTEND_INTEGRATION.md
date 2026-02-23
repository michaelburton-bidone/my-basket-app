# Discount Feature - Integration Guide for Frontend

## Overview
This guide helps frontend developers integrate the discount feature into the user interface.

## TypeScript Interfaces

### Product Type with Discount
```typescript
interface Product {
  id: string;
  name: string;
  price: number;
  description: string;
  image: string;
  dataAiHint: string;
  category?: string;
  inStock?: boolean;
  discount?: {
    percentage: number;  // 0-100
    endsAt: Date;
  };
  createdAt?: Date;
  updatedAt?: Date;
}
```

## React Component Examples

### Product Card with Discount Badge
```typescript
import React from 'react';

interface ProductCardProps {
  product: Product;
}

const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
  const hasDiscount = product.discount && new Date(product.discount.endsAt) > new Date();
  const discountedPrice = product.price * (1 - (product.discount?.percentage || 0) / 100);

  return (
    <div className="product-card">
      <img src={product.image} alt={product.name} />
      
      {hasDiscount && (
        <div className="discount-badge">
          <span className="discount-percent">
            {product.discount?.percentage}% OFF
          </span>
        </div>
      )}

      <h3>{product.name}</h3>
      <p>{product.description}</p>

      <div className="pricing">
        {hasDiscount ? (
          <>
            <span className="original-price">
              ${product.price.toFixed(2)}
            </span>
            <span className="discount-price">
              ${discountedPrice.toFixed(2)}
            </span>
          </>
        ) : (
          <span className="price">
            ${product.price.toFixed(2)}
          </span>
        )}
      </div>

      {hasDiscount && (
        <p className="discount-expires">
          Sale ends: {new Date(product.discount!.endsAt).toLocaleDateString()}
        </p>
      )}

      <button>Add to Cart</button>
    </div>
  );
};

export default ProductCard;
```

### Discount Price Calculator Hook
```typescript
import { useMemo } from 'react';

interface UseDiscountPriceProps {
  originalPrice: number;
  discountPercentage?: number;
  isActive?: boolean;
}

interface DiscountedPriceResult {
  originalPrice: number;
  discountedPrice: number;
  discountAmount: number;
  hasDiscount: boolean;
}

const useDiscountPrice = ({
  originalPrice,
  discountPercentage = 0,
  isActive = true,
}: UseDiscountPriceProps): DiscountedPriceResult => {
  return useMemo(() => {
    const hasDiscount = isActive && discountPercentage > 0;
    const discountAmount = hasDiscount 
      ? originalPrice * (discountPercentage / 100)
      : 0;
    const discountedPrice = originalPrice - discountAmount;

    return {
      originalPrice,
      discountedPrice: Number(discountedPrice.toFixed(2)),
      discountAmount: Number(discountAmount.toFixed(2)),
      hasDiscount,
    };
  }, [originalPrice, discountPercentage, isActive]);
};

export default useDiscountPrice;
```

### Product List with Discount Filtering
```typescript
import React, { useState, useMemo } from 'react';

interface ProductListProps {
  products: Product[];
}

const ProductList: React.FC<ProductListProps> = ({ products }) => {
  const [showOnlyDiscounted, setShowOnlyDiscounted] = useState(false);

  const filteredProducts = useMemo(() => {
    if (!showOnlyDiscounted) return products;

    return products.filter(product => {
      if (!product.discount) return false;
      const expiryDate = new Date(product.discount.endsAt);
      return expiryDate > new Date();
    });
  }, [products, showOnlyDiscounted]);

  return (
    <div className="product-list">
      <div className="filter-controls">
        <label>
          <input
            type="checkbox"
            checked={showOnlyDiscounted}
            onChange={(e) => setShowOnlyDiscounted(e.target.checked)}
          />
          Show only discounted products
        </label>
      </div>

      <div className="products-grid">
        {filteredProducts.map(product => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>

      {filteredProducts.length === 0 && (
        <p className="no-results">
          {showOnlyDiscounted 
            ? 'No discounted products available' 
            : 'No products found'}
        </p>
      )}
    </div>
  );
};

export default ProductList;
```

### Discount Timer Component
```typescript
import React, { useState, useEffect } from 'react';

interface DiscountTimerProps {
  expiryDate: string | Date;
  onExpired?: () => void;
}

interface TimeRemaining {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  isExpired: boolean;
}

const DiscountTimer: React.FC<DiscountTimerProps> = ({ expiryDate, onExpired }) => {
  const [timeRemaining, setTimeRemaining] = useState<TimeRemaining | null>(null);

  useEffect(() => {
    const calculateTimeRemaining = () => {
      const expiry = new Date(expiryDate).getTime();
      const now = new Date().getTime();
      const difference = expiry - now;

      if (difference <= 0) {
        setTimeRemaining({
          days: 0,
          hours: 0,
          minutes: 0,
          seconds: 0,
          isExpired: true,
        });
        onExpired?.();
        return;
      }

      const days = Math.floor(difference / (1000 * 60 * 60 * 24));
      const hours = Math.floor((difference / (1000 * 60 * 60)) % 24);
      const minutes = Math.floor((difference / 1000 / 60) % 60);
      const seconds = Math.floor((difference / 1000) % 60);

      setTimeRemaining({
        days,
        hours,
        minutes,
        seconds,
        isExpired: false,
      });
    };

    calculateTimeRemaining();
    const interval = setInterval(calculateTimeRemaining, 1000);

    return () => clearInterval(interval);
  }, [expiryDate, onExpired]);

  if (!timeRemaining) return null;

  if (timeRemaining.isExpired) {
    return <span className="discount-expired">Sale ended</span>;
  }

  return (
    <div className="discount-timer">
      <span className="timer-label">Offer expires in:</span>
      {timeRemaining.days > 0 && (
        <span className="timer-segment">
          {timeRemaining.days}d
        </span>
      )}
      <span className="timer-segment">
        {String(timeRemaining.hours).padStart(2, '0')}h
      </span>
      <span className="timer-segment">
        {String(timeRemaining.minutes).padStart(2, '0')}m
      </span>
      <span className="timer-segment">
        {String(timeRemaining.seconds).padStart(2, '0')}s
      </span>
    </div>
  );
};

export default DiscountTimer;
```

## CSS Styling Examples

### Discount Badge
```css
.discount-badge {
  position: absolute;
  top: 10px;
  right: 10px;
  background: #ff4444;
  color: white;
  padding: 8px 12px;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.discount-percent {
  text-align: center;
  font-size: 14px;
  line-height: 1.2;
}
```

### Price Display with Discount
```css
.pricing {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 12px 0;
}

.original-price {
  text-decoration: line-through;
  color: #999;
  font-size: 16px;
}

.discount-price {
  font-size: 24px;
  font-weight: bold;
  color: #ff4444;
}

.price {
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.discount-expires {
  font-size: 12px;
  color: #ff4444;
  font-weight: 500;
  margin: 8px 0;
}
```

### Discount Timer
```css
.discount-timer {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #ff4444;
  background: #fff5f5;
  padding: 8px 12px;
  border-radius: 4px;
  margin-top: 8px;
}

.timer-label {
  font-weight: 600;
}

.timer-segment {
  background: #ff4444;
  color: white;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
  min-width: 24px;
  text-align: center;
}

.discount-expired {
  color: #666;
  font-size: 12px;
  font-style: italic;
}
```

## API Integration

### Fetch Products with Discounts
```typescript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

export const fetchProducts = async (): Promise<Product[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/products`);
    return response.data.products;
  } catch (error) {
    console.error('Failed to fetch products:', error);
    throw error;
  }
};

export const createProductWithDiscount = async (
  productData: Omit<Product, 'id' | 'createdAt' | 'updatedAt'>
): Promise<Product> => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/api/products`,
      productData
    );
    return response.data;
  } catch (error) {
    console.error('Failed to create product:', error);
    throw error;
  }
};

export const updateProductDiscount = async (
  productId: string,
  discount: Product['discount']
): Promise<Product> => {
  try {
    const response = await axios.put(
      `${API_BASE_URL}/api/products/${productId}`,
      { discount }
    );
    return response.data;
  } catch (error) {
    console.error('Failed to update product discount:', error);
    throw error;
  }
};
```

## State Management (Redux Example)

### Slice
```typescript
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

interface ProductState {
  items: Product[];
  loading: boolean;
  error: string | null;
}

export const fetchProducts = createAsyncThunk(
  'products/fetchProducts',
  async () => {
    const response = await fetch('http://localhost:3001/api/products');
    return (await response.json()).products;
  }
);

const productSlice = createSlice({
  name: 'products',
  initialState: {
    items: [],
    loading: false,
    error: null,
  } as ProductState,
  extraReducers: (builder) => {
    builder
      .addCase(fetchProducts.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchProducts.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
      })
      .addCase(fetchProducts.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch products';
      });
  },
});

export default productSlice.reducer;
```

## Testing Examples

### Jest Test
```typescript
import { render, screen } from '@testing-library/react';
import ProductCard from './ProductCard';

describe('ProductCard with Discount', () => {
  const mockProduct: Product = {
    id: '1',
    name: 'Test Product',
    price: 10,
    description: 'Test Description',
    image: 'https://example.com/image.jpg',
    dataAiHint: 'test',
    discount: {
      percentage: 20,
      endsAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    },
  };

  test('displays discount badge', () => {
    render(<ProductCard product={mockProduct} />);
    expect(screen.getByText('20% OFF')).toBeInTheDocument();
  });

  test('displays discounted price', () => {
    render(<ProductCard product={mockProduct} />);
    expect(screen.getByText('$8.00')).toBeInTheDocument();
    expect(screen.getByText('$10.00')).toBeInTheDocument();
  });

  test('displays original price as strikethrough', () => {
    const { container } = render(<ProductCard product={mockProduct} />);
    const strikethrough = container.querySelector('.original-price');
    expect(strikethrough).toHaveTextContent('$10.00');
  });
});
```

## Best Practices

1. **Always check discount expiry**: Use `new Date(product.discount.endsAt) > new Date()`
2. **Handle missing discounts**: Use optional chaining and nullish coalescing
3. **Format prices consistently**: Always use `.toFixed(2)` for display
4. **Cache calculations**: Use `useMemo` for discount price calculations
5. **Update timers efficiently**: Use `setInterval` with cleanup
6. **Validate discount data**: Ensure percentage is 0-100
7. **Provide loading states**: Show skeleton or spinner while fetching
8. **Handle errors gracefully**: Show user-friendly error messages

## Performance Tips

- Use `React.memo` for product cards to prevent unnecessary re-renders
- Implement pagination to reduce DOM nodes
- Lazy load images with `Intersection Observer`
- Cache API responses with `SWR` or `React Query`
- Debounce filter/search input

---

**Integration Status**: Ready for Implementation
**Last Updated**: 2026-02-09
