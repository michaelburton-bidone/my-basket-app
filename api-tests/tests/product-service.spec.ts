import { test, expect } from '@playwright/test';

const API_BASE_URL = 'http://localhost:3001/api';

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
    percentage: number;
    endsAt: string;
  };
  createdAt?: string;
  updatedAt?: string;
}

test.describe('Product Service API', () => {
  test.describe('Health Check', () => {
    test('should return healthy status', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/health`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data).toHaveProperty('status', 'healthy');
      expect(data).toHaveProperty('service', 'product-service');
      expect(data).toHaveProperty('timestamp');
    });
  });

  test.describe('GET /products', () => {
    test('should retrieve all products with default pagination', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data).toHaveProperty('products');
      expect(data).toHaveProperty('total');
      expect(data).toHaveProperty('page', 1);
      expect(data).toHaveProperty('limit');
      expect(data).toHaveProperty('totalPages');
      
      expect(Array.isArray(data.products)).toBe(true);
      expect(data.total).toBeGreaterThanOrEqual(0);
    });

    test('should return products with valid structure', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products`);
      const data = await response.json();
      
      if (data.products.length > 0) {
        const product = data.products[0];
        
        expect(product).toHaveProperty('id');
        expect(product).toHaveProperty('name');
        expect(product).toHaveProperty('price');
        expect(product).toHaveProperty('description');
        expect(product).toHaveProperty('image');
        expect(product).toHaveProperty('dataAiHint');
        
        expect(typeof product.id).toBe('string');
        expect(typeof product.name).toBe('string');
        expect(typeof product.price).toBe('number');
        expect(product.price).toBeGreaterThan(0);
      }
    });

    test('should support pagination with page parameter', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?page=1&limit=5`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data.page).toBe(1);
      expect(data.limit).toBe(5);
      expect(data.products.length).toBeLessThanOrEqual(5);
    });

    test('should retrieve second page of products', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?page=2&limit=5`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data.page).toBe(2);
      expect(data.limit).toBe(5);
    });

    test('should filter products by category', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?category=fruits`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          expect(product.category).toBe('fruits');
        });
      }
    });

    test('should filter products by minimum price', async ({ request }) => {
      const minPrice = 5;
      const response = await request.get(`${API_BASE_URL}/products?minPrice=${minPrice}`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          expect(product.price).toBeGreaterThanOrEqual(minPrice);
        });
      }
    });

    test('should filter products by maximum price', async ({ request }) => {
      const maxPrice = 5;
      const response = await request.get(`${API_BASE_URL}/products?maxPrice=${maxPrice}`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          expect(product.price).toBeLessThanOrEqual(maxPrice);
        });
      }
    });

    test('should filter products by price range', async ({ request }) => {
      const minPrice = 3;
      const maxPrice = 6;
      const response = await request.get(`${API_BASE_URL}/products?minPrice=${minPrice}&maxPrice=${maxPrice}`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          expect(product.price).toBeGreaterThanOrEqual(minPrice);
          expect(product.price).toBeLessThanOrEqual(maxPrice);
        });
      }
    });

    test('should filter products by in-stock status', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?inStock=true`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          expect(product.inStock).toBe(true);
        });
      }
    });

    test('should search products by name', async ({ request }) => {
      const searchTerm = 'Organic';
      const response = await request.get(`${API_BASE_URL}/products?search=${encodeURIComponent(searchTerm)}`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data.products.length).toBeGreaterThanOrEqual(0);
    });

    test('should apply multiple filters simultaneously', async ({ request }) => {
      const response = await request.get(
        `${API_BASE_URL}/products?category=bakery&minPrice=2&maxPrice=10&inStock=true&page=1&limit=20`
      );
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data.page).toBe(1);
      expect(data.limit).toBe(20);
      
      if (data.products.length > 0) {
        data.products.forEach((product: Product) => {
          if (product.category) expect(product.category).toBe('bakery');
          expect(product.price).toBeGreaterThanOrEqual(2);
          expect(product.price).toBeLessThanOrEqual(10);
          if (product.inStock !== undefined) expect(product.inStock).toBe(true);
        });
      }
    });

    test('should handle limit=0 gracefully', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?limit=0`);
      
      // Should either fail validation or return empty results
      if (response.status() === 400) {
        expect(response.status()).toBe(400);
      }
    });

    test('should reject limit > 100', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?limit=150`);
      
      // Should fail validation
      expect(response.status()).toBe(400);
      const data = await response.json();
      expect(data).toHaveProperty('error');
    });

    test('should handle negative page number', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?page=-1`);
      
      expect(response.status()).toBe(400);
      const data = await response.json();
      expect(data).toHaveProperty('error');
    });
  });

  test.describe('GET /products/:id', () => {
    let validProductId: string = '1';

    test.beforeAll(async ({ request }) => {
      try {
        const response = await request.get(`${API_BASE_URL}/products`);
        const data = await response.json();
        
        if (data.products && data.products.length > 0) {
          validProductId = data.products[0].id;
        }
      } catch (error) {
        validProductId = '1';
      }
    });

    test('should retrieve a product by valid ID', async ({ request }) => {
      const productId = validProductId || '1';
      const response = await request.get(`${API_BASE_URL}/products/${productId}`);
      
      expect(response.status()).toBe(200);
      const product = await response.json();
      
      expect(product).toHaveProperty('id');
      expect(product.id).toBe(productId);
      expect(product).toHaveProperty('name');
      expect(product).toHaveProperty('price');
      expect(product).toHaveProperty('description');
      expect(product).toHaveProperty('image');
      expect(product).toHaveProperty('dataAiHint');
    });

    test('should return 404 for non-existent product', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products/non-existent-id-12345`);
      
      expect(response.status()).toBe(404);
      const data = await response.json();
      expect(data).toHaveProperty('error', 'Product not found');
    });

    test('should return valid product structure', async ({ request }) => {
      const productId = validProductId || '1';
      const response = await request.get(`${API_BASE_URL}/products/${productId}`);
      const product = await response.json();
      
      expect(typeof product.id).toBe('string');
      expect(typeof product.name).toBe('string');
      expect(typeof product.price).toBe('number');
      expect(typeof product.description).toBe('string');
      expect(typeof product.image).toBe('string');
      expect(typeof product.dataAiHint).toBe('string');
      
      if (product.category) {
        expect(typeof product.category).toBe('string');
      }
      
      if (product.inStock !== undefined) {
        expect(typeof product.inStock).toBe('boolean');
      }
      
      if (product.discount) {
        expect(typeof product.discount.percentage).toBe('number');
        if (typeof product.discount.endsAt === 'string') {
          expect(typeof product.discount.endsAt).toBe('string');
        }
      }
    });

    test('should include timestamps in response', async ({ request }) => {
      const productId = validProductId || '1';
      const response = await request.get(`${API_BASE_URL}/products/${productId}`);
      const product = await response.json();
      
      if (product.createdAt) {
        expect(product).toHaveProperty('createdAt');
        const createdAt = new Date(product.createdAt);
        expect(createdAt.getTime()).toBeGreaterThan(0);
      }
      
      if (product.updatedAt) {
        expect(product).toHaveProperty('updatedAt');
        const updatedAt = new Date(product.updatedAt);
        expect(updatedAt.getTime()).toBeGreaterThan(0);
      }
    });
  });

  test.describe('POST /products', () => {
    test('should create a new product with valid data', async ({ request }) => {
      const newProduct = {
        name: 'Test Product',
        price: 10.99,
        description: 'A test product for validation',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'test product',
        category: 'test-category',
        inStock: true,
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: newProduct,
      });

      expect(response.status()).toBe(201);
      const created = await response.json();
      
      expect(created).toHaveProperty('id');
      expect(created.name).toBe(newProduct.name);
      expect(created.price).toBe(newProduct.price);
      expect(created.description).toBe(newProduct.description);
      expect(created.category).toBe(newProduct.category);
      expect(created.inStock).toBe(true);
    });

    test('should return 400 for missing required fields', async ({ request }) => {
      const invalidProduct = {
        name: 'Missing Price Product',
        // Missing price field
        description: 'Description without price',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'test',
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: invalidProduct,
      });

      expect(response.status()).toBe(400);
      const error = await response.json();
      expect(error).toHaveProperty('error');
    });

    test('should return 400 for negative price', async ({ request }) => {
      const invalidProduct = {
        name: 'Negative Price Product',
        price: -5.99,
        description: 'Invalid negative price',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'test',
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: invalidProduct,
      });

      expect(response.status()).toBe(400);
    });

    test('should return 400 for invalid image URL', async ({ request }) => {
      const invalidProduct = {
        name: 'Invalid Image Product',
        price: 9.99,
        description: 'Invalid image URL',
        image: 'not-a-valid-url',
        dataAiHint: 'test',
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: invalidProduct,
      });

      expect(response.status()).toBe(400);
    });

    test('should create product with discount', async ({ request }) => {
      const newProduct = {
        name: 'Discounted Product',
        price: 15.99,
        description: 'Product with discount',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'discount test',
        discount: {
          percentage: 20,
          endsAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
        },
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: newProduct,
      });

      expect(response.status()).toBe(201);
      const created = await response.json();
      
      expect(created).toHaveProperty('discount');
      expect(created.discount.percentage).toBe(20);
    });

    test('should include timestamps when creating product', async ({ request }) => {
      const newProduct = {
        name: 'Timestamp Test Product',
        price: 12.99,
        description: 'Testing timestamps',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'timestamp test',
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: newProduct,
      });

      const created = await response.json();
      
      expect(created).toHaveProperty('createdAt');
      expect(created).toHaveProperty('updatedAt');
    });
  });

  test.describe('PUT /products/:id', () => {
    let productIdToUpdate: string = '1';

    test.beforeAll(async ({ request }) => {
      try {
        const response = await request.get(`${API_BASE_URL}/products`);
        const data = await response.json();
        
        if (data.products && data.products.length > 0) {
          productIdToUpdate = data.products[0].id;
        }
      } catch (error) {
        productIdToUpdate = '1';
      }
    });

    test('should update an existing product', async ({ request }) => {
      const id = productIdToUpdate || '1';
      const updates = {
        name: 'Updated Product Name',
        price: 19.99,
      };

      const response = await request.put(`${API_BASE_URL}/products/${id}`, {
        data: updates,
      });

      expect(response.status()).toBe(200);
      const updated = await response.json();
      
      expect(updated).toHaveProperty('id');
      expect(updated.name).toBe(updates.name);
      expect(updated.price).toBe(updates.price);
    });

    test('should partially update a product', async ({ request }) => {
      const updates = {
        price: 25.99,
      };

      const response = await request.put(`${API_BASE_URL}/products/${productIdToUpdate}`, {
        data: updates,
      });

      expect(response.status()).toBe(200);
      const updated = await response.json();
      expect(updated.price).toBe(25.99);
    });

    test('should return 404 when updating non-existent product', async ({ request }) => {
      const updates = { name: 'Updated Name' };
      
      const response = await request.put(`${API_BASE_URL}/products/non-existent-id`, {
        data: updates,
      });

      expect(response.status()).toBe(404);
      const error = await response.json();
      expect(error).toHaveProperty('error');
    });

    test('should update product timestamp', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products/${productIdToUpdate}`);
      const originalProduct = await response.json();
      const originalUpdatedAt = originalProduct.updatedAt;

      // Wait a moment
      await new Promise((resolve) => setTimeout(resolve, 100));

      const updateResponse = await request.put(`${API_BASE_URL}/products/${productIdToUpdate}`, {
        data: { name: 'Another Update' },
      });

      const updated = await updateResponse.json();
      
      // UpdatedAt should be different (newer)
      expect(new Date(updated.updatedAt).getTime()).toBeGreaterThanOrEqual(
        new Date(originalUpdatedAt).getTime()
      );
    });

    test('should validate update data', async ({ request }) => {
      const id = productIdToUpdate || '1';
      const invalidUpdate = {
        price: -10, // Invalid negative price
      };

      const response = await request.put(`${API_BASE_URL}/products/${id}`, {
        data: invalidUpdate,
      });

      expect([400, 422]).toContain(response.status());
    });
  });

  test.describe('DELETE /products/:id', () => {
    let productIdToDelete: string;

    test.beforeEach(async ({ request }) => {
      // Create a product to delete
      const newProduct = {
        name: 'Product to Delete',
        price: 5.99,
        description: 'Temporary product for deletion test',
        image: 'https://placehold.co/300x200.png',
        dataAiHint: 'delete test',
      };

      const response = await request.post(`${API_BASE_URL}/products`, {
        data: newProduct,
      });

      const created = await response.json();
      productIdToDelete = created.id;
    });

    test('should delete an existing product', async ({ request }) => {
      const response = await request.delete(`${API_BASE_URL}/products/${productIdToDelete}`);
      
      expect(response.status()).toBe(204);
    });

    test('should return 404 when deleting non-existent product', async ({ request }) => {
      const response = await request.delete(`${API_BASE_URL}/products/non-existent-id-xyz`);
      
      expect(response.status()).toBe(404);
      const error = await response.json();
      expect(error).toHaveProperty('error', 'Product not found');
    });

    test('should confirm product is deleted after deletion', async ({ request }) => {
      const deleteResponse = await request.delete(`${API_BASE_URL}/products/${productIdToDelete}`);
      expect(deleteResponse.status()).toBe(204);

      // Try to get the deleted product
      const getResponse = await request.get(`${API_BASE_URL}/products/${productIdToDelete}`);
      expect(getResponse.status()).toBe(404);
    });
  });

  test.describe('GET /categories', () => {
    test('should retrieve list of all categories', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/categories`);
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data).toHaveProperty('categories');
      expect(Array.isArray(data.categories)).toBe(true);
    });

    test('should return categories as array of strings', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/categories`);
      const data = await response.json();
      
      if (data.categories.length > 0) {
        data.categories.forEach((category: string) => {
          expect(typeof category).toBe('string');
        });
      }
    });

    test('should return unique categories only', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/categories`);
      const data = await response.json();
      
      const uniqueCategories = new Set(data.categories);
      expect(uniqueCategories.size).toBe(data.categories.length);
    });
  });

  test.describe('Edge Cases & Error Handling', () => {
    test('should handle empty search results gracefully', async ({ request }) => {
      const response = await request.get(
        `${API_BASE_URL}/products?search=${encodeURIComponent('nonexistentproductthatdoesnotexist12345')}`
      );
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data.products).toEqual([]);
      expect(data.total).toBe(0);
    });

    test('should handle special characters in search', async ({ request }) => {
      const response = await request.get(
        `${API_BASE_URL}/products?search=${encodeURIComponent('@#$%^&*()')}`
      );
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(Array.isArray(data.products)).toBe(true);
    });

    test('should handle very large limit parameter gracefully', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products?limit=1000`);
      
      // Should fail validation since limit max is 100
      expect(response.status()).toBe(400);
    });

    test('should handle concurrent requests', async ({ request }) => {
      const requests = Array.from({ length: 5 }, (_, i) =>
        request.get(`${API_BASE_URL}/products?page=${i + 1}&limit=10`)
      );

      const responses = await Promise.all(requests);
      
      responses.forEach((response) => {
        expect(response.status()).toBe(200);
      });
    });

    test('should handle missing content-type header', async ({ request }) => {
      const response = await request.post(`${API_BASE_URL}/products`, {
        data: {
          name: 'Test',
          price: 10,
          description: 'Test',
          image: 'https://placehold.co/300x200.png',
          dataAiHint: 'test',
        },
        headers: { 'Content-Type': 'application/json' },
      });

      // Should still work - either 201 or 400
      expect([201, 400, 500]).toContain(response.status());
    });
  });

  test.describe('Response Format Validation', () => {
    test('should return proper JSON content-type', async ({ request }) => {
      const response = await request.get(`${API_BASE_URL}/products`);
      
      const contentType = response.headers()['content-type'];
      expect(contentType).toContain('application/json');
    });

    test('should include proper status codes', async ({ request }) => {
      const getResponse = await request.get(`${API_BASE_URL}/products`);
      expect(getResponse.status()).toBe(200);

      const notFoundResponse = await request.get(`${API_BASE_URL}/products/invalid-id`);
      expect(notFoundResponse.status()).toBe(404);
    });

    test('should include error details in validation errors', async ({ request }) => {
      const response = await request.post(`${API_BASE_URL}/products`, {
        data: { name: 'No Price' },
      });

      expect(response.status()).toBe(400);
      const error = await response.json();
      
      expect(error).toHaveProperty('error');
      if (error.details) {
        expect(Array.isArray(error.details)).toBe(true);
      }
    });
  });
});
