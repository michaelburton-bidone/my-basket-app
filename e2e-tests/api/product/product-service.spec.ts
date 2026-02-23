import { test, expect } from '@playwright/test';

test.describe('Product Service API Tests - Discount Feature', () => {
  const baseURL = 'http://localhost:3001';

  test('GET /products - List all products with discount information', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('products');
    expect(data).toHaveProperty('total');
    expect(data).toHaveProperty('page', 1);
    expect(data).toHaveProperty('limit');
    expect(data).toHaveProperty('totalPages');

    // Verify products array
    expect(Array.isArray(data.products)).toBe(true);
    expect(data.products.length).toBeGreaterThan(0);

    // Verify product structure includes discount field
    const product = data.products[0];
    expect(product).toHaveProperty('id');
    expect(product).toHaveProperty('name');
    expect(product).toHaveProperty('price');
    expect(product).toHaveProperty('description');
    expect(product).toHaveProperty('image');
    expect(product).toHaveProperty('dataAiHint');

    // Check for products with discounts
    const productsWithDiscount = data.products.filter((p: any) => p.discount);
    expect(productsWithDiscount.length).toBeGreaterThan(0);

    // Verify discount structure
    const discountedProduct = productsWithDiscount[0];
    expect(discountedProduct.discount).toHaveProperty('percentage');
    expect(discountedProduct.discount).toHaveProperty('endsAt');
    expect(typeof discountedProduct.discount.percentage).toBe('number');
    expect(discountedProduct.discount.percentage).toBeGreaterThanOrEqual(0);
    expect(discountedProduct.discount.percentage).toBeLessThanOrEqual(100);
  });

  test('GET /products/:id - Get product by ID with discount', async ({ request }) => {
    // Get product 1 (Organic Apples - has 10% discount)
    const response = await request.get(`${baseURL}/api/products/1`);
    expect(response.status()).toBe(200);

    const product = await response.json();
    expect(product).toHaveProperty('id', '1');
    expect(product).toHaveProperty('name', 'Organic Apples');
    expect(product).toHaveProperty('price');
    expect(product).toHaveProperty('discount');

    // Verify discount details
    expect(product.discount).toHaveProperty('percentage', 10);
    expect(product.discount).toHaveProperty('endsAt');
    expect(new Date(product.discount.endsAt).getTime()).toBeGreaterThan(Date.now());
  });

  test('GET /products/:id - Get product by ID without discount', async ({ request }) => {
    // Get product 3 (Free-Range Eggs - no discount)
    const response = await request.get(`${baseURL}/api/products/3`);
    expect(response.status()).toBe(200);

    const product = await response.json();
    expect(product).toHaveProperty('id', '3');
    expect(product).toHaveProperty('name', 'Free-Range Eggs');
    // Discount should not be present or be undefined
    expect(!product.discount || Object.keys(product.discount).length === 0).toBe(true);
  });

  test('GET /products/:id - Not found error', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products/999999`);
    expect(response.status()).toBe(404);

    const error = await response.json();
    expect(error).toHaveProperty('error');
  });

  test('POST /products - Create product with discount', async ({ request }) => {
    const newProduct = {
      name: 'Organic Bananas',
      price: 2.99,
      description: 'Fresh organic bananas, rich in potassium',
      image: 'https://placehold.co/300x200.png',
      dataAiHint: 'bananas fruit',
      category: 'fruits',
      inStock: true,
      discount: {
        percentage: 12,
        endsAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      },
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(response.status()).toBe(201);

    const createdProduct = await response.json();
    expect(createdProduct).toHaveProperty('id');
    expect(createdProduct).toHaveProperty('name', 'Organic Bananas');
    expect(createdProduct).toHaveProperty('price', 2.99);
    expect(createdProduct).toHaveProperty('discount');
    expect(createdProduct.discount).toHaveProperty('percentage', 12);
    expect(createdProduct.discount).toHaveProperty('endsAt');
    expect(createdProduct).toHaveProperty('createdAt');
    expect(createdProduct).toHaveProperty('updatedAt');
  });

  test('POST /products - Create product without discount', async ({ request }) => {
    const newProduct = {
      name: 'Cheddar Cheese',
      price: 6.99,
      description: 'Sharp cheddar cheese block',
      image: 'https://placehold.co/300x200.png',
      dataAiHint: 'cheese dairy',
      category: 'dairy',
      inStock: true,
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(response.status()).toBe(201);

    const createdProduct = await response.json();
    expect(createdProduct).toHaveProperty('id');
    expect(createdProduct).toHaveProperty('name', 'Cheddar Cheese');
    expect(createdProduct).toHaveProperty('price', 6.99);
    // Discount should not be present or be optional
    expect(!createdProduct.discount || createdProduct.discount === null).toBe(true);
  });

  test('POST /products - Invalid discount percentage (too high)', async ({ request }) => {
    const newProduct = {
      name: 'Invalid Discount Product',
      price: 4.99,
      description: 'Test product with invalid discount',
      image: 'https://placehold.co/300x200.png',
      dataAiHint: 'test',
      discount: {
        percentage: 150, // Invalid: > 100
        endsAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      },
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(response.status()).toBe(400);

    const error = await response.json();
    expect(error).toHaveProperty('error');
  });

  test('POST /products - Invalid discount percentage (negative)', async ({ request }) => {
    const newProduct = {
      name: 'Invalid Discount Product',
      price: 4.99,
      description: 'Test product with invalid discount',
      image: 'https://placehold.co/300x200.png',
      dataAiHint: 'test',
      discount: {
        percentage: -10, // Invalid: < 0
        endsAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      },
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(response.status()).toBe(400);

    const error = await response.json();
    expect(error).toHaveProperty('error');
  });

  test('PUT /products/:id - Update product to add discount', async ({ request }) => {
    // First, get product 3 to verify no discount
    const getResponse = await request.get(`${baseURL}/api/products/3`);
    const originalProduct = await getResponse.json();
    expect(!originalProduct.discount).toBe(true);

    // Update product 3 to add discount
    const updates = {
      discount: {
        percentage: 8,
        endsAt: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toISOString(),
      },
    };

    const updateResponse = await request.put(`${baseURL}/api/products/3`, {
      data: updates,
    });
    expect(updateResponse.status()).toBe(200);

    const updatedProduct = await updateResponse.json();
    expect(updatedProduct).toHaveProperty('discount');
    expect(updatedProduct.discount).toHaveProperty('percentage', 8);
    expect(updatedProduct.discount).toHaveProperty('endsAt');
  });

  test('PUT /products/:id - Update product to modify discount', async ({ request }) => {
    // Update product 1 discount from 10% to 25%
    const updates = {
      discount: {
        percentage: 25,
        endsAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
      },
    };

    const response = await request.put(`${baseURL}/api/products/1`, {
      data: updates,
    });
    expect(response.status()).toBe(200);

    const updatedProduct = await response.json();
    expect(updatedProduct).toHaveProperty('discount');
    expect(updatedProduct.discount).toHaveProperty('percentage', 25);
  });

  test('PUT /products/:id - Product not found', async ({ request }) => {
    const updates = {
      name: 'Updated Name',
    };

    const response = await request.put(`${baseURL}/api/products/999999`, {
      data: updates,
    });
    expect(response.status()).toBe(404);

    const error = await response.json();
    expect(error).toHaveProperty('error', 'Product not found');
  });

  test('DELETE /products/:id - Delete product', async ({ request }) => {
    // First create a product
    const newProduct = {
      name: 'Deletable Product',
      price: 3.99,
      description: 'This product will be deleted',
      image: 'https://placehold.co/300x200.png',
      dataAiHint: 'delete test',
      category: 'test',
    };

    const createResponse = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(createResponse.status()).toBe(201);

    const createdProduct = await createResponse.json();
    const productId = createdProduct.id;

    // Delete the product
    const deleteResponse = await request.delete(`${baseURL}/api/products/${productId}`);
    expect(deleteResponse.status()).toBe(204);

    // Verify product is deleted
    const getResponse = await request.get(`${baseURL}/api/products/${productId}`);
    expect(getResponse.status()).toBe(404);
  });

  test('GET /categories - List product categories', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/categories`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('categories');
    expect(Array.isArray(data.categories)).toBe(true);
    expect(data.categories.length).toBeGreaterThan(0);
  });

  test('GET /health - Health check endpoint', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/health`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('status', 'healthy');
    expect(data).toHaveProperty('service', 'product-service');
    expect(data).toHaveProperty('timestamp');
  });

  test('GET /products - Filter by category (fruits)', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products?category=fruits`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(Array.isArray(data.products)).toBe(true);

    // Verify all returned products are in fruits category
    data.products.forEach((product: any) => {
      expect(product.category).toBe('fruits');
    });
  });

  test('GET /products - Filter by min price', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products?minPrice=5`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(Array.isArray(data.products)).toBe(true);

    // Verify all products meet min price requirement
    data.products.forEach((product: any) => {
      expect(product.price).toBeGreaterThanOrEqual(5);
    });
  });

  test('GET /products - Filter by max price', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products?maxPrice=4`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(Array.isArray(data.products)).toBe(true);

    // Verify all products meet max price requirement
    data.products.forEach((product: any) => {
      expect(product.price).toBeLessThanOrEqual(4);
    });
  });

  test('GET /products - Search by name', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products?search=organic`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(Array.isArray(data.products)).toBe(true);
    expect(data.products.length).toBeGreaterThan(0);

    // Verify search results contain the search term
    data.products.forEach((product: any) => {
      const searchableText = `${product.name} ${product.description} ${product.dataAiHint}`.toLowerCase();
      expect(searchableText).toContain('organic');
    });
  });

  test('GET /products - Pagination', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/products?page=1&limit=5`);
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data).toHaveProperty('page', 1);
    expect(data).toHaveProperty('limit', 5);
    expect(data.products.length).toBeLessThanOrEqual(5);
  });

  test('POST /products - Missing required fields', async ({ request }) => {
    const incompleteProduct = {
      name: 'Test Product',
      // Missing price, description, image, dataAiHint
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: incompleteProduct,
    });
    expect(response.status()).toBe(400);

    const error = await response.json();
    expect(error).toHaveProperty('error');
  });

  test('POST /products - Invalid image URL', async ({ request }) => {
    const newProduct = {
      name: 'Invalid URL Product',
      price: 4.99,
      description: 'Product with invalid image URL',
      image: 'not-a-valid-url', // Invalid URL
      dataAiHint: 'test',
    };

    const response = await request.post(`${baseURL}/api/products`, {
      data: newProduct,
    });
    expect(response.status()).toBe(400);

    const error = await response.json();
    expect(error).toHaveProperty('error');
  });

  test('Discount Calculation - Verify price and discount relationship', async ({ request }) => {
    // Get product with discount
    const response = await request.get(`${baseURL}/api/products/1`);
    const product = await response.json();

    expect(product).toHaveProperty('price');
    expect(product).toHaveProperty('discount');

    const originalPrice = product.price;
    const discountPercentage = product.discount.percentage;
    const expectedDiscountedPrice = originalPrice * (1 - discountPercentage / 100);

    // Verify calculation (for reference in UI implementation)
    expect(typeof originalPrice).toBe('number');
    expect(typeof discountPercentage).toBe('number');
    expect(discountPercentage).toBeGreaterThanOrEqual(0);
    expect(discountPercentage).toBeLessThanOrEqual(100);
  });
});
