# API Documentation

All microservices in the My Basket App now include comprehensive Swagger/OpenAPI documentation.

## 📚 Swagger Documentation URLs

### API Gateway (Unified Documentation)
**URL:** [http://localhost:3000/api-docs](http://localhost:3000/api-docs)

The API Gateway provides a unified overview and links to all service-specific documentation.

### Individual Service Documentation

#### 📦 Product Service
- **Swagger UI:** [http://localhost:3001/api-docs](http://localhost:3001/api-docs)
- **OpenAPI JSON:** [http://localhost:3001/api-docs.json](http://localhost:3001/api-docs.json)
- **Port:** 3001
- **Endpoints:**
  - `GET /api/products` - Get all products with filters and pagination
  - `GET /api/products/{id}` - Get product by ID
  - `POST /api/products` - Create a new product
  - `PUT /api/products/{id}` - Update a product
  - `DELETE /api/products/{id}` - Delete a product
  - `GET /api/categories` - Get all categories
  - `GET /api/health` - Health check

#### 🛒 Cart Service
- **Swagger UI:** [http://localhost:3002/api-docs](http://localhost:3002/api-docs)
- **OpenAPI JSON:** [http://localhost:3002/api-docs.json](http://localhost:3002/api-docs.json)
- **Port:** 3002
- **Endpoints:**
  - `GET /api/cart/{userId}` - Get user's cart
  - `POST /api/cart/{userId}/items` - Add item to cart
  - `PUT /api/cart/{userId}/items/{productId}` - Update cart item quantity
  - `DELETE /api/cart/{userId}/items/{productId}` - Remove item from cart
  - `DELETE /api/cart/{userId}` - Clear cart
  - `GET /api/cart/{userId}/summary` - Get cart summary
  - `GET /api/health` - Health check

#### 📋 Order Service
- **Swagger UI:** [http://localhost:3003/api-docs](http://localhost:3003/api-docs)
- **OpenAPI JSON:** [http://localhost:3003/api-docs.json](http://localhost:3003/api-docs.json)
- **Port:** 3003
- **Endpoints:**
  - `POST /api/orders/{userId}` - Create a new order
  - `GET /api/orders/{userId}` - Get all orders for a user
  - `GET /api/orders/{userId}/{orderId}` - Get specific order details
  - `PUT /api/orders/{userId}/{orderId}/status` - Update order status
  - `DELETE /api/orders/{userId}/{orderId}` - Cancel an order
  - `GET /api/health` - Health check

#### 🤖 AI Service
- **Swagger UI:** [http://localhost:3004/api-docs](http://localhost:3004/api-docs)
- **OpenAPI JSON:** [http://localhost:3004/api-docs.json](http://localhost:3004/api-docs.json)
- **Port:** 3004
- **Endpoints:**
  - `POST /api/recommendations/grocery-suggestions` - Get grocery suggestions
  - `POST /api/recommendations/personalized` - Get personalized recommendations
  - `POST /api/grocery-suggestions` - Legacy endpoint for suggestions
  - `GET /api/health` - Health check

## 🚀 Quick Start

1. **Start all microservices:**
   ```bash
   npm run microservices:start
   ```

2. **Access the main API documentation:**
   - Open [http://localhost:3000/api-docs](http://localhost:3000/api-docs) in your browser

3. **Explore individual services:**
   - Each service has its own detailed Swagger documentation at `/api-docs`

## 📝 Features

- **Interactive API Testing:** Try out API endpoints directly from the Swagger UI
- **Request/Response Examples:** See example payloads for all endpoints
- **Schema Definitions:** Complete data models and validation rules
- **Error Responses:** Documented error codes and messages
- **Authentication:** (Ready for future implementation)

## 🔍 Using Swagger UI

1. **Navigate to any Swagger URL** listed above
2. **Expand an endpoint** to see details
3. **Click "Try it out"** to test the endpoint
4. **Fill in parameters** and request body
5. **Click "Execute"** to send the request
6. **View the response** with status code and data

## 📦 Swagger Dependencies

Each service includes:
- `swagger-ui-express` - Swagger UI middleware
- `swagger-jsdoc` - Generate OpenAPI specs from JSDoc comments
- `@types/swagger-ui-express` - TypeScript definitions
- `@types/swagger-jsdoc` - TypeScript definitions

## 🔧 Customization

To modify the Swagger documentation:

1. **Update JSDoc comments** in route files (`src/routes.ts`)
2. **Edit Swagger config** in `src/swagger.ts`
3. **Restart the service** to see changes

## 📡 API Gateway Routing

All requests can be routed through the API Gateway at port 3000:

- `/api/products/*` → Product Service (3001)
- `/api/cart/*` → Cart Service (3002)
- `/api/orders/*` → Order Service (3003)
- `/api/recommendations/*` → AI Service (3004)

## 🛠️ Troubleshooting

If Swagger UI is not loading:

1. Check if the service is running: `curl http://localhost:300X/api/health`
2. Verify no CORS issues in browser console
3. Ensure all dependencies are installed: `npm run microservices:install`
4. Restart services: `npm run microservices:start`

## 📚 Additional Resources

- [Swagger/OpenAPI Specification](https://swagger.io/specification/)
- [swagger-jsdoc Documentation](https://github.com/Surnet/swagger-jsdoc)
- [swagger-ui-express Documentation](https://github.com/scottie1984/swagger-ui-express)
