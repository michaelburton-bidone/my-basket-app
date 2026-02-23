# Agent Workflow Configuration & API Testing

## Overview

This document outlines the agent workflow setup for the MyBasket Lite e-commerce platform and includes comprehensive API testing documentation for the microservices architecture.

---

## 📋 Table of Contents

1. [VS Code Configuration](#vs-code-configuration)
2. [MCP Server Setup](#mcp-server-setup)
3. [Product Service API Testing](#product-service-api-testing)
4. [Running Tests](#running-tests)

---

## VS Code Configuration

### Settings Setup

The VS Code configuration for this project is located in `.vscode/settings.json`:

```jsonc
{
    "IDX.aI.enableInlineCompletion": true,
    "IDX.aI.enableCodebaseIndexing": true,
    "git.ignoreLimitWarning": true,
    "yaml.schemas": {
        "https://www.artillery.io/schema.json": []
    }
}
```

**Configuration Details:**
- **IDX.aI.enableInlineCompletion**: Enables AI-powered inline code completion
- **IDX.aI.enableCodebaseIndexing**: Enables codebase indexing for better AI context
- **git.ignoreLimitWarning**: Suppresses git ignore file size warnings
- **yaml.schemas**: Adds Artillery.io schema for YAML file validation

### MCP Server Configuration

The Model Context Protocol (MCP) server is configured in `.vscode/mcp.json`:

```json
{
  "servers": {
    "democratize-quality": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "@democratize-quality/mcp-server@latest"
      ],
      "cwd": "${workspaceFolder}",
      "env": {
        "OUTPUT_DIR": "./api-test-reports"
      }
    }
  },
  "inputs": []
}
```

**MCP Server Details:**
- **Server Name**: democratize-quality
- **Type**: stdio (Standard Input/Output)
- **Command**: npx @democratize-quality/mcp-server@latest
- **Working Directory**: Project root (${workspaceFolder})
- **Output Directory**: ./api-test-reports
- **Purpose**: Provides AI-powered API testing and quality assurance capabilities

---

## Product Service API Testing

### Service Overview

**Base URL**: http://localhost:3001
**Documentation**: Available at http://localhost:3001/api-docs

The Product Service is responsible for:
- Managing product catalog
- Handling product filtering and search
- Managing product inventory
- Providing product recommendations based on AI hints

### API Endpoints

#### 1. Get All Products
**Endpoint**: `GET /api/products`

**Description**: Retrieve a paginated list of products with optional filters

**Query Parameters**:
- `category` (string, optional): Filter by product category
- `minPrice` (number, optional): Minimum price filter
- `maxPrice` (number, optional): Maximum price filter
- `inStock` (boolean, optional): Filter for in-stock items
- `search` (string, optional): Search products by name or description
- `page` (number, optional): Page number (default: 1)
- `limit` (number, optional): Items per page (max: 100, default: 10)

**Example Request**:
```bash
GET http://localhost:3001/api/products?category=Electronics&minPrice=100&maxPrice=1000&page=1&limit=10
```

**Expected Response** (200 OK):
```json
{
  "products": [
    {
      "id": "prod_001",
      "name": "Wireless Headphones",
      "price": 79.99,
      "description": "High-quality wireless headphones with noise cancellation",
      "image": "https://example.com/headphones.jpg",
      "dataAiHint": "audio-electronics",
      "category": "Electronics",
      "inStock": true,
      "discount": null,
      "createdAt": "2025-10-19T10:30:00Z",
      "updatedAt": "2025-10-19T10:30:00Z"
    }
  ],
  "total": 42,
  "page": 1,
  "limit": 10,
  "hasMore": true
}
```

#### 2. Get Product by ID
**Endpoint**: `GET /api/products/:id`

**Description**: Retrieve a specific product by its ID

**Path Parameters**:
- `id` (string, required): Product ID

**Example Request**:
```bash
GET http://localhost:3001/api/products/prod_001
```

**Expected Response** (200 OK):
```json
{
  "id": "prod_001",
  "name": "Wireless Headphones",
  "price": 79.99,
  "description": "High-quality wireless headphones with noise cancellation",
  "image": "https://example.com/headphones.jpg",
  "dataAiHint": "audio-electronics",
  "category": "Electronics",
  "inStock": true,
  "discount": null,
  "createdAt": "2025-10-19T10:30:00Z",
  "updatedAt": "2025-10-19T10:30:00Z"
}
```

#### 3. Create Product
**Endpoint**: `POST /api/products`

**Description**: Create a new product (Admin only)

**Request Body**:
```json
{
  "name": "Wireless Headphones",
  "price": 79.99,
  "description": "High-quality wireless headphones with noise cancellation",
  "image": "https://example.com/headphones.jpg",
  "dataAiHint": "audio-electronics",
  "category": "Electronics",
  "inStock": true,
  "discount": {
    "percentage": 15,
    "endsAt": "2025-12-31T23:59:59Z"
  }
}
```

**Expected Response** (201 Created):
```json
{
  "id": "prod_new_001",
  "name": "Wireless Headphones",
  "price": 79.99,
  "description": "High-quality wireless headphones with noise cancellation",
  "image": "https://example.com/headphones.jpg",
  "dataAiHint": "audio-electronics",
  "category": "Electronics",
  "inStock": true,
  "discount": {
    "percentage": 15,
    "endsAt": "2025-12-31T23:59:59Z"
  },
  "createdAt": "2025-10-19T10:30:00Z",
  "updatedAt": "2025-10-19T10:30:00Z"
}
```

#### 4. Update Product
**Endpoint**: `PUT /api/products/:id`

**Description**: Update an existing product (Admin only)

**Path Parameters**:
- `id` (string, required): Product ID

**Request Body** (all fields optional):
```json
{
  "name": "Updated Product Name",
  "price": 89.99,
  "description": "Updated description",
  "inStock": true,
  "discount": {
    "percentage": 20,
    "endsAt": "2025-12-31T23:59:59Z"
  }
}
```

**Expected Response** (200 OK):
```json
{
  "id": "prod_001",
  "name": "Updated Product Name",
  "price": 89.99,
  "description": "Updated description",
  "image": "https://example.com/headphones.jpg",
  "dataAiHint": "audio-electronics",
  "category": "Electronics",
  "inStock": true,
  "discount": {
    "percentage": 20,
    "endsAt": "2025-12-31T23:59:59Z"
  },
  "createdAt": "2025-10-19T10:30:00Z",
  "updatedAt": "2025-10-19T10:30:00Z"
}
```

#### 5. Delete Product
**Endpoint**: `DELETE /api/products/:id`

**Description**: Delete a product (Admin only)

**Path Parameters**:
- `id` (string, required): Product ID

**Expected Response** (204 No Content):
```
(empty response body)
```

#### 6. Health Check
**Endpoint**: `GET /health`

**Description**: Health check endpoint for service monitoring

**Expected Response** (200 OK):
```json
{
  "status": "healthy",
  "service": "product-service",
  "port": 3001,
  "uptime": 3600
}
```

---

## Running Tests

### Prerequisites

1. Ensure all microservices are running:
   ```bash
   npm run microservices:start:win  # Windows
   npm run microservices:start      # macOS/Linux
   ```

2. Verify Product Service is accessible:
   ```bash
   curl http://localhost:3001/health
   ```

3. Access Swagger documentation:
   - Open http://localhost:3001/api-docs in your browser

### Test Execution

#### Manual Testing with curl

Test get all products:
```bash
curl -X GET "http://localhost:3001/api/products?category=Electronics&page=1&limit=10"
```

Test get product by ID:
```bash
curl -X GET "http://localhost:3001/api/products/prod_001"
```

Test create product:
```bash
curl -X POST "http://localhost:3001/api/products" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Product",
    "price": 99.99,
    "description": "Test product description",
    "image": "https://example.com/test.jpg",
    "dataAiHint": "test-category",
    "category": "Electronics",
    "inStock": true
  }'
```

#### API Test Plan Generation

To generate a comprehensive test plan for the Product Service:

```bash
npm run api:plan -- --schema=http://localhost:3001/api-docs
```

Or using the MCP server:

```bash
# Via the democratize-quality MCP server
# Results saved to: ./api-test-reports/product-service-test-plan.md
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | OK | Product retrieved successfully |
| 201 | Created | Product created successfully |
| 204 | No Content | Product deleted successfully |
| 400 | Bad Request | Invalid query parameters or request body |
| 404 | Not Found | Product ID does not exist |
| 500 | Internal Server Error | Server error |

### Error Response Format

```json
{
  "error": "Product not found",
  "code": "PRODUCT_NOT_FOUND",
  "statusCode": 404
}
```

---

## Workflow Best Practices

1. **Always verify health status** before running tests
2. **Use pagination** for large product lists
3. **Validate input data** before sending requests
4. **Test edge cases** (empty results, boundary values)
5. **Monitor response times** for performance issues
6. **Document API changes** in this file
7. **Use staging environment** for non-destructive testing

---

## Integration with AI Agent

The AI agent (Copilot) uses this configuration to:

1. **Understand codebase structure** via `enableCodebaseIndexing`
2. **Generate code completions** via `enableInlineCompletion`
3. **Generate test plans** via `democratize-quality` MCP server
4. **Output test reports** to `./api-test-reports` directory

### Using the AI Agent for API Testing

Ask Copilot to:
- "Generate a test plan for the Product Service API"
- "Create integration tests for the product endpoints"
- "Validate the Product Service against its OpenAPI spec"
- "Document the API response schemas"

---

## References

- [MyBasket Lite Architecture](../README.md)
- [Product Service Implementation](../microservices/product-service/README.md)
- [API Test Reports](../api-test-reports/)
- [Swagger Documentation](http://localhost:3001/api-docs)
- [OpenAPI Specification](http://localhost:3001/api-docs.json)

---

**Last Updated**: February 23, 2026
**Status**: Active
**Maintainer**: Development Team
