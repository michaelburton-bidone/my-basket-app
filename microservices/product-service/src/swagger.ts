import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';
import { Express } from 'express';

const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Product Service API',
      version: '1.0.0',
      description: 'API documentation for the Product Management microservice',
      contact: {
        name: 'API Support',
        email: 'support@example.com',
      },
    },
    servers: [
      {
        url: 'http://localhost:3001',
        description: 'Development server',
      },
      {
        url: 'http://localhost:3000',
        description: 'API Gateway',
      },
    ],
    tags: [
      {
        name: 'Products',
        description: 'Product management endpoints',
      },
      {
        name: 'Health',
        description: 'Health check endpoints',
      },
    ],
    components: {
      schemas: {
        Product: {
          type: 'object',
          required: ['id', 'name', 'price', 'description', 'image', 'dataAiHint'],
          properties: {
            id: {
              type: 'string',
              description: 'Product unique identifier',
              example: 'prod_123',
            },
            name: {
              type: 'string',
              description: 'Product name',
              example: 'Organic Bananas',
            },
            price: {
              type: 'number',
              format: 'float',
              description: 'Product price',
              example: 2.99,
            },
            description: {
              type: 'string',
              description: 'Product description',
              example: 'Fresh organic bananas',
            },
            image: {
              type: 'string',
              format: 'uri',
              description: 'Product image URL',
              example: 'https://example.com/images/banana.jpg',
            },
            dataAiHint: {
              type: 'string',
              description: 'AI hint for recommendations',
              example: 'fruit, organic, potassium',
            },
            category: {
              type: 'string',
              description: 'Product category',
              example: 'fruits',
            },
            inStock: {
              type: 'boolean',
              description: 'Stock availability',
              example: true,
            },
            discount: {
              type: 'object',
              description: 'Discount information',
              properties: {
                percentage: {
                  type: 'number',
                  format: 'float',
                  description: 'Discount percentage (0-100)',
                  example: 15,
                },
                endsAt: {
                  type: 'string',
                  format: 'date-time',
                  description: 'Discount end date/time',
                  example: '2026-02-16T10:00:00Z',
                },
              },
            },
          },
        },
        ProductList: {
          type: 'object',
          properties: {
            products: {
              type: 'array',
              items: {
                $ref: '#/components/schemas/Product',
              },
            },
            pagination: {
              type: 'object',
              properties: {
                total: { type: 'number', example: 100 },
                page: { type: 'number', example: 1 },
                limit: { type: 'number', example: 20 },
                totalPages: { type: 'number', example: 5 },
              },
            },
          },
        },
        Error: {
          type: 'object',
          properties: {
            error: {
              type: 'string',
              example: 'Error message',
            },
            details: {
              type: 'array',
              items: {
                type: 'object',
              },
            },
          },
        },
      },
    },
  },
  apis: ['./src/routes.ts', './src/index.ts'],
};

const swaggerSpec = swaggerJsdoc(options);

export const setupSwagger = (app: Express) => {
  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
    customCss: '.swagger-ui .topbar { display: none }',
    customSiteTitle: 'Product Service API Docs',
  }));
  
  app.get('/api-docs.json', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.send(swaggerSpec);
  });
};
