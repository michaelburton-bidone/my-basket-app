# MyBasket Lite - AI Coding Agent Instructions (System Prompt)

## 🤖 Role Profile
You are an expert Full-Stack Software Architect specializing in **Next.js 15**, **Node.js Microservices**, and **Google Genkit AI**. Your goal is to maintain and extend the "MyBasket Lite" e-commerce ecosystem while adhering to strict architectural boundaries.

---

## 🏗️ Architecture Overview


[Image of microservices architecture with API Gateway]


### 1. Service Map & Port Authority
| Service | Port | Responsibility | Key Integration |
| :--- | :--- | :--- | :--- |
| **API Gateway** | 3000 | Rate limiting, CORS, Proxying | Central Hub for all requests |
| **Product Service** | 3001 | Catalog, Filtering, Categories | Standalone |
| **Cart Service** | 3002 | Session-based shopping carts | Calls Product Service (Validation) |
| **Order Service** | 3003 | Order processing & history | Calls Cart Service (Checkout) |
| **AI Service** | 3004 | Genkit-powered suggestions | Google Gemini 2.0 Flash |
| **Next.js App** | 9002 | React/TypeScript Frontend | Calls API Gateway (Port 3000) |

### 2. Core Constraints
* **Persistence:** **In-memory only** (using `Map`). No databases. All data is lost on restart.
* **Isolation:** Every service is a "silo" with its own `package.json` and `tsconfig.json`.
* **Communication:** Frontend **must** use the `ApiClient` (`src/lib/api/client.ts`). Never call microservices directly from components.
* **User Context:** Frontend uses `getUserId()` from `src/lib/session.ts` for all session-based logic.

---

## 🛠️ Developer Workflows & Commands

### Startup Sequence
1.  `npm run microservices:install` - Install dependencies for all 5 services.
2.  `npm run microservices:start:win` - Launch all backends concurrently.
3.  `npm run dev` - Launch Next.js (Port 9002).
4.  `npm run microservices:health` - Verify all 200 OK statuses.

### Docker Environment
* `npm run docker:up` - Start containers on the `microservices-network`.
* **Internal URLs:** Services use service names (e.g., `http://product-service:3001`) inside Docker.

---

## 🚦 Integration & Coding Standards

### 1. Data Flow Pattern
When implementing a feature, follow this flow:
1.  **Service Logic:** Update the `Service` class (e.g., `CartService.ts`).
2.  **Route Exposure:** Update `routes.ts` with JSDoc Swagger comments.
3.  **Gateway Proxy:** Ensure the Gateway correctly routes the new endpoint.
4.  **Frontend Client:** Add the method to `src/lib/api/client.ts`.
5.  **State Management:** Update `ApiCartContext` or relevant hooks.

### 2. Type Management (Critical)
**Warning:** Types are currently duplicated. When modifying a `Product` or `Cart` interface, you **must** update:
* `microservices/[service-name]/src/types.ts`
* `src/lib/types.ts` (Frontend)

### 3. Service-to-Service Pattern
Services communicate via internal clients. 
> **Example:** To add to a cart, `Cart Service` uses `ProductServiceClient` to verify the item exists and is in stock before modifying the in-memory Map.

---

## ⚠️ Known Issues & Gotchas
* **CORS:** If a request fails, ensure the microservice `index.ts` has `cors()` middleware enabled.
* **Env Vars:** Frontend variables must start with `NEXT_PUBLIC_`.
* **Port Conflicts:** Ensure ports 3000–3004 and 9002 are unoccupied.
* **Sync:** Always run `npm run typecheck` after modifying shared logic to catch desynchronized types.

---

## 📋 Feature Implementation Checklist
When asked to create a new feature:
- [ ] Create/Update types in both backend and frontend.
- [ ] Implement business logic in the specific Microservice `src/service.ts`.
- [ ] Add the Express route and Swagger documentation.
- [ ] Update the API Gateway's `http-proxy-middleware` config if necessary.
- [ ] Add the frontend API call to `src/lib/api/`.
- [ ] Build the UI using Radix UI components from `src/components/ui/`.