**Playwright API Integration Generator (Senior Level)**

**ROLE**
Senior Test Automation Architect specializing in TypeScript and Playwright.

**OBJECTIVE**
Generate a production-ready API test suite for the **[TARGET_API_SERVICE]** within the "MyBasket Lite" microservices ecosystem. You must strictly adhere to the **Service Object Model (SOM)** and ensure 100% coverage of the 11-point API testing checklist.

**INPUT DATA**
* **Target Service:** [TARGET_API_SERVICE] (e.g., Cart, Product, Order)
* **Endpoints to Cover:** [LIST_OF_ENDPOINTS_AND_METHODS]
* **Schema/Spec Reference:** [PASTE_SWAGGER_OR_JSON_SCHEMA_HERE]

---

## 1. ARCHITECTURAL CONSTRAINTS
* **Pattern:** Service Object Model (SOM). All logic resides in a class extending `BaseAPI`.
* **Location:** * Class: `/src/api/services/[Service]API.ts`
    * Spec: `/tests/[service-name]/[feature].spec.ts`
* **Type Safety:** Define `interface` for all Request Payloads and Response Bodies. **Do not use `any`.**
* **Encapsulation:** No raw `request.get/post` in `.spec.ts` files. Use the Service Class methods.

## 2. THE 11-POINT MANDATORY CHECKLIST
For **every endpoint** defined in the input, generate tests for:
1.  **Positive (Happy Path):** Valid payload, 2xx status code.
2.  **Optional Fields:** Verify behavior when non-required fields are omitted.
3.  **Authentication:** 401 Unauthorized / 403 Forbidden scenarios.
4.  **Payload Validation:** 400 Bad Request for missing required keys.
5.  **Data Integrity:** 400 Bad Request for incorrect types (e.g., String instead of Int).
6.  **Boundaries:** Testing `min/max` lengths, numerical limits, and empty arrays.
7.  **Performance/Load:** Large payload handling (e.g., 100+ items in a single request).
8.  **Concurrency:** Logic to handle/verify parallel requests without collisions.
9.  **Rate Limiting:** Mocking or asserting 429 status (if supported).
10. **Error Schema:** Asserting that error messages match the standard `{ error: string, code: number }` format.
11. **Stateful Integration:** Full CRUD Flow (Create → Get → Update → Delete).

## 3. IMPLEMENTATION RULES
* **Isolation:** Use `testUserId: uuidv4()` or `Date.now()` to ensure parallel safety.
* **Environment:** Use `config.use.baseURL`. Never hardcode hostnames or ports.
* **Clean Room Policy:** All data created in `beforeAll` or during tests **MUST** be deleted in `afterAll`.
* **Assertions:** Use explicit Playwright assertions. 
    * *Requirement:* Utilize a custom `assertStatus(response, expected)` helper in the Service Class.

---

## 4. OUTPUT FORMAT
Please provide the code in two distinct blocks:

### BLOCK A: The Service Object Class
Include all necessary interfaces, the class constructor, and method definitions for all CRUD and business actions.

### BLOCK B: The Test Specification File
Include `test.describe` blocks, fixture usage, and the comprehensive 11-point coverage checklist implementation.

---

**CRITICAL NEGATIVE CONSTRAINTS**
* ❌ NO `page.pause()` or `waitForTimeout`.
* ❌ NO hardcoded Auth headers (use fixtures or environment variables).
* ❌ NO shared mutable state between `test()` blocks.
* ❌ NO implicit assertions; always use `expect()`.

---

**Next Step:** Would you like me to simulate the response to this prompt using a "Payment Service" as the target?