---
name: api-planning
description: Analyze API schemas (OpenAPI, Swagger, GraphQL) and create comprehensive test plans with realistic sample data and optional endpoint validation. Use when user mentions API testing, test plans, test coverage, API documentation, schema analysis, REST APIs, GraphQL APIs, test scenarios, or needs to plan API test strategy.
---

# API Test Planning Skill

Use this skill to analyze API schemas and generate comprehensive test plans with realistic, context-aware sample data.

## When to Use This Skill

- User provides an API schema URL or file path
- User mentions creating test plans for APIs
- User needs to analyze API documentation
- User wants to validate API endpoints
- User requests test coverage analysis for REST or GraphQL APIs

## Core Workflow

### Step 1: Generate Test Plan with Realistic Samples

When user provides a schema URL or file path, use the `api_planner` tool from the democratize-quality MCP server:

```javascript
await tools.api_planner({
  schemaUrl: "https://api.example.com/swagger.json",
  // OR schemaPath: "./schema.graphql" for local files
  apiBaseUrl: "https://api.example.com",
  includeAuth: true,
  includeSecurity: true,
  includeErrorHandling: true,
  outputPath: "./api-test-plan.md",
  testCategories: ["functional", "security", "performance", "integration", "edge-cases"],
  validateEndpoints: false  // Set to true for live validation
})
```

**Important Workflow Rules:**
1. Call `api_planner` tool ONCE to generate the test plan
2. Review the results and explain what was generated to the user
3. Answer questions based on the generated output
4. Only call api_planner again if user explicitly requests a new/different test plan

### Step 2: Optional Endpoint Validation

When API is accessible, enable validation to verify schemas match reality:

```javascript
await tools.api_planner({
  schemaUrl: "https://api.example.com/swagger.json",
  validateEndpoints: true,
  validationSampleSize: 3,  // Default: 3, use -1 for all endpoints
  validationTimeout: 5000   // 5 seconds per request
})
```

**Validation Features:**
- Real API testing with actual responses
- Response time metrics
- Success/failure indicators (✅/❌)
- Validation summary with success rate
- Graceful error handling

## What You Get

### Without Validation (Fast):
- Test plan with realistic sample data
- Context-aware field values (names, emails, dates)
- Ready-to-use test data

### With Validation (Comprehensive):
- Test plan with realistic samples
- Validation summary (success rate, statistics)
- Per-endpoint validation results
- Actual API responses captured
- Response time metrics for each endpoint

## Working with Schema Files

### GraphQL SDL Files (.graphql, .gql)
**ALWAYS use `schemaPath` parameter for SDL files:**

```javascript
await tools.api_planner({
  schemaPath: "./schema.graphql",  // Tool reads full file, converts SDL to introspection
  outputPath: "./test-plan.md"
})
```

**What happens:**
- Tool reads full file (no truncation)
- Automatically converts SDL → Introspection JSON
- Saves `schema.json` alongside `schema.graphql`
- Generates test plan from introspection

### OpenAPI/Swagger Files
- Use `schemaPath` for local files (`.json`, `.yaml`, `.yml`)
- Use `schemaUrl` for remote URLs

## Realistic Sample Data Generation

The tool generates context-aware sample data automatically:

**50+ Field Patterns:**
- Personal info: `firstName` → "John", `email` → "john.doe@example.com"
- Contact: `phoneNumber` → "+1-555-0123", `city` → "New York"
- Business: `company` → "Acme Corp", `jobTitle` → "Engineer"
- Identifiers: `uuid` → valid UUID, `token` → realistic token
- Content: `title` → "Report Title", `description` → meaningful text
- Numeric: `price` → 19.99, `age` → 25, `rating` → 4.5
- Dates: `createdAt` → "2026-02-15T10:30:00Z"

## Advanced Scenarios

### Scenario 1: Local Schema File
```javascript
await tools.api_planner({
  schemaPath: "./openapi.json",
  apiBaseUrl: "https://staging-api.example.com",
  validateEndpoints: true
})
```

### Scenario 2: GraphQL API
```javascript
await tools.api_planner({
  schemaUrl: "https://api.example.com/graphql",
  schemaType: "graphql",
  testCategories: ["functional", "edge-cases"]
})
```

### Scenario 3: Multiple Related Services
```javascript
const services = [
  { url: "https://api1.example.com/swagger.json", name: "auth-service" },
  { url: "https://api2.example.com/swagger.json", name: "user-service" }
]

for (const service of services) {
  await tools.api_planner({
    schemaUrl: service.url,
    outputPath: `./${service.name}-test-plan.md`
  })
}
```

## Best Practices

### DO:
- Use api_planner once per schema
- Enable validation when API is accessible
- Use schemaPath for local files (especially GraphQL SDL)
- Review validation results for API issues
- Save plans to files using outputPath parameter
- Include security and edge case testing

### DON'T:
- Don't call api_planner multiple times without user request
- Don't regenerate plans just to answer questions
- Don't skip validation if API is accessible
- Don't ignore failed validations
- Don't validate all endpoints for large APIs (use validationSampleSize)

## Parameter Reference

**Required (one of):**
- `schemaUrl` - URL to fetch schema
- `schemaPath` - Local file path

**Common Optional:**
- `apiBaseUrl` - Override base URL from schema
- `outputPath` - Save test plan to file
- `includeAuth` - Include auth scenarios (default: false)
- `includeSecurity` - Include security scenarios (default: false)
- `includeErrorHandling` - Include error scenarios (default: false)
- `testCategories` - Array of test types

**Validation Optional:**
- `validateEndpoints` - Enable actual API testing (default: false)
- `validationSampleSize` - Number to validate (default: 3, -1 = all)
- `validationTimeout` - Timeout in ms (default: 5000)

## Troubleshooting

### Issue: Validation Not Running
- Ensure `validateEndpoints: true` is set explicitly

### Issue: All Validations Fail with 401/403
- API requires authentication
- Use api_request tool to test with proper auth headers
- Document auth requirements in manual review

### Issue: Validation Timeout
- Increase `validationTimeout` parameter (e.g., 10000 for 10s)

### Issue: Schema Parse Errors
- Verify schema URL is accessible
- Check schema format compatibility
- Try using `schemaPath` for local files

## Additional Validation Tools

If you need to validate specific endpoints manually:

```javascript
await tools.api_request({
  sessionId: "validation-session",
  method: "GET",
  url: "https://api.example.com/endpoint",
  expect: { status: 200 }
})

// Check session status
await tools.api_session_status({
  sessionId: "validation-session"
})

// Generate report
await tools.api_session_report({
  sessionId: "validation-session",
  outputPath: "./validation-report.html"
})
```
