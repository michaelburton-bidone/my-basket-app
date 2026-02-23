---
name: test-healing
description: Debug and automatically fix failing API tests using intelligent analysis and healing strategies. Repairs authentication issues, endpoint changes, schema mismatches, and assertion failures. Use when tests are failing, broken, or need debugging after API changes.
---

# API Test Healing Skill

Use this skill to automatically diagnose and fix failing API tests through systematic analysis and intelligent healing strategies.

## When to Use This Skill

- User mentions failing tests or broken tests
- Tests are not passing after API changes
- Authentication errors in API tests
- Schema mismatches or validation failures
- User needs to debug API test failures
- Tests need updating after API deployment

## Core Workflow

### Step 1: Automated Healing (Primary Approach)

**ALWAYS use `api_healer` tool first** for automated healing:

```javascript
// Heal specific test file
await tools.api_healer({
  testPath: "./tests/api-tests.spec.js",
  testType: "auto",                    // jest, playwright, postman, auto
  sessionId: "healing-session",
  maxHealingAttempts: 3,
  autoFix: true,
  backupOriginal: true,
  healingStrategies: [
    "schema-update",
    "endpoint-fix",
    "auth-repair",
    "data-correction",
    "assertion-update"
  ]
})

// Heal multiple test files
await tools.api_healer({
  testFiles: [
    "./tests/auth.test.js",
    "./tests/users.test.js",
    "./tests/orders.test.js"
  ],
  autoFix: true,
  analysisOnly: false  // set to true for analysis without fixing
})
```

### Step 2: Healing Strategies

The `api_healer` tool implements these automatic strategies:

#### Schema Update
- Detects API response schema changes
- Updates test assertions to match current API structure
- Fixes property name changes and type mismatches

#### Endpoint Fix
- Identifies 404 errors from changed endpoints
- Attempts to discover correct endpoint URLs
- Updates test configurations with working endpoints

#### Authentication Repair
- Fixes expired or invalid authentication tokens
- Updates authentication methods and headers
- Repairs OAuth flows and API key issues

#### Data Correction
- Fixes request payload validation errors
- Updates test data to match current API requirements
- Corrects data types and required fields

#### Assertion Update
- Updates test expectations based on actual API responses
- Fixes assertion mismatches and validation rules
- Aligns test expectations with current API behavior

### Step 3: Manual Healing (When Automated Healing Needs Assistance)

#### Analysis Mode
```javascript
// Use analysis-only mode first
await tools.api_healer({
  testPath: "./failing-test.js",
  analysisOnly: true,
  sessionId: "analysis-session"
})
```

#### Diagnostic Testing
```javascript
// Test authentication manually
await tools.api_request({
  sessionId: "diagnostic-session",
  method: "POST",
  url: "https://api.example.com/auth/login",
  data: { email: "test@example.com", password: "test123" },
  expect: { status: [200, 401] },
  extract: { token: "access_token" }
})

// Test failing endpoint with auth
await tools.api_request({
  sessionId: "diagnostic-session",
  method: "GET",
  url: "https://api.example.com/protected-resource",
  headers: { "Authorization": "Bearer {{token}}" },
  expect: { status: [200, 401, 403, 404] }
})
```

#### Targeted Fixes
```javascript
// Apply specific healing strategies
await tools.api_healer({
  testPath: "./failing-test.js",
  healingStrategies: ["auth-repair"],  // Target specific issues
  maxHealingAttempts: 1,
  autoFix: true
})
```

## Common Failure Categories

### 1. Authentication Failures
**Symptoms:** 401 Unauthorized, 403 Forbidden responses

**Causes:**
- Expired tokens
- Invalid credentials
- Missing authorization headers

**Healing Actions:**
- Refresh authentication flow
- Update token generation
- Fix header format

### 2. Request Format Issues
**Symptoms:** 400 Bad Request, validation errors

**Causes:**
- Invalid JSON
- Missing required fields
- Incorrect data types

**Healing Actions:**
- Update request payload structure
- Fix field mappings
- Correct data types

### 3. Response Validation Mismatches
**Symptoms:** Test assertions failing on response content

**Causes:**
- API schema changes
- New fields added/removed
- Data format updates

**Healing Actions:**
- Update validation rules
- Adjust expected response structure
- Fix assertion patterns

### 4. Endpoint Changes
**Symptoms:** 404 Not Found, method not allowed errors

**Causes:**
- API versioning
- Endpoint deprecation
- URL structure changes

**Healing Actions:**
- Update endpoint URLs
- Change HTTP methods
- Handle API versioning

### 5. Data Dependencies
**Symptoms:** Tests failing due to missing/invalid test data

**Causes:**
- Test data cleanup
- External service dependencies
- Race conditions

**Healing Actions:**
- Improve test data setup
- Add proper cleanup
- Handle async operations

## Debugging Methodology

### 1. Parse Error Messages
- Analyze validation failures, HTTP errors, timeout issues
- Examine differences between expected and actual responses
- Identify error patterns

### 2. Compare Expected vs Actual
- Look at response structure differences
- Check data type mismatches
- Identify missing or extra fields

### 3. Trace Request Flow
- Follow the complete request/response cycle
- Identify break points in the flow
- Check intermediate steps

### 4. Check Dependencies
- Verify prerequisite API calls work
- Validate test data setup
- Confirm authentication is valid

## Session Management and Reporting

```javascript
// Check healing session status
await tools.api_session_status({
  sessionId: "healing-session"
})

// Generate healing report
await tools.api_session_report({
  sessionId: "healing-session",
  outputPath: "./healing-report.html"
})
```

## Best Practices

### DO:
- Use api_healer tool immediately for automated fixing
- Create backups before making changes (default: true)
- Analyze patterns in multiple failing tests
- Apply targeted healing strategies when possible
- Validate fixes by re-running tests
- Document changes made during healing
- Use session tracking for comprehensive analysis

### DON'T:
- Don't manually edit tests before trying automated healing
- Don't ignore error patterns across multiple tests
- Don't skip analysis mode for complex failures
- Don't apply all strategies blindly
- Don't forget to verify fixes work
- Don't lose track of what was changed

## Standard Test Pattern Examples

### Jest Test Structure
```javascript
const axios = require('axios');

describe('API Test Suite', () => {
  const baseUrl = 'https://api.example.com/v1';
  let authToken;

  beforeAll(async () => {
    // Setup code
    const authResponse = await axios.post(`${baseUrl}/auth/login`, {
      username: 'test',
      password: 'password'
    });
    authToken = authResponse.data.token;
  });

  test('should get data successfully', async () => {
    const response = await axios.get(`${baseUrl}/data`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('id');
  });
});
```

### Playwright Test Structure
```typescript
import { test, expect } from '@playwright/test';

test.describe('API Test Suite', () => {
  let authToken: string;

  test.beforeAll(async ({ request }) => {
    const response = await request.post('/auth/login', {
      data: { username: 'test', password: 'password' }
    });
    const data = await response.json();
    authToken = data.token;
  });

  test('should get data successfully', async ({ request }) => {
    const response = await request.get('/data', {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty('id');
  });
});
```

## Common Fix Patterns

### Authentication Fix
```javascript
// Before (failing)
const response = await axios.get('/protected-endpoint');

// After (fixed with proper auth)
const response = await axios.get('/protected-endpoint', {
  headers: {
    'Authorization': `Bearer ${authToken}`,
    'Content-Type': 'application/json'
  }
});
```

### Response Validation Fix
```javascript
// Before (failing due to wrong expectations)
expect(response.data.items).toHaveLength(10);

// After (fixed based on actual API response)
expect(Array.isArray(response.data.items)).toBe(true);
expect(response.data.items.length).toBeGreaterThan(0);
```

### Error Handling Improvement
```javascript
// Before (basic error handling)
try {
  const response = await axios.get('/endpoint');
  expect(response.status).toBe(200);
} catch (error) {
  throw error;
}

// After (proper error handling)
test('should handle 404 errors correctly', async () => {
  await expect(axios.get('/nonexistent-endpoint'))
    .rejects
    .toMatchObject({
      response: {
        status: 404
      }
    });
});
```

## Output Requirements

When healing tests, provide:

1. **Detailed failure analysis** - Categorize error types
2. **Applied healing strategies** - List what was attempted
3. **Fixed test files** - Show modified code with backups
4. **Healing report** - Summary of changes and success rate
5. **Recommendations** - Suggest how to prevent similar issues
6. **Verification status** - Confirm tests now pass

## Error Resolution Strategy

- Make reasonable assumptions and fix issues automatically
- Fix one issue at a time and re-test to verify
- Use standard test patterns (Jest/Playwright conventions)
- Preserve test coverage while updating for API changes
- Improve test maintainability during fixes
- If API is fundamentally broken, mark tests as skipped with comments
- Continue until all tests pass or are properly documented

## Additional Tools Available

Use these tools for manual investigation when needed:
- `search/fileSearch` - Find test files
- `search/textSearch` - Search for error patterns
- `search/readFile` - Read test file contents
- `edit/editFiles` - Manually fix tests after analysis
- `search/listDirectory` - List test directories

## Troubleshooting

### Automated healing not working
- Check if test file path is correct
- Verify test type detection is accurate
- Try analysis mode first to understand issues
- Apply targeted strategies instead of all

### Tests still failing after healing
- Use api_request for manual endpoint testing
- Check if API is actually working
- Validate authentication separately
- Consider if API changes are breaking

### Cannot identify failure cause
- Use analysis-only mode
- Check session status for more details
- Generate healing report for comprehensive view
- Test API endpoints manually with api_request
