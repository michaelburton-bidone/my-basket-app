---
name: test-generation
description: Generate executable API tests from test plans in multiple formats (Playwright, Jest, Postman). Automatically detects project configuration and creates tests with proper authentication, setup, and error handling. Use when user wants to generate tests, create test files, convert test plans to code, or needs automated API test creation.
---

# API Test Generation Skill

Use this skill to convert API test plans into executable test code in Playwright, Jest, or Postman formats.

## When to Use This Skill

- User has a test plan and wants to generate executable tests
- User mentions generating Playwright tests or Jest tests
- User needs to create Postman collections
- User wants to convert test documentation to code
- User requests automated test generation

## Core Workflow

### Step 1: Project Setup Detection (REQUIRED FIRST STEP)

**ALWAYS call `api_project_setup` tool FIRST** before generating any tests:

```javascript
const setupResult = await tools.api_project_setup({
  outputDir: "./tests"  // or user-specified directory
})
```

**The tool uses Smart Detection:**
- ✅ **Has playwright.config.ts** → Auto-detects: Playwright + TypeScript
- ✅ **Has playwright.config.js** → Auto-detects: Playwright + JavaScript
- ✅ **Has jest.config.ts** → Auto-detects: Jest + TypeScript
- ✅ **Has jest.config.js** → Auto-detects: Jest + JavaScript
- ⚠️ **Has tsconfig.json only** → Asks user: Which framework?
- ❓ **No config files** → Asks user: Framework + Language

### Step 2: Handle Detection Results

**Case A: Auto-Detected Configuration**
```javascript
if (setupResult.autoDetected) {
  // Configuration found - proceed directly to generation
  const framework = setupResult.config.framework;  // 'playwright' or 'jest'
  const language = setupResult.config.language;    // 'typescript' or 'javascript'
  
  console.log(`✓ Detected ${framework} project with ${language}`);
  // Proceed to Step 3
}
```

**Case B: Partial Detection (TypeScript found, need framework)**
```javascript
if (setupResult.needsUserInput && setupResult.detected.hasTypeScript) {
  // Ask user: "I found TypeScript. Which framework would you like to use?"
  // Options: Playwright (recommended), Jest, Postman, All formats
  // After user responds, store: language='typescript', framework=<user choice>
}
```

**Case C: No Configuration (Empty project)**
```javascript
if (setupResult.needsUserInput && !setupResult.detected.hasTypeScript) {
  // Ask user both:
  // 1. Which test framework? Playwright / Jest / Postman / All
  // 2. Which language? TypeScript (recommended) / JavaScript
  // After user responds, store both values
}
```

### Step 3: Extract Test Plan Content (When User Specifies Sections)

When user requests specific sections (e.g., "generate tests for section 1"):

```javascript
// Read test plan file
const testPlanContent = await readFile("./api-test-plan.md");

// Parse sections using markdown headers (##)
// Extract requested sections preserving structure

// User says: "generate tests for section 1"
// Extract: Section at index 0 (first ## after title)

// User says: "tests for GET /api/v1/Users"
// Extract: Section(s) matching pattern in title
```

**Include base URL** from plan overview in extracted content.

### Step 4: Generate Tests with api_generator

```javascript
await tools.api_generator({
  // Use extracted content OR full plan path
  testPlanContent: extractedContent,  // for specific sections
  // OR testPlanPath: "./api-test-plan.md",  // for full plan
  
  // Use detected/chosen configuration from Step 1
  outputFormat: detectedConfig.framework,  // 'playwright', 'jest', 'postman', 'all'
  language: detectedConfig.language,       // 'typescript' or 'javascript'
  
  // Pass project info from Step 1
  projectInfo: {
    hasTypeScript: detectedConfig.hasTypeScript,
    hasPlaywrightConfig: detectedConfig.hasPlaywrightConfig,
    hasJestConfig: detectedConfig.hasJestConfig
  },
  
  // Additional parameters
  outputDir: "./tests",
  sessionId: "api-gen-" + Date.now(),
  includeAuth: true,
  includeSetup: true,
  baseUrl: "https://api.example.com"  // optional override
})
```

## Output Formats

### Playwright Tests (TypeScript/JavaScript)
- Browser-based API testing with request fixture
- Full HTTP client with assertions
- TypeScript types for better IDE support

### Jest Tests (TypeScript/JavaScript)
- Node.js API testing with axios
- Popular testing framework
- Easy integration with existing projects

### Postman Collections
- Import-ready JSON format
- Environment variables included
- Can be used in Postman or Newman

### All Formats
- Generate all three formats simultaneously
- Maximum compatibility
- Ready for different testing scenarios

## Example Generation Scenarios

### Scenario 1: New Empty Project
```
1. User asks to generate tests
2. Call api_project_setup → No config detected
3. Ask user: Framework? Language?
4. User chooses: Playwright + JavaScript
5. Call api_generator with choices
6. Generate tests + setup instructions
```

### Scenario 2: Existing TypeScript Project
```
1. User asks to generate tests
2. Call api_project_setup → Auto-detects Playwright + TypeScript
3. Call api_generator with detected config
4. Generate tests (no user input needed)
```

### Scenario 3: Specific Section Request
```
1. User: "Generate tests for section 2"
2. Call api_project_setup → Detect config
3. Read test plan and extract section 2
4. Call api_generator with extracted content + config
5. Generate tests for that section only
```

### Scenario 4: Override Auto-Detection
```
1. User: "Generate Jest tests in JavaScript"
2. Call api_project_setup (may detect Playwright)
3. User explicitly wants Jest + JS → Use user preference
4. Call api_generator with outputFormat='jest', language='javascript'
5. Generate requested format
```

## Session Management and Validation

After generation, validate using API request tools:

```javascript
// Validate generated tests work
await tools.api_request({
  sessionId: "validation-session",
  method: "POST",
  url: "https://api.example.com/auth/login",
  data: { email: "test@example.com", password: "test123" },
  expect: { status: 200 },
  extract: { token: "access_token" }
})

// Check session status
await tools.api_session_status({
  sessionId: "validation-session"
})

// Generate validation report
await tools.api_session_report({
  sessionId: "validation-session",
  outputPath: "./validation-report.html"
})
```

## Manual Test Creation (Fallback)

When automatic generation needs refinement:

```javascript
// Create custom test files
await tools.edit_createFile({
  path: "./tests/custom-api-test.spec.ts",
  content: `import { test, expect } from '@playwright/test';

test.describe('Custom API Tests', () => {
  test('should validate custom scenario', async ({ request }) => {
    const response = await request.get('https://api.example.com/custom');
    expect(response.status()).toBe(200);
  });
});`
})
```

## Best Practices

### DO:
- Always call `api_project_setup` first
- Let tool auto-detect configuration when possible
- Extract specific sections when user requests them
- Validate generated tests with api_request
- Provide clear feedback about detected configuration
- Use consistent sessionId for related operations
- Generate comprehensive reports

### DON'T:
- Don't skip project setup detection
- Don't guess configuration - detect or ask
- Don't generate without confirmation if detection fails
- Don't ignore user's explicit format preferences
- Don't generate without preserving test plan structure

## Error Handling

If test generation fails:
1. Check if project setup was called first
2. Verify test plan format is correct
3. Ensure configuration matches project structure
4. Try manual file creation as fallback
5. Provide clear error messages to user

## Generated Test Structure

### Playwright Example:
```typescript
import { test, expect } from '@playwright/test';

test.describe('Books API', () => {
  test('GET /api/v1/Books - should return all books', async ({ request }) => {
    const response = await request.get('https://api.example.com/api/v1/Books');
    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);
    const books = await response.json();
    expect(Array.isArray(books)).toBeTruthy();
  });
});
```

### Jest Example:
```typescript
import axios from 'axios';

describe('Books API', () => {
  const baseUrl = 'https://api.example.com';
  
  test('GET /api/v1/Books - should return all books', async () => {
    const response = await axios.get(`${baseUrl}/api/v1/Books`);
    expect(response.status).toBe(200);
    expect(Array.isArray(response.data)).toBe(true);
  });
});
```

## Additional Tools Available

Use these file manipulation tools when needed:
- `search/fileSearch` - Find test files
- `search/textSearch` - Search within files
- `search/listDirectory` - List test directory
- `search/readFile` - Read existing tests
- `edit/createFile` - Create new test files
- `edit/editFiles` - Modify existing tests

## Troubleshooting

### Project setup not detecting configuration
- Verify config files exist in correct location
- Check file names match expected patterns
- Manually ask user for preferences

### Generated tests don't match project structure
- Ensure project setup was called first
- Verify outputFormat matches project framework
- Check language setting matches project

### Tests fail to run after generation
- Validate with api_request tool
- Check API endpoint availability
- Verify authentication configuration
