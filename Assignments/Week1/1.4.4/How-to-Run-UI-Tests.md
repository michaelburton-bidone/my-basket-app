# UI Tests - Complete Running Instructions

## Quick Start

```bash
# Navigate to the UI tests directory
cd ui-tests

# Install dependencies
npm install

# Run all tests with 6 workers (recommended)
npm test -- --workers=6

# View the test report
npx playwright show-report
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup & Installation](#setup--installation)
3. [Running Tests](#running-tests)
4. [Viewing Reports](#viewing-reports)
5. [Debugging Tests](#debugging-tests)
6. [Configuration Options](#configuration-options)
7. [Common Commands](#common-commands)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before running the UI tests, ensure you have:

- **Node.js** v16+ installed
- **npm** v7+ installed
- **Git** installed (for version control)
- A terminal/command prompt
- ~500MB disk space for dependencies and reports

### Check Prerequisites

```bash
# Check Node.js version
node --version
# Expected: v16.0.0 or higher

# Check npm version
npm --version
# Expected: v7.0.0 or higher
```

---

## Setup & Installation

### Step 1: Navigate to UI Tests Directory

```bash
cd c:\Work\TalentDojo\Mike\my-basket-app\ui-tests
```

Or from the project root:

```bash
cd my-basket-app
cd ui-tests
```

### Step 2: Install Dependencies

```bash
# Install all dependencies from package.json
npm install

# This will install:
# - @playwright/test (testing framework)
# - playwright browsers (Chromium, Firefox, WebKit)
# - TypeScript
# - Other dependencies
```

### Step 3: Verify Installation

```bash
# Check if playwright is installed
npx playwright --version

# Check if browsers are installed
npx playwright install --with-deps
```

### Step 4: (Optional) Start Application Server

If testing against a local server, ensure it's running:

```bash
# From the project root (in another terminal)
cd my-basket-app
npm run dev
# Server should start on http://localhost:3000
```

---

## Running Tests

### Basic Commands

#### Run All Tests (Recommended)
```bash
# Run all tests with 6 parallel workers (default, fastest)
npm test -- --workers=6
```

**Expected Output:**
```
Running 19 tests using 6 workers

✓  1 [chromium] › tests\add-product-to-cart.spec.ts (6.6s)
✓  2 [chromium] › tests\add-product-to-cart.spec.ts (7.1s)
...
19 passed (32.2s)
```

#### Run with Single Worker (Slower, More Stable)
```bash
# Run tests sequentially (one at a time)
npm test -- --workers=1
```

#### Run Specific Test File
```bash
# Run only add-product-to-cart tests
npm test -- add-product-to-cart.spec.ts

# Run only ui-validation tests
npm test -- ui-validation.spec.ts

# Run with specific worker count
npm test -- add-product-to-cart.spec.ts --workers=2
```

#### Run Specific Test by Name
```bash
# Run tests matching a pattern (case-insensitive)
npm test -- -g "should add first product"

# Run empty cart message test
npm test -- -g "should show empty cart message"

# Run all toast-related tests
npm test -- -g "toast"

# Run all cart-related tests
npm test -- -g "cart"
```

#### Run with Different Worker Counts
```bash
# Run with 1 worker (sequential, most stable)
npm test -- --workers=1

# Run with 2 workers
npm test -- --workers=2

# Run with 4 workers
npm test -- --workers=4

# Run with 6 workers (default, fastest)
npm test -- --workers=6

# Run with maximum available workers
npm test -- --workers=0
```

#### Run with Tags
```bash
# Run all tests tagged with @cart
npm test -- --grep "@cart"

# Run all tests tagged with @ui
npm test -- --grep "@ui"

# Run all error handling tests
npm test -- --grep "@error-handling"
```

#### Run with Additional Options
```bash
# Run tests and update snapshots
npm test -- --update-snapshots

# Run tests with headed browser (see browser window)
npm test -- --headed

# Run tests with debug mode
npm test -- --debug

# Run tests and keep browser open
npm test -- --headed --workers=1
```

---

## Viewing Reports

### After Test Execution

```bash
# Open the HTML report (automatically generated)
npx playwright show-report

# This opens an interactive report showing:
# - All test results
# - Duration of each test
# - Screenshots on failure
# - Video recordings
# - Trace files for debugging
```

### Report Contents

The report includes:

- ✅/❌ **Test Results** - Pass/fail status for each test
- ⏱️ **Execution Time** - Duration for each test and total
- 📸 **Screenshots** - Captured on test failure
- 🎬 **Videos** - Full test execution recordings
- 🔍 **Traces** - Interactive execution traces for debugging
- 📊 **Stats** - Total tests, pass rate, duration

### Finding Previous Reports

```bash
# Reports are stored in:
test-results/
  ├── add-product-to-cart-[test-name]-chromium/
  │   ├── test-failed-1.png
  │   ├── video.webm
  │   └── trace.zip
  └── ui-validation-[test-name]-chromium/
      ├── test-failed-1.png
      ├── video.webm
      └── trace.zip
```

### View Specific Trace Files

```bash
# View a specific trace file directly
npx playwright show-trace test-results/[test-folder]/trace.zip

# Example:
npx playwright show-trace test-results/add-product-to-cart-Add-Pr-17b43-and-show-toast-notification-chromium/trace.zip
```

---

## Debugging Tests

### Debug Mode (Interactive)

```bash
# Run single test in debug mode
npm test -- add-product-to-cart.spec.ts --debug

# Browser opens with Playwright Inspector
# Step through test execution line by line
# Inspect elements in real-time
# Execute commands in console
```

### Headed Mode (See Browser)

```bash
# Run tests with visible browser window
npm test -- --headed

# See what's happening on screen during test
# Good for understanding test behavior
# Less stable than headless mode
```

### Single Test with Headed Mode

```bash
# Perfect for debugging a specific test
npm test -- -g "should add first product" --headed --workers=1
```

### Screenshot on Failure

```bash
# Screenshots are automatically captured on failure
# View them in the test report:
npx playwright show-report

# Or manually check:
ls test-results/*/test-failed-*.png
```

### View Video Recordings

```bash
# Videos are captured for all tests
# View them in the test report:
npx playwright show-report

# Or manually:
ls test-results/*/video.webm
```

---

## Configuration Options

### Playwright Configuration
**File:** `playwright.config.ts`

```typescript
// Key configuration settings:

// Browser to test
use: {
  browser: 'chromium',    // chromium, firefox, webkit
}

// Base URL for navigation
baseURL: 'http://localhost:3000',

// Timeout for test actions
timeout: 30000,           // 30 seconds

// Screenshot on failure
screenshot: 'only-on-failure',

// Video recording
video: 'retain-on-failure',

// Trace recording
trace: 'on-first-retry',
```

### Override Configuration via CLI

```bash
# Override base URL
npm test -- --base-url=http://localhost:3001

# Override timeout (in milliseconds)
npm test -- --timeout=60000

# Use specific browser
npm test -- --project=chromium
npm test -- --project=firefox
```

---

## Common Commands Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `npm test` | Run all tests | `npm test -- --workers=6` |
| `npm test -- [file]` | Run specific file | `npm test -- add-product-to-cart.spec.ts` |
| `npm test -- -g [pattern]` | Run matching tests | `npm test -- -g "toast"` |
| `npm test -- --headed` | See browser window | `npm test -- --headed --workers=1` |
| `npm test -- --debug` | Debug mode | `npm test -- --debug --workers=1` |
| `npx playwright show-report` | View test report | `npx playwright show-report` |
| `npm test -- --help` | Show all options | `npm test -- --help` |

---

## Complete Test Execution Examples

### Example 1: Run All Tests with Report

```bash
cd ui-tests
npm test -- --workers=6
npx playwright show-report
```

**Use Case:** Full test suite execution after code changes

---

### Example 2: Debug a Single Failing Test

```bash
cd ui-tests
npm test -- -g "should add first product" --headed --debug --workers=1
```

**Use Case:** Debugging why a specific test is failing

---

### Example 3: Run Add-to-Cart Tests Only

```bash
cd ui-tests
npm test -- add-product-to-cart.spec.ts --workers=4
npx playwright show-report
```

**Use Case:** Testing only cart functionality after changes

---

### Example 4: Run UI Validation Tests

```bash
cd ui-tests
npm test -- ui-validation.spec.ts --workers=4
npx playwright show-report
```

**Use Case:** Validating UI components and layout

---

### Example 5: Run Empty Cart Message Test

```bash
cd ui-tests
npm test -- -g "should show empty cart message" --headed --workers=1
```

**Use Case:** Testing the required empty cart message feature

---

### Example 6: Run with Video Capture (Slower)

```bash
cd ui-tests
npm test -- --workers=1
npx playwright show-report
# Watch videos in report
```

**Use Case:** Recording test execution for documentation

---

## Test File Structure

```
ui-tests/
├── tests/
│   ├── add-product-to-cart.spec.ts    (7 tests)
│   └── ui-validation.spec.ts          (12 tests)
├── src/
│   ├── pages/                         (Page Objects)
│   │   ├── BasePage.ts
│   │   ├── ProductPage.ts
│   │   ├── CartPage.ts
│   │   └── CheckoutPage.ts
│   ├── fixtures/                      (Test Fixtures)
│   │   └── page-fixtures.ts
│   ├── types/                         (TypeScript Types)
│   └── utils/                         (Utilities)
├── config/
│   ├── environments.ts                (Environment Config)
│   └── test-data.ts                   (Test Data)
├── playwright.config.ts               (Main Config)
├── tsconfig.json                      (TypeScript Config)
└── package.json                       (Dependencies)
```

---

## Troubleshooting

### Issue 1: Tests Timeout

**Problem:** Tests fail with timeout error

```
Error: Timeout of 30000ms exceeded
```

**Solution:**
```bash
# Increase timeout for specific test
npm test -- --timeout=60000

# Or run with single worker (less resource contention)
npm test -- --workers=1
```

---

### Issue 2: Element Not Found

**Problem:** Tests fail because element can't be found

```
Error: locator('[data-testid="..."]').first()
  Expected: visible
  Timeout: 5000ms
```

**Solution:**
```bash
# Run in headed mode to see what's happening
npm test -- -g "[test name]" --headed --workers=1

# Or run with debug mode
npm test -- -g "[test name]" --debug
```

---

### Issue 3: Application Not Running

**Problem:** Tests fail with connection refused

```
Error: connect ECONNREFUSED 127.0.0.1:3000
```

**Solution:**
```bash
# Start the application first (in another terminal)
npm run dev

# Then run tests
npm test -- --workers=6
```

---

### Issue 4: Playwright Browsers Not Installed

**Problem:** Tests fail with browser installation error

```
Error: Playwright browsers are not installed
```

**Solution:**
```bash
# Install browsers
npx playwright install

# Or with system dependencies
npx playwright install --with-deps
```

---

### Issue 5: Port Already in Use

**Problem:** Application won't start due to port conflict

```
Error: listen EADDRINUSE: address already in use :::3000
```

**Solution:**
```bash
# Kill process on port 3000 (Windows)
netstat -ano | findstr :3000
taskkill /PID [PID] /F

# Or use different port
PORT=3001 npm run dev
```

---

### Issue 6: Insufficient Memory

**Problem:** Tests fail with out of memory error

```
Error: JavaScript heap out of memory
```

**Solution:**
```bash
# Run with fewer workers
npm test -- --workers=2

# Or increase Node memory limit
set NODE_OPTIONS=--max-old-space-size=4096
npm test -- --workers=6
```

---

### Issue 7: Tests Pass Locally but Fail in CI

**Problem:** Tests work on local machine but fail in CI/CD

**Solution:**
```bash
# Run with single worker (more stable in CI)
npm test -- --workers=1

# Run without headed/debug modes
npm test -- --workers=1

# Check environment-specific config
cat config/environments.ts
```

---

## Performance Tips

### Speed Up Test Execution

1. **Use Multiple Workers**
   ```bash
   npm test -- --workers=6
   # Default for fast execution
   ```

2. **Run Only Changed Tests**
   ```bash
   npm test -- -g "[pattern]"
   # Only run tests matching pattern
   ```

3. **Disable Video/Screenshots**
   Edit `playwright.config.ts`:
   ```typescript
   video: 'off',           // Disable video
   screenshot: 'off',      // Disable screenshots
   ```

4. **Use Faster Browser**
   ```typescript
   // chromium is fastest, webkit is slowest
   // Use chromium for normal testing
   ```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/ui-tests.yml
name: UI Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: 18
      - run: cd ui-tests && npm install
      - run: npm test -- --workers=4
      - uses: actions/upload-artifact@v2
        if: always()
        with:
          name: test-results
          path: ui-tests/test-results/
```

---

## Additional Resources

### Playwright Documentation
- Official Docs: https://playwright.dev
- API Reference: https://playwright.dev/docs/api/class-test
- Best Practices: https://playwright.dev/docs/best-practices

### Project Documentation
- UI Test Summary: `./1.4.4-UI-Test-Summary.md`
- Test Cases Coverage: `./Test-Cases-Coverage-Table.md`
- Page Object Models: `ui-tests/src/pages/`

---

## Summary

### Quick Reference Card

```
# Navigate to tests
cd ui-tests

# Install dependencies
npm install

# Run all tests
npm test -- --workers=6

# View report
npx playwright show-report

# Debug specific test
npm test -- -g "test name" --headed --debug

# Run specific file
npm test -- add-product-to-cart.spec.ts

# Run with single worker
npm test -- --workers=1
```

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review test files in `tests/` directory
3. Check `playwright.config.ts` for configuration
4. Run with `--debug` flag for more information
5. View test report for screenshots and videos

---

**Last Updated:** February 3, 2026  
**Test Framework:** Playwright + TypeScript  
**Status:** ✅ All Tests Passing (19/19)
