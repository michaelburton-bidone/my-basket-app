📦 UI Testing Framework - Complete File Structure
================================================

ui-tests/
│
├── 📄 Configuration Files
│   ├── package.json                    # NPM dependencies and scripts
│   ├── tsconfig.json                   # TypeScript configuration
│   ├── playwright.config.ts            # Playwright test configuration
│   ├── .eslintrc.json                  # ESLint configuration
│   └── .gitignore                      # Git ignore patterns
│
├── 📚 Documentation
│   ├── README.md                       # Quick start guide
│   ├── GUIDE.md                        # Comprehensive implementation guide
│   ├── FRAMEWORK_SUMMARY.md            # Complete framework summary
│   └── FILE_TREE.md                    # This file
│
├── 🔐 Global Setup
│   └── global-setup.ts                 # Auth and global state setup
│
├── 📁 Source Code (src/)
│   │
│   ├── pages/                          # Page Object Models
│   │   ├── BasePage.ts                 # Base class (30+ methods)
│   │   ├── ProductPage.ts              # Product listing page
│   │   ├── CartPage.ts                 # Shopping cart page
│   │   ├── CheckoutPage.ts             # Checkout page
│   │   └── index.ts                    # Export all pages
│   │
│   ├── utils/                          # Utility Functions
│   │   ├── waiters.ts                  # 8 wait utility functions
│   │   ├── assertions.ts               # 12 assertion functions
│   │   ├── helpers.ts                  # 18 general utilities
│   │   └── index.ts                    # Export all utils
│   │
│   ├── types/                          # TypeScript Interfaces
│   │   ├── page.types.ts               # Page-related types
│   │   ├── test.types.ts               # Test data types
│   │   └── index.ts                    # Export all types
│   │
│   └── fixtures/                       # Playwright Fixtures
│       ├── page-fixtures.ts            # Page object injection
│       └── index.ts                    # Export fixtures
│
├── ⚙️ Configuration (config/)
│   ├── environments.ts                 # Environment URLs and settings
│   └── test-data.ts                    # Test data and fixtures
│
└── 🧪 Test Specifications (tests/)
    ├── cart-addition.spec.ts           # Happy path tests (7 tests)
    │   └── @smoke @cart tags
    │
    ├── cart-crud.spec.ts               # CRUD operations (7 tests)
    │   └── @cart tag
    │
    ├── checkout.spec.ts                # Checkout flow (8 tests)
    │   └── @checkout tag
    │
    ├── ui-validation.spec.ts           # UI validation (12 tests)
    │   └── @ui tag
    │
    └── form-validation.spec.ts         # Form validation (10 tests)
        └── @validation tag


📊 FILE STATISTICS
==================

Total Files Created: 35
Total Test Scenarios: 44

Breakdown by Category:
├── Configuration & Setup Files: 5
├── Page Object Models: 4
├── Utility Functions: 3
├── Type Definitions: 3
├── Fixtures: 2
├── Configuration Files: 2
├── Documentation: 4
├── Global Setup: 1
└── Test Specifications: 5


🔍 KEY FILE DESCRIPTIONS
========================

PROJECT SETUP
─────────────
package.json
  - Playwright v1.45.0 and TypeScript v5.3.3
  - Test scripts: test, test:ui, test:debug, test:headed, test:smoke, test:cart
  - Report viewing: report, codegen

playwright.config.ts
  - Parallel execution settings
  - 2 retries in CI, 0 locally
  - 60s global timeout, 10s expect timeout
  - Trace recording on first retry
  - Screenshot and video on failure
  - HTML, JSON, JUnit reporters
  - Chrome/Chromium browser support

tsconfig.json
  - ES2020 target with strict mode
  - Path aliases: @pages, @utils, @types, @config, @fixtures
  - Esmodule and import support
  - Declaration map and source maps

.eslintrc.json
  - TypeScript ESLint configuration
  - Recommended preset with best practices
  - Warning for explicit any types
  - Console method restrictions


PAGE OBJECT MODELS (src/pages/)
──────────────────────────────

BasePage.ts (Base Class)
  Methods:
  - Navigation: navigateTo(), navigateToHome()
  - Clicks: click()
  - Input: fill(), typeText(), clearInput()
  - Text: getText(), getInputValue()
  - State: isVisible(), isEnabled(), isChecked()
  - Waits: waitForElement(), waitForElementCount(), waitForNavigation()
  - Info: getCurrentUrl(), getPageTitle(), getAttribute()
  - Utils: hover(), takeScreenshot(), reload(), pressKey()

ProductPage.ts
  Locators:
  - productContainer, productItem, productName, productPrice, productImage
  - addToCartButton, viewDetailsButton
  - filterButton, sortDropdown
  - searchInput, searchButton
  - paginationContainer, nextPageButton, previousPageButton
  - loadingSpinner, emptyStateMessage
  
  Methods:
  - navigateTo(), waitForProductsToLoad()
  - getProductCount(), getProductByIndex()
  - addProductToCartByIndex(), addProductToCartByName()
  - getProductIndexByName()
  - searchProduct(), sortProducts(), applyFilter()
  - goToNextPage(), goToPreviousPage()
  - isEmptyStateShown(), isLoading()
  - getAllProductNames(), getAllProductPrices()

CartPage.ts
  Locators:
  - cartItemContainer, cartItemName, cartItemPrice, cartItemQuantity
  - quantityInput, increaseQuantityButton, decreaseQuantityButton
  - removeButton
  - subtotalAmount, taxAmount, totalAmount, discountAmount
  - checkoutButton, clearCartButton, continueShoppingButton
  - emptyCartMessage
  - promoCodeInput, applyPromoButton, promoError
  
  Methods:
  - navigateTo(), waitForCartToLoad()
  - getCartItemCount(), isCartEmpty()
  - getCartItemByIndex(), getAllCartItems()
  - getCartTotals()
  - updateQuantity(), increaseQuantity(), decreaseQuantity()
  - removeItem(), removeItemByName(), clearCart()
  - applyPromoCode(), getPromoErrorMessage()
  - proceedToCheckout(), continueShopping()
  - assertProductInCart(), assertCartTotal()
  - isCheckoutButtonEnabled(), getDiscountAmount()

CheckoutPage.ts
  Locators:
  - firstNameInput, lastNameInput, emailInput, phoneInput
  - addressInput, cityInput, stateInput, zipCodeInput, countrySelect
  - shippingMethodContainer, standardShipping, expressShipping, overnightShipping
  - cardNumberInput, cardNameInput, cardExpiryInput, cardCVVInput
  - billingAddressCheckbox
  - orderReviewContainer, orderItemContainer, orderSubtotal, orderTotal
  - continueToBillingButton, continueToPaymentButton, placeOrderButton
  - formError, fieldError, successMessage, orderNumber, checkoutStep
  
  Methods:
  - navigateTo()
  - fillShippingInfo(), fillShippingAddress(), fillCheckoutForm()
  - selectShippingMethod()
  - fillPaymentInfo(), useSameAddressForBilling()
  - continueToBilling(), continueToPayment(), placeOrder(), goBack()
  - getFormError(), getFieldError(), getSuccessMessage()
  - getOrderNumber(), getOrderTotals()
  - isStepActive(), getOrderItemCount()
  - validateOrderSummaryMatches(), isPlaceOrderButtonEnabled()
  - completeCheckout() - Full workflow


UTILITY FUNCTIONS (src/utils/)
──────────────────────────────

waiters.ts (8 Functions)
  - waitForElementVisible()
  - waitForElementHidden()
  - waitForElementsVisible()
  - waitForElementStable()
  - waitForCondition()
  - waitForElementCount()
  - waitForText()
  - waitForPageLoad()
  - waitForResponse()
  - waitForElements()

assertions.ts (12 Functions)
  - assertElementVisible()
  - assertElementHidden()
  - assertElementEnabled()
  - assertElementDisabled()
  - assertElementContainsText()
  - assertElementHasText()
  - assertElementHasAttribute()
  - assertElementChecked()
  - assertElementNotChecked()
  - assertElementCount()
  - assertUrlContains()
  - assertUrlEquals()
  - assertPageTitle()
  - assertPageTitleContains()
  - assertElementInViewport()
  - assertElementHasClass()
  - assertTextInPage()
  - assertInputValue()

helpers.ts (18 Functions)
  - formatCurrency()
  - parseCurrency()
  - generateUniqueId()
  - getRandomItem()
  - getRandomNumber()
  - sleep()
  - retry()
  - extractNumbers()
  - extractFirstNumber()
  - formatDate()
  - elementExists()
  - getAllLocatorTexts()
  - isValidEmail()
  - isValidPhoneNumber()
  - getCurrentTimestamp()
  - buildUrl()
  - delayExecution()


TYPES (src/types/)
──────────────────

page.types.ts
  - PageLocators
  - NavigationOptions
  - ElementState
  - ClickOptions
  - TextInputOptions
  - WaitOptions

test.types.ts
  - Product
  - CartItem
  - Cart
  - CheckoutData
  - User
  - AuthCredentials
  - TestResult
  - ValidationError


FIXTURES (src/fixtures/)
────────────────────────

page-fixtures.ts
  Custom test fixture providing:
  - productPage: ProductPage instance
  - cartPage: CartPage instance
  - checkoutPage: CheckoutPage instance
  
  Usage:
  test('test name', async ({ productPage, cartPage }) => {
    // Page objects available
  });


CONFIGURATION (config/)
──────────────────────

environments.ts
  - Development configuration
    baseUrl: http://localhost:9002
    apiUrl: http://localhost:3001/api
    timeout: 30000
    retries: 1
  
  - Staging configuration
    baseUrl: https://staging.basket.app
    apiUrl: https://staging.basket.app/api
    timeout: 60000
    retries: 2
  
  - Production configuration
    baseUrl: https://basket.app
    apiUrl: https://basket.app/api
    timeout: 60000
    retries: 3
  
  Functions:
  - getEnvironment()
  - getBaseUrl()
  - getApiUrl()
  - buildUrl()

test-data.ts
  Sample Data:
  - sampleProducts (5 products)
  - testCreditCards (Visa, MasterCard, AMEX, Invalid)
  - testPromoCodes (Valid 10%, 20%, Expired, Invalid)
  
  Generators:
  - generateTestUser()
  - generateCheckoutData()
  - generateRandomCheckoutData()
  
  Utilities:
  - getSampleProductByName()
  - getSampleProductByIndex()
  - createCartItem()
  - calculateCartTotal()
  - getTestCreditCard()
  - getTestPromoCode()


TEST SPECIFICATIONS (tests/)
───────────────────────────

cart-addition.spec.ts (7 Tests) @smoke @cart
  1. should add single product to cart
  2. should add multiple products to cart
  3. should increase product quantity in cart
  4. should remove product from cart
  5. should clear entire cart
  6. should calculate cart totals correctly
  7. should persist cart after navigation

cart-crud.spec.ts (7 Tests) @cart
  1. should read cart items correctly
  2. should update item quantity correctly
  3. should recalculate totals after quantity change
  4. should handle duplicate products correctly
  5. should remove specific item by name
  6. should validate item prices in cart match product prices
  7. should validate cart empty state

checkout.spec.ts (8 Tests) @checkout
  1. should complete full checkout process
  2. should validate order summary
  3. should require all shipping information
  4. should handle different shipping methods
  5. should validate payment information
  6. should display order summary with all items
  7. should use same address for billing and shipping
  8. should go back to previous checkout steps

ui-validation.spec.ts (12 Tests) @ui
  1. should display product list correctly
  2. should display product information correctly
  3. should have functional add to cart buttons
  4. should show and hide loading spinner appropriately
  5. should display cart correctly
  6. should display cart totals correctly
  7. should show empty cart message when no items
  8. should enable/disable checkout button based on cart state
  9. should display correct product count in cart
  10. should have visible product prices
  11. should display quantity controls in cart
  12. should maintain visual hierarchy in checkout form

form-validation.spec.ts (10 Tests) @validation
  1. should validate required fields in checkout form
  2. should validate email format
  3. should validate phone number format
  4. should validate zip code format
  5. should validate card number format
  6. should validate CVV format
  7. should show error for empty required field
  8. should clear errors when correcting input
  9. should validate address fields are not empty
  10. should handle special characters in name fields
  11. should validate country selection


📋 USAGE EXAMPLES
=================

# Install
npm install
npx playwright install

# Run all tests
npm test

# Run with UI
npm run test:ui

# Run smoke tests only
npm run test:smoke

# Run cart tests only
npm run test:cart

# Debug mode
npm run test:debug

# Generate code from browser
npm run codegen

# View HTML report
npm run report

# Set environment
TEST_ENV=staging npm test
BASE_URL=http://localhost:3000 npm test


🎯 FEATURE HIGHLIGHTS
====================

✅ Type-Safe Development
   - Full TypeScript support
   - Interfaces for all data structures
   - Strict mode enabled

✅ Page Object Model
   - Locators centralized
   - Common methods in BasePage
   - DRY principle throughout

✅ Fixture Injection
   - Page objects auto-injected
   - Dependency management
   - Clean test code

✅ Comprehensive Utilities
   - 8 waiter functions
   - 12 assertion functions
   - 18 helper functions

✅ Test Data Management
   - Sample products and users
   - Random data generation
   - Test credit cards and promo codes

✅ Environment Configuration
   - Dev, staging, prod support
   - Dynamic URL switching
   - Timeout and retry configuration

✅ CI/CD Ready
   - Auto-detect CI environment
   - 2 retries in CI
   - Trace recording
   - JUnit XML reports
   - Screenshot on failure
   - Video recording

✅ Comprehensive Testing
   - 44 test scenarios
   - Happy path tests
   - CRUD operations
   - Form validation
   - UI validation
   - Checkout flow

✅ Well Documented
   - README with quick start
   - GUIDE with detailed instructions
   - Code comments throughout
   - This file tree


End of File Tree
================
