/**
 * Checkout Page Object Model
 * 
 * Handles interactions with the checkout page
 */

import { Page } from '@playwright/test';
import { BasePage } from './BasePage';
import { CheckoutData } from '@types/test.types';

export class CheckoutPage extends BasePage {
  // Shipping Information
  readonly firstNameInput = this.page.getByLabel(/first name/i);
  readonly lastNameInput = this.page.getByLabel(/last name/i);
  readonly emailInput = this.page.getByLabel(/email/i);
  readonly phoneInput = this.page.getByLabel(/phone/i);

  // Shipping Address
  readonly addressInput = this.page.getByLabel(/address|street/i);
  readonly cityInput = this.page.getByLabel(/city/i);
  readonly stateInput = this.page.getByLabel(/state|province/i);
  readonly zipCodeInput = this.page.getByLabel(/zip code|postal code/i);
  readonly countrySelect = this.page.getByLabel(/country/i);

  // Shipping Method
  readonly shippingMethodContainer = this.page.locator('[data-testid="shipping-method"]');
  readonly standardShipping = this.page.locator('[data-testid="shipping-standard"]');
  readonly expressShipping = this.page.locator('[data-testid="shipping-express"]');
  readonly overnightShipping = this.page.locator('[data-testid="shipping-overnight"]');

  // Payment Information
  readonly cardNumberInput = this.page.getByLabel(/card number/i);
  readonly cardNameInput = this.page.getByLabel(/name on card/i);
  readonly cardExpiryInput = this.page.getByLabel(/expiry|expiration/i);
  readonly cardCVVInput = this.page.getByLabel(/cvv|cvc/i);

  // Billing Address
  readonly billingAddressCheckbox = this.page.getByLabel(/billing address|same as shipping/i);
  readonly billingAddressInput = this.page.getByLabel(/billing address/i);

  // Order Review
  readonly orderReviewContainer = this.page.locator('[data-testid="order-review"]');
  readonly orderItemContainer = this.page.locator('[data-testid="order-item"]');
  readonly orderSubtotal = this.page.locator('[data-testid="order-subtotal"]');
  readonly orderTax = this.page.locator('[data-testid="order-tax"]');
  readonly orderShipping = this.page.locator('[data-testid="order-shipping"]');
  readonly orderTotal = this.page.locator('[data-testid="order-total"]');

  // Action buttons
  readonly continueToBillingButton = this.page.getByRole('button', { name: /continue.*billing/i });
  readonly continueToPaymentButton = this.page.getByRole('button', { name: /continue.*payment/i });
  readonly placeOrderButton = this.page.getByRole('button', { name: /place order|submit/i });
  readonly backButton = this.page.getByRole('button', { name: /back|previous/i });

  // Form errors
  readonly formError = this.page.locator('[data-testid="form-error"]');
  readonly fieldError = (fieldName: string) =>
    this.page.locator(`[data-testid="error-${fieldName.toLowerCase()}"]`);

  // Success message
  readonly successMessage = this.page.locator('[data-testid="success-message"]');
  readonly orderNumber = this.page.locator('[data-testid="order-number"]');

  // Progress indicator
  readonly checkoutStep = (step: number) =>
    this.page.locator(`[data-testid="step-${step}"]`);

  constructor(page: Page) {
    super(page);
  }

  /**
   * Navigate to checkout page
   */
  async navigateTo(path: string = '/checkout'): Promise<void> {
    await super.navigateTo(path);
  }

  /**
   * Fill shipping information
   */
  async fillShippingInfo(data: Partial<CheckoutData>): Promise<void> {
    if (data.firstName) await this.fill(this.firstNameInput, data.firstName);
    if (data.lastName) await this.fill(this.lastNameInput, data.lastName);
    if (data.email) await this.fill(this.emailInput, data.email);
    if (data.phone) await this.fill(this.phoneInput, data.phone);
  }

  /**
   * Fill shipping address
   */
  async fillShippingAddress(data: Partial<CheckoutData>): Promise<void> {
    if (data.address) await this.fill(this.addressInput, data.address);
    if (data.city) await this.fill(this.cityInput, data.city);
    if (data.state) await this.fill(this.stateInput, data.state);
    if (data.zipCode) await this.fill(this.zipCodeInput, data.zipCode);
    if (data.country) {
      await this.click(this.countrySelect);
      const countryOption = this.page.getByRole('option', { name: data.country });
      await this.click(countryOption);
    }
  }

  /**
   * Fill complete checkout form
   */
  async fillCheckoutForm(data: CheckoutData): Promise<void> {
    await this.fillShippingInfo(data);
    await this.fillShippingAddress(data);
  }

  /**
   * Select shipping method
   */
  async selectShippingMethod(method: 'standard' | 'express' | 'overnight'): Promise<void> {
    let button;
    switch (method) {
      case 'express':
        button = this.expressShipping;
        break;
      case 'overnight':
        button = this.overnightShipping;
        break;
      default:
        button = this.standardShipping;
    }
    await this.click(button);
  }

  /**
   * Fill payment information
   */
  async fillPaymentInfo(
    cardNumber: string,
    cardName: string,
    expiry: string,
    cvv: string
  ): Promise<void> {
    await this.fill(this.cardNumberInput, cardNumber);
    await this.fill(this.cardNameInput, cardName);
    await this.fill(this.cardExpiryInput, expiry);
    await this.fill(this.cardCVVInput, cvv);
  }

  /**
   * Use same address for billing
   */
  async useSameAddressForBilling(): Promise<void> {
    const isChecked = await this.isChecked(this.billingAddressCheckbox);
    if (!isChecked) {
      await this.click(this.billingAddressCheckbox);
    }
  }

  /**
   * Continue to billing
   */
  async continueToBilling(): Promise<void> {
    await this.click(this.continueToBillingButton);
  }

  /**
   * Continue to payment
   */
  async continueToPayment(): Promise<void> {
    await this.click(this.continueToPaymentButton);
  }

  /**
   * Place order
   */
  async placeOrder(): Promise<void> {
    await this.click(this.placeOrderButton);
  }

  /**
   * Go back
   */
  async goBack(): Promise<void> {
    await this.click(this.backButton);
  }

  /**
   * Get form error message
   */
  async getFormError(): Promise<string | null> {
    const isVisible = await this.isVisible(this.formError);
    if (isVisible) {
      return this.getText(this.formError);
    }
    return null;
  }

  /**
   * Get field error message
   */
  async getFieldError(fieldName: string): Promise<string | null> {
    const error = this.fieldError(fieldName);
    const isVisible = await this.isVisible(error);
    if (isVisible) {
      return this.getText(error);
    }
    return null;
  }

  /**
   * Get success message
   */
  async getSuccessMessage(): Promise<string | null> {
    const isVisible = await this.isVisible(this.successMessage);
    if (isVisible) {
      return this.getText(this.successMessage);
    }
    return null;
  }

  /**
   * Get order number
   */
  async getOrderNumber(): Promise<string | null> {
    const isVisible = await this.isVisible(this.orderNumber);
    if (isVisible) {
      return this.getText(this.orderNumber);
    }
    return null;
  }

  /**
   * Get order totals
   */
  async getOrderTotals(): Promise<{
    subtotal: number;
    tax: number;
    shipping: number;
    total: number;
  }> {
    const subtotalText = await this.getText(this.orderSubtotal);
    const taxText = await this.getText(this.orderTax);
    const shippingText = await this.getText(this.orderShipping);
    const totalText = await this.getText(this.orderTotal);

    return {
      subtotal: parseFloat(subtotalText.replace(/[^0-9.]/g, '')),
      tax: parseFloat(taxText.replace(/[^0-9.]/g, '')),
      shipping: parseFloat(shippingText.replace(/[^0-9.]/g, '')),
      total: parseFloat(totalText.replace(/[^0-9.]/g, '')),
    };
  }

  /**
   * Check if current step is visible
   */
  async isStepActive(step: number): Promise<boolean> {
    const stepElement = this.checkoutStep(step);
    return this.isVisible(stepElement);
  }

  /**
   * Get order item count
   */
  async getOrderItemCount(): Promise<number> {
    return this.orderItemContainer.count();
  }

  /**
   * Validate order summary matches cart
   */
  async validateOrderSummaryMatches(expectedTotal: number): Promise<void> {
    const totals = await this.getOrderTotals();
    if (Math.abs(totals.total - expectedTotal) > 0.01) {
      throw new Error(`Order total mismatch. Expected: ${expectedTotal}, Got: ${totals.total}`);
    }
  }

  /**
   * Check if place order button is enabled
   */
  async isPlaceOrderButtonEnabled(): Promise<boolean> {
    return this.isEnabled(this.placeOrderButton);
  }

  /**
   * Fill and complete checkout
   */
  async completeCheckout(
    checkoutData: CheckoutData,
    cardNumber: string,
    cardName: string,
    expiry: string,
    cvv: string,
    shippingMethod: 'standard' | 'express' | 'overnight' = 'standard'
  ): Promise<void> {
    // Step 1: Shipping Info
    await this.fillCheckoutForm(checkoutData);
    await this.continueToBilling();

    // Step 2: Select shipping
    await this.selectShippingMethod(shippingMethod);
    await this.continueToPayment();

    // Step 3: Payment
    await this.fillPaymentInfo(cardNumber, cardName, expiry, cvv);
    await this.useSameAddressForBilling();

    // Step 4: Place order
    await this.placeOrder();
  }
}
