/**
 * Response validation utilities
 */

import { ApiResponse, ValidationResult } from '@types/index';
import { logger } from './logger';

export class ResponseValidator {
  /**
   * Validate HTTP status code
   */
  static validateStatus(
    status: number,
    expectedStatus: number | number[]
  ): ValidationResult {
    const expected = Array.isArray(expectedStatus)
      ? expectedStatus
      : [expectedStatus];

    const isValid = expected.includes(status);

    return {
      isValid,
      errors: isValid
        ? []
        : [
            `Expected status ${expected.join(' or ')}, but got ${status}`,
          ],
    };
  }

  /**
   * Validate response structure
   */
  static validateResponseStructure(
    response: any,
    requiredFields: string[]
  ): ValidationResult {
    const errors: string[] = [];

    for (const field of requiredFields) {
      if (!(field in response)) {
        errors.push(`Required field '${field}' is missing in response`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate response field types
   */
  static validateFieldTypes(
    response: any,
    fieldTypes: Record<string, string>
  ): ValidationResult {
    const errors: string[] = [];

    for (const [field, expectedType] of Object.entries(fieldTypes)) {
      if (!(field in response)) {
        errors.push(`Field '${field}' is missing`);
        continue;
      }

      const actualType = typeof response[field];
      if (actualType !== expectedType) {
        errors.push(
          `Field '${field}' should be of type '${expectedType}' but got '${actualType}'`
        );
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate response contains expected data
   */
  static validateResponseData(
    response: any,
    expectedData: Record<string, any>
  ): ValidationResult {
    const errors: string[] = [];

    for (const [key, expectedValue] of Object.entries(expectedData)) {
      if (!(key in response)) {
        errors.push(`Expected field '${key}' not found in response`);
        continue;
      }

      if (response[key] !== expectedValue) {
        errors.push(
          `Field '${key}' expected '${expectedValue}' but got '${response[key]}'`
        );
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate array response
   */
  static validateArrayResponse(
    response: any,
    isArray: boolean = true
  ): ValidationResult {
    const isValid = Array.isArray(response) === isArray;

    return {
      isValid,
      errors: isValid
        ? []
        : [
            `Expected response to be ${isArray ? 'an array' : 'not an array'}, but got ${typeof response}`,
          ],
    };
  }

  /**
   * Log validation errors
   */
  static logValidationErrors(errors: string[], testName: string): void {
    if (errors.length > 0) {
      logger.error(`Validation failed for ${testName}:`, errors);
    }
  }

  /**
   * Assert validation result
   */
  static assertValid(
    result: ValidationResult,
    message: string = 'Validation failed'
  ): void {
    if (!result.isValid) {
      const errorMessage = `${message}\nErrors: ${result.errors.join('\n')}`;
      throw new Error(errorMessage);
    }
  }
}
