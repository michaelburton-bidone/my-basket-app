/**
 * Error handling utilities
 */

import { logger } from './logger';

export class ApiError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public response?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class ValidationError extends Error {
  constructor(public errors: string[]) {
    super(`Validation failed: ${errors.join(', ')}`);
    this.name = 'ValidationError';
  }
}

export class NetworkError extends Error {
  constructor(
    public message: string,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class ErrorHandler {
  /**
   * Handle API errors
   */
  static handleApiError(statusCode: number, response: any): void {
    let message = `API Error (${statusCode})`;

    if (response?.error) {
      message += `: ${response.error}`;
    } else if (response?.message) {
      message += `: ${response.message}`;
    }

    logger.error(message, response);
    throw new ApiError(statusCode, message, response);
  }

  /**
   * Handle validation errors
   */
  static handleValidationError(errors: string[]): void {
    logger.error('Validation failed', errors);
    throw new ValidationError(errors);
  }

  /**
   * Handle network errors
   */
  static handleNetworkError(error: Error): void {
    logger.error('Network error occurred', error);
    throw new NetworkError('Network request failed', error);
  }

  /**
   * Check if error is retryable
   */
  static isRetryable(error: any): boolean {
    if (error instanceof NetworkError) {
      return true;
    }

    if (error instanceof ApiError) {
      // Retry on 5xx errors, 429 (Too Many Requests), and 408 (Request Timeout)
      return error.statusCode >= 500 || error.statusCode === 429 || error.statusCode === 408;
    }

    return false;
  }

  /**
   * Retry mechanism
   */
  static async retry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;

        if (!this.isRetryable(error)) {
          throw error;
        }

        if (i < maxRetries - 1) {
          logger.warn(
            `Request failed (attempt ${i + 1}/${maxRetries}), retrying in ${delayMs}ms...`
          );
          await new Promise((resolve) => setTimeout(resolve, delayMs));
          delayMs *= 2; // Exponential backoff
        }
      }
    }

    throw lastError || new Error('Max retries exceeded');
  }
}
