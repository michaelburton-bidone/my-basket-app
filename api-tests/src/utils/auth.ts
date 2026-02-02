/**
 * Authentication handler for different auth types
 */

import { AuthConfig } from '@types/index';
import { logger } from './logger';

export class AuthHandler {
  /**
   * Generate authorization headers based on auth configuration
   */
  static getAuthHeaders(authConfig: AuthConfig): Record<string, string> {
    const headers: Record<string, string> = {};

    switch (authConfig.type) {
      case 'bearer':
        if (authConfig.token) {
          headers['Authorization'] = `Bearer ${authConfig.token}`;
          logger.debug('Bearer token added to headers');
        } else {
          logger.warn('Bearer token auth type selected but no token provided');
        }
        break;

      case 'api-key':
        if (authConfig.apiKey) {
          const keyHeader = authConfig.apiKeyHeader || 'X-API-Key';
          headers[keyHeader] = authConfig.apiKey;
          logger.debug(`API key added to headers with key: ${keyHeader}`);
        } else {
          logger.warn('API key auth type selected but no key provided');
        }
        break;

      case 'basic':
        if (authConfig.username && authConfig.password) {
          const credentials = Buffer.from(
            `${authConfig.username}:${authConfig.password}`
          ).toString('base64');
          headers['Authorization'] = `Basic ${credentials}`;
          logger.debug('Basic authentication added to headers');
        } else {
          logger.warn('Basic auth type selected but credentials not provided');
        }
        break;

      case 'none':
      default:
        logger.debug('No authentication configured');
        break;
    }

    return headers;
  }

  /**
   * Check if authentication is configured
   */
  static isAuthConfigured(authConfig: AuthConfig): boolean {
    return authConfig.type !== 'none';
  }

  /**
   * Validate authentication configuration
   */
  static validateAuthConfig(authConfig: AuthConfig): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    switch (authConfig.type) {
      case 'bearer':
        if (!authConfig.token) {
          errors.push('Bearer token is required for bearer authentication');
        }
        break;

      case 'api-key':
        if (!authConfig.apiKey) {
          errors.push('API key is required for API key authentication');
        }
        break;

      case 'basic':
        if (!authConfig.username || !authConfig.password) {
          errors.push('Username and password are required for basic authentication');
        }
        break;

      case 'none':
        // No validation needed
        break;

      default:
        errors.push(`Unknown authentication type: ${authConfig.type}`);
        break;
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}
