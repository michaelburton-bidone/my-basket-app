/**
 * Base API client for handling HTTP requests
 */

import { APIRequestContext } from '@playwright/test';
import { ApiResponse } from '@types/index';
import {
  logger,
  configManager,
  AuthHandler,
  ErrorHandler,
  ResponseValidator,
} from '@utils/index';

export class BaseAPI {
  protected apiContext: APIRequestContext;
  protected baseURL: string;
  protected headers: Record<string, string>;

  constructor(apiContext: APIRequestContext) {
    this.apiContext = apiContext;
    this.baseURL = configManager.getBaseURL();
    this.headers = this.initializeHeaders();
  }

  /**
   * Initialize headers with authentication
   */
  private initializeHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    const authHeaders = AuthHandler.getAuthHeaders(
      configManager.getAuthConfig()
    );

    return { ...headers, ...authHeaders };
  }

  /**
   * Make GET request
   */
  async get<T = any>(
    endpoint: string,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint, undefined, options);
  }

  /**
   * Make POST request
   */
  async post<T = any>(
    endpoint: string,
    data?: Record<string, any>,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, data, options);
  }

  /**
   * Make PUT request
   */
  async put<T = any>(
    endpoint: string,
    data?: Record<string, any>,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', endpoint, data, options);
  }

  /**
   * Make PATCH request
   */
  async patch<T = any>(
    endpoint: string,
    data?: Record<string, any>,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', endpoint, data, options);
  }

  /**
   * Make DELETE request
   */
  async delete<T = any>(
    endpoint: string,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint, undefined, options);
  }

  /**
   * Core request method
   */
  protected async request<T = any>(
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
    endpoint: string,
    data?: Record<string, any>,
    options?: {
      params?: Record<string, any>;
      headers?: Record<string, string>;
    }
  ): Promise<ApiResponse<T>> {
    const url = this.buildUrl(endpoint, options?.params);
    const headers = { ...this.headers, ...options?.headers };

    logger.debug(`${method} ${url}`);

    try {
      const response = await ErrorHandler.retry(
        async () => {
          return await this.apiContext.fetch(url, {
            method,
            headers,
            data: data ? JSON.stringify(data) : undefined,
            timeout: configManager.getTimeout(),
          });
        },
        3,
        1000
      );

      const responseBody = await response.json().catch(() => ({}));
      const statusCode = response.status();

      logger.debug(`Response Status: ${statusCode}`, responseBody);

      // Validate status is OK
      if (!response.ok()) {
        ErrorHandler.handleApiError(statusCode, responseBody);
      }

      return {
        status: statusCode,
        data: responseBody as T,
        headers: response.headers(),
      };
    } catch (error) {
      if (error instanceof Error) {
        ErrorHandler.handleNetworkError(error);
      }
      throw error;
    }
  }

  /**
   * Build full URL with query parameters
   */
  private buildUrl(endpoint: string, params?: Record<string, any>): string {
    let url = `${this.baseURL}${endpoint}`;

    if (params && Object.keys(params).length > 0) {
      const searchParams = new URLSearchParams();
      for (const [key, value] of Object.entries(params)) {
        if (value !== null && value !== undefined) {
          searchParams.append(key, String(value));
        }
      }
      url += `?${searchParams.toString()}`;
    }

    return url;
  }

  /**
   * Add custom header
   */
  addHeader(key: string, value: string): void {
    this.headers[key] = value;
  }

  /**
   * Remove header
   */
  removeHeader(key: string): void {
    delete this.headers[key];
  }

  /**
   * Get current headers
   */
  getHeaders(): Record<string, string> {
    return { ...this.headers };
  }

  /**
   * Reset headers to default
   */
  resetHeaders(): void {
    this.headers = this.initializeHeaders();
  }
}
