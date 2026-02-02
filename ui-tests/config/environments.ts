/**
 * Environment Configuration
 * 
 * Manages URLs and configurations for different environments (dev, staging, prod)
 */

export interface EnvironmentConfig {
  baseUrl: string;
  apiUrl: string;
  timeout: number;
  retries: number;
}

export const environments = {
  dev: {
    baseUrl: process.env.DEV_BASE_URL || 'http://localhost:9002',
    apiUrl: process.env.DEV_API_URL || 'http://localhost:3001/api',
    timeout: 30000,
    retries: 1,
  } as EnvironmentConfig,

  staging: {
    baseUrl: process.env.STAGING_BASE_URL || 'https://staging.basket.app',
    apiUrl: process.env.STAGING_API_URL || 'https://staging.basket.app/api',
    timeout: 60000,
    retries: 2,
  } as EnvironmentConfig,

  prod: {
    baseUrl: process.env.PROD_BASE_URL || 'https://basket.app',
    apiUrl: process.env.PROD_API_URL || 'https://basket.app/api',
    timeout: 60000,
    retries: 3,
  } as EnvironmentConfig,
};

/**
 * Get current environment
 */
export function getEnvironment(): EnvironmentConfig {
  const env = process.env.TEST_ENV || 'dev';

  if (env === 'staging') {
    return environments.staging;
  } else if (env === 'prod') {
    return environments.prod;
  }

  return environments.dev;
}

/**
 * Get base URL
 */
export function getBaseUrl(): string {
  return getEnvironment().baseUrl;
}

/**
 * Get API URL
 */
export function getApiUrl(): string {
  return getEnvironment().apiUrl;
}

/**
 * Build full URL
 */
export function buildUrl(path: string): string {
  const baseUrl = getBaseUrl();
  return `${baseUrl}${path}`;
}
