/**
 * Authentication and configuration types
 */

export type AuthType = 'none' | 'bearer' | 'api-key' | 'basic';

export interface AuthConfig {
  type: AuthType;
  token?: string;
  apiKey?: string;
  apiKeyHeader?: string;
  username?: string;
  password?: string;
}

export interface RequestConfig {
  baseURL?: string;
  timeout?: number;
  headers?: Record<string, string>;
  auth?: AuthConfig;
}

export interface TestConfig {
  baseURL: string;
  environment: 'dev' | 'staging' | 'prod';
  auth: AuthConfig;
  requestTimeout: number;
  apiTimeout: number;
  debug: boolean;
}

/**
 * Test execution types
 */

export interface TestResult {
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
}

export interface TestSuite {
  name: string;
  tests: TestResult[];
  totalDuration: number;
}
