/**
 * Configuration loader and management
 */

import dotenv from 'dotenv';
import { TestConfig, AuthConfig, AuthType } from '@types/index';

dotenv.config();

export class ConfigManager {
  private static instance: ConfigManager;
  private config: TestConfig;

  private constructor() {
    this.config = this.loadConfig();
  }

  static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  private loadConfig(): TestConfig {
    const authType = (process.env.AUTH_TYPE || 'none') as AuthType;

    const authConfig: AuthConfig = {
      type: authType,
    };

    if (authType === 'bearer') {
      authConfig.token = process.env.BEARER_TOKEN;
    } else if (authType === 'api-key') {
      authConfig.apiKey = process.env.API_KEY;
      authConfig.apiKeyHeader = process.env.API_KEY_HEADER || 'X-API-Key';
    } else if (authType === 'basic') {
      authConfig.username = process.env.BASIC_AUTH_USERNAME;
      authConfig.password = process.env.BASIC_AUTH_PASSWORD;
    }

    return {
      baseURL: process.env.BASE_URL || 'http://localhost:3002',
      environment: (process.env.ENVIRONMENT as any) || 'dev',
      auth: authConfig,
      requestTimeout: parseInt(process.env.REQUEST_TIMEOUT || '30000', 10),
      apiTimeout: parseInt(process.env.API_TIMEOUT || '10000', 10),
      debug: process.env.DEBUG === 'true',
    };
  }

  getConfig(): TestConfig {
    return this.config;
  }

  getBaseURL(): string {
    return this.config.baseURL;
  }

  getAuthConfig(): AuthConfig {
    return this.config.auth;
  }

  getTimeout(): number {
    return this.config.requestTimeout;
  }

  getEnvironment(): string {
    return this.config.environment;
  }

  isDebugMode(): boolean {
    return this.config.debug;
  }
}

export const configManager = ConfigManager.getInstance();
