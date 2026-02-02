/**
 * Environment-specific configurations
 * Import the appropriate config based on ENVIRONMENT variable
 */

export const environments = {
  dev: {
    baseURL: 'http://localhost:9002',
    timeout: 30000,
    retries: 2,
    debugMode: false,
  },
  staging: {
    baseURL: 'https://staging.basketapp.com',
    timeout: 45000,
    retries: 3,
    debugMode: false,
  },
  prod: {
    baseURL: 'https://api.basketapp.com',
    timeout: 60000,
    retries: 2,
    debugMode: false,
  },
};

export const getEnvironmentConfig = (env: string = process.env.ENVIRONMENT || 'dev') => {
  return environments[env as keyof typeof environments] || environments.dev;
};
