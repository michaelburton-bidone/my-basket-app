/**
 * Page-related types and interfaces
 */

export interface PageLocators {
  [key: string]: string;
}

export interface NavigationOptions {
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle';
  referer?: string;
  timeout?: number;
}

export interface ElementState {
  visible: boolean;
  enabled: boolean;
  checked?: boolean;
}

export interface ClickOptions {
  button?: 'left' | 'right' | 'middle';
  clickCount?: number;
  delay?: number;
  force?: boolean;
  noWaitAfter?: boolean;
  timeout?: number;
}

export interface TextInputOptions {
  delay?: number;
  noWaitAfter?: boolean;
  timeout?: number;
}

export interface WaitOptions {
  state?: 'attached' | 'visible' | 'hidden' | 'stable';
  timeout?: number;
}
