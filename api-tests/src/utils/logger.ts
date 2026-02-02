/**
 * Logger utility for consistent logging across tests
 */

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR',
}

class Logger {
  private logLevel: LogLevel;

  constructor(level: LogLevel = LogLevel.INFO) {
    this.logLevel = level;
  }

  private getTimestamp(): string {
    return new Date().toISOString();
  }

  private shouldLog(level: LogLevel): boolean {
    const levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR];
    return levels.indexOf(level) >= levels.indexOf(this.logLevel);
  }

  debug(message: string, data?: any): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      console.log(
        `[${this.getTimestamp()}] [${LogLevel.DEBUG}] ${message}`,
        data || ''
      );
    }
  }

  info(message: string, data?: any): void {
    if (this.shouldLog(LogLevel.INFO)) {
      console.log(
        `[${this.getTimestamp()}] [${LogLevel.INFO}] ${message}`,
        data || ''
      );
    }
  }

  warn(message: string, data?: any): void {
    if (this.shouldLog(LogLevel.WARN)) {
      console.warn(
        `[${this.getTimestamp()}] [${LogLevel.WARN}] ${message}`,
        data || ''
      );
    }
  }

  error(message: string, error?: any): void {
    if (this.shouldLog(LogLevel.ERROR)) {
      console.error(
        `[${this.getTimestamp()}] [${LogLevel.ERROR}] ${message}`,
        error || ''
      );
    }
  }

  setLevel(level: LogLevel): void {
    this.logLevel = level;
  }
}

export const logger = new Logger(LogLevel.INFO);
