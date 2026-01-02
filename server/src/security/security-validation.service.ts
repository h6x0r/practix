import { Injectable, ForbiddenException, Logger } from '@nestjs/common';
import { CodeScannerService, CodeScanResult } from './code-scanner.service';
import { ActivityLoggerService } from './activity-logger.service';
import { IpBanService } from './ip-ban.service';

/**
 * SecurityValidationService
 *
 * Centralized service for validating code submissions against security threats.
 * Extracted from SubmissionsService to follow Single Responsibility Principle.
 *
 * Responsibilities:
 * - Scan code for malicious patterns
 * - Log security events
 * - Handle IP banning for repeat offenders
 */
@Injectable()
export class SecurityValidationService {
  private readonly logger = new Logger(SecurityValidationService.name);

  constructor(
    private readonly codeScannerService: CodeScannerService,
    private readonly activityLogger: ActivityLoggerService,
    private readonly ipBanService: IpBanService,
  ) {}

  /**
   * Validate code for security threats
   *
   * @param code - The code to validate
   * @param language - Programming language
   * @param context - Optional context for logging (ip, userId)
   * @throws ForbiddenException if code contains malicious patterns
   */
  async validateCode(
    code: string,
    language: string,
    context?: { ip?: string; userId?: string },
  ): Promise<void> {
    const scanResult = this.codeScannerService.scan(code, language);

    if (!scanResult.isSafe) {
      await this.handleMaliciousCode(scanResult, code, language, context);
    }
  }

  /**
   * Handle malicious code detection
   * - Logs the security event
   * - Adds strikes to IP
   * - Throws ForbiddenException
   */
  private async handleMaliciousCode(
    scanResult: CodeScanResult,
    code: string,
    language: string,
    context?: { ip?: string; userId?: string },
  ): Promise<never> {
    const { ip, userId } = context || {};

    this.logger.warn(
      `Malicious code detected: ip=${ip}, user=${userId}, lang=${language}, threat=${scanResult.threatLevel}`,
    );

    if (ip) {
      // Log the security event
      await this.activityLogger.logMaliciousCode(
        ip,
        userId,
        code,
        language,
        scanResult.threatLevel,
        scanResult.threats,
      );

      // Add strikes and potentially ban the IP
      await this.ipBanService.handleMaliciousCode(ip, scanResult.threatLevel);
    }

    throw new ForbiddenException({
      message: scanResult.message,
      threatLevel: scanResult.threatLevel,
      threats: scanResult.threats.map(t => t.pattern),
    });
  }

  /**
   * Quick check if code is safe (without throwing)
   * Useful for pre-validation or conditional logic
   */
  isCodeSafe(code: string, language: string): boolean {
    const scanResult = this.codeScannerService.scan(code, language);
    return scanResult.isSafe;
  }

  /**
   * Get scan result without throwing
   * Useful when you need detailed threat information
   */
  scanCode(code: string, language: string): CodeScanResult {
    return this.codeScannerService.scan(code, language);
  }
}
