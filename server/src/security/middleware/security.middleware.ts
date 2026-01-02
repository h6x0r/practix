import { Injectable, NestMiddleware, Logger } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { ActivityLoggerService, SecurityEventType } from '../activity-logger.service';
import { IpBanService } from '../ip-ban.service';

/**
 * Security Middleware
 * Detects and logs suspicious requests (SQL injection, XSS, path traversal)
 */
@Injectable()
export class SecurityMiddleware implements NestMiddleware {
  private readonly logger = new Logger(SecurityMiddleware.name);

  // Patterns for detecting attacks
  private readonly SQL_INJECTION_PATTERNS = [
    /(\%27)|(\')|(\-\-)|(\%23)|(#)/i,
    /((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))/i,
    /\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))/i,
    /union\s+select/i,
    /exec(\s|\+)+(s|x)p\w+/i,
    /insert\s+into/i,
    /drop\s+table/i,
    /update\s+.*\s+set/i,
    /delete\s+from/i,
  ];

  private readonly XSS_PATTERNS = [
    /<script[^>]*>[\s\S]*?<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi,
    /<iframe[^>]*>/gi,
    /<object[^>]*>/gi,
    /<embed[^>]*>/gi,
    /<svg[^>]*onload/gi,
    /expression\s*\(/gi,
  ];

  private readonly PATH_TRAVERSAL_PATTERNS = [
    /\.\.\//g,
    /\.\.%2f/gi,
    /\.\.%5c/gi,
    /%2e%2e%2f/gi,
    /%252e%252e%252f/gi,
    /\.\.\\/g,
  ];

  constructor(
    private activityLogger: ActivityLoggerService,
    private ipBanService: IpBanService,
  ) {}

  async use(req: Request, res: Response, next: NextFunction) {
    const ip = this.getClientIp(req);
    const userId = (req as any).user?.userId;

    // Check if IP is banned
    try {
      await this.ipBanService.checkAndThrow(ip);
    } catch (error) {
      return res.status(403).json({
        statusCode: 403,
        message: 'Access denied',
        error: 'IP_BANNED',
      });
    }

    // Check for suspicious patterns in the request
    const suspiciousResult = await this.checkForSuspiciousPatterns(req, ip, userId);

    if (suspiciousResult.isSuspicious) {
      // Add strike for suspicious activity
      await this.ipBanService.addStrike(
        ip,
        'suspicious_activity' as any,
        suspiciousResult.type,
      );

      // Block SQL injection and path traversal attempts immediately
      // These are always malicious - never allow them through
      if (suspiciousResult.type === 'sql_injection' || suspiciousResult.type === 'path_traversal') {
        return res.status(400).json({
          statusCode: 400,
          message: 'Invalid request',
          error: 'BAD_REQUEST',
        });
      }

      // XSS: Just log and continue (might be false positives in legitimate content)
      // The IP ban service will handle blocking if threshold is reached
    }

    next();
  }

  private async checkForSuspiciousPatterns(
    req: Request,
    ip: string,
    userId?: string,
  ): Promise<{ isSuspicious: boolean; type?: string }> {
    const requestString = this.buildRequestString(req);

    // Check SQL Injection
    for (const pattern of this.SQL_INJECTION_PATTERNS) {
      if (pattern.test(requestString)) {
        await this.activityLogger.logSuspiciousRequest(
          SecurityEventType.SQL_INJECTION_ATTEMPT,
          ip,
          userId,
          {
            path: req.path,
            method: req.method,
            query: JSON.stringify(req.query),
          },
        );
        this.logger.warn(`SQL injection attempt from ${ip}: ${req.path}`);
        return { isSuspicious: true, type: 'sql_injection' };
      }
    }

    // Check XSS
    for (const pattern of this.XSS_PATTERNS) {
      if (pattern.test(requestString)) {
        await this.activityLogger.logSuspiciousRequest(
          SecurityEventType.XSS_ATTEMPT,
          ip,
          userId,
          {
            path: req.path,
            method: req.method,
          },
        );
        this.logger.warn(`XSS attempt from ${ip}: ${req.path}`);
        return { isSuspicious: true, type: 'xss' };
      }
    }

    // Check Path Traversal
    for (const pattern of this.PATH_TRAVERSAL_PATTERNS) {
      if (pattern.test(req.path) || pattern.test(requestString)) {
        await this.activityLogger.logSuspiciousRequest(
          SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
          ip,
          userId,
          {
            path: req.path,
            method: req.method,
          },
        );
        this.logger.warn(`Path traversal attempt from ${ip}: ${req.path}`);
        return { isSuspicious: true, type: 'path_traversal' };
      }
    }

    return { isSuspicious: false };
  }

  private buildRequestString(req: Request): string {
    const parts: string[] = [
      req.path,
      JSON.stringify(req.query || {}),
      JSON.stringify(req.params || {}),
    ];

    // Only include body for non-code-submission endpoints
    // (code submissions intentionally contain code that might trigger patterns)
    if (req.body && !req.path.includes('/submissions/run')) {
      parts.push(JSON.stringify(req.body));
    }

    return parts.join(' ');
  }

  private getClientIp(request: Request): string {
    const forwardedFor = request.headers['x-forwarded-for'];
    if (forwardedFor) {
      const ips = Array.isArray(forwardedFor)
        ? forwardedFor[0]
        : forwardedFor.split(',')[0];
      return ips.trim();
    }

    const realIp = request.headers['x-real-ip'];
    if (realIp) {
      return Array.isArray(realIp) ? realIp[0] : realIp;
    }

    return request.ip || request.socket.remoteAddress || 'unknown';
  }
}
