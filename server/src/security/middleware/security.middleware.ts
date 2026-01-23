import { Injectable, NestMiddleware, Logger } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { ActivityLoggerService, SecurityEventType } from '../activity-logger.service';
import { IpBanService } from '../ip-ban.service';

/**
 * Security Middleware
 * Detects and logs suspicious requests (SQL injection, XSS, path traversal)
 *
 * Note: Prisma ORM already protects against SQL injection via parameterized queries.
 * These patterns serve as defense-in-depth and for detecting/logging attack attempts.
 * Patterns are intentionally less aggressive to reduce false positives.
 */
@Injectable()
export class SecurityMiddleware implements NestMiddleware {
  private readonly logger = new Logger(SecurityMiddleware.name);

  // SQL Injection patterns - only explicit attack signatures
  // Removed: single quotes, #, --, ; (too many false positives in normal text/code)
  private readonly SQL_INJECTION_PATTERNS = [
    /union\s+(all\s+)?select/i,           // UNION SELECT attack
    /exec(\s|\+)+(s|x)p\w+/i,             // SQL Server stored procedure execution
    /drop\s+(table|database|index)/i,     // DROP statements
    /truncate\s+table/i,                  // TRUNCATE TABLE
    /alter\s+table/i,                     // ALTER TABLE
    /create\s+(table|database|index)/i,   // CREATE statements
    /;\s*(drop|delete|truncate|alter)/i,  // Statement chaining with dangerous commands
    /'\s*or\s+'?\d*'?\s*=\s*'?\d*'?/i,    // Classic ' OR '1'='1' pattern
    /'\s*or\s+true/i,                     // ' OR true
    /'\s*and\s+'?\d*'?\s*=\s*'?\d*'?/i,   // ' AND '1'='1' pattern
    /benchmark\s*\(/i,                    // MySQL benchmark attack
    /sleep\s*\(\s*\d+\s*\)/i,             // Time-based injection (SLEEP)
    /waitfor\s+delay/i,                   // SQL Server time-based injection
    /load_file\s*\(/i,                    // MySQL file read
    /into\s+(out|dump)file/i,             // MySQL file write
  ];

  // XSS patterns - focus on actual script execution vectors
  private readonly XSS_PATTERNS = [
    /<script[^>]*>[\s\S]*?<\/script>/gi,  // Script tags
    /javascript\s*:/gi,                    // javascript: protocol
    /on(load|error|click|mouse|focus|blur|change|submit|key)\s*=/gi, // Event handlers (specific ones)
    /<iframe[^>]*src\s*=/gi,               // iframes with src
    /<object[^>]*data\s*=/gi,              // object tags with data
    /<embed[^>]*src\s*=/gi,                // embed tags with src
    /<svg[^>]*on\w+\s*=/gi,                // SVG with event handlers
    /expression\s*\([^)]*\)/gi,            // CSS expression (IE)
    /vbscript\s*:/gi,                      // vbscript: protocol
    /data\s*:\s*text\/html/gi,             // data: URI with HTML
  ];

  // Path traversal patterns - unchanged, these are always malicious
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

    // Include body in security checks
    // New patterns are less aggressive and won't false-positive on normal code
    if (req.body) {
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
