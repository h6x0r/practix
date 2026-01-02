import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { CacheService } from '../cache/cache.service';
import { ThreatLevel } from './code-scanner.service';

/**
 * Types of security events
 */
export enum SecurityEventType {
  // Authentication events
  LOGIN_SUCCESS = 'login_success',
  LOGIN_FAILED = 'login_failed',
  LOGIN_BRUTE_FORCE = 'login_brute_force',
  LOGOUT = 'logout',
  TOKEN_EXPIRED = 'token_expired',
  TOKEN_INVALID = 'token_invalid',

  // Rate limiting events
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  RATE_LIMIT_ABUSE = 'rate_limit_abuse', // Repeated rate limit violations

  // Code execution events
  MALICIOUS_CODE_DETECTED = 'malicious_code_detected',
  CODE_EXECUTION_ERROR = 'code_execution_error',
  TIMEOUT_EXECUTION = 'timeout_execution',

  // Access control events
  UNAUTHORIZED_ACCESS = 'unauthorized_access',
  FORBIDDEN_RESOURCE = 'forbidden_resource',
  SUBSCRIPTION_BYPASS_ATTEMPT = 'subscription_bypass_attempt',

  // Suspicious behavior
  SUSPICIOUS_REQUEST = 'suspicious_request',
  SQL_INJECTION_ATTEMPT = 'sql_injection_attempt',
  XSS_ATTEMPT = 'xss_attempt',
  PATH_TRAVERSAL_ATTEMPT = 'path_traversal_attempt',

  // IP-related events
  IP_BANNED = 'ip_banned',
  IP_UNBANNED = 'ip_unbanned',
  NEW_IP_FOR_USER = 'new_ip_for_user',
}

/**
 * Severity levels for security events
 */
export enum EventSeverity {
  INFO = 'info',
  WARNING = 'warning',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * Security event data structure
 */
export interface SecurityEvent {
  type: SecurityEventType;
  severity: EventSeverity;
  ip: string;
  userId?: string;
  userEmail?: string;
  details: Record<string, any>;
  timestamp: Date;
}

/**
 * Activity Logger Service
 * Logs security events and suspicious activities for analysis and alerting
 */
@Injectable()
export class ActivityLoggerService {
  private readonly logger = new Logger(ActivityLoggerService.name);
  private readonly enabled: boolean;
  private readonly storeInDb: boolean;

  // Track failed login attempts for brute force detection
  private readonly BRUTE_FORCE_THRESHOLD = 5;
  private readonly BRUTE_FORCE_WINDOW = 15 * 60; // 15 minutes in seconds

  constructor(
    private configService: ConfigService,
    private prisma: PrismaService,
    private cacheService: CacheService,
  ) {
    this.enabled = this.configService.get<boolean>('SUSPICIOUS_ACTIVITY_LOG', true);
    this.storeInDb = this.configService.get<string>('NODE_ENV') === 'production';
    this.logger.log(`Activity logger ${this.enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Log a security event
   */
  async logEvent(event: Omit<SecurityEvent, 'timestamp'>): Promise<void> {
    if (!this.enabled) return;

    const fullEvent: SecurityEvent = {
      ...event,
      timestamp: new Date(),
    };

    // Always log to console for immediate visibility
    this.logToConsole(fullEvent);

    // Store in database for production environments
    if (this.storeInDb) {
      await this.storeEvent(fullEvent);
    }

    // Track patterns for alerting
    await this.trackPatterns(fullEvent);
  }

  /**
   * Log failed login attempt and check for brute force
   */
  async logFailedLogin(ip: string, email: string): Promise<{ isBruteForce: boolean }> {
    const cacheKey = `login_failed:${ip}:${email}`;

    // Get current count
    const currentCount = (await this.cacheService.get<number>(cacheKey)) || 0;
    const newCount = currentCount + 1;

    // Update count with TTL
    await this.cacheService.set(cacheKey, newCount, this.BRUTE_FORCE_WINDOW);

    const isBruteForce = newCount >= this.BRUTE_FORCE_THRESHOLD;

    await this.logEvent({
      type: isBruteForce ? SecurityEventType.LOGIN_BRUTE_FORCE : SecurityEventType.LOGIN_FAILED,
      severity: isBruteForce ? EventSeverity.CRITICAL : EventSeverity.WARNING,
      ip,
      userEmail: email,
      details: {
        attemptCount: newCount,
        threshold: this.BRUTE_FORCE_THRESHOLD,
      },
    });

    return { isBruteForce };
  }

  /**
   * Log successful login
   */
  async logSuccessfulLogin(ip: string, userId: string, email: string): Promise<void> {
    // Clear failed login count on success
    const cacheKey = `login_failed:${ip}:${email}`;
    await this.cacheService.delete(cacheKey);

    // Check if this is a new IP for this user
    const userIpsKey = `user_ips:${userId}`;
    const knownIps = (await this.cacheService.get<string[]>(userIpsKey)) || [];

    const isNewIp = !knownIps.includes(ip);
    if (isNewIp) {
      knownIps.push(ip);
      // Keep last 10 IPs, TTL 30 days
      const recentIps = knownIps.slice(-10);
      await this.cacheService.set(userIpsKey, recentIps, 30 * 24 * 60 * 60);

      // Log new IP event
      await this.logEvent({
        type: SecurityEventType.NEW_IP_FOR_USER,
        severity: EventSeverity.INFO,
        ip,
        userId,
        userEmail: email,
        details: {
          knownIpCount: recentIps.length,
        },
      });
    }

    await this.logEvent({
      type: SecurityEventType.LOGIN_SUCCESS,
      severity: EventSeverity.INFO,
      ip,
      userId,
      userEmail: email,
      details: { isNewIp },
    });
  }

  /**
   * Log malicious code detection
   * NOTE: We intentionally do NOT log the code itself to avoid:
   * - Storing potentially sensitive user data
   * - Creating a repository of attack patterns
   * - PII exposure in obfuscated code
   */
  async logMaliciousCode(
    ip: string,
    userId: string | undefined,
    code: string,
    language: string,
    threatLevel: ThreatLevel,
    threats: Array<{ description: string; pattern: string }>,
  ): Promise<void> {
    await this.logEvent({
      type: SecurityEventType.MALICIOUS_CODE_DETECTED,
      severity: this.threatLevelToSeverity(threatLevel),
      ip,
      userId,
      details: {
        language,
        threatLevel,
        threatDescriptions: threats.map(t => t.description),
        // Do NOT log code content - only metadata
        codeLength: code.length,
        codeHash: this.hashForReference(code), // Hash for correlation only
      },
    });
  }

  /**
   * Create a short hash for log correlation (not for security)
   */
  private hashForReference(data: string): string {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(data).digest('hex').substring(0, 16);
  }

  /**
   * Log rate limit violation
   */
  async logRateLimitExceeded(
    ip: string,
    userId: string | undefined,
    endpoint: string,
    retryAfter: number,
  ): Promise<void> {
    // Track repeated violations
    const violationKey = `rate_violations:${ip}`;
    const violations = (await this.cacheService.get<number>(violationKey)) || 0;
    const newViolations = violations + 1;

    await this.cacheService.set(violationKey, newViolations, 60 * 60); // 1 hour window

    const isAbuse = newViolations >= 10; // 10 violations in an hour = abuse

    await this.logEvent({
      type: isAbuse ? SecurityEventType.RATE_LIMIT_ABUSE : SecurityEventType.RATE_LIMIT_EXCEEDED,
      severity: isAbuse ? EventSeverity.HIGH : EventSeverity.WARNING,
      ip,
      userId,
      details: {
        endpoint,
        retryAfter,
        violationCount: newViolations,
      },
    });
  }

  /**
   * Log unauthorized access attempt
   */
  async logUnauthorizedAccess(
    ip: string,
    endpoint: string,
    method: string,
    headers: Record<string, string>,
  ): Promise<void> {
    await this.logEvent({
      type: SecurityEventType.UNAUTHORIZED_ACCESS,
      severity: EventSeverity.WARNING,
      ip,
      details: {
        endpoint,
        method,
        userAgent: headers['user-agent'],
        origin: headers['origin'],
      },
    });
  }

  /**
   * Log suspicious request (SQL injection, XSS, path traversal)
   */
  async logSuspiciousRequest(
    type: SecurityEventType.SQL_INJECTION_ATTEMPT | SecurityEventType.XSS_ATTEMPT | SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
    ip: string,
    userId: string | undefined,
    requestDetails: { path: string; method: string; body?: string; query?: string },
  ): Promise<void> {
    await this.logEvent({
      type,
      severity: EventSeverity.HIGH,
      ip,
      userId,
      details: requestDetails,
    });
  }

  /**
   * Log IP ban event
   */
  async logIpBan(ip: string, reason: string, duration: number): Promise<void> {
    await this.logEvent({
      type: SecurityEventType.IP_BANNED,
      severity: EventSeverity.HIGH,
      ip,
      details: {
        reason,
        durationHours: duration,
        expiresAt: new Date(Date.now() + duration * 60 * 60 * 1000).toISOString(),
      },
    });
  }

  /**
   * Get recent security events (for admin dashboard)
   */
  async getRecentEvents(limit = 100, severity?: EventSeverity): Promise<any[]> {
    if (!this.storeInDb) {
      return [];
    }

    const where: any = {};
    if (severity) {
      where.severity = severity;
    }

    return this.prisma.securityEvent.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
  }

  /**
   * Get event count by type (for metrics)
   */
  async getEventCounts(since: Date): Promise<Record<SecurityEventType, number>> {
    if (!this.storeInDb) {
      return {} as Record<SecurityEventType, number>;
    }

    const counts = await this.prisma.securityEvent.groupBy({
      by: ['type'],
      where: {
        createdAt: { gte: since },
      },
      _count: true,
    });

    return counts.reduce((acc, { type, _count }) => {
      acc[type as SecurityEventType] = _count;
      return acc;
    }, {} as Record<SecurityEventType, number>);
  }

  // ================================
  // Private methods
  // ================================

  private logToConsole(event: SecurityEvent): void {
    const logMessage = `[SECURITY] ${event.type} | IP: ${event.ip} | User: ${event.userId || 'anonymous'}`;

    switch (event.severity) {
      case EventSeverity.CRITICAL:
        this.logger.error(logMessage, event.details);
        break;
      case EventSeverity.HIGH:
        this.logger.warn(logMessage, event.details);
        break;
      case EventSeverity.WARNING:
        this.logger.warn(logMessage, event.details);
        break;
      default:
        this.logger.log(logMessage);
    }
  }

  private async storeEvent(event: SecurityEvent): Promise<void> {
    try {
      await this.prisma.securityEvent.create({
        data: {
          type: event.type,
          severity: event.severity,
          ip: event.ip,
          userId: event.userId,
          userEmail: event.userEmail,
          details: event.details,
          createdAt: event.timestamp,
        },
      });
    } catch (error) {
      this.logger.error('Failed to store security event', error);
    }
  }

  private async trackPatterns(event: SecurityEvent): Promise<void> {
    // Track critical events for potential alerting
    if (event.severity === EventSeverity.CRITICAL) {
      const criticalKey = `critical_events:${event.ip}`;
      const count = (await this.cacheService.get<number>(criticalKey)) || 0;
      await this.cacheService.set(criticalKey, count + 1, 60 * 60); // 1 hour

      // Could trigger alerts here (email, Slack, etc.)
      if (count + 1 >= 3) {
        this.logger.error(`ALERT: Multiple critical events from IP ${event.ip}`);
      }
    }
  }

  private threatLevelToSeverity(level: ThreatLevel): EventSeverity {
    switch (level) {
      case ThreatLevel.CRITICAL:
        return EventSeverity.CRITICAL;
      case ThreatLevel.HIGH:
        return EventSeverity.HIGH;
      case ThreatLevel.MEDIUM:
        return EventSeverity.WARNING;
      default:
        return EventSeverity.INFO;
    }
  }
}
