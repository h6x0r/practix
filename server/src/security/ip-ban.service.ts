import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { CacheService } from '../cache/cache.service';
import { ActivityLoggerService, SecurityEventType } from './activity-logger.service';
import { ThreatLevel } from './code-scanner.service';

/**
 * Reason for IP ban
 */
export enum BanReason {
  BRUTE_FORCE = 'brute_force',
  RATE_LIMIT_ABUSE = 'rate_limit_abuse',
  MALICIOUS_CODE = 'malicious_code',
  SUSPICIOUS_ACTIVITY = 'suspicious_activity',
  MANUAL = 'manual',
}

/**
 * Ban record structure
 */
export interface BanRecord {
  ip: string;
  reason: BanReason;
  bannedAt: number; // timestamp
  expiresAt: number; // timestamp
  strikes: number;
  details?: string;
}

/**
 * IP Ban Service
 * Manages IP bans for abusive behavior
 */
@Injectable()
export class IpBanService {
  private readonly logger = new Logger(IpBanService.name);
  private readonly enabled: boolean;
  private readonly banThreshold: number;
  private readonly banDurationHours: number;

  // Cache key prefixes
  private readonly BAN_KEY_PREFIX = 'ip_ban:';
  private readonly STRIKE_KEY_PREFIX = 'ip_strikes:';

  constructor(
    private configService: ConfigService,
    private cacheService: CacheService,
    private activityLogger: ActivityLoggerService,
  ) {
    this.enabled = this.configService.get<boolean>('ENABLE_IP_BAN', true);
    this.banThreshold = this.configService.get<number>('IP_BAN_THRESHOLD', 5);
    this.banDurationHours = this.configService.get<number>('IP_BAN_DURATION_HOURS', 24);

    this.logger.log(`IP ban service ${this.enabled ? 'enabled' : 'disabled'}`);
    this.logger.log(`Ban threshold: ${this.banThreshold} strikes, duration: ${this.banDurationHours} hours`);
  }

  /**
   * Check if an IP is banned
   */
  async isBanned(ip: string): Promise<boolean> {
    if (!this.enabled) return false;

    const banRecord = await this.getBanRecord(ip);
    if (!banRecord) return false;

    // Check if ban has expired
    if (Date.now() > banRecord.expiresAt) {
      await this.unban(ip, 'expired');
      return false;
    }

    return true;
  }

  /**
   * Get ban record for an IP
   */
  async getBanRecord(ip: string): Promise<BanRecord | null> {
    return this.cacheService.get<BanRecord>(`${this.BAN_KEY_PREFIX}${ip}`);
  }

  /**
   * Add a strike to an IP
   * Returns true if IP was banned as a result
   */
  async addStrike(ip: string, reason: BanReason, details?: string): Promise<boolean> {
    if (!this.enabled) return false;

    const strikeKey = `${this.STRIKE_KEY_PREFIX}${ip}`;

    // Get current strikes
    const currentStrikes = (await this.cacheService.get<number>(strikeKey)) || 0;
    const newStrikes = currentStrikes + 1;

    // Update strikes with 24 hour TTL
    await this.cacheService.set(strikeKey, newStrikes, 24 * 60 * 60);

    this.logger.warn(`Strike added for IP ${ip}: ${newStrikes}/${this.banThreshold} (${reason})`);

    // Check if threshold reached
    if (newStrikes >= this.banThreshold) {
      await this.ban(ip, reason, details);
      return true;
    }

    return false;
  }

  /**
   * Ban an IP address
   */
  async ban(ip: string, reason: BanReason, details?: string): Promise<void> {
    if (!this.enabled) return;

    const banRecord: BanRecord = {
      ip,
      reason,
      bannedAt: Date.now(),
      expiresAt: Date.now() + this.banDurationHours * 60 * 60 * 1000,
      strikes: (await this.cacheService.get<number>(`${this.STRIKE_KEY_PREFIX}${ip}`)) || 0,
      details,
    };

    // Store ban with TTL
    await this.cacheService.set(
      `${this.BAN_KEY_PREFIX}${ip}`,
      banRecord,
      this.banDurationHours * 60 * 60,
    );

    // Log the ban
    await this.activityLogger.logIpBan(ip, reason, this.banDurationHours);

    this.logger.warn(`IP ${ip} banned for ${this.banDurationHours} hours. Reason: ${reason}`);
  }

  /**
   * Unban an IP address
   */
  async unban(ip: string, reason: 'manual' | 'expired'): Promise<void> {
    await this.cacheService.delete(`${this.BAN_KEY_PREFIX}${ip}`);
    await this.cacheService.delete(`${this.STRIKE_KEY_PREFIX}${ip}`);

    await this.activityLogger.logEvent({
      type: SecurityEventType.IP_UNBANNED,
      severity: 'info' as any,
      ip,
      details: { reason },
    });

    this.logger.log(`IP ${ip} unbanned. Reason: ${reason}`);
  }

  /**
   * Get strikes count for an IP
   */
  async getStrikes(ip: string): Promise<number> {
    return (await this.cacheService.get<number>(`${this.STRIKE_KEY_PREFIX}${ip}`)) || 0;
  }

  /**
   * Handle brute force login attempt
   * Called when login brute force is detected
   */
  async handleBruteForce(ip: string): Promise<void> {
    await this.addStrike(ip, BanReason.BRUTE_FORCE, 'Multiple failed login attempts');
  }

  /**
   * Handle rate limit abuse
   * Called when repeated rate limit violations are detected
   */
  async handleRateLimitAbuse(ip: string, endpoint: string): Promise<void> {
    await this.addStrike(ip, BanReason.RATE_LIMIT_ABUSE, `Rate limit abuse on ${endpoint}`);
  }

  /**
   * Handle malicious code detection
   * Called when malicious code is detected
   */
  async handleMaliciousCode(ip: string, threatLevel: ThreatLevel): Promise<void> {
    // For critical threats, ban immediately
    if (threatLevel === ThreatLevel.CRITICAL) {
      await this.ban(ip, BanReason.MALICIOUS_CODE, 'Critical malicious code detected');
    } else if (threatLevel === ThreatLevel.HIGH) {
      // For high threats, add 2 strikes
      await this.addStrike(ip, BanReason.MALICIOUS_CODE, 'High-level malicious code');
      await this.addStrike(ip, BanReason.MALICIOUS_CODE, 'High-level malicious code');
    } else if (threatLevel === ThreatLevel.MEDIUM) {
      // For medium threats, add 1 strike
      await this.addStrike(ip, BanReason.MALICIOUS_CODE, 'Medium-level suspicious code');
    }
  }

  /**
   * Middleware-style check that throws if banned
   */
  async checkAndThrow(ip: string): Promise<void> {
    if (!this.enabled) return;

    if (await this.isBanned(ip)) {
      const banRecord = await this.getBanRecord(ip);
      const remainingTime = banRecord
        ? Math.ceil((banRecord.expiresAt - Date.now()) / (1000 * 60 * 60))
        : this.banDurationHours;

      throw new HttpException(
        {
          statusCode: HttpStatus.FORBIDDEN,
          message: `Your IP address has been temporarily banned due to suspicious activity. Please try again in ${remainingTime} hours.`,
          error: 'IP_BANNED',
          reason: banRecord?.reason,
        },
        HttpStatus.FORBIDDEN,
      );
    }
  }

  /**
   * Get ban info for display (without exposing internal details)
   */
  async getBanInfo(ip: string): Promise<{ isBanned: boolean; remainingHours?: number; reason?: string }> {
    const banRecord = await this.getBanRecord(ip);

    if (!banRecord) {
      return { isBanned: false };
    }

    if (Date.now() > banRecord.expiresAt) {
      await this.unban(ip, 'expired');
      return { isBanned: false };
    }

    return {
      isBanned: true,
      remainingHours: Math.ceil((banRecord.expiresAt - Date.now()) / (1000 * 60 * 60)),
      reason: banRecord.reason,
    };
  }
}
