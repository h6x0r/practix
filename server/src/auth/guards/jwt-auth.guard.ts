import { Injectable, ExecutionContext, UnauthorizedException, Logger, Inject, Optional } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { CacheService } from '../../cache/cache.service';
import { Request } from 'express';

// Rate limiting constants
const AUTH_FAIL_LIMIT = 10; // Max failed auth attempts per window
const AUTH_FAIL_WINDOW = 300; // 5 minute window in seconds
const AUTH_BLOCK_DURATION = 900; // Block for 15 minutes after limit exceeded

@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {
  private readonly logger = new Logger(JwtAuthGuard.name);

  constructor(
    @Optional() @Inject(CacheService) private readonly cacheService?: CacheService,
  ) {
    super();
  }

  /**
   * Get client IP from request
   */
  private getClientIp(request: Request): string {
    // Support for reverse proxies (X-Forwarded-For header)
    const forwarded = request.headers['x-forwarded-for'];
    if (forwarded) {
      const ips = Array.isArray(forwarded) ? forwarded[0] : forwarded.split(',')[0];
      return ips.trim();
    }
    return request.ip || request.socket?.remoteAddress || 'unknown';
  }

  /**
   * Check if IP is rate limited
   */
  private async isRateLimited(ip: string): Promise<boolean> {
    if (!this.cacheService) return false;

    const blockKey = `auth:blocked:${ip}`;
    const blocked = await this.cacheService.get<boolean>(blockKey);
    return !!blocked;
  }

  /**
   * Track failed auth attempt and potentially block IP
   */
  private async trackFailedAttempt(ip: string): Promise<void> {
    if (!this.cacheService) return;

    const countKey = `auth:fail:${ip}`;
    const currentCount = (await this.cacheService.get<number>(countKey)) || 0;
    const newCount = currentCount + 1;

    // Update count with window TTL
    await this.cacheService.set(countKey, newCount, AUTH_FAIL_WINDOW);

    if (newCount >= AUTH_FAIL_LIMIT) {
      // Block the IP
      const blockKey = `auth:blocked:${ip}`;
      await this.cacheService.set(blockKey, true, AUTH_BLOCK_DURATION);
      this.logger.warn(`IP ${ip} blocked for excessive auth failures (${newCount} attempts)`);
    }
  }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    const ip = this.getClientIp(request);

    // Check if IP is blocked
    if (await this.isRateLimited(ip)) {
      this.logger.warn(`Blocked auth attempt from rate-limited IP: ${ip}`);
      throw new UnauthorizedException('Too many authentication failures. Please try again later.');
    }

    try {
      // Call parent canActivate (validates JWT)
      const result = await super.canActivate(context);
      return result as boolean;
    } catch (error) {
      // Track failed attempt
      await this.trackFailedAttempt(ip);
      throw error;
    }
  }
}