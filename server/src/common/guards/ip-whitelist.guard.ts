import {
  Injectable,
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Logger,
  SetMetadata,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { Request } from 'express';

/**
 * Metadata key for specifying which IP whitelist to use
 */
export const IP_WHITELIST_KEY = 'ip_whitelist_provider';

/**
 * Decorator to specify which provider's IP whitelist to use
 * @param provider - 'payme' or 'click'
 */
export const IpWhitelist = (provider: 'payme' | 'click') =>
  SetMetadata(IP_WHITELIST_KEY, provider);

/**
 * Known IP addresses/ranges for payment providers
 * These are documented in provider APIs and can be overridden via env vars
 *
 * Payme IPs: https://developer.payme.uz/uz/protocols/merchant-api
 * Click IPs: https://docs.click.uz/click-api-request
 */
const DEFAULT_PAYME_IPS = [
  // Payme production servers
  '185.8.212.0/24',
  '195.158.31.0/24',
  // Development/test mode - allow localhost
];

const DEFAULT_CLICK_IPS = [
  // Click production servers
  '185.8.212.0/24',
  '195.158.28.0/24',
  '195.158.29.0/24',
  // Development/test mode - allow localhost
];

/**
 * IP Whitelist Guard for Payment Webhooks
 *
 * Validates that incoming requests come from known payment provider IPs.
 * This adds an extra layer of security beyond Basic Auth (Payme) and
 * signature verification (Click).
 *
 * Usage:
 * @UseGuards(IpWhitelistGuard)
 * @IpWhitelist('payme')
 * handlePaymeWebhook() { ... }
 */
@Injectable()
export class IpWhitelistGuard implements CanActivate {
  private readonly logger = new Logger(IpWhitelistGuard.name);
  private readonly paymeIps: string[];
  private readonly clickIps: string[];
  private readonly isEnabled: boolean;
  private readonly isDevelopment: boolean;

  constructor(
    private reflector: Reflector,
    private configService: ConfigService,
  ) {
    this.isDevelopment =
      this.configService.get<string>('NODE_ENV') !== 'production';

    // Load IP whitelist settings from config
    this.isEnabled = this.configService.get<boolean>(
      'WEBHOOK_IP_WHITELIST_ENABLED',
      true,
    );

    // Load provider IPs from env or use defaults
    const paymeIpsEnv = this.configService.get<string>('PAYME_ALLOWED_IPS', '');
    const clickIpsEnv = this.configService.get<string>('CLICK_ALLOWED_IPS', '');

    this.paymeIps = paymeIpsEnv
      ? paymeIpsEnv.split(',').map((ip) => ip.trim())
      : DEFAULT_PAYME_IPS;

    this.clickIps = clickIpsEnv
      ? clickIpsEnv.split(',').map((ip) => ip.trim())
      : DEFAULT_CLICK_IPS;

    this.logger.log(
      `IP Whitelist Guard initialized. Enabled: ${this.isEnabled}, Development: ${this.isDevelopment}`,
    );
  }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    // If disabled, allow all requests
    if (!this.isEnabled) {
      return true;
    }

    const request = context.switchToHttp().getRequest<Request>();
    const clientIp = this.getClientIp(request);
    const provider = this.reflector.get<string>(
      IP_WHITELIST_KEY,
      context.getHandler(),
    );

    // If no provider specified, skip IP check
    if (!provider) {
      return true;
    }

    // In development, allow localhost and private IPs
    if (this.isDevelopment && this.isLocalOrPrivateIp(clientIp)) {
      this.logger.debug(
        `Development mode: allowing local IP ${clientIp} for ${provider}`,
      );
      return true;
    }

    // Get the appropriate whitelist
    const allowedIps = provider === 'payme' ? this.paymeIps : this.clickIps;

    // Check if client IP is in the whitelist
    const isAllowed = this.isIpAllowed(clientIp, allowedIps);

    if (!isAllowed) {
      this.logger.warn(
        `Blocked ${provider} webhook from unauthorized IP: ${clientIp}`,
      );
      throw new ForbiddenException(
        `IP address ${clientIp} is not authorized for ${provider} webhooks`,
      );
    }

    this.logger.debug(`Allowed ${provider} webhook from IP: ${clientIp}`);
    return true;
  }

  /**
   * Extract client IP from request, handling proxies
   */
  private getClientIp(request: Request): string {
    // Check x-forwarded-for header (set by reverse proxies)
    const forwardedFor = request.headers['x-forwarded-for'];
    if (forwardedFor) {
      const ips = Array.isArray(forwardedFor)
        ? forwardedFor[0]
        : forwardedFor.split(',')[0];
      return ips.trim();
    }

    // Check x-real-ip header (Nginx)
    const realIp = request.headers['x-real-ip'];
    if (realIp) {
      return Array.isArray(realIp) ? realIp[0] : realIp;
    }

    // Fall back to socket address
    return request.ip || request.socket.remoteAddress || 'unknown';
  }

  /**
   * Check if IP is localhost or private network
   */
  private isLocalOrPrivateIp(ip: string): boolean {
    // Normalize IPv6 localhost
    if (ip === '::1' || ip === '::ffff:127.0.0.1') {
      return true;
    }

    // Check IPv4 patterns
    if (
      ip.startsWith('127.') ||
      ip.startsWith('10.') ||
      ip.startsWith('192.168.') ||
      ip.startsWith('172.16.') ||
      ip.startsWith('172.17.') ||
      ip.startsWith('172.18.') ||
      ip.startsWith('172.19.') ||
      ip.startsWith('172.2') ||
      ip.startsWith('172.30.') ||
      ip.startsWith('172.31.') ||
      ip === 'localhost'
    ) {
      return true;
    }

    return false;
  }

  /**
   * Check if IP is in the allowed list (supports CIDR notation)
   */
  private isIpAllowed(clientIp: string, allowedIps: string[]): boolean {
    for (const allowedIp of allowedIps) {
      if (allowedIp.includes('/')) {
        // CIDR notation
        if (this.isIpInCidr(clientIp, allowedIp)) {
          return true;
        }
      } else {
        // Exact match
        if (clientIp === allowedIp) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Check if IP is within CIDR range
   * @param ip - IP address to check
   * @param cidr - CIDR notation (e.g., '192.168.1.0/24')
   */
  private isIpInCidr(ip: string, cidr: string): boolean {
    try {
      const [range, bits] = cidr.split('/');
      const mask = parseInt(bits, 10);

      const ipNum = this.ipToNumber(ip);
      const rangeNum = this.ipToNumber(range);

      if (ipNum === null || rangeNum === null) {
        return false;
      }

      const maskBits = ~(2 ** (32 - mask) - 1);
      return (ipNum & maskBits) === (rangeNum & maskBits);
    } catch {
      return false;
    }
  }

  /**
   * Convert IP address to number for CIDR comparison
   */
  private ipToNumber(ip: string): number | null {
    // Handle IPv4-mapped IPv6 addresses
    if (ip.startsWith('::ffff:')) {
      ip = ip.substring(7);
    }

    const parts = ip.split('.');
    if (parts.length !== 4) {
      return null;
    }

    let num = 0;
    for (const part of parts) {
      const octet = parseInt(part, 10);
      if (isNaN(octet) || octet < 0 || octet > 255) {
        return null;
      }
      num = (num << 8) + octet;
    }
    return num >>> 0; // Convert to unsigned
  }
}
