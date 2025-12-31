import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';
import * as crypto from 'crypto';

/**
 * Cache service using Redis for storing execution results
 */
@Injectable()
export class CacheService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(CacheService.name);
  private redis: Redis;
  private isConnected = false;

  // Cache TTL in seconds
  private readonly DEFAULT_TTL = 3600; // 1 hour
  private readonly EXECUTION_CACHE_TTL = 1800; // 30 minutes for code execution

  constructor(private config: ConfigService) {}

  async onModuleInit() {
    try {
      this.redis = new Redis({
        host: this.config.get('REDIS_HOST') || 'redis',
        port: parseInt(this.config.get('REDIS_PORT') || '6379', 10),
        maxRetriesPerRequest: 3,
        retryStrategy: (times) => {
          if (times > 3) {
            this.logger.warn('Redis connection failed, caching disabled');
            return null;
          }
          return Math.min(times * 200, 2000);
        },
      });

      this.redis.on('connect', () => {
        this.isConnected = true;
        this.logger.log('Cache service connected to Redis');
      });

      this.redis.on('error', (err) => {
        this.isConnected = false;
        this.logger.warn(`Redis error: ${err.message}`);
      });

      // Test connection
      await this.redis.ping();
    } catch (error) {
      this.logger.warn('Failed to connect to Redis for caching, continuing without cache');
      this.isConnected = false;
    }
  }

  async onModuleDestroy() {
    if (this.redis) {
      await this.redis.quit();
    }
  }

  /**
   * Generate hash for cache key
   */
  generateHash(data: string): string {
    return crypto.createHash('sha256').update(data).digest('hex').substring(0, 16);
  }

  /**
   * Generate cache key for code execution
   */
  getExecutionCacheKey(code: string, language: string, stdin?: string): string {
    const hash = this.generateHash(`${code}:${language}:${stdin || ''}`);
    return `exec:${language}:${hash}`;
  }

  /**
   * Get cached execution result
   */
  async getExecutionResult<T = Record<string, any>>(code: string, language: string, stdin?: string): Promise<T | null> {
    if (!this.isConnected) return null;

    try {
      const key = this.getExecutionCacheKey(code, language, stdin);
      const cached = await this.redis.get(key);

      if (cached) {
        this.logger.debug(`Cache hit for ${key}`);
        return JSON.parse(cached) as T;
      }

      return null;
    } catch (error) {
      this.logger.warn(`Cache get error: ${error.message}`);
      return null;
    }
  }

  /**
   * Cache execution result
   */
  async setExecutionResult<T = Record<string, any>>(
    code: string,
    language: string,
    stdin: string | undefined,
    result: T & { status?: string },
  ): Promise<void> {
    if (!this.isConnected) return;

    // Only cache successful executions
    if (result.status !== 'passed') return;

    try {
      const key = this.getExecutionCacheKey(code, language, stdin);
      await this.redis.setex(key, this.EXECUTION_CACHE_TTL, JSON.stringify(result));
      this.logger.debug(`Cached result for ${key}`);
    } catch (error) {
      this.logger.warn(`Cache set error: ${error.message}`);
    }
  }

  /**
   * Generic get
   */
  async get<T>(key: string): Promise<T | null> {
    if (!this.isConnected) return null;

    try {
      const value = await this.redis.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      this.logger.warn(`Cache get error: ${error.message}`);
      return null;
    }
  }

  /**
   * Generic set with TTL
   */
  async set<T = any>(key: string, value: T, ttl: number = this.DEFAULT_TTL): Promise<void> {
    if (!this.isConnected) return;

    try {
      await this.redis.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      this.logger.warn(`Cache set error: ${error.message}`);
    }
  }

  /**
   * Delete cache entry
   */
  async delete(key: string): Promise<void> {
    if (!this.isConnected) return;

    try {
      await this.redis.del(key);
    } catch (error) {
      this.logger.warn(`Cache delete error: ${error.message}`);
    }
  }

  /**
   * Delete cache entries by pattern (e.g., "courses:*")
   */
  async deleteByPattern(pattern: string): Promise<number> {
    if (!this.isConnected) return 0;

    try {
      const keys = await this.redis.keys(pattern);
      if (keys.length > 0) {
        await this.redis.del(...keys);
        this.logger.log(`Deleted ${keys.length} cache entries matching: ${pattern}`);
      }
      return keys.length;
    } catch (error) {
      this.logger.warn(`Cache deleteByPattern error: ${error.message}`);
      return 0;
    }
  }

  /**
   * Get cache stats
   */
  async getStats(): Promise<{ connected: boolean; keys?: number }> {
    if (!this.isConnected) {
      return { connected: false };
    }

    try {
      const info = await this.redis.dbsize();
      return { connected: true, keys: info };
    } catch {
      return { connected: false };
    }
  }
}
