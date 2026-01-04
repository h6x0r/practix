import { Injectable } from '@nestjs/common';
import { HealthIndicator, HealthIndicatorResult, HealthCheckError } from '@nestjs/terminus';
import { CacheService } from '../cache/cache.service';

@Injectable()
export class RedisHealthIndicator extends HealthIndicator {
  constructor(private cache: CacheService) {
    super();
  }

  async isHealthy(key: string): Promise<HealthIndicatorResult> {
    try {
      // Perform a simple set/get to verify Redis connectivity
      const testKey = '__health_check__';
      const testValue = Date.now().toString();

      await this.cache.set(testKey, testValue, 10); // 10 second TTL
      const retrieved = await this.cache.get<string>(testKey);

      if (retrieved === testValue) {
        await this.cache.delete(testKey);
        return this.getStatus(key, true, { message: 'Redis is reachable' });
      }

      const result = this.getStatus(key, false, { message: 'Redis read/write mismatch' });
      throw new HealthCheckError('Redis check failed', result);
    } catch (error) {
      if (error instanceof HealthCheckError) throw error;

      const result = this.getStatus(key, false, {
        message: error instanceof Error ? error.message : 'Redis check failed',
      });
      throw new HealthCheckError('Redis check failed', result);
    }
  }
}
