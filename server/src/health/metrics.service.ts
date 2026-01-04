import { Injectable, OnModuleInit } from '@nestjs/common';
import * as client from 'prom-client';

@Injectable()
export class MetricsService implements OnModuleInit {
  private readonly registry: client.Registry;

  // Custom metrics
  private readonly httpRequestsTotal: client.Counter<string>;
  private readonly httpRequestDuration: client.Histogram<string>;
  private readonly submissionsTotal: client.Counter<string>;
  private readonly submissionDuration: client.Histogram<string>;
  private readonly aiRequestsTotal: client.Counter<string>;
  private readonly activeUsers: client.Gauge<string>;

  constructor() {
    this.registry = new client.Registry();

    // Add default metrics (CPU, memory, event loop, etc.)
    client.collectDefaultMetrics({ register: this.registry });

    // HTTP requests counter
    this.httpRequestsTotal = new client.Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'path', 'status'],
      registers: [this.registry],
    });

    // HTTP request duration histogram
    this.httpRequestDuration = new client.Histogram({
      name: 'http_request_duration_seconds',
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'path', 'status'],
      buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
      registers: [this.registry],
    });

    // Code submissions counter
    this.submissionsTotal = new client.Counter({
      name: 'code_submissions_total',
      help: 'Total number of code submissions',
      labelNames: ['language', 'status'],
      registers: [this.registry],
    });

    // Submission execution duration
    this.submissionDuration = new client.Histogram({
      name: 'code_submission_duration_seconds',
      help: 'Duration of code submission execution in seconds',
      labelNames: ['language'],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60],
      registers: [this.registry],
    });

    // AI requests counter
    this.aiRequestsTotal = new client.Counter({
      name: 'ai_requests_total',
      help: 'Total number of AI API requests',
      labelNames: ['type', 'status'],
      registers: [this.registry],
    });

    // Active users gauge
    this.activeUsers = new client.Gauge({
      name: 'active_users',
      help: 'Number of currently active users',
      registers: [this.registry],
    });
  }

  onModuleInit() {
    // Initialize gauge to 0
    this.activeUsers.set(0);
  }

  /**
   * Get all metrics in Prometheus format
   */
  async getMetrics(): Promise<string> {
    return this.registry.metrics();
  }

  /**
   * Record an HTTP request
   */
  recordHttpRequest(method: string, path: string, status: number, durationMs: number) {
    const normalizedPath = this.normalizePath(path);
    this.httpRequestsTotal.inc({ method, path: normalizedPath, status: String(status) });
    this.httpRequestDuration.observe(
      { method, path: normalizedPath, status: String(status) },
      durationMs / 1000,
    );
  }

  /**
   * Record a code submission
   */
  recordSubmission(language: string, status: 'passed' | 'failed' | 'error', durationMs: number) {
    this.submissionsTotal.inc({ language, status });
    this.submissionDuration.observe({ language }, durationMs / 1000);
  }

  /**
   * Record an AI request
   */
  recordAiRequest(type: 'hint' | 'explain' | 'roadmap', status: 'success' | 'error') {
    this.aiRequestsTotal.inc({ type, status });
  }

  /**
   * Update active users count
   */
  setActiveUsers(count: number) {
    this.activeUsers.set(count);
  }

  /**
   * Normalize path to avoid high cardinality
   * Replaces dynamic segments like IDs and slugs with placeholders
   */
  private normalizePath(path: string): string {
    return path
      .replace(/\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, '/:id')
      .replace(/\/\d+/g, '/:id')
      .replace(/\/[a-z0-9-]+(?=\/|$)/gi, (match, offset, string) => {
        // Keep known route segments, replace others
        const knownSegments = [
          'auth', 'courses', 'tasks', 'submissions', 'roadmaps', 'gamification',
          'subscriptions', 'admin', 'ai', 'health', 'metrics', 'api', 'docs',
        ];
        const segment = match.slice(1);
        if (knownSegments.includes(segment.toLowerCase())) {
          return match;
        }
        return '/:slug';
      });
  }
}
