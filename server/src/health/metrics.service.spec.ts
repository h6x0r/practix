import { Test, TestingModule } from '@nestjs/testing';
import { MetricsService } from './metrics.service';

describe('MetricsService', () => {
  let service: MetricsService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [MetricsService],
    }).compile();

    service = module.get<MetricsService>(MetricsService);
    // Trigger onModuleInit
    service.onModuleInit();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // getMetrics() - Prometheus format output
  // ============================================
  describe('getMetrics()', () => {
    it('should return metrics in Prometheus format', async () => {
      const metrics = await service.getMetrics();

      expect(typeof metrics).toBe('string');
      expect(metrics).toContain('# HELP');
      expect(metrics).toContain('# TYPE');
    });

    it('should include default Node.js metrics', async () => {
      const metrics = await service.getMetrics();

      // Default metrics include process and nodejs prefixed metrics
      expect(metrics).toMatch(/nodejs_|process_/);
    });

    it('should include custom metrics definitions', async () => {
      const metrics = await service.getMetrics();

      expect(metrics).toContain('http_requests_total');
      expect(metrics).toContain('http_request_duration_seconds');
      expect(metrics).toContain('code_submissions_total');
      expect(metrics).toContain('code_submission_duration_seconds');
      expect(metrics).toContain('ai_requests_total');
      expect(metrics).toContain('active_users');
    });
  });

  // ============================================
  // recordHttpRequest() - HTTP metrics
  // ============================================
  describe('recordHttpRequest()', () => {
    it('should record HTTP request with correct labels', async () => {
      service.recordHttpRequest('GET', '/api/courses', 200, 50);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('http_requests_total{method="GET",path="/api/courses",status="200"}');
    });

    it('should record request duration in seconds', async () => {
      service.recordHttpRequest('POST', '/api/submissions', 201, 1500);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('http_request_duration_seconds');
      // Duration should be 1.5 seconds
      expect(metrics).toMatch(/http_request_duration_seconds_sum.*1\.5/);
    });

    it('should normalize paths with UUIDs', async () => {
      service.recordHttpRequest(
        'GET',
        '/api/tasks/123e4567-e89b-12d3-a456-426614174000',
        200,
        100,
      );

      const metrics = await service.getMetrics();

      // 'api' is not in known segments so it becomes :slug, 'tasks' is known, UUID becomes :id
      expect(metrics).toContain('/api/tasks/:id');
      expect(metrics).not.toContain('123e4567');
    });

    it('should normalize paths with numeric IDs', async () => {
      service.recordHttpRequest('GET', '/api/users/12345/profile', 200, 50);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('/:id');
      expect(metrics).not.toContain('12345');
    });

    it('should keep known route segments unchanged', async () => {
      service.recordHttpRequest('GET', '/api/courses/submissions', 200, 50);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('/courses');
      expect(metrics).toContain('/submissions');
    });

    it('should handle different HTTP methods', async () => {
      service.recordHttpRequest('GET', '/api/test', 200, 10);
      service.recordHttpRequest('POST', '/api/test', 201, 20);
      service.recordHttpRequest('PUT', '/api/test', 200, 30);
      service.recordHttpRequest('DELETE', '/api/test', 204, 15);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('method="GET"');
      expect(metrics).toContain('method="POST"');
      expect(metrics).toContain('method="PUT"');
      expect(metrics).toContain('method="DELETE"');
    });

    it('should handle different status codes', async () => {
      service.recordHttpRequest('GET', '/api/test', 200, 10);
      service.recordHttpRequest('GET', '/api/test', 400, 10);
      service.recordHttpRequest('GET', '/api/test', 500, 10);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('status="200"');
      expect(metrics).toContain('status="400"');
      expect(metrics).toContain('status="500"');
    });
  });

  // ============================================
  // recordSubmission() - Code submission metrics
  // ============================================
  describe('recordSubmission()', () => {
    it('should record code submission with language label', async () => {
      service.recordSubmission('python', 'passed', 2000);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('code_submissions_total{language="python",status="passed"}');
    });

    it('should record submission duration in seconds', async () => {
      service.recordSubmission('go', 'passed', 5000);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('code_submission_duration_seconds');
      // Duration should be 5 seconds
      expect(metrics).toMatch(/code_submission_duration_seconds_sum{language="go"}.*5/);
    });

    it('should handle different submission statuses', async () => {
      service.recordSubmission('javascript', 'passed', 1000);
      service.recordSubmission('javascript', 'failed', 1000);
      service.recordSubmission('javascript', 'error', 1000);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('status="passed"');
      expect(metrics).toContain('status="failed"');
      expect(metrics).toContain('status="error"');
    });

    it('should track multiple languages', async () => {
      service.recordSubmission('python', 'passed', 1000);
      service.recordSubmission('java', 'passed', 2000);
      service.recordSubmission('go', 'failed', 500);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('language="python"');
      expect(metrics).toContain('language="java"');
      expect(metrics).toContain('language="go"');
    });
  });

  // ============================================
  // recordAiRequest() - AI API metrics
  // ============================================
  describe('recordAiRequest()', () => {
    it('should record AI request with type label', async () => {
      service.recordAiRequest('hint', 'success');

      const metrics = await service.getMetrics();

      expect(metrics).toContain('ai_requests_total{type="hint",status="success"}');
    });

    it('should handle different AI request types', async () => {
      service.recordAiRequest('hint', 'success');
      service.recordAiRequest('explain', 'success');
      service.recordAiRequest('roadmap', 'error');

      const metrics = await service.getMetrics();

      expect(metrics).toContain('type="hint"');
      expect(metrics).toContain('type="explain"');
      expect(metrics).toContain('type="roadmap"');
    });

    it('should track success and error statuses', async () => {
      service.recordAiRequest('hint', 'success');
      service.recordAiRequest('hint', 'error');

      const metrics = await service.getMetrics();

      expect(metrics).toContain('status="success"');
      expect(metrics).toContain('status="error"');
    });
  });

  // ============================================
  // setActiveUsers() - Active users gauge
  // ============================================
  describe('setActiveUsers()', () => {
    it('should set active users count', async () => {
      service.setActiveUsers(42);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('active_users 42');
    });

    it('should update active users count', async () => {
      service.setActiveUsers(10);
      service.setActiveUsers(25);
      service.setActiveUsers(15);

      const metrics = await service.getMetrics();

      // Gauge should show the last set value
      expect(metrics).toContain('active_users 15');
    });

    it('should handle zero active users', async () => {
      service.setActiveUsers(100);
      service.setActiveUsers(0);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('active_users 0');
    });

    it('should initialize to 0 in onModuleInit', async () => {
      // Service already called onModuleInit in beforeEach
      const metrics = await service.getMetrics();

      expect(metrics).toContain('active_users 0');
    });
  });

  // ============================================
  // normalizePath() - Path normalization
  // ============================================
  describe('path normalization', () => {
    it('should normalize UUID in path', async () => {
      service.recordHttpRequest(
        'GET',
        '/api/items/a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        200,
        10,
      );

      const metrics = await service.getMetrics();

      expect(metrics).toContain('/:id');
      expect(metrics).not.toContain('a1b2c3d4');
    });

    it('should normalize multiple path parameters', async () => {
      service.recordHttpRequest(
        'GET',
        '/api/courses/go-basics/tasks/123',
        200,
        10,
      );

      const metrics = await service.getMetrics();

      // 'courses' and 'tasks' are known segments, 'go-basics' and '123' should be normalized
      expect(metrics).toContain('/courses');
      expect(metrics).toContain('/tasks');
    });

    it('should preserve health and metrics paths', async () => {
      service.recordHttpRequest('GET', '/health', 200, 5);
      service.recordHttpRequest('GET', '/health/metrics', 200, 5);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('path="/health"');
      expect(metrics).toContain('path="/health/metrics"');
    });

    it('should normalize dynamic slugs', async () => {
      service.recordHttpRequest('GET', '/api/courses/my-awesome-course-123', 200, 10);

      const metrics = await service.getMetrics();

      expect(metrics).not.toContain('my-awesome-course-123');
    });

    it('should handle empty path segments', async () => {
      service.recordHttpRequest('GET', '/api//courses/', 200, 10);

      const metrics = await service.getMetrics();

      // Should not throw and should record something
      expect(metrics).toContain('http_requests_total');
    });
  });

  // ============================================
  // Counter increments
  // ============================================
  describe('counter increments', () => {
    it('should increment HTTP request counter', async () => {
      service.recordHttpRequest('GET', '/api/test', 200, 10);
      service.recordHttpRequest('GET', '/api/test', 200, 10);
      service.recordHttpRequest('GET', '/api/test', 200, 10);

      const metrics = await service.getMetrics();

      // Path "/api/test" normalizes to "/api/:slug"
      expect(metrics).toContain('http_requests_total{method="GET",path="/api/:slug",status="200"} 3');
    });

    it('should increment submission counter', async () => {
      service.recordSubmission('python', 'passed', 1000);
      service.recordSubmission('python', 'passed', 1000);

      const metrics = await service.getMetrics();

      expect(metrics).toContain('code_submissions_total{language="python",status="passed"} 2');
    });

    it('should increment AI request counter', async () => {
      service.recordAiRequest('hint', 'success');
      service.recordAiRequest('hint', 'success');
      service.recordAiRequest('hint', 'success');
      service.recordAiRequest('hint', 'success');

      const metrics = await service.getMetrics();

      expect(metrics).toContain('ai_requests_total{type="hint",status="success"} 4');
    });
  });
});
