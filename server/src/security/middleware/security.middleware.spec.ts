import { Test, TestingModule } from '@nestjs/testing';
import { Request, Response, NextFunction } from 'express';
import { SecurityMiddleware } from './security.middleware';
import { ActivityLoggerService, SecurityEventType } from '../activity-logger.service';
import { IpBanService } from '../ip-ban.service';
import { ForbiddenException } from '@nestjs/common';

describe('SecurityMiddleware', () => {
  let middleware: SecurityMiddleware;
  let activityLogger: ActivityLoggerService;
  let ipBanService: IpBanService;

  const mockActivityLogger = {
    logSuspiciousRequest: jest.fn().mockResolvedValue(undefined),
  };

  const mockIpBanService = {
    checkAndThrow: jest.fn().mockResolvedValue(undefined),
    addStrike: jest.fn().mockResolvedValue(undefined),
  };

  const createMockRequest = (overrides: Partial<Request> = {}): Request => ({
    path: '/api/test',
    method: 'GET',
    query: {},
    params: {},
    body: {},
    headers: {},
    ip: '127.0.0.1',
    socket: { remoteAddress: '127.0.0.1' },
    ...overrides,
  } as unknown as Request);

  const createMockResponse = (): Response => {
    const res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
    return res as unknown as Response;
  };

  const createMockNext = (): NextFunction => jest.fn();

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SecurityMiddleware,
        { provide: ActivityLoggerService, useValue: mockActivityLogger },
        { provide: IpBanService, useValue: mockIpBanService },
      ],
    }).compile();

    middleware = module.get<SecurityMiddleware>(SecurityMiddleware);
    activityLogger = module.get<ActivityLoggerService>(ActivityLoggerService);
    ipBanService = module.get<IpBanService>(IpBanService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(middleware).toBeDefined();
  });

  // ============================================
  // Normal Request Flow
  // ============================================
  describe('normal requests', () => {
    it('should pass through normal requests', async () => {
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('should pass through requests with safe query params', async () => {
      const req = createMockRequest({
        query: { page: '1', limit: '10', search: 'hello world' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should pass through requests with safe body', async () => {
      const req = createMockRequest({
        method: 'POST',
        body: { email: 'test@example.com', name: 'John Doe' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  // ============================================
  // IP Ban Check
  // ============================================
  describe('IP ban check', () => {
    it('should block banned IPs', async () => {
      mockIpBanService.checkAndThrow.mockRejectedValueOnce(new ForbiddenException('IP banned'));
      const req = createMockRequest();
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(res.status).toHaveBeenCalledWith(403);
      expect(res.json).toHaveBeenCalledWith({
        statusCode: 403,
        message: 'Access denied',
        error: 'IP_BANNED',
      });
      expect(next).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // SQL Injection Detection
  // ============================================
  describe('SQL injection detection', () => {
    it('should detect SQL injection with UNION SELECT', async () => {
      const req = createMockRequest({
        query: { id: "1 UNION SELECT * FROM users" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        SecurityEventType.SQL_INJECTION_ATTEMPT,
        '127.0.0.1',
        undefined,
        expect.any(Object),
      );
      expect(mockIpBanService.addStrike).toHaveBeenCalled();
      expect(res.status).toHaveBeenCalledWith(400);
      expect(next).not.toHaveBeenCalled();
    });

    it('should detect SQL injection with single quote', async () => {
      const req = createMockRequest({
        query: { name: "'; DROP TABLE users;--" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        SecurityEventType.SQL_INJECTION_ATTEMPT,
        expect.any(String),
        undefined,
        expect.any(Object),
      );
      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should detect DROP TABLE injection', async () => {
      const req = createMockRequest({
        body: { query: "DROP TABLE students" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    // Note: INSERT, UPDATE, DELETE patterns were intentionally removed
    // to reduce false positives in legitimate code submissions
    it('should allow INSERT INTO (not blocked to reduce false positives)', async () => {
      const req = createMockRequest({
        body: { data: "INSERT INTO users (admin) VALUES (1)" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      // Should pass through - these patterns are not checked anymore
      expect(next).toHaveBeenCalled();
    });

    it('should allow UPDATE SET (not blocked to reduce false positives)', async () => {
      const req = createMockRequest({
        body: { data: "UPDATE users SET admin = 1" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(next).toHaveBeenCalled();
    });

    it('should allow DELETE FROM (not blocked to reduce false positives)', async () => {
      const req = createMockRequest({
        body: { data: "DELETE FROM users WHERE 1=1" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(next).toHaveBeenCalled();
    });
  });

  // ============================================
  // XSS Detection
  // ============================================
  describe('XSS detection', () => {
    it('should detect script tags', async () => {
      const req = createMockRequest({
        body: { comment: '<script>alert("xss")</script>' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        SecurityEventType.XSS_ATTEMPT,
        expect.any(String),
        undefined,
        expect.any(Object),
      );
      expect(mockIpBanService.addStrike).toHaveBeenCalled();
      // XSS allows request through (might be false positive), just logs
      expect(next).toHaveBeenCalled();
    });

    it('should detect javascript: protocol', async () => {
      const req = createMockRequest({
        body: { url: 'javascript:alert(1)' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        SecurityEventType.XSS_ATTEMPT,
        expect.any(String),
        undefined,
        expect.any(Object),
      );
    });

    it('should detect onclick handlers', async () => {
      const req = createMockRequest({
        body: { html: '<div onclick="evil()">Click</div>' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalled();
    });

    it('should detect iframe tags', async () => {
      const req = createMockRequest({
        body: { content: '<iframe src="evil.com"></iframe>' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalled();
    });

    it('should detect svg onload', async () => {
      const req = createMockRequest({
        body: { content: '<svg onload="alert(1)">' },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalled();
    });
  });

  // ============================================
  // Path Traversal Detection
  // ============================================
  describe('path traversal detection', () => {
    it('should detect ../ in path', async () => {
      const req = createMockRequest({
        path: '/api/files/../../../etc/passwd',
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
        expect.any(String),
        undefined,
        expect.any(Object),
      );
      expect(res.status).toHaveBeenCalledWith(400);
      expect(next).not.toHaveBeenCalled();
    });

    it('should detect URL-encoded path traversal', async () => {
      const req = createMockRequest({
        path: '/api/files/..%2f..%2fetc/passwd',
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should detect double-encoded path traversal', async () => {
      const req = createMockRequest({
        path: '/api/%252e%252e%252f',
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should detect backslash path traversal', async () => {
      const req = createMockRequest({
        path: '/api/files/..\\..\\windows\\system32',
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
    });
  });

  // ============================================
  // IP Detection
  // ============================================
  describe('IP detection', () => {
    it('should extract IP from x-forwarded-for header', async () => {
      const req = createMockRequest({
        headers: { 'x-forwarded-for': '192.168.1.1, 10.0.0.1' },
        query: { id: "1 UNION SELECT *" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        '192.168.1.1',
        undefined,
        expect.any(Object),
      );
    });

    it('should extract IP from x-forwarded-for array', async () => {
      const req = createMockRequest({
        headers: { 'x-forwarded-for': ['10.0.0.1', '192.168.1.1'] },
        query: { id: "1 UNION SELECT *" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        '10.0.0.1',
        undefined,
        expect.any(Object),
      );
    });

    it('should extract IP from x-real-ip header', async () => {
      const req = createMockRequest({
        headers: { 'x-real-ip': '10.10.10.10' },
        query: { id: "1 UNION SELECT *" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        '10.10.10.10',
        undefined,
        expect.any(Object),
      );
    });

    it('should extract IP from x-real-ip array', async () => {
      const req = createMockRequest({
        headers: { 'x-real-ip': ['10.10.10.10', '20.20.20.20'] },
        query: { id: "1 UNION SELECT *" },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        '10.10.10.10',
        undefined,
        expect.any(Object),
      );
    });

    it('should fallback to socket remoteAddress', async () => {
      const req = createMockRequest({
        ip: undefined,
        query: { id: "1 UNION SELECT *" },
      });
      // Override socket.remoteAddress
      (req as any).socket = { remoteAddress: '192.168.0.1' };
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        '192.168.0.1',
        undefined,
        expect.any(Object),
      );
    });
  });

  // ============================================
  // User ID Logging
  // ============================================
  describe('user ID logging', () => {
    it('should include userId when user is authenticated', async () => {
      const req = createMockRequest({
        query: { id: "1 UNION SELECT *" },
      });
      (req as any).user = { userId: 'user-123' };
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      expect(mockActivityLogger.logSuspiciousRequest).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(String),
        'user-123',
        expect.any(Object),
      );
    });
  });

  // ============================================
  // Code Submission Exclusion
  // ============================================
  describe('code submission exclusion', () => {
    it('should check body for /submissions/run endpoint (DROP TABLE blocked)', async () => {
      const req = createMockRequest({
        path: '/submissions/run',
        method: 'POST',
        body: {
          code: "SELECT * FROM users; DROP TABLE users;--",
          language: 'sql',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      // DROP TABLE pattern is detected and blocked
      // Note: This is intentional - malicious SQL should be blocked even in code submissions
      // Legitimate SQL learning should use proper escaping/formatting
      expect(res.status).toHaveBeenCalledWith(400);
    });

    it('should allow safe code in /submissions/run endpoint', async () => {
      const req = createMockRequest({
        path: '/submissions/run',
        method: 'POST',
        body: {
          code: "SELECT * FROM users WHERE id = 1",
          language: 'sql',
        },
      });
      const res = createMockResponse();
      const next = createMockNext();

      await middleware.use(req, res, next);

      // Safe SQL patterns should pass
      expect(next).toHaveBeenCalled();
    });
  });
});
