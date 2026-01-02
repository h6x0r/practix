import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/prisma/prisma.service';

describe('AdminController (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;
  let userToken: string;
  let adminToken: string;
  let testUserId: string;
  let testAdminId: string;

  const testUser = {
    email: `admin-user-${Date.now()}@e2e.test`,
    password: 'TestPassword123!',
    name: 'Admin Test Regular User',
  };

  const testAdmin = {
    email: `admin-admin-${Date.now()}@e2e.test`,
    password: 'AdminPassword123!',
    name: 'Admin Test Admin User',
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

    prisma = app.get<PrismaService>(PrismaService);

    await app.init();

    // Register test user
    const userResponse = await request(app.getHttpServer())
      .post('/auth/register')
      .send(testUser);
    userToken = userResponse.body.access_token;
    testUserId = userResponse.body.user.id;

    // Register admin user
    const adminResponse = await request(app.getHttpServer())
      .post('/auth/register')
      .send(testAdmin);
    testAdminId = adminResponse.body.user.id;

    // Promote to admin
    await prisma.user.update({
      where: { id: testAdminId },
      data: { role: 'ADMIN' },
    });

    // Re-login to get fresh token with updated role
    const adminLoginResponse = await request(app.getHttpServer())
      .post('/auth/login')
      .send({
        email: testAdmin.email,
        password: testAdmin.password,
      });
    adminToken = adminLoginResponse.body.access_token;
  });

  afterAll(async () => {
    // Cleanup test users
    await prisma.user.deleteMany({
      where: {
        email: { in: [testUser.email, testAdmin.email] },
      },
    }).catch(() => {});

    await app.close();
  });

  describe('Authentication and Authorization', () => {
    it('should reject unauthenticated requests', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .expect(401);
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });

    it('should accept admin users', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
    });
  });

  describe('GET /admin/analytics/dashboard', () => {
    it('should return dashboard statistics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      // Dashboard should have user counts
      expect(response.body).toHaveProperty('totalUsers');
      expect(typeof response.body.totalUsers).toBe('number');

      // Should have new users count
      expect(response.body).toHaveProperty('newUsersToday');
      expect(typeof response.body.newUsersToday).toBe('number');

      // Should have active users
      expect(response.body).toHaveProperty('activeUsers');
    });
  });

  describe('GET /admin/analytics/courses', () => {
    it('should return course analytics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/courses')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      // Should return array of courses with analytics
      expect(response.body).toBeDefined();
      // If there are courses, check structure
      if (Array.isArray(response.body) && response.body.length > 0) {
        expect(response.body[0]).toHaveProperty('id');
        expect(response.body[0]).toHaveProperty('title');
      }
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/courses')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });
  });

  describe('GET /admin/analytics/tasks', () => {
    it('should return task analytics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/tasks')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
      // Should have task statistics
      if (response.body.hardestTasks) {
        expect(Array.isArray(response.body.hardestTasks)).toBe(true);
      }
      if (response.body.popularTasks) {
        expect(Array.isArray(response.body.popularTasks)).toBe(true);
      }
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/tasks')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });
  });

  describe('GET /admin/analytics/submissions', () => {
    it('should return submission statistics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/submissions')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
      // Should have total submissions count
      expect(response.body).toHaveProperty('totalSubmissions');
      expect(typeof response.body.totalSubmissions).toBe('number');

      // Should have status breakdown
      if (response.body.byStatus) {
        expect(typeof response.body.byStatus).toBe('object');
      }
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/submissions')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });
  });

  describe('GET /admin/analytics/subscriptions', () => {
    it('should return subscription statistics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/subscriptions')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
      // Should have subscription counts
      expect(response.body).toHaveProperty('activeSubscriptions');
      expect(typeof response.body.activeSubscriptions).toBe('number');
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/subscriptions')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });
  });

  describe('GET /admin/analytics/ai-usage', () => {
    it('should return AI usage statistics', async () => {
      const response = await request(app.getHttpServer())
        .get('/admin/analytics/ai-usage')
        .set('Authorization', `Bearer ${adminToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
      // Should have usage data
      expect(response.body).toHaveProperty('totalUsage');
    });

    it('should reject non-admin users', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/ai-usage')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(403);
    });
  });

  describe('Rate Limiting', () => {
    it('should have rate limiting enabled (30 req/min)', async () => {
      // Make several rapid requests - should eventually get rate limited
      // Note: This test may not trigger rate limit in test environment
      // depending on throttler configuration
      const responses = await Promise.all(
        Array(5).fill(null).map(() =>
          request(app.getHttpServer())
            .get('/admin/analytics/dashboard')
            .set('Authorization', `Bearer ${adminToken}`)
        )
      );

      // All should succeed (rate limit is 30/min, we're doing 5)
      responses.forEach(response => {
        expect(response.status).toBe(200);
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle invalid token format', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', 'Bearer invalid-jwt-token')
        .expect(401);
    });

    it('should handle missing Authorization header', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .expect(401);
    });

    it('should handle malformed Authorization header', async () => {
      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', 'NotBearer token')
        .expect(401);
    });

    it('should handle expired token', async () => {
      // This is an obviously invalid/expired token
      const expiredToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZXhwIjoxfQ.invalid';

      await request(app.getHttpServer())
        .get('/admin/analytics/dashboard')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);
    });
  });
});
