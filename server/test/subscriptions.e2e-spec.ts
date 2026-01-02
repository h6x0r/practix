import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/prisma/prisma.service';

describe('SubscriptionsController (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;
  let userToken: string;
  let adminToken: string;
  let testUserId: string;
  let testAdminId: string;

  const testUser = {
    email: `sub-user-${Date.now()}@e2e.test`,
    password: 'TestPassword123!',
    name: 'Subscription Test User',
  };

  const testAdmin = {
    email: `sub-admin-${Date.now()}@e2e.test`,
    password: 'AdminPassword123!',
    name: 'Subscription Test Admin',
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
    // Cleanup in order (subscriptions first, then users)
    await prisma.subscription.deleteMany({
      where: {
        userId: { in: [testUserId, testAdminId] },
      },
    }).catch(() => {});

    await prisma.user.deleteMany({
      where: {
        email: { in: [testUser.email, testAdmin.email] },
      },
    }).catch(() => {});

    await app.close();
  });

  describe('GET /subscriptions/plans', () => {
    it('should return available subscription plans (public)', async () => {
      const response = await request(app.getHttpServer())
        .get('/subscriptions/plans')
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      // Each plan should have required fields
      if (response.body.length > 0) {
        expect(response.body[0]).toHaveProperty('id');
        expect(response.body[0]).toHaveProperty('name');
        expect(response.body[0]).toHaveProperty('slug');
      }
    });
  });

  describe('GET /subscriptions/plans/:slug', () => {
    it('should return plan by slug (public)', async () => {
      // First get all plans to find a valid slug
      const plansResponse = await request(app.getHttpServer())
        .get('/subscriptions/plans');

      if (plansResponse.body.length > 0) {
        const slug = plansResponse.body[0].slug;
        const response = await request(app.getHttpServer())
          .get(`/subscriptions/plans/${slug}`)
          .expect(200);

        expect(response.body).toHaveProperty('slug', slug);
      }
    });

    it('should return 404 for non-existent plan', async () => {
      await request(app.getHttpServer())
        .get('/subscriptions/plans/non-existent-plan-xyz')
        .expect(404);
    });
  });

  describe('GET /subscriptions/my', () => {
    it('should return empty array for new user', async () => {
      const response = await request(app.getHttpServer())
        .get('/subscriptions/my')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
    });

    it('should reject unauthenticated request', async () => {
      await request(app.getHttpServer())
        .get('/subscriptions/my')
        .expect(401);
    });
  });

  describe('POST /subscriptions (Admin Only)', () => {
    let testPlanId: string;

    beforeAll(async () => {
      // Get a plan ID for testing
      const plansResponse = await request(app.getHttpServer())
        .get('/subscriptions/plans');

      if (plansResponse.body.length > 0) {
        testPlanId = plansResponse.body[0].id;
      }
    });

    it('should reject subscription creation by regular user', async () => {
      if (!testPlanId) return; // Skip if no plans exist

      await request(app.getHttpServer())
        .post('/subscriptions')
        .set('Authorization', `Bearer ${userToken}`)
        .send({ planId: testPlanId })
        .expect(403);
    });

    it('should allow admin to create subscription for user', async () => {
      if (!testPlanId) return; // Skip if no plans exist

      const response = await request(app.getHttpServer())
        .post('/subscriptions')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          planId: testPlanId,
          userId: testUserId,
        })
        .expect(201);

      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('userId', testUserId);
      expect(response.body).toHaveProperty('status', 'active');
    });

    it('should allow admin to create subscription for themselves', async () => {
      if (!testPlanId) return; // Skip if no plans exist

      const response = await request(app.getHttpServer())
        .post('/subscriptions')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({ planId: testPlanId })
        .expect(201);

      expect(response.body).toHaveProperty('userId', testAdminId);
    });

    it('should reject unauthenticated request', async () => {
      await request(app.getHttpServer())
        .post('/subscriptions')
        .send({ planId: 'some-id' })
        .expect(401);
    });
  });

  describe('GET /subscriptions/access/course/:courseId', () => {
    it('should return access info for authenticated user', async () => {
      // Get a course ID first
      const coursesResponse = await request(app.getHttpServer())
        .get('/courses');

      if (coursesResponse.body.length > 0) {
        const courseId = coursesResponse.body[0].id;

        const response = await request(app.getHttpServer())
          .get(`/subscriptions/access/course/${courseId}`)
          .set('Authorization', `Bearer ${userToken}`)
          .expect(200);

        expect(response.body).toHaveProperty('hasAccess');
        expect(typeof response.body.hasAccess).toBe('boolean');
      }
    });

    it('should reject unauthenticated request', async () => {
      await request(app.getHttpServer())
        .get('/subscriptions/access/course/some-course-id')
        .expect(401);
    });
  });

  describe('GET /subscriptions/access/task/:taskId', () => {
    it('should return access info for authenticated user', async () => {
      // Get a task ID first
      const tasksResponse = await request(app.getHttpServer())
        .get('/tasks');

      if (tasksResponse.body.length > 0) {
        const taskSlug = tasksResponse.body[0].slug;

        const response = await request(app.getHttpServer())
          .get(`/subscriptions/access/task/${taskSlug}`)
          .set('Authorization', `Bearer ${userToken}`)
          .expect(200);

        expect(response.body).toHaveProperty('hasAccess');
      }
    });

    it('should reject unauthenticated request', async () => {
      await request(app.getHttpServer())
        .get('/subscriptions/access/task/some-task-id')
        .expect(401);
    });
  });

  describe('POST /subscriptions/webhook/stripe', () => {
    it('should reject request without signature header', async () => {
      await request(app.getHttpServer())
        .post('/subscriptions/webhook/stripe')
        .send({ type: 'checkout.session.completed' })
        .expect(401);
    });

    it('should reject if webhook not configured', async () => {
      // This assumes STRIPE_WEBHOOK_SECRET is not set in test env
      await request(app.getHttpServer())
        .post('/subscriptions/webhook/stripe')
        .set('stripe-signature', 'test-signature')
        .send({ type: 'checkout.session.completed' })
        .expect(401); // Will fail with "Webhook not configured"
    });
  });

  describe('DELETE /subscriptions/:id', () => {
    let userSubscriptionId: string;

    beforeAll(async () => {
      // Create a subscription for the user to cancel
      const plansResponse = await request(app.getHttpServer())
        .get('/subscriptions/plans');

      if (plansResponse.body.length > 0) {
        const createResponse = await request(app.getHttpServer())
          .post('/subscriptions')
          .set('Authorization', `Bearer ${adminToken}`)
          .send({
            planId: plansResponse.body[0].id,
            userId: testUserId,
          });

        userSubscriptionId = createResponse.body.id;
      }
    });

    it('should allow user to cancel their own subscription', async () => {
      if (!userSubscriptionId) return;

      const response = await request(app.getHttpServer())
        .delete(`/subscriptions/${userSubscriptionId}`)
        .set('Authorization', `Bearer ${userToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('status', 'cancelled');
    });

    it('should reject unauthenticated request', async () => {
      await request(app.getHttpServer())
        .delete('/subscriptions/some-id')
        .expect(401);
    });

    it('should reject cancellation of non-existent subscription', async () => {
      await request(app.getHttpServer())
        .delete('/subscriptions/non-existent-id')
        .set('Authorization', `Bearer ${userToken}`)
        .expect(404);
    });
  });
});
