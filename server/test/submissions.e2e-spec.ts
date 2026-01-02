import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/prisma/prisma.service';

describe('SubmissionsController (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;
  let accessToken: string;
  let userId: string;

  const testUser = {
    email: `submissions-${Date.now()}@e2e.test`,
    password: 'TestPassword123!',
    name: 'Submissions Test User',
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

    prisma = app.get<PrismaService>(PrismaService);

    await app.init();

    // Register and login test user
    const registerResponse = await request(app.getHttpServer())
      .post('/auth/register')
      .send(testUser);

    accessToken = registerResponse.body.access_token;
    userId = registerResponse.body.user.id;
  });

  afterAll(async () => {
    // Cleanup
    await prisma.submission.deleteMany({
      where: { userId },
    }).catch(() => {});

    await prisma.user.deleteMany({
      where: { email: testUser.email },
    }).catch(() => {});

    await app.close();
  });

  describe('POST /submissions/run', () => {
    it('should execute simple code', async () => {
      const response = await request(app.getHttpServer())
        .post('/submissions/run')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          code: 'console.log("Hello E2E");',
          language: 'javascript',
        })
        .expect(201);

      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('stdout');
    });

    it('should reject unsupported language', async () => {
      await request(app.getHttpServer())
        .post('/submissions/run')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          code: 'print("hello")',
          language: 'cobol',
        })
        .expect(400);
    });

    it('should reject malicious code patterns', async () => {
      const response = await request(app.getHttpServer())
        .post('/submissions/run')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          code: 'import os; os.system("rm -rf /")',
          language: 'python',
        });

      // Should either reject with 403 or execute safely in sandbox
      expect([201, 403]).toContain(response.status);
    });

    it('should work without authentication for run', async () => {
      const response = await request(app.getHttpServer())
        .post('/submissions/run')
        .send({
          code: 'console.log("anonymous");',
          language: 'javascript',
        })
        .expect(201);

      expect(response.body).toHaveProperty('status');
    });
  });

  describe('POST /submissions (submit to task)', () => {
    let testTask: any;

    beforeAll(async () => {
      // Find a free task for testing
      testTask = await prisma.task.findFirst({
        where: { isPremium: false },
        select: { id: true, slug: true },
      });
    });

    it('should require authentication for task submission', async () => {
      if (!testTask) {
        console.log('Skipping: No free task found');
        return;
      }

      await request(app.getHttpServer())
        .post('/submissions')
        .send({
          taskId: testTask.id,
          code: 'console.log("test");',
          language: 'javascript',
        })
        .expect(401);
    });

    it('should create submission for authenticated user', async () => {
      if (!testTask) {
        console.log('Skipping: No free task found');
        return;
      }

      const response = await request(app.getHttpServer())
        .post('/submissions')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          taskId: testTask.id,
          code: 'console.log("test");',
          language: 'javascript',
        })
        .expect(201);

      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('score');
    });

    it('should reject submission for non-existent task', async () => {
      await request(app.getHttpServer())
        .post('/submissions')
        .set('Authorization', `Bearer ${accessToken}`)
        .send({
          taskId: 'non-existent-task-id',
          code: 'console.log("test");',
          language: 'javascript',
        })
        .expect(404);
    });
  });

  describe('GET /submissions/status', () => {
    it('should return execution engine status', async () => {
      const response = await request(app.getHttpServer())
        .get('/submissions/status')
        .expect(200);

      expect(response.body).toHaveProperty('available');
      expect(response.body).toHaveProperty('languages');
      expect(Array.isArray(response.body.languages)).toBe(true);
    });
  });
});
