import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';
import { PrismaService } from '../src/prisma/prisma.service';

describe('CoursesController (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;
  let accessToken: string;
  let userId: string;

  const testUser = {
    email: `courses-${Date.now()}@e2e.test`,
    password: 'TestPassword123!',
    name: 'Courses Test User',
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
    await prisma.user.deleteMany({
      where: { email: testUser.email },
    }).catch(() => {});

    await app.close();
  });

  describe('GET /courses', () => {
    it('should return list of courses', async () => {
      const response = await request(app.getHttpServer())
        .get('/courses')
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);

      if (response.body.length > 0) {
        const course = response.body[0];
        expect(course).toHaveProperty('id');
        expect(course).toHaveProperty('title');
        expect(course).toHaveProperty('description');
      }
    });

    it('should include progress for authenticated user', async () => {
      const response = await request(app.getHttpServer())
        .get('/courses')
        .set('Authorization', `Bearer ${accessToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);

      if (response.body.length > 0) {
        const course = response.body[0];
        expect(course).toHaveProperty('progress');
        expect(typeof course.progress).toBe('number');
      }
    });
  });

  describe('GET /courses/:slug/structure', () => {
    let testCourse: any;

    beforeAll(async () => {
      testCourse = await prisma.course.findFirst({
        select: { slug: true },
      });
    });

    it('should return course structure', async () => {
      if (!testCourse) {
        console.log('Skipping: No course found');
        return;
      }

      const response = await request(app.getHttpServer())
        .get(`/courses/${testCourse.slug}/structure`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);

      if (response.body.length > 0) {
        const module = response.body[0];
        expect(module).toHaveProperty('id');
        expect(module).toHaveProperty('title');
        expect(module).toHaveProperty('topics');
      }
    });

    it('should return 404 for non-existent course', async () => {
      await request(app.getHttpServer())
        .get('/courses/non-existent-course/structure')
        .expect(404);
    });
  });
});

describe('TasksController (e2e)', () => {
  let app: INestApplication;
  let prisma: PrismaService;
  let accessToken: string;

  const testUser = {
    email: `tasks-${Date.now()}@e2e.test`,
    password: 'TestPassword123!',
    name: 'Tasks Test User',
  };

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

    prisma = app.get<PrismaService>(PrismaService);

    await app.init();

    const registerResponse = await request(app.getHttpServer())
      .post('/auth/register')
      .send(testUser);

    accessToken = registerResponse.body.access_token;
  });

  afterAll(async () => {
    await prisma.user.deleteMany({
      where: { email: testUser.email },
    }).catch(() => {});

    await app.close();
  });

  describe('GET /tasks/:identifier', () => {
    let freeTask: any;
    let premiumTask: any;

    beforeAll(async () => {
      freeTask = await prisma.task.findFirst({
        where: { isPremium: false },
        select: { id: true, slug: true, title: true },
      });

      premiumTask = await prisma.task.findFirst({
        where: { isPremium: true },
        select: { id: true, slug: true, title: true },
      });
    });

    it('should return free task details by slug', async () => {
      if (!freeTask) {
        console.log('Skipping: No free task found');
        return;
      }

      const response = await request(app.getHttpServer())
        .get(`/tasks/${freeTask.slug}`)
        .set('Authorization', `Bearer ${accessToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('title');
      expect(response.body).toHaveProperty('description');
      expect(response.body).toHaveProperty('initialCode');
    });

    it('should return free task details by id', async () => {
      if (!freeTask) {
        console.log('Skipping: No free task found');
        return;
      }

      const response = await request(app.getHttpServer())
        .get(`/tasks/${freeTask.id}`)
        .set('Authorization', `Bearer ${accessToken}`)
        .expect(200);

      expect(response.body.slug).toBe(freeTask.slug);
    });

    it('should return 404 for non-existent task', async () => {
      await request(app.getHttpServer())
        .get('/tasks/non-existent-task')
        .set('Authorization', `Bearer ${accessToken}`)
        .expect(404);
    });

    it('should restrict premium task access for free user', async () => {
      if (!premiumTask) {
        console.log('Skipping: No premium task found');
        return;
      }

      const response = await request(app.getHttpServer())
        .get(`/tasks/${premiumTask.slug}`)
        .set('Authorization', `Bearer ${accessToken}`);

      // Either returns locked version (200) or forbidden (403)
      expect([200, 403]).toContain(response.status);

      if (response.status === 200) {
        // Should have isPremium flag
        expect(response.body.isPremium).toBe(true);
      }
    });
  });
});
