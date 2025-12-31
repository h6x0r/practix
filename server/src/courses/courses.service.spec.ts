import { Test, TestingModule } from '@nestjs/testing';
import { CoursesService } from './courses.service';
import { PrismaService } from '../prisma/prisma.service';
import { NotFoundException } from '@nestjs/common';

describe('CoursesService', () => {
  let service: CoursesService;
  let prisma: PrismaService;

  const mockPrisma = {
    course: {
      findMany: jest.fn(),
      findUnique: jest.fn(),
    },
    submission: {
      findMany: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CoursesService,
        {
          provide: PrismaService,
          useValue: mockPrisma,
        },
      ],
    }).compile();

    service = module.get<CoursesService>(CoursesService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('findAll', () => {
    it('should return all courses with progress', async () => {
      const mockCourses = [
        {
          id: 'course-1',
          slug: 'go-basics',
          title: 'Go Basics',
          description: 'Learn Go fundamentals',
          category: 'go',
          icon: 'go',
          estimatedTime: '10h',
          order: 1,
          translations: {},
          modules: [
            {
              topics: [
                { tasks: [{ id: 'task-1' }, { id: 'task-2' }] },
              ],
            },
          ],
        },
      ];

      mockPrisma.course.findMany.mockResolvedValue(mockCourses);
      mockPrisma.submission.findMany.mockResolvedValue([]);

      const result = await service.findAll();

      expect(result).toHaveLength(1);
      expect(result[0].id).toBe('go-basics');
      expect(result[0].totalModules).toBe(1);
      expect(result[0].progress).toBe(0);
    });

    it('should calculate progress correctly when user has completed tasks', async () => {
      const mockCourses = [
        {
          id: 'course-1',
          slug: 'go-basics',
          title: 'Go Basics',
          description: 'Learn Go fundamentals',
          category: 'go',
          icon: 'go',
          estimatedTime: '10h',
          order: 1,
          translations: {},
          modules: [
            {
              topics: [
                { tasks: [{ id: 'task-1' }, { id: 'task-2' }] },
              ],
            },
          ],
        },
      ];

      const mockSubmissions = [
        {
          taskId: 'task-1',
          task: { topic: { module: { courseId: 'course-1' } } },
        },
      ];

      mockPrisma.course.findMany.mockResolvedValue(mockCourses);
      mockPrisma.submission.findMany.mockResolvedValue(mockSubmissions);

      const result = await service.findAll('user-123');

      expect(result[0].progress).toBe(50); // 1 out of 2 tasks completed
    });
  });

  describe('getStructure', () => {
    it('should return course structure with modules and topics', async () => {
      const mockCourse = {
        id: 'course-1',
        slug: 'go-basics',
        modules: [
          {
            id: 'module-1',
            title: 'Getting Started',
            description: 'Introduction',
            section: 'core',
            estimatedTime: '2h',
            translations: {},
            order: 1,
            topics: [
              {
                id: 'topic-1',
                title: 'Hello World',
                description: 'First program',
                difficulty: 'easy',
                estimatedTime: '30m',
                translations: {},
                order: 1,
                tasks: [
                  {
                    id: 'task-1',
                    slug: 'hello-world',
                    title: 'Hello World',
                    difficulty: 'easy',
                    estimatedTime: '10m',
                    isPremium: false,
                    translations: {},
                  },
                ],
              },
            ],
          },
        ],
      };

      mockPrisma.course.findUnique.mockResolvedValue(mockCourse);
      mockPrisma.submission.findMany.mockResolvedValue([]);

      const result = await service.getStructure('go-basics');

      expect(result).toHaveLength(1);
      expect(result[0].title).toBe('Getting Started');
      expect(result[0].topics).toHaveLength(1);
      expect(result[0].topics[0].tasks).toHaveLength(1);
      expect(result[0].topics[0].tasks[0].status).toBe('pending');
    });

    it('should throw NotFoundException if course not found', async () => {
      mockPrisma.course.findUnique.mockResolvedValue(null);

      await expect(service.getStructure('non-existent')).rejects.toThrow(NotFoundException);
    });

    it('should mark completed tasks correctly', async () => {
      const mockCourse = {
        id: 'course-1',
        slug: 'go-basics',
        modules: [
          {
            id: 'module-1',
            title: 'Getting Started',
            description: 'Introduction',
            section: 'core',
            estimatedTime: '2h',
            translations: {},
            order: 1,
            topics: [
              {
                id: 'topic-1',
                title: 'Hello World',
                description: 'First program',
                difficulty: 'easy',
                estimatedTime: '30m',
                translations: {},
                order: 1,
                tasks: [
                  { id: 'task-1', slug: 'task-1', title: 'Task 1', difficulty: 'easy', estimatedTime: '10m', isPremium: false, translations: {} },
                  { id: 'task-2', slug: 'task-2', title: 'Task 2', difficulty: 'easy', estimatedTime: '10m', isPremium: false, translations: {} },
                ],
              },
            ],
          },
        ],
      };

      mockPrisma.course.findUnique.mockResolvedValue(mockCourse);
      mockPrisma.submission.findMany.mockResolvedValue([{ taskId: 'task-1' }]);

      const result = await service.getStructure('go-basics', 'user-123');

      expect(result[0].topics[0].tasks[0].status).toBe('completed');
      expect(result[0].topics[0].tasks[1].status).toBe('pending');
    });
  });
});
