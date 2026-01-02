import { Test, TestingModule } from '@nestjs/testing';
import { CoursesService } from './courses.service';
import { PrismaService } from '../prisma/prisma.service';
import { CacheService } from '../cache/cache.service';
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

  const mockCacheService = {
    get: jest.fn().mockResolvedValue(null),
    set: jest.fn(),
    delete: jest.fn(),
    deleteByPattern: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CoursesService,
        {
          provide: PrismaService,
          useValue: mockPrisma,
        },
        {
          provide: CacheService,
          useValue: mockCacheService,
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
                {
                  title: 'Hello World',
                  translations: {},
                  _count: { tasks: 2 },
                  tasks: [{ id: 'task-1' }],
                },
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
                {
                  title: 'Hello World',
                  translations: {},
                  _count: { tasks: 2 },
                  tasks: [{ id: 'task-1' }],
                },
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

    it('should return cached structure on cache hit', async () => {
      const cachedStructure = {
        courseId: 'course-1',
        modules: [
          {
            id: 'module-1',
            title: 'Cached Module',
            description: 'From cache',
            section: 'core',
            estimatedTime: '2h',
            translations: {},
            topics: [
              {
                id: 'topic-1',
                title: 'Cached Topic',
                description: 'From cache',
                difficulty: 'easy',
                estimatedTime: '30m',
                translations: {},
                tasks: [
                  { id: 'task-1', slug: 'task-1', title: 'Cached Task', difficulty: 'easy', estimatedTime: '10m', isPremium: false, translations: {} }
                ]
              }
            ]
          }
        ]
      };

      mockCacheService.get.mockResolvedValue(cachedStructure);

      const result = await service.getStructure('go-basics');

      expect(result[0].title).toBe('Cached Module');
      expect(mockPrisma.course.findUnique).not.toHaveBeenCalled();
    });
  });

  describe('findOne', () => {
    const mockCourse = {
      id: 'course-1',
      slug: 'go-basics',
      title: 'Go Basics',
      description: 'Learn Go',
      category: 'go',
      icon: 'go',
      estimatedTime: '10h',
      order: 1,
      translations: {},
    };

    it('should return course by slug', async () => {
      // findOne calls findAll internally, so we need to set up findAll mocks
      const cachedCourses = [{
        id: 'go-basics',
        slug: 'go-basics',
        uuid: 'course-1',
        title: 'Go Basics',
        description: 'Learn Go',
        category: 'go',
        icon: 'go',
        estimatedTime: '10h',
        totalModules: 1,
        totalTasks: 2,
        translations: {},
        sampleTopics: []
      }];

      mockPrisma.course.findUnique.mockResolvedValue(mockCourse);
      mockCacheService.get.mockResolvedValue(cachedCourses);

      const result = await service.findOne('go-basics');

      expect(result?.id).toBe('go-basics');
      expect(mockPrisma.course.findUnique).toHaveBeenCalledWith({ where: { slug: 'go-basics' } });
    });

    it('should fallback to ID lookup if slug not found', async () => {
      mockPrisma.course.findUnique
        .mockResolvedValueOnce(null) // First call with slug returns null
        .mockResolvedValueOnce(mockCourse); // Second call with ID

      const result = await service.findOne('course-1');

      expect(mockPrisma.course.findUnique).toHaveBeenCalledWith({ where: { slug: 'course-1' } });
      expect(mockPrisma.course.findUnique).toHaveBeenCalledWith({ where: { id: 'course-1' } });
    });

    it('should return null if course not found by slug or id', async () => {
      mockPrisma.course.findUnique.mockResolvedValue(null);

      const result = await service.findOne('non-existent');

      // When slug lookup fails, it returns the result of ID lookup (which is null)
      expect(result).toBeNull();
    });
  });

  describe('findAll with cache', () => {
    it('should return cached courses on cache hit', async () => {
      const cachedCourses = [
        {
          id: 'cached-course',
          slug: 'cached-course',
          uuid: 'uuid-1',
          title: 'Cached Course',
          description: 'From cache',
          category: 'go',
          icon: 'go',
          estimatedTime: '5h',
          totalModules: 2,
          totalTasks: 10,
          translations: {},
          sampleTopics: []
        }
      ];

      mockCacheService.get.mockResolvedValue(cachedCourses);

      const result = await service.findAll();

      expect(result[0].title).toBe('Cached Course');
      expect(result[0].progress).toBe(0);
      expect(mockPrisma.course.findMany).not.toHaveBeenCalled();
    });

    it('should calculate progress from cached courses with user submissions', async () => {
      const cachedCourses = [
        {
          id: 'cached-course',
          slug: 'cached-course',
          uuid: 'uuid-1',
          title: 'Cached Course',
          description: 'From cache',
          category: 'go',
          icon: 'go',
          estimatedTime: '5h',
          totalModules: 2,
          totalTasks: 10,
          translations: {},
          sampleTopics: []
        }
      ];

      mockCacheService.get.mockResolvedValue(cachedCourses);
      mockPrisma.submission.findMany.mockResolvedValue([
        { taskId: 't1', task: { topic: { module: { courseId: 'uuid-1' } } } },
        { taskId: 't2', task: { topic: { module: { courseId: 'uuid-1' } } } },
      ]);

      const result = await service.findAll('user-123');

      expect(result[0].progress).toBe(20); // 2 out of 10 tasks = 20%
    });

    it('should handle errors in findAll gracefully', async () => {
      mockCacheService.get.mockResolvedValue(null);
      mockPrisma.course.findMany.mockRejectedValue(new Error('Database error'));

      await expect(service.findAll()).rejects.toThrow('Database error');
    });
  });

  describe('invalidateCache', () => {
    it('should delete all course caches', async () => {
      mockCacheService.deleteByPattern.mockResolvedValue(5);

      const result = await service.invalidateCache();

      expect(result).toEqual({ deleted: 5 });
      expect(mockCacheService.deleteByPattern).toHaveBeenCalledWith('courses:*');
    });

    it('should return 0 if no caches to delete', async () => {
      mockCacheService.deleteByPattern.mockResolvedValue(0);

      const result = await service.invalidateCache();

      expect(result).toEqual({ deleted: 0 });
    });
  });
});
