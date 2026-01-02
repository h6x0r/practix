import { Test, TestingModule } from '@nestjs/testing';
import { CoursesController } from './courses.controller';
import { CoursesService } from './courses.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { OptionalJwtAuthGuard } from '../auth/guards/optional-jwt.guard';
import { AdminGuard } from '../auth/guards/admin.guard';
import { NotFoundException } from '@nestjs/common';

describe('CoursesController', () => {
  let controller: CoursesController;
  let coursesService: CoursesService;

  const mockCoursesService = {
    findAll: jest.fn(),
    findOne: jest.fn(),
    getStructure: jest.fn(),
    invalidateCache: jest.fn(),
  };

  const mockCourses = [
    {
      id: 'go-basics',
      title: 'Go Basics',
      description: 'Learn Go fundamentals',
      category: 'go',
      icon: 'go',
      totalModules: 5,
      totalTopics: 20,
      progress: 0,
    },
    {
      id: 'java-core',
      title: 'Java Core',
      description: 'Java fundamentals',
      category: 'java',
      icon: 'java',
      totalModules: 8,
      totalTopics: 35,
      progress: 25,
    },
  ];

  const mockCourseStructure = [
    {
      id: 'module-1',
      title: 'Getting Started',
      description: 'Introduction to Go',
      section: 'core',
      estimatedTime: '2h',
      order: 1,
      topics: [
        {
          id: 'topic-1',
          title: 'Hello World',
          description: 'First Go program',
          difficulty: 'easy',
          estimatedTime: '30m',
          tasks: [
            {
              id: 'task-1',
              slug: 'hello-world',
              title: 'Hello World',
              difficulty: 'easy',
              estimatedTime: '10m',
              isPremium: false,
              status: 'pending',
            },
          ],
        },
      ],
    },
  ];

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [CoursesController],
      providers: [
        {
          provide: CoursesService,
          useValue: mockCoursesService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(OptionalJwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(AdminGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<CoursesController>(CoursesController);
    coursesService = module.get<CoursesService>(CoursesService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('findAll', () => {
    it('should return all courses for guest user', async () => {
      mockCoursesService.findAll.mockResolvedValue(mockCourses);

      const result = await controller.findAll({ user: null });

      expect(result).toEqual(mockCourses);
      expect(mockCoursesService.findAll).toHaveBeenCalledWith(undefined);
    });

    it('should return all courses with progress for authenticated user', async () => {
      const coursesWithProgress = mockCourses.map(c => ({
        ...c,
        progress: 50,
      }));
      mockCoursesService.findAll.mockResolvedValue(coursesWithProgress);

      const result = await controller.findAll({ user: { userId: 'user-123' } });

      expect(result).toEqual(coursesWithProgress);
      expect(mockCoursesService.findAll).toHaveBeenCalledWith('user-123');
    });

    it('should handle empty courses list', async () => {
      mockCoursesService.findAll.mockResolvedValue([]);

      const result = await controller.findAll({ user: null });

      expect(result).toEqual([]);
    });

    it('should handle service errors gracefully', async () => {
      mockCoursesService.findAll.mockRejectedValue(new Error('Database error'));

      await expect(controller.findAll({ user: null })).rejects.toThrow('Database error');
    });
  });

  describe('findOne', () => {
    const singleCourse = mockCourses[0];

    it('should return a course by id for guest', async () => {
      mockCoursesService.findOne.mockResolvedValue(singleCourse);

      const result = await controller.findOne('go-basics', { user: null });

      expect(result).toEqual(singleCourse);
      expect(mockCoursesService.findOne).toHaveBeenCalledWith('go-basics', undefined);
    });

    it('should return a course with user progress for authenticated user', async () => {
      const courseWithProgress = { ...singleCourse, progress: 75 };
      mockCoursesService.findOne.mockResolvedValue(courseWithProgress);

      const result = await controller.findOne('go-basics', { user: { userId: 'user-123' } });

      expect(result).toEqual(courseWithProgress);
      expect(mockCoursesService.findOne).toHaveBeenCalledWith('go-basics', 'user-123');
    });

    it('should throw NotFoundException for non-existent course', async () => {
      mockCoursesService.findOne.mockRejectedValue(
        new NotFoundException('Course not found')
      );

      await expect(
        controller.findOne('non-existent', { user: null })
      ).rejects.toThrow(NotFoundException);
    });

    it('should handle course slug with special characters', async () => {
      mockCoursesService.findOne.mockResolvedValue(singleCourse);

      await controller.findOne('c_go_basics', { user: null });

      expect(mockCoursesService.findOne).toHaveBeenCalledWith('c_go_basics', undefined);
    });
  });

  describe('getStructure', () => {
    it('should return course structure for guest', async () => {
      mockCoursesService.getStructure.mockResolvedValue(mockCourseStructure);

      const result = await controller.getStructure('go-basics', { user: null });

      expect(result).toEqual(mockCourseStructure);
      expect(mockCoursesService.getStructure).toHaveBeenCalledWith('go-basics', undefined);
    });

    it('should return course structure with task statuses for authenticated user', async () => {
      const structureWithStatus = mockCourseStructure.map(m => ({
        ...m,
        topics: m.topics.map(t => ({
          ...t,
          tasks: t.tasks.map(task => ({
            ...task,
            status: 'completed',
          })),
        })),
      }));
      mockCoursesService.getStructure.mockResolvedValue(structureWithStatus);

      const result = await controller.getStructure('go-basics', { user: { userId: 'user-123' } });

      expect(result[0].topics[0].tasks[0].status).toBe('completed');
      expect(mockCoursesService.getStructure).toHaveBeenCalledWith('go-basics', 'user-123');
    });

    it('should throw NotFoundException for non-existent course', async () => {
      mockCoursesService.getStructure.mockRejectedValue(
        new NotFoundException('Course not found')
      );

      await expect(
        controller.getStructure('non-existent', { user: null })
      ).rejects.toThrow(NotFoundException);
    });

    it('should handle course with multiple modules', async () => {
      const multiModuleStructure = [
        ...mockCourseStructure,
        {
          id: 'module-2',
          title: 'Advanced Topics',
          description: 'Advanced Go',
          section: 'advanced',
          estimatedTime: '4h',
          order: 2,
          topics: [],
        },
      ];
      mockCoursesService.getStructure.mockResolvedValue(multiModuleStructure);

      const result = await controller.getStructure('go-basics', { user: null });

      expect(result).toHaveLength(2);
    });
  });

  describe('invalidateCache', () => {
    it('should invalidate cache successfully (admin)', async () => {
      mockCoursesService.invalidateCache.mockResolvedValue({
        deleted: 5,
      });

      const result = await controller.invalidateCache();

      expect(result).toEqual({
        deleted: 5,
      });
      expect(mockCoursesService.invalidateCache).toHaveBeenCalled();
    });

    it('should return proper response even when no cache exists', async () => {
      mockCoursesService.invalidateCache.mockResolvedValue({
        deleted: 0,
      });

      const result = await controller.invalidateCache();

      expect(result.deleted).toBe(0);
    });
  });

  describe('edge cases', () => {
    it('should handle undefined user in request', async () => {
      mockCoursesService.findAll.mockResolvedValue(mockCourses);

      const result = await controller.findAll({ user: undefined });

      expect(mockCoursesService.findAll).toHaveBeenCalledWith(undefined);
      expect(result).toBeDefined();
    });

    it('should handle request with user but no userId', async () => {
      mockCoursesService.findAll.mockResolvedValue(mockCourses);

      const result = await controller.findAll({ user: {} });

      expect(mockCoursesService.findAll).toHaveBeenCalledWith(undefined);
      expect(result).toBeDefined();
    });

    it('should handle large course list', async () => {
      const largeCourseList = Array.from({ length: 100 }, (_, i) => ({
        id: `course-${i}`,
        title: `Course ${i}`,
        description: `Description ${i}`,
        category: 'misc',
        icon: 'default',
        totalModules: i,
        totalTopics: i * 5,
        progress: 0,
      }));
      mockCoursesService.findAll.mockResolvedValue(largeCourseList);

      const result = await controller.findAll({ user: null });

      expect(result).toHaveLength(100);
    });

    it('should handle structure with deeply nested tasks', async () => {
      const deepStructure = [
        {
          id: 'module-1',
          title: 'Module',
          description: 'Desc',
          section: 'core',
          estimatedTime: '1h',
          order: 1,
          topics: Array.from({ length: 10 }, (_, i) => ({
            id: `topic-${i}`,
            title: `Topic ${i}`,
            description: `Topic desc ${i}`,
            difficulty: 'medium',
            estimatedTime: '10m',
            tasks: Array.from({ length: 5 }, (_, j) => ({
              id: `task-${i}-${j}`,
              slug: `task-${i}-${j}`,
              title: `Task ${j}`,
              difficulty: 'easy',
              estimatedTime: '5m',
              isPremium: false,
              status: 'pending',
            })),
          })),
        },
      ];
      mockCoursesService.getStructure.mockResolvedValue(deepStructure);

      const result = await controller.getStructure('complex-course', { user: null });

      expect(result[0].topics).toHaveLength(10);
      expect(result[0].topics[0].tasks).toHaveLength(5);
    });
  });
});
