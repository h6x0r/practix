import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { UserCoursesController } from './user-courses.controller';
import { UserCoursesService } from './user-courses.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

describe('UserCoursesController', () => {
  let controller: UserCoursesController;
  let userCoursesService: UserCoursesService;

  const mockCourse = {
    id: 'course-123',
    slug: 'go-basics',
    title: 'Go Basics',
    description: 'Learn Go fundamentals',
    category: 'go',
    icon: '🐹',
    estimatedTime: '20h',
    translations: {},
    progress: 50,
    startedAt: new Date('2025-01-01'),
    lastAccessedAt: new Date('2025-01-15'),
    completedAt: null,
  };

  const mockUserCoursesService = {
    getUserCourses: jest.fn(),
    startCourse: jest.fn(),
    updateProgress: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserCoursesController],
      providers: [
        {
          provide: UserCoursesService,
          useValue: mockUserCoursesService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<UserCoursesController>(UserCoursesController);
    userCoursesService = module.get<UserCoursesService>(UserCoursesService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getUserCourses', () => {
    it('should return user courses', async () => {
      mockUserCoursesService.getUserCourses.mockResolvedValue([mockCourse]);

      const result = await controller.getUserCourses({ user: { userId: 'user-123' } });

      expect(result).toEqual([mockCourse]);
      expect(result[0].progress).toBe(50);
      expect(mockUserCoursesService.getUserCourses).toHaveBeenCalledWith('user-123');
    });

    it('should return empty array for new user', async () => {
      mockUserCoursesService.getUserCourses.mockResolvedValue([]);

      const result = await controller.getUserCourses({ user: { userId: 'new-user' } });

      expect(result).toEqual([]);
    });

    it('should return multiple courses', async () => {
      const multipleCourses = [
        mockCourse,
        { ...mockCourse, slug: 'java-basics', title: 'Java Basics', progress: 75 },
        { ...mockCourse, slug: 'python-ml', title: 'Python ML', progress: 25 },
      ];
      mockUserCoursesService.getUserCourses.mockResolvedValue(multipleCourses);

      const result = await controller.getUserCourses({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(3);
    });

    it('should handle service errors', async () => {
      mockUserCoursesService.getUserCourses.mockRejectedValue(new Error('Database error'));

      await expect(
        controller.getUserCourses({ user: { userId: 'user-123' } })
      ).rejects.toThrow('Database error');
    });
  });

  describe('startCourse', () => {
    it('should start a new course', async () => {
      mockUserCoursesService.startCourse.mockResolvedValue(mockCourse);

      const result = await controller.startCourse(
        { user: { userId: 'user-123' } },
        'go-basics'
      );

      expect(result).toEqual(mockCourse);
      expect(mockUserCoursesService.startCourse).toHaveBeenCalledWith('user-123', 'go-basics');
    });

    it('should throw NotFoundException for non-existent course', async () => {
      mockUserCoursesService.startCourse.mockRejectedValue(
        new NotFoundException('Course not found: nonexistent-course')
      );

      await expect(
        controller.startCourse({ user: { userId: 'user-123' } }, 'nonexistent-course')
      ).rejects.toThrow(NotFoundException);
    });

    it('should resume existing course', async () => {
      const resumedCourse = { ...mockCourse, progress: 50 };
      mockUserCoursesService.startCourse.mockResolvedValue(resumedCourse);

      const result = await controller.startCourse(
        { user: { userId: 'user-123' } },
        'go-basics'
      );

      expect(result.progress).toBe(50);
    });

    it('should handle courses with special characters in slug', async () => {
      mockUserCoursesService.startCourse.mockResolvedValue(mockCourse);

      await controller.startCourse(
        { user: { userId: 'user-123' } },
        'c_go_advanced'
      );

      expect(mockUserCoursesService.startCourse).toHaveBeenCalledWith('user-123', 'c_go_advanced');
    });
  });

  describe('updateProgress', () => {
    const mockProgressResult = {
      courseSlug: 'go-basics',
      progress: 75,
      lastAccessedAt: new Date(),
      completedAt: null,
    };

    it('should update course progress', async () => {
      mockUserCoursesService.updateProgress.mockResolvedValue(mockProgressResult);

      const result = await controller.updateProgress(
        { user: { userId: 'user-123' } },
        'go-basics',
        { progress: 75 }
      );

      expect(result).toEqual(mockProgressResult);
      expect(mockUserCoursesService.updateProgress).toHaveBeenCalledWith(
        'user-123',
        'go-basics',
        75
      );
    });

    it('should mark course as complete when progress is 100', async () => {
      const completedResult = {
        ...mockProgressResult,
        progress: 100,
        completedAt: new Date(),
      };
      mockUserCoursesService.updateProgress.mockResolvedValue(completedResult);

      const result = await controller.updateProgress(
        { user: { userId: 'user-123' } },
        'go-basics',
        { progress: 100 }
      );

      expect(result.progress).toBe(100);
      expect(result.completedAt).toBeDefined();
    });

    it('should throw NotFoundException if course not started', async () => {
      mockUserCoursesService.updateProgress.mockRejectedValue(
        new NotFoundException('User has not started course: go-basics')
      );

      await expect(
        controller.updateProgress(
          { user: { userId: 'user-123' } },
          'go-basics',
          { progress: 50 }
        )
      ).rejects.toThrow(NotFoundException);
    });

    it('should handle zero progress', async () => {
      const zeroProgressResult = { ...mockProgressResult, progress: 0 };
      mockUserCoursesService.updateProgress.mockResolvedValue(zeroProgressResult);

      const result = await controller.updateProgress(
        { user: { userId: 'user-123' } },
        'go-basics',
        { progress: 0 }
      );

      expect(result.progress).toBe(0);
    });

    it('should throw error for invalid progress value', async () => {
      mockUserCoursesService.updateProgress.mockRejectedValue(
        new Error('Progress must be between 0 and 100')
      );

      await expect(
        controller.updateProgress(
          { user: { userId: 'user-123' } },
          'go-basics',
          { progress: 150 }
        )
      ).rejects.toThrow('Progress must be between 0 and 100');
    });
  });
});
