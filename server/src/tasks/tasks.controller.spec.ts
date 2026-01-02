import { Test, TestingModule } from '@nestjs/testing';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { OptionalJwtAuthGuard } from '../auth/guards/optional-jwt.guard';
import { NotFoundException } from '@nestjs/common';

describe('TasksController', () => {
  let controller: TasksController;
  let tasksService: TasksService;

  const mockTasksService = {
    findAll: jest.fn(),
    findOne: jest.fn(),
  };

  const mockTasks = [
    {
      id: 'task-1',
      slug: 'hello-world',
      title: 'Hello World',
      difficulty: 'easy',
      tags: ['beginner', 'syntax'],
      isPremium: false,
      estimatedTime: '10m',
      description: 'Write your first program',
      status: 'pending',
    },
    {
      id: 'task-2',
      slug: 'variables',
      title: 'Variables',
      difficulty: 'easy',
      tags: ['beginner', 'variables'],
      isPremium: false,
      estimatedTime: '15m',
      description: 'Learn about variables',
      status: 'pending',
    },
    {
      id: 'task-3',
      slug: 'advanced-concurrency',
      title: 'Advanced Concurrency',
      difficulty: 'hard',
      tags: ['advanced', 'concurrency'],
      isPremium: true,
      estimatedTime: '45m',
      description: 'Master concurrency patterns',
      status: 'pending',
    },
  ];

  const mockSingleTask = {
    id: 'task-1',
    slug: 'hello-world',
    title: 'Hello World',
    difficulty: 'easy',
    tags: ['beginner', 'syntax'],
    isPremium: false,
    estimatedTime: '10m',
    description: 'Write your first Go program',
    initialCode: 'package main\n\nfunc main() {\n\t// Your code here\n}',
    solution: 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Hello, World!")\n}',
    testCode: 'func TestHelloWorld(t *testing.T) { /* tests */ }',
    translations: {},
    status: 'pending',
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TasksController],
      providers: [
        {
          provide: TasksService,
          useValue: mockTasksService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(OptionalJwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<TasksController>(TasksController);
    tasksService = module.get<TasksService>(TasksService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('findAll', () => {
    it('should return all tasks for guest user', async () => {
      mockTasksService.findAll.mockResolvedValue(mockTasks);

      const result = await controller.findAll({ user: null });

      expect(result).toEqual(mockTasks);
      expect(mockTasksService.findAll).toHaveBeenCalledWith(undefined);
    });

    it('should return all tasks with completion status for authenticated user', async () => {
      const tasksWithStatus = mockTasks.map((t, i) => ({
        ...t,
        status: i === 0 ? 'completed' : 'pending',
      }));
      mockTasksService.findAll.mockResolvedValue(tasksWithStatus);

      const result = await controller.findAll({ user: { userId: 'user-123' } });

      expect(result[0].status).toBe('completed');
      expect(result[1].status).toBe('pending');
      expect(mockTasksService.findAll).toHaveBeenCalledWith('user-123');
    });

    it('should handle empty tasks list', async () => {
      mockTasksService.findAll.mockResolvedValue([]);

      const result = await controller.findAll({ user: null });

      expect(result).toEqual([]);
    });

    it('should handle service errors', async () => {
      mockTasksService.findAll.mockRejectedValue(new Error('Database error'));

      await expect(controller.findAll({ user: null })).rejects.toThrow('Database error');
    });

    it('should handle undefined user in request', async () => {
      mockTasksService.findAll.mockResolvedValue(mockTasks);

      const result = await controller.findAll({ user: undefined });

      expect(mockTasksService.findAll).toHaveBeenCalledWith(undefined);
      expect(result).toBeDefined();
    });

    it('should filter premium tasks visibility', async () => {
      const tasksWithPremium = mockTasks.filter(t => t.isPremium);
      mockTasksService.findAll.mockResolvedValue(mockTasks);

      const result = await controller.findAll({ user: null });

      expect(result.some(t => t.isPremium)).toBe(true);
    });
  });

  describe('findOne', () => {
    it('should return a task by slug for guest', async () => {
      mockTasksService.findOne.mockResolvedValue(mockSingleTask);

      const result = await controller.findOne('hello-world', { user: null });

      expect(result).toEqual(mockSingleTask);
      expect(mockTasksService.findOne).toHaveBeenCalledWith('hello-world', undefined);
    });

    it('should return a task with completed status for authenticated user', async () => {
      const completedTask = { ...mockSingleTask, status: 'completed' };
      mockTasksService.findOne.mockResolvedValue(completedTask);

      const result = await controller.findOne('hello-world', { user: { userId: 'user-123' } });

      expect(result.status).toBe('completed');
      expect(mockTasksService.findOne).toHaveBeenCalledWith('hello-world', 'user-123');
    });

    it('should throw NotFoundException for non-existent task', async () => {
      mockTasksService.findOne.mockRejectedValue(
        new NotFoundException('Task with slug non-existent not found')
      );

      await expect(
        controller.findOne('non-existent', { user: null })
      ).rejects.toThrow(NotFoundException);
    });

    it('should handle task slug with underscores', async () => {
      mockTasksService.findOne.mockResolvedValue(mockSingleTask);

      await controller.findOne('hello_world_task', { user: null });

      expect(mockTasksService.findOne).toHaveBeenCalledWith('hello_world_task', undefined);
    });

    it('should handle task slug with numbers', async () => {
      mockTasksService.findOne.mockResolvedValue(mockSingleTask);

      await controller.findOne('task-01-intro', { user: null });

      expect(mockTasksService.findOne).toHaveBeenCalledWith('task-01-intro', undefined);
    });

    it('should return task with all fields for authenticated user', async () => {
      const fullTask = {
        ...mockSingleTask,
        initialCode: 'package main...',
        solutionCode: 'package main...',
        testCode: 'func TestMain...',
      };
      mockTasksService.findOne.mockResolvedValue(fullTask);

      const result = await controller.findOne('hello-world', { user: { userId: 'user-123' } });

      expect(result.initialCode).toBeDefined();
      expect(result.solutionCode).toBeDefined();
      expect(result.testCode).toBeDefined();
    });

    it('should handle premium task access', async () => {
      const premiumTask = {
        ...mockSingleTask,
        isPremium: true,
      };
      mockTasksService.findOne.mockResolvedValue(premiumTask);

      const result = await controller.findOne('advanced-concurrency', { user: { userId: 'user-123' } });

      expect(result.isPremium).toBe(true);
    });
  });

  describe('edge cases', () => {
    it('should handle very long slug', async () => {
      const longSlug = 'this-is-a-very-long-task-slug-that-describes-something-in-detail';
      mockTasksService.findOne.mockResolvedValue(mockSingleTask);

      await controller.findOne(longSlug, { user: null });

      expect(mockTasksService.findOne).toHaveBeenCalledWith(longSlug, undefined);
    });

    it('should handle request with user but missing userId', async () => {
      mockTasksService.findAll.mockResolvedValue(mockTasks);

      const result = await controller.findAll({ user: {} });

      expect(mockTasksService.findAll).toHaveBeenCalledWith(undefined);
      expect(result).toBeDefined();
    });

    it('should handle large number of tasks', async () => {
      const manyTasks = Array.from({ length: 500 }, (_, i) => ({
        id: `task-${i}`,
        slug: `task-${i}`,
        title: `Task ${i}`,
        difficulty: 'medium',
        tags: ['test'],
        isPremium: false,
        estimatedTime: '10m',
        description: `Description ${i}`,
        status: 'pending',
      }));
      mockTasksService.findAll.mockResolvedValue(manyTasks);

      const result = await controller.findAll({ user: null });

      expect(result).toHaveLength(500);
    });

    it('should handle tasks with different difficulties', async () => {
      const variedTasks = [
        { ...mockTasks[0], difficulty: 'easy' },
        { ...mockTasks[1], difficulty: 'medium' },
        { ...mockTasks[2], difficulty: 'hard' },
      ];
      mockTasksService.findAll.mockResolvedValue(variedTasks);

      const result = await controller.findAll({ user: null });

      expect(result.map(t => t.difficulty)).toEqual(['easy', 'medium', 'hard']);
    });

    it('should handle tasks with empty tags', async () => {
      const taskWithNoTags = { ...mockSingleTask, tags: [] };
      mockTasksService.findOne.mockResolvedValue(taskWithNoTags);

      const result = await controller.findOne('no-tags', { user: null });

      expect(result.tags).toEqual([]);
    });

    it('should handle task with unicode in title', async () => {
      const unicodeTask = {
        ...mockSingleTask,
        title: 'Привет мир! 你好世界!',
        description: 'Description with unicode: émoji 🚀'
      };
      mockTasksService.findOne.mockResolvedValue(unicodeTask);

      const result = await controller.findOne('unicode-task', { user: null });

      expect(result.title).toBe('Привет мир! 你好世界!');
    });
  });
});
