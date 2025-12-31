import { Test, TestingModule } from '@nestjs/testing';
import { TasksService } from './tasks.service';
import { PrismaService } from '../prisma/prisma.service';
import { NotFoundException } from '@nestjs/common';

describe('TasksService', () => {
  let service: TasksService;
  let prisma: PrismaService;

  const mockPrisma = {
    task: {
      findMany: jest.fn(),
      findUnique: jest.fn(),
    },
    submission: {
      findMany: jest.fn(),
      findFirst: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        TasksService,
        {
          provide: PrismaService,
          useValue: mockPrisma,
        },
      ],
    }).compile();

    service = module.get<TasksService>(TasksService);
    prisma = module.get<PrismaService>(PrismaService);

    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('findAll', () => {
    it('should return tasks with pending status when no userId provided', async () => {
      const mockTasks = [
        { id: '1', slug: 'task-1', title: 'Task 1', difficulty: 'easy', tags: ['go'], isPremium: false },
        { id: '2', slug: 'task-2', title: 'Task 2', difficulty: 'medium', tags: ['java'], isPremium: true },
      ];

      mockPrisma.task.findMany.mockResolvedValue(mockTasks);

      const result = await service.findAll();

      expect(result).toHaveLength(2);
      expect(result[0].status).toBe('pending');
      expect(result[1].status).toBe('pending');
    });

    it('should return tasks with completion status when userId provided', async () => {
      const mockTasks = [
        { id: '1', slug: 'task-1', title: 'Task 1', difficulty: 'easy', tags: ['go'], isPremium: false },
        { id: '2', slug: 'task-2', title: 'Task 2', difficulty: 'medium', tags: ['java'], isPremium: true },
      ];

      const mockSubmissions = [{ taskId: '1' }];

      mockPrisma.task.findMany.mockResolvedValue(mockTasks);
      mockPrisma.submission.findMany.mockResolvedValue(mockSubmissions);

      const result = await service.findAll('user-123');

      expect(result).toHaveLength(2);
      expect(result.find((t) => t.id === '1')?.status).toBe('completed');
      expect(result.find((t) => t.id === '2')?.status).toBe('pending');
    });
  });

  describe('findOne', () => {
    it('should return a task by slug', async () => {
      const mockTask = {
        id: '1',
        slug: 'hello-world',
        title: 'Hello World',
        difficulty: 'easy',
        tags: ['go'],
        isPremium: false,
        description: 'Write a hello world program',
        initialCode: 'package main',
        solutionCode: 'package main...',
      };

      mockPrisma.task.findUnique.mockResolvedValue(mockTask);

      const result = await service.findOne('hello-world');

      expect(result.slug).toBe('hello-world');
      expect(result.status).toBe('pending');
    });

    it('should throw NotFoundException if task not found', async () => {
      mockPrisma.task.findUnique.mockResolvedValue(null);

      await expect(service.findOne('non-existent')).rejects.toThrow(NotFoundException);
    });

    it('should mark task as completed if user has passed submission', async () => {
      const mockTask = {
        id: '1',
        slug: 'hello-world',
        title: 'Hello World',
        difficulty: 'easy',
      };

      mockPrisma.task.findUnique.mockResolvedValue(mockTask);
      mockPrisma.submission.findFirst.mockResolvedValue({ id: 'sub-1', status: 'passed' });

      const result = await service.findOne('hello-world', 'user-123');

      expect(result.status).toBe('completed');
    });
  });
});
