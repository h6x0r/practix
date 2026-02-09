import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { SnippetsService } from './snippets.service';
import { PrismaService } from '../prisma/prisma.service';

describe('SnippetsService', () => {
  let service: SnippetsService;
  let prisma: PrismaService;

  const mockSnippet = {
    id: 'uuid-123',
    shortId: 'abc12345',
    userId: 'user-123',
    title: 'Test Snippet',
    code: 'console.log("hello")',
    language: 'typescript',
    isPublic: true,
    viewCount: 0,
    expiresAt: null,
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  const mockPrisma = {
    codeSnippet: {
      create: jest.fn(),
      findUnique: jest.fn(),
      findMany: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SnippetsService,
        { provide: PrismaService, useValue: mockPrisma },
      ],
    }).compile();

    service = module.get<SnippetsService>(SnippetsService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a snippet with user', async () => {
      mockPrisma.codeSnippet.create.mockResolvedValue(mockSnippet);

      const result = await service.create(
        { code: 'test', language: 'go' },
        'user-123',
      );

      expect(result).toHaveProperty('shortId');
      expect(mockPrisma.codeSnippet.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          userId: 'user-123',
          code: 'test',
          language: 'go',
        }),
      });
    });

    it('should create anonymous snippet without user', async () => {
      mockPrisma.codeSnippet.create.mockResolvedValue({
        ...mockSnippet,
        userId: null,
      });

      const result = await service.create(
        { code: 'test', language: 'python' },
        undefined,
      );

      expect(result).toHaveProperty('shortId');
      expect(mockPrisma.codeSnippet.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          userId: undefined,
        }),
      });
    });

    it('should set expiration date if provided', async () => {
      const expiresAt = '2026-12-31T23:59:59Z';
      mockPrisma.codeSnippet.create.mockResolvedValue({
        ...mockSnippet,
        expiresAt: new Date(expiresAt),
      });

      await service.create({ code: 'test', language: 'go', expiresAt }, 'user');

      expect(mockPrisma.codeSnippet.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          expiresAt: new Date(expiresAt),
        }),
      });
    });
  });

  describe('findByShortId', () => {
    it('should return snippet and increment view count', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue(mockSnippet);
      mockPrisma.codeSnippet.update.mockResolvedValue(mockSnippet);

      const result = await service.findByShortId('abc12345');

      expect(result.code).toBe(mockSnippet.code);
      expect(result.viewCount).toBe(1);
      expect(mockPrisma.codeSnippet.update).toHaveBeenCalledWith({
        where: { id: mockSnippet.id },
        data: { viewCount: { increment: 1 } },
      });
    });

    it('should throw NotFoundException if snippet not found', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue(null);

      await expect(service.findByShortId('notexist')).rejects.toThrow(
        NotFoundException,
      );
    });

    it('should throw NotFoundException if snippet expired', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue({
        ...mockSnippet,
        expiresAt: new Date('2020-01-01'),
      });

      await expect(service.findByShortId('abc12345')).rejects.toThrow(
        NotFoundException,
      );
    });
  });

  describe('findUserSnippets', () => {
    it('should return user snippets ordered by createdAt desc', async () => {
      mockPrisma.codeSnippet.findMany.mockResolvedValue([mockSnippet]);

      const result = await service.findUserSnippets('user-123');

      expect(result).toHaveLength(1);
      expect(mockPrisma.codeSnippet.findMany).toHaveBeenCalledWith({
        where: { userId: 'user-123' },
        orderBy: { createdAt: 'desc' },
        take: 20,
        select: expect.any(Object),
      });
    });

    it('should respect limit parameter', async () => {
      mockPrisma.codeSnippet.findMany.mockResolvedValue([]);

      await service.findUserSnippets('user-123', 5);

      expect(mockPrisma.codeSnippet.findMany).toHaveBeenCalledWith(
        expect.objectContaining({ take: 5 }),
      );
    });
  });

  describe('delete', () => {
    it('should delete snippet owned by user', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue(mockSnippet);
      mockPrisma.codeSnippet.delete.mockResolvedValue(mockSnippet);

      const result = await service.delete('abc12345', 'user-123');

      expect(result).toEqual({ success: true });
      expect(mockPrisma.codeSnippet.delete).toHaveBeenCalled();
    });

    it('should throw NotFoundException if snippet not found', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue(null);

      await expect(service.delete('notexist', 'user-123')).rejects.toThrow(
        NotFoundException,
      );
    });

    it('should throw NotFoundException if not owner', async () => {
      mockPrisma.codeSnippet.findUnique.mockResolvedValue({
        ...mockSnippet,
        userId: 'other-user',
      });

      await expect(service.delete('abc12345', 'user-123')).rejects.toThrow(
        NotFoundException,
      );
    });
  });
});
