import { Test, TestingModule } from '@nestjs/testing';
import { AiService } from './ai.service';
import { PrismaService } from '../prisma/prisma.service';
import { UsersService } from '../users/users.service';
import { ConfigService } from '@nestjs/config';
import { AccessControlService } from '../subscriptions/access-control.service';
import { ForbiddenException, ServiceUnavailableException } from '@nestjs/common';

// Mock GoogleGenAI
jest.mock('@google/genai', () => ({
  GoogleGenAI: jest.fn().mockImplementation(() => ({
    models: {
      generateContent: jest.fn().mockResolvedValue({
        text: 'This is a helpful AI response about your code.',
      }),
    },
  })),
}));

describe('AiService', () => {
  let service: AiService;
  let prisma: PrismaService;
  let usersService: UsersService;
  let accessControlService: AccessControlService;

  const mockFreeUser = {
    id: 'user-free',
    email: 'free@example.com',
    name: 'Free User',
    isPremium: false,
  };

  const mockPremiumUser = {
    id: 'user-premium',
    email: 'premium@example.com',
    name: 'Premium User',
    isPremium: true,
  };

  const mockPrismaService = {
    aiUsage: {
      upsert: jest.fn(),
      updateMany: jest.fn(),
    },
    $transaction: jest.fn(),
  };

  const mockUsersService = {
    findById: jest.fn(),
  };

  const mockConfigService = {
    get: jest.fn().mockReturnValue('test-api-key'),
  };

  const mockAccessControlService = {
    canUseAiTutor: jest.fn(),
    hasGlobalAccess: jest.fn().mockResolvedValue(true),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AiService,
        { provide: PrismaService, useValue: mockPrismaService },
        { provide: UsersService, useValue: mockUsersService },
        { provide: ConfigService, useValue: mockConfigService },
        { provide: AccessControlService, useValue: mockAccessControlService },
      ],
    }).compile();

    service = module.get<AiService>(AiService);
    prisma = module.get<PrismaService>(PrismaService);
    usersService = module.get<UsersService>(UsersService);
    accessControlService = module.get<AccessControlService>(AccessControlService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('askTutor()', () => {
    const defaultParams = {
      userId: 'user-premium',
      taskId: 'task-123',
      taskTitle: 'Hello World',
      userCode: 'func main() { println("Hello") }',
      question: 'Why is my code not working?',
      language: 'go',
      uiLanguage: 'en',
    };

    describe('access control', () => {
      it('should check subscription access before processing', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({ userId: 'user-premium', date: '2025-01-01', count: 0 }),
              updateMany: jest.fn().mockResolvedValue({ count: 1 }),
            },
          });
        });

        await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(mockAccessControlService.canUseAiTutor).toHaveBeenCalledWith(
          defaultParams.userId,
          defaultParams.taskId
        );
      });

      it('should throw ForbiddenException if no AI access', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(false);

        await expect(
          service.askTutor(
            'user-free',
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ForbiddenException);

        await expect(
          service.askTutor(
            'user-free',
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow('AI Tutor requires a subscription');
      });
    });

    describe('rate limiting', () => {
      it('should check daily usage limit', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({ userId: 'user-premium', date: '2025-01-01', count: 5 }),
              updateMany: jest.fn().mockResolvedValue({ count: 1 }),
            },
          });
        });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.remaining).toBeDefined();
      });

      it('should throw ForbiddenException when free user limit exceeded', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockFreeUser);
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({ userId: 'user-free', date: '2025-01-01', count: 3 }),
              updateMany: jest.fn().mockResolvedValue({ count: 0 }),
            },
          });
        });

        // First call returns exceeded: true
        mockPrismaService.$transaction.mockResolvedValueOnce({ exceeded: true, count: 3 });

        await expect(
          service.askTutor(
            'user-free',
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ForbiddenException);
      });

      it('should throw ForbiddenException when premium user limit exceeded', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValueOnce({ exceeded: true, count: 15 });

        await expect(
          service.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ForbiddenException);
      });

      it('should handle race conditions in usage tracking', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);

        // Simulate race condition where updateMany returns 0 (another request beat us)
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({ userId: 'user-premium', date: '2025-01-01', count: 14 }),
              updateMany: jest.fn().mockResolvedValue({ count: 0 }), // No rows updated = race condition
            },
          });
        });

        // The service should detect this and return exceeded: true
        mockPrismaService.$transaction.mockResolvedValueOnce({ exceeded: true, count: 15 });

        await expect(
          service.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ForbiddenException);
      });

      it('should increment usage count atomically', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValueOnce({ exceeded: false, count: 6, limit: 50 });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.remaining).toBe(44); // 50 - 6 = 44 (global premium limit)
      });
    });

    describe('API interaction (mocked)', () => {
      beforeEach(() => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({ exceeded: false, count: 1, limit: 50 });
      });

      it('should return AI response', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.answer).toBe('This is a helpful AI response about your code.');
      });

      it('should include remaining count in response', async () => {
        mockPrismaService.$transaction.mockResolvedValueOnce({ exceeded: false, count: 5, limit: 50 });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.remaining).toBe(45); // 50 - 5 = 45 (global premium limit)
      });

      it('should respect UI language setting (en)', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          'en'
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'English' for 'en' language
      });

      it('should respect UI language setting (ru)', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          'ru'
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'Russian' for 'ru' language
      });

      it('should respect UI language setting (uz)', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          'uz'
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'Uzbek' for 'uz' language
      });
    });

    describe('error handling', () => {
      afterEach(() => {
        // Reset GoogleGenAI mock to success state
        const { GoogleGenAI } = require('@google/genai');
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockResolvedValue({
              text: 'This is a helpful AI response about your code.',
            }),
          },
        }));
      });

      it('should throw ServiceUnavailableException when API key is missing', async () => {
        // Create a new service instance without API key
        const moduleWithoutKey: TestingModule = await Test.createTestingModule({
          providers: [
            AiService,
            { provide: PrismaService, useValue: mockPrismaService },
            { provide: UsersService, useValue: mockUsersService },
            { provide: ConfigService, useValue: { get: jest.fn().mockReturnValue(null) } },
            { provide: AccessControlService, useValue: mockAccessControlService },
          ],
        }).compile();

        const serviceWithoutKey = moduleWithoutKey.get<AiService>(AiService);

        await expect(
          serviceWithoutKey.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ServiceUnavailableException);
      });

      it('should rollback usage on API failure and throw ServiceUnavailableException', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({ exceeded: false, count: 5, limit: 50 });
        mockPrismaService.aiUsage.updateMany.mockResolvedValue({ count: 1 });

        // Mock API failure
        const { GoogleGenAI } = require('@google/genai');
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockRejectedValue(new Error('API Error')),
          },
        }));

        // Recreate service with failing API
        const moduleWithFailingApi: TestingModule = await Test.createTestingModule({
          providers: [
            AiService,
            { provide: PrismaService, useValue: mockPrismaService },
            { provide: UsersService, useValue: mockUsersService },
            { provide: ConfigService, useValue: mockConfigService },
            { provide: AccessControlService, useValue: mockAccessControlService },
          ],
        }).compile();

        const failingService = moduleWithFailingApi.get<AiService>(AiService);

        await expect(
          failingService.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow(ServiceUnavailableException);

        expect(mockPrismaService.aiUsage.updateMany).toHaveBeenCalledWith(
          expect.objectContaining({
            data: { count: { decrement: 1 } },
          })
        );
      });

      it('should handle rollback errors silently', async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({ exceeded: false, count: 5, limit: 50 });
        mockPrismaService.aiUsage.updateMany.mockRejectedValue(new Error('Rollback failed'));

        // Mock API failure
        const { GoogleGenAI } = require('@google/genai');
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockRejectedValue(new Error('API Error')),
          },
        }));

        const moduleWithFailingApi: TestingModule = await Test.createTestingModule({
          providers: [
            AiService,
            { provide: PrismaService, useValue: mockPrismaService },
            { provide: UsersService, useValue: mockUsersService },
            { provide: ConfigService, useValue: mockConfigService },
            { provide: AccessControlService, useValue: mockAccessControlService },
          ],
        }).compile();

        const failingService = moduleWithFailingApi.get<AiService>(AiService);

        // Should still throw ServiceUnavailableException even if rollback fails
        await expect(
          failingService.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage
          )
        ).rejects.toThrow('AI is currently overloaded');
      });
    });

    describe('edge cases', () => {
      beforeEach(async () => {
        // Reset GoogleGenAI mock to success state
        const { GoogleGenAI } = require('@google/genai');
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockResolvedValue({
              text: 'This is a helpful AI response about your code.',
            }),
          },
        }));

        // Recreate service with fresh mock
        const module: TestingModule = await Test.createTestingModule({
          providers: [
            AiService,
            { provide: PrismaService, useValue: mockPrismaService },
            { provide: UsersService, useValue: mockUsersService },
            { provide: ConfigService, useValue: mockConfigService },
            { provide: AccessControlService, useValue: mockAccessControlService },
          ],
        }).compile();

        service = module.get<AiService>(AiService);

        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({ exceeded: false, count: 1, limit: 50 });
      });

      it('should handle empty user code', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          '', // Empty code
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.answer).toBeDefined();
      });

      it('should handle empty question', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          '', // Empty question
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.answer).toBeDefined();
      });

      it('should handle unknown language', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          'unknown-lang',
          defaultParams.uiLanguage
        );

        expect(result.answer).toBeDefined();
      });

      it('should default to English for unknown UI language', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          'unknown' // Unknown UI language
        );

        expect(result.answer).toBeDefined();
      });

      it('should handle null language gracefully', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          null as any,
          defaultParams.uiLanguage
        );

        expect(result.answer).toBeDefined();
      });

      it('should handle null task title gracefully', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          null as any,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage
        );

        expect(result.answer).toBeDefined();
      });

      it('should use default uiLanguage when not provided', async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language
          // uiLanguage not provided, should default to 'en'
        );

        expect(result.answer).toBeDefined();
      });
    });
  });
});
