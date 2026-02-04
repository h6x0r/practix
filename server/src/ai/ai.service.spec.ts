import { Test, TestingModule } from "@nestjs/testing";
import { AiService } from "./ai.service";
import { PrismaService } from "../prisma/prisma.service";
import { UsersService } from "../users/users.service";
import { ConfigService } from "@nestjs/config";
import { AccessControlService } from "../subscriptions/access-control.service";
import { SettingsService } from "../admin/settings/settings.service";
import {
  ForbiddenException,
  ServiceUnavailableException,
} from "@nestjs/common";

// Mock GoogleGenAI
jest.mock("@google/genai", () => ({
  GoogleGenAI: jest.fn().mockImplementation(() => ({
    models: {
      generateContent: jest.fn().mockResolvedValue({
        text: "This is a helpful AI response about your code.",
      }),
    },
  })),
}));

describe("AiService", () => {
  let service: AiService;
  let prisma: PrismaService;
  let usersService: UsersService;
  let accessControlService: AccessControlService;

  const mockFreeUser = {
    id: "user-free",
    email: "free@example.com",
    name: "Free User",
    isPremium: false,
  };

  const mockPremiumUser = {
    id: "user-premium",
    email: "premium@example.com",
    name: "Premium User",
    isPremium: true,
  };

  const mockPrismaService = {
    aiUsage: {
      upsert: jest.fn(),
      updateMany: jest.fn(),
      findUnique: jest.fn(),
    },
    course: {
      findUnique: jest.fn(),
    },
    $transaction: jest.fn(),
  };

  const mockUsersService = {
    findById: jest.fn(),
  };

  const mockConfigService = {
    get: jest.fn().mockReturnValue("test-api-key"),
  };

  const mockAccessControlService = {
    canUseAiTutor: jest.fn(),
    hasGlobalAccess: jest.fn().mockResolvedValue(true),
    hasCourseAccess: jest.fn().mockResolvedValue(false),
  };

  const mockSettingsService = {
    getAiSettings: jest.fn().mockResolvedValue({
      enabled: true,
      limits: {
        free: 5,
        course: 30,
        premium: 100,
        promptEngineering: 100,
      },
    }),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AiService,
        { provide: PrismaService, useValue: mockPrismaService },
        { provide: UsersService, useValue: mockUsersService },
        { provide: ConfigService, useValue: mockConfigService },
        { provide: AccessControlService, useValue: mockAccessControlService },
        { provide: SettingsService, useValue: mockSettingsService },
      ],
    }).compile();

    service = module.get<AiService>(AiService);
    prisma = module.get<PrismaService>(PrismaService);
    usersService = module.get<UsersService>(UsersService);
    accessControlService =
      module.get<AccessControlService>(AccessControlService);

    jest.clearAllMocks();
  });

  it("should be defined", () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // getUserAiLimit() - Determine user's AI limit tier
  // ============================================
  describe("getUserAiLimit()", () => {
    beforeEach(() => {
      mockPrismaService.course.findUnique.mockResolvedValue(null);
    });

    it("should return global tier with 100 limit for global premium users", async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);

      const result = await service.getUserAiLimit("user-premium", "task-123");

      expect(result.tier).toBe("global");
      expect(result.limit).toBe(100);
    });

    it("should return prompt_engineering tier with 100 limit for PE course subscribers", async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
      mockPrismaService.course.findUnique.mockResolvedValue({
        id: "pe-course-id",
        slug: "prompt-engineering",
      });
      mockAccessControlService.hasCourseAccess.mockResolvedValue(true);

      const result = await service.getUserAiLimit("user-pe", "task-123");

      expect(result.tier).toBe("prompt_engineering");
      expect(result.limit).toBe(100);
    });

    it("should return course tier with 30 limit for course subscribers", async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
      mockAccessControlService.hasCourseAccess.mockResolvedValue(false);
      mockAccessControlService.canUseAiTutor.mockResolvedValue(true);

      const result = await service.getUserAiLimit("user-course", "task-123");

      expect(result.tier).toBe("course");
      expect(result.limit).toBe(30);
    });

    it("should return free tier with 5 limit for users without subscriptions", async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
      mockAccessControlService.hasCourseAccess.mockResolvedValue(false);
      mockAccessControlService.canUseAiTutor.mockResolvedValue(false);

      const result = await service.getUserAiLimit("user-free", "task-123");

      expect(result.tier).toBe("free");
      expect(result.limit).toBe(5);
    });

    it("should return free tier when taskId is not provided", async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
      mockAccessControlService.hasCourseAccess.mockResolvedValue(false);

      const result = await service.getUserAiLimit("user-free");

      expect(result.tier).toBe("free");
      expect(result.limit).toBe(5);
    });
  });

  // ============================================
  // getAiLimitInfo() - Get current usage info
  // ============================================
  describe("getAiLimitInfo()", () => {
    beforeEach(() => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);
      mockPrismaService.course.findUnique.mockResolvedValue(null);
    });

    it("should return full limit info with usage data", async () => {
      mockPrismaService.aiUsage.findUnique.mockResolvedValue({
        userId: "user-premium",
        date: "2025-01-01",
        count: 25,
      });

      const result = await service.getAiLimitInfo("user-premium", "task-123");

      expect(result.tier).toBe("global");
      expect(result.limit).toBe(100);
      expect(result.used).toBe(25);
      expect(result.remaining).toBe(75);
    });

    it("should return 0 used when no usage record exists", async () => {
      mockPrismaService.aiUsage.findUnique.mockResolvedValue(null);

      const result = await service.getAiLimitInfo("user-premium", "task-123");

      expect(result.used).toBe(0);
      expect(result.remaining).toBe(100);
    });

    it("should return 0 remaining when limit is exceeded", async () => {
      mockPrismaService.aiUsage.findUnique.mockResolvedValue({
        userId: "user-premium",
        date: "2025-01-01",
        count: 150,
      });

      const result = await service.getAiLimitInfo("user-premium", "task-123");

      expect(result.remaining).toBe(0);
    });
  });

  describe("askTutor()", () => {
    const defaultParams = {
      userId: "user-premium",
      taskId: "task-123",
      taskTitle: "Hello World",
      userCode: 'func main() { println("Hello") }',
      question: "Why is my code not working?",
      language: "go",
      uiLanguage: "en",
    };

    describe("access control", () => {
      it("should check global access first when determining limit tier", async () => {
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);
        mockPrismaService.course.findUnique.mockResolvedValue(null);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 1,
          limit: 100,
          tier: "global",
        });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(mockAccessControlService.hasGlobalAccess).toHaveBeenCalledWith(
          defaultParams.userId,
        );
        expect(result.tier).toBe("global");
        expect(result.limit).toBe(100);
      });

      it("should allow free tier users with 5 requests/day limit", async () => {
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
        mockAccessControlService.canUseAiTutor.mockResolvedValue(false);
        mockPrismaService.course.findUnique.mockResolvedValue(null);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 1,
          limit: 5,
          tier: "free",
        });

        const result = await service.askTutor(
          "user-free",
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.tier).toBe("free");
        expect(result.limit).toBe(5);
        expect(result.remaining).toBe(4);
      });
    });

    describe("rate limiting", () => {
      it("should check daily usage limit", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({
                userId: "user-premium",
                date: "2025-01-01",
                count: 5,
              }),
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
          defaultParams.uiLanguage,
        );

        expect(result.remaining).toBeDefined();
      });

      it("should throw ForbiddenException when free user limit exceeded", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockFreeUser);
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({
                userId: "user-free",
                date: "2025-01-01",
                count: 3,
              }),
              updateMany: jest.fn().mockResolvedValue({ count: 0 }),
            },
          });
        });

        // First call returns exceeded: true
        mockPrismaService.$transaction.mockResolvedValueOnce({
          exceeded: true,
          count: 3,
        });

        await expect(
          service.askTutor(
            "user-free",
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow(ForbiddenException);
      });

      it("should throw ForbiddenException when premium user limit exceeded", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValueOnce({
          exceeded: true,
          count: 15,
        });

        await expect(
          service.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow(ForbiddenException);
      });

      it("should handle race conditions in usage tracking", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);

        // Simulate race condition where updateMany returns 0 (another request beat us)
        mockPrismaService.$transaction.mockImplementation(async (fn) => {
          return fn({
            aiUsage: {
              upsert: jest.fn().mockResolvedValue({
                userId: "user-premium",
                date: "2025-01-01",
                count: 14,
              }),
              updateMany: jest.fn().mockResolvedValue({ count: 0 }), // No rows updated = race condition
            },
          });
        });

        // The service should detect this and return exceeded: true
        mockPrismaService.$transaction.mockResolvedValueOnce({
          exceeded: true,
          count: 15,
        });

        await expect(
          service.askTutor(
            defaultParams.userId,
            defaultParams.taskId,
            defaultParams.taskTitle,
            defaultParams.userCode,
            defaultParams.question,
            defaultParams.language,
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow(ForbiddenException);
      });

      it("should increment usage count atomically", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValueOnce({
          exceeded: false,
          count: 6,
          limit: 50,
        });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.remaining).toBe(44); // 50 - 6 = 44 (global premium limit)
      });
    });

    describe("API interaction (mocked)", () => {
      beforeEach(() => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 1,
          limit: 50,
        });
      });

      it("should return AI response", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBe(
          "This is a helpful AI response about your code.",
        );
      });

      it("should include remaining count in response", async () => {
        mockPrismaService.$transaction.mockResolvedValueOnce({
          exceeded: false,
          count: 5,
          limit: 50,
        });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.remaining).toBe(45); // 50 - 5 = 45 (global premium limit)
      });

      it("should respect UI language setting (en)", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          "en",
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'English' for 'en' language
      });

      it("should respect UI language setting (ru)", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          "ru",
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'Russian' for 'ru' language
      });

      it("should respect UI language setting (uz)", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          "uz",
        );

        expect(result.answer).toBeDefined();
        // The prompt template should include 'Uzbek' for 'uz' language
      });
    });

    describe("error handling", () => {
      afterEach(() => {
        // Reset GoogleGenAI mock to success state
        const { GoogleGenAI } = require("@google/genai");
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockResolvedValue({
              text: "This is a helpful AI response about your code.",
            }),
          },
        }));
      });

      it("should throw ServiceUnavailableException when API key is missing", async () => {
        // Create a new service instance without API key
        const moduleWithoutKey: TestingModule = await Test.createTestingModule({
          providers: [
            AiService,
            { provide: PrismaService, useValue: mockPrismaService },
            { provide: UsersService, useValue: mockUsersService },
            {
              provide: ConfigService,
              useValue: { get: jest.fn().mockReturnValue(null) },
            },
            {
              provide: AccessControlService,
              useValue: mockAccessControlService,
            },
            { provide: SettingsService, useValue: mockSettingsService },
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
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow(ServiceUnavailableException);
      });

      it("should rollback usage on API failure and throw ServiceUnavailableException", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 5,
          limit: 50,
        });
        mockPrismaService.aiUsage.updateMany.mockResolvedValue({ count: 1 });

        // Mock API failure
        const { GoogleGenAI } = require("@google/genai");
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest
              .fn()
              .mockRejectedValue(new Error("API Error")),
          },
        }));

        // Recreate service with failing API
        const moduleWithFailingApi: TestingModule =
          await Test.createTestingModule({
            providers: [
              AiService,
              { provide: PrismaService, useValue: mockPrismaService },
              { provide: UsersService, useValue: mockUsersService },
              { provide: ConfigService, useValue: mockConfigService },
              {
                provide: AccessControlService,
                useValue: mockAccessControlService,
              },
              { provide: SettingsService, useValue: mockSettingsService },
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
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow(ServiceUnavailableException);

        expect(mockPrismaService.aiUsage.updateMany).toHaveBeenCalledWith(
          expect.objectContaining({
            data: { count: { decrement: 1 } },
          }),
        );
      });

      it("should handle rollback errors silently", async () => {
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 5,
          limit: 50,
        });
        mockPrismaService.aiUsage.updateMany.mockRejectedValue(
          new Error("Rollback failed"),
        );

        // Mock API failure
        const { GoogleGenAI } = require("@google/genai");
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest
              .fn()
              .mockRejectedValue(new Error("API Error")),
          },
        }));

        const moduleWithFailingApi: TestingModule =
          await Test.createTestingModule({
            providers: [
              AiService,
              { provide: PrismaService, useValue: mockPrismaService },
              { provide: UsersService, useValue: mockUsersService },
              { provide: ConfigService, useValue: mockConfigService },
              {
                provide: AccessControlService,
                useValue: mockAccessControlService,
              },
              { provide: SettingsService, useValue: mockSettingsService },
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
            defaultParams.uiLanguage,
          ),
        ).rejects.toThrow("AI is currently overloaded");
      });
    });

    describe("edge cases", () => {
      beforeEach(async () => {
        // Reset GoogleGenAI mock to success state
        const { GoogleGenAI } = require("@google/genai");
        GoogleGenAI.mockImplementation(() => ({
          models: {
            generateContent: jest.fn().mockResolvedValue({
              text: "This is a helpful AI response about your code.",
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
            {
              provide: AccessControlService,
              useValue: mockAccessControlService,
            },
            { provide: SettingsService, useValue: mockSettingsService },
          ],
        }).compile();

        service = module.get<AiService>(AiService);

        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockUsersService.findById.mockResolvedValue(mockPremiumUser);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 1,
          limit: 50,
        });
      });

      it("should handle empty user code", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          "", // Empty code
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBeDefined();
      });

      it("should handle empty question", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          "", // Empty question
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBeDefined();
      });

      it("should handle unknown language", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          "unknown-lang",
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBeDefined();
      });

      it("should default to English for unknown UI language", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          "unknown", // Unknown UI language
        );

        expect(result.answer).toBeDefined();
      });

      it("should handle null language gracefully", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          null as any,
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBeDefined();
      });

      it("should handle null task title gracefully", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          null as any,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.answer).toBeDefined();
      });

      it("should use default uiLanguage when not provided", async () => {
        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          // uiLanguage not provided, should default to 'en'
        );

        expect(result.answer).toBeDefined();
      });

      it("should use course subscription limit when no global access", async () => {
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
        mockAccessControlService.canUseAiTutor.mockResolvedValue(true);
        mockPrismaService.course.findUnique.mockResolvedValue(null);
        mockPrismaService.$transaction.mockResolvedValue({
          exceeded: false,
          count: 5,
          limit: 30,
          tier: "course",
        });

        const result = await service.askTutor(
          defaultParams.userId,
          defaultParams.taskId,
          defaultParams.taskTitle,
          defaultParams.userCode,
          defaultParams.question,
          defaultParams.language,
          defaultParams.uiLanguage,
        );

        expect(result.remaining).toBe(25); // 30 - 5 = 25 (course subscription limit)
        expect(result.tier).toBe("course");
      });
    });
  });

  // ============================================
  // evaluatePrompt() - Prompt Engineering Evaluation
  // ============================================
  describe("evaluatePrompt()", () => {
    const mockPromptConfig = {
      testScenarios: [
        {
          input: "What is machine learning?",
          expectedCriteria: [
            "Explains supervised learning",
            "Mentions examples",
          ],
          rubric: "Score based on clarity and completeness",
        },
        {
          input: "Explain neural networks",
          expectedCriteria: ["Explains layers", "Mentions weights"],
        },
      ],
      judgePrompt:
        "Evaluate this output: {{OUTPUT}}\n\nCriteria:\n- {{CRITERIA}}\n\nRubric: {{RUBRIC}}",
      passingScore: 7,
    };

    beforeEach(async () => {
      // Reset GoogleGenAI mock for evaluatePrompt tests
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "ML is about learning patterns..." }) // First scenario execution
            .mockResolvedValueOnce({
              text: '{"score": 8, "feedback": "Good explanation"}',
            }) // First judge
            .mockResolvedValueOnce({ text: "Neural networks have layers..." }) // Second scenario execution
            .mockResolvedValueOnce({
              text: '{"score": 7, "feedback": "Covers basics"}',
            }), // Second judge
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
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      service = module.get<AiService>(AiService);
    });

    it("should throw ServiceUnavailableException when API key is missing", async () => {
      const moduleWithoutKey: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          {
            provide: ConfigService,
            useValue: { get: jest.fn().mockReturnValue(null) },
          },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const serviceWithoutKey = moduleWithoutKey.get<AiService>(AiService);

      await expect(
        serviceWithoutKey.evaluatePrompt(
          "user-123",
          "test prompt",
          mockPromptConfig,
        ),
      ).rejects.toThrow(ServiceUnavailableException);
    });

    it("should evaluate all test scenarios", async () => {
      const result = await service.evaluatePrompt(
        "user-123",
        "You are an AI that explains {{INPUT}}. Be concise.",
        mockPromptConfig,
      );

      expect(result.scenarioResults).toHaveLength(2);
      expect(result.scenarioResults[0].scenarioIndex).toBe(0);
      expect(result.scenarioResults[1].scenarioIndex).toBe(1);
    });

    it("should return passed=true when average score meets passing threshold", async () => {
      const result = await service.evaluatePrompt(
        "user-123",
        "Explain {{INPUT}}",
        mockPromptConfig,
      );

      expect(result.score).toBeGreaterThanOrEqual(7);
      expect(result.passed).toBe(true);
    });

    it("should return passed=false when average score below threshold", async () => {
      // Mock low scores
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Short answer" })
            .mockResolvedValueOnce({
              text: '{"score": 3, "feedback": "Too brief"}',
            })
            .mockResolvedValueOnce({ text: "Another short answer" })
            .mockResolvedValueOnce({
              text: '{"score": 4, "feedback": "Missing details"}',
            }),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const failingService = module.get<AiService>(AiService);

      const result = await failingService.evaluatePrompt(
        "user-123",
        "Explain {{INPUT}}",
        mockPromptConfig,
      );

      expect(result.score).toBeLessThan(7);
      expect(result.passed).toBe(false);
      expect(result.summary).toContain("Not passed");
    });

    it("should handle JSON parsing errors from judge", async () => {
      // Mock invalid JSON response from judge
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Valid output" })
            .mockResolvedValueOnce({ text: "This is not valid JSON" }),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const serviceWithBadJson = module.get<AiService>(AiService);

      const result = await serviceWithBadJson.evaluatePrompt(
        "user-123",
        "Test prompt {{INPUT}}",
        {
          ...mockPromptConfig,
          testScenarios: [mockPromptConfig.testScenarios[0]],
        },
      );

      expect(result.scenarioResults[0].score).toBe(0);
      expect(result.scenarioResults[0].feedback).toBe(
        "Failed to parse evaluation",
      );
    });

    it("should handle API errors during scenario execution", async () => {
      // Mock API error
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockRejectedValue(new Error("API timeout")),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const serviceWithError = module.get<AiService>(AiService);

      const result = await serviceWithError.evaluatePrompt(
        "user-123",
        "Test prompt {{INPUT}}",
        {
          ...mockPromptConfig,
          testScenarios: [mockPromptConfig.testScenarios[0]],
        },
      );

      expect(result.scenarioResults[0].passed).toBe(false);
      expect(result.scenarioResults[0].score).toBe(0);
      expect(result.scenarioResults[0].feedback).toContain("Execution failed");
    });

    it("should handle empty judge response", async () => {
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Valid output" })
            .mockResolvedValueOnce({ text: "" }), // Empty response
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: SettingsService, useValue: mockSettingsService },
          { provide: AccessControlService, useValue: mockAccessControlService },
        ],
      }).compile();

      const serviceWithEmptyResponse = module.get<AiService>(AiService);

      const result = await serviceWithEmptyResponse.evaluatePrompt(
        "user-123",
        "Test prompt {{INPUT}}",
        {
          ...mockPromptConfig,
          testScenarios: [mockPromptConfig.testScenarios[0]],
        },
      );

      // Should use default fallback
      expect(result.scenarioResults[0].score).toBe(0);
    });

    it("should truncate long input/output in results", async () => {
      const longInput = "X".repeat(300);
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Y".repeat(600) }) // Long output
            .mockResolvedValueOnce({
              text: '{"score": 8, "feedback": "Good"}',
            }),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const serviceWithLongContent = module.get<AiService>(AiService);

      const result = await serviceWithLongContent.evaluatePrompt(
        "user-123",
        "Test {{INPUT}}",
        {
          ...mockPromptConfig,
          testScenarios: [
            {
              input: longInput,
              expectedCriteria: ["Test"],
            },
          ],
        },
      );

      expect(result.scenarioResults[0].input.length).toBeLessThanOrEqual(203); // 200 + '...'
      expect(result.scenarioResults[0].output.length).toBeLessThanOrEqual(503); // 500 + '...'
    });

    it("should handle empty test scenarios array", async () => {
      const result = await service.evaluatePrompt("user-123", "Test prompt", {
        ...mockPromptConfig,
        testScenarios: [],
      });

      expect(result.scenarioResults).toHaveLength(0);
      expect(result.score).toBe(0);
      expect(result.passed).toBe(false);
    });

    it("should replace {{INPUT}} placeholder in prompt", async () => {
      const { GoogleGenAI } = require("@google/genai");
      const mockGenerateContent = jest
        .fn()
        .mockResolvedValueOnce({ text: "Response about ML" })
        .mockResolvedValueOnce({ text: '{"score": 8, "feedback": "Good"}' });

      GoogleGenAI.mockImplementation(() => ({
        models: { generateContent: mockGenerateContent },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      await module
        .get<AiService>(AiService)
        .evaluatePrompt("user-123", "Explain this: {{INPUT}}", {
          ...mockPromptConfig,
          testScenarios: [
            {
              input: "Machine Learning",
              expectedCriteria: ["Explains ML"],
            },
          ],
        });

      // The first call should have the input substituted
      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          contents: "Explain this: Machine Learning",
        }),
      );
    });

    it("should extract JSON from markdown-wrapped response", async () => {
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Valid output" })
            .mockResolvedValueOnce({
              text: '```json\n{"score": 9, "feedback": "Excellent"}\n```',
            }),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const result = await module
        .get<AiService>(AiService)
        .evaluatePrompt("user-123", "Test {{INPUT}}", {
          ...mockPromptConfig,
          testScenarios: [mockPromptConfig.testScenarios[0]],
        });

      expect(result.scenarioResults[0].score).toBe(9);
      expect(result.scenarioResults[0].feedback).toBe("Excellent");
    });

    it("should clamp score to 0-10 range", async () => {
      const { GoogleGenAI } = require("@google/genai");
      GoogleGenAI.mockImplementation(() => ({
        models: {
          generateContent: jest
            .fn()
            .mockResolvedValueOnce({ text: "Output" })
            .mockResolvedValueOnce({
              text: '{"score": 15, "feedback": "Over max"}',
            }),
        },
      }));

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AiService,
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: UsersService, useValue: mockUsersService },
          { provide: ConfigService, useValue: mockConfigService },
          { provide: AccessControlService, useValue: mockAccessControlService },
          { provide: SettingsService, useValue: mockSettingsService },
        ],
      }).compile();

      const result = await module
        .get<AiService>(AiService)
        .evaluatePrompt("user-123", "Test {{INPUT}}", {
          ...mockPromptConfig,
          testScenarios: [mockPromptConfig.testScenarios[0]],
        });

      expect(result.scenarioResults[0].score).toBe(10);
    });
  });

  // ============================================
  // getPromptEvaluationCost()
  // ============================================
  describe("getPromptEvaluationCost()", () => {
    it("should return 2 calls per scenario", () => {
      const cost = service.getPromptEvaluationCost(3);
      expect(cost).toBe(6); // 3 scenarios * 2 calls each
    });

    it("should return 0 for 0 scenarios", () => {
      const cost = service.getPromptEvaluationCost(0);
      expect(cost).toBe(0);
    });

    it("should calculate correctly for single scenario", () => {
      const cost = service.getPromptEvaluationCost(1);
      expect(cost).toBe(2);
    });
  });
});
