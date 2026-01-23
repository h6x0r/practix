import { Test, TestingModule } from '@nestjs/testing';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ThrottlerModule } from '@nestjs/throttler';
import { ForbiddenException, ServiceUnavailableException } from '@nestjs/common';

describe('AiController', () => {
  let controller: AiController;
  let aiService: AiService;

  const mockAiService = {
    askTutor: jest.fn(),
  };

  const mockRequest = {
    user: { userId: 'user-123' },
  };

  const mockAskAiDto = {
    taskId: 'task-456',
    taskTitle: 'Hello World',
    userCode: 'func main() { println("Hello") }',
    question: 'Why is my code not working?',
    language: 'go',
    uiLanguage: 'en',
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      imports: [
        ThrottlerModule.forRoot([{
          ttl: 60000,
          limit: 100,
        }]),
      ],
      controllers: [AiController],
      providers: [
        { provide: AiService, useValue: mockAiService },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<AiController>(AiController);
    aiService = module.get<AiService>(AiService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('askTutor()', () => {
    it('should return AI response', async () => {
      const mockResponse = {
        answer: 'Here is a hint about your code...',
        remaining: 45,
        isGlobalPremium: true,
      };
      mockAiService.askTutor.mockResolvedValue(mockResponse);

      const result = await controller.askTutor(mockRequest, mockAskAiDto);

      expect(result).toEqual(mockResponse);
      expect(mockAiService.askTutor).toHaveBeenCalledWith(
        'user-123',
        'task-456',
        'Hello World',
        'func main() { println("Hello") }',
        'Why is my code not working?',
        'go',
        'en'
      );
    });

    it('should default to "en" when uiLanguage is not provided', async () => {
      const mockResponse = { answer: 'Response', remaining: 45, isGlobalPremium: true };
      mockAiService.askTutor.mockResolvedValue(mockResponse);

      const dtoWithoutUiLanguage = { ...mockAskAiDto };
      delete (dtoWithoutUiLanguage as any).uiLanguage;

      await controller.askTutor(mockRequest, dtoWithoutUiLanguage);

      expect(mockAiService.askTutor).toHaveBeenCalledWith(
        'user-123',
        'task-456',
        'Hello World',
        'func main() { println("Hello") }',
        'Why is my code not working?',
        'go',
        'en'
      );
    });

    it('should pass through ForbiddenException from service', async () => {
      mockAiService.askTutor.mockRejectedValue(
        new ForbiddenException('Daily AI Tutor limit reached')
      );

      await expect(
        controller.askTutor(mockRequest, mockAskAiDto)
      ).rejects.toThrow(ForbiddenException);
    });

    it('should pass through ServiceUnavailableException from service', async () => {
      mockAiService.askTutor.mockRejectedValue(
        new ServiceUnavailableException('AI is currently overloaded')
      );

      await expect(
        controller.askTutor(mockRequest, mockAskAiDto)
      ).rejects.toThrow(ServiceUnavailableException);
    });

    it('should handle different UI languages', async () => {
      const mockResponse = { answer: 'Ответ на русском', remaining: 45, isGlobalPremium: true };
      mockAiService.askTutor.mockResolvedValue(mockResponse);

      const russianDto = { ...mockAskAiDto, uiLanguage: 'ru' };
      await controller.askTutor(mockRequest, russianDto);

      expect(mockAiService.askTutor).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        'ru'
      );
    });

    it('should handle different programming languages', async () => {
      const mockResponse = { answer: 'Python hint', remaining: 30, isGlobalPremium: false };
      mockAiService.askTutor.mockResolvedValue(mockResponse);

      const pythonDto = { ...mockAskAiDto, language: 'python', userCode: 'print("Hello")' };
      await controller.askTutor(mockRequest, pythonDto);

      expect(mockAiService.askTutor).toHaveBeenCalledWith(
        'user-123',
        'task-456',
        'Hello World',
        'print("Hello")',
        'Why is my code not working?',
        'python',
        'en'
      );
    });

    it('should pass user ID from request', async () => {
      const mockResponse = { answer: 'Response', remaining: 10, isGlobalPremium: false };
      mockAiService.askTutor.mockResolvedValue(mockResponse);

      const differentUser = { user: { userId: 'different-user-789' } };
      await controller.askTutor(differentUser, mockAskAiDto);

      expect(mockAiService.askTutor).toHaveBeenCalledWith(
        'different-user-789',
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String)
      );
    });
  });
});
