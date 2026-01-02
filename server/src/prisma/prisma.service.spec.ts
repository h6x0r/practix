import { Test, TestingModule } from '@nestjs/testing';
import { PrismaService } from './prisma.service';

describe('PrismaService', () => {
  let service: PrismaService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [PrismaService],
    }).compile();

    service = module.get<PrismaService>(PrismaService);

    // Mock the $connect and $disconnect methods
    jest.spyOn(service, '$connect').mockResolvedValue(undefined);
    jest.spyOn(service, '$disconnect').mockResolvedValue(undefined);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('onModuleInit', () => {
    it('should connect to database on module init', async () => {
      await service.onModuleInit();

      expect(service.$connect).toHaveBeenCalled();
    });

    it('should handle connection errors', async () => {
      const connectError = new Error('Connection failed');
      jest.spyOn(service, '$connect').mockRejectedValue(connectError);

      await expect(service.onModuleInit()).rejects.toThrow('Connection failed');
    });
  });

  describe('onModuleDestroy', () => {
    it('should disconnect from database on module destroy', async () => {
      await service.onModuleDestroy();

      expect(service.$disconnect).toHaveBeenCalled();
    });

    it('should handle disconnect errors', async () => {
      const disconnectError = new Error('Disconnect failed');
      jest.spyOn(service, '$disconnect').mockRejectedValue(disconnectError);

      await expect(service.onModuleDestroy()).rejects.toThrow('Disconnect failed');
    });
  });

  describe('lifecycle', () => {
    it('should handle full lifecycle (init -> destroy)', async () => {
      await service.onModuleInit();
      expect(service.$connect).toHaveBeenCalledTimes(1);

      await service.onModuleDestroy();
      expect(service.$disconnect).toHaveBeenCalledTimes(1);
    });

    it('should be able to reconnect after disconnect', async () => {
      await service.onModuleInit();
      await service.onModuleDestroy();
      await service.onModuleInit();

      expect(service.$connect).toHaveBeenCalledTimes(2);
    });
  });
});
