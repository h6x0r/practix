import { Test, TestingModule } from "@nestjs/testing";
import { PromoCodesService } from "./promocodes.service";
import { PrismaService } from "../prisma/prisma.service";
import { PromoCodeType, PromoCodeApplicableTo } from "@prisma/client";

describe("PromoCodesService", () => {
  let service: PromoCodesService;
  let prismaService: PrismaService;

  const mockPrismaService = {
    promoCode: {
      findUnique: jest.fn(),
      findMany: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      count: jest.fn(),
    },
    promoCodeUsage: {
      create: jest.fn(),
      count: jest.fn(),
      aggregate: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PromoCodesService,
        {
          provide: PrismaService,
          useValue: mockPrismaService,
        },
      ],
    }).compile();

    service = module.get<PromoCodesService>(PromoCodesService);
    prismaService = module.get<PrismaService>(PrismaService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("createPromoCode", () => {
    it("should create a new promo code", async () => {
      const dto = {
        code: "TEST20",
        type: PromoCodeType.PERCENTAGE,
        discount: 20,
        validFrom: new Date("2026-01-01"),
        validUntil: new Date("2026-12-31"),
      };

      mockPrismaService.promoCode.findUnique.mockResolvedValue(null);
      mockPrismaService.promoCode.create.mockResolvedValue({
        id: "promo-1",
        ...dto,
        code: "TEST20",
        maxUses: null,
        maxUsesPerUser: 1,
        usesCount: 0,
        isActive: true,
        applicableTo: PromoCodeApplicableTo.ALL,
        courseIds: [],
        description: null,
        createdBy: "admin-1",
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      const result = await service.createPromoCode(dto, "admin-1");

      expect(mockPrismaService.promoCode.findUnique).toHaveBeenCalledWith({
        where: { code: "TEST20" },
      });
      expect(mockPrismaService.promoCode.create).toHaveBeenCalled();
      expect(result.code).toBe("TEST20");
    });

    it("should throw if code already exists", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({ id: "existing" });

      await expect(
        service.createPromoCode(
          {
            code: "EXISTING",
            type: PromoCodeType.PERCENTAGE,
            discount: 10,
            validFrom: new Date(),
            validUntil: new Date(),
          },
          "admin-1",
        ),
      ).rejects.toThrow("Promo code already exists");
    });

    it("should normalize code to uppercase", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue(null);
      mockPrismaService.promoCode.create.mockResolvedValue({
        id: "promo-1",
        code: "LOWERCASE",
      });

      await service.createPromoCode(
        {
          code: "lowercase",
          type: PromoCodeType.FIXED,
          discount: 10000,
          validFrom: new Date(),
          validUntil: new Date(),
        },
        "admin-1",
      );

      expect(mockPrismaService.promoCode.findUnique).toHaveBeenCalledWith({
        where: { code: "LOWERCASE" },
      });
    });
  });

  describe("validatePromoCode", () => {
    const validPromoCode = {
      id: "promo-1",
      code: "VALID20",
      type: PromoCodeType.PERCENTAGE,
      discount: 20,
      maxUses: 100,
      maxUsesPerUser: 1,
      usesCount: 10,
      minPurchaseAmount: null,
      validFrom: new Date("2026-01-01"),
      validUntil: new Date("2026-12-31"),
      isActive: true,
      applicableTo: PromoCodeApplicableTo.ALL,
      courseIds: [],
    };

    it("should return valid for a valid promo code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue(validPromoCode);
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(0);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(true);
      expect(result.discountAmount).toBe(20000); // 20% of 100000
    });

    it("should return invalid for non-existent code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue(null);

      const result = await service.validatePromoCode(
        "INVALID",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("Invalid promo code");
    });

    it("should return invalid for inactive code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        isActive: false,
      });

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("Promo code is inactive");
    });

    it("should return invalid for expired code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        validUntil: new Date("2020-01-01"),
      });

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("Promo code has expired");
    });

    it("should return invalid when usage limit reached", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        maxUses: 10,
        usesCount: 10,
      });

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("Promo code usage limit reached");
    });

    it("should return invalid when user already used code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue(validPromoCode);
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(1);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("You have already used this promo code");
    });

    it("should calculate fixed discount correctly", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        type: PromoCodeType.FIXED,
        discount: 50000, // 50000 tiyn fixed discount
      });
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(0);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(true);
      expect(result.discountAmount).toBe(50000);
    });

    it("should cap fixed discount at order amount", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        type: PromoCodeType.FIXED,
        discount: 200000, // More than order amount
      });
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(0);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(true);
      expect(result.discountAmount).toBe(100000); // Capped at order amount
    });

    it("should check minimum purchase amount", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        minPurchaseAmount: 200000,
      });
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(0);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "subscription",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toContain("Minimum purchase amount");
    });

    it("should validate subscription-only codes", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        ...validPromoCode,
        applicableTo: PromoCodeApplicableTo.SUBSCRIPTIONS,
      });
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(0);

      const result = await service.validatePromoCode(
        "VALID20",
        "user-1",
        "purchase",
        100000,
      );

      expect(result.valid).toBe(false);
      expect(result.error).toBe("Promo code not valid for this purchase");
    });
  });

  describe("applyPromoCode", () => {
    it("should create usage record and increment count", async () => {
      mockPrismaService.promoCodeUsage.create.mockResolvedValue({});
      mockPrismaService.promoCode.update.mockResolvedValue({});

      await service.applyPromoCode(
        "promo-1",
        "user-1",
        "order-1",
        "subscription",
        20000,
      );

      expect(mockPrismaService.promoCodeUsage.create).toHaveBeenCalledWith({
        data: {
          promoCodeId: "promo-1",
          userId: "user-1",
          orderId: "order-1",
          orderType: "subscription",
          discountAmount: 20000,
        },
      });

      expect(mockPrismaService.promoCode.update).toHaveBeenCalledWith({
        where: { id: "promo-1" },
        data: { usesCount: { increment: 1 } },
      });
    });
  });

  describe("getAllPromoCodes", () => {
    it("should return paginated promo codes", async () => {
      mockPrismaService.promoCode.findMany.mockResolvedValue([]);
      mockPrismaService.promoCode.count.mockResolvedValue(0);

      const result = await service.getAllPromoCodes({ limit: 10, offset: 0 });

      expect(result).toEqual({ promoCodes: [], total: 0 });
    });

    it("should filter by isActive", async () => {
      mockPrismaService.promoCode.findMany.mockResolvedValue([]);
      mockPrismaService.promoCode.count.mockResolvedValue(0);

      await service.getAllPromoCodes({ isActive: true });

      expect(mockPrismaService.promoCode.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { isActive: true },
        }),
      );
    });
  });

  describe("deletePromoCode", () => {
    it("should delete unused promo code", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        id: "promo-1",
        _count: { usages: 0 },
      });
      mockPrismaService.promoCode.delete.mockResolvedValue({});

      await service.deletePromoCode("promo-1");

      expect(mockPrismaService.promoCode.delete).toHaveBeenCalledWith({
        where: { id: "promo-1" },
      });
    });

    it("should throw if promo code has been used", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue({
        id: "promo-1",
        _count: { usages: 5 },
      });

      await expect(service.deletePromoCode("promo-1")).rejects.toThrow(
        "Cannot delete promo code that has been used",
      );
    });

    it("should throw if promo code not found", async () => {
      mockPrismaService.promoCode.findUnique.mockResolvedValue(null);

      await expect(service.deletePromoCode("promo-1")).rejects.toThrow(
        "Promo code not found",
      );
    });
  });

  describe("getPromoCodeStats", () => {
    it("should return aggregated stats", async () => {
      mockPrismaService.promoCode.count
        .mockResolvedValueOnce(10) // total
        .mockResolvedValueOnce(5) // active
        .mockResolvedValueOnce(5); // expired
      mockPrismaService.promoCodeUsage.count.mockResolvedValue(100);
      mockPrismaService.promoCodeUsage.aggregate.mockResolvedValue({
        _sum: { discountAmount: 500000 },
      });

      const result = await service.getPromoCodeStats();

      expect(result).toEqual({
        total: 10,
        active: 5,
        expired: 5,
        totalUsages: 100,
        totalDiscountGiven: 500000,
      });
    });
  });
});
