import { Injectable } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";
import {
  PromoCodeType,
  PromoCodeApplicableTo,
  PromoCode,
} from "@prisma/client";

export interface CreatePromoCodeDto {
  code: string;
  type: PromoCodeType;
  discount: number;
  maxUses?: number;
  maxUsesPerUser?: number;
  minPurchaseAmount?: number;
  validFrom: Date;
  validUntil: Date;
  applicableTo?: PromoCodeApplicableTo;
  courseIds?: string[];
  description?: string;
}

export interface ValidatePromoCodeResult {
  valid: boolean;
  error?: string;
  promoCode?: PromoCode;
  discountAmount?: number;
}

@Injectable()
export class PromoCodesService {
  constructor(private readonly prisma: PrismaService) {}

  /**
   * Create a new promo code (admin action)
   */
  async createPromoCode(dto: CreatePromoCodeDto, adminId: string) {
    // Normalize code to uppercase
    const code = dto.code.toUpperCase().trim();

    // Check if code already exists
    const existing = await this.prisma.promoCode.findUnique({
      where: { code },
    });

    if (existing) {
      throw new Error("Promo code already exists");
    }

    return this.prisma.promoCode.create({
      data: {
        code,
        type: dto.type,
        discount: dto.discount,
        maxUses: dto.maxUses || null,
        maxUsesPerUser: dto.maxUsesPerUser || 1,
        minPurchaseAmount: dto.minPurchaseAmount || null,
        validFrom: dto.validFrom,
        validUntil: dto.validUntil,
        applicableTo: dto.applicableTo || PromoCodeApplicableTo.ALL,
        courseIds: dto.courseIds || [],
        description: dto.description || null,
        createdBy: adminId,
      },
    });
  }

  /**
   * Update promo code (admin action)
   */
  async updatePromoCode(
    promoCodeId: string,
    updates: Partial<Omit<CreatePromoCodeDto, "code">>,
  ) {
    const promoCode = await this.prisma.promoCode.findUnique({
      where: { id: promoCodeId },
    });

    if (!promoCode) {
      throw new Error("Promo code not found");
    }

    return this.prisma.promoCode.update({
      where: { id: promoCodeId },
      data: {
        ...(updates.type !== undefined && { type: updates.type }),
        ...(updates.discount !== undefined && { discount: updates.discount }),
        ...(updates.maxUses !== undefined && { maxUses: updates.maxUses }),
        ...(updates.maxUsesPerUser !== undefined && {
          maxUsesPerUser: updates.maxUsesPerUser,
        }),
        ...(updates.minPurchaseAmount !== undefined && {
          minPurchaseAmount: updates.minPurchaseAmount,
        }),
        ...(updates.validFrom !== undefined && { validFrom: updates.validFrom }),
        ...(updates.validUntil !== undefined && {
          validUntil: updates.validUntil,
        }),
        ...(updates.applicableTo !== undefined && {
          applicableTo: updates.applicableTo,
        }),
        ...(updates.courseIds !== undefined && { courseIds: updates.courseIds }),
        ...(updates.description !== undefined && {
          description: updates.description,
        }),
      },
    });
  }

  /**
   * Deactivate promo code (admin action)
   */
  async deactivatePromoCode(promoCodeId: string) {
    return this.prisma.promoCode.update({
      where: { id: promoCodeId },
      data: { isActive: false },
    });
  }

  /**
   * Activate promo code (admin action)
   */
  async activatePromoCode(promoCodeId: string) {
    return this.prisma.promoCode.update({
      where: { id: promoCodeId },
      data: { isActive: true },
    });
  }

  /**
   * Get all promo codes (admin)
   */
  async getAllPromoCodes(params: {
    isActive?: boolean;
    limit?: number;
    offset?: number;
  }) {
    const { isActive, limit = 50, offset = 0 } = params;

    const where = isActive !== undefined ? { isActive } : {};

    const [promoCodes, total] = await Promise.all([
      this.prisma.promoCode.findMany({
        where,
        include: {
          _count: {
            select: { usages: true },
          },
        },
        orderBy: { createdAt: "desc" },
        take: limit,
        skip: offset,
      }),
      this.prisma.promoCode.count({ where }),
    ]);

    return { promoCodes, total };
  }

  /**
   * Get promo code by ID with usage stats
   */
  async getPromoCodeById(promoCodeId: string) {
    const promoCode = await this.prisma.promoCode.findUnique({
      where: { id: promoCodeId },
      include: {
        usages: {
          include: {
            user: {
              select: { id: true, email: true, name: true },
            },
          },
          orderBy: { createdAt: "desc" },
          take: 50,
        },
        _count: {
          select: { usages: true },
        },
      },
    });

    return promoCode;
  }

  /**
   * Validate promo code for user (public endpoint)
   */
  async validatePromoCode(
    code: string,
    userId: string,
    orderType: "subscription" | "purchase",
    amount: number,
    courseId?: string,
  ): Promise<ValidatePromoCodeResult> {
    const normalizedCode = code.toUpperCase().trim();

    const promoCode = await this.prisma.promoCode.findUnique({
      where: { code: normalizedCode },
    });

    if (!promoCode) {
      return { valid: false, error: "Invalid promo code" };
    }

    // Check if active
    if (!promoCode.isActive) {
      return { valid: false, error: "Promo code is inactive" };
    }

    // Check date validity
    const now = new Date();
    if (now < promoCode.validFrom) {
      return { valid: false, error: "Promo code is not yet valid" };
    }
    if (now > promoCode.validUntil) {
      return { valid: false, error: "Promo code has expired" };
    }

    // Check max uses
    if (promoCode.maxUses !== null && promoCode.usesCount >= promoCode.maxUses) {
      return { valid: false, error: "Promo code usage limit reached" };
    }

    // Check user usage limit
    const userUsageCount = await this.prisma.promoCodeUsage.count({
      where: {
        promoCodeId: promoCode.id,
        userId,
      },
    });

    if (userUsageCount >= promoCode.maxUsesPerUser) {
      return {
        valid: false,
        error: "You have already used this promo code",
      };
    }

    // Check applicable to
    if (promoCode.applicableTo !== PromoCodeApplicableTo.ALL) {
      if (
        orderType === "subscription" &&
        promoCode.applicableTo !== PromoCodeApplicableTo.SUBSCRIPTIONS
      ) {
        return {
          valid: false,
          error: "Promo code not valid for subscriptions",
        };
      }
      if (
        orderType === "purchase" &&
        promoCode.applicableTo !== PromoCodeApplicableTo.PURCHASES &&
        promoCode.applicableTo !== PromoCodeApplicableTo.COURSES
      ) {
        return { valid: false, error: "Promo code not valid for this purchase" };
      }
    }

    // Check course restriction
    if (
      promoCode.applicableTo === PromoCodeApplicableTo.COURSES &&
      promoCode.courseIds.length > 0 &&
      courseId
    ) {
      if (!promoCode.courseIds.includes(courseId)) {
        return { valid: false, error: "Promo code not valid for this course" };
      }
    }

    // Check minimum purchase amount
    if (
      promoCode.minPurchaseAmount !== null &&
      amount < promoCode.minPurchaseAmount
    ) {
      const minAmountUzs = promoCode.minPurchaseAmount / 100;
      return {
        valid: false,
        error: `Minimum purchase amount is ${minAmountUzs.toLocaleString()} UZS`,
      };
    }

    // Calculate discount amount
    let discountAmount = 0;
    if (promoCode.type === PromoCodeType.PERCENTAGE) {
      discountAmount = Math.floor((amount * promoCode.discount) / 100);
    } else if (promoCode.type === PromoCodeType.FIXED) {
      discountAmount = Math.min(promoCode.discount, amount);
    }
    // FREE_TRIAL doesn't have a discount amount, it adds trial days

    return {
      valid: true,
      promoCode,
      discountAmount,
    };
  }

  /**
   * Apply promo code (called during payment completion)
   */
  async applyPromoCode(
    promoCodeId: string,
    userId: string,
    orderId: string,
    orderType: "subscription" | "purchase",
    discountAmount: number,
  ) {
    // Create usage record
    await this.prisma.promoCodeUsage.create({
      data: {
        promoCodeId,
        userId,
        orderId,
        orderType,
        discountAmount,
      },
    });

    // Increment usage count
    await this.prisma.promoCode.update({
      where: { id: promoCodeId },
      data: {
        usesCount: { increment: 1 },
      },
    });
  }

  /**
   * Get promo code stats (admin)
   */
  async getPromoCodeStats() {
    const now = new Date();

    const [total, active, expired, totalUsages, totalDiscount] =
      await Promise.all([
        this.prisma.promoCode.count(),
        this.prisma.promoCode.count({
          where: {
            isActive: true,
            validUntil: { gt: now },
          },
        }),
        this.prisma.promoCode.count({
          where: {
            OR: [{ isActive: false }, { validUntil: { lt: now } }],
          },
        }),
        this.prisma.promoCodeUsage.count(),
        this.prisma.promoCodeUsage.aggregate({
          _sum: { discountAmount: true },
        }),
      ]);

    return {
      total,
      active,
      expired,
      totalUsages,
      totalDiscountGiven: totalDiscount._sum.discountAmount || 0,
    };
  }

  /**
   * Delete promo code (admin action) - only if never used
   */
  async deletePromoCode(promoCodeId: string) {
    const promoCode = await this.prisma.promoCode.findUnique({
      where: { id: promoCodeId },
      include: {
        _count: {
          select: { usages: true },
        },
      },
    });

    if (!promoCode) {
      throw new Error("Promo code not found");
    }

    if (promoCode._count.usages > 0) {
      throw new Error(
        "Cannot delete promo code that has been used. Deactivate instead.",
      );
    }

    return this.prisma.promoCode.delete({
      where: { id: promoCodeId },
    });
  }
}
