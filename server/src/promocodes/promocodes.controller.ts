import {
  Controller,
  Get,
  Post,
  Delete,
  Body,
  Param,
  Query,
  UseGuards,
  Request,
  BadRequestException,
  NotFoundException,
} from "@nestjs/common";
import { Throttle } from "@nestjs/throttler";
import { PromoCodesService, CreatePromoCodeDto } from "./promocodes.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { AdminGuard } from "../auth/guards/admin.guard";
import { PromoCodeType, PromoCodeApplicableTo } from "@prisma/client";

@Controller("admin/promocodes")
@UseGuards(JwtAuthGuard, AdminGuard)
@Throttle({ default: { limit: 30, ttl: 60000 } })
export class PromoCodesController {
  constructor(private readonly promoCodesService: PromoCodesService) {}

  /**
   * GET /admin/promocodes
   * Get all promo codes with pagination
   */
  @Get()
  async getAllPromoCodes(
    @Query("isActive") isActive?: string,
    @Query("limit") limit?: string,
    @Query("offset") offset?: string,
  ) {
    return this.promoCodesService.getAllPromoCodes({
      isActive: isActive !== undefined ? isActive === "true" : undefined,
      limit: limit ? parseInt(limit, 10) : 50,
      offset: offset ? parseInt(offset, 10) : 0,
    });
  }

  /**
   * GET /admin/promocodes/stats
   * Get promo code statistics
   */
  @Get("stats")
  async getPromoCodeStats() {
    return this.promoCodesService.getPromoCodeStats();
  }

  /**
   * GET /admin/promocodes/:id
   * Get promo code details with usage history
   */
  @Get(":id")
  async getPromoCodeById(@Param("id") promoCodeId: string) {
    const promoCode = await this.promoCodesService.getPromoCodeById(promoCodeId);
    if (!promoCode) {
      throw new NotFoundException("Promo code not found");
    }
    return promoCode;
  }

  /**
   * POST /admin/promocodes
   * Create a new promo code
   */
  @Post()
  async createPromoCode(
    @Body()
    body: {
      code: string;
      type: string;
      discount: number;
      maxUses?: number;
      maxUsesPerUser?: number;
      minPurchaseAmount?: number;
      validFrom: string;
      validUntil: string;
      applicableTo?: string;
      courseIds?: string[];
      description?: string;
    },
    @Request() req: { user: { userId: string } },
  ) {
    // Validate code
    if (!body.code || body.code.trim().length < 3) {
      throw new BadRequestException("Code must be at least 3 characters");
    }

    // Validate type
    const validTypes = Object.values(PromoCodeType);
    if (!validTypes.includes(body.type as PromoCodeType)) {
      throw new BadRequestException(
        `Invalid type. Must be one of: ${validTypes.join(", ")}`,
      );
    }

    // Validate discount
    if (body.discount === undefined || body.discount <= 0) {
      throw new BadRequestException("Discount must be a positive number");
    }

    // Validate dates
    const validFrom = new Date(body.validFrom);
    const validUntil = new Date(body.validUntil);
    if (isNaN(validFrom.getTime()) || isNaN(validUntil.getTime())) {
      throw new BadRequestException("Invalid date format");
    }
    if (validUntil <= validFrom) {
      throw new BadRequestException("validUntil must be after validFrom");
    }

    // Validate applicableTo
    let applicableTo: PromoCodeApplicableTo | undefined;
    if (body.applicableTo) {
      const validApplicableTo = Object.values(PromoCodeApplicableTo);
      if (!validApplicableTo.includes(body.applicableTo as PromoCodeApplicableTo)) {
        throw new BadRequestException(
          `Invalid applicableTo. Must be one of: ${validApplicableTo.join(", ")}`,
        );
      }
      applicableTo = body.applicableTo as PromoCodeApplicableTo;
    }

    const dto: CreatePromoCodeDto = {
      code: body.code,
      type: body.type as PromoCodeType,
      discount: body.discount,
      maxUses: body.maxUses,
      maxUsesPerUser: body.maxUsesPerUser,
      minPurchaseAmount: body.minPurchaseAmount,
      validFrom,
      validUntil,
      applicableTo,
      courseIds: body.courseIds,
      description: body.description,
    };

    try {
      return await this.promoCodesService.createPromoCode(dto, req.user.userId);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to create promo code",
      );
    }
  }

  /**
   * POST /admin/promocodes/:id/update
   * Update promo code
   */
  @Post(":id/update")
  async updatePromoCode(
    @Param("id") promoCodeId: string,
    @Body()
    body: {
      type?: string;
      discount?: number;
      maxUses?: number;
      maxUsesPerUser?: number;
      minPurchaseAmount?: number;
      validFrom?: string;
      validUntil?: string;
      applicableTo?: string;
      courseIds?: string[];
      description?: string;
    },
  ) {
    const updates: Partial<Omit<CreatePromoCodeDto, "code">> = {};

    if (body.type) {
      const validTypes = Object.values(PromoCodeType);
      if (!validTypes.includes(body.type as PromoCodeType)) {
        throw new BadRequestException(`Invalid type`);
      }
      updates.type = body.type as PromoCodeType;
    }

    if (body.discount !== undefined) {
      if (body.discount <= 0) {
        throw new BadRequestException("Discount must be positive");
      }
      updates.discount = body.discount;
    }

    if (body.maxUses !== undefined) updates.maxUses = body.maxUses;
    if (body.maxUsesPerUser !== undefined)
      updates.maxUsesPerUser = body.maxUsesPerUser;
    if (body.minPurchaseAmount !== undefined)
      updates.minPurchaseAmount = body.minPurchaseAmount;

    if (body.validFrom) updates.validFrom = new Date(body.validFrom);
    if (body.validUntil) updates.validUntil = new Date(body.validUntil);

    if (body.applicableTo) {
      const validApplicableTo = Object.values(PromoCodeApplicableTo);
      if (!validApplicableTo.includes(body.applicableTo as PromoCodeApplicableTo)) {
        throw new BadRequestException(`Invalid applicableTo`);
      }
      updates.applicableTo = body.applicableTo as PromoCodeApplicableTo;
    }

    if (body.courseIds !== undefined) updates.courseIds = body.courseIds;
    if (body.description !== undefined) updates.description = body.description;

    try {
      return await this.promoCodesService.updatePromoCode(promoCodeId, updates);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to update promo code",
      );
    }
  }

  /**
   * POST /admin/promocodes/:id/activate
   * Activate promo code
   */
  @Post(":id/activate")
  async activatePromoCode(@Param("id") promoCodeId: string) {
    try {
      return await this.promoCodesService.activatePromoCode(promoCodeId);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to activate",
      );
    }
  }

  /**
   * POST /admin/promocodes/:id/deactivate
   * Deactivate promo code
   */
  @Post(":id/deactivate")
  async deactivatePromoCode(@Param("id") promoCodeId: string) {
    try {
      return await this.promoCodesService.deactivatePromoCode(promoCodeId);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to deactivate",
      );
    }
  }

  /**
   * DELETE /admin/promocodes/:id
   * Delete promo code (only if never used)
   */
  @Delete(":id")
  async deletePromoCode(@Param("id") promoCodeId: string) {
    try {
      await this.promoCodesService.deletePromoCode(promoCodeId);
      return { success: true };
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to delete promo code",
      );
    }
  }
}
