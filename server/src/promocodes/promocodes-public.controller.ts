import {
  Controller,
  Post,
  Body,
  UseGuards,
  Request,
  BadRequestException,
} from "@nestjs/common";
import { Throttle } from "@nestjs/throttler";
import { PromoCodesService } from "./promocodes.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";

@Controller("promocodes")
@UseGuards(JwtAuthGuard)
@Throttle({ default: { limit: 20, ttl: 60000 } }) // 20 validations per minute
export class PromoCodesPublicController {
  constructor(private readonly promoCodesService: PromoCodesService) {}

  /**
   * POST /promocodes/validate
   * Validate a promo code (public endpoint for users)
   */
  @Post("validate")
  async validatePromoCode(
    @Body()
    body: {
      code: string;
      orderType: "subscription" | "purchase";
      amount: number;
      courseId?: string;
    },
    @Request() req: { user: { userId: string } },
  ) {
    if (!body.code || body.code.trim().length === 0) {
      throw new BadRequestException("Promo code is required");
    }

    if (!body.orderType || !["subscription", "purchase"].includes(body.orderType)) {
      throw new BadRequestException("Invalid order type");
    }

    if (body.amount === undefined || body.amount <= 0) {
      throw new BadRequestException("Amount must be positive");
    }

    const result = await this.promoCodesService.validatePromoCode(
      body.code,
      req.user.userId,
      body.orderType,
      body.amount,
      body.courseId,
    );

    if (!result.valid) {
      return {
        valid: false,
        error: result.error,
      };
    }

    return {
      valid: true,
      code: result.promoCode?.code,
      type: result.promoCode?.type,
      discount: result.promoCode?.discount,
      discountAmount: result.discountAmount,
    };
  }
}
