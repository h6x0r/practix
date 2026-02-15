import {
  Controller,
  Get,
  Body,
  Patch,
  UseGuards,
  Request,
  Query,
} from "@nestjs/common";
import { UsersService } from "./users.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { UpdatePreferencesDto } from "./dto/users.dto";
import { User } from "@prisma/client";
import { AuthenticatedRequest } from "../common/types";

@Controller("users")
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @UseGuards(JwtAuthGuard)
  @Get("me")
  async getProfile(@Request() req: AuthenticatedRequest) {
    const user = await this.usersService.findById(req.user.userId);

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(req.user.userId);
    const plan = await this.usersService.getActivePlan(req.user.userId);

    return this.transformUser(user, isPremium, plan);
  }

  /**
   * Get user statistics for Dashboard
   * Returns: totalSolved, hoursSpent, globalRank, skillPoints, streak
   */
  @UseGuards(JwtAuthGuard)
  @Get("me/stats")
  async getStats(@Request() req: AuthenticatedRequest) {
    return this.usersService.getUserStats(req.user.userId);
  }

  /**
   * Get weekly activity for Dashboard charts
   * Query: ?days=7 (default 7), ?offset=0 (days offset from today)
   */
  @UseGuards(JwtAuthGuard)
  @Get("me/activity")
  async getActivity(
    @Request() req: AuthenticatedRequest,
    @Query("days") days?: string,
    @Query("offset") offset?: string,
  ) {
    const numDays = parseInt(days || "7", 10);
    const numOffset = parseInt(offset || "0", 10);
    return this.usersService.getWeeklyActivity(
      req.user.userId,
      numDays,
      numOffset,
    );
  }

  /**
   * Get yearly activity for heatmap (Analytics page)
   */
  @UseGuards(JwtAuthGuard)
  @Get("me/activity/yearly")
  async getYearlyActivity(@Request() req: AuthenticatedRequest) {
    return this.usersService.getYearlyActivity(req.user.userId);
  }

  @UseGuards(JwtAuthGuard)
  @Patch("me/preferences")
  async updatePreferences(
    @Request() req: AuthenticatedRequest,
    @Body() preferences: UpdatePreferencesDto,
  ) {
    const user = await this.usersService.updatePreferences(
      req.user.userId,
      preferences,
    );

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(req.user.userId);
    const plan = await this.usersService.getActivePlan(req.user.userId);

    return this.transformUser(user, isPremium, plan);
  }

  /**
   * Update user avatar
   * Accepts base64 image data or a preset avatar URL
   */
  @UseGuards(JwtAuthGuard)
  @Patch("me/avatar")
  async updateAvatar(
    @Request() req: AuthenticatedRequest,
    @Body() body: { avatarUrl: string },
  ) {
    const user = await this.usersService.updateAvatar(
      req.user.userId,
      body.avatarUrl,
    );

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(req.user.userId);
    const plan = await this.usersService.getActivePlan(req.user.userId);

    return this.transformUser(user, isPremium, plan);
  }

  // NOTE: /users/upgrade endpoint removed for security.
  // Subscription upgrades should only be processed through:
  // 1. POST /subscriptions (admin-only) - for manual subscriptions
  // 2. POST /webhooks/stripe - for payment gateway callbacks

  private transformUser(
    user: Partial<User> | null,
    isPremium?: boolean,
    plan?: { name: string; expiresAt: string } | null,
  ) {
    // Create a copy and ensure password is never included
    const { password, ...result } = user as any;

    // Override with computed values if provided
    if (isPremium !== undefined) {
      result.isPremium = isPremium;
    }
    if (plan !== undefined) {
      result.plan = plan;
    }

    return result;
  }
}
