
import { Controller, Get, Post, Body, Patch, UseGuards, Request, Query } from '@nestjs/common';
import { UsersService } from './users.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { UpdatePreferencesDto, UpdatePlanDto } from './dto/users.dto';
import { User } from '@prisma/client';

@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @UseGuards(JwtAuthGuard)
  @Get('me')
  async getProfile(@Request() req) {
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
  @Get('me/stats')
  async getStats(@Request() req) {
    return this.usersService.getUserStats(req.user.userId);
  }

  /**
   * Get weekly activity for Dashboard charts
   * Query: ?days=7 (default 7), ?offset=0 (days offset from today)
   */
  @UseGuards(JwtAuthGuard)
  @Get('me/activity')
  async getActivity(
    @Request() req,
    @Query('days') days?: string,
    @Query('offset') offset?: string
  ) {
    const numDays = parseInt(days || '7', 10);
    const numOffset = parseInt(offset || '0', 10);
    return this.usersService.getWeeklyActivity(req.user.userId, numDays, numOffset);
  }

  /**
   * Get yearly activity for heatmap (Analytics page)
   */
  @UseGuards(JwtAuthGuard)
  @Get('me/activity/yearly')
  async getYearlyActivity(@Request() req) {
    return this.usersService.getYearlyActivity(req.user.userId);
  }

  @UseGuards(JwtAuthGuard)
  @Patch('me/preferences')
  async updatePreferences(@Request() req, @Body() preferences: UpdatePreferencesDto) {
    const user = await this.usersService.updatePreferences(req.user.userId, preferences);

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
  @Patch('me/avatar')
  async updateAvatar(@Request() req, @Body() body: { avatarUrl: string }) {
    const user = await this.usersService.updateAvatar(req.user.userId, body.avatarUrl);

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(req.user.userId);
    const plan = await this.usersService.getActivePlan(req.user.userId);

    return this.transformUser(user, isPremium, plan);
  }

  // New Endpoint: Simulate a payment/upgrade action
  @UseGuards(JwtAuthGuard)
  @Post('upgrade')
  async upgradeToPremium(@Request() req, @Body() planData?: UpdatePlanDto) {
    // In a real app, this would be a webhook from Stripe.
    // Here we manually update the DB to reflect a successful subscription.
    const oneYearFromNow = new Date();
    oneYearFromNow.setFullYear(oneYearFromNow.getFullYear() + 1);

    const plan: UpdatePlanDto = planData || {
        name: 'Pro Annual',
        expiresAt: oneYearFromNow.toISOString()
    };

    const user = await this.usersService.updatePlan(req.user.userId, true, plan);

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(req.user.userId);
    const activePlan = await this.usersService.getActivePlan(req.user.userId);

    return this.transformUser(user, isPremium, activePlan);
  }

  private transformUser(user: User, isPremium?: boolean, plan?: { name: string; expiresAt: string } | null) {
    const { password, ...result } = user;

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
