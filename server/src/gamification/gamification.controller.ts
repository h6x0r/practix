import { Controller, Get, UseGuards, Request, Query } from '@nestjs/common';
import { GamificationService } from './gamification.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('gamification')
export class GamificationController {
  constructor(private readonly gamificationService: GamificationService) {}

  /**
   * Get current user's gamification stats (XP, level, badges)
   */
  @UseGuards(JwtAuthGuard)
  @Get('me')
  async getMyStats(@Request() req) {
    return this.gamificationService.getUserStats(req.user.userId);
  }

  /**
   * Get current user's rank
   */
  @UseGuards(JwtAuthGuard)
  @Get('me/rank')
  async getMyRank(@Request() req) {
    const rank = await this.gamificationService.getUserRank(req.user.userId);
    return { rank };
  }

  /**
   * Get leaderboard
   * Requires authentication to protect user data (GDPR compliance)
   */
  @UseGuards(JwtAuthGuard)
  @Get('leaderboard')
  async getLeaderboard(@Query('limit') limit?: string) {
    const parsedLimit = limit ? parseInt(limit, 10) : 50;
    return this.gamificationService.getLeaderboard(parsedLimit);
  }
}
