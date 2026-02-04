import {
  Controller,
  Get,
  Post,
  Query,
  Param,
  Body,
  UseGuards,
  Request,
  NotFoundException,
  BadRequestException,
} from "@nestjs/common";
import { Throttle } from "@nestjs/throttler";
import { AdminService } from "./admin.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { AdminGuard } from "../auth/guards/admin.guard";

@Controller("admin/analytics")
@UseGuards(JwtAuthGuard, AdminGuard)
@Throttle({ default: { limit: 30, ttl: 60000 } }) // 30 requests per minute for admin endpoints
export class AdminController {
  constructor(private readonly adminService: AdminService) {}

  /**
   * GET /admin/analytics/dashboard
   * Returns dashboard statistics including total users, new users, and active users
   */
  @Get("dashboard")
  async getDashboardStats() {
    return this.adminService.getDashboardStats();
  }

  /**
   * GET /admin/analytics/courses
   * Returns course analytics including popularity, completion rates, and average progress
   */
  @Get("courses")
  async getCourseAnalytics() {
    return this.adminService.getCourseAnalytics();
  }

  /**
   * GET /admin/analytics/tasks
   * Returns task analytics including hardest tasks and most popular tasks
   */
  @Get("tasks")
  async getTaskAnalytics() {
    return this.adminService.getTaskAnalytics();
  }

  /**
   * GET /admin/analytics/submissions
   * Returns submission statistics including total submissions, by status, and daily trends
   */
  @Get("submissions")
  async getSubmissionStats() {
    return this.adminService.getSubmissionStats();
  }

  /**
   * GET /admin/analytics/subscriptions
   * Returns subscription statistics including active subscriptions, new subscriptions, and revenue
   */
  @Get("subscriptions")
  async getSubscriptionStats() {
    return this.adminService.getSubscriptionStats();
  }

  /**
   * GET /admin/analytics/ai-usage
   * Returns AI usage statistics including total usage and daily trends
   */
  @Get("ai-usage")
  async getAiUsageStats() {
    return this.adminService.getAiUsageStats();
  }

  /**
   * GET /admin/analytics/users/search?q=query
   * Search users by email or name
   */
  @Get("users/search")
  async searchUsers(@Query("q") query: string) {
    return this.adminService.searchUsers(query);
  }

  /**
   * GET /admin/analytics/users/:id
   * Get user details by ID
   */
  @Get("users/:id")
  async getUserById(@Param("id") userId: string) {
    const user = await this.adminService.getUserById(userId);
    if (!user) {
      throw new NotFoundException("User not found");
    }
    return user;
  }

  /**
   * GET /admin/analytics/users/banned
   * Get list of banned users
   */
  @Get("users/banned/list")
  async getBannedUsers(
    @Query("limit") limit?: string,
    @Query("offset") offset?: string,
  ) {
    return this.adminService.getBannedUsers(
      limit ? parseInt(limit, 10) : 50,
      offset ? parseInt(offset, 10) : 0,
    );
  }

  /**
   * POST /admin/analytics/users/:id/ban
   * Ban a user
   */
  @Post("users/:id/ban")
  async banUser(
    @Param("id") userId: string,
    @Body("reason") reason: string,
    @Request() req: { user: { userId: string } },
  ) {
    if (!reason || reason.trim().length === 0) {
      throw new BadRequestException("Ban reason is required");
    }
    try {
      return await this.adminService.banUser(
        userId,
        reason.trim(),
        req.user.userId,
      );
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to ban user",
      );
    }
  }

  /**
   * POST /admin/analytics/users/:id/unban
   * Unban a user
   */
  @Post("users/:id/unban")
  async unbanUser(@Param("id") userId: string) {
    try {
      return await this.adminService.unbanUser(userId);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : "Failed to unban user",
      );
    }
  }
}
