import {
  Controller,
  Get,
  Post,
  Delete,
  Body,
  Param,
  UseGuards,
  Request,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SubscriptionsService } from './subscriptions.service';
import { AccessControlService } from './access-control.service';
import { CreateSubscriptionDto } from './dto/subscription.dto';

@Controller('subscriptions')
export class SubscriptionsController {
  constructor(
    private subscriptionsService: SubscriptionsService,
    private accessControlService: AccessControlService,
  ) {}

  /**
   * Get all available subscription plans
   */
  @Get('plans')
  async getPlans() {
    return this.subscriptionsService.getPlans();
  }

  /**
   * Get plan by slug
   */
  @Get('plans/:slug')
  async getPlanBySlug(@Param('slug') slug: string) {
    return this.subscriptionsService.getPlanBySlug(slug);
  }

  /**
   * Get current user's subscriptions
   */
  @Get('my')
  @UseGuards(JwtAuthGuard)
  async getMySubscriptions(@Request() req) {
    return this.subscriptionsService.getUserSubscriptions(req.user.userId);
  }

  /**
   * Get current user's access info for a course
   */
  @Get('access/course/:courseId')
  @UseGuards(JwtAuthGuard)
  async getCourseAccess(
    @Request() req,
    @Param('courseId') courseId: string,
  ) {
    return this.accessControlService.getCourseAccess(req.user.userId, courseId);
  }

  /**
   * Get current user's access info for a task
   */
  @Get('access/task/:taskId')
  @UseGuards(JwtAuthGuard)
  async getTaskAccess(
    @Request() req,
    @Param('taskId') taskId: string,
  ) {
    return this.accessControlService.getTaskAccess(req.user.userId, taskId);
  }

  /**
   * Create a new subscription (called after payment)
   * In production, this would be called by payment webhook
   */
  @Post()
  @UseGuards(JwtAuthGuard)
  async createSubscription(
    @Request() req,
    @Body() dto: CreateSubscriptionDto,
  ) {
    return this.subscriptionsService.createSubscription(req.user.userId, dto);
  }

  /**
   * Cancel a subscription
   */
  @Delete(':id')
  @UseGuards(JwtAuthGuard)
  async cancelSubscription(
    @Request() req,
    @Param('id') id: string,
  ) {
    return this.subscriptionsService.cancelSubscription(req.user.userId, id);
  }
}
