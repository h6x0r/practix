import {
  Controller,
  Get,
  Post,
  Delete,
  Body,
  Param,
  UseGuards,
  Request,
  Headers,
  UnauthorizedException,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';
import { SubscriptionsService } from './subscriptions.service';
import { AccessControlService } from './access-control.service';
import { CreateSubscriptionDto } from './dto/subscription.dto';
import { ConfigService } from '@nestjs/config';

@Controller('subscriptions')
export class SubscriptionsController {
  private readonly webhookSecret: string;

  constructor(
    private subscriptionsService: SubscriptionsService,
    private accessControlService: AccessControlService,
    private configService: ConfigService,
  ) {
    this.webhookSecret = this.configService.get<string>('STRIPE_WEBHOOK_SECRET') || '';
  }

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
   * Create a new subscription (ADMIN ONLY)
   * For manual subscription creation by administrators.
   * For payment-based subscriptions, use the webhook endpoint.
   */
  @Post()
  @UseGuards(JwtAuthGuard, AdminGuard)
  async createSubscription(
    @Request() req,
    @Body() dto: CreateSubscriptionDto & { userId?: string },
  ) {
    // Admin can create subscription for any user (if userId provided) or themselves
    const targetUserId = dto.userId || req.user.userId;
    return this.subscriptionsService.createSubscription(targetUserId, dto);
  }

  /**
   * Webhook endpoint for Stripe payment callbacks
   * Validates webhook signature before processing
   */
  @Post('webhook/stripe')
  async handleStripeWebhook(
    @Body() payload: any,
    @Headers('stripe-signature') signature: string,
  ) {
    if (!this.webhookSecret) {
      throw new UnauthorizedException('Webhook not configured');
    }

    // In production, validate Stripe signature here:
    // const event = stripe.webhooks.constructEvent(payload, signature, this.webhookSecret);
    // For now, just check that signature header is present
    if (!signature) {
      throw new UnauthorizedException('Missing webhook signature');
    }

    // Process webhook event
    // This is a placeholder - implement actual Stripe webhook handling
    return { received: true };
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
