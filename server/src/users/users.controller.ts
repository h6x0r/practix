
import { Controller, Get, Post, Body, Patch, UseGuards, Request } from '@nestjs/common';
import { UsersService } from './users.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @UseGuards(JwtAuthGuard)
  @Get('me')
  async getProfile(@Request() req) {
    const user = await this.usersService.findById(req.user.userId);
    return this.transformUser(user);
  }

  @UseGuards(JwtAuthGuard)
  @Patch('me/preferences')
  async updatePreferences(@Request() req, @Body() preferences: any) {
    const user = await this.usersService.updatePreferences(req.user.userId, preferences);
    return this.transformUser(user);
  }

  // New Endpoint: Simulate a payment/upgrade action
  @UseGuards(JwtAuthGuard)
  @Post('upgrade')
  async upgradeToPremium(@Request() req) {
    // In a real app, this would be a webhook from Stripe.
    // Here we manually update the DB to reflect a successful subscription.
    const oneYearFromNow = new Date();
    oneYearFromNow.setFullYear(oneYearFromNow.getFullYear() + 1);

    const planData = {
        name: 'Pro Annual',
        expiresAt: oneYearFromNow.toISOString()
    };

    const user = await this.usersService.updatePlan(req.user.userId, true, planData);
    return this.transformUser(user);
  }

  private transformUser(user: any) {
    const { password, ...result } = user;
    return result;
  }
}
