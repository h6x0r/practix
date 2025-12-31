import { Controller, Get, Delete, Param, UseGuards, Request, HttpException, HttpStatus } from '@nestjs/common';
import { SessionsService } from './sessions.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('users/me/sessions')
@UseGuards(JwtAuthGuard)
export class SessionsController {
  constructor(private readonly sessionsService: SessionsService) {}

  /**
   * Get all active sessions for the authenticated user
   * GET /users/me/sessions
   */
  @Get()
  async getUserSessions(@Request() req) {
    const sessions = await this.sessionsService.getUserSessions(req.user.userId);

    // Transform sessions to remove sensitive token data
    return sessions.map(session => ({
      id: session.id,
      deviceInfo: session.deviceInfo,
      ipAddress: session.ipAddress,
      createdAt: session.createdAt,
      lastActiveAt: session.lastActiveAt,
      expiresAt: session.expiresAt,
      isActive: session.isActive,
    }));
  }

  /**
   * Invalidate a specific session
   * DELETE /users/me/sessions/:sessionId
   */
  @Delete(':sessionId')
  async invalidateSession(@Request() req, @Param('sessionId') sessionId: string) {
    // First, verify the session belongs to the authenticated user
    const sessions = await this.sessionsService.getUserSessions(req.user.userId);
    const session = sessions.find(s => s.id === sessionId);

    if (!session) {
      throw new HttpException(
        'Session not found or does not belong to you',
        HttpStatus.NOT_FOUND,
      );
    }

    // Invalidate the session
    await this.sessionsService.invalidateSession(sessionId);

    return {
      message: 'Session invalidated successfully',
      sessionId,
    };
  }
}
