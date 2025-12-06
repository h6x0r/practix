
import { Controller, Post, Body, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SubmissionsService } from './submissions.service';

@Controller('submissions')
export class SubmissionsController {
  constructor(private readonly submissionsService: SubmissionsService) {}

  @UseGuards(JwtAuthGuard)
  @Post()
  async create(@Request() req, @Body() body: { taskId: string; code: string; language: string }) {
    // body.taskId comes from frontend, which might be a slug or UUID.
    return this.submissionsService.create(req.user.userId, body.taskId, body.code, body.language);
  }
}
