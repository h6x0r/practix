import { Controller, Post, Body, UseGuards, Request } from '@nestjs/common';
import { AiService } from './ai.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @UseGuards(JwtAuthGuard)
  @Post('tutor')
  async askTutor(
    @Request() req, 
    @Body() body: { taskTitle: string; userCode: string; question: string; language: string }
  ) {
    return this.aiService.askTutor(
        req.user.userId,
        body.taskTitle,
        body.userCode,
        body.question,
        body.language
    );
  }
}