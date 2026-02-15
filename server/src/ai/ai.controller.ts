import {
  Controller,
  Post,
  Get,
  Body,
  Query,
  UseGuards,
  Request,
} from "@nestjs/common";
import { Throttle } from "@nestjs/throttler";
import { AiService } from "./ai.service";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { AskAiDto } from "./dto/ai.dto";
import { AuthenticatedRequest } from '../common/types';

@Controller("ai")
export class AiController {
  constructor(private readonly aiService: AiService) {}

  /**
   * Get current AI usage limits for the authenticated user
   * Returns tier, limit, used, and remaining counts
   */
  @UseGuards(JwtAuthGuard)
  @Get("limits")
  async getLimits(@Request() req: AuthenticatedRequest, @Query("taskId") taskId?: string) {
    return this.aiService.getAiLimitInfo(req.user.userId, taskId);
  }

  /**
   * Ask AI Tutor for help with a task
   * Rate limited: 1 request per 5 seconds per user
   */
  @UseGuards(JwtAuthGuard)
  @Post("tutor")
  @Throttle({ default: { limit: 1, ttl: 5000 } }) // 1 request per 5 seconds
  async askTutor(@Request() req: AuthenticatedRequest, @Body() body: AskAiDto) {
    return this.aiService.askTutor(
      req.user.userId,
      body.taskId,
      body.taskTitle,
      body.userCode,
      body.question,
      body.language,
      body.uiLanguage || "en",
    );
  }
}
