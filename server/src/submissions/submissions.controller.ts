import {
  Controller,
  Post,
  Get,
  Body,
  Param,
  UseGuards,
  Request,
  Query,
  HttpCode,
  HttpStatus,
  ForbiddenException,
} from '@nestjs/common';
import { ThrottlerGuard, Throttle, SkipThrottle } from '@nestjs/throttler';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { SubmissionsService } from './submissions.service';
import { CodeExecutionService } from '../queue/code-execution.service';
import { CreateSubmissionDto, RunCodeDto, RunTestsDto } from './dto/submissions.dto';

@Controller('submissions')
@UseGuards(ThrottlerGuard)
export class SubmissionsController {
  constructor(
    private readonly submissionsService: SubmissionsService,
    private readonly codeExecutionService: CodeExecutionService
  ) {}

  /**
   * POST /submissions
   * Submit code for a task (requires auth, saves to DB)
   * Rate limit: 10 submissions per minute
   */
  @UseGuards(JwtAuthGuard)
  @Post()
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 10, ttl: 60000 } }) // 10 submissions per minute
  async create(@Request() req, @Body() body: CreateSubmissionDto) {
    return this.submissionsService.create(
      req.user.userId,
      body.taskId,
      body.code,
      body.language
    );
  }

  /**
   * POST /submissions/run
   * Run code without saving (playground mode)
   * Rate limit: 20 runs per minute (more lenient for testing)
   */
  @Post('run')
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 20, ttl: 60000 } }) // 20 runs per minute
  async runCode(@Body() body: RunCodeDto) {
    return this.submissionsService.runCode(body.code, body.language, body.stdin);
  }

  /**
   * POST /submissions/run-tests
   * Run quick tests (5 tests) without saving to database
   * Used for "Run Code" button - fast feedback without full submission
   * Rate limit: 15 runs per minute
   */
  @Post('run-tests')
  @HttpCode(HttpStatus.OK)
  @Throttle({ default: { limit: 15, ttl: 60000 } }) // 15 runs per minute
  async runTests(@Body() body: RunTestsDto) {
    return this.submissionsService.runQuickTests(
      body.taskId,
      body.code,
      body.language
    );
  }

  /**
   * GET /submissions/judge/status
   * Get execution engine health and queue status
   */
  @Get('judge/status')
  @SkipThrottle() // No rate limit for status checks
  async getJudgeStatus() {
    return this.submissionsService.getJudgeStatus();
  }

  /**
   * GET /submissions/languages
   * Get list of supported languages
   */
  @Get('languages')
  @SkipThrottle() // No rate limit for language list
  getLanguages() {
    return {
      languages: this.codeExecutionService.getSupportedLanguages(),
      default: 'go',
    };
  }

  /**
   * GET /submissions/:id
   * Get submission by ID (requires auth, owner only)
   */
  @UseGuards(JwtAuthGuard)
  @Get(':id')
  async findOne(@Request() req, @Param('id') id: string) {
    const submission = await this.submissionsService.findOne(id);

    // Check ownership - users can only access their own submissions
    if (submission.userId !== req.user.userId) {
      throw new ForbiddenException('You can only access your own submissions');
    }

    return submission;
  }

  /**
   * GET /submissions/user/recent
   * Get user's recent submissions (requires auth)
   */
  @UseGuards(JwtAuthGuard)
  @Get('user/recent')
  async getUserRecent(@Request() req, @Query('limit') limit?: string) {
    const limitNum = limit ? parseInt(limit, 10) : 10;
    return this.submissionsService.findRecentByUser(req.user.userId, limitNum);
  }

  /**
   * GET /submissions/task/:taskId
   * Get user's submissions for a specific task (requires auth)
   */
  @UseGuards(JwtAuthGuard)
  @Get('task/:taskId')
  async getByTask(@Request() req, @Param('taskId') taskId: string) {
    return this.submissionsService.findByUserAndTask(req.user.userId, taskId);
  }
}
