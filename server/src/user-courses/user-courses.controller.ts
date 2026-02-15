import {
  Controller,
  Get,
  Post,
  Patch,
  Param,
  Body,
  UseGuards,
  Request,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { UserCoursesService } from './user-courses.service';
import { UpdateProgressDto } from './dto/user-courses.dto';
import { AuthenticatedRequest } from '../common/types';

@Controller('users/me/courses')
@UseGuards(JwtAuthGuard)
export class UserCoursesController {
  constructor(private readonly userCoursesService: UserCoursesService) {}

  /**
   * GET /users/me/courses
   * Get all courses started by the authenticated user
   */
  @Get()
  async getUserCourses(@Request() req: AuthenticatedRequest) {
    return this.userCoursesService.getUserCourses(req.user.userId);
  }

  /**
   * POST /users/me/courses/:courseSlug/start
   * Start a new course or resume an existing one
   */
  @Post(':courseSlug/start')
  @HttpCode(HttpStatus.OK)
  async startCourse(@Request() req: AuthenticatedRequest, @Param('courseSlug') courseSlug: string) {
    return this.userCoursesService.startCourse(req.user.userId, courseSlug);
  }

  /**
   * PATCH /users/me/courses/:courseSlug/progress
   * Update progress for a course
   */
  @Patch(':courseSlug/progress')
  @HttpCode(HttpStatus.OK)
  async updateProgress(
    @Request() req: AuthenticatedRequest,
    @Param('courseSlug') courseSlug: string,
    @Body() dto: UpdateProgressDto
  ) {
    return this.userCoursesService.updateProgress(
      req.user.userId,
      courseSlug,
      dto.progress
    );
  }

  /**
   * PATCH /users/me/courses/:courseSlug/access
   * Update last accessed time for a course (moves it to top of My Tasks)
   */
  @Patch(':courseSlug/access')
  @HttpCode(HttpStatus.OK)
  async updateLastAccessed(
    @Request() req: AuthenticatedRequest,
    @Param('courseSlug') courseSlug: string
  ) {
    return this.userCoursesService.updateLastAccessed(
      req.user.userId,
      courseSlug
    );
  }
}
