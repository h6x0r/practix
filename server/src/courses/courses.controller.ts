import { Controller, Get, Post, Param, UseGuards, Request } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { CoursesService } from './courses.service';
import { OptionalJwtAuthGuard } from '../auth/guards/optional-jwt.guard';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';

@Controller('courses')
@Throttle({ default: { limit: 60, ttl: 60000 } }) // 60 requests per minute for public endpoints
export class CoursesController {
  constructor(private readonly coursesService: CoursesService) {}

  @UseGuards(OptionalJwtAuthGuard)
  @Get()
  findAll(@Request() req) {
    return this.coursesService.findAll(req.user?.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get(':id')
  findOne(@Param('id') id: string, @Request() req) {
    return this.coursesService.findOne(id, req.user?.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get(':id/structure')
  getStructure(@Param('id') id: string, @Request() req) {
    return this.coursesService.getStructure(id, req.user?.userId);
  }

  /**
   * Invalidate all course caches (Admin only)
   * Call this after updating course content via seed
   */
  @UseGuards(JwtAuthGuard, AdminGuard)
  @Post('cache/invalidate')
  invalidateCache() {
    return this.coursesService.invalidateCache();
  }
}