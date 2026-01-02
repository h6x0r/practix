import { Controller, Get, Param, UseGuards, Request } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { TasksService } from './tasks.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { OptionalJwtAuthGuard } from '../auth/guards/optional-jwt.guard';

@Controller('tasks')
@Throttle({ default: { limit: 60, ttl: 60000 } }) // 60 requests per minute for public endpoints
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @UseGuards(OptionalJwtAuthGuard)
  @Get()
  findAll(@Request() req) {
    // req.user might be undefined if not logged in
    return this.tasksService.findAll(req.user?.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get(':slug')
  findOne(@Param('slug') slug: string, @Request() req) {
    return this.tasksService.findOne(slug, req.user?.userId);
  }
}