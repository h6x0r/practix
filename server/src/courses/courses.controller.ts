import { Controller, Get, Param, UseGuards, Request } from '@nestjs/common';
import { CoursesService } from './courses.service';
import { OptionalJwtAuthGuard } from '../auth/guards/optional-jwt.guard';

@Controller('courses')
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
}