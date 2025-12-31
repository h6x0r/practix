import { Controller, Get, Post, Patch, Body, Param, Query, UseGuards, Request } from '@nestjs/common';
import { BugReportsService } from './bugreports.service';
import { CreateBugReportDto, UpdateBugStatusDto } from './dto/bugreports.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';

@Controller('bugreports')
export class BugReportsController {
  constructor(private readonly bugReportsService: BugReportsService) {}

  /**
   * Submit a new bug report
   * Requires authentication
   */
  @UseGuards(JwtAuthGuard)
  @Post()
  async create(@Request() req, @Body() createDto: CreateBugReportDto) {
    return this.bugReportsService.create(req.user.userId, createDto);
  }

  /**
   * Get current user's bug reports
   */
  @UseGuards(JwtAuthGuard)
  @Get('my')
  async findMy(@Request() req) {
    return this.bugReportsService.findByUser(req.user.userId);
  }

  /**
   * Get all bug reports (admin only)
   * Requires admin role
   */
  @UseGuards(JwtAuthGuard, AdminGuard)
  @Get()
  async findAll(
    @Query('status') status?: string,
    @Query('severity') severity?: string,
    @Query('category') category?: string,
  ) {
    return this.bugReportsService.findAll({ status, severity, category });
  }

  /**
   * Get a single bug report by ID
   */
  @UseGuards(JwtAuthGuard)
  @Get(':id')
  async findOne(@Param('id') id: string) {
    return this.bugReportsService.findOne(id);
  }

  /**
   * Update bug report status (admin only)
   * Uses UpdateBugStatusDto for validation
   */
  @UseGuards(JwtAuthGuard, AdminGuard)
  @Patch(':id/status')
  async updateStatus(
    @Param('id') id: string,
    @Body() updateDto: UpdateBugStatusDto,
  ) {
    return this.bugReportsService.updateStatus(id, updateDto.status);
  }
}
