import { Controller, Get, Put, Body, UseGuards, Req } from "@nestjs/common";
import { Throttle } from "@nestjs/throttler";
import { ApiTags, ApiOperation, ApiBearerAuth } from "@nestjs/swagger";
import { SettingsService } from "./settings.service";
import {
  UpdateAiSettingsDto,
  AiSettingsResponseDto,
} from "./dto/settings.dto";
import { JwtAuthGuard } from "../../auth/guards/jwt-auth.guard";
import { AdminGuard } from "../../auth/guards/admin.guard";

interface RequestWithUser {
  user: { userId: string };
}

@ApiTags("Admin Settings")
@Controller("admin/settings")
@UseGuards(JwtAuthGuard, AdminGuard)
@ApiBearerAuth()
@Throttle({ default: { limit: 30, ttl: 60000 } })
export class SettingsController {
  constructor(private readonly settingsService: SettingsService) {}

  /**
   * GET /admin/settings/ai
   * Returns current AI settings
   */
  @Get("ai")
  @ApiOperation({ summary: "Get AI Tutor settings" })
  async getAiSettings(): Promise<AiSettingsResponseDto> {
    return this.settingsService.getAiSettings();
  }

  /**
   * PUT /admin/settings/ai
   * Update AI settings (enabled status and/or limits)
   */
  @Put("ai")
  @ApiOperation({ summary: "Update AI Tutor settings" })
  async updateAiSettings(
    @Body() dto: UpdateAiSettingsDto,
    @Req() req: RequestWithUser,
  ): Promise<AiSettingsResponseDto> {
    return this.settingsService.updateAiSettings(dto, req.user.userId);
  }
}
