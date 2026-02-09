import {
  Controller,
  Get,
  Post,
  Delete,
  Body,
  Param,
  UseGuards,
  Request,
} from "@nestjs/common";
import { ApiTags, ApiOperation, ApiBearerAuth } from "@nestjs/swagger";
import { SnippetsService } from "./snippets.service";
import { CreateSnippetDto } from "./dto/create-snippet.dto";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { OptionalJwtAuthGuard } from "../auth/guards/optional-jwt.guard";

@ApiTags("snippets")
@Controller("snippets")
export class SnippetsController {
  constructor(private snippetsService: SnippetsService) {}

  @Post()
  @UseGuards(OptionalJwtAuthGuard)
  @ApiOperation({ summary: "Create a new code snippet" })
  async create(@Body() dto: CreateSnippetDto, @Request() req) {
    const userId = req.user?.id;
    return this.snippetsService.create(dto, userId);
  }

  @Get(":shortId")
  @ApiOperation({ summary: "Get a snippet by short ID (public)" })
  async findOne(@Param("shortId") shortId: string) {
    return this.snippetsService.findByShortId(shortId);
  }

  @Get("user/my")
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: "Get current user snippets" })
  async findMySnippets(@Request() req) {
    return this.snippetsService.findUserSnippets(req.user.id);
  }

  @Delete(":shortId")
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: "Delete a snippet" })
  async delete(@Param("shortId") shortId: string, @Request() req) {
    return this.snippetsService.delete(shortId, req.user.id);
  }
}
