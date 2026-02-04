import {
  IsBoolean,
  IsInt,
  IsOptional,
  Min,
  Max,
  ValidateNested,
} from "class-validator";
import { Type } from "class-transformer";
import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";

export class AiLimitsDto {
  @ApiPropertyOptional({ description: "Daily limit for free users", example: 5 })
  @IsOptional()
  @IsInt()
  @Min(0)
  @Max(1000)
  free?: number;

  @ApiPropertyOptional({
    description: "Daily limit for course subscribers",
    example: 30,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  @Max(1000)
  course?: number;

  @ApiPropertyOptional({
    description: "Daily limit for premium users",
    example: 100,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  @Max(1000)
  premium?: number;

  @ApiPropertyOptional({
    description: "Daily limit for Prompt Engineering course",
    example: 100,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  @Max(1000)
  promptEngineering?: number;
}

export class UpdateAiSettingsDto {
  @ApiPropertyOptional({ description: "Enable/disable AI Tutor globally" })
  @IsOptional()
  @IsBoolean()
  enabled?: boolean;

  @ApiPropertyOptional({ description: "Daily request limits by tier" })
  @IsOptional()
  @ValidateNested()
  @Type(() => AiLimitsDto)
  limits?: AiLimitsDto;
}

export class AiSettingsResponseDto {
  @ApiProperty({ description: "AI Tutor enabled status" })
  enabled: boolean;

  @ApiProperty({ description: "Daily request limits by tier" })
  limits: {
    free: number;
    course: number;
    premium: number;
    promptEngineering: number;
  };
}
