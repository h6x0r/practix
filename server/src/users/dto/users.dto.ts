import { IsString, IsOptional, IsNumber, IsBoolean, IsDateString, IsObject, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';

class NotificationsDto {
  @IsBoolean()
  @IsOptional()
  emailDigest?: boolean;

  @IsBoolean()
  @IsOptional()
  newCourses?: boolean;
}

export class UpdatePreferencesDto {
  @IsNumber()
  @IsOptional()
  editorFontSize?: number;

  @IsString()
  @IsOptional()
  editorFontFamily?: string;

  @IsString()
  @IsOptional()
  editorTheme?: string;

  @IsBoolean()
  @IsOptional()
  editorMinimap?: boolean;

  @IsBoolean()
  @IsOptional()
  editorLineNumbers?: boolean;

  @IsObject()
  @ValidateNested()
  @Type(() => NotificationsDto)
  @IsOptional()
  notifications?: NotificationsDto;
}

export class UpdatePlanDto {
  @IsString()
  name: string;

  @IsDateString()
  expiresAt: string;
}
