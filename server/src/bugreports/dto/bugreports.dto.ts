import { IsString, IsNotEmpty, IsOptional, IsEnum } from 'class-validator';

export enum BugCategory {
  DESCRIPTION = 'description',
  SOLUTION = 'solution',
  EDITOR = 'editor',
  HINTS = 'hints',
  AI_TUTOR = 'ai-tutor',
  OTHER = 'other',
}

export enum BugSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

export enum BugStatus {
  OPEN = 'open',
  IN_PROGRESS = 'in-progress',
  RESOLVED = 'resolved',
  CLOSED = 'closed',
  WONT_FIX = 'wont-fix',
}

export class CreateBugReportDto {
  @IsString()
  @IsNotEmpty()
  title: string;

  @IsString()
  @IsNotEmpty()
  description: string;

  @IsEnum(BugCategory)
  category: BugCategory;

  @IsEnum(BugSeverity)
  @IsOptional()
  severity?: BugSeverity = BugSeverity.MEDIUM;

  @IsString()
  @IsOptional()
  taskId?: string;

  @IsOptional()
  metadata?: Record<string, unknown>;
}

export class UpdateBugStatusDto {
  @IsEnum(BugStatus)
  status: BugStatus;
}
