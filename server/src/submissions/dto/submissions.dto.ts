import { IsString, IsNotEmpty, IsOptional, MaxLength } from 'class-validator';

// 50KB limit for code - industry standard (LeetCode uses 50KB, HackerRank 64KB)
const CODE_MAX_LENGTH = 50000;
const STDIN_MAX_LENGTH = 10000;

export class CreateSubmissionDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(CODE_MAX_LENGTH, { message: 'Code exceeds maximum size of 50KB' })
  code: string;

  @IsString()
  @IsNotEmpty()
  taskId: string;

  @IsString()
  @IsNotEmpty()
  language: string;
}

export class RunCodeDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(CODE_MAX_LENGTH, { message: 'Code exceeds maximum size of 50KB' })
  code: string;

  @IsString()
  @IsNotEmpty()
  language: string;

  @IsString()
  @IsOptional()
  @MaxLength(STDIN_MAX_LENGTH, { message: 'Input exceeds maximum size of 10KB' })
  stdin?: string;
}

export class RunTestsDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(CODE_MAX_LENGTH, { message: 'Code exceeds maximum size of 50KB' })
  code: string;

  @IsString()
  @IsNotEmpty()
  taskId: string;

  @IsString()
  @IsNotEmpty()
  language: string;
}

// 10KB limit for prompts (they're typically smaller than code)
const PROMPT_MAX_LENGTH = 10000;

export class SubmitPromptDto {
  @IsString()
  @IsNotEmpty()
  @MaxLength(PROMPT_MAX_LENGTH, { message: 'Prompt exceeds maximum size of 10KB' })
  prompt: string;

  @IsString()
  @IsNotEmpty()
  taskId: string;
}
