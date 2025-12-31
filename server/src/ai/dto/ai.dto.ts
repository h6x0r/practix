import { IsString, IsNotEmpty, IsOptional, MaxLength } from 'class-validator';

export class AskAiDto {
  @IsString()
  @IsNotEmpty()
  taskId: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(200, { message: 'Task title exceeds maximum length' })
  taskTitle: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(50000, { message: 'Code exceeds maximum size of 50KB' })
  userCode: string;

  @IsString()
  @IsNotEmpty()
  @MaxLength(2000, { message: 'Question exceeds maximum length of 2000 characters' })
  question: string;

  @IsString()
  @IsNotEmpty()
  language: string;

  @IsString()
  @IsOptional()
  uiLanguage?: string;
}
