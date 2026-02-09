import {
  IsString,
  IsOptional,
  IsBoolean,
  MaxLength,
  IsIn,
  IsDateString,
} from 'class-validator';

const SUPPORTED_LANGUAGES = [
  'go',
  'java',
  'python',
  'typescript',
  'javascript',
  'c',
  'cpp',
  'rust',
];

export class CreateSnippetDto {
  @IsOptional()
  @IsString()
  @MaxLength(100)
  title?: string;

  @IsString()
  @MaxLength(50000) // 50KB max
  code: string;

  @IsString()
  @IsIn(SUPPORTED_LANGUAGES)
  language: string;

  @IsOptional()
  @IsBoolean()
  isPublic?: boolean;

  @IsOptional()
  @IsDateString()
  expiresAt?: string;
}
