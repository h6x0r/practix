import { IsString, IsNotEmpty, IsOptional, IsEmail, MinLength } from 'class-validator';

export class SubmitCodeDto {
  @IsString()
  @IsNotEmpty()
  taskId: string;

  @IsString()
  @IsNotEmpty()
  code: string;

  @IsString()
  @IsNotEmpty()
  language: string;
}

export class RunCodeDto {
  @IsString()
  @IsNotEmpty()
  code: string;

  @IsString()
  @IsNotEmpty()
  language: string;

  @IsString()
  @IsOptional()
  stdin?: string;
}

export class AskTutorDto {
  @IsString()
  @IsNotEmpty()
  taskTitle: string;

  @IsString()
  @IsNotEmpty()
  question: string;

  @IsString()
  @IsNotEmpty()
  userCode: string;

  @IsString()
  @IsNotEmpty()
  language: string;

  @IsString()
  @IsOptional()
  uiLanguage?: string;
}

export class LoginDto {
  @IsEmail()
  email: string;

  @IsNotEmpty()
  password: string;
}

export class RegisterDto {
  @IsNotEmpty()
  @IsString()
  name: string;

  @IsEmail()
  email: string;

  @IsNotEmpty()
  @MinLength(6)
  password: string;
}
