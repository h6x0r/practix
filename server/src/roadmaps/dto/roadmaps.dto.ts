import { IsString, IsNotEmpty, IsOptional, IsArray, IsNumber, IsIn } from 'class-validator';

// Legacy DTO - kept for backward compatibility
export class GenerateRoadmapDto {
  @IsString()
  @IsNotEmpty()
  role: string;

  @IsString()
  @IsNotEmpty()
  level: string;

  @IsString()
  @IsOptional()
  goal?: string;

  @IsArray()
  @IsString({ each: true })
  @IsOptional()
  preferredTopics?: string[];

  @IsNumber()
  @IsOptional()
  hoursPerWeek?: number;
}

// ============================================================================
// NEW: Extended Roadmap Generation DTO (v2)
// ============================================================================

export class GenerateRoadmapVariantsDto {
  // Step 1: Current knowledge
  @IsArray()
  @IsString({ each: true })
  knownLanguages: string[];

  @IsNumber()
  yearsOfExperience: number;

  // Step 2: Interests (multi-select)
  @IsArray()
  @IsString({ each: true })
  interests: string[];

  // Step 3: Goal
  @IsString()
  @IsIn(['first-job', 'senior', 'startup', 'master-skill'])
  goal: string;

  // Step 4: Time commitment
  @IsNumber()
  hoursPerWeek: number;

  @IsNumber()
  targetMonths: number;
}

export class SelectRoadmapVariantDto {
  @IsString()
  @IsNotEmpty()
  variantId: string;

  // Full variant data (sent from frontend after generation)
  @IsString()
  @IsNotEmpty()
  name: string;

  @IsString()
  description: string;

  @IsNumber()
  totalTasks: number;

  @IsNumber()
  estimatedHours: number;

  @IsNumber()
  estimatedMonths: number;

  @IsString()
  targetRole: string;

  @IsString()
  @IsIn(['easy', 'medium', 'hard'])
  difficulty: string;

  // Phases JSON - will be parsed on backend
  phases: any[];
}
