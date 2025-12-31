import { IsString, IsOptional, IsBoolean, IsDateString } from 'class-validator';

export class CreateSubscriptionDto {
  @IsString()
  planId: string;

  @IsOptional()
  @IsBoolean()
  autoRenew?: boolean;
}

export class SubscriptionPlanDto {
  id: string;
  slug: string;
  name: string;
  nameRu?: string;
  type: 'global' | 'course';
  courseId?: string;
  priceMonthly: number;
  currency: string;
  isActive: boolean;
}

export class SubscriptionDto {
  id: string;
  userId: string;
  planId: string;
  plan: SubscriptionPlanDto;
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  startDate: string;
  endDate: string;
  autoRenew: boolean;
}

export class TaskAccessDto {
  canView: boolean;
  canRun: boolean;
  canSubmit: boolean;
  canSeeSolution: boolean;
  canUseAiTutor: boolean;
  queuePriority: number; // 1 = high (premium), 10 = low (free)
}

export class CourseAccessDto {
  hasAccess: boolean;
  queuePriority: number;
  canUseAiTutor: boolean;
}
