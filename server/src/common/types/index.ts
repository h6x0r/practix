import { Prisma } from '@prisma/client';

/**
 * User Preferences Interface
 */
export interface UserPreferences extends Record<string, any> {
  editorFontSize?: number;
  editorMinimap?: boolean;
  editorTheme?: string;
  editorLineNumbers?: boolean;
  notifications?: {
    emailDigest?: boolean;
    newCourses?: boolean;
  };
}

/**
 * User Plan Interface
 */
export interface UserPlan extends Record<string, any> {
  name: string;
  expiresAt: string;
}

/**
 * User Create Data Interface
 * NOTE: isPremium is always computed from active subscriptions, not stored.
 * New users start with isPremium=false (database default).
 */
export interface CreateUserData {
  email: string;
  name: string;
  password: string;
  preferences?: UserPreferences;
  avatarUrl?: string;
}

/**
 * JWT Payload Interface
 */
export interface JwtPayload {
  email: string;
  sub: string; // user id
}

/**
 * Request User Interface (from JWT strategy)
 */
export interface RequestUser {
  userId: string;
  email: string;
}

/**
 * HTTP Exception Response Interface
 */
export interface HttpExceptionResponse {
  statusCode?: number;
  message?: string | string[] | Record<string, any>;
  errors?: string[];
  error?: string;
}

/**
 * Course Topic Translation Interface
 */
export interface TopicTranslations {
  ru?: {
    title?: string;
    translations?: any;
  };
  uz?: {
    title?: string;
    translations?: any;
  };
  [key: string]: any;
}

/**
 * Cache Result Interface (generic)
 */
export interface CacheResult<T = any> {
  data: T;
  cached: boolean;
  timestamp: number;
}
