import React, { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react';
import { subscriptionService } from '@/features/subscriptions/api/subscriptionService';
import { Subscription, TaskAccess, CourseAccess } from '@/features/subscriptions/model/types';
import { useAuth } from '@/features/auth/model/useAuth';

interface SubscriptionContextValue {
  subscriptions: Subscription[];
  isLoading: boolean;
  error: string | null;

  // Quick access checks
  hasGlobalAccess: boolean;
  hasCourseAccess: (courseId: string) => boolean;

  // Detailed access info
  getTaskAccess: (taskId: string) => Promise<TaskAccess>;
  getCourseAccess: (courseId: string) => Promise<CourseAccess>;

  // Refresh subscriptions
  refreshSubscriptions: () => Promise<void>;
}

const defaultTaskAccess: TaskAccess = {
  canView: true,
  canRun: true,
  canSubmit: true,
  canSeeSolution: false,
  canUseAiTutor: false,
  queuePriority: 10,
};

const defaultCourseAccess: CourseAccess = {
  hasAccess: false,
  queuePriority: 10,
  canUseAiTutor: false,
};

const SubscriptionContext = createContext<SubscriptionContextValue | undefined>(undefined);

interface SubscriptionProviderProps {
  children: ReactNode;
}

export function SubscriptionProvider({ children }: SubscriptionProviderProps) {
  const { isAuthenticated, user } = useAuth();
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cache for access checks - using ref to avoid dependency issues in callbacks
  const accessCacheRef = useRef<Map<string, TaskAccess | CourseAccess>>(new Map());

  // Fetch subscriptions when user is authenticated
  const refreshSubscriptions = useCallback(async () => {
    if (!isAuthenticated) {
      setSubscriptions([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const subs = await subscriptionService.getMySubscriptions();
      setSubscriptions(subs);
      // Clear access cache when subscriptions change
      accessCacheRef.current = new Map();
    } catch (err) {
      console.error('Failed to fetch subscriptions:', err);
      setError('Failed to load subscription information');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    refreshSubscriptions();
  }, [refreshSubscriptions]);

  // Check if user has global access
  const hasGlobalAccess = subscriptions.some(
    sub => sub.status === 'active' && sub.plan.type === 'global'
  );

  // Check if user has access to a specific course
  const hasCourseAccess = useCallback((courseId: string): boolean => {
    if (hasGlobalAccess) return true;

    return subscriptions.some(
      sub => sub.status === 'active' &&
             sub.plan.type === 'course' &&
             sub.plan.courseId === courseId
    );
  }, [subscriptions, hasGlobalAccess]);

  // Get task access with caching
  const getTaskAccess = useCallback(async (taskId: string): Promise<TaskAccess> => {
    if (!isAuthenticated) {
      return defaultTaskAccess;
    }

    const cacheKey = `task:${taskId}`;
    const cached = accessCacheRef.current.get(cacheKey);
    if (cached) {
      return cached as TaskAccess;
    }

    try {
      const access = await subscriptionService.getTaskAccess(taskId);
      accessCacheRef.current.set(cacheKey, access);
      return access;
    } catch (err) {
      console.error('Failed to fetch task access:', err);
      return defaultTaskAccess;
    }
  }, [isAuthenticated]);

  // Get course access with caching
  const getCourseAccess = useCallback(async (courseId: string): Promise<CourseAccess> => {
    if (!isAuthenticated) {
      return defaultCourseAccess;
    }

    const cacheKey = `course:${courseId}`;
    const cached = accessCacheRef.current.get(cacheKey);
    if (cached) {
      return cached as CourseAccess;
    }

    try {
      const access = await subscriptionService.getCourseAccess(courseId);
      accessCacheRef.current.set(cacheKey, access);
      return access;
    } catch (err) {
      console.error('Failed to fetch course access:', err);
      return defaultCourseAccess;
    }
  }, [isAuthenticated]);

  const value: SubscriptionContextValue = {
    subscriptions,
    isLoading,
    error,
    hasGlobalAccess,
    hasCourseAccess,
    getTaskAccess,
    getCourseAccess,
    refreshSubscriptions,
  };

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
}

export function useSubscription(): SubscriptionContextValue {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
}
