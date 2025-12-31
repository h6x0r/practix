import React from 'react';

export enum AppRoute {
  DASHBOARD = '/',
  COURSES = '/courses',
  ROADMAP = '/roadmap',
  TASK = '/task',
  PROFILE = '/profile',
  PREMIUM = '/premium',
  ANALYTICS = '/analytics',
  SETTINGS = '/settings'
}

export interface NavItemConfig {
  label: string;
  path: string;
  iconKey: string; // String key mapping to an icon component (e.g. 'dashboard', 'book')
  translationKey: string; // Translation key for UI (e.g. 'nav.dashboard')
  roles?: string[]; // RBAC: which user roles can see this
  adminOnly?: boolean; // Only visible to admin users
  authRequired?: boolean; // Only visible to authenticated users
}

export interface AppRouteConfig {
  path: string;
  element: React.ReactNode;
  layout?: boolean; // Should wrap in Main Layout?
  protected?: boolean; // Requires Auth? (redirects to login)
  hidden?: boolean; // Show 404 for unauthenticated users (routes that "don't exist")
}
