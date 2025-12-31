
import React from 'react';
import { AppRouteConfig } from '../types';

// Pages
import DashboardPage from '../features/dashboard/ui/DashboardPage';
import CoursesPage from '../features/courses/ui/CoursesPage';
import CourseDetailsPage from '../features/courses/ui/CourseDetailsPage';
import TaskWorkspace from '../features/tasks/ui/TaskWorkspace';
import PlaygroundPage from '../features/playground/ui/PlaygroundPage';
import PaymentsPage from '../features/payments/ui/PaymentsPage';
import SettingsPage from '../features/settings/ui/SettingsPage';
import AnalyticsPage from '../features/analytics/ui/AnalyticsPage';
import MyTasksPage from '../features/my-tasks/ui/MyTasksPage';
import RoadmapPage from '../features/roadmap/ui/RoadmapPage';
import LeaderboardPage from '../features/gamification/ui/LeaderboardPage';
import AuthPage from '../features/auth/ui/AuthPage';
import AdminDashboard from '../features/admin/ui/AdminDashboard';

export const routes: AppRouteConfig[] = [
  // Auth Routes (No Layout)
  { path: '/login', element: <AuthPage />, layout: false },
  { path: '/register', element: <AuthPage />, layout: false },

  // Public/Hybrid Routes
  { path: '/', element: <DashboardPage />, layout: true },
  { path: '/courses', element: <CoursesPage />, layout: true },
  { path: '/course/:courseId', element: <CourseDetailsPage />, layout: true },
  { path: '/playground', element: <PlaygroundPage />, layout: true },

  // Semi-Protected Routes (Show preview with auth overlay)
  { path: '/my-tasks', element: <MyTasksPage />, layout: true },
  { path: '/roadmap', element: <RoadmapPage />, layout: true },
  { path: '/leaderboard', element: <LeaderboardPage />, layout: true },
  
  // Task Workspace 
  { path: '/course/:courseId/task/:slug', element: <TaskWorkspace />, layout: true, protected: true },
  { path: '/task/:slug', element: <TaskWorkspace />, layout: true, protected: true },
  
  { path: '/premium', element: <PaymentsPage />, layout: true, hidden: true },
  { path: '/payments', element: <PaymentsPage />, layout: true, hidden: true },
  { path: '/analytics', element: <AnalyticsPage />, layout: true },
  { path: '/settings', element: <SettingsPage />, layout: true, hidden: true },

  // Admin Routes (Require Admin Role)
  { path: '/admin', element: <AdminDashboard />, layout: true, protected: true },
];
