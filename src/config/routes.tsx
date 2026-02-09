import React, { Suspense, lazy } from "react";
import { AppRouteConfig } from "../types";

// Loading spinner for lazy-loaded components
const PageLoader = () => (
  <div className="min-h-screen bg-gray-900 flex items-center justify-center">
    <div className="flex flex-col items-center gap-4">
      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm font-mono">Loading...</span>
    </div>
  </div>
);

// Lazy wrapper helper
const lazyWithSuspense = (
  importFn: () => Promise<{ default: React.ComponentType }>,
) => {
  const LazyComponent = lazy(importFn);
  return (
    <Suspense fallback={<PageLoader />}>
      <LazyComponent />
    </Suspense>
  );
};

// Heavy pages - lazy loaded (Monaco Editor ~600KB, Recharts ~200KB)
const PlaygroundPage = lazyWithSuspense(
  () => import("../features/playground/ui/PlaygroundPage"),
);
const SharedSnippetPage = lazyWithSuspense(
  () => import("../features/playground/ui/SharedSnippetPage"),
);
const TaskWorkspace = lazyWithSuspense(
  () => import("../features/tasks/ui/TaskWorkspace"),
);
const AnalyticsPage = lazyWithSuspense(
  () => import("../features/analytics/ui/AnalyticsPage"),
);
const AdminDashboard = lazyWithSuspense(
  () => import("../features/admin/ui/AdminDashboard"),
);
const DashboardPage = lazyWithSuspense(
  () => import("../features/dashboard/ui/DashboardPage"),
);

// Light pages - regular imports
import CoursesPage from "../features/courses/ui/CoursesPage";
import CourseDetailsPage from "../features/courses/ui/CourseDetailsPage";
import PaymentsPage from "../features/payments/ui/PaymentsPage";
import PricingPage from "../features/payments/ui/PricingPage";
import SettingsPage from "../features/settings/ui/SettingsPage";
import MyTasksPage from "../features/my-tasks/ui/MyTasksPage";
import RoadmapPage from "../features/roadmap/ui/RoadmapPage";
import LeaderboardPage from "../features/gamification/ui/LeaderboardPage";
import AuthPage from "../features/auth/ui/AuthPage";

export const routes: AppRouteConfig[] = [
  // Auth Routes (No Layout)
  { path: "/login", element: <AuthPage />, layout: false },
  { path: "/register", element: <AuthPage />, layout: false },

  // Public/Hybrid Routes
  { path: "/", element: DashboardPage, layout: true },
  { path: "/courses", element: <CoursesPage />, layout: true },
  { path: "/course/:courseId", element: <CourseDetailsPage />, layout: true },
  { path: "/playground", element: PlaygroundPage, layout: true },
  { path: "/playground/:shortId", element: SharedSnippetPage, layout: true },
  { path: "/pricing", element: <PricingPage />, layout: true },

  // Semi-Protected Routes (Show preview with auth overlay)
  { path: "/my-tasks", element: <MyTasksPage />, layout: true },
  { path: "/roadmap", element: <RoadmapPage />, layout: true },
  { path: "/leaderboard", element: <LeaderboardPage />, layout: true },

  // Task Workspace
  {
    path: "/course/:courseId/task/:slug",
    element: TaskWorkspace,
    layout: true,
    protected: true,
  },
  {
    path: "/task/:slug",
    element: TaskWorkspace,
    layout: true,
    protected: true,
  },

  { path: "/premium", element: <PaymentsPage />, layout: true, hidden: true },
  { path: "/payments", element: <PaymentsPage />, layout: true, hidden: true },
  { path: "/analytics", element: AnalyticsPage, layout: true },
  { path: "/settings", element: <SettingsPage />, layout: true, hidden: true },

  // Admin Routes (Require Admin Role)
  { path: "/admin", element: AdminDashboard, layout: true, protected: true },
];
