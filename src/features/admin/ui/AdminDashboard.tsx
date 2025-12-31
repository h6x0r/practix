
import React, { useState, useEffect, useContext } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts';
import {
  adminService,
  DashboardStats,
  CourseAnalyticsItem,
  TaskAnalyticsItem,
  SubmissionStatusStat,
} from '../api/adminService';
import { AuthContext } from '@/components/Layout';
import { Link, Navigate } from 'react-router-dom';
import { createLogger } from '@/lib/logger';

const log = createLogger('AdminDashboard');

// Status colors for pie chart
const STATUS_COLORS: Record<string, string> = {
  'Accepted': '#22c55e',
  'Wrong Answer': '#ef4444',
  'Runtime Error': '#f97316',
  'Time Limit Exceeded': '#eab308',
  'Compilation Error': '#a855f7',
  'Pending': '#6b7280',
};

const AdminDashboard = () => {
  const { user } = useContext(AuthContext);
  const [loading, setLoading] = useState(true);

  // Data States
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [courseAnalytics, setCourseAnalytics] = useState<CourseAnalyticsItem[]>([]);
  const [hardestTasks, setHardestTasks] = useState<TaskAnalyticsItem[]>([]);
  const [mostPopularTasks, setMostPopularTasks] = useState<TaskAnalyticsItem[]>([]);
  const [submissionsByStatus, setSubmissionsByStatus] = useState<SubmissionStatusStat[]>([]);
  const [dailySubmissions, setDailySubmissions] = useState<{ date: string; count: number }[]>([]);
  const [totalSubmissions, setTotalSubmissions] = useState(0);

  // Check admin access
  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <div className="w-16 h-16 bg-gray-100 dark:bg-dark-surface rounded-2xl flex items-center justify-center mb-6">
          <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Login Required</h2>
        <p className="text-gray-500 dark:text-gray-400 max-w-sm mb-6">Please login to access the admin dashboard.</p>
        <Link to="/login" className="px-6 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5">
          Sign In
        </Link>
      </div>
    );
  }

  // Redirect if not admin
  if (user.role !== 'ADMIN') {
    return <Navigate to="/" replace />;
  }

  // Load all analytics data
  useEffect(() => {
    if (user && user.role === 'ADMIN') {
      setLoading(true);
      Promise.all([
        adminService.getDashboardStats(),
        adminService.getCourseAnalytics(),
        adminService.getTaskAnalytics(),
        adminService.getSubmissionStats(),
      ])
        .then(([stats, coursesResponse, tasksResponse, submissionsResponse]) => {
          setDashboardStats(stats);
          setCourseAnalytics(coursesResponse.courses || []);
          setHardestTasks(tasksResponse.hardestTasks || []);
          setMostPopularTasks(tasksResponse.mostPopularTasks || []);
          setSubmissionsByStatus(submissionsResponse.byStatus || []);
          setTotalSubmissions(submissionsResponse.totalSubmissions || 0);

          // Convert dailySubmissions object to array
          const dailyData = Object.entries(submissionsResponse.dailySubmissions || {})
            .map(([date, count]) => ({ date, count }))
            .sort((a, b) => a.date.localeCompare(b.date));
          setDailySubmissions(dailyData);

          setLoading(false);
        })
        .catch((error) => {
          log.error('Failed to load admin analytics', error);
          setLoading(false);
        });
    }
  }, [user]);

  if (loading) {
    return (
      <div className="p-10 text-center text-gray-500 animate-pulse">
        Loading Admin Analytics...
      </div>
    );
  }

  // Top 5 courses by completion rate
  const topCourses = [...courseAnalytics]
    .sort((a, b) => b.completionRate - a.completionRate)
    .slice(0, 5);

  return (
    <div className="max-w-[1600px] mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">Admin Analytics</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">Platform-wide statistics and insights.</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          {
            label: 'Total Users',
            value: dashboardStats?.totalUsers.toLocaleString() || '0',
            color: 'text-blue-500',
            bg: 'bg-blue-500/10',
            icon: (
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            )
          },
          {
            label: 'New Users (30d)',
            value: dashboardStats?.newUsers.toLocaleString() || '0',
            color: 'text-green-500',
            bg: 'bg-green-500/10',
            icon: (
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
              </svg>
            )
          },
          {
            label: 'Active Users (Monthly)',
            value: dashboardStats?.activeUsers.monthly.toLocaleString() || '0',
            color: 'text-brand-500',
            bg: 'bg-brand-500/10',
            icon: (
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            )
          },
          {
            label: 'Total Submissions',
            value: totalSubmissions.toLocaleString(),
            color: 'text-purple-500',
            bg: 'bg-purple-500/10',
            icon: (
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            )
          },
        ].map((stat, i) => (
          <div
            key={i}
            className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border flex items-center justify-between shadow-sm hover:shadow-md transition-shadow"
          >
            <div>
              <div className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">{stat.label}</div>
              <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">{stat.value}</div>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${stat.bg} ${stat.color}`}>
              {stat.icon}
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Submissions by Status (Pie Chart) */}
        <div className="bg-white dark:bg-dark-surface p-6 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Submissions by Status</h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={submissionsByStatus}
                  dataKey="count"
                  nameKey="status"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={(entry) => `${entry.status} (${entry.percentage}%)`}
                  labelLine={false}
                >
                  {submissionsByStatus.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={STATUS_COLORS[entry.status] || '#6366f1'}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    borderRadius: '12px',
                    border: '1px solid #27272a',
                    color: '#fff',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Submissions by Day (Line Chart) */}
        <div className="lg:col-span-2 bg-white dark:bg-dark-surface p-6 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Submissions by Day (Last 30 Days)</h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dailySubmissions}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
                <XAxis
                  dataKey="date"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#9CA3AF', fontSize: 12 }}
                  dy={10}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return `${date.getMonth() + 1}/${date.getDate()}`;
                  }}
                />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <Tooltip
                  cursor={{ stroke: '#FF6B35', strokeWidth: 2 }}
                  contentStyle={{
                    backgroundColor: '#18181b',
                    borderRadius: '12px',
                    border: '1px solid #27272a',
                    color: '#fff',
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="#FF6B35"
                  strokeWidth={3}
                  dot={{ fill: '#FF6B35', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Course Popularity (Bar Chart) */}
      <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Course Popularity (Enrollments)</h2>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={courseAnalytics}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
              <XAxis
                dataKey="courseTitle"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#9CA3AF', fontSize: 12 }}
                dy={10}
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                contentStyle={{
                  backgroundColor: '#18181b',
                  borderRadius: '12px',
                  border: '1px solid #27272a',
                  color: '#fff',
                }}
              />
              <Bar dataKey="totalEnrolled" fill="#FF6B35" radius={[6, 6, 0, 0]} barSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tables Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Top Courses by Completion */}
        <div className="bg-white dark:bg-dark-surface p-6 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Top Courses by Completion Rate</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-100 dark:border-dark-border">
                  <th className="text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Course</th>
                  <th className="text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Completions</th>
                  <th className="text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
                {topCourses.map((course) => (
                  <tr key={course.courseId} className="hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors">
                    <td className="py-3 text-sm text-gray-900 dark:text-white font-medium">{course.courseTitle}</td>
                    <td className="py-3 text-sm text-gray-600 dark:text-gray-300 text-right">{course.completed}</td>
                    <td className="py-3 text-right">
                      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                        {course.completionRate.toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Hardest Tasks */}
        <div className="bg-white dark:bg-dark-surface p-6 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Hardest Tasks (Lowest Pass Rate)</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-100 dark:border-dark-border">
                  <th className="text-left text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Task</th>
                  <th className="text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Submissions</th>
                  <th className="text-right text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider pb-3">Pass Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
                {hardestTasks.map((task) => (
                  <tr key={task.taskId} className="hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors">
                    <td className="py-3 text-sm text-gray-900 dark:text-white font-medium">{task.taskTitle}</td>
                    <td className="py-3 text-sm text-gray-600 dark:text-gray-300 text-right">{task.totalSubmissions}</td>
                    <td className="py-3 text-right">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${
                          task.passRate < 30
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                            : task.passRate < 60
                            ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
                            : 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                        }`}
                      >
                        {task.passRate.toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
