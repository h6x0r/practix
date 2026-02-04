import React, { useState, useEffect, useCallback } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import {
  adminService,
  DashboardStats,
  AnalyticsTimelineResponse,
} from "../api/adminService";
import { useUITranslation } from "@/contexts/LanguageContext";
import { createLogger } from "@/lib/logger";

const log = createLogger("AnalyticsPanel");

interface AnalyticsPanelProps {
  dashboardStats: DashboardStats | null;
}

const AnalyticsPanel = ({ dashboardStats }: AnalyticsPanelProps) => {
  const { tUI } = useUITranslation();
  const [timelineData, setTimelineData] =
    useState<AnalyticsTimelineResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState<7 | 14 | 30 | 90>(30);

  const loadTimeline = useCallback(async () => {
    setLoading(true);
    try {
      const data = await adminService.getAnalyticsTimeline(period);
      setTimelineData(data);
    } catch (error) {
      log.error("Failed to load analytics timeline", error);
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    loadTimeline();
  }, [loadTimeline]);

  // Format currency (UZS in tiyn to readable format)
  const formatRevenue = (amount: number) => {
    const uzs = amount / 100;
    if (uzs >= 1000000) {
      return `${(uzs / 1000000).toFixed(1)}M`;
    }
    if (uzs >= 1000) {
      return `${(uzs / 1000).toFixed(0)}K`;
    }
    return uzs.toLocaleString();
  };

  // Format date for chart
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  return (
    <div className="space-y-6">
      {/* DAU/WAU/MAU Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* DAU */}
        <div className="bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-100 dark:border-dark-border">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              {tUI("admin.dau") || "Daily Active Users"}
            </h3>
            <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
              <svg
                className="w-4 h-4 text-blue-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">
            {dashboardStats?.activeUsers.daily.toLocaleString() || "0"}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {tUI("admin.last24h") || "Last 24 hours"}
          </div>
        </div>

        {/* WAU */}
        <div className="bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-100 dark:border-dark-border">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              {tUI("admin.wau") || "Weekly Active Users"}
            </h3>
            <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center">
              <svg
                className="w-4 h-4 text-purple-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">
            {dashboardStats?.activeUsers.weekly.toLocaleString() || "0"}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {tUI("admin.last7d") || "Last 7 days"}
          </div>
        </div>

        {/* MAU */}
        <div className="bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-100 dark:border-dark-border">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              {tUI("admin.mau") || "Monthly Active Users"}
            </h3>
            <div className="w-8 h-8 rounded-full bg-brand-500/10 flex items-center justify-center">
              <svg
                className="w-4 h-4 text-brand-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                />
              </svg>
            </div>
          </div>
          <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">
            {dashboardStats?.activeUsers.monthly.toLocaleString() || "0"}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {tUI("admin.last30d") || "Last 30 days"}
          </div>
        </div>
      </div>

      {/* Timeline Charts */}
      <div className="bg-white dark:bg-dark-surface rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm overflow-hidden">
        {/* Header with period selector */}
        <div className="flex items-center justify-between p-6 border-b border-gray-100 dark:border-dark-border">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            {tUI("admin.analyticsTimeline") || "Analytics Timeline"}
          </h2>
          <div className="flex gap-2">
            {[7, 14, 30, 90].map((days) => (
              <button
                key={days}
                onClick={() => setPeriod(days as 7 | 14 | 30 | 90)}
                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-colors ${
                  period === days
                    ? "bg-brand-500 text-white"
                    : "bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-dark-border"
                }`}
              >
                {days}d
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="h-80 flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : timelineData ? (
          <div className="p-6 space-y-8">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
                <div className="text-2xl font-bold text-blue-500">
                  {timelineData.summary.avgDau}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {tUI("admin.avgDau") || "Avg DAU"}
                </div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
                <div className="text-2xl font-bold text-green-500">
                  {timelineData.summary.totalNewUsers}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {tUI("admin.newUsers") || "New Users"}
                </div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
                <div className="text-2xl font-bold text-purple-500">
                  {timelineData.summary.totalNewSubscriptions}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {tUI("admin.newSubs") || "New Subs"}
                </div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
                <div className="text-2xl font-bold text-amber-500">
                  {timelineData.summary.totalPayments}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {tUI("admin.payments") || "Payments"}
                </div>
              </div>
              <div className="text-center p-3 bg-gradient-to-br from-brand-500/10 to-purple-500/10 rounded-xl">
                <div className="text-2xl font-bold text-brand-600">
                  {formatRevenue(timelineData.summary.totalRevenue)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {tUI("admin.totalRevenue") || "Revenue (UZS)"}
                </div>
              </div>
            </div>

            {/* DAU Chart */}
            <div>
              <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-4">
                {tUI("admin.dauChart") || "Daily Active Users"}
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={timelineData.timeline}>
                    <defs>
                      <linearGradient id="dauGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#3f3f46"
                      strokeOpacity={0.1}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                      tickFormatter={formatDate}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#18181b",
                        borderRadius: "12px",
                        border: "1px solid #27272a",
                        color: "#fff",
                      }}
                      labelFormatter={(value) =>
                        new Date(value).toLocaleDateString()
                      }
                    />
                    <Area
                      type="monotone"
                      dataKey="dau"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      fill="url(#dauGradient)"
                      name="DAU"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Revenue & Payments Chart */}
            <div>
              <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-4">
                {tUI("admin.revenueChart") || "Revenue & Payments"}
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={timelineData.timeline}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#3f3f46"
                      strokeOpacity={0.1}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                      tickFormatter={formatDate}
                    />
                    <YAxis
                      yAxisId="revenue"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                      tickFormatter={(value) => formatRevenue(value)}
                    />
                    <YAxis
                      yAxisId="payments"
                      orientation="right"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#18181b",
                        borderRadius: "12px",
                        border: "1px solid #27272a",
                        color: "#fff",
                      }}
                      labelFormatter={(value) =>
                        new Date(value).toLocaleDateString()
                      }
                      formatter={(value: number, name: string) => {
                        if (name === "revenue") {
                          return [`${formatRevenue(value)} UZS`, "Revenue"];
                        }
                        return [value, name];
                      }}
                    />
                    <Bar
                      yAxisId="revenue"
                      dataKey="revenue"
                      fill="#FF6B35"
                      radius={[4, 4, 0, 0]}
                      name="revenue"
                    />
                    <Line
                      yAxisId="payments"
                      type="monotone"
                      dataKey="payments"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      dot={{ fill: "#8B5CF6", r: 3 }}
                      name="Payments"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* New Users & Subscriptions Chart */}
            <div>
              <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-4">
                {tUI("admin.growthChart") || "User Growth"}
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timelineData.timeline}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#3f3f46"
                      strokeOpacity={0.1}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                      tickFormatter={formatDate}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#9CA3AF", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#18181b",
                        borderRadius: "12px",
                        border: "1px solid #27272a",
                        color: "#fff",
                      }}
                      labelFormatter={(value) =>
                        new Date(value).toLocaleDateString()
                      }
                    />
                    <Line
                      type="monotone"
                      dataKey="newUsers"
                      stroke="#22C55E"
                      strokeWidth={2}
                      dot={{ fill: "#22C55E", r: 3 }}
                      name="New Users"
                    />
                    <Line
                      type="monotone"
                      dataKey="newSubscriptions"
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      dot={{ fill: "#8B5CF6", r: 3 }}
                      name="New Subscriptions"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-80 flex items-center justify-center text-gray-500">
            {tUI("admin.noData") || "No data available"}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsPanel;
