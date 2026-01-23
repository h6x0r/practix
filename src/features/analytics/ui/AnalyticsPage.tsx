
import React, { useState, useEffect, useContext, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { analyticsService, AnalyticsSummary, WeeklyStats, YearlyContribution } from '../api/analyticsService';
import { AuthContext } from '@/components/Layout';
import { AuthRequiredOverlay } from '@/components/AuthRequiredOverlay';
import { useUITranslation, useLanguage } from '@/contexts/LanguageContext';
import { createLogger } from '@/lib/logger';

const log = createLogger('Analytics');

// Day name keys for i18n
const DAY_KEYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'] as const;

const AnalyticsPage = () => {
  const { user } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const { language } = useLanguage();
  const [weekOffset, setWeekOffset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{date: string, count: number, x: number, y: number} | null>(null);

  // Data States
  const [weeklyData, setWeeklyData] = useState<WeeklyStats[]>([]);
  const [yearlyData, setYearlyData] = useState<YearlyContribution[]>([]);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [weekLoading, setWeekLoading] = useState(false);

  // Initial load - fetch all data
  useEffect(() => {
    if (user) {
        setLoading(true);
        Promise.all([
            analyticsService.getWeeklyStats(0),
            analyticsService.getYearlyContributions(),
            analyticsService.getSummary()
        ]).then(([week, year, stats]) => {
            setWeeklyData(week);
            setYearlyData(year);
            setSummary(stats);
            setLoading(false);
        }).catch((error) => {
            log.error('Failed to load analytics data', error);
            setLoading(false);
        });
    }
  }, [user]);

  // Update weekly data when offset changes (skip initial load)
  useEffect(() => {
    if (user && !loading) {
        setWeekLoading(true);
        analyticsService.getWeeklyStats(weekOffset)
          .then((data) => {
            setWeeklyData(data);
            setWeekLoading(false);
          })
          .catch((error) => {
            log.error('Failed to load weekly stats', error);
            setWeekLoading(false);
          });
    }
  }, [weekOffset]); // eslint-disable-line react-hooks/exhaustive-deps

  // Show auth overlay for unauthenticated users
  if (!user) {
    return (
      <AuthRequiredOverlay
        title={tUI('analytics.loginRequired')}
        description={tUI('analytics.loginRequiredDesc')}
      >
        <AnalyticsPreview tUI={tUI} />
      </AuthRequiredOverlay>
    );
  }

  // Get locale based on language
  const getLocale = () => {
    switch (language) {
      case 'ru': return 'ru-RU';
      case 'uz': return 'uz-UZ';
      default: return 'en-US';
    }
  };

  // Helper to calculate date range string for calendar week (Monday-based)
  // When weekOffset = 0, we show the CURRENT calendar week (Mon-Sun containing today)
  const getDateRangeLabel = (offset: number) => {
    const today = new Date();

    // Calculate Monday of the CURRENT week first
    const dayOfWeek = today.getDay(); // 0 = Sunday, 1 = Monday, etc.
    const diffToMonday = dayOfWeek === 0 ? -6 : 1 - dayOfWeek; // Days to subtract to get to Monday

    const currentMonday = new Date(today);
    currentMonday.setDate(today.getDate() + diffToMonday);
    currentMonday.setHours(0, 0, 0, 0);

    // Now go back 'offset' weeks from current Monday
    const targetMonday = new Date(currentMonday);
    targetMonday.setDate(currentMonday.getDate() - (offset * 7));

    // Calculate Sunday of that week
    const targetSunday = new Date(targetMonday);
    targetSunday.setDate(targetMonday.getDate() + 6);

    const locale = getLocale();
    const fmt = (d: Date) => d.toLocaleDateString(locale, { month: 'short', day: 'numeric' });

    return `${fmt(targetMonday)} - ${fmt(targetSunday)}`;
  };

  const formatHeatmapDate = (isoDate: string) => {
    const locale = getLocale();
    return new Date(isoDate).toLocaleDateString(locale, { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const handlePrevWeek = () => setWeekOffset(prev => prev + 1);
  const handleNextWeek = () => setWeekOffset(prev => Math.max(0, prev - 1));

  const handleCellHover = (e: React.MouseEvent, day: {date: string, count: number}) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoveredCell({
      date: day.date,
      count: day.count,
      x: rect.left + rect.width / 2,
      y: rect.top - 8 // Offset above cell
    });
  };

  const handleCellLeave = () => {
    setHoveredCell(null);
  };

  // Localize day names in weekly data and include date
  const weeklyDataLocalized = useMemo(() => {
    return weeklyData.map((d, i) => {
      // Extract day of month from the date (if available)
      const dayOfMonth = d.date ? new Date(d.date).getDate() : '';
      const dayName = tUI(`analytics.day.${DAY_KEYS[i % 7]}`);
      // Compact format: "Пн 6" or "Mon 6"
      const label = dayOfMonth ? `${dayName} ${dayOfMonth}` : dayName;

      return {
        ...d,
        name: label,
        Tasks: d.tasks // Use capitalized "Tasks" for tooltip
      };
    });
  }, [weeklyData, tUI]);

  // Calculate actual weekly total from displayed data
  const weeklyTotal = useMemo(() => {
    return weeklyData.reduce((sum, day) => sum + day.tasks, 0);
  }, [weeklyData]);

  if (loading) {
      return <div className="p-10 text-center text-gray-500 animate-pulse">{tUI('analytics.loading')}</div>;
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('analytics.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">{tUI('analytics.subtitle')}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Weekly Activity Bar Chart (with Navigation) */}
        <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <div className="flex justify-between items-center mb-6">
             <h2 className="text-xl font-bold text-gray-900 dark:text-white">{tUI('analytics.weeklyActivity')}</h2>

             {/* Simple Navigation */}
             <div className="flex items-center gap-3 bg-gray-50 dark:bg-dark-bg p-1 rounded-lg">
                <button
                  onClick={handlePrevWeek}
                  disabled={weekLoading}
                  className={`p-1 rounded shadow-sm text-gray-500 ${weekLoading ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white dark:hover:bg-dark-surface'}`}
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                </button>
                <span className="text-xs font-bold text-gray-600 dark:text-gray-300 min-w-[100px] text-center select-none">
                  {weekLoading ? '...' : getDateRangeLabel(weekOffset)}
                </span>
                <button
                  onClick={handleNextWeek}
                  disabled={weekOffset === 0 || weekLoading}
                  className={`p-1 rounded shadow-sm text-gray-500 ${weekOffset === 0 || weekLoading ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white dark:hover:bg-dark-surface'}`}
                >
                   <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                </button>
             </div>
          </div>

          <div className="h-80 w-full relative">
            {weekLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-dark-surface/50 z-10">
                <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
              </div>
            )}
            {weeklyDataLocalized.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weeklyDataLocalized}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} dy={10} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                  <Tooltip
                    cursor={{fill: 'transparent'}}
                    contentStyle={{backgroundColor: '#18181b', borderRadius: '12px', border: '1px solid #27272a', color: '#fff'}}
                  />
                  <Bar dataKey="Tasks" fill="#0ea5e9" radius={[6, 6, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            ) : !weekLoading && (
              <div className="h-full flex flex-col items-center justify-center text-gray-400">
                <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <p className="text-sm font-medium">{tUI('analytics.noActivityThisWeek')}</p>
              </div>
            )}
          </div>
          <div className="mt-4 text-center text-sm text-gray-500">
             {weekOffset === 0
               ? <span>{tUI('analytics.youAreOnFire')} <span className="text-brand-500 font-bold">{tUI('analytics.tasksSolvedThisWeek').replace('{count}', String(weeklyTotal))}</span></span>
               : <span>{tUI('analytics.historicalData').replace('{dateRange}', getDateRangeLabel(weekOffset))}</span>
             }
          </div>
        </div>

        {/* Contribution Heatmap (GitHub Style) */}
        <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm flex flex-col relative">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">{tUI('analytics.yearlyContributions')}</h2>
          <div className="flex-1 flex flex-col justify-center">
            {(() => {
              // Transform flat array into weeks structure: 7 rows (days) x 53 columns (weeks)
              const weeks: YearlyContribution[][] = [];
              const numWeeks = Math.ceil(yearlyData.length / 7);

              for (let week = 0; week < numWeeks; week++) {
                const weekData: YearlyContribution[] = [];
                for (let day = 0; day < 7; day++) {
                  const index = week * 7 + day;
                  if (index < yearlyData.length) {
                    weekData.push(yearlyData[index]);
                  }
                }
                if (weekData.length > 0) {
                  weeks.push(weekData);
                }
              }

              // Calculate month labels for each week column
              const monthLabels: { month: string; startCol: number }[] = [];
              let currentMonth = -1;

              const locale = getLocale();
              weeks.forEach((week, weekIndex) => {
                if (week.length > 0) {
                  const firstDayOfWeek = new Date(week[0].date);
                  const month = firstDayOfWeek.getMonth();

                  // Only add label if it's a new month and not the very first column (to avoid cramping)
                  if (month !== currentMonth && weekIndex > 0) {
                    monthLabels.push({
                      month: firstDayOfWeek.toLocaleDateString(locale, { month: 'short' }),
                      startCol: weekIndex
                    });
                    currentMonth = month;
                  } else if (weekIndex === 0) {
                    // Always show first month
                    monthLabels.push({
                      month: firstDayOfWeek.toLocaleDateString(locale, { month: 'short' }),
                      startCol: 0
                    });
                    currentMonth = month;
                  }
                }
              });

              return (
                <div className="space-y-2">
                  {/* Month labels */}
                  <div className="grid gap-1 mb-1" style={{ gridTemplateColumns: `repeat(${weeks.length}, 1fr)` }}>
                    {weeks.map((_, weekIndex) => {
                      const label = monthLabels.find(m => m.startCol === weekIndex);
                      return (
                        <div key={weekIndex} className="text-[10px] text-gray-400 font-medium">
                          {label?.month || ''}
                        </div>
                      );
                    })}
                  </div>

                  {/* Grid: 7 rows (days) */}
                  {[0, 1, 2, 3, 4, 5, 6].map((dayIndex) => (
                    <div
                      key={dayIndex}
                      className="grid gap-1"
                      style={{ gridTemplateColumns: `repeat(${weeks.length}, 1fr)` }}
                    >
                      {weeks.map((week, weekIndex) => {
                        const day = week[dayIndex];
                        if (!day) {
                          return <div key={weekIndex} className="w-full aspect-square" />;
                        }

                        return (
                          <div
                            key={weekIndex}
                            onMouseEnter={(e) => handleCellHover(e, day)}
                            onMouseLeave={handleCellLeave}
                            className={`w-full aspect-square rounded-[2px] cursor-pointer transition-colors ${
                              day.intensity === 0 ? 'bg-gray-200 dark:bg-gray-700/60 border border-gray-300 dark:border-gray-600' :
                              day.intensity === 1 ? 'bg-green-200 dark:bg-green-800/70' :
                              day.intensity === 2 ? 'bg-green-400 dark:bg-green-600' :
                              day.intensity === 3 ? 'bg-green-500 dark:bg-green-500' :
                              'bg-green-600 dark:bg-green-400'
                            } hover:ring-2 ring-gray-400 dark:ring-gray-400`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              );
            })()}

            <div className="flex justify-end items-center gap-2 mt-4 text-xs text-gray-400">
              <span>{tUI('analytics.less')}</span>
              <div className="w-3 h-3 bg-gray-200 dark:bg-gray-700/60 border border-gray-300 dark:border-gray-600 rounded-[2px]"></div>
              <div className="w-3 h-3 bg-green-200 dark:bg-green-800/70 rounded-[2px]"></div>
              <div className="w-3 h-3 bg-green-400 dark:bg-green-600 rounded-[2px]"></div>
              <div className="w-3 h-3 bg-green-600 dark:bg-green-400 rounded-[2px]"></div>
              <span>{tUI('analytics.more')}</span>
            </div>

            <div className="mt-8 pt-6 border-t border-gray-100 dark:border-dark-border grid grid-cols-3 text-center">
               <div>
                 <div className="text-2xl font-bold text-gray-900 dark:text-white">
                   {summary?.totalSubmissions?.toLocaleString() || '0'}
                 </div>
                 <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.totalSubmissions')}</div>
               </div>
               <div>
                 <div className="text-2xl font-bold text-gray-900 dark:text-white">
                   {summary?.maxStreak || 0}
                 </div>
                 <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.maxStreak')}</div>
               </div>
               <div>
                 <div className="text-2xl font-bold text-gray-900 dark:text-white">
                   {summary?.currentStreak || 0}
                 </div>
                 <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.currentStreak')}</div>
               </div>
            </div>
          </div>

          {/* Floating Tooltip */}
          {hoveredCell && (
            <div
              className="fixed z-50 bg-gray-900 text-white text-xs px-3 py-2 rounded-lg shadow-xl pointer-events-none transform -translate-x-1/2 -translate-y-full whitespace-nowrap"
              style={{ left: hoveredCell.x, top: hoveredCell.y }}
            >
               <div className="font-bold">{hoveredCell.count === 0 ? tUI('analytics.noActivity') : tUI('analytics.tasksCompleted').replace('{count}', String(hoveredCell.count))}</div>
               <div className="text-gray-400 text-[10px] uppercase font-bold mt-0.5">{formatHeatmapDate(hoveredCell.date)}</div>
               {/* Arrow */}
               <div className="absolute left-1/2 -bottom-1 w-2 h-2 bg-gray-900 transform -translate-x-1/2 rotate-45"></div>
            </div>
          )}

        </div>

      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          {
            label: tUI('analytics.completionRate'),
            val: `${summary?.completionRate || 0}%`,
            color: 'text-green-500',
            bg: 'bg-green-500/10'
          },
          {
            label: tUI('analytics.avgRuntime'),
            val: summary?.avgRuntime || '~15ms',
            color: 'text-purple-500',
            bg: 'bg-purple-500/10'
          },
          {
            label: tUI('analytics.totalXP'),
            val: summary?.totalXP?.toLocaleString() || '0',
            color: 'text-amber-500',
            bg: 'bg-amber-500/10'
          },
        ].map((s, i) => (
          <div key={i} className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border flex items-center justify-between">
            <div>
              <div className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">{s.label}</div>
              <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">{s.val}</div>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${s.bg} ${s.color}`}>
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Mock preview component for unauthenticated users
const AnalyticsPreview = ({ tUI }: { tUI: (key: string) => string }) => {
  const mockWeeklyData = [
    { name: tUI('analytics.day.mon'), Tasks: 4 },
    { name: tUI('analytics.day.tue'), Tasks: 6 },
    { name: tUI('analytics.day.wed'), Tasks: 3 },
    { name: tUI('analytics.day.thu'), Tasks: 8 },
    { name: tUI('analytics.day.fri'), Tasks: 5 },
    { name: tUI('analytics.day.sat'), Tasks: 7 },
    { name: tUI('analytics.day.sun'), Tasks: 4 }
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('analytics.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">{tUI('analytics.subtitle')}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Weekly Activity Bar Chart */}
        <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">{tUI('analytics.weeklyActivity')}</h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockWeeklyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <Bar dataKey="Tasks" fill="#0ea5e9" radius={[6, 6, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Contribution Heatmap Placeholder */}
        <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-100 dark:border-dark-border shadow-sm">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">{tUI('analytics.yearlyContributions')}</h2>
          <div className="grid grid-cols-12 gap-1">
            {Array.from({ length: 84 }).map((_, i) => (
              <div
                key={i}
                className={`aspect-square rounded-sm ${
                  Math.random() > 0.6
                    ? ['bg-green-200 dark:bg-green-900/40', 'bg-green-400 dark:bg-green-700', 'bg-green-600 dark:bg-green-500'][Math.floor(Math.random() * 3)]
                    : 'bg-gray-100 dark:bg-dark-bg/50'
                }`}
              />
            ))}
          </div>
          <div className="mt-6 pt-6 border-t border-gray-100 dark:border-dark-border grid grid-cols-3 text-center">
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">156</div>
              <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.totalSubmissions')}</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">14</div>
              <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.maxStreak')}</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">7</div>
              <div className="text-xs text-gray-400 uppercase font-bold mt-1">{tUI('analytics.currentStreak')}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { label: tUI('analytics.completionRate'), val: '78%', color: 'text-green-500', bg: 'bg-green-500/10' },
          { label: tUI('analytics.avgRuntime'), val: '~15ms', color: 'text-purple-500', bg: 'bg-purple-500/10' },
          { label: tUI('analytics.totalXP'), val: '2,450', color: 'text-amber-500', bg: 'bg-amber-500/10' },
        ].map((s, i) => (
          <div key={i} className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border flex items-center justify-between">
            <div>
              <div className="text-gray-500 dark:text-gray-400 text-sm font-medium mb-1">{s.label}</div>
              <div className="text-3xl font-display font-bold text-gray-900 dark:text-white">{s.val}</div>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${s.bg} ${s.color}`}>
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnalyticsPage;
