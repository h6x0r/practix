import React, { useContext, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AuthContext } from '@/components/Layout';
import { AuthRequiredOverlay } from '@/components/AuthRequiredOverlay';
import { taskService } from '@/features/tasks/api/taskService';
import { dashboardService, UserStats, DayActivity } from '../api/dashboardService';
import { IconCheckCircle, IconClock, IconPlay, IconChart } from '@/components/Icons';
import { Submission } from '@/types';
import { useUITranslation } from '@/contexts/LanguageContext';
import { createLogger } from '@/lib/logger';

const log = createLogger('Dashboard');

const DashboardPage = () => {
  const { user } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const [recentSubmissions, setRecentSubmissions] = useState<Submission[]>([]);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [chartData, setChartData] = useState<DayActivity[]>([]);
  const [chartDays, setChartDays] = useState<number>(7);
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);

  // Load Data via Services
  useEffect(() => {
    if (user) {
        setLoading(true);
        Promise.all([
            taskService.getRecentSubmissions(15), // Last 15 submissions
            dashboardService.getStats(),
            dashboardService.getWeeklyActivity(chartDays)
        ]).then(([submissionsData, statsData, activityData]) => {
            setRecentSubmissions(submissionsData);
            setUserStats(statsData);
            setChartData(activityData);
            setLoading(false);
        }).catch((error) => {
            log.error('Failed to load dashboard data', error);
            setLoading(false);
        });
    }
  }, [user]);

  // Handle chart period change
  const handleChartPeriodChange = async (days: number) => {
    setChartDays(days);
    setChartLoading(true);
    try {
      const activityData = await dashboardService.getWeeklyActivity(days);
      setChartData(activityData);
    } catch (error) {
      log.error('Failed to load chart data', error);
    } finally {
      setChartLoading(false);
    }
  };

  if (!user) {
    return (
      <AuthRequiredOverlay
        title={tUI('dashboard.loginRequired') || 'Sign in to view your dashboard'}
        description={tUI('dashboard.loginRequiredDesc') || 'Track your progress, see your stats, and continue learning'}
      >
        <DashboardPreview tUI={tUI} />
      </AuthRequiredOverlay>
    );
  }

  // Split submissions by status
  const pendingSubmissions = recentSubmissions.filter(s => s.status !== 'passed');
  const passedSubmissions = recentSubmissions.filter(s => s.status === 'passed');

  if (loading) {
      return <div className="p-10 text-center text-gray-500 animate-pulse">{tUI('common.loading')}</div>;
  }

  // Format skill points (e.g. 3200 -> 3.2k)
  const formatSkillPoints = (points: number) => {
    if (points >= 1000) {
      return `${(points / 1000).toFixed(1)}k`;
    }
    return points.toString();
  };

  // Format time spent (minutes -> hours or minutes with i18n)
  const formatTimeSpent = (totalMinutes: number) => {
    const hours = Math.floor(totalMinutes / 60);
    if (hours > 0) {
      return `${hours}${tUI('dashboard.hours')}`;
    }
    return `${totalMinutes}${tUI('dashboard.minutes')}`;
  };

  const stats = [
    {
      label: tUI('dashboard.totalSolved'),
      val: userStats?.totalSolved?.toString() || '0',
      sub: `+${userStats?.weekThisWeek || 0} ${tUI('dashboard.thisWeek')}`,
      color: 'text-brand-500',
      bg: 'bg-brand-500/10'
    },
    {
      label: tUI('dashboard.hoursSpent'),
      val: formatTimeSpent(userStats?.totalMinutes || 0),
      sub: `${userStats?.totalSubmissions || 0} ${tUI('dashboard.submissions')}`,
      color: 'text-purple-500',
      bg: 'bg-purple-500/10'
    },
    {
      label: tUI('dashboard.globalRank'),
      val: userStats?.globalRank ? `#${userStats.globalRank}` : '-',
      sub: `${tUI('dashboard.topPercent')} ${userStats?.topPercent || 0}%`,
      color: 'text-emerald-500',
      bg: 'bg-emerald-500/10'
    },
    {
      label: tUI('dashboard.skillPoints'),
      val: formatSkillPoints(userStats?.skillPoints || 0),
      sub: `${tUI('dashboard.maxStreak')}: ${userStats?.maxStreak || 0}`,
      color: 'text-amber-500',
      bg: 'bg-amber-500/10'
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div className="flex justify-between items-end">
        <div>
           <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('dashboard.pageTitle')}</h1>
           <p className="text-gray-500 dark:text-gray-400 mt-1">{tUI('dashboard.pageSubtitle')}</p>
        </div>
        <div className="text-right hidden md:block">
           <div className="text-sm font-bold text-gray-400 uppercase tracking-wider">{tUI('dashboard.currentStreak')}</div>
           <div className="text-3xl font-display font-bold text-orange-500">
             🔥 {userStats?.currentStreak || 0} {tUI('dashboard.days')}
           </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-2">
              <div className={`p-2 rounded-lg ${stat.bg} ${stat.color}`}>
                <IconChart className="w-5 h-5" />
              </div>
            </div>
            <div className="text-3xl font-display font-bold text-gray-900 dark:text-white mb-1">{stat.val}</div>
            <div className="flex justify-between items-center">
              <div className="text-xs font-bold text-gray-400 uppercase">{stat.label}</div>
              <div className="text-xs font-medium text-green-500">{stat.sub}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart */}
        <div className="lg:col-span-2 bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">{tUI('dashboard.activityOverview')}</h2>
            <select
              value={chartDays}
              onChange={(e) => handleChartPeriodChange(Number(e.target.value))}
              disabled={chartLoading}
              className="bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border text-xs rounded-lg px-2 py-1 outline-none dark:text-gray-300 disabled:opacity-50"
            >
              <option value={7}>{tUI('dashboard.last7Days')}</option>
              <option value={30}>{tUI('dashboard.last30Days')}</option>
            </select>
          </div>
          <div className="h-72 w-full relative">
            {chartLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-dark-surface/50 z-10">
                <div className="text-gray-500 animate-pulse">{tUI('common.loading')}</div>
              </div>
            )}
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorSolved" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#9CA3AF', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#9CA3AF', fontSize: 12}} />
                <Tooltip 
                  contentStyle={{backgroundColor: '#18181b', borderRadius: '12px', border: '1px solid #27272a', color: '#fff'}}
                  itemStyle={{color: '#fff'}}
                />
                <Area type="monotone" dataKey="solved" stroke="#0ea5e9" strokeWidth={3} fillOpacity={1} fill="url(#colorSolved)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Task Lists */}
        <div data-testid="my-courses-section" className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm flex flex-col overflow-hidden h-[380px]">
          <div className="p-4 border-b border-gray-100 dark:border-dark-border bg-gray-50/50 dark:bg-dark-surface flex-shrink-0">
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">{tUI('dashboard.recentActivity')}</h2>
          </div>
          <div className="overflow-y-auto p-4 space-y-4">

            {/* Pending/Failed Submissions */}
            {pendingSubmissions.length > 0 && (
              <div>
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <IconClock className="w-3 h-3" /> {tUI('dashboard.pending')}
                </h3>
                <div className="space-y-2">
                  {pendingSubmissions.slice(0, 5).map(sub => (
                    <Link to={`/task/${sub.task?.slug || sub.taskId}`} key={sub.id} className="group flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-dark-bg border border-gray-100 dark:border-dark-border hover:border-brand-300 transition-colors cursor-pointer">
                       <div className="flex items-center gap-3">
                         <div className={`w-2 h-2 rounded-full ${sub.status === 'failed' ? 'bg-red-400' : 'bg-amber-400'}`}></div>
                         <div>
                           <div className="text-sm font-medium text-gray-900 dark:text-white group-hover:text-brand-500 transition-colors">{sub.task?.title || 'Task'}</div>
                           <div className="text-xs text-gray-500">{sub.status} • {sub.testsPassed || 0}/{sub.testsTotal || 0} tests</div>
                         </div>
                       </div>
                       <IconPlay className="w-4 h-4 text-gray-300 group-hover:text-brand-500" />
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Passed Submissions */}
            {passedSubmissions.length > 0 && (
              <div className="mt-4">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <IconCheckCircle className="w-3 h-3" /> {tUI('dashboard.completed')}
                </h3>
                <div className="space-y-2">
                  {passedSubmissions.slice(0, 5).map(sub => (
                    <Link to={`/task/${sub.task?.slug || sub.taskId}`} key={sub.id} data-testid="course-progress-card" className="group flex items-center justify-between p-3 rounded-xl bg-green-50/50 dark:bg-green-900/10 border border-green-100 dark:border-green-900/20 hover:border-green-300 transition-colors cursor-pointer">
                       <div className="flex items-center gap-3">
                         <div className="w-2 h-2 rounded-full bg-green-500"></div>
                         <div>
                           <div className="text-sm font-medium text-gray-900 dark:text-white">{sub.task?.title || 'Task'}</div>
                           <div className="text-xs text-gray-500">{tUI('dashboard.score')}: {sub.score || 100}</div>
                         </div>
                       </div>
                       <IconCheckCircle className="w-4 h-4 text-green-500" />
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Empty state */}
            {recentSubmissions.length === 0 && (
              <div className="text-center py-8 text-gray-400">
                <p className="text-sm">{tUI('dashboard.noActivity') || 'No recent activity'}</p>
                <Link to="/courses" className="text-brand-500 text-sm hover:underline mt-2 inline-block">
                  {tUI('dashboard.startLearning') || 'Start learning →'}
                </Link>
              </div>
            )}
            
          </div>
        </div>
      </div>
      
    </div>
  );
};

// Mock preview component for unauthenticated users
const DashboardPreview = ({ tUI }: { tUI: (key: string) => string }) => {
  const mockStats = [
    { label: tUI('dashboard.totalSolved'), val: '47', color: 'text-brand-500', bg: 'bg-brand-500/10' },
    { label: tUI('dashboard.hoursSpent'), val: '12h', color: 'text-purple-500', bg: 'bg-purple-500/10' },
    { label: tUI('dashboard.globalRank'), val: '#128', color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    { label: tUI('dashboard.skillPoints'), val: '2.4k', color: 'text-amber-500', bg: 'bg-amber-500/10' },
  ];

  const mockChartData = [
    { name: 'Mon', solved: 3 }, { name: 'Tue', solved: 5 }, { name: 'Wed', solved: 2 },
    { name: 'Thu', solved: 8 }, { name: 'Fri', solved: 4 }, { name: 'Sat', solved: 6 }, { name: 'Sun', solved: 3 }
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('dashboard.pageTitle')}</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">{tUI('dashboard.pageSubtitle')}</p>
        </div>
        <div className="text-right hidden md:block">
          <div className="text-sm font-bold text-gray-400 uppercase tracking-wider">{tUI('dashboard.currentStreak')}</div>
          <div className="text-3xl font-display font-bold text-orange-500">🔥 7 {tUI('dashboard.days')}</div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {mockStats.map((stat, idx) => (
          <div key={idx} className="bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
            <div className="flex justify-between items-start mb-2">
              <div className={`p-2 rounded-lg ${stat.bg} ${stat.color}`}>
                <IconChart className="w-5 h-5" />
              </div>
            </div>
            <div className="text-3xl font-display font-bold text-gray-900 dark:text-white mb-1">{stat.val}</div>
            <div className="text-xs font-bold text-gray-400 uppercase">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
        <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-6">{tUI('dashboard.activityOverview')}</h2>
        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={mockChartData}>
              <defs>
                <linearGradient id="colorSolvedMock" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" strokeOpacity={0.1} />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#9CA3AF', fontSize: 12}} />
              <YAxis axisLine={false} tickLine={false} tick={{fill: '#9CA3AF', fontSize: 12}} />
              <Area type="monotone" dataKey="solved" stroke="#0ea5e9" strokeWidth={2} fillOpacity={1} fill="url(#colorSolvedMock)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;