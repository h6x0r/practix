import React, { useContext, useEffect, useState } from 'react';
import { AuthContext } from '@/components/Layout';
import { gamificationService, LeaderboardEntry, UserGamificationStats } from '../api/gamificationService';
import { useUITranslation } from '@/contexts/LanguageContext';
import { IconTrophy, IconFire, IconStar } from '@/components/Icons';

const LeaderboardPage = () => {
  const { user } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [myStats, setMyStats] = useState<UserGamificationStats | null>(null);
  const [myRank, setMyRank] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [board, stats, rank] = await Promise.all([
          gamificationService.getLeaderboard(50),
          user ? gamificationService.getMyStats() : null,
          user ? gamificationService.getMyRank() : null,
        ]);
        setLeaderboard(board);
        if (stats) setMyStats(stats);
        if (rank) setMyRank(rank.rank);
      } catch (error) {
        console.error('Failed to load leaderboard:', error);
      }
      setLoading(false);
    };
    loadData();
  }, [user]);

  const getRankBadge = (rank: number) => {
    if (rank === 1) return { bg: 'bg-yellow-500', text: 'text-white', icon: '🥇' };
    if (rank === 2) return { bg: 'bg-gray-400', text: 'text-white', icon: '🥈' };
    if (rank === 3) return { bg: 'bg-amber-600', text: 'text-white', icon: '🥉' };
    return { bg: 'bg-gray-100 dark:bg-dark-bg', text: 'text-gray-600 dark:text-gray-400', icon: null };
  };

  if (loading) {
    return (
      <div className="p-10 text-center text-gray-500 animate-pulse">
        {tUI('common.loading')}
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white flex items-center justify-center gap-3">
          <IconTrophy className="w-8 h-8 text-yellow-500" />
          {tUI('leaderboard.title') || 'Leaderboard'}
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          {tUI('leaderboard.subtitle') || 'Top coders ranked by XP'}
        </p>
      </div>

      {/* My Stats Card (if logged in) */}
      {user && myStats && (
        <div className="bg-gradient-to-r from-brand-500 to-purple-600 rounded-2xl p-6 text-white shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-white/20 flex items-center justify-center text-2xl font-bold">
                {myRank ? `#${myRank}` : '-'}
              </div>
              <div>
                <h2 className="text-xl font-bold">{user.name}</h2>
                <p className="text-white/80">
                  Level {myStats.level} • {myStats.xp.toLocaleString()} XP
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-2 text-lg">
                <IconFire className="w-5 h-5 text-orange-300" />
                <span>{myStats.currentStreak} day streak</span>
              </div>
              <div className="text-white/70 text-sm mt-1">
                {myStats.badges.length} badges earned
              </div>
            </div>
          </div>
          {/* XP Progress */}
          <div className="mt-4">
            <div className="flex justify-between text-sm text-white/80 mb-1">
              <span>Progress to Level {myStats.level + 1}</span>
              <span>{myStats.xpProgress} / {myStats.xpNeeded} XP</span>
            </div>
            <div className="w-full h-2 bg-white/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-white rounded-full transition-all"
                style={{ width: `${myStats.progressPercent}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Leaderboard Table */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm overflow-hidden">
        <div className="p-4 border-b border-gray-100 dark:border-dark-border">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            {tUI('leaderboard.top50') || 'Top 50 Coders'}
          </h2>
        </div>
        <div className="divide-y divide-gray-100 dark:divide-dark-border">
          {leaderboard.map((entry) => {
            const rankStyle = getRankBadge(entry.rank);
            const isMe = user?.id === entry.id;

            return (
              <div
                key={entry.id}
                className={`flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors ${
                  isMe ? 'bg-brand-50 dark:bg-brand-900/20' : ''
                }`}
              >
                <div className="flex items-center gap-4">
                  {/* Rank */}
                  <div className={`w-10 h-10 rounded-full ${rankStyle.bg} ${rankStyle.text} flex items-center justify-center font-bold text-sm`}>
                    {rankStyle.icon || entry.rank}
                  </div>

                  {/* Avatar & Name */}
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-brand-400 to-purple-500 flex items-center justify-center text-white font-bold">
                      {entry.avatarUrl ? (
                        <img src={entry.avatarUrl} alt={entry.name} className="w-full h-full rounded-full object-cover" />
                      ) : (
                        entry.name.charAt(0).toUpperCase()
                      )}
                    </div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                        {entry.name}
                        {isMe && (
                          <span className="text-xs bg-brand-100 dark:bg-brand-900/30 text-brand-600 dark:text-brand-400 px-2 py-0.5 rounded-full">
                            You
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        Level {entry.level} • {entry.tasksSolved} tasks
                      </div>
                    </div>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {entry.xp.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">XP</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center gap-1 text-lg font-bold text-orange-500">
                      <IconFire className="w-4 h-4" />
                      {entry.streak}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Streak</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Empty State */}
      {leaderboard.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <IconStar className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>{tUI('leaderboard.empty') || 'No entries yet. Be the first!'}</p>
        </div>
      )}
    </div>
  );
};

export default LeaderboardPage;
