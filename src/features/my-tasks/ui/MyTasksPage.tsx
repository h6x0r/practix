
import React, { useState, useEffect, useContext } from 'react';
import { Link } from 'react-router-dom';
import { userCoursesService, UserCourse } from '@/features/courses/api/userCoursesService';
import { getCourseTheme } from '@/utils/themeUtils';
import { IconPlay, IconBook } from '@/components/Icons';
import { useUITranslation } from '@/contexts/LanguageContext';
import { AuthContext } from '@/components/Layout';
import { AuthRequiredOverlay } from '@/components/AuthRequiredOverlay';

// Mock data for unauthenticated users preview
const MOCK_COURSES = [
  { id: '1', slug: 'go-basics', title: 'Go Basics', description: 'Master Go fundamentals', icon: '🐹', progress: 45 },
  { id: '2', slug: 'java-core', title: 'Java Core', description: 'Learn Java programming', icon: '☕', progress: 30 },
  { id: '3', slug: 'python-ml', title: 'Python ML', description: 'Machine learning with Python', icon: '🐍', progress: 15 },
];

const MyTasksPage = () => {
  const { tUI } = useUITranslation();
  const { user } = useContext(AuthContext);
  const [myCourses, setMyCourses] = useState<UserCourse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCourses = async () => {
        if (!user) {
          setLoading(false);
          return;
        }
        setLoading(true);
        try {
          // Fetch courses directly from API - includes progress
          const startedCourses = await userCoursesService.getStartedCourses();
          setMyCourses(startedCourses);
        } catch (error) {
          console.error('Failed to fetch user courses:', error);
        } finally {
          setLoading(false);
        }
    };
    fetchCourses();
  }, [user]);

  // Show auth overlay for unauthenticated users with mock preview
  if (!user) {
    return (
      <AuthRequiredOverlay
        title={tUI('myTasks.loginRequired')}
        description={tUI('myTasks.loginRequiredDesc')}
      >
        <MyTasksContent courses={MOCK_COURSES as any} tUI={tUI} />
      </AuthRequiredOverlay>
    );
  }

  if (loading) {
      return <div className="text-center pt-20 text-gray-500 animate-pulse">{tUI('myTasks.loadingProgress')}</div>;
  }

  if (myCourses.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <div className="w-24 h-24 bg-gray-100 dark:bg-dark-surface rounded-full flex items-center justify-center mb-6">
          <IconBook className="w-10 h-10 text-gray-400" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">{tUI('myTasks.noActiveCourses')}</h2>
        <p className="text-gray-500 dark:text-gray-400 max-w-md mb-8">
          {tUI('myTasks.noActiveCoursesDesc')}
        </p>
        <Link
          to="/courses"
          className="px-8 py-3 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
        >
          {tUI('myTasks.browseCourses')}
        </Link>
      </div>
    );
  }

  return <MyTasksContent courses={myCourses} tUI={tUI} />;
};

// Extracted content component for reuse in preview
const MyTasksContent = ({ courses, tUI }: { courses: any[]; tUI: (key: string) => string }) => {
  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white">{tUI('myTasks.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">{tUI('myTasks.description')}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {courses.map(course => {
          const theme = getCourseTheme(course.slug);
          return (
            <div key={course.id} className="group relative bg-white dark:bg-dark-surface rounded-3xl border border-gray-100 dark:border-dark-border overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300">
              {/* Gradient accent bar */}
              <div className={`h-1.5 bg-gradient-to-r ${theme.from} ${theme.to}`} />

              <div className="p-6">
                <div className="flex items-start gap-5">
                  {/* Icon with gradient background */}
                  <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${theme.from} ${theme.to} flex items-center justify-center text-2xl shadow-lg text-white flex-shrink-0 group-hover:scale-110 transition-transform duration-300`}>
                    {course.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1 group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">{course.title}</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-1">{course.description}</p>
                  </div>
                </div>

                {/* Progress section */}
                <div className="mt-5 pt-5 border-t border-gray-100 dark:border-dark-border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">{tUI('myTasks.progress') || 'Progress'}</span>
                    <span className="text-sm font-bold text-gray-900 dark:text-white">{course.progress}%</span>
                  </div>
                  <div className="h-2.5 bg-gray-100 dark:bg-dark-bg rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${theme.from} ${theme.to} transition-all duration-500`}
                      style={{width: `${course.progress}%`}}
                    />
                  </div>
                </div>

                {/* Action button */}
                <Link
                  to={`/course/${course.slug}`}
                  className="mt-5 w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 text-white dark:text-black font-bold rounded-xl hover:from-gray-800 hover:to-gray-600 dark:hover:from-gray-100 dark:hover:to-gray-300 transition-all transform hover:-translate-y-0.5 shadow-sm hover:shadow-lg"
                >
                  <IconPlay className="w-4 h-4" />
                  {tUI('myTasks.continueLearning')}
                </Link>
              </div>

              {/* Decorative gradient blob */}
              <div className={`absolute -bottom-16 -right-16 w-32 h-32 bg-gradient-to-br ${theme.from} ${theme.to} opacity-5 rounded-full blur-2xl group-hover:opacity-10 transition-opacity pointer-events-none`} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MyTasksPage;
