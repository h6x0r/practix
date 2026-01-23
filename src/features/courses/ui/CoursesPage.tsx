import React, { useState, useEffect, useMemo, useContext } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { courseService } from '../api/courseService';
import { userCoursesService, UserCourse } from '../api/userCoursesService';
import { getCourseTheme } from '@/utils/themeUtils';
import { Course } from '@/types';
import { IconClock } from '@/components/Icons';
import { useLanguage, useUITranslation } from '@/contexts/LanguageContext';
import { AuthContext } from '@/components/Layout';
import { createLogger } from '@/lib/logger';

const log = createLogger('Courses');

type CourseFilter =
  | 'all'
  // Languages
  | 'go' | 'java' | 'python'
  // CS Fundamentals
  | 'algo_ds' | 'patterns_se' | 'math_ds'
  // Applied
  | 'ml_ai';

const CoursesPage = () => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { tUI, formatTimeLocalized, plural } = useUITranslation();
  const { user } = useContext(AuthContext);
  const [rawCourses, setRawCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [startedCourseSlugs, setStartedCourseSlugs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<CourseFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Apply translations to courses and their sample topics
  const courses = useMemo(() => rawCourses.map(c => ({
    ...t(c),
    sampleTopics: c.sampleTopics?.map(topic => t(topic)) || []
  })), [rawCourses, t]);

  // Fetch user's started courses from API
  useEffect(() => {
    const fetchStartedCourses = async () => {
      if (!user) {
        setStartedCourseSlugs([]);
        return;
      }
      try {
        const userCourses = await userCoursesService.getStartedCourses();
        setStartedCourseSlugs(userCourses.map(c => c.slug));
      } catch (error) {
        log.error('Failed to load started courses', error);
      }
    };
    fetchStartedCourses();
  }, [user]);

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        setLoading(true);
        const data = await courseService.getAllCourses();
        setRawCourses(data);
      } catch (error) {
        log.error('Failed to load courses', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCourses();
  }, []);

  const handleStartCourse = async (e: React.MouseEvent, courseSlug: string) => {
    // Prevent bubbling if wrapped in link
    e.preventDefault();
    e.stopPropagation();

    if (user && !startedCourseSlugs.includes(courseSlug)) {
      try {
        await userCoursesService.startCourse(courseSlug);
        setStartedCourseSlugs(prev => [...prev, courseSlug]);
      } catch (error) {
        log.error('Failed to start course', error);
      }
    }
    navigate(`/course/${courseSlug}`);
  };

  // Filter courses based on course ID patterns (kebab-case slugs)
  const filterCourse = (course: Course, filter: CourseFilter): boolean => {
    const id = course.id.toLowerCase();
    switch (filter) {
      case 'all':
        return true;
      case 'go':
        // Go language courses (excluding ML/design patterns which have their own filters)
        return id.startsWith('go-') && !id.includes('design-patterns');
      case 'java':
        // Java language courses (excluding ML/NLP/design patterns)
        return id.startsWith('java-') && !id.includes('design-patterns');
      case 'python':
        // Python courses + Algorithms course (uses Python)
        return id.startsWith('python-') || id.startsWith('algo-');
      case 'algo_ds':
        // Algorithms & Data Structures
        return id.startsWith('algo-');
      case 'math_ds':
        // Math for Data Science
        return id === 'math-for-ds';
      case 'patterns_se':
        // Design Patterns & Software Engineering
        return id.includes('design-patterns') || id === 'software-engineering';
      case 'ml_ai':
        // ML/AI courses across all languages
        return id.includes('-ml') || id.includes('-nlp') || id.includes('ml-') ||
               id.includes('deep-learning') || id.includes('-llm');
      default:
        return true;
    }
  };

  const filteredCourses = useMemo(() => {
    let result = courses.filter(c => filterCourse(c, activeTab));

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      result = result.filter(c =>
        c.title.toLowerCase().includes(query) ||
        c.description.toLowerCase().includes(query) ||
        c.sampleTopics?.some(topic => topic.title.toLowerCase().includes(query))
      );
    }

    return result;
  }, [courses, activeTab, searchQuery]);

  // Get course badge based on ID (kebab-case slugs)
  const getCourseBadge = (courseId: string): { label: string; colorClass: string } => {
    const id = courseId.toLowerCase();

    // ML/AI courses
    if (id.includes('-ml') || id.includes('-nlp') || id.includes('ml-') || id.includes('deep-learning') || id.includes('-llm')) {
      return {
        label: tUI('courses.mlAi'),
        colorClass: 'bg-purple-50 text-purple-600 border-purple-100 dark:bg-purple-900/10 dark:text-purple-400 dark:border-purple-900/30'
      };
    }

    // Design Patterns
    if (id.includes('design-patterns')) {
      return {
        label: tUI('courses.designPatterns'),
        colorClass: 'bg-indigo-50 text-indigo-600 border-indigo-100 dark:bg-indigo-900/10 dark:text-indigo-400 dark:border-indigo-900/30'
      };
    }

    // Software Engineering
    if (id === 'software-engineering') {
      return {
        label: tUI('courses.softwareEng'),
        colorClass: 'bg-slate-50 text-slate-600 border-slate-100 dark:bg-slate-900/10 dark:text-slate-400 dark:border-slate-900/30'
      };
    }

    // Algorithms
    if (id.startsWith('algo-')) {
      return {
        label: tUI('courses.algoDs'),
        colorClass: 'bg-emerald-50 text-emerald-600 border-emerald-100 dark:bg-emerald-900/10 dark:text-emerald-400 dark:border-emerald-900/30'
      };
    }

    // Math for Data Science
    if (id === 'math-for-ds') {
      return {
        label: tUI('courses.mathDs'),
        colorClass: 'bg-blue-50 text-blue-600 border-blue-100 dark:bg-blue-900/10 dark:text-blue-400 dark:border-blue-900/30'
      };
    }

    // Go
    if (id.startsWith('go-')) {
      return {
        label: 'Go',
        colorClass: 'bg-cyan-50 text-cyan-600 border-cyan-100 dark:bg-cyan-900/10 dark:text-cyan-400 dark:border-cyan-900/30'
      };
    }

    // Java
    if (id.startsWith('java-')) {
      return {
        label: 'Java',
        colorClass: 'bg-orange-50 text-orange-600 border-orange-100 dark:bg-orange-900/10 dark:text-orange-400 dark:border-orange-900/30'
      };
    }

    // Python
    if (id.startsWith('python-')) {
      return {
        label: 'Python',
        colorClass: 'bg-green-50 text-green-600 border-green-100 dark:bg-green-900/10 dark:text-green-400 dark:border-green-900/30'
      };
    }

    return {
      label: tUI('courses.csCore'),
      colorClass: 'bg-gray-50 text-gray-600 border-gray-100 dark:bg-gray-900/10 dark:text-gray-400 dark:border-gray-900/30'
    };
  };

  // Filter groups for organized UI
  const filterGroups: { label: string; filters: { id: CourseFilter; label: string; icon: string; disabled?: boolean }[] }[] = [
    {
      label: tUI('courses.groupLanguages'),
      filters: [
        { id: 'go', label: 'Go', icon: '🐹' },
        { id: 'java', label: 'Java', icon: '☕' },
        { id: 'python', label: 'Python', icon: '🐍' },
      ]
    },
    {
      label: tUI('courses.groupCsFundamentals'),
      filters: [
        { id: 'algo_ds', label: tUI('courses.filterAlgoDS'), icon: '🧮' },
        { id: 'math_ds', label: tUI('courses.filterMathDS'), icon: '📐' },
        { id: 'patterns_se', label: tUI('courses.filterPatternsSE'), icon: '🏗' },
      ]
    },
    {
      label: tUI('courses.groupApplied'),
      filters: [
        { id: 'ml_ai', label: tUI('courses.filterMlAi'), icon: '🤖' },
      ]
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="text-gray-400 animate-pulse text-lg">{tUI('courses.loading')}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-white dark:bg-dark-surface rounded-3xl p-10 border border-gray-100 dark:border-dark-border shadow-sm">
        <div className="relative z-10">
          <h1 className="text-4xl font-display font-bold text-gray-900 dark:text-white mb-4">{tUI('courses.title')}</h1>
          <p className="text-lg text-gray-500 dark:text-gray-400 max-w-2xl leading-relaxed">
            {tUI('courses.description')}
          </p>
        </div>
        <div className="absolute top-0 right-0 w-96 h-96 bg-brand-500/5 rounded-full blur-3xl transform translate-x-1/3 -translate-y-1/3 pointer-events-none"></div>
      </div>

      {/* Search & Filter Bar */}
      <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-100 dark:border-dark-border p-4 relative z-20">
        {/* Search Input */}
        <div className="mb-4">
          <div className="relative">
            <svg
              className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder={tUI('courses.searchPlaceholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              data-testid="course-search"
              className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent transition-all"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Filter Groups */}
        <div className="flex flex-wrap items-start gap-6">
          {/* All Courses Button */}
          <button
            onClick={() => setActiveTab('all')}
            className={`px-4 py-2 rounded-xl text-sm font-bold whitespace-nowrap transition-all ${
              activeTab === 'all'
                ? 'bg-gray-900 dark:bg-white text-white dark:text-black shadow-lg'
                : 'bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-dark-border'
            }`}
          >
            {tUI('courses.allTracks')}
          </button>

          {/* Separator */}
          <div className="h-10 w-px bg-gray-200 dark:bg-dark-border hidden sm:block" />

          {/* Filter Groups */}
          {filterGroups.map((group, groupIdx) => (
            <div key={groupIdx} className="flex flex-col gap-2">
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider px-1">
                {group.label}
              </span>
              <div className="flex flex-wrap gap-2">
                {group.filters.map(filter => (
                  <button
                    key={filter.id}
                    onClick={() => !filter.disabled && setActiveTab(filter.id)}
                    disabled={filter.disabled}
                    data-testid={`category-filter-${filter.id}`}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                      filter.disabled
                        ? 'bg-gray-50 dark:bg-dark-bg text-gray-300 dark:text-gray-600 cursor-not-allowed'
                        : activeTab === filter.id
                          ? 'bg-gray-900 dark:bg-white text-white dark:text-black shadow-md'
                          : 'bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-dark-border border border-transparent hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <span>{filter.icon}</span>
                    <span>{filter.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filteredCourses.map(course => {
          const isStarted = startedCourseSlugs.includes(course.slug);
          const theme = getCourseTheme(course.slug);
          return (
            <div
              key={course.id}
              data-testid="course-card"
              className="group relative flex flex-col bg-white dark:bg-dark-surface rounded-3xl p-1 border border-gray-100 dark:border-dark-border hover:border-brand-200 dark:hover:border-gray-700 transition-all duration-300 hover:shadow-xl hover:shadow-brand-900/5"
            >
              {/* Card Content Wrapper */}
              <div className="p-7 flex flex-col h-full rounded-[20px] bg-white dark:bg-dark-surface overflow-hidden relative z-10">
                
                {/* Header */}
                <div className="flex justify-between items-start mb-6 relative z-10 pointer-events-none">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${theme.from} ${theme.to} text-white flex items-center justify-center text-3xl shadow-lg transform group-hover:scale-110 group-hover:rotate-3 transition-transform duration-300`}>
                    {course.icon}
                  </div>
                  <div className="flex flex-col gap-2 items-end">
                     {(() => {
                       const badge = getCourseBadge(course.id);
                       return (
                         <span className={`px-3 py-1 text-[10px] font-bold uppercase tracking-wider rounded-full border ${badge.colorClass}`}>
                           {badge.label}
                         </span>
                       );
                     })()}
                  </div>
                </div>

                <Link to={`/course/${course.slug}`} className="block relative z-20">
                    <h3 className="text-2xl font-display font-bold text-gray-900 dark:text-white mb-3 group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">
                    {course.title}
                    </h3>
                    <p className="text-gray-500 dark:text-gray-400 leading-relaxed mb-8 flex-1">
                    {course.description}
                    </p>
                </Link>

                {/* Topic Previews - Fixed height */}
                <Link to={`/course/${course.slug}`} className="mb-6 relative z-20 cursor-pointer flex-1 flex flex-col">
                   <div className="flex flex-wrap gap-2 min-h-[72px] content-start">
                      {course.sampleTopics?.slice(0, 3).map((topic, i) => (
                        <span
                          key={i}
                          className="px-3 py-1.5 bg-gray-50 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs font-medium rounded-lg border border-gray-100 dark:border-dark-border hover:border-gray-300 dark:hover:border-gray-600 transition-colors h-fit"
                        >
                          {topic.title}
                        </span>
                      ))}
                      {(course.totalModules > 3) && (
                        <span className="px-3 py-1.5 bg-gray-100 dark:bg-dark-border text-gray-500 text-xs font-bold rounded-lg h-fit">
                          +{course.totalModules - 3}
                        </span>
                      )}
                   </div>
                   <div className="flex items-center gap-3 mt-auto pt-3 text-xs text-gray-400">
                     <span className="flex items-center gap-1">
                       <IconClock className="w-3 h-3"/> {formatTimeLocalized(course.estimatedTime)}
                     </span>
                     <span>•</span>
                     <span>{plural(course.totalModules, 'module')}</span>
                   </div>
                </Link>

                {/* Action - High Z-Index to prevent click blocking */}
                <div className="mt-auto pt-6 border-t border-gray-100 dark:border-dark-border flex items-center justify-between relative z-30">
                   {isStarted ? (
                     <div className="flex items-center gap-2 text-green-500 font-bold text-sm">
                       <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                       {tUI('courses.inProgress')} ({course.progress}%)
                     </div>
                   ) : (
                     <div className="text-sm font-bold text-gray-400">{tUI('courses.notStarted')}</div>
                   )}

                   {isStarted ? (
                      <Link
                        to={`/course/${course.slug}`}
                        className="px-6 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
                      >
                        {tUI('courses.resume')}
                      </Link>
                   ) : (
                      <button
                        onClick={(e) => handleStartCourse(e, course.slug)}
                        className="px-6 py-2.5 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-bold rounded-xl shadow-lg shadow-emerald-500/25 transition-all transform hover:-translate-y-0.5"
                      >
                        {tUI('courses.startLearning')}
                      </button>
                   )}
                </div>

              </div>
              
              {/* Background Decor - Explicitly Z-0 to stay behind */}
              <div className={`absolute -bottom-12 -right-12 w-48 h-48 bg-gradient-to-br ${theme.from} ${theme.to} opacity-5 rounded-full blur-2xl group-hover:opacity-10 transition-opacity z-0 pointer-events-none`}></div>
            </div>
          );
        })}
      </div>
      
      {filteredCourses.length === 0 && (
          <div className="text-center py-20 bg-white dark:bg-dark-surface rounded-3xl border border-gray-100 dark:border-dark-border border-dashed">
              <div className="text-4xl mb-4">🚧</div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">{tUI('courses.comingSoon')}</h3>
              <p className="text-gray-500">{tUI('courses.comingSoonDesc')}</p>
          </div>
      )}
    </div>
  );
};

export default CoursesPage;