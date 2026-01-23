
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useParams, Link, useLocation, useNavigate } from 'react-router-dom';
import { courseService } from '../api/courseService';
import { getCourseTheme, getModuleIcon } from '@/utils/themeUtils';
import { IconChevronDown, IconPlay, IconCheckCircle, IconLock, IconClock, IconLayers } from '@/components/Icons';
import { Course, CourseModule, Topic, Task } from '@/types';
import { useLanguage, useUITranslation } from '@/contexts/LanguageContext';
import { createLogger } from '@/lib/logger';

const log = createLogger('CourseDetails');

const CourseDetailsPage = () => {
  const { courseId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { tUI, formatTimeLocalized, plural, difficulty: tDifficulty } = useUITranslation();
  const [rawCourse, setRawCourse] = useState<Course | undefined>(undefined);
  const [rawModules, setRawModules] = useState<CourseModule[]>([]);
  const [loading, setLoading] = useState(true);

  const [expandedModuleId, setExpandedModuleId] = useState<string | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Apply translations to course and nested modules/topics/tasks
  const course = useMemo(() => t(rawCourse), [rawCourse, t]);
  const modules = useMemo(() => rawModules.map(m => ({
    ...t(m),
    topics: m.topics.map(topic => ({
      ...t(topic),
      tasks: topic.tasks.map(task => t(task))
    }))
  })), [rawModules, t]);

  // Fetch course details and structure
  useEffect(() => {
      if (courseId) {
          setLoading(true);
          Promise.all([
            courseService.getCourseById(courseId),
            courseService.getCourseStructure(courseId)
          ]).then(([c, m]) => {
              setRawCourse(c);
              setRawModules(m);
              setLoading(false);
          }).catch((error) => {
              log.error('Failed to load course details', error);
              setLoading(false);
          });
      }
  }, [courseId]);

  // Update Page Title
  useEffect(() => {
    if (course) {
      document.title = `${course.title} — Practix`;
    }
  }, [course]);


  // Find the first pending task across all modules to continue journey
  const nextPendingTask = useMemo(() => {
    for (const module of rawModules) {
      for (const topic of module.topics) {
        for (const task of topic.tasks) {
          if (task.status !== 'completed') {
            return task;
          }
        }
      }
    }
    return null;
  }, [rawModules]);

  const handleContinueJourney = () => {
    if (nextPendingTask && courseId) {
      navigate(`/course/${courseId}/task/${nextPendingTask.slug}`);
    }
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (location.hash && modules.length > 0) {
      const topicId = location.hash.replace('#', '');
      const parentModule = modules.find(m => m.topics.some(t => t.id === topicId));
      if (parentModule) {
        setExpandedModuleId(parentModule.id);

        timeoutRef.current = setTimeout(() => {
            const el = document.getElementById(topicId);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 400);
      }
    }
  }, [location.hash, modules]);

  if (loading) {
      return <div className="p-10 text-center text-gray-500 animate-pulse">{tUI('courseDetails.loading')}</div>;
  }

  if (!course) {
    return <div className="p-8 text-center text-gray-500">{tUI('courseDetails.notFound')}</div>;
  }

  const theme = getCourseTheme(course.id);

  return (
    <div className="max-w-7xl mx-auto space-y-10 pb-20">
      
      {/* 1. Modern Header */}
      <div className="relative bg-white dark:bg-dark-surface rounded-[2rem] overflow-hidden border border-gray-100 dark:border-dark-border shadow-lg">
         {/* Background Gradient Mesh */}
         <div className={`absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-br ${theme.from} ${theme.to} opacity-10 rounded-full blur-3xl transform translate-x-1/3 -translate-y-1/3 pointer-events-none`}></div>
         
         <div className="relative z-10 p-10 flex flex-col md:flex-row gap-10 items-center md:items-start">
            <div className={`w-32 h-32 rounded-3xl bg-gradient-to-br ${theme.from} ${theme.to} text-white flex items-center justify-center text-6xl shadow-2xl transform rotate-3 flex-shrink-0`}>
              {course.icon}
            </div>
            
            <div className="flex-1 text-center md:text-left">
               <div className="flex flex-wrap items-center justify-center md:justify-start gap-3 mb-4">
                  <span className="px-3 py-1 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs font-bold rounded-full border border-gray-200 dark:border-dark-border uppercase tracking-wide">
                     {course.category === 'language' ? tUI('courseDetails.languageTrack') : tUI('courseDetails.csFundamental')}
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs font-bold rounded-full border border-gray-200 dark:border-dark-border">
                     <IconLayers className="w-3 h-3"/> {plural(modules.length, 'module')}
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs font-bold rounded-full border border-gray-200 dark:border-dark-border">
                     <IconClock className="w-3 h-3"/> {formatTimeLocalized(course.estimatedTime)}
                  </span>
               </div>
               
               <h1 data-testid="course-title" className="text-5xl font-display font-bold text-gray-900 dark:text-white mb-4 leading-tight">
                 {course.title}
               </h1>
               <p className="text-xl text-gray-500 dark:text-gray-400 max-w-2xl leading-relaxed">
                 {course.description}
               </p>

               {/* Progress & Continue Journey */}
               <div className="mt-8 flex flex-col sm:flex-row items-start sm:items-end gap-4 w-full max-w-xl mx-auto md:mx-0">
                   {/* Progress Bar */}
                   <div className="flex-1 w-full">
                     <div className="flex justify-between text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">
                       <span>{tUI('courseDetails.trackProgress')}</span>
                       <span>{course.progress}%</span>
                     </div>
                     <div data-testid="course-progress" className="h-3 bg-gray-100 dark:bg-dark-bg rounded-full overflow-hidden shadow-inner">
                        <div className={`h-full bg-gradient-to-r ${theme.from} ${theme.to}`} style={{width: `${course.progress}%`}}></div>
                     </div>
                   </div>

                   {/* Continue Journey Button */}
                   {nextPendingTask && (
                     <button
                       onClick={handleContinueJourney}
                       className={`flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${theme.from} ${theme.to} text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5 whitespace-nowrap`}
                     >
                       <IconPlay className="w-4 h-4" />
                       <span>{tUI('courseDetails.continueJourney')}</span>
                     </button>
                   )}
               </div>
            </div>
         </div>
      </div>


      {/* Modules List */}
      <div>
        <h2 className="text-2xl font-display font-bold text-gray-900 dark:text-white mb-6 pl-2">
            {tUI('courseDetails.curriculumModules')}
        </h2>

        {modules.length === 0 && (
            <div className="text-center py-20 text-gray-400">{tUI('courseDetails.noContent')}</div>
        )}

        <div className="grid grid-cols-1 gap-6">
           {modules.map((module, index) => {
             const isExpanded = expandedModuleId === module.id;
             // Calculate module progress based on topics/tasks
             const totalTasks = module.topics.reduce((acc, t) => acc + t.tasks.length, 0);
             const completedTasks = module.topics.reduce((acc, t) => acc + t.tasks.filter(tk => tk.status === 'completed').length, 0);
             const moduleProgress = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
             
             return (
               <div key={module.id} data-testid="module-item" className={`bg-white dark:bg-dark-surface rounded-3xl border transition-all duration-300 overflow-hidden ${isExpanded ? 'border-brand-500 ring-4 ring-brand-500/10 shadow-xl' : 'border-gray-100 dark:border-dark-border shadow-sm hover:border-gray-300 dark:hover:border-gray-600'}`}>
                  
                  {/* Module Header */}
                  <div 
                    onClick={() => setExpandedModuleId(isExpanded ? null : module.id)}
                    className="p-6 md:p-8 cursor-pointer flex flex-col md:flex-row items-start md:items-center gap-6 relative z-10 group/card"
                  >
                     <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-2xl flex-shrink-0 transition-all duration-300 shadow-lg ${
                       isExpanded
                         ? `bg-gradient-to-br ${theme.from} ${theme.to} text-white shadow-xl scale-110`
                         : `bg-gradient-to-br ${theme.from} ${theme.to} opacity-80 group-hover/card:opacity-100 group-hover/card:scale-105`
                     }`}>
                        <span className="text-3xl">{getModuleIcon(index)}</span>
                     </div>
                     
                     <div className="flex-1">
                        <div className="flex items-center gap-3 mb-1">
                          <span className={`px-2 py-0.5 bg-gradient-to-r ${theme.from} ${theme.to} text-white text-[10px] font-bold uppercase rounded-md shadow-sm`}>
                            #{index + 1}
                          </span>
                          <h3 className={`text-xl font-bold transition-colors ${isExpanded ? 'text-gray-900 dark:text-white' : 'text-gray-900 dark:text-white'}`}>
                            {module.title}
                          </h3>
                          <span className="flex items-center gap-1 px-2 py-0.5 bg-gray-100 dark:bg-dark-bg text-gray-500 text-[10px] font-bold uppercase rounded border border-gray-200 dark:border-dark-border">
                            <IconLayers className="w-3 h-3"/> {plural(totalTasks, 'task')}
                          </span>
                          <span className="flex items-center gap-1 px-2 py-0.5 bg-gray-100 dark:bg-dark-bg text-gray-500 text-[10px] font-bold uppercase rounded border border-gray-200 dark:border-dark-border">
                            <IconClock className="w-3 h-3"/> {formatTimeLocalized(module.estimatedTime || '1h')}
                          </span>
                        </div>
                        <p className="text-gray-500 dark:text-gray-400 text-sm">{module.description}</p>
                     </div>

                     <div className="flex items-center gap-6 w-full md:w-auto justify-between md:justify-end">
                        <div className="w-24 hidden md:block">
                           <div className="h-1.5 bg-gray-100 dark:bg-dark-bg rounded-full overflow-hidden">
                             <div className="h-full bg-green-500" style={{width: `${moduleProgress}%`}}></div>
                           </div>
                           <div className="text-[10px] text-right text-gray-400 mt-1 font-bold">{moduleProgress}% {tUI('courseDetails.done')}</div>
                        </div>
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                          isExpanded
                            ? `bg-gradient-to-br ${theme.from} ${theme.to} text-white rotate-180 shadow-lg`
                            : 'bg-gray-100 dark:bg-dark-bg text-gray-400 group-hover/card:bg-gray-200 dark:group-hover/card:bg-dark-border'
                        }`}>
                           <IconChevronDown className="w-5 h-5"/>
                        </div>
                     </div>
                  </div>

                  {/* Expanded Content (Topics List) */}
                  {isExpanded && (
                    <div className="border-t border-gray-100 dark:border-dark-border bg-gray-50/50 dark:bg-black/20 p-4 md:p-8 relative z-0">
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {module.topics.map((topic, i) => (
                             <div key={topic.id} id={topic.id} data-testid="topic-item" className="relative bg-white dark:bg-dark-surface p-5 rounded-2xl border border-gray-200 dark:border-dark-border hover:border-gray-300 dark:hover:border-gray-600 transition-all hover:shadow-lg group overflow-hidden">
                                {/* Gradient accent line */}
                                <div className={`absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b ${theme.from} ${theme.to}`}></div>

                                <div className="flex justify-between items-start mb-3 pl-3">
                                   <div className="flex items-center gap-2">
                                      {/* Topic number */}
                                      <span className={`w-6 h-6 rounded-lg bg-gradient-to-br ${theme.from} ${theme.to} text-white text-[10px] font-bold flex items-center justify-center shadow-sm`}>
                                        {i + 1}
                                      </span>
                                      <div className={`px-2 py-1 text-[10px] font-bold uppercase rounded border ${
                                        topic.difficulty === 'easy' ? 'bg-green-50 text-green-600 border-green-100 dark:bg-green-900/10 dark:text-green-400 dark:border-green-900/30' :
                                        topic.difficulty === 'medium' ? 'bg-yellow-50 text-yellow-600 border-yellow-100 dark:bg-yellow-900/10 dark:text-yellow-400 dark:border-yellow-900/30' :
                                        'bg-red-50 text-red-600 border-red-100 dark:bg-red-900/10 dark:text-red-400 dark:border-red-900/30'
                                      }`}>
                                        {tDifficulty(topic.difficulty)}
                                      </div>
                                   </div>
                                   {/* Tasks Count & Time */}
                                   <div className="flex items-center gap-2">
                                       <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-100 dark:bg-dark-bg text-[10px] font-bold text-gray-400">
                                            <IconClock className="w-3 h-3"/> {formatTimeLocalized(topic.estimatedTime)}
                                       </div>
                                       <div className="text-[10px] font-bold text-gray-400">{plural(topic.tasks.length, 'task')}</div>
                                   </div>
                                </div>
                                <h4 className="text-base font-bold text-gray-900 dark:text-white mb-1 pl-3 group-hover:text-brand-600 transition-colors">
                                  {topic.title}
                                </h4>
                                <p className="text-xs text-gray-500 mb-4 line-clamp-2 pl-3">{topic.description}</p>
                                
                                <div className="space-y-2 pl-3">
                                   {topic.tasks.map(task => (
                                     <Link
                                       to={`/course/${course.id}/task/${task.slug}`}
                                       key={task.id}
                                       data-testid="task-link"
                                       className="flex items-center justify-between p-2 rounded-lg bg-gray-50 dark:bg-dark-bg hover:bg-gray-100 dark:hover:bg-dark-border transition-colors cursor-pointer group/task relative z-10"
                                     >
                                        <div className="flex items-center gap-2 overflow-hidden flex-1">
                                           <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${task.status === 'completed' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                                           <span className={`text-xs font-medium truncate ${task.status === 'completed' ? 'text-gray-500 line-through' : 'text-gray-700 dark:text-gray-300'}`}>
                                             {task.title}
                                           </span>
                                        </div>
                                        <div className="flex items-center gap-2 pl-2">
                                            <span className="text-[10px] text-gray-400 font-mono hidden group-hover/task:block">{formatTimeLocalized(task.estimatedTime)}</span>
                                            {task.isPremium && <IconLock className="w-3 h-3 text-amber-500 flex-shrink-0"/>}
                                        </div>
                                     </Link>
                                   ))}
                                </div>
                             </div>
                          ))}
                       </div>
                    </div>
                  )}
               </div>
             );
           })}
        </div>
      </div>
    </div>
  );
};

export default CourseDetailsPage;
