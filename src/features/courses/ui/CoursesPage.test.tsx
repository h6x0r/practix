import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import CoursesPage from './CoursesPage';
import { AuthContext } from '@/components/Layout';

// Mock dependencies
vi.mock('../api/courseService', () => ({
  courseService: {
    getAllCourses: vi.fn(),
  },
}));

vi.mock('../api/userCoursesService', () => ({
  userCoursesService: {
    getStartedCourses: vi.fn(),
    startCourse: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useLanguage: () => ({
    t: (entity: any) => entity,
    language: 'en',
  }),
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'courses.title': 'Courses',
        'courses.description': 'Learn programming',
        'courses.subtitle': 'Learn programming',
        'courses.all': 'All',
        'courses.allTracks': 'All Tracks',
        'courses.go': 'Go',
        'courses.java': 'Java',
        'courses.python': 'Python',
        'courses.startCourse': 'Start Course',
        'courses.startLearning': 'Start Learning',
        'courses.continueCourse': 'Continue',
        'courses.resume': 'Resume',
        'courses.loading': 'Loading...',
        'courses.inProgress': 'In Progress',
        'courses.notStarted': 'Not Started',
        'courses.searchPlaceholder': 'Search courses...',
        'courses.groupLanguages': 'Languages',
        'courses.groupCsFundamentals': 'CS Fundamentals',
        'courses.groupApplied': 'Applied',
        'courses.filterAlgoDS': 'Algorithms & DS',
        'courses.filterMathDS': 'Math for DS',
        'courses.filterPatternsSE': 'Design Patterns',
        'courses.filterMlAi': 'ML/AI',
        'courses.mlAi': 'ML/AI',
        'courses.designPatterns': 'Design Patterns',
        'courses.softwareEng': 'Software Eng',
        'courses.algoDs': 'Algorithms',
        'courses.mathDs': 'Math DS',
        'courses.csCore': 'CS Core',
        'courses.comingSoon': 'Coming Soon',
        'courses.comingSoonDesc': 'More courses are on the way!',
        'common.loading': 'Loading...',
      };
      return translations[key] || key;
    },
    formatTimeLocalized: (time: string) => time,
    plural: (count: number, word: string) => `${count} ${word}s`,
  }),
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

vi.mock('@/components/Toast', () => ({
  useToast: () => ({
    showToast: vi.fn(),
  }),
}));

vi.mock('@/utils/themeUtils', () => ({
  getCourseTheme: () => ({
    from: 'from-blue-500',
    to: 'to-purple-500',
    gradient: 'from-blue-500 to-purple-500',
    accent: 'blue',
    icon: '📘',
  }),
}));

import { courseService } from '../api/courseService';
import { userCoursesService } from '../api/userCoursesService';

describe('CoursesPage', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com', name: 'Test User' };

  const mockCourses = [
    {
      id: 'go-basics',
      slug: 'go-basics',
      title: 'Go Basics',
      description: 'Learn Go programming',
      taskCount: 50,
      moduleCount: 5,
      estimatedHours: 20,
      difficulty: 'beginner',
      sampleTopics: [],
    },
    {
      id: 'java-core',
      slug: 'java-core',
      title: 'Java Core',
      description: 'Master Java fundamentals',
      taskCount: 80,
      moduleCount: 8,
      estimatedHours: 40,
      difficulty: 'intermediate',
      sampleTopics: [],
    },
  ];

  const mockUserCourses = [
    { slug: 'go-basics', progress: 50 },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(courseService.getAllCourses).mockResolvedValue(mockCourses);
    vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue(mockUserCourses);
    vi.mocked(userCoursesService.startCourse).mockResolvedValue({ slug: 'java-core', progress: 0 });
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <CoursesPage />
        </MemoryRouter>
      </AuthContext.Provider>
    );
  };

  describe('loading state', () => {
    it('should show loading state initially', () => {
      renderWithAuth(mockUser);

      expect(courseService.getAllCourses).toHaveBeenCalled();
    });
  });

  describe('course display', () => {
    it('should load and display courses', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      expect(screen.getByText('Java Core')).toBeInTheDocument();
    });

    it('should display course descriptions', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Learn Go programming')).toBeInTheDocument();
      });
    });
  });

  describe('user courses', () => {
    it('should load user started courses', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(userCoursesService.getStartedCourses).toHaveBeenCalled();
      });
    });

    it('should not load user courses when not authenticated', async () => {
      renderWithAuth(null);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // User courses should still be called but return empty
      expect(userCoursesService.getStartedCourses).not.toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should handle courses load error', async () => {
      vi.mocked(courseService.getAllCourses).mockRejectedValue(new Error('API Error'));

      renderWithAuth(mockUser);

      // Should not crash
      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });
    });
  });

  describe('filtering', () => {
    it('should show All Tracks filter by default', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Both courses should be visible initially (all tab)
      expect(screen.getByText('Java Core')).toBeInTheDocument();
      expect(screen.getByText('All Tracks')).toBeInTheDocument();
    });

    it('should filter courses by Go tab', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Click Go filter
      fireEvent.click(screen.getByTestId('category-filter-go'));

      // Only Go course should be visible
      expect(screen.getByText('Go Basics')).toBeInTheDocument();
      expect(screen.queryByText('Java Core')).not.toBeInTheDocument();
    });

    it('should filter courses by Java tab', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Click Java filter
      fireEvent.click(screen.getByTestId('category-filter-java'));

      // Only Java course should be visible
      expect(screen.queryByText('Go Basics')).not.toBeInTheDocument();
      expect(screen.getByText('Java Core')).toBeInTheDocument();
    });

    it('should show all courses when All Tracks clicked', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Click Go filter first
      fireEvent.click(screen.getByTestId('category-filter-go'));
      expect(screen.queryByText('Java Core')).not.toBeInTheDocument();

      // Click All Tracks
      fireEvent.click(screen.getByText('All Tracks'));

      // Both should be visible again
      expect(screen.getByText('Go Basics')).toBeInTheDocument();
      expect(screen.getByText('Java Core')).toBeInTheDocument();
    });
  });

  describe('search', () => {
    it('should render search input', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      expect(screen.getByTestId('course-search')).toBeInTheDocument();
    });

    it('should filter courses by search query', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Type in search
      const searchInput = screen.getByTestId('course-search');
      fireEvent.change(searchInput, { target: { value: 'Java' } });

      // Only Java course should be visible
      expect(screen.queryByText('Go Basics')).not.toBeInTheDocument();
      expect(screen.getByText('Java Core')).toBeInTheDocument();
    });

    it('should show empty state when no results', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Type search that matches nothing
      const searchInput = screen.getByTestId('course-search');
      fireEvent.change(searchInput, { target: { value: 'NonExistentCourse' } });

      // Empty state should appear
      expect(screen.getByText('Coming Soon')).toBeInTheDocument();
    });

    it('should show clear button when search has value', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      const searchInput = screen.getByTestId('course-search');
      fireEvent.change(searchInput, { target: { value: 'Test' } });

      // Clear button should appear (X icon)
      const clearButton = document.querySelector('button svg path[d*="M6 18L18 6"]')?.closest('button');
      expect(clearButton).toBeInTheDocument();
    });
  });

  describe('course actions', () => {
    it('should show Start Learning button for not started courses', async () => {
      // User has no started courses
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue([]);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      const startButtons = screen.getAllByText('Start Learning');
      expect(startButtons.length).toBe(2);
    });

    it('should show Resume button for started courses', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // go-basics is in mockUserCourses
      expect(screen.getByText('Resume')).toBeInTheDocument();
    });

    it('should show In Progress status for started courses', async () => {
      const mockStartedCourses = [
        { slug: 'go-basics', progress: 50 },
      ];
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue(mockStartedCourses);

      // Update mock courses to have progress
      const coursesWithProgress = [
        { ...mockCourses[0], progress: 50 },
        { ...mockCourses[1], progress: 0 },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithProgress);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText(/In Progress/)).toBeInTheDocument();
      });
    });

    it('should call startCourse when clicking Start Learning', async () => {
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue([]);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Find the Start Learning button for Go Basics
      const startButtons = screen.getAllByText('Start Learning');
      fireEvent.click(startButtons[0]);

      await waitFor(() => {
        expect(userCoursesService.startCourse).toHaveBeenCalledWith('go-basics');
      });
    });
  });

  describe('course badges', () => {
    it('should display Go badge for Go courses', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Find badge by class - it's the span with uppercase styling
      const goBadges = screen.getAllByText('Go');
      const goBadge = goBadges.find(el => el.className.includes('uppercase'));
      expect(goBadge).toBeInTheDocument();
    });

    it('should display Java badge for Java courses', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Java Core')).toBeInTheDocument();
      });

      // Find badge by class - it's the span with uppercase styling
      const javaBadges = screen.getAllByText('Java');
      const javaBadge = javaBadges.find(el => el.className.includes('uppercase'));
      expect(javaBadge).toBeInTheDocument();
    });

    it('should display ML/AI badge for ML courses', async () => {
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue([]);
      const mlCourses = [
        {
          id: 'python-ml-basics',
          slug: 'python-ml-basics',
          title: 'Python ML',
          description: 'Machine learning with Python',
          taskCount: 60,
          moduleCount: 6,
          totalModules: 6,
          estimatedHours: 30,
          estimatedTime: '30h',
          difficulty: 'intermediate',
          sampleTopics: [],
          icon: '🤖',
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(mlCourses);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Python ML')).toBeInTheDocument();
      });

      // Find badge - check the badge element exists
      const badges = screen.getAllByText('ML/AI');
      expect(badges.length).toBeGreaterThan(0);
    });

    it('should display Algorithms badge for algo courses', async () => {
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue([]);
      const algoCourses = [
        {
          id: 'algo-fundamentals',
          slug: 'algo-fundamentals',
          title: 'Algorithm Course',
          description: 'Learn algorithms',
          taskCount: 100,
          moduleCount: 10,
          totalModules: 10,
          estimatedHours: 50,
          estimatedTime: '50h',
          difficulty: 'intermediate',
          sampleTopics: [],
          icon: '🧮',
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(algoCourses);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Algorithm Course')).toBeInTheDocument();
      });

      // Algorithms badge should be present
      expect(screen.getByText('Algorithms')).toBeInTheDocument();
    });
  });

  describe('filter groups', () => {
    it('should display filter group labels', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      expect(screen.getByText('Languages')).toBeInTheDocument();
      expect(screen.getByText('CS Fundamentals')).toBeInTheDocument();
      expect(screen.getByText('Applied')).toBeInTheDocument();
    });

    it('should display all filter buttons', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      expect(screen.getByTestId('category-filter-go')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-java')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-python')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-algo_ds')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-math_ds')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-patterns_se')).toBeInTheDocument();
      expect(screen.getByTestId('category-filter-ml_ai')).toBeInTheDocument();
    });
  });

  describe('course card details', () => {
    it('should display course icon', async () => {
      vi.mocked(userCoursesService.getStartedCourses).mockResolvedValue([]);
      const coursesWithIcon = [
        {
          ...mockCourses[0],
          icon: '🐹',
          totalModules: 5,
          estimatedTime: '20h',
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithIcon);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      // Icon should be in a div - use getAllByText since it appears in both course card and filter button
      const icons = screen.getAllByText('🐹');
      expect(icons.length).toBeGreaterThan(0);
    });

    it('should display estimated time', async () => {
      const coursesWithTime = [
        {
          ...mockCourses[0],
          estimatedTime: '20h',
          totalModules: 5,
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithTime);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('20h')).toBeInTheDocument();
      });
    });

    it('should display module count', async () => {
      const coursesWithModules = [
        {
          ...mockCourses[0],
          totalModules: 5,
          estimatedTime: '20h',
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithModules);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('5 modules')).toBeInTheDocument();
      });
    });

    it('should display sample topics', async () => {
      const coursesWithTopics = [
        {
          ...mockCourses[0],
          totalModules: 5,
          estimatedTime: '20h',
          sampleTopics: [
            { title: 'Variables' },
            { title: 'Functions' },
            { title: 'Loops' },
          ],
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithTopics);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Variables')).toBeInTheDocument();
      });

      expect(screen.getByText('Functions')).toBeInTheDocument();
      expect(screen.getByText('Loops')).toBeInTheDocument();
    });

    it('should show +N for extra modules', async () => {
      const coursesWithManyTopics = [
        {
          ...mockCourses[0],
          totalModules: 10,
          estimatedTime: '40h',
          sampleTopics: [
            { title: 'Topic 1' },
            { title: 'Topic 2' },
            { title: 'Topic 3' },
            { title: 'Topic 4' },
          ],
        },
      ];
      vi.mocked(courseService.getAllCourses).mockResolvedValue(coursesWithManyTopics);

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('+7')).toBeInTheDocument();
      });
    });
  });

  describe('navigation', () => {
    it('should have links to course detail page', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });

      const courseLinks = screen.getAllByRole('link');
      const goBasicsLink = courseLinks.find(link => link.getAttribute('href') === '/course/go-basics');
      expect(goBasicsLink).toBeInTheDocument();
    });
  });
});
