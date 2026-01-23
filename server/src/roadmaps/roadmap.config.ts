/**
 * Roadmap Configuration
 * Centralized configuration for roadmap generation and display
 */

// Cache TTL for roadmap variants (24 hours)
export const VARIANTS_CACHE_TTL = 86400;

// Salary ranges by experience level (USD)
export const SALARY_RANGES: Record<string, { min: number; max: number }> = {
  'junior': { min: 800, max: 1500 },
  'junior-plus': { min: 1200, max: 2500 },
  'middle': { min: 2000, max: 4000 },
  'middle-plus': { min: 3000, max: 5000 },
  'senior': { min: 3500, max: 6000 },
  'senior-plus': { min: 5000, max: 8000 },
};

// Course icons for UI display
export const COURSE_ICONS: Record<string, string> = {
  'c_go_basics': '🐹',
  'c_go_concurrency': '🐹',
  'c_go_web_apis': '🐹',
  'c_go_production': '🐹',
  'c_go_design_patterns': '🐹',
  'c_java_core': '☕',
  'c_java_modern': '☕',
  'c_java_advanced': '☕',
  'c_java_design_patterns': '☕',
  'c_python_ml_fundamentals': '🐍',
  'c_python_deep_learning': '🐍',
  'c_python_llm': '🐍',
  'c_java_ml': '☕',
  'c_java_nlp': '☕',
  'c_go_ml_inference': '🐹',
  'software-engineering': '🏗️',
  'algo-fundamentals': '🧮',
  'algo-advanced': '🧮',
};

// Phase color palettes for UI (Tailwind gradient classes)
export const PHASE_PALETTES = [
  'from-cyan-400 to-blue-500',
  'from-emerald-400 to-green-500',
  'from-orange-400 to-red-500',
  'from-purple-400 to-indigo-500',
  'from-pink-400 to-rose-500',
  'from-amber-400 to-yellow-500',
  'from-teal-400 to-cyan-500',
  'from-fuchsia-400 to-purple-600',
];

// Category to course pattern mapping for filtering
export const CATEGORY_PATTERNS: Record<string, RegExp[]> = {
  'backend-go': [/^c_go/],
  'backend-java': [/^c_java/],
  'python-data': [/^c_python/, /^algo/],
  'ai-ml': [/ml|deep|llm|nlp/i],
  'software-design': [/software-engineering/, /design-patterns/],
  'algorithms': [/^algo/],
  'fullstack': [/.*/], // All courses
};

// Default AI model for roadmap generation (can be overridden via env)
export const DEFAULT_AI_MODEL = 'gemini-2.0-flash';

// Generation limits
export const GENERATION_LIMITS = {
  FREE_GENERATIONS: 1,
  PREMIUM_GENERATIONS: 5,
  VARIANTS_COUNT: 3,
};

// Time estimates multipliers by experience level
export const TIME_MULTIPLIERS: Record<string, number> = {
  '0': 1.5,      // No experience - takes longer
  '<1': 1.2,    // Less than 1 year
  '1-2': 1.0,   // 1-2 years - baseline
  '3-5': 0.8,   // 3-5 years - faster
  '5+': 0.6,    // 5+ years - much faster
};
