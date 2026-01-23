
// This utility maps data IDs to visual presentation styles.
// This allows the backend to be agnostic of the UI design system (Tailwind).

interface CourseTheme {
  from: string;
  to: string;
  accent: string; // For borders and highlights
}

export const getCourseTheme = (courseId: string): CourseTheme => {
  const id = courseId.toLowerCase();

  // Go Courses - Cyan/Blue family
  if (id === 'go-basics' || id.startsWith('go-') && !id.includes('ml') && !id.includes('design')) {
    if (id === 'go-concurrency') return { from: 'from-violet-500', to: 'to-purple-600', accent: 'violet' };
    if (id === 'go-web-apis') return { from: 'from-emerald-500', to: 'to-teal-600', accent: 'emerald' };
    if (id === 'go-production') return { from: 'from-orange-500', to: 'to-amber-600', accent: 'orange' };
    return { from: 'from-cyan-500', to: 'to-blue-600', accent: 'cyan' };
  }

  // Java Courses - Orange/Red family
  if (id.startsWith('java-') && !id.includes('ml') && !id.includes('nlp') && !id.includes('design')) {
    if (id === 'java-modern') return { from: 'from-pink-500', to: 'to-rose-600', accent: 'pink' };
    if (id === 'java-advanced') return { from: 'from-indigo-500', to: 'to-blue-600', accent: 'indigo' };
    return { from: 'from-orange-500', to: 'to-red-600', accent: 'orange' };
  }

  // Python Courses - Green/Yellow family
  if (id.startsWith('python-')) {
    if (id === 'python-deep-learning') return { from: 'from-orange-500', to: 'to-red-500', accent: 'orange' };
    if (id === 'python-llm') return { from: 'from-purple-500', to: 'to-pink-500', accent: 'purple' };
    return { from: 'from-green-500', to: 'to-yellow-500', accent: 'green' };
  }

  // ML/AI Courses - Purple/Pink family
  if (id.includes('-ml') || id.includes('-nlp') || id.includes('ml-')) {
    if (id.includes('java')) return { from: 'from-orange-400', to: 'to-pink-500', accent: 'orange' };
    if (id.includes('go')) return { from: 'from-cyan-400', to: 'to-purple-500', accent: 'cyan' };
    return { from: 'from-purple-500', to: 'to-pink-600', accent: 'purple' };
  }

  // Design Patterns - Purple/Indigo family
  if (id.includes('design-patterns')) {
    if (id.includes('go')) return { from: 'from-cyan-500', to: 'to-indigo-600', accent: 'cyan' };
    if (id.includes('java')) return { from: 'from-orange-500', to: 'to-indigo-600', accent: 'orange' };
    return { from: 'from-purple-500', to: 'to-indigo-600', accent: 'purple' };
  }

  // Software Engineering - Slate/Blue family
  if (id === 'software-engineering' || id.includes('se-')) {
    return { from: 'from-slate-500', to: 'to-blue-600', accent: 'slate' };
  }

  // Algorithms & DS - Green/Emerald family
  if (id.startsWith('algo-')) {
    if (id === 'algo-advanced') return { from: 'from-teal-500', to: 'to-green-600', accent: 'teal' };
    return { from: 'from-emerald-400', to: 'to-green-600', accent: 'emerald' };
  }

  // Prompt Engineering - Pink/Purple family
  if (id.includes('prompt') || id === 'c_prompt_engineering') {
    return { from: 'from-pink-500', to: 'to-purple-600', accent: 'pink' };
  }

  // Default
  return { from: 'from-gray-500', to: 'to-gray-700', accent: 'gray' };
};

// Module icons - rotating set of meaningful icons for variety
export const getModuleIcon = (index: number): string => {
  const icons = ['📚', '⚡', '🔧', '🎯', '💡', '🚀', '🔐', '🌐', '📊', '🧩'];
  return icons[index % icons.length];
};
