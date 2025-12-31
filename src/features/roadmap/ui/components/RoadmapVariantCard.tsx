import React from 'react';
import { RoadmapVariant } from '../../model/types';
import { IconCheck, IconClock, IconCode, IconTrophy, IconChartBar } from '@/components/Icons';

interface RoadmapVariantCardProps {
  variant: RoadmapVariant;
  isSelected: boolean;
  onSelect: (variant: RoadmapVariant) => void;
}

const difficultyConfig = {
  easy: { color: 'green', label: 'Easy', bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400' },
  medium: { color: 'amber', label: 'Medium', bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400' },
  hard: { color: 'red', label: 'Hard', bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400' },
};

export const RoadmapVariantCard: React.FC<RoadmapVariantCardProps> = ({
  variant,
  isSelected,
  onSelect,
}) => {
  const difficulty = difficultyConfig[variant.difficulty] || difficultyConfig.medium;

  return (
    <button
      onClick={() => onSelect(variant)}
      className={`relative w-full text-left p-6 rounded-2xl border-2 transition-all duration-300 transform hover:scale-[1.02] ${
        isSelected
          ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20 shadow-lg shadow-brand-500/20'
          : 'border-gray-200 dark:border-dark-border bg-white dark:bg-dark-surface hover:border-brand-300 dark:hover:border-brand-700 hover:shadow-md'
      }`}
    >
      {/* Recommended badge */}
      {variant.isRecommended && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-gradient-to-r from-brand-500 to-purple-500 text-white text-xs font-bold rounded-full shadow-lg">
          ⭐ Recommended
        </div>
      )}

      {/* Selected checkmark */}
      {isSelected && (
        <div className="absolute top-4 right-4 w-6 h-6 bg-brand-500 rounded-full flex items-center justify-center">
          <IconCheck className="w-4 h-4 text-white" />
        </div>
      )}

      {/* Header */}
      <div className="mb-4">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
          {variant.name}
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {variant.description}
        </p>
      </div>

      {/* Target role & difficulty */}
      <div className="flex items-center gap-2 mb-4">
        <span className="px-2 py-1 bg-brand-100 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-xs font-bold rounded-lg">
          {variant.targetRole}
        </span>
        <span className={`px-2 py-1 ${difficulty.bg} ${difficulty.text} text-xs font-bold rounded-lg`}>
          {difficulty.label}
        </span>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {/* Tasks */}
        <div className="flex items-center gap-2 p-2.5 bg-gray-50 dark:bg-dark-bg rounded-lg">
          <IconCode className="w-4 h-4 text-brand-500" />
          <div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">{variant.totalTasks}</div>
            <div className="text-[10px] text-gray-500 uppercase">Tasks</div>
          </div>
        </div>

        {/* Time */}
        <div className="flex items-center gap-2 p-2.5 bg-gray-50 dark:bg-dark-bg rounded-lg">
          <IconClock className="w-4 h-4 text-purple-500" />
          <div>
            <div className="text-lg font-bold text-gray-900 dark:text-white">{variant.estimatedMonths}m</div>
            <div className="text-[10px] text-gray-500 uppercase">{variant.estimatedHours}h total</div>
          </div>
        </div>
      </div>

      {/* Salary range */}
      <div className="mb-4 p-3 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-100 dark:border-green-900/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <IconChartBar className="w-4 h-4 text-green-600 dark:text-green-400" />
            <span className="text-xs font-bold text-green-700 dark:text-green-400">Expected Salary</span>
          </div>
          <div className="text-lg font-bold text-green-700 dark:text-green-400">
            ${variant.salaryRange.min.toLocaleString()} - ${variant.salaryRange.max.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Topics */}
      <div className="mb-4">
        <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase">Topics</div>
        <div className="flex flex-wrap gap-1.5">
          {variant.topics.slice(0, 5).map((topic, i) => (
            <span key={i} className="px-2 py-0.5 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs rounded-md">
              {topic}
            </span>
          ))}
          {variant.topics.length > 5 && (
            <span className="px-2 py-0.5 text-gray-400 text-xs">+{variant.topics.length - 5}</span>
          )}
        </div>
      </div>

      {/* Source courses */}
      <div className="mb-4">
        <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase">From Courses</div>
        <div className="flex flex-wrap gap-2">
          {variant.sources.slice(0, 4).map((source, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2 py-1 bg-gray-50 dark:bg-dark-bg rounded-lg">
              <span className="text-sm">{source.icon}</span>
              <span className="text-xs text-gray-600 dark:text-gray-400">{source.percentage}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Preview tasks */}
      <div>
        <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase">First Tasks</div>
        <div className="space-y-1.5">
          {variant.previewTasks.slice(0, 3).map((task, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-4 h-4 flex items-center justify-center bg-brand-100 dark:bg-brand-900/30 text-brand-600 dark:text-brand-400 rounded text-[10px] font-bold">
                {i + 1}
              </span>
              <span className="text-gray-700 dark:text-gray-300 truncate flex-1">{task.title}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                task.difficulty === 'easy' ? 'bg-green-100 text-green-600' :
                task.difficulty === 'hard' ? 'bg-red-100 text-red-600' :
                'bg-amber-100 text-amber-600'
              }`}>
                {task.difficulty}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Selection indicator at bottom */}
      <div className={`mt-4 pt-4 border-t ${isSelected ? 'border-brand-200 dark:border-brand-800' : 'border-gray-100 dark:border-dark-border'}`}>
        <div className={`text-center text-sm font-bold ${
          isSelected ? 'text-brand-600 dark:text-brand-400' : 'text-gray-400 dark:text-gray-500'
        }`}>
          {isSelected ? '✓ Selected' : 'Click to select'}
        </div>
      </div>
    </button>
  );
};

export default RoadmapVariantCard;
