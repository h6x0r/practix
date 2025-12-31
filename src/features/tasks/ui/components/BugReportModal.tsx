import React, { useState, useCallback } from 'react';
import { IconX, IconBug } from '@/components/Icons';
import { useUITranslation } from '@/contexts/LanguageContext';
import { bugReportService, BugCategory, BugSeverity, BugReportData } from '../../api/bugReportService';
import { useToast } from '@/components/Toast';

interface BugReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  taskId?: string;
  taskSlug?: string;
  isPremium: boolean;
  userCode?: string;
}

interface CategoryOption {
  id: BugCategory;
  icon: string;
  labelKey: string;
  premiumOnly?: boolean;
}

const CATEGORIES: CategoryOption[] = [
  { id: 'description', icon: '📝', labelKey: 'bugReport.categoryDescription' },
  { id: 'solution', icon: '💡', labelKey: 'bugReport.categorySolution', premiumOnly: true },
  { id: 'editor', icon: '💻', labelKey: 'bugReport.categoryEditor' },
  { id: 'hints', icon: '💭', labelKey: 'bugReport.categoryHints' },
  { id: 'ai-tutor', icon: '🤖', labelKey: 'bugReport.categoryAiTutor', premiumOnly: true },
  { id: 'other', icon: '❓', labelKey: 'bugReport.categoryOther' },
];

const SEVERITY_OPTIONS: { id: BugSeverity; labelKey: string; descKey: string }[] = [
  { id: 'low', labelKey: 'bugReport.severityLow', descKey: 'bugReport.severityLowDesc' },
  { id: 'medium', labelKey: 'bugReport.severityMedium', descKey: 'bugReport.severityMediumDesc' },
  { id: 'high', labelKey: 'bugReport.severityHigh', descKey: 'bugReport.severityHighDesc' },
];

export const BugReportModal: React.FC<BugReportModalProps> = ({
  isOpen,
  onClose,
  taskId,
  taskSlug,
  isPremium,
  userCode,
}) => {
  const { tUI } = useUITranslation();
  const { showToast } = useToast();

  const [selectedCategory, setSelectedCategory] = useState<BugCategory | null>(null);
  const [severity, setSeverity] = useState<BugSeverity>('medium');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Filter categories based on premium status
  const availableCategories = CATEGORIES.filter(
    (cat) => !cat.premiumOnly || isPremium
  );

  const resetForm = useCallback(() => {
    setSelectedCategory(null);
    setSeverity('medium');
    setTitle('');
    setDescription('');
  }, []);

  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [onClose, resetForm]);

  const handleSubmit = async () => {
    if (!selectedCategory || !title.trim() || !description.trim()) {
      showToast(tUI('bugReport.fillAllFields'), 'error');
      return;
    }

    setIsSubmitting(true);

    try {
      const reportData: BugReportData = {
        title: title.trim(),
        description: description.trim(),
        category: selectedCategory,
        severity,
        taskId,
        metadata: {
          userCode: userCode?.substring(0, 2000), // Limit code size
          browserInfo: navigator.userAgent,
          url: window.location.href,
        },
      };

      await bugReportService.submit(reportData);
      showToast(tUI('bugReport.submitSuccess'), 'success');
      handleClose();
    } catch (error) {
      console.error('Failed to submit bug report:', error);
      showToast(tUI('bugReport.submitError'), 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg mx-4 bg-white dark:bg-dark-surface rounded-xl shadow-2xl border border-gray-200 dark:border-dark-border overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-dark-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <IconBug className="w-5 h-5 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900 dark:text-white">
                {tUI('bugReport.title')}
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {taskSlug || tUI('bugReport.generalFeedback')}
              </p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 dark:hover:bg-dark-bg text-gray-500 transition-colors"
          >
            <IconX className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-5 max-h-[70vh] overflow-y-auto">
          {/* Category Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              {tUI('bugReport.selectCategory')}
            </label>
            <div className="grid grid-cols-3 gap-2">
              {availableCategories.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border-2 transition-all ${
                    selectedCategory === cat.id
                      ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                      : 'border-gray-200 dark:border-dark-border hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <span className="text-2xl">{cat.icon}</span>
                  <span className={`text-xs font-medium text-center ${
                    selectedCategory === cat.id
                      ? 'text-brand-700 dark:text-brand-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}>
                    {tUI(cat.labelKey)}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Severity Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              {tUI('bugReport.severity')}
            </label>
            <div className="space-y-2">
              {SEVERITY_OPTIONS.map((opt) => (
                <label
                  key={opt.id}
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    severity === opt.id
                      ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                      : 'border-gray-200 dark:border-dark-border hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <input
                    type="radio"
                    name="severity"
                    value={opt.id}
                    checked={severity === opt.id}
                    onChange={() => setSeverity(opt.id)}
                    className="w-4 h-4 text-brand-600 border-gray-300 focus:ring-brand-500"
                  />
                  <div>
                    <span className={`text-sm font-medium ${
                      severity === opt.id
                        ? 'text-brand-700 dark:text-brand-400'
                        : 'text-gray-700 dark:text-gray-300'
                    }`}>
                      {tUI(opt.labelKey)}
                    </span>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {tUI(opt.descKey)}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Title Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              {tUI('bugReport.issueTitle')}
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={tUI('bugReport.issueTitlePlaceholder')}
              className="w-full px-4 py-2.5 text-sm border border-gray-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-bg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-brand-500 focus:border-transparent outline-none transition-all"
              maxLength={200}
            />
          </div>

          {/* Description Textarea */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              {tUI('bugReport.description')}
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder={tUI('bugReport.descriptionPlaceholder')}
              rows={4}
              className="w-full px-4 py-2.5 text-sm border border-gray-200 dark:border-dark-border rounded-lg bg-white dark:bg-dark-bg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-brand-500 focus:border-transparent outline-none transition-all resize-none"
              maxLength={2000}
            />
            <p className="text-xs text-gray-400 mt-1 text-right">
              {description.length}/2000
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-dark-border bg-gray-50 dark:bg-dark-bg/50">
          <button
            onClick={handleClose}
            className="px-5 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-dark-bg rounded-lg transition-colors"
          >
            {tUI('common.cancel')}
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || !selectedCategory || !title.trim() || !description.trim()}
            className={`px-5 py-2 text-sm font-bold text-white rounded-lg transition-all flex items-center gap-2 ${
              isSubmitting || !selectedCategory || !title.trim() || !description.trim()
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-500 shadow-lg shadow-red-900/20'
            }`}
          >
            {isSubmitting && <span className="animate-spin">⟳</span>}
            {tUI('bugReport.submit')}
          </button>
        </div>
      </div>
    </div>
  );
};
