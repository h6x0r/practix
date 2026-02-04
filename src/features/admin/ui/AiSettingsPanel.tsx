import React, { useState, useEffect, useCallback } from "react";
import { adminService, AiSettings, AiLimits } from "../api/adminService";
import { useUITranslation } from "@/contexts/LanguageContext";
import { createLogger } from "@/lib/logger";

const log = createLogger("AiSettingsPanel");

interface LimitInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  description: string;
}

const LimitInput: React.FC<LimitInputProps> = ({
  label,
  value,
  onChange,
  description,
}) => (
  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-bg rounded-xl">
    <div className="flex-1">
      <div className="font-medium text-gray-900 dark:text-white">{label}</div>
      <div className="text-sm text-gray-500 dark:text-gray-400">
        {description}
      </div>
    </div>
    <input
      type="number"
      min={0}
      max={1000}
      value={value}
      onChange={(e) => onChange(Math.max(0, parseInt(e.target.value) || 0))}
      className="w-24 px-3 py-2 text-center font-bold text-lg bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-border rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-transparent"
    />
  </div>
);

const AiSettingsPanel: React.FC = () => {
  const { tUI } = useUITranslation();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [editedLimits, setEditedLimits] = useState<AiLimits | null>(null);
  const [enabled, setEnabled] = useState(true);

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await adminService.getAiSettings();
      setSettings(data);
      setEditedLimits(data.limits);
      setEnabled(data.enabled);
    } catch (err) {
      log.error("Failed to load AI settings", err);
      setError(tUI("admin.aiSettings.loadError"));
    } finally {
      setLoading(false);
    }
  }, [tUI]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleSave = async () => {
    if (!editedLimits) return;

    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      const updated = await adminService.updateAiSettings({
        enabled,
        limits: editedLimits,
      });

      setSettings(updated);
      setEditedLimits(updated.limits);
      setEnabled(updated.enabled);
      setSuccess(true);

      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      log.error("Failed to save AI settings", err);
      setError(tUI("admin.aiSettings.saveError"));
    } finally {
      setSaving(false);
    }
  };

  const handleLimitChange = (key: keyof AiLimits, value: number) => {
    if (!editedLimits) return;
    setEditedLimits({ ...editedLimits, [key]: value });
  };

  const hasChanges =
    settings &&
    editedLimits &&
    (enabled !== settings.enabled ||
      editedLimits.free !== settings.limits.free ||
      editedLimits.course !== settings.limits.course ||
      editedLimits.premium !== settings.limits.premium ||
      editedLimits.promptEngineering !== settings.limits.promptEngineering);

  if (loading) {
    return (
      <div className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 dark:bg-dark-bg rounded w-1/3" />
          <div className="h-20 bg-gray-200 dark:bg-dark-bg rounded" />
          <div className="h-20 bg-gray-200 dark:bg-dark-bg rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <svg
              className="w-6 h-6 text-brand-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
            {tUI("admin.aiSettings.title")}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {tUI("admin.aiSettings.subtitle")}
          </p>
        </div>

        {/* Enable/Disable Toggle */}
        <button
          onClick={() => setEnabled(!enabled)}
          className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
            enabled ? "bg-brand-500" : "bg-gray-300 dark:bg-dark-border"
          }`}
        >
          <span
            className={`inline-block h-6 w-6 transform rounded-full bg-white shadow-lg transition-transform ${
              enabled ? "translate-x-7" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {/* Status Badge */}
      <div className="mb-6">
        <span
          className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            enabled
              ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
              : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full mr-2 ${enabled ? "bg-green-500" : "bg-red-500"}`}
          />
          {enabled
            ? tUI("admin.aiSettings.enabled")
            : tUI("admin.aiSettings.disabled")}
        </span>
      </div>

      {/* Error/Success Messages */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="mb-4 p-3 bg-green-100 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg text-green-700 dark:text-green-400 text-sm">
          {tUI("admin.aiSettings.saveSuccess")}
        </div>
      )}

      {/* Limits Section */}
      {editedLimits && (
        <div className="space-y-3">
          <h3 className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
            {tUI("admin.aiSettings.dailyLimits")}
          </h3>

          <LimitInput
            label={tUI("admin.aiSettings.freeTier")}
            value={editedLimits.free}
            onChange={(v) => handleLimitChange("free", v)}
            description={tUI("admin.aiSettings.freeTierDesc")}
          />

          <LimitInput
            label={tUI("admin.aiSettings.courseTier")}
            value={editedLimits.course}
            onChange={(v) => handleLimitChange("course", v)}
            description={tUI("admin.aiSettings.courseTierDesc")}
          />

          <LimitInput
            label={tUI("admin.aiSettings.premiumTier")}
            value={editedLimits.premium}
            onChange={(v) => handleLimitChange("premium", v)}
            description={tUI("admin.aiSettings.premiumTierDesc")}
          />

          <LimitInput
            label={tUI("admin.aiSettings.promptEngineeringTier")}
            value={editedLimits.promptEngineering}
            onChange={(v) => handleLimitChange("promptEngineering", v)}
            description={tUI("admin.aiSettings.promptEngineeringTierDesc")}
          />
        </div>
      )}

      {/* Save Button */}
      <div className="mt-6 flex justify-end">
        <button
          onClick={handleSave}
          disabled={!hasChanges || saving}
          className={`px-6 py-2.5 rounded-xl font-bold transition-all ${
            hasChanges && !saving
              ? "bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white shadow-lg shadow-brand-500/25 transform hover:-translate-y-0.5"
              : "bg-gray-200 dark:bg-dark-border text-gray-400 dark:text-gray-500 cursor-not-allowed"
          }`}
        >
          {saving ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              {tUI("admin.aiSettings.saving")}
            </span>
          ) : (
            tUI("admin.aiSettings.saveChanges")
          )}
        </button>
      </div>
    </div>
  );
};

export default AiSettingsPanel;
