import React, { useState, useEffect, useCallback } from "react";
import {
  adminService,
  BugReport,
  BugStatus,
  BugSeverity,
  BugCategory,
} from "../api/adminService";
import { useUITranslation } from "@/contexts/LanguageContext";
import { createLogger } from "@/lib/logger";

const log = createLogger("BugReportsPanel");

const STATUS_COLORS: Record<BugStatus, string> = {
  open: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  "in-progress":
    "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
  resolved:
    "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  closed: "bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400",
  "wont-fix": "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
};

const SEVERITY_COLORS: Record<BugSeverity, string> = {
  low: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  medium:
    "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
  high: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
};

const ALL_STATUSES: BugStatus[] = [
  "open",
  "in-progress",
  "resolved",
  "closed",
  "wont-fix",
];

const BugReportsPanel: React.FC = () => {
  const { tUI } = useUITranslation();
  const [loading, setLoading] = useState(true);
  const [reports, setReports] = useState<BugReport[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<BugStatus | "">("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [updatingId, setUpdatingId] = useState<string | null>(null);

  const loadReports = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const filters = statusFilter ? { status: statusFilter } : undefined;
      const data = await adminService.getBugReports(filters);
      setReports(data);
    } catch (err) {
      log.error("Failed to load bug reports", err);
      setError(tUI("admin.bugReports.loadError"));
    } finally {
      setLoading(false);
    }
  }, [statusFilter, tUI]);

  useEffect(() => {
    loadReports();
  }, [loadReports]);

  const handleStatusChange = async (id: string, newStatus: BugStatus) => {
    try {
      setUpdatingId(id);
      await adminService.updateBugReportStatus(id, newStatus);
      setReports((prev) =>
        prev.map((r) => (r.id === id ? { ...r, status: newStatus } : r)),
      );
    } catch (err) {
      log.error("Failed to update status", err);
    } finally {
      setUpdatingId(null);
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-dark-surface p-6 rounded-2xl border border-gray-100 dark:border-dark-border shadow-sm">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 dark:bg-dark-bg rounded w-1/3" />
          <div className="h-12 bg-gray-200 dark:bg-dark-bg rounded" />
          <div className="h-12 bg-gray-200 dark:bg-dark-bg rounded" />
          <div className="h-12 bg-gray-200 dark:bg-dark-bg rounded" />
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
              className="w-6 h-6 text-red-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            {tUI("admin.bugReports.title")}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {tUI("admin.bugReports.subtitle")} ({reports.length})
          </p>
        </div>

        {/* Filter */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as BugStatus | "")}
          className="px-3 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-lg text-sm"
        >
          <option value="">{tUI("admin.bugReports.allStatuses")}</option>
          {ALL_STATUSES.map((s) => (
            <option key={s} value={s}>
              {tUI(`admin.bugReports.status.${s}`)}
            </option>
          ))}
        </select>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Reports List */}
      {reports.length === 0 ? (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          {tUI("admin.bugReports.noReports")}
        </div>
      ) : (
        <div className="space-y-3 max-h-[500px] overflow-y-auto">
          {reports.map((report) => (
            <div
              key={report.id}
              className="border border-gray-100 dark:border-dark-border rounded-xl overflow-hidden"
            >
              {/* Report Header */}
              <div
                className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors"
                onClick={() =>
                  setExpandedId(expandedId === report.id ? null : report.id)
                }
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_COLORS[report.severity]}`}
                      >
                        {report.severity.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {report.category}
                      </span>
                    </div>
                    <h3 className="font-medium text-gray-900 dark:text-white truncate">
                      {report.title}
                    </h3>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {report.user.name || report.user.email} •{" "}
                      {formatDate(report.createdAt)}
                    </div>
                  </div>

                  {/* Status Dropdown */}
                  <div className="flex items-center gap-2">
                    <select
                      value={report.status}
                      onChange={(e) => {
                        e.stopPropagation();
                        handleStatusChange(report.id, e.target.value as BugStatus);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      disabled={updatingId === report.id}
                      className={`px-2 py-1 rounded text-xs font-medium border-0 cursor-pointer ${STATUS_COLORS[report.status]}`}
                    >
                      {ALL_STATUSES.map((s) => (
                        <option key={s} value={s}>
                          {tUI(`admin.bugReports.status.${s}`)}
                        </option>
                      ))}
                    </select>
                    <svg
                      className={`w-5 h-5 text-gray-400 transition-transform ${expandedId === report.id ? "rotate-180" : ""}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {expandedId === report.id && (
                <div className="px-4 pb-4 border-t border-gray-100 dark:border-dark-border bg-gray-50 dark:bg-dark-bg/30">
                  <div className="pt-4 space-y-3">
                    <div>
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                        {tUI("admin.bugReports.description")}
                      </div>
                      <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                        {report.description}
                      </p>
                    </div>

                    {report.task && (
                      <div>
                        <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                          {tUI("admin.bugReports.relatedTask")}
                        </div>
                        <a
                          href={`/task/${report.task.slug}`}
                          className="text-sm text-brand-500 hover:underline"
                        >
                          {report.task.title}
                        </a>
                      </div>
                    )}

                    {report.metadata && Object.keys(report.metadata).length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                          {tUI("admin.bugReports.metadata")}
                        </div>
                        <pre className="text-xs bg-gray-100 dark:bg-dark-bg p-2 rounded overflow-x-auto">
                          {JSON.stringify(report.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BugReportsPanel;
