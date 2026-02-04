import React, { useState, useEffect, useCallback } from "react";
import {
  adminService,
  PromoCodeItem,
  PromoCodeStats,
  PromoCodeType,
  PromoCodeApplicableTo,
} from "../api/adminService";
import { useUITranslation } from "@/contexts/LanguageContext";
import { createLogger } from "@/lib/logger";

const log = createLogger("PromoCodesPanel");

const PromoCodesPanel: React.FC = () => {
  const { tUI } = useUITranslation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [promoCodes, setPromoCodes] = useState<PromoCodeItem[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<PromoCodeStats | null>(null);
  const [filterActive, setFilterActive] = useState<string>("");

  // Create modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createForm, setCreateForm] = useState({
    code: "",
    type: "PERCENTAGE" as PromoCodeType,
    discount: 0,
    maxUses: "",
    maxUsesPerUser: "1",
    minPurchaseAmount: "",
    validFrom: "",
    validUntil: "",
    applicableTo: "ALL" as PromoCodeApplicableTo,
    description: "",
  });
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const formatAmount = (tiyn: number) => {
    const uzs = tiyn / 100;
    return new Intl.NumberFormat("uz-UZ").format(uzs) + " UZS";
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("ru-RU", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
  };

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [codesResponse, statsResponse] = await Promise.all([
        adminService.getPromoCodes({
          isActive: filterActive === "" ? undefined : filterActive === "true",
        }),
        adminService.getPromoCodeStats(),
      ]);
      setPromoCodes(codesResponse.promoCodes);
      setTotal(codesResponse.total);
      setStats(statsResponse);
    } catch (err) {
      log.error("Failed to load promo codes", err);
      setError(tUI("admin.promocodes.loadError"));
    } finally {
      setLoading(false);
    }
  }, [filterActive, tUI]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async () => {
    if (!createForm.code.trim() || createForm.discount <= 0) {
      setError(tUI("admin.promocodes.invalidForm"));
      return;
    }

    try {
      setActionLoading("create");
      await adminService.createPromoCode({
        code: createForm.code.trim(),
        type: createForm.type,
        discount: createForm.discount,
        maxUses: createForm.maxUses ? parseInt(createForm.maxUses) : undefined,
        maxUsesPerUser: createForm.maxUsesPerUser
          ? parseInt(createForm.maxUsesPerUser)
          : 1,
        minPurchaseAmount: createForm.minPurchaseAmount
          ? parseInt(createForm.minPurchaseAmount) * 100
          : undefined,
        validFrom: createForm.validFrom,
        validUntil: createForm.validUntil,
        applicableTo: createForm.applicableTo,
        description: createForm.description || undefined,
      });
      setShowCreateModal(false);
      setCreateForm({
        code: "",
        type: "PERCENTAGE",
        discount: 0,
        maxUses: "",
        maxUsesPerUser: "1",
        minPurchaseAmount: "",
        validFrom: "",
        validUntil: "",
        applicableTo: "ALL",
        description: "",
      });
      await loadData();
    } catch (err) {
      log.error("Failed to create promo code", err);
      setError(tUI("admin.promocodes.createError"));
    } finally {
      setActionLoading(null);
    }
  };

  const handleToggleActive = async (promoCode: PromoCodeItem) => {
    try {
      setActionLoading(promoCode.id);
      if (promoCode.isActive) {
        await adminService.deactivatePromoCode(promoCode.id);
      } else {
        await adminService.activatePromoCode(promoCode.id);
      }
      setPromoCodes((prev) =>
        prev.map((p) =>
          p.id === promoCode.id ? { ...p, isActive: !p.isActive } : p,
        ),
      );
    } catch (err) {
      log.error("Failed to toggle promo code", err);
      setError(tUI("admin.promocodes.toggleError"));
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (promoCode: PromoCodeItem) => {
    if (!window.confirm(tUI("admin.promocodes.confirmDelete"))) return;

    try {
      setActionLoading(promoCode.id);
      await adminService.deletePromoCode(promoCode.id);
      setPromoCodes((prev) => prev.filter((p) => p.id !== promoCode.id));
      setTotal((prev) => prev - 1);
    } catch (err) {
      log.error("Failed to delete promo code", err);
      setError(tUI("admin.promocodes.deleteError"));
    } finally {
      setActionLoading(null);
    }
  };

  const getTypeLabel = (type: PromoCodeType) => {
    switch (type) {
      case "PERCENTAGE":
        return tUI("admin.promocodes.typePercentage");
      case "FIXED":
        return tUI("admin.promocodes.typeFixed");
      case "FREE_TRIAL":
        return tUI("admin.promocodes.typeFreeTrial");
    }
  };

  const getDiscountDisplay = (promoCode: PromoCodeItem) => {
    switch (promoCode.type) {
      case "PERCENTAGE":
        return `${promoCode.discount}%`;
      case "FIXED":
        return formatAmount(promoCode.discount);
      case "FREE_TRIAL":
        return `${promoCode.discount} ${tUI("admin.promocodes.days")}`;
    }
  };

  const isExpired = (validUntil: string) => {
    return new Date(validUntil) < new Date();
  };

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
                d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
              />
            </svg>
            {tUI("admin.promocodes.title")}
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {tUI("admin.promocodes.subtitle")}
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all"
        >
          {tUI("admin.promocodes.create")}
        </button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
          <div className="p-3 bg-gray-50 dark:bg-dark-bg rounded-xl">
            <div className="text-xs text-gray-500">{tUI("admin.promocodes.totalCodes")}</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {stats.total}
            </div>
          </div>
          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-xl">
            <div className="text-xs text-green-600">{tUI("admin.promocodes.activeCodes")}</div>
            <div className="text-xl font-bold text-green-700 dark:text-green-400">
              {stats.active}
            </div>
          </div>
          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-xl">
            <div className="text-xs text-red-600">{tUI("admin.promocodes.expiredCodes")}</div>
            <div className="text-xl font-bold text-red-700 dark:text-red-400">
              {stats.expired}
            </div>
          </div>
          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
            <div className="text-xs text-blue-600">{tUI("admin.promocodes.totalUsages")}</div>
            <div className="text-xl font-bold text-blue-700 dark:text-blue-400">
              {stats.totalUsages}
            </div>
          </div>
          <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
            <div className="text-xs text-purple-600">{tUI("admin.promocodes.totalDiscount")}</div>
            <div className="text-xl font-bold text-purple-700 dark:text-purple-400">
              {formatAmount(stats.totalDiscountGiven)}
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-4 mb-4">
        <select
          value={filterActive}
          onChange={(e) => setFilterActive(e.target.value)}
          className="px-3 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-lg text-sm"
        >
          <option value="">{tUI("admin.promocodes.allCodes")}</option>
          <option value="true">{tUI("admin.promocodes.activeOnly")}</option>
          <option value="false">{tUI("admin.promocodes.inactiveOnly")}</option>
        </select>
        <div className="ml-auto text-sm text-gray-500 dark:text-gray-400">
          {tUI("admin.payments.total")}: {total}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex justify-center py-8">
          <svg
            className="animate-spin h-8 w-8 text-brand-500"
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
        </div>
      )}

      {/* Promo Codes List */}
      {!loading && (
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {promoCodes.map((promoCode) => (
            <div
              key={promoCode.id}
              className={`p-4 rounded-xl transition-colors ${
                !promoCode.isActive || isExpired(promoCode.validUntil)
                  ? "bg-gray-100 dark:bg-gray-800/50 opacity-60"
                  : "bg-gray-50 dark:bg-dark-bg"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono font-bold text-lg text-gray-900 dark:text-white">
                      {promoCode.code}
                    </span>
                    <span
                      className={`px-2 py-0.5 text-xs font-medium rounded ${
                        promoCode.isActive && !isExpired(promoCode.validUntil)
                          ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
                          : "bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                      }`}
                    >
                      {promoCode.isActive && !isExpired(promoCode.validUntil)
                        ? tUI("admin.promocodes.active")
                        : isExpired(promoCode.validUntil)
                          ? tUI("admin.promocodes.expired")
                          : tUI("admin.promocodes.inactive")}
                    </span>
                    <span className="px-2 py-0.5 text-xs font-medium rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                      {getTypeLabel(promoCode.type)}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {tUI("admin.promocodes.discount")}:{" "}
                    <span className="font-bold text-brand-600">
                      {getDiscountDisplay(promoCode)}
                    </span>
                    {promoCode.maxUses && (
                      <>
                        {" "}
                        • {tUI("admin.promocodes.uses")}: {promoCode.usesCount}/
                        {promoCode.maxUses}
                      </>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {formatDate(promoCode.validFrom)} -{" "}
                    {formatDate(promoCode.validUntil)}
                    {promoCode.description && (
                      <> • {promoCode.description}</>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleToggleActive(promoCode)}
                    disabled={actionLoading === promoCode.id}
                    className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors disabled:opacity-50 ${
                      promoCode.isActive
                        ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 hover:bg-yellow-200"
                        : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 hover:bg-green-200"
                    }`}
                  >
                    {actionLoading === promoCode.id
                      ? "..."
                      : promoCode.isActive
                        ? tUI("admin.promocodes.deactivate")
                        : tUI("admin.promocodes.activate")}
                  </button>
                  {promoCode.usesCount === 0 && (
                    <button
                      onClick={() => handleDelete(promoCode)}
                      disabled={actionLoading === promoCode.id}
                      className="px-3 py-1.5 text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 font-medium rounded-lg hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors disabled:opacity-50"
                    >
                      {tUI("common.delete")}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
          {promoCodes.length === 0 && (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              {tUI("admin.promocodes.noCodes")}
            </div>
          )}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          onClick={() => setShowCreateModal(false)}
        >
          <div
            className="relative w-full max-w-lg bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border shadow-2xl p-6 max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              {tUI("admin.promocodes.createTitle")}
            </h3>

            <div className="space-y-4">
              {/* Code */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {tUI("admin.promocodes.codeLabel")}
                </label>
                <input
                  type="text"
                  value={createForm.code}
                  onChange={(e) =>
                    setCreateForm({
                      ...createForm,
                      code: e.target.value.toUpperCase(),
                    })
                  }
                  placeholder="SUMMER2026"
                  className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl font-mono"
                />
              </div>

              {/* Type & Discount */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI("admin.promocodes.typeLabel")}
                  </label>
                  <select
                    value={createForm.type}
                    onChange={(e) =>
                      setCreateForm({
                        ...createForm,
                        type: e.target.value as PromoCodeType,
                      })
                    }
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  >
                    <option value="PERCENTAGE">
                      {tUI("admin.promocodes.typePercentage")}
                    </option>
                    <option value="FIXED">{tUI("admin.promocodes.typeFixed")}</option>
                    <option value="FREE_TRIAL">
                      {tUI("admin.promocodes.typeFreeTrial")}
                    </option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {createForm.type === "PERCENTAGE"
                      ? tUI("admin.promocodes.discountPercent")
                      : createForm.type === "FIXED"
                        ? tUI("admin.promocodes.discountAmount")
                        : tUI("admin.promocodes.trialDays")}
                  </label>
                  <input
                    type="number"
                    value={createForm.discount}
                    onChange={(e) =>
                      setCreateForm({
                        ...createForm,
                        discount: parseInt(e.target.value) || 0,
                      })
                    }
                    min={1}
                    max={createForm.type === "PERCENTAGE" ? 100 : undefined}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  />
                </div>
              </div>

              {/* Dates */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI("admin.promocodes.validFrom")}
                  </label>
                  <input
                    type="date"
                    value={createForm.validFrom}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, validFrom: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI("admin.promocodes.validUntil")}
                  </label>
                  <input
                    type="date"
                    value={createForm.validUntil}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, validUntil: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  />
                </div>
              </div>

              {/* Limits */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI("admin.promocodes.maxUses")}
                  </label>
                  <input
                    type="number"
                    value={createForm.maxUses}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, maxUses: e.target.value })
                    }
                    placeholder={tUI("admin.promocodes.unlimited")}
                    min={1}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI("admin.promocodes.maxUsesPerUser")}
                  </label>
                  <input
                    type="number"
                    value={createForm.maxUsesPerUser}
                    onChange={(e) =>
                      setCreateForm({
                        ...createForm,
                        maxUsesPerUser: e.target.value,
                      })
                    }
                    min={1}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                  />
                </div>
              </div>

              {/* Applicable To */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {tUI("admin.promocodes.applicableTo")}
                </label>
                <select
                  value={createForm.applicableTo}
                  onChange={(e) =>
                    setCreateForm({
                      ...createForm,
                      applicableTo: e.target.value as PromoCodeApplicableTo,
                    })
                  }
                  className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                >
                  <option value="ALL">{tUI("admin.promocodes.applicableAll")}</option>
                  <option value="SUBSCRIPTIONS">
                    {tUI("admin.promocodes.applicableSubscriptions")}
                  </option>
                  <option value="PURCHASES">
                    {tUI("admin.promocodes.applicablePurchases")}
                  </option>
                  <option value="COURSES">
                    {tUI("admin.promocodes.applicableCourses")}
                  </option>
                </select>
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {tUI("admin.promocodes.descriptionLabel")}
                </label>
                <input
                  type="text"
                  value={createForm.description}
                  onChange={(e) =>
                    setCreateForm({ ...createForm, description: e.target.value })
                  }
                  placeholder={tUI("admin.promocodes.descriptionPlaceholder")}
                  className="w-full px-4 py-2 bg-gray-100 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl"
                />
              </div>
            </div>

            <div className="flex gap-3 justify-end mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                {tUI("common.cancel")}
              </button>
              <button
                onClick={handleCreate}
                disabled={
                  actionLoading === "create" ||
                  !createForm.code.trim() ||
                  createForm.discount <= 0 ||
                  !createForm.validFrom ||
                  !createForm.validUntil
                }
                className="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
              >
                {actionLoading === "create"
                  ? "..."
                  : tUI("admin.promocodes.createBtn")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PromoCodesPanel;
