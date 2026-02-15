import React, { useContext, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  paymentService,
  PaymentHistoryItem,
  PaymentProvider,
  PromoCodeValidation,
} from "../api/paymentService";
import { subscriptionService } from "@/features/subscriptions/api/subscriptionService";
import { SubscriptionPlan } from "@/features/subscriptions/model/types";
import { AuthContext } from "@/components/Layout";
import { useLanguage } from "@/contexts/LanguageContext";
import { IconCrown, IconRocket, IconLightning } from "@/components/Icons";
import { createLogger } from "@/lib/logger";
import StatusCard from "./components/StatusCard";
import CheckoutPanel from "./components/CheckoutPanel";
import PaymentHistoryTab from "./components/PaymentHistoryTab";

const log = createLogger("Payments");

const formatPrice = (amountInTiyn: number): string => {
  const uzs = amountInTiyn / 100;
  return new Intl.NumberFormat("uz-UZ").format(uzs);
};

type TabType = "subscribe" | "purchases" | "history";

const PaymentsPage = () => {
  const { user } = useContext(AuthContext);
  const { language } = useLanguage();
  const [searchParams] = useSearchParams();

  const [activeTab, setActiveTab] = useState<TabType>("subscribe");
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [providers, setProviders] = useState<PaymentProvider[]>([]);
  const [history, setHistory] = useState<PaymentHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [checkoutLoading, setCheckoutLoading] = useState(false);

  const [selectedPlanId, setSelectedPlanId] = useState<string>("");
  const [selectedProvider, setSelectedProvider] = useState<"payme" | "click">(
    "payme",
  );
  const [purchaseType, setPurchaseType] = useState<"subscription" | "roadmap">(
    "subscription",
  );

  const [promoCode, setPromoCode] = useState<string>("");
  const [promoValidation, setPromoValidation] =
    useState<PromoCodeValidation | null>(null);
  const [promoLoading, setPromoLoading] = useState(false);

  const globalPlan = plans.find((p) => p.type === "global");
  const coursePlans = plans.filter((p) => p.type === "course");
  const selectedPlan = plans.find((p) => p.id === selectedPlanId);

  useEffect(() => {
    if (user) {
      Promise.all([
        subscriptionService.getPlans(),
        paymentService.getProviders(),
        paymentService.getPaymentHistory(),
      ])
        .then(([plansData, providersData, historyData]) => {
          setPlans(plansData);
          setProviders(providersData);
          setHistory(historyData);

          const planFromUrl = searchParams.get("plan");
          if (planFromUrl) {
            const plan = plansData.find(
              (p) => p.slug === planFromUrl || p.id === planFromUrl,
            );
            if (plan) setSelectedPlanId(plan.id);
          }
          setLoading(false);
        })
        .catch((error) => {
          log.error("Failed to load payment data", error);
          setLoading(false);
        });
    }
  }, [user, searchParams]);

  const handleValidatePromo = async () => {
    if (!promoCode.trim() || !selectedPlan) return;
    setPromoLoading(true);
    try {
      const result = await paymentService.validatePromoCode(
        promoCode.trim(),
        purchaseType === "subscription" ? "subscription" : "purchase",
        selectedPlan.priceMonthly,
      );
      setPromoValidation(result);
    } catch (error) {
      log.error("Promo validation failed", error);
      setPromoValidation({
        valid: false,
        error: "Failed to validate promo code",
      });
    } finally {
      setPromoLoading(false);
    }
  };

  const handlePlanChange = (planId: string) => {
    setSelectedPlanId(planId);
    setPromoCode("");
    setPromoValidation(null);
  };

  const handleCheckout = async () => {
    if (!selectedPlanId && purchaseType === "subscription") return;
    setCheckoutLoading(true);
    try {
      const response = await paymentService.createCheckout({
        orderType:
          purchaseType === "subscription" ? "subscription" : "purchase",
        planId: purchaseType === "subscription" ? selectedPlanId : undefined,
        purchaseType:
          purchaseType === "roadmap" ? "roadmap_generation" : undefined,
        quantity: 1,
        provider: selectedProvider,
        returnUrl: window.location.origin + "/payments?status=success",
        promoCode: promoValidation?.valid ? promoCode.trim() : undefined,
      });
      window.location.href = response.paymentUrl;
    } catch (error) {
      log.error("Checkout failed", error);
      setCheckoutLoading(false);
    }
  };

  const getFinalPrice = () => {
    if (!selectedPlan) return 0;
    if (promoValidation?.valid && promoValidation.discountAmount) {
      return Math.max(
        0,
        selectedPlan.priceMonthly - promoValidation.discountAmount,
      );
    }
    return selectedPlan.priceMonthly;
  };

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto p-8" data-testid="loading-spinner">
        <div className="animate-pulse space-y-8">
          <div className="h-8 bg-gray-200 dark:bg-dark-border rounded w-1/3"></div>
          <div className="h-64 bg-gray-200 dark:bg-dark-border rounded-3xl"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1
          className="text-3xl font-display font-bold text-gray-900 dark:text-white"
          data-testid="payments-title"
        >
          {language === "ru"
            ? "Подписки и платежи"
            : language === "uz"
              ? "Obuna va to'lovlar"
              : "Subscriptions & Payments"}
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          {language === "ru"
            ? "Управляйте подписками и просматривайте историю платежей"
            : language === "uz"
              ? "Obunalarni boshqaring va to'lovlar tarixini ko'ring"
              : "Manage subscriptions and view payment history"}
        </p>
      </div>

      <StatusCard
        isPremium={!!user?.isPremium}
        expiresAt={user?.plan?.expiresAt}
        language={language}
      />

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200 dark:border-dark-border">
        {[
          {
            id: "subscribe",
            label: language === "ru" ? "Подписки" : "Subscribe",
            icon: IconCrown,
            testId: "subscribe-tab",
          },
          {
            id: "purchases",
            label: language === "ru" ? "Покупки" : "Purchases",
            icon: IconRocket,
            testId: "purchases-tab",
          },
          {
            id: "history",
            label: language === "ru" ? "История" : "History",
            icon: IconLightning,
            testId: "history-tab",
          },
        ].map((tab) => (
          <button
            key={tab.id}
            data-testid={tab.testId}
            onClick={() => setActiveTab(tab.id as TabType)}
            className={`flex items-center gap-2 px-6 py-3 font-bold text-sm transition-all border-b-2 -mb-[2px] ${
              activeTab === tab.id
                ? "border-brand-500 text-brand-600 dark:text-brand-400"
                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "subscribe" && (
        <div
          className="grid lg:grid-cols-3 gap-8"
          data-testid="subscription-plans"
        >
          <div className="lg:col-span-2 space-y-6">
            {/* Global Premium */}
            {globalPlan && (
              <PlanCard
                plan={globalPlan}
                isSelected={selectedPlanId === globalPlan.id}
                onSelect={() => {
                  handlePlanChange(globalPlan.id);
                  setPurchaseType("subscription");
                }}
                isGlobal
                language={language}
                testId="plan-global-premium"
              />
            )}
            {/* Course Plans */}
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                {language === "ru"
                  ? "Или выберите отдельный курс"
                  : "Or select a single course"}
              </h3>
              <div className="space-y-3">
                {coursePlans.map((plan, index) => (
                  <PlanCard
                    key={plan.id}
                    plan={plan}
                    isSelected={selectedPlanId === plan.id}
                    onSelect={() => {
                      handlePlanChange(plan.id);
                      setPurchaseType("subscription");
                    }}
                    language={language}
                    testId={
                      index === 0
                        ? "plan-course-premium"
                        : `plan-course-${index}`
                    }
                  />
                ))}
              </div>
            </div>
          </div>

          <CheckoutPanel
            selectedPlan={selectedPlan}
            providers={providers}
            selectedProvider={selectedProvider}
            onProviderChange={setSelectedProvider}
            promoCode={promoCode}
            onPromoCodeChange={setPromoCode}
            promoValidation={promoValidation}
            onPromoValidationChange={setPromoValidation}
            promoLoading={promoLoading}
            onValidatePromo={handleValidatePromo}
            checkoutLoading={checkoutLoading}
            onCheckout={handleCheckout}
            getFinalPrice={getFinalPrice}
            language={language}
          />
        </div>
      )}

      {activeTab === "purchases" && (
        <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            {language === "ru" ? "Разовые покупки" : "One-time Purchases"}
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div
              data-testid="plan-roadmap-credits"
              className="p-6 rounded-xl border border-gray-200 dark:border-dark-border"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center text-white text-2xl">
                  🗺️
                </div>
                <div>
                  <h4 className="font-bold text-gray-900 dark:text-white">
                    {language === "ru"
                      ? "Генерация Roadmap"
                      : "Roadmap Generation"}
                  </h4>
                  <p className="text-sm text-gray-500">
                    {language === "ru"
                      ? "Персональный план обучения"
                      : "Personal learning plan"}
                  </p>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  15,000{" "}
                  <span className="text-sm font-normal text-gray-500">UZS</span>
                </div>
                <button
                  onClick={() => {
                    setPurchaseType("roadmap");
                    setSelectedPlanId("");
                    setActiveTab("subscribe");
                  }}
                  className="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white font-bold text-sm rounded-lg transition-colors"
                >
                  {language === "ru" ? "Купить" : "Buy"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "history" && (
        <PaymentHistoryTab history={history} language={language} />
      )}
    </div>
  );
};

// Plan card component
const PlanCard: React.FC<{
  plan: SubscriptionPlan;
  isSelected: boolean;
  onSelect: () => void;
  isGlobal?: boolean;
  language: string;
  testId: string;
}> = ({ plan, isSelected, onSelect, isGlobal, language, testId }) => {
  if (isGlobal) {
    return (
      <div
        data-testid={testId}
        data-selected={isSelected ? "true" : "false"}
        onClick={onSelect}
        className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${
          isSelected
            ? "border-brand-500 bg-brand-50 dark:bg-brand-900/20"
            : "border-gray-200 dark:border-dark-border hover:border-brand-300 bg-white dark:bg-dark-surface"
        }`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white text-2xl">
              👑
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  {language === "ru" ? "Глобальный Premium" : "Global Premium"}
                </h3>
                <span className="px-2 py-0.5 bg-gradient-to-r from-amber-400 to-orange-500 text-white text-xs font-bold rounded-full">
                  BEST VALUE
                </span>
              </div>
              <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
                {language === "ru"
                  ? "Доступ ко всем курсам и функциям платформы"
                  : "Access to all courses and platform features"}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatPrice(plan.priceMonthly)}{" "}
              <span className="text-sm font-normal text-gray-500">UZS</span>
            </div>
            <div className="text-xs text-gray-500">
              {language === "ru" ? "/ месяц" : "/ month"}
            </div>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {[
            "Все курсы",
            "100 AI запросов/день",
            "Приоритетная очередь",
            "Без рекламы",
          ].map((feature) => (
            <span
              key={feature}
              className="px-3 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-medium rounded-full"
            >
              {feature}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid={testId}
      data-selected={isSelected ? "true" : "false"}
      onClick={onSelect}
      className={`p-4 rounded-xl border-2 cursor-pointer transition-all flex items-center justify-between ${
        isSelected
          ? "border-brand-500 bg-brand-50 dark:bg-brand-900/20"
          : "border-gray-200 dark:border-dark-border hover:border-brand-300 bg-white dark:bg-dark-surface"
      }`}
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gray-100 dark:bg-dark-bg flex items-center justify-center text-xl">
          {plan.course?.icon || "📚"}
        </div>
        <div>
          <h4 className="font-bold text-gray-900 dark:text-white">
            {language === "ru" ? plan.nameRu || plan.name : plan.name}
          </h4>
          <p className="text-xs text-gray-500">{plan.course?.title}</p>
        </div>
      </div>
      <div className="text-right">
        <div className="font-bold text-gray-900 dark:text-white">
          {formatPrice(plan.priceMonthly)}{" "}
          <span className="text-xs font-normal text-gray-500">UZS</span>
        </div>
      </div>
    </div>
  );
};

export default PaymentsPage;
