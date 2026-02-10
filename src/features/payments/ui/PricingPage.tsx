import React, { useEffect, useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { subscriptionService } from "@/features/subscriptions/api/subscriptionService";
import { SubscriptionPlan } from "@/features/subscriptions/model/types";
import { AuthContext } from "@/components/Layout";
import { useUITranslation } from "@/contexts/LanguageContext";
import { IconCheck, IconX } from "@/components/Icons";

const formatPrice = (amountInTiyn: number): string => {
  const uzs = amountInTiyn / 100;
  return new Intl.NumberFormat("uz-UZ").format(uzs);
};

// Feature comparison data
const FEATURES = [
  { key: "courses", free: "3 курса", course: "1 курс", premium: "Все курсы" },
  { key: "tasks", free: true, course: true, premium: true },
  { key: "aiTutor", free: "5/день", course: "30/день", premium: "100/день" },
  { key: "playground", free: true, course: true, premium: true },
  { key: "codeSharing", free: true, course: true, premium: true },
  { key: "leaderboard", free: true, course: true, premium: true },
  { key: "certificates", free: false, course: true, premium: true },
  { key: "priorityQueue", free: false, course: false, premium: true },
  { key: "noAds", free: false, course: false, premium: true },
  { key: "earlyAccess", free: false, course: false, premium: true },
];

const PricingPage: React.FC = () => {
  const { tUI } = useUITranslation();
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [billingCycle, setBillingCycle] = useState<"monthly" | "yearly">(
    "monthly",
  );

  const globalPlan = plans.find((p) => p.type === "global");
  const coursePlans = plans.filter((p) => p.type === "course");

  useEffect(() => {
    subscriptionService
      .getPlans()
      .then(setPlans)
      .finally(() => setLoading(false));
  }, []);

  const handleSelectPlan = (planSlug?: string) => {
    if (!user) {
      navigate(
        "/login?redirect=/payments" + (planSlug ? `?plan=${planSlug}` : ""),
      );
      return;
    }
    navigate("/payments" + (planSlug ? `?plan=${planSlug}` : ""));
  };

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-display font-bold text-gray-900 dark:text-white mb-4">
          {tUI("pricing.title")}
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          {tUI("pricing.subtitle")}
        </p>

        {/* Billing Toggle */}
        <div className="flex items-center justify-center gap-4 mt-8">
          <span
            className={`text-sm font-medium ${billingCycle === "monthly" ? "text-gray-900 dark:text-white" : "text-gray-500"}`}
          >
            {tUI("pricing.monthly")}
          </span>
          <button
            onClick={() =>
              setBillingCycle(billingCycle === "monthly" ? "yearly" : "monthly")
            }
            className="relative w-14 h-7 bg-gray-200 dark:bg-gray-700 rounded-full transition-colors"
          >
            <div
              className={`absolute top-1 w-5 h-5 bg-brand-500 rounded-full transition-transform ${
                billingCycle === "yearly" ? "translate-x-8" : "translate-x-1"
              }`}
            />
          </button>
          <span
            className={`text-sm font-medium ${billingCycle === "yearly" ? "text-gray-900 dark:text-white" : "text-gray-500"}`}
          >
            {tUI("pricing.yearly")}
            <span className="ml-2 px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-xs font-bold rounded-full">
              -20%
            </span>
          </span>
        </div>
      </div>

      {/* Pricing Cards */}
      <div className="grid md:grid-cols-3 gap-6 mb-16">
        {/* Free Plan */}
        <div className="bg-white dark:bg-dark-surface rounded-3xl border border-gray-200 dark:border-dark-border p-8 flex flex-col">
          <div className="mb-6">
            <div className="text-4xl mb-3">🆓</div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
              {tUI("pricing.free")}
            </h3>
            <p className="text-gray-500 dark:text-gray-400 text-sm mt-2">
              {tUI("pricing.freeDesc")}
            </p>
          </div>

          <div className="mb-6">
            <div className="text-4xl font-display font-bold text-gray-900 dark:text-white">
              0 <span className="text-lg font-normal text-gray-500">UZS</span>
            </div>
            <div className="text-sm text-gray-500">
              {tUI("pricing.forever")}
            </div>
          </div>

          <ul className="space-y-3 mb-8 flex-1">
            {[
              "3 бесплатных курса",
              "5 AI запросов/день",
              "Playground",
              "Leaderboard",
            ].map((feature) => (
              <li
                key={feature}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400"
              >
                <IconCheck className="w-4 h-4 text-green-500 flex-shrink-0" />
                {feature}
              </li>
            ))}
          </ul>

          <button
            data-testid="select-free"
            onClick={() => navigate("/courses")}
            className="w-full py-3 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-bold rounded-xl transition-colors"
          >
            {tUI("pricing.startFree")}
          </button>
        </div>

        {/* Course Plan */}
        <div className="bg-white dark:bg-dark-surface rounded-3xl border border-gray-200 dark:border-dark-border p-8 flex flex-col">
          <div className="mb-6">
            <div className="text-4xl mb-3">📚</div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
              {tUI("pricing.course")}
            </h3>
            <p className="text-gray-500 dark:text-gray-400 text-sm mt-2">
              {tUI("pricing.courseDesc")}
            </p>
          </div>

          <div className="mb-6">
            <div className="text-4xl font-display font-bold text-gray-900 dark:text-white">
              {coursePlans[0]
                ? formatPrice(
                    billingCycle === "yearly"
                      ? Math.round(coursePlans[0].priceMonthly * 0.8)
                      : coursePlans[0].priceMonthly,
                  )
                : "49,000"}{" "}
              <span className="text-lg font-normal text-gray-500">UZS</span>
            </div>
            <div className="text-sm text-gray-500">
              {billingCycle === "yearly"
                ? tUI("pricing.perYear")
                : tUI("pricing.perMonth")}
            </div>
          </div>

          <ul className="space-y-3 mb-8 flex-1">
            {[
              "1 курс на выбор",
              "30 AI запросов/день",
              "Сертификат",
              "Приоритетная поддержка",
            ].map((feature) => (
              <li
                key={feature}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400"
              >
                <IconCheck className="w-4 h-4 text-green-500 flex-shrink-0" />
                {feature}
              </li>
            ))}
          </ul>

          <button
            data-testid="select-course"
            onClick={() => handleSelectPlan(coursePlans[0]?.slug)}
            className="w-full py-3 bg-gray-900 dark:bg-white hover:bg-gray-800 dark:hover:bg-gray-100 text-white dark:text-gray-900 font-bold rounded-xl transition-colors"
          >
            {tUI("pricing.selectCourse")}
          </button>
        </div>

        {/* Premium Plan */}
        <div className="bg-gradient-to-br from-brand-600 to-purple-600 rounded-3xl p-8 flex flex-col text-white relative overflow-hidden">
          {/* Popular badge */}
          <div className="absolute top-4 right-4 px-3 py-1 bg-white/20 backdrop-blur-sm text-white text-xs font-bold rounded-full">
            {tUI("pricing.popular")}
          </div>

          <div className="mb-6">
            <div className="text-4xl mb-3">👑</div>
            <h3 className="text-2xl font-bold">{tUI("pricing.premium")}</h3>
            <p className="text-white/80 text-sm mt-2">
              {tUI("pricing.premiumDesc")}
            </p>
          </div>

          <div className="mb-6">
            <div className="text-4xl font-display font-bold">
              {globalPlan
                ? formatPrice(
                    billingCycle === "yearly"
                      ? Math.round(globalPlan.priceMonthly * 0.8)
                      : globalPlan.priceMonthly,
                  )
                : "149,000"}{" "}
              <span className="text-lg font-normal text-white/80">UZS</span>
            </div>
            <div className="text-sm text-white/80">
              {billingCycle === "yearly"
                ? tUI("pricing.perYear")
                : tUI("pricing.perMonth")}
            </div>
          </div>

          <ul className="space-y-3 mb-8 flex-1">
            {[
              "Все курсы (21+)",
              "100 AI запросов/день",
              "Приоритетная очередь",
              "Без рекламы",
              "Ранний доступ",
            ].map((feature) => (
              <li
                key={feature}
                className="flex items-center gap-2 text-sm text-white/90"
              >
                <IconCheck className="w-4 h-4 text-white flex-shrink-0" />
                {feature}
              </li>
            ))}
          </ul>

          <button
            data-testid="select-premium"
            onClick={() => handleSelectPlan(globalPlan?.slug)}
            className="w-full py-3 bg-white hover:bg-gray-100 text-brand-600 font-bold rounded-xl transition-colors shadow-lg"
          >
            {tUI("pricing.getPremium")}
          </button>

          {/* Background decoration */}
          <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-white/10 rounded-full blur-3xl pointer-events-none" />
        </div>
      </div>

      {/* Feature Comparison Table */}
      <div className="bg-white dark:bg-dark-surface rounded-3xl border border-gray-200 dark:border-dark-border overflow-hidden">
        <div className="p-6 border-b border-gray-200 dark:border-dark-border">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            {tUI("pricing.compareFeatures")}
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 dark:bg-dark-bg">
                <th className="px-6 py-4 text-left text-sm font-bold text-gray-500 uppercase tracking-wider">
                  {tUI("pricing.feature")}
                </th>
                <th className="px-6 py-4 text-center text-sm font-bold text-gray-500 uppercase tracking-wider">
                  {tUI("pricing.free")}
                </th>
                <th className="px-6 py-4 text-center text-sm font-bold text-gray-500 uppercase tracking-wider">
                  {tUI("pricing.course")}
                </th>
                <th className="px-6 py-4 text-center text-sm font-bold text-brand-600 uppercase tracking-wider">
                  {tUI("pricing.premium")}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-dark-border">
              {FEATURES.map((feature) => (
                <tr
                  key={feature.key}
                  className="hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors"
                >
                  <td className="px-6 py-4 text-sm font-medium text-gray-900 dark:text-white">
                    {tUI(`pricing.features.${feature.key}`)}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {typeof feature.free === "boolean" ? (
                      feature.free ? (
                        <IconCheck className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <IconX className="w-5 h-5 text-gray-300 dark:text-gray-600 mx-auto" />
                      )
                    ) : (
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {feature.free}
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {typeof feature.course === "boolean" ? (
                      feature.course ? (
                        <IconCheck className="w-5 h-5 text-green-500 mx-auto" />
                      ) : (
                        <IconX className="w-5 h-5 text-gray-300 dark:text-gray-600 mx-auto" />
                      )
                    ) : (
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {feature.course}
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center bg-brand-50/50 dark:bg-brand-900/10">
                    {typeof feature.premium === "boolean" ? (
                      feature.premium ? (
                        <IconCheck className="w-5 h-5 text-brand-500 mx-auto" />
                      ) : (
                        <IconX className="w-5 h-5 text-gray-300 dark:text-gray-600 mx-auto" />
                      )
                    ) : (
                      <span className="text-sm font-medium text-brand-600 dark:text-brand-400">
                        {feature.premium}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-8">
          {tUI("pricing.faq")}
        </h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {[
            { q: "pricing.faq.cancel", a: "pricing.faq.cancelAnswer" },
            { q: "pricing.faq.refund", a: "pricing.faq.refundAnswer" },
            { q: "pricing.faq.upgrade", a: "pricing.faq.upgradeAnswer" },
            { q: "pricing.faq.payment", a: "pricing.faq.paymentAnswer" },
          ].map((item, i) => (
            <div
              key={i}
              className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border p-6"
            >
              <h3 className="font-bold text-gray-900 dark:text-white mb-2">
                {tUI(item.q)}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {tUI(item.a)}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="mt-16 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm font-medium mb-4">
          <span>🔒</span> {tUI("pricing.securePayment")}
        </div>
        <p className="text-gray-500 dark:text-gray-400">
          {tUI("pricing.questions")}{" "}
          <a
            href="mailto:support@practix.uz"
            className="text-brand-600 hover:underline"
          >
            support@practix.uz
          </a>
        </p>
      </div>
    </div>
  );
};

export default PricingPage;
