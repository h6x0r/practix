import React, { useContext, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { paymentService, PaymentHistoryItem, PaymentProvider } from '../api/paymentService';
import { subscriptionService } from '@/features/subscriptions/api/subscriptionService';
import { SubscriptionPlan } from '@/features/subscriptions/model/types';
import { AuthContext } from '@/components/Layout';
import { useLanguage } from '@/contexts/LanguageContext';
import {
  IconCheck,
  IconX,
  IconSparkles,
  IconCrown,
  IconRocket,
  IconLightning,
} from '@/components/Icons';
import { createLogger } from '@/lib/logger';

const log = createLogger('Payments');

// Format price in UZS
const formatPrice = (amountInTiyn: number): string => {
  const uzs = amountInTiyn / 100;
  return new Intl.NumberFormat('uz-UZ').format(uzs);
};

// Format date
const formatDate = (isoDate: string | undefined): string => {
  if (!isoDate) return 'N/A';
  try {
    return new Date(isoDate).toLocaleDateString('ru-RU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  } catch {
    return isoDate;
  }
};

type TabType = 'subscribe' | 'purchases' | 'history';

const PaymentsPage = () => {
  const { user } = useContext(AuthContext);
  const { language } = useLanguage();
  const [searchParams] = useSearchParams();

  // State
  const [activeTab, setActiveTab] = useState<TabType>('subscribe');
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [providers, setProviders] = useState<PaymentProvider[]>([]);
  const [history, setHistory] = useState<PaymentHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [checkoutLoading, setCheckoutLoading] = useState(false);

  // Checkout form state
  const [selectedPlanId, setSelectedPlanId] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<'payme' | 'click'>('payme');
  const [purchaseType, setPurchaseType] = useState<'subscription' | 'roadmap'>('subscription');

  // Get global and course plans
  const globalPlan = plans.find(p => p.type === 'global');
  const coursePlans = plans.filter(p => p.type === 'course');

  // Load data
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

          // Pre-select from URL params
          const planFromUrl = searchParams.get('plan');
          if (planFromUrl) {
            const plan = plansData.find(p => p.slug === planFromUrl || p.id === planFromUrl);
            if (plan) {
              setSelectedPlanId(plan.id);
            }
          }

          setLoading(false);
        })
        .catch(error => {
          log.error('Failed to load payment data', error);
          setLoading(false);
        });
    }
  }, [user, searchParams]);

  // Handle checkout
  const handleCheckout = async () => {
    if (!selectedPlanId && purchaseType === 'subscription') {
      return;
    }

    setCheckoutLoading(true);

    try {
      const response = await paymentService.createCheckout({
        orderType: purchaseType === 'subscription' ? 'subscription' : 'purchase',
        planId: purchaseType === 'subscription' ? selectedPlanId : undefined,
        purchaseType: purchaseType === 'roadmap' ? 'roadmap_generation' : undefined,
        quantity: 1,
        provider: selectedProvider,
        returnUrl: window.location.origin + '/payments?status=success',
      });

      // Redirect to payment page
      window.location.href = response.paymentUrl;
    } catch (error) {
      log.error('Checkout failed', error);
      setCheckoutLoading(false);
    }
  };

  // Get selected plan details
  const selectedPlan = plans.find(p => p.id === selectedPlanId);

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
        <h1 className="text-3xl font-display font-bold text-gray-900 dark:text-white" data-testid="payments-title">
          {language === 'ru' ? 'Подписки и платежи' : language === 'uz' ? "Obuna va to'lovlar" : 'Subscriptions & Payments'}
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          {language === 'ru'
            ? 'Управляйте подписками и просматривайте историю платежей'
            : language === 'uz'
            ? "Obunalarni boshqaring va to'lovlar tarixini ko'ring"
            : 'Manage subscriptions and view payment history'}
        </p>
      </div>

      {/* Current Status Card */}
      <div
        className={`rounded-3xl p-8 text-white shadow-xl relative overflow-hidden transition-all duration-500 ${
          user?.isPremium
            ? 'bg-gradient-to-r from-brand-600 to-purple-600'
            : 'bg-gradient-to-r from-gray-900 to-gray-800 dark:from-dark-surface dark:to-black'
        }`}
      >
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          {user?.isPremium ? (
            <div data-testid="subscription-active">
              <div className="flex items-center gap-2 text-white/90 font-bold uppercase tracking-wider mb-2 text-xs">
                <IconSparkles className="w-4 h-4" /> {language === 'ru' ? 'Текущий план' : 'Current Plan'}
              </div>
              <h2 className="text-3xl font-display font-bold mb-2" data-testid="current-plan-name">Premium</h2>
              <p className="text-white/90 max-w-md">
                {language === 'ru'
                  ? `Активная подписка до ${formatDate(user.plan?.expiresAt)}. Полный доступ ко всем функциям.`
                  : `Active subscription until ${formatDate(user.plan?.expiresAt)}. Full access to all features.`}
              </p>
            </div>
          ) : (
            <div>
              <div className="text-brand-400 font-bold uppercase tracking-wider mb-2 text-xs">
                {language === 'ru' ? 'Текущий план' : 'Current Plan'}
              </div>
              <h2 className="text-3xl font-display font-bold mb-2">Free</h2>
              <p className="text-gray-400 max-w-md">
                {language === 'ru'
                  ? 'Базовый доступ. Обновитесь для полного доступа к курсам и AI-тьютору.'
                  : 'Basic access. Upgrade for full course access and AI Tutor.'}
              </p>
            </div>
          )}
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-10 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2 pointer-events-none"></div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200 dark:border-dark-border">
        {[
          { id: 'subscribe', label: language === 'ru' ? 'Подписки' : 'Subscribe', icon: IconCrown, testId: 'subscribe-tab' },
          { id: 'purchases', label: language === 'ru' ? 'Покупки' : 'Purchases', icon: IconRocket, testId: 'purchases-tab' },
          { id: 'history', label: language === 'ru' ? 'История' : 'History', icon: IconLightning, testId: 'history-tab' },
        ].map(tab => (
          <button
            key={tab.id}
            data-testid={tab.testId}
            onClick={() => setActiveTab(tab.id as TabType)}
            className={`flex items-center gap-2 px-6 py-3 font-bold text-sm transition-all border-b-2 -mb-[2px] ${
              activeTab === tab.id
                ? 'border-brand-500 text-brand-600 dark:text-brand-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'subscribe' && (
        <div className="grid lg:grid-cols-3 gap-8" data-testid="subscription-plans">
          {/* Plan Selection */}
          <div className="lg:col-span-2 space-y-6">
            {/* Global Premium */}
            {globalPlan && (
              <div
                data-testid="plan-global-premium"
                data-selected={selectedPlanId === globalPlan.id ? 'true' : 'false'}
                onClick={() => {
                  setSelectedPlanId(globalPlan.id);
                  setPurchaseType('subscription');
                }}
                className={`p-6 rounded-2xl border-2 cursor-pointer transition-all ${
                  selectedPlanId === globalPlan.id
                    ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                    : 'border-gray-200 dark:border-dark-border hover:border-brand-300 bg-white dark:bg-dark-surface'
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
                          {language === 'ru' ? 'Глобальный Premium' : 'Global Premium'}
                        </h3>
                        <span className="px-2 py-0.5 bg-gradient-to-r from-amber-400 to-orange-500 text-white text-xs font-bold rounded-full">
                          BEST VALUE
                        </span>
                      </div>
                      <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
                        {language === 'ru'
                          ? 'Доступ ко всем курсам и функциям платформы'
                          : 'Access to all courses and platform features'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      {formatPrice(globalPlan.priceMonthly)} <span className="text-sm font-normal text-gray-500">UZS</span>
                    </div>
                    <div className="text-xs text-gray-500">{language === 'ru' ? '/ месяц' : '/ month'}</div>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {['Все курсы', '100 AI запросов/день', 'Приоритетная очередь', 'Без рекламы'].map(feature => (
                    <span
                      key={feature}
                      className="px-3 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-medium rounded-full"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Course Plans */}
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                {language === 'ru' ? 'Или выберите отдельный курс' : 'Or select a single course'}
              </h3>
              <div className="space-y-3">
                {coursePlans.map((plan, index) => (
                  <div
                    key={plan.id}
                    data-testid={index === 0 ? 'plan-course-premium' : `plan-course-${index}`}
                    data-selected={selectedPlanId === plan.id ? 'true' : 'false'}
                    onClick={() => {
                      setSelectedPlanId(plan.id);
                      setPurchaseType('subscription');
                    }}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition-all flex items-center justify-between ${
                      selectedPlanId === plan.id
                        ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                        : 'border-gray-200 dark:border-dark-border hover:border-brand-300 bg-white dark:bg-dark-surface'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-gray-100 dark:bg-dark-bg flex items-center justify-center text-xl">
                        {plan.course?.icon || '📚'}
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white">
                          {language === 'ru' ? plan.nameRu || plan.name : plan.name}
                        </h4>
                        <p className="text-xs text-gray-500">{plan.course?.title}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-gray-900 dark:text-white">
                        {formatPrice(plan.priceMonthly)} <span className="text-xs font-normal text-gray-500">UZS</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Checkout Panel */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border p-6 space-y-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                {language === 'ru' ? 'Оформление' : 'Checkout'}
              </h3>

              {/* Selected Plan Summary */}
              {selectedPlan ? (
                <div className="p-4 bg-gray-50 dark:bg-dark-bg rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-brand-100 dark:bg-brand-900/30 flex items-center justify-center text-xl">
                      {selectedPlan.type === 'global' ? '👑' : selectedPlan.course?.icon || '📚'}
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-gray-900 dark:text-white text-sm">
                        {language === 'ru' ? selectedPlan.nameRu || selectedPlan.name : selectedPlan.name}
                      </div>
                      <div className="text-xs text-gray-500">{language === 'ru' ? '1 месяц' : '1 month'}</div>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-dark-border flex justify-between">
                    <span className="text-gray-500">{language === 'ru' ? 'Итого' : 'Total'}</span>
                    <span className="font-bold text-gray-900 dark:text-white">
                      {formatPrice(selectedPlan.priceMonthly)} UZS
                    </span>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-gray-50 dark:bg-dark-bg rounded-xl text-center text-gray-500 text-sm">
                  {language === 'ru' ? 'Выберите план слева' : 'Select a plan'}
                </div>
              )}

              {/* Payment Provider Selection */}
              <div>
                <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-3">
                  {language === 'ru' ? 'Способ оплаты' : 'Payment Method'}
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {providers.map(provider => (
                    <button
                      key={provider.id}
                      data-testid={`provider-${provider.id}`}
                      onClick={() => setSelectedProvider(provider.id as 'payme' | 'click')}
                      disabled={!provider.configured}
                      className={`p-4 rounded-xl border-2 transition-all ${
                        selectedProvider === provider.id
                          ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20 selected'
                          : 'border-gray-200 dark:border-dark-border hover:border-brand-300'
                      } ${!provider.configured ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className="text-2xl mb-1">{provider.id === 'payme' ? '💳' : '📱'}</div>
                      <div className="font-bold text-sm text-gray-900 dark:text-white">{provider.name}</div>
                      {!provider.configured && (
                        <div className="text-xs text-gray-400 mt-1">
                          {language === 'ru' ? 'Скоро' : 'Coming soon'}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Checkout Button */}
              <button
                data-testid="checkout-button"
                onClick={handleCheckout}
                disabled={!selectedPlanId || checkoutLoading || !providers.some(p => p.id === selectedProvider && p.configured)}
                className="w-full py-4 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 disabled:from-gray-400 disabled:to-gray-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 disabled:shadow-none transition-all transform hover:-translate-y-0.5 disabled:transform-none disabled:cursor-not-allowed"
              >
                {checkoutLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    {language === 'ru' ? 'Обработка...' : 'Processing...'}
                  </span>
                ) : (
                  <>
                    {language === 'ru' ? 'Оплатить' : 'Pay'}{' '}
                    {selectedPlan ? `${formatPrice(selectedPlan.priceMonthly)} UZS` : ''}
                  </>
                )}
              </button>

              <p className="text-xs text-gray-400 text-center">
                {language === 'ru'
                  ? 'Нажимая "Оплатить", вы соглашаетесь с условиями использования'
                  : 'By clicking "Pay", you agree to our terms of service'}
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'purchases' && (
        <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            {language === 'ru' ? 'Разовые покупки' : 'One-time Purchases'}
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            {/* Roadmap Generation */}
            <div data-testid="plan-roadmap-credits" className="p-6 rounded-xl border border-gray-200 dark:border-dark-border">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center text-white text-2xl">
                  🗺️
                </div>
                <div>
                  <h4 className="font-bold text-gray-900 dark:text-white">
                    {language === 'ru' ? 'Генерация Roadmap' : 'Roadmap Generation'}
                  </h4>
                  <p className="text-sm text-gray-500">
                    {language === 'ru' ? 'Персональный план обучения' : 'Personal learning plan'}
                  </p>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  15,000 <span className="text-sm font-normal text-gray-500">UZS</span>
                </div>
                <button
                  onClick={() => {
                    setPurchaseType('roadmap');
                    setSelectedPlanId('');
                    setActiveTab('subscribe');
                  }}
                  className="px-4 py-2 bg-brand-600 hover:bg-brand-500 text-white font-bold text-sm rounded-lg transition-colors"
                >
                  {language === 'ru' ? 'Купить' : 'Buy'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border overflow-hidden">
          <div className="p-6 border-b border-gray-100 dark:border-dark-border">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
              {language === 'ru' ? 'История платежей' : 'Payment History'}
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table data-testid="payment-history-table" className="w-full text-left text-sm">
              <thead className="bg-gray-50 dark:bg-dark-bg text-gray-500 uppercase font-bold text-xs">
                <tr>
                  <th className="px-6 py-4">{language === 'ru' ? 'Дата' : 'Date'}</th>
                  <th className="px-6 py-4">{language === 'ru' ? 'Описание' : 'Description'}</th>
                  <th className="px-6 py-4">{language === 'ru' ? 'Сумма' : 'Amount'}</th>
                  <th className="px-6 py-4">{language === 'ru' ? 'Статус' : 'Status'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
                {history.length === 0 ? (
                  <tr data-testid="history-empty">
                    <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                      {language === 'ru' ? 'История платежей пуста' : 'No payment history'}
                    </td>
                  </tr>
                ) : (
                  history.map(item => (
                    <tr key={item.id} data-testid="payment-history-row" className="hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors">
                      <td className="px-6 py-4 text-gray-900 dark:text-white">{formatDate(item.createdAt)}</td>
                      <td className="px-6 py-4 text-gray-500">{item.description}</td>
                      <td className="px-6 py-4 font-bold text-gray-900 dark:text-white">
                        {formatPrice(item.amount)} {item.currency}
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                            item.status === 'completed'
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                              : item.status === 'pending'
                              ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                              : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          }`}
                        >
                          {item.status === 'completed' ? (
                            <IconCheck className="w-3 h-3" />
                          ) : (
                            <IconX className="w-3 h-3" />
                          )}
                          {item.status}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default PaymentsPage;
