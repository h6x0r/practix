import React from 'react';
import { PaymentProvider, PromoCodeValidation } from '../../api/paymentService';
import { SubscriptionPlan } from '@/features/subscriptions/model/types';
import { IconCheck, IconX, IconGift } from '@/components/Icons';

interface CheckoutPanelProps {
  selectedPlan: SubscriptionPlan | undefined;
  providers: PaymentProvider[];
  selectedProvider: 'payme' | 'click';
  onProviderChange: (provider: 'payme' | 'click') => void;
  promoCode: string;
  onPromoCodeChange: (code: string) => void;
  promoValidation: PromoCodeValidation | null;
  onPromoValidationChange: (v: PromoCodeValidation | null) => void;
  promoLoading: boolean;
  onValidatePromo: () => void;
  checkoutLoading: boolean;
  onCheckout: () => void;
  getFinalPrice: () => number;
  language: string;
}

const formatPrice = (amountInTiyn: number): string => {
  const uzs = amountInTiyn / 100;
  return new Intl.NumberFormat('uz-UZ').format(uzs);
};

const CheckoutPanel: React.FC<CheckoutPanelProps> = ({
  selectedPlan,
  providers,
  selectedProvider,
  onProviderChange,
  promoCode,
  onPromoCodeChange,
  promoValidation,
  onPromoValidationChange,
  promoLoading,
  onValidatePromo,
  checkoutLoading,
  onCheckout,
  getFinalPrice,
  language,
}) => {
  return (
    <div className="lg:col-span-1">
      <div className="sticky top-24 bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border p-6 space-y-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white">
          {language === 'ru' ? 'Оформление' : 'Checkout'}
        </h3>

        {/* Selected Plan Summary */}
        {selectedPlan ? (
          <PlanSummary
            plan={selectedPlan}
            promoValidation={promoValidation}
            getFinalPrice={getFinalPrice}
            language={language}
          />
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
            {providers.map((provider) => (
              <button
                key={provider.id}
                data-testid={`provider-${provider.id}`}
                onClick={() => onProviderChange(provider.id as 'payme' | 'click')}
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

        {/* Promo Code Input */}
        {selectedPlan && (
          <PromoCodeInput
            promoCode={promoCode}
            onPromoCodeChange={onPromoCodeChange}
            promoValidation={promoValidation}
            onPromoValidationChange={onPromoValidationChange}
            promoLoading={promoLoading}
            onValidatePromo={onValidatePromo}
            language={language}
          />
        )}

        {/* Checkout Button */}
        <button
          data-testid="checkout-button"
          onClick={onCheckout}
          disabled={
            !selectedPlan || checkoutLoading || !providers.some((p) => p.id === selectedProvider && p.configured)
          }
          className="w-full py-4 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 disabled:from-gray-400 disabled:to-gray-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 disabled:shadow-none transition-all transform hover:-translate-y-0.5 disabled:transform-none disabled:cursor-not-allowed"
        >
          {checkoutLoading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              {language === 'ru' ? 'Обработка...' : 'Processing...'}
            </span>
          ) : (
            <>
              {language === 'ru' ? 'Оплатить' : 'Pay'}{' '}
              {selectedPlan ? `${formatPrice(getFinalPrice())} UZS` : ''}
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
  );
};

// Sub-component: Plan Summary
const PlanSummary: React.FC<{
  plan: SubscriptionPlan;
  promoValidation: PromoCodeValidation | null;
  getFinalPrice: () => number;
  language: string;
}> = ({ plan, promoValidation, getFinalPrice, language }) => (
  <div className="p-4 bg-gray-50 dark:bg-dark-bg rounded-xl">
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 rounded-xl bg-brand-100 dark:bg-brand-900/30 flex items-center justify-center text-xl">
        {plan.type === 'global' ? '👑' : plan.course?.icon || '📚'}
      </div>
      <div className="flex-1">
        <div className="font-bold text-gray-900 dark:text-white text-sm">
          {language === 'ru' ? plan.nameRu || plan.name : plan.name}
        </div>
        <div className="text-xs text-gray-500">{language === 'ru' ? '1 месяц' : '1 month'}</div>
      </div>
    </div>
    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-dark-border space-y-2">
      {promoValidation?.valid && promoValidation.discountAmount ? (
        <>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">{language === 'ru' ? 'Цена' : 'Price'}</span>
            <span className="text-gray-500 line-through">{formatPrice(plan.priceMonthly)} UZS</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-green-600 dark:text-green-400">
              {language === 'ru' ? 'Скидка' : 'Discount'}
            </span>
            <span className="text-green-600 dark:text-green-400">
              -{formatPrice(promoValidation.discountAmount)} UZS
            </span>
          </div>
          <div className="flex justify-between pt-2 border-t border-gray-200 dark:border-dark-border">
            <span className="text-gray-500">{language === 'ru' ? 'Итого' : 'Total'}</span>
            <span className="font-bold text-gray-900 dark:text-white">{formatPrice(getFinalPrice())} UZS</span>
          </div>
        </>
      ) : (
        <div className="flex justify-between">
          <span className="text-gray-500">{language === 'ru' ? 'Итого' : 'Total'}</span>
          <span className="font-bold text-gray-900 dark:text-white">{formatPrice(plan.priceMonthly)} UZS</span>
        </div>
      )}
    </div>
  </div>
);

// Sub-component: Promo Code Input
const PromoCodeInput: React.FC<{
  promoCode: string;
  onPromoCodeChange: (code: string) => void;
  promoValidation: PromoCodeValidation | null;
  onPromoValidationChange: (v: PromoCodeValidation | null) => void;
  promoLoading: boolean;
  onValidatePromo: () => void;
  language: string;
}> = ({ promoCode, onPromoCodeChange, promoValidation, onPromoValidationChange, promoLoading, onValidatePromo, language }) => (
  <div>
    <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-2">
      <IconGift className="w-4 h-4 inline mr-1" />
      {language === 'ru' ? 'Промокод' : language === 'uz' ? 'Promokod' : 'Promo Code'}
    </label>
    <div className="flex gap-2">
      <input
        type="text"
        data-testid="promo-code-input"
        value={promoCode}
        onChange={(e) => {
          onPromoCodeChange(e.target.value.toUpperCase());
          if (promoValidation) onPromoValidationChange(null);
        }}
        placeholder={language === 'ru' ? 'Введите код' : 'Enter code'}
        className="flex-1 px-3 py-2 rounded-lg border border-gray-200 dark:border-dark-border bg-white dark:bg-dark-bg text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
      />
      <button
        data-testid="apply-promo-button"
        onClick={onValidatePromo}
        disabled={!promoCode.trim() || promoLoading}
        className="px-4 py-2 bg-gray-100 dark:bg-dark-bg hover:bg-gray-200 dark:hover:bg-dark-border text-gray-700 dark:text-gray-300 font-medium text-sm rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {promoLoading ? (
          <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        ) : language === 'ru' ? (
          'Применить'
        ) : (
          'Apply'
        )}
      </button>
    </div>
    {promoValidation && (
      <div
        className={`mt-2 text-sm flex items-center gap-1 ${promoValidation.valid ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}
      >
        {promoValidation.valid ? (
          <>
            <IconCheck className="w-4 h-4" />
            {promoValidation.type === 'PERCENTAGE'
              ? `${promoValidation.discount}% ${language === 'ru' ? 'скидка применена' : 'discount applied'}`
              : promoValidation.type === 'FIXED'
                ? `${formatPrice(promoValidation.discountAmount || 0)} UZS ${language === 'ru' ? 'скидка' : 'off'}`
                : language === 'ru'
                  ? 'Промокод применён'
                  : 'Promo applied'}
          </>
        ) : (
          <>
            <IconX className="w-4 h-4" />
            {promoValidation.error || (language === 'ru' ? 'Неверный промокод' : 'Invalid promo code')}
          </>
        )}
      </div>
    )}
  </div>
);

export default CheckoutPanel;
