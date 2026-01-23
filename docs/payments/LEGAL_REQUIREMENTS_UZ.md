# Legal Requirements for Payment Integration in Uzbekistan

## ИП vs ООО Comparison

### Quick Comparison Table

| Criteria | ИП (Individual Entrepreneur) | ООО (LLC) |
|----------|------------------------------|-----------|
| **Registration Fee** | 337,500 UZS (online) | 3,750,000 UZS |
| **Registration Time** | 1-2 days, online | 3-5 days, more documents |
| **Liability** | Personal assets | Limited to charter capital |
| **Closure** | Simple | Complex liquidation |
| **Works with Payme/Click** | Yes | Yes |
| **Employees** | Up to 3 | Unlimited |
| **Investments/Partners** | No | Yes (shares) |
| **B2B Reputation** | Medium | High |

---

## Taxation in 2025

### Tax Regimes

| Regime | Rate | Best For |
|--------|------|----------|
| **Fixed Tax (ИП)** | 375,000 UZS/month | Income < 100M UZS/year |
| **Turnover Tax (УСН)** | 4% (services) | Income 100M - 1B UZS/year |
| **General System** | 10% profit + VAT | Income > 1B UZS/year |

### Changes from 2026

Starting January 1, 2026:
- Turnover tax reduced to **1%** for ИП with annual turnover < 1B UZS
- Fixed income tax for ИП abolished
- Simplified registration with biometric identification

### IT Sector Specifics

- Profit tax increased from 7.5% to 10%
- Mandatory integration with state e-document systems
- IT startup preferences have been removed

---

## Recommendation

### Start with ИП because:

1. **Fast and Cheap** - registration in 1-2 days, ~350,000 UZS
2. **Payme/Click work with ИП** - full payment acceptance
3. **Minimal Reporting** - especially with fixed tax
4. **Easy to Close/Pause** - if things don't work out
5. **1% Tax from 2026** - very profitable for IT services

### When to Switch to ООО:

- Annual turnover > 1B UZS
- Need investors/partners
- More than 3 employees
- B2B contracts require legal entity

---

## Documents for ИП Registration

### Required Documents

1. **Passport** (copy)
2. **Photo 3x4** (2 pcs.)
3. **Registration Address** (propiska)
4. **OKЭД Codes** (activity types):
   - 62.01 - Software Development
   - 62.09 - Other IT Services
   - 63.11 - Data Processing
5. **State Fee** - 337,500 UZS online

### Where to Register

- **Online**: [new.birdarcha.uz](https://new.birdarcha.uz)
- **In Person**: State Services Center (YAGXM)

---

## After Registration

### Step-by-Step

1. **Open Business Bank Account**
   - Kapital Bank, Uzcard Bank, Agrobank
   - Required documents: passport, registration certificate

2. **Get EDS (Electronic Digital Signature)**
   - Required for electronic reporting
   - Apply at certification centers

3. **Apply to Payment Providers**

   **Payme**:
   - Go to [merchant.payme.uz](https://merchant.payme.uz)
   - Submit application with documents
   - Wait for approval (5-10 business days)
   - Get Merchant ID and Secret Key

   **Click**:
   - Go to [my.click.uz/services/create](https://my.click.uz)
   - Create new service
   - Submit for verification
   - Get Service ID, Merchant ID, Secret Key

4. **Configure Environment**
   ```bash
   # server/.env
   PAYME_MERCHANT_ID=your_merchant_id
   PAYME_SECRET_KEY=your_secret_key

   CLICK_SERVICE_ID=your_service_id
   CLICK_MERCHANT_ID=your_merchant_id
   CLICK_SECRET_KEY=your_secret_key
   ```

5. **Test Integration**
   - Use test mode first
   - Verify webhooks work
   - Test payment flow

6. **Go Live**
   - Set `*_TEST_MODE=false`
   - Update webhook URLs to production
   - Monitor first real transactions

---

## New E-Commerce Regulations (July 1, 2025)

According to Cabinet of Ministers Resolution No. 885 (December 26, 2024):

### Requirements

1. **Mandatory Registration**
   - All companies selling online must register in Uzbekistan
   - Applies to local and foreign businesses

2. **Special Bank Accounts**
   - Required for e-commerce transactions
   - Enhances financial transparency

3. **Integration Requirements**
   - State electronic document systems
   - Tax reporting integration

---

## Costs Summary

### One-Time Costs

| Item | Cost (UZS) |
|------|------------|
| ИП Registration | 337,500 |
| EDS Certificate | 150,000-300,000 |
| Bank Account Opening | Free (most banks) |
| **Total** | ~500,000-650,000 |

### Monthly Costs (Fixed Tax)

| Item | Cost (UZS) |
|------|------------|
| Fixed Income Tax | 375,000 |
| Social Tax (if no employees) | 0 |
| Bank Maintenance | 50,000-100,000 |
| **Total** | ~425,000-475,000 |

### From 2026 (Turnover Tax 1%)

At 10M UZS monthly revenue:
- Tax: 100,000 UZS
- Bank: 50,000-100,000 UZS
- **Total**: ~150,000-200,000 UZS

---

## Useful Links

- [Birdarcha - ИП Registration](https://new.birdarcha.uz)
- [Tax Code of Uzbekistan](https://lex.uz/docs/7302)
- [Payme Merchant Portal](https://merchant.payme.uz)
- [Click Services](https://my.click.uz)
- [Legalise.uz - Business Registration](https://legalise.uz/)
- [NORMA.UZ - Legal Info](https://www.norma.uz/)

---

## Contact Information

### Tax Authorities
- Hotline: 1198
- Website: soliq.uz

### Payme Support
- Phone: +998 78 150-22-24
- Email: support@payme.uz

### Click Support
- Phone: +998 71 200-11-00
- Email: info@click.uz
