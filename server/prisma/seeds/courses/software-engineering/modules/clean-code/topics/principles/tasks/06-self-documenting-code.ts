import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-self-documenting',
	title: 'Self-Documenting Code',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'documentation', 'naming'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Transform cryptic code into self-documenting code by using clear names, extracting functions, and replacing magic numbers with named constants.

**You will refactor:**

1. Replace magic numbers with named constants
2. Extract boolean expressions into well-named functions
3. Use descriptive variable names
4. Make code read like prose

**Key Concepts:**
- **Named Constants**: Replace magic numbers (3.14, 365, etc.)
- **Intention-Revealing Functions**: Extract complex conditions
- **Ubiquitous Language**: Use domain terminology
- **Code as Documentation**: Code explains itself without comments

**Constraints:**
- No explanatory comments allowed
- Use domain-specific terminology
- Extract at least 3 named constants
- Extract at least 2 query functions (Is/Has/Can)`,
	initialCode: `package principles

import "time"

type Account struct {
	Balance     float64
	CreatedAt   time.Time
	IsPremium   bool
	FailedLogins int
}

func ProcessAccount(a *Account) string {
	if a.Balance < 100 && !a.IsPremium {
	}

	if d < 30 {
	}

	if a.FailedLogins >= 3 && a.FailedLogins < 10 {
	}

	if a.FailedLogins >= 10 {
	}

	if a.Balance >= 1000 && a.IsPremium {
	}

}`,
	solutionCode: `package principles

import "time"

type Account struct {
	Balance      float64
	CreatedAt    time.Time
	IsPremium    bool
	FailedLogins int
}

const (
	minimumActiveBalance = 100.0
	vipBalanceThreshold  = 1000.0
	trialPeriodDays      = 30
	lockThreshold        = 3
	suspensionThreshold  = 10
)

// ProcessAccount determines account status based on balance, age, and security metrics
func ProcessAccount(account *Account) string {
	if isInactiveAccount(account) {
		return "inactive"
	}

	if isTrialAccount(account) {
		return "trial"
	}

	if isSuspendedAccount(account) {
		return "suspended"
	}

	if isLockedAccount(account) {
		return "locked"
	}

	if isVIPAccount(account) {
		return "vip"
	}

	return "active"
}

func isInactiveAccount(account *Account) bool {
	return account.Balance < minimumActiveBalance && !account.IsPremium
}

func isTrialAccount(account *Account) bool {
	accountAgeDays := time.Since(account.CreatedAt).Hours() / 24
	return accountAgeDays < trialPeriodDays
}

func isLockedAccount(account *Account) bool {
	return account.FailedLogins >= lockThreshold &&
	       account.FailedLogins < suspensionThreshold
}

func isSuspendedAccount(account *Account) bool {
	return account.FailedLogins >= suspensionThreshold
}

func isVIPAccount(account *Account) bool {
	return account.Balance >= vipBalanceThreshold && account.IsPremium
}`,
	hint1: `Extract all magic numbers (100, 1000, 30, 3, 10) to named constants with descriptive names like minimumActiveBalance, vipBalanceThreshold, etc.`,
	hint2: `Extract each condition into a query function: isInactiveAccount, isTrialAccount, isLockedAccount, isSuspendedAccount, isVIPAccount. Each should have a clear boolean purpose.`,
	whyItMatters: `Self-documenting code eliminates the need for comments and makes intent crystal clear.

**Why Self-Documenting Code Matters:**

**Magic Numbers are Evil:**
\`\`\`go
// BAD
if age >= 18 && balance > 1000 {  // What are 18 and 1000?

// GOOD
const legalAdultAge = 18
const premiumBalanceThreshold = 1000.0
if age >= legalAdultAge && balance > premiumBalanceThreshold {
\`\`\`

**Complex Conditions Need Names:**
\`\`\`go
// BAD
if user.Age >= 18 && user.KYCVerified && user.Balance > 0 {

// GOOD
if isEligibleForTrading(user) {

func isEligibleForTrading(user *User) bool {
    return user.Age >= legalAdultAge &&
           user.KYCVerified &&
           user.Balance > 0
}
\`\`\``,
	order: 5,
	testCode: `package principles

import (
	"testing"
	"time"
)

// Test1: ProcessAccount returns inactive for low balance non-premium
func Test1(t *testing.T) {
	account := &Account{Balance: 50.0, IsPremium: false, CreatedAt: time.Now().AddDate(0, -2, 0)}
	result := ProcessAccount(account)
	if result != "inactive" {
		t.Errorf("expected 'inactive', got: %s", result)
	}
}

// Test2: ProcessAccount returns trial for new account
func Test2(t *testing.T) {
	account := &Account{Balance: 500.0, IsPremium: false, CreatedAt: time.Now().AddDate(0, 0, -10)}
	result := ProcessAccount(account)
	if result != "trial" {
		t.Errorf("expected 'trial', got: %s", result)
	}
}

// Test3: ProcessAccount returns locked for 3-9 failed logins
func Test3(t *testing.T) {
	account := &Account{Balance: 200.0, CreatedAt: time.Now().AddDate(0, -2, 0), FailedLogins: 5}
	result := ProcessAccount(account)
	if result != "locked" {
		t.Errorf("expected 'locked', got: %s", result)
	}
}

// Test4: ProcessAccount returns suspended for 10+ failed logins
func Test4(t *testing.T) {
	account := &Account{Balance: 200.0, CreatedAt: time.Now().AddDate(0, -2, 0), FailedLogins: 10}
	result := ProcessAccount(account)
	if result != "suspended" {
		t.Errorf("expected 'suspended', got: %s", result)
	}
}

// Test5: ProcessAccount returns vip for high balance premium
func Test5(t *testing.T) {
	account := &Account{Balance: 1500.0, IsPremium: true, CreatedAt: time.Now().AddDate(0, -2, 0)}
	result := ProcessAccount(account)
	if result != "vip" {
		t.Errorf("expected 'vip', got: %s", result)
	}
}

// Test6: ProcessAccount returns active for normal account
func Test6(t *testing.T) {
	account := &Account{Balance: 200.0, IsPremium: false, CreatedAt: time.Now().AddDate(0, -2, 0)}
	result := ProcessAccount(account)
	if result != "active" {
		t.Errorf("expected 'active', got: %s", result)
	}
}

// Test7: ProcessAccount inactive check: premium with low balance is NOT inactive
func Test7(t *testing.T) {
	account := &Account{Balance: 50.0, IsPremium: true, CreatedAt: time.Now().AddDate(0, -2, 0)}
	result := ProcessAccount(account)
	if result == "inactive" {
		t.Error("premium user with low balance should not be inactive")
	}
}

// Test8: ProcessAccount boundary: exactly 30 days old is NOT trial
func Test8(t *testing.T) {
	account := &Account{Balance: 200.0, CreatedAt: time.Now().AddDate(0, 0, -30)}
	result := ProcessAccount(account)
	if result == "trial" {
		t.Error("account exactly 30 days old should not be trial")
	}
}

// Test9: ProcessAccount boundary: exactly 3 failed logins is locked
func Test9(t *testing.T) {
	account := &Account{Balance: 200.0, CreatedAt: time.Now().AddDate(0, -2, 0), FailedLogins: 3}
	result := ProcessAccount(account)
	if result != "locked" {
		t.Errorf("expected 'locked' at exactly 3 failed logins, got: %s", result)
	}
}

// Test10: ProcessAccount VIP requires BOTH high balance AND premium
func Test10(t *testing.T) {
	account := &Account{Balance: 1500.0, IsPremium: false, CreatedAt: time.Now().AddDate(0, -2, 0)}
	result := ProcessAccount(account)
	if result == "vip" {
		t.Error("high balance without premium should not be VIP")
	}
}
`,
	translations: {
		ru: {
			title: 'Самодокументируемый код',
			description: `Преобразуйте загадочный код в самодокументируемый используя чёткие имена, извлекая функции и заменяя магические числа именованными константами.`,
			hint1: `Извлеките все магические числа в именованные константы с описательными именами.`,
			hint2: `Извлеките каждое условие в функцию-запрос с чёткой булевой целью.`,
			whyItMatters: `Самодокументируемый код устраняет необходимость в комментариях и делает намерение кристально чистым.`,
			solutionCode: `package principles

import "time"

type Account struct {
	Balance      float64
	CreatedAt    time.Time
	IsPremium    bool
	FailedLogins int
}

const (
	minimumActiveBalance = 100.0
	vipBalanceThreshold  = 1000.0
	trialPeriodDays      = 30
	lockThreshold        = 3
	suspensionThreshold  = 10
)

// ProcessAccount определяет статус аккаунта на основе баланса, возраста и метрик безопасности
func ProcessAccount(account *Account) string {
	if isInactiveAccount(account) {
		return "inactive"
	}

	if isTrialAccount(account) {
		return "trial"
	}

	if isSuspendedAccount(account) {
		return "suspended"
	}

	if isLockedAccount(account) {
		return "locked"
	}

	if isVIPAccount(account) {
		return "vip"
	}

	return "active"
}

func isInactiveAccount(account *Account) bool {
	return account.Balance < minimumActiveBalance && !account.IsPremium
}

func isTrialAccount(account *Account) bool {
	accountAgeDays := time.Since(account.CreatedAt).Hours() / 24
	return accountAgeDays < trialPeriodDays
}

func isLockedAccount(account *Account) bool {
	return account.FailedLogins >= lockThreshold &&
	       account.FailedLogins < suspensionThreshold
}

func isSuspendedAccount(account *Account) bool {
	return account.FailedLogins >= suspensionThreshold
}

func isVIPAccount(account *Account) bool {
	return account.Balance >= vipBalanceThreshold && account.IsPremium
}`
		},
		uz: {
			title: "O'z-o'zini hujjatlaydigan kod",
			description: `Aniq nomlardan foydalanib, funksiyalarni ajratib va sehrli raqamlarni nomlangan konstantalar bilan almashtirish orqali sirli kodni o'z-o'zini hujjatlaydigan kodga aylantiring.`,
			hint1: `Barcha sehrli raqamlarni ta'riflovchi nomlarga ega konstantalarga ajrating.`,
			hint2: `Har bir shartni aniq boolean maqsadga ega so'rov funksiyasiga ajrating.`,
			whyItMatters: `O'z-o'zini hujjatlaydigan kod izohlar zaruriyatini bartaraf etadi va niyatni kristal toza qiladi.`,
			solutionCode: `package principles

import "time"

type Account struct {
	Balance      float64
	CreatedAt    time.Time
	IsPremium    bool
	FailedLogins int
}

const (
	minimumActiveBalance = 100.0
	vipBalanceThreshold  = 1000.0
	trialPeriodDays      = 30
	lockThreshold        = 3
	suspensionThreshold  = 10
)

// ProcessAccount balans, yosh va xavfsizlik ko'rsatkichlariga asoslangan holda hisob holatini aniqlaydi
func ProcessAccount(account *Account) string {
	if isInactiveAccount(account) {
		return "inactive"
	}

	if isTrialAccount(account) {
		return "trial"
	}

	if isSuspendedAccount(account) {
		return "suspended"
	}

	if isLockedAccount(account) {
		return "locked"
	}

	if isVIPAccount(account) {
		return "vip"
	}

	return "active"
}

func isInactiveAccount(account *Account) bool {
	return account.Balance < minimumActiveBalance && !account.IsPremium
}

func isTrialAccount(account *Account) bool {
	accountAgeDays := time.Since(account.CreatedAt).Hours() / 24
	return accountAgeDays < trialPeriodDays
}

func isLockedAccount(account *Account) bool {
	return account.FailedLogins >= lockThreshold &&
	       account.FailedLogins < suspensionThreshold
}

func isSuspendedAccount(account *Account) bool {
	return account.FailedLogins >= suspensionThreshold
}

func isVIPAccount(account *Account) bool {
	return account.Balance >= vipBalanceThreshold && account.IsPremium
}`
		}
	}
};

export default task;
