import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-ocp-advanced',
	title: 'Open/Closed Principle - Advanced',
	difficulty: 'medium',
	tags: ['go', 'solid', 'ocp', 'extensibility', 'advanced'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply OCP to a complex validation system with composable rules.

**Current Problem:**

A UserValidator with hardcoded validation rules that requires modification to add new validation logic.

**Your task:**

Create a flexible validation system using the Strategy pattern:

1. **ValidationRule interface** - Contract for all validation rules
2. **EmailRule** - Validates email format
3. **PasswordRule** - Validates password strength
4. **AgeRule** - Validates age requirements
5. **CompositeRule** - Combines multiple rules
6. **Validator** - Validates using any rule combination

**Key Concepts:**
- **Strategy Pattern**: Encapsulate algorithms in separate classes
- **Composite Pattern**: Combine simple rules into complex validations
- **Extensibility**: Add validation rules without modifying core code

**Example Usage:**

\`\`\`go
// Create individual rules
emailRule := &EmailRule{}
passwordRule := &PasswordRule{MinLength: 8}
ageRule := &AgeRule{MinAge: 18}

// Compose rules
allRules := &CompositeRule{
    Rules: []ValidationRule{emailRule, passwordRule, ageRule},
}

// Validate user
user := User{Email: "test@example.com", Password: "Secure123", Age: 25}
validator := &Validator{}
errors := validator.Validate(user, allRules)

// Add custom rule? Just implement ValidationRule interface!
type UsernameRule struct{}
func (ur *UsernameRule) Validate(user User) error { ... }
\`\`\`

**Real-World Impact:**
- Different validation rules for different user types
- Reusable validation logic across application
- Easy A/B testing of validation strategies
- Plugin-based validation systems

**Constraints:**
- All rules must implement ValidationRule interface
- CompositeRule must support any number of rules
- Adding new rule should not modify existing code`,
	initialCode: `package principles

import (
	"fmt"
	"regexp"
	"strings"
)

type User struct {
	Email    string
	Password string
	Age      int
}

// ValidationRule defines contract for all validation rules
type ValidationRule interface {
	Validate(user User) error
}

// EmailRule validates email format
type EmailRule struct{}

func (er *EmailRule) Validate(user User) error {
	// TODO: Check if email contains "@", return error if not
	panic("TODO: implement")
}

// PasswordRule validates password strength
type PasswordRule struct {
	MinLength int
}

func (pr *PasswordRule) Validate(user User) error {
	// TODO: Check length, uppercase, and digit requirements
	panic("TODO: implement")
}

// AgeRule validates age requirements
type AgeRule struct {
	MinAge int
}

func (ar *AgeRule) Validate(user User) error {
	// TODO: Check if age meets minimum requirement
	panic("TODO: implement")
}

// CompositeRule combines multiple validation rules
type CompositeRule struct {
	Rules []ValidationRule
}

func (cr *CompositeRule) Validate(user User) error {
	// TODO: Run all rules, collect errors, return combined error
	panic("TODO: implement")
}

// Validator executes validation rules
type Validator struct{}

func (v *Validator) Validate(user User, rule ValidationRule) []string {
	// TODO: Run rule.Validate, return errors as slice
	panic("TODO: implement")
}`,
	solutionCode: `package principles

import (
	"fmt"
	"regexp"
	"strings"
)

type User struct {
	Email    string
	Password string
	Age      int
}

// ValidationRule interface makes validation system open for extension
// Add new rule? Implement this interface - no changes to Validator needed!
type ValidationRule interface {
	Validate(user User) error	// returns nil if valid, error if invalid
}

// EmailRule validates email format
type EmailRule struct{}

// Validate checks email contains @ symbol
func (er *EmailRule) Validate(user User) error {
	if !strings.Contains(user.Email, "@") {	// simple email check
		return fmt.Errorf("invalid email format")	// return error if invalid
	}
	return nil	// valid email
}

// PasswordRule validates password strength with configurable requirements
type PasswordRule struct {
	MinLength int	// minimum password length
}

// Validate checks password meets strength requirements
func (pr *PasswordRule) Validate(user User) error {
	if len(user.Password) < pr.MinLength {	// check minimum length
		return fmt.Errorf("password must be at least %d characters", pr.MinLength)
	}
	if !regexp.MustCompile(\`[A-Z]\`).MatchString(user.Password) {	// check uppercase
		return fmt.Errorf("password must contain uppercase letter")
	}
	if !regexp.MustCompile(\`[0-9]\`).MatchString(user.Password) {	// check digit
		return fmt.Errorf("password must contain digit")
	}
	return nil	// password meets all requirements
}

// AgeRule validates age requirements
type AgeRule struct {
	MinAge int	// minimum required age
}

// Validate checks user meets age requirement
func (ar *AgeRule) Validate(user User) error {
	if user.Age < ar.MinAge {	// check minimum age
		return fmt.Errorf("must be at least %d years old", ar.MinAge)
	}
	return nil	// age requirement met
}

// CompositeRule combines multiple rules using Composite pattern
// Allows building complex validations from simple rules
type CompositeRule struct {
	Rules []ValidationRule	// collection of rules to apply
}

// Validate runs all rules and collects errors
func (cr *CompositeRule) Validate(user User) error {
	var errorMessages []string

	// Run each rule
	for _, rule := range cr.Rules {
		if err := rule.Validate(user); err != nil {	// if rule fails
			errorMessages = append(errorMessages, err.Error())	// collect error
		}
	}

	if len(errorMessages) > 0 {	// if any rules failed
		return fmt.Errorf(strings.Join(errorMessages, "; "))	// combine errors
	}
	return nil	// all rules passed
}

// Validator is closed for modification
// It works with any ValidationRule without knowing implementation
type Validator struct{}

// Validate runs validation rule and returns error messages
func (v *Validator) Validate(user User, rule ValidationRule) []string {
	if err := rule.Validate(user); err != nil {	// run validation
		return strings.Split(err.Error(), "; ")	// split composite errors
	}
	return nil	// no errors
}

// Want to add UsernameRule? Just implement ValidationRule!
// type UsernameRule struct { MinLength int }
// func (ur *UsernameRule) Validate(user User) error { ... }
// NO changes to Validator needed!`,
	hint1: `For EmailRule.Validate, check if user.Email contains "@" using strings.Contains. Return fmt.Errorf if not, return nil if valid. For PasswordRule, check length and use regexp to check for uppercase and digits.`,
	hint2: `For CompositeRule.Validate, loop through cr.Rules, call Validate on each, and collect error messages. If any errors exist, join them with "; " and return as error. For Validator.Validate, call rule.Validate(user) and split the error message by "; " to return []string.`,
	testCode: `package principles

import (
	"testing"
)

// Test1: EmailRule accepts valid email
func Test1(t *testing.T) {
	rule := &EmailRule{}
	user := User{Email: "test@example.com"}
	if err := rule.Validate(user); err != nil {
		t.Errorf("valid email rejected: %v", err)
	}
}

// Test2: EmailRule rejects email without @
func Test2(t *testing.T) {
	rule := &EmailRule{}
	user := User{Email: "invalid-email"}
	if err := rule.Validate(user); err == nil {
		t.Error("invalid email should be rejected")
	}
}

// Test3: PasswordRule accepts valid password
func Test3(t *testing.T) {
	rule := &PasswordRule{MinLength: 8}
	user := User{Password: "ValidPass123"}
	if err := rule.Validate(user); err != nil {
		t.Errorf("valid password rejected: %v", err)
	}
}

// Test4: PasswordRule rejects short password
func Test4(t *testing.T) {
	rule := &PasswordRule{MinLength: 8}
	user := User{Password: "Short1"}
	if err := rule.Validate(user); err == nil {
		t.Error("short password should be rejected")
	}
}

// Test5: AgeRule accepts valid age
func Test5(t *testing.T) {
	rule := &AgeRule{MinAge: 18}
	user := User{Age: 25}
	if err := rule.Validate(user); err != nil {
		t.Errorf("valid age rejected: %v", err)
	}
}

// Test6: AgeRule rejects underage
func Test6(t *testing.T) {
	rule := &AgeRule{MinAge: 18}
	user := User{Age: 15}
	if err := rule.Validate(user); err == nil {
		t.Error("underage should be rejected")
	}
}

// Test7: CompositeRule passes when all rules pass
func Test7(t *testing.T) {
	composite := &CompositeRule{
		Rules: []ValidationRule{
			&EmailRule{},
			&AgeRule{MinAge: 18},
		},
	}
	user := User{Email: "test@test.com", Age: 20}
	if err := composite.Validate(user); err != nil {
		t.Errorf("valid user rejected: %v", err)
	}
}

// Test8: CompositeRule fails when any rule fails
func Test8(t *testing.T) {
	composite := &CompositeRule{
		Rules: []ValidationRule{
			&EmailRule{},
			&AgeRule{MinAge: 18},
		},
	}
	user := User{Email: "invalid", Age: 20}
	if err := composite.Validate(user); err == nil {
		t.Error("user with invalid email should be rejected")
	}
}

// Test9: Validator returns errors as slice
func Test9(t *testing.T) {
	validator := &Validator{}
	rule := &EmailRule{}
	user := User{Email: "invalid"}
	errors := validator.Validate(user, rule)
	if len(errors) == 0 {
		t.Error("should return errors for invalid user")
	}
}

// Test10: Validator returns nil for valid user
func Test10(t *testing.T) {
	validator := &Validator{}
	rule := &EmailRule{}
	user := User{Email: "valid@example.com"}
	errors := validator.Validate(user, rule)
	if errors != nil {
		t.Errorf("should return nil for valid user, got: %v", errors)
	}
}
`,
	whyItMatters: `Advanced OCP with validation shows how to build flexible, composable systems.

**Why Advanced OCP Matters:**

**1. Composable Business Rules**

\`\`\`go
// Build complex rules from simple ones
premiumUserRules := &CompositeRule{
	Rules: []ValidationRule{
		&EmailRule{},
		&PasswordRule{MinLength: 12},	// stricter for premium
		&AgeRule{MinAge: 21},
		&CreditCardRule{},		// premium only
	},
}

regularUserRules := &CompositeRule{
	Rules: []ValidationRule{
		&EmailRule{},
		&PasswordRule{MinLength: 8},	// less strict
		&AgeRule{MinAge: 18},
		// no credit card required
	},
}

// Easy to create different rule sets for different contexts!
\`\`\`

**2. Real-World: Form Validation**

\`\`\`go
// Registration form vs Profile update form
registrationRules := &CompositeRule{
	Rules: []ValidationRule{
		&EmailRule{},
		&PasswordRule{MinLength: 8},
		&AgeRule{MinAge: 18},
		&TermsAcceptedRule{},	// only for registration
		&RecaptchaRule{},	// only for registration
	},
}

profileUpdateRules := &CompositeRule{
	Rules: []ValidationRule{
		&EmailRule{},
		// no password validation on update
		// no age validation on update
		// no terms/recaptcha
	},
}
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Принцип открытости/закрытости - Продвинутый',
			description: `Примените OCP к сложной системе валидации с компонуемыми правилами.`,
			hint1: `Для EmailRule.Validate проверьте содержит ли user.Email "@" используя strings.Contains. Верните fmt.Errorf если нет, верните nil если валидно.`,
			hint2: `Для CompositeRule.Validate переберите cr.Rules, вызовите Validate на каждом, соберите сообщения об ошибках. Если есть ошибки, объедините их с "; ".`,
			whyItMatters: `Продвинутый OCP с валидацией показывает как строить гибкие, компонуемые системы.`,
			solutionCode: `package principles

import (
	"fmt"
	"regexp"
	"strings"
)

type User struct {
	Email    string
	Password string
	Age      int
}

type ValidationRule interface {
	Validate(user User) error
}

type EmailRule struct{}

func (er *EmailRule) Validate(user User) error {
	if !strings.Contains(user.Email, "@") {
		return fmt.Errorf("неверный формат email")
	}
	return nil
}

type PasswordRule struct {
	MinLength int
}

func (pr *PasswordRule) Validate(user User) error {
	if len(user.Password) < pr.MinLength {
		return fmt.Errorf("пароль должен быть минимум %d символов", pr.MinLength)
	}
	if !regexp.MustCompile(\`[A-Z]\`).MatchString(user.Password) {
		return fmt.Errorf("пароль должен содержать заглавную букву")
	}
	if !regexp.MustCompile(\`[0-9]\`).MatchString(user.Password) {
		return fmt.Errorf("пароль должен содержать цифру")
	}
	return nil
}

type AgeRule struct {
	MinAge int
}

func (ar *AgeRule) Validate(user User) error {
	if user.Age < ar.MinAge {
		return fmt.Errorf("должно быть минимум %d лет", ar.MinAge)
	}
	return nil
}

type CompositeRule struct {
	Rules []ValidationRule
}

func (cr *CompositeRule) Validate(user User) error {
	var errorMessages []string
	for _, rule := range cr.Rules {
		if err := rule.Validate(user); err != nil {
			errorMessages = append(errorMessages, err.Error())
		}
	}
	if len(errorMessages) > 0 {
		return fmt.Errorf(strings.Join(errorMessages, "; "))
	}
	return nil
}

type Validator struct{}

func (v *Validator) Validate(user User, rule ValidationRule) []string {
	if err := rule.Validate(user); err != nil {
		return strings.Split(err.Error(), "; ")
	}
	return nil
}`
		},
		uz: {
			title: 'Ochiq/Yopiq printsipi - Kengaytirilgan',
			description: `Kompozitsion qoidalar bilan murakkab tekshirish tizimiga OCP ni qo'llang.`,
			hint1: `EmailRule.Validate uchun user.Email "@" ni o'z ichiga oladimi strings.Contains yordamida tekshiring. Agar yo'q bo'lsa fmt.Errorf qaytaring, to'g'ri bo'lsa nil.`,
			hint2: `CompositeRule.Validate uchun cr.Rules ni aylanib o'ting, har birida Validate ni chaqiring, xato xabarlarini yig'ing. Agar xatolar bo'lsa "; " bilan birlashtiring.`,
			whyItMatters: `Tekshirish bilan kengaytirilgan OCP moslashuvchan, kompozitsion tizimlarni qanday qurishni ko'rsatadi.`,
			solutionCode: `package principles

import (
	"fmt"
	"regexp"
	"strings"
)

type User struct {
	Email    string
	Password string
	Age      int
}

type ValidationRule interface {
	Validate(user User) error
}

type EmailRule struct{}

func (er *EmailRule) Validate(user User) error {
	if !strings.Contains(user.Email, "@") {
		return fmt.Errorf("noto'g'ri email formati")
	}
	return nil
}

type PasswordRule struct {
	MinLength int
}

func (pr *PasswordRule) Validate(user User) error {
	if len(user.Password) < pr.MinLength {
		return fmt.Errorf("parol kamida %d belgi bo'lishi kerak", pr.MinLength)
	}
	if !regexp.MustCompile(\`[A-Z]\`).MatchString(user.Password) {
		return fmt.Errorf("parol katta harf o'z ichiga olishi kerak")
	}
	if !regexp.MustCompile(\`[0-9]\`).MatchString(user.Password) {
		return fmt.Errorf("parol raqam o'z ichiga olishi kerak")
	}
	return nil
}

type AgeRule struct {
	MinAge int
}

func (ar *AgeRule) Validate(user User) error {
	if user.Age < ar.MinAge {
		return fmt.Errorf("kamida %d yoshda bo'lish kerak", ar.MinAge)
	}
	return nil
}

type CompositeRule struct {
	Rules []ValidationRule
}

func (cr *CompositeRule) Validate(user User) error {
	var errorMessages []string
	for _, rule := range cr.Rules {
		if err := rule.Validate(user); err != nil {
			errorMessages = append(errorMessages, err.Error())
		}
	}
	if len(errorMessages) > 0 {
		return fmt.Errorf(strings.Join(errorMessages, "; "))
	}
	return nil
}

type Validator struct{}

func (v *Validator) Validate(user User, rule ValidationRule) []string {
	if err := rule.Validate(user); err != nil {
		return strings.Split(err.Error(), "; ")
	}
	return nil
}`
		}
	}
};

export default task;
