import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-testable',
	title: 'Writing Testable Code',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'testing', 'testability'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor code to be testable by removing dependencies on external systems, using dependency injection, and making functions pure.

**You will refactor:**

1. Extract dependencies into interfaces
2. Use dependency injection instead of global state
3. Make functions pure (same input → same output)
4. Write tests for refactored code

**Key Concepts:**
- **Dependency Injection**: Pass dependencies as parameters
- **Interfaces**: Abstract external dependencies
- **Pure Functions**: No side effects, deterministic
- **Testability**: Code should be easy to test in isolation

**Constraints:**
- No global variables for dependencies
- Define at least one interface
- Make at least one function pure
- Write 2+ test cases`,
	initialCode: `package principles

import (
	"fmt"
	"time"
)

type Database struct{}

func (db *Database) Save(key, value string) error {
	return nil
}

type User struct {
	Name      string
	CreatedAt time.Time
}

func CreateUser(name string) error {

	}

	}

}`,
	solutionCode: `package principles

import (
	"fmt"
	"testing"
	"time"
)

// Storage interface abstracts database operations
type Storage interface {
	Save(key, value string) error
}

// TimeProvider interface abstracts time operations
type TimeProvider interface {
	Now() time.Time
}

// Database implements Storage interface
type Database struct{}

func (db *Database) Save(key, value string) error {
	fmt.Printf("Saving %s=%s\n", key, value)
	return nil
}

// RealTimeProvider implements TimeProvider for production
type RealTimeProvider struct{}

func (r *RealTimeProvider) Now() time.Time {
	return time.Now()
}

type User struct {
	Name      string
	CreatedAt time.Time
}

// UserService handles user operations with injected dependencies
type UserService struct {
	storage Storage
	time    TimeProvider
}

// NewUserService creates a new service with dependencies
func NewUserService(storage Storage, timeProvider TimeProvider) *UserService {
	return &UserService{
		storage: storage,
		time:    timeProvider,
	}
}

// CreateUser creates a user with injected dependencies (testable!)
func (s *UserService) CreateUser(name string) error {
	now := s.time.Now()

	user := &User{
		Name:      name,
		CreatedAt: now,
	}

	return s.storage.Save("user", user.Name)
}

// Test example:

// MockStorage for testing
type MockStorage struct {
	SavedKeys   []string
	SavedValues []string
	SaveError   error
}

func (m *MockStorage) Save(key, value string) error {
	m.SavedKeys = append(m.SavedKeys, key)
	m.SavedValues = append(m.SavedValues, value)
	return m.SaveError
}

// MockTimeProvider for testing
type MockTimeProvider struct {
	FixedTime time.Time
}

func (m *MockTimeProvider) Now() time.Time {
	return m.FixedTime
}

// Test cases
func TestCreateUser(t *testing.T) {
	// Arrange
	mockStorage := &MockStorage{}
	fixedTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	mockTime := &MockTimeProvider{FixedTime: fixedTime}
	service := NewUserService(mockStorage, mockTime)

	// Act
	err := service.CreateUser("John")

	// Assert
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(mockStorage.SavedKeys) != 1 {
		t.Errorf("Expected 1 save call, got %d", len(mockStorage.SavedKeys))
	}
	if mockStorage.SavedValues[0] != "John" {
		t.Errorf("Expected 'John', got '%s'", mockStorage.SavedValues[0])
	}
}`,
	hint1: `Define Storage and TimeProvider interfaces. Create a UserService struct that holds these dependencies. Use constructor injection via NewUserService.`,
	hint2: `Create mock implementations of both interfaces for testing. Write tests that inject mocks and verify behavior without touching real database or time.`,
	whyItMatters: `Testable code is maintainable code. Dependency injection enables testing in isolation.

**Why Testability Matters:**

**Global Dependencies Kill Testing:**
\`\`\`go
// BAD: Can't test without real database
var db *sql.DB
func SaveUser(u *User) error {
    return db.Save(u)  // Requires real DB!
}

// GOOD: Can inject mock
type Storage interface {
    Save(*User) error
}
func SaveUser(storage Storage, u *User) error {
    return storage.Save(u)  // Use mock in tests!
}
\`\`\`

**Pure Functions Are Easy to Test:**
\`\`\`go
// BAD: Impure, uses time.Now()
func IsExpired(expires time.Time) bool {
    return time.Now().After(expires)
}

// GOOD: Pure, testable
func IsExpired(now, expires time.Time) bool {
    return now.After(expires)
}
\`\`\`

**Dependency Injection:**
- Pass dependencies as parameters
- Use interfaces to abstract implementations
- Constructor injection for services
- Makes testing trivial`,
	order: 10,
	testCode: `package principles

import (
	"errors"
	"testing"
	"time"
)

// Test1: NewUserService creates service with dependencies
func Test1(t *testing.T) {
	mockStorage := &MockStorage{}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)
	if service == nil {
		t.Error("expected non-nil service")
	}
}

// Test2: CreateUser saves user name
func Test2(t *testing.T) {
	mockStorage := &MockStorage{}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)

	service.CreateUser("Alice")

	if len(mockStorage.SavedValues) != 1 || mockStorage.SavedValues[0] != "Alice" {
		t.Error("expected user name to be saved")
	}
}

// Test3: CreateUser uses correct key
func Test3(t *testing.T) {
	mockStorage := &MockStorage{}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)

	service.CreateUser("Bob")

	if len(mockStorage.SavedKeys) != 1 || mockStorage.SavedKeys[0] != "user" {
		t.Error("expected key to be 'user'")
	}
}

// Test4: CreateUser returns nil on success
func Test4(t *testing.T) {
	mockStorage := &MockStorage{}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)

	err := service.CreateUser("Charlie")
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
}

// Test5: CreateUser propagates storage error
func Test5(t *testing.T) {
	mockStorage := &MockStorage{SaveError: errors.New("db error")}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)

	err := service.CreateUser("Dan")
	if err == nil {
		t.Error("expected error from storage")
	}
}

// Test6: MockStorage implements Storage interface
func Test6(t *testing.T) {
	var storage Storage = &MockStorage{}
	err := storage.Save("key", "value")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test7: MockTimeProvider returns fixed time
func Test7(t *testing.T) {
	fixedTime := time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC)
	mockTime := &MockTimeProvider{FixedTime: fixedTime}

	result := mockTime.Now()
	if !result.Equal(fixedTime) {
		t.Error("expected fixed time to be returned")
	}
}

// Test8: RealTimeProvider implements TimeProvider
func Test8(t *testing.T) {
	var provider TimeProvider = &RealTimeProvider{}
	now := provider.Now()
	if now.IsZero() {
		t.Error("expected non-zero time")
	}
}

// Test9: Database implements Storage interface
func Test9(t *testing.T) {
	var storage Storage = &Database{}
	err := storage.Save("test", "value")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test10: Multiple CreateUser calls accumulate in mock
func Test10(t *testing.T) {
	mockStorage := &MockStorage{}
	mockTime := &MockTimeProvider{FixedTime: time.Now()}
	service := NewUserService(mockStorage, mockTime)

	service.CreateUser("User1")
	service.CreateUser("User2")
	service.CreateUser("User3")

	if len(mockStorage.SavedValues) != 3 {
		t.Errorf("expected 3 saves, got: %d", len(mockStorage.SavedValues))
	}
}
`,
	translations: {
		ru: {
			title: 'Написание тестируемого кода',
			description: `Выполните рефакторинг кода для тестируемости удалив зависимости от внешних систем, используя внедрение зависимостей и делая функции чистыми.`,
			hint1: `Определите интерфейсы Storage и TimeProvider. Создайте структуру UserService которая хранит эти зависимости. Используйте инъекцию через конструктор NewUserService.`,
			hint2: `Создайте мок реализации обоих интерфейсов для тестирования. Напишите тесты которые внедряют моки и проверяют поведение без реальной БД или времени.`,
			whyItMatters: `Тестируемый код — это поддерживаемый код. Внедрение зависимостей позволяет тестировать изолированно.`,
			solutionCode: `package principles

import (
	"fmt"
	"testing"
	"time"
)

// Storage интерфейс абстрагирует операции с БД
type Storage interface {
	Save(key, value string) error
}

// TimeProvider интерфейс абстрагирует операции со временем
type TimeProvider interface {
	Now() time.Time
}

type Database struct{}

func (db *Database) Save(key, value string) error {
	fmt.Printf("Saving %s=%s\n", key, value)
	return nil
}

type RealTimeProvider struct{}

func (r *RealTimeProvider) Now() time.Time {
	return time.Now()
}

type User struct {
	Name      string
	CreatedAt time.Time
}

// UserService обрабатывает операции с пользователем с внедрёнными зависимостями
type UserService struct {
	storage Storage
	time    TimeProvider
}

// NewUserService создаёт новый сервис с зависимостями
func NewUserService(storage Storage, timeProvider TimeProvider) *UserService {
	return &UserService{
		storage: storage,
		time:    timeProvider,
	}
}

// CreateUser создаёт пользователя с внедрёнными зависимостями (тестируемо!)
func (s *UserService) CreateUser(name string) error {
	now := s.time.Now()

	user := &User{
		Name:      name,
		CreatedAt: now,
	}

	return s.storage.Save("user", user.Name)
}

// Мок для тестирования
type MockStorage struct {
	SavedKeys   []string
	SavedValues []string
	SaveError   error
}

func (m *MockStorage) Save(key, value string) error {
	m.SavedKeys = append(m.SavedKeys, key)
	m.SavedValues = append(m.SavedValues, value)
	return m.SaveError
}

type MockTimeProvider struct {
	FixedTime time.Time
}

func (m *MockTimeProvider) Now() time.Time {
	return m.FixedTime
}

func TestCreateUser(t *testing.T) {
	mockStorage := &MockStorage{}
	fixedTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	mockTime := &MockTimeProvider{FixedTime: fixedTime}
	service := NewUserService(mockStorage, mockTime)

	err := service.CreateUser("John")

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(mockStorage.SavedKeys) != 1 {
		t.Errorf("Expected 1 save call, got %d", len(mockStorage.SavedKeys))
	}
	if mockStorage.SavedValues[0] != "John" {
		t.Errorf("Expected 'John', got '%s'", mockStorage.SavedValues[0])
	}
}`
		},
		uz: {
			title: 'Testlanadigan kod yozish',
			description: `Tashqi tizimlardan bog'liqlikni olib tashlash, bog'liqlik in'ektsiyasidan foydalanish va funksiyalarni toza qilish orqali kodni testlanadigan qilish uchun refaktoring qiling.`,
			hint1: `Storage va TimeProvider interfeyslarini aniqlang. Bu bog'liqliklarni saqlaydigan UserService strukturasini yarating. NewUserService orqali konstruktor in'ektsiyasidan foydalaning.`,
			hint2: `Test uchun ikkala interfeys uchun ham mok implementatsiyalarini yarating. Moklar in'ektsiya qiladigan va haqiqiy ma'lumotlar bazasi yoki vaqtsiz harakatni tekshiradigan testlar yozing.`,
			whyItMatters: `Testlanadigan kod qo'llab-quvvatlanadigan kod. Bog'liqlik in'ektsiyasi izolyatsiyada testlashga imkon beradi.`,
			solutionCode: `package principles

import (
	"fmt"
	"testing"
	"time"
)

// Storage interfeysi ma'lumotlar bazasi operatsiyalarini abstrakt qiladi
type Storage interface {
	Save(key, value string) error
}

// TimeProvider interfeysi vaqt operatsiyalarini abstrakt qiladi
type TimeProvider interface {
	Now() time.Time
}

type Database struct{}

func (db *Database) Save(key, value string) error {
	fmt.Printf("Saving %s=%s\n", key, value)
	return nil
}

type RealTimeProvider struct{}

func (r *RealTimeProvider) Now() time.Time {
	return time.Now()
}

type User struct {
	Name      string
	CreatedAt time.Time
}

// UserService in'ektsiya qilingan bog'liqliklar bilan foydalanuvchi operatsiyalarini boshqaradi
type UserService struct {
	storage Storage
	time    TimeProvider
}

// NewUserService bog'liqliklar bilan yangi servisni yaratadi
func NewUserService(storage Storage, timeProvider TimeProvider) *UserService {
	return &UserService{
		storage: storage,
		time:    timeProvider,
	}
}

// CreateUser in'ektsiya qilingan bog'liqliklar bilan foydalanuvchi yaratadi (testlanadigan!)
func (s *UserService) CreateUser(name string) error {
	now := s.time.Now()

	user := &User{
		Name:      name,
		CreatedAt: now,
	}

	return s.storage.Save("user", user.Name)
}

// Test uchun mok
type MockStorage struct {
	SavedKeys   []string
	SavedValues []string
	SaveError   error
}

func (m *MockStorage) Save(key, value string) error {
	m.SavedKeys = append(m.SavedKeys, key)
	m.SavedValues = append(m.SavedValues, value)
	return m.SaveError
}

type MockTimeProvider struct {
	FixedTime time.Time
}

func (m *MockTimeProvider) Now() time.Time {
	return m.FixedTime
}

func TestCreateUser(t *testing.T) {
	mockStorage := &MockStorage{}
	fixedTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	mockTime := &MockTimeProvider{FixedTime: fixedTime}
	service := NewUserService(mockStorage, mockTime)

	err := service.CreateUser("John")

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(mockStorage.SavedKeys) != 1 {
		t.Errorf("Expected 1 save call, got %d", len(mockStorage.SavedKeys))
	}
	if mockStorage.SavedValues[0] != "John" {
		t.Errorf("Expected 'John', got '%s'", mockStorage.SavedValues[0])
	}
}`
		}
	}
};

export default task;
