import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-testing-test-main',
	title: 'TestMain Setup',
	difficulty: 'medium',	tags: ['go', 'testing', 'setup', 'teardown'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Use **TestMain** for test setup and teardown logic.

**Requirements:**
1. Implement \`TestMain(m *M)\` function
2. Set up test resources (initialize TestConfig)
3. Run tests with \`m.Run()\`
4. Clean up resources after tests
5. Track exit code for verification

**Example:**
\`\`\`go
var testConfig *Config
var exitCode int

func TestMain(m *M) {
    // Setup
    testConfig = &Config{Ready: true}

    // Run tests
    exitCode = m.Run()

    // Teardown
    testConfig = nil
}
\`\`\`

**Why TestMain:**
- Runs once per package, before all tests
- Perfect for expensive setup (DB connections, services)
- Cleanup guaranteed after all tests finish
- Exit code tells CI if tests passed

**Constraints:**
- Must call m.Run() to execute tests
- Setup before m.Run(), cleanup after
- Store exit code for verification`,
	initialCode: `package testmain_test

// TestConfig represents shared test configuration
type TestConfig struct {
	DatabaseURL string
	LogLevel    string
	Initialized bool
	Counter     int
}

var testConfig *TestConfig
var exitCode int  // Track exit code from m.Run()

// TODO: Implement TestMain with setup and teardown
func TestMain(m *M) {
	// TODO: Setup - initialize testConfig with:
	//   DatabaseURL: "postgres://test@localhost/testdb"
	//   LogLevel: "debug"
	//   Initialized: true

	// TODO: Run all tests and capture exit code

	// TODO: Teardown - set testConfig to nil
}

// TODO: Write test that uses testConfig
func TestDatabaseConnection(t *T) {
	// TODO: Verify testConfig is initialized
	// TODO: Check DatabaseURL is set
}`,
	solutionCode: `package testmain_test

// TestConfig represents shared test configuration
type TestConfig struct {
	DatabaseURL string
	LogLevel    string
	Initialized bool
	Counter     int
}

var testConfig *TestConfig
var exitCode int  // Track exit code from m.Run()

func TestMain(m *M) {
	// Setup: Initialize test configuration
	testConfig = &TestConfig{
		DatabaseURL: "postgres://test@localhost/testdb",
		LogLevel:    "debug",
		Initialized: true,
		Counter:     0,
	}

	// Run all tests
	exitCode = m.Run()

	// Teardown: Clean up resources
	testConfig = nil
}

func TestDatabaseConnection(t *T) {
	// Verify TestMain initialized config
	if testConfig == nil {
		t.Fatal("testConfig should be initialized by TestMain")
	}

	if !testConfig.Initialized {
		t.Error("testConfig.Initialized should be true")
	}

	// Check DatabaseURL is set
	expected := "postgres://test@localhost/testdb"
	if testConfig.DatabaseURL != expected {
		t.Errorf("DatabaseURL = %q, want %q", testConfig.DatabaseURL, expected)
	}
}`,
		hint1: `TestMain runs before all tests. Use it for package-level setup like database connections.`,
		hint2: `m.Run() returns exit code (0 = all passed). Store it to verify tests ran correctly.`,
		testCode: `package testmain_test

// Test1: testConfig is initialized by TestMain
func Test1(t *T) {
	if testConfig == nil {
		t.Error("testConfig should be set by TestMain")
	}
}

// Test2: testConfig.Initialized is true
func Test2(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	if !testConfig.Initialized {
		t.Error("testConfig.Initialized should be true")
	}
}

// Test3: DatabaseURL is set correctly
func Test3(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	expected := "postgres://test@localhost/testdb"
	if testConfig.DatabaseURL != expected {
		t.Errorf("DatabaseURL = %q, want %q", testConfig.DatabaseURL, expected)
	}
}

// Test4: LogLevel is set
func Test4(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	if testConfig.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want %q", testConfig.LogLevel, "debug")
	}
}

// Test5: Counter starts at 0
func Test5(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	if testConfig.Counter != 0 {
		t.Errorf("Counter = %d, want 0", testConfig.Counter)
	}
}

// Test6: Can modify shared config
func Test6(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	testConfig.Counter++
	if testConfig.Counter < 1 {
		t.Error("Counter should be incrementable")
	}
}

// Test7: Config struct has all fields
func Test7(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	// Verify all fields are accessible
	_ = testConfig.DatabaseURL
	_ = testConfig.LogLevel
	_ = testConfig.Initialized
	_ = testConfig.Counter
}

// Test8: m.Run() was called (exitCode is set)
func Test8(t *T) {
	// exitCode should be 0 if we reach this test
	// (tests only run if m.Run() was called)
	if testConfig == nil {
		t.Error("tests only run if m.Run() is called in TestMain")
	}
}

// Test9: Multiple tests can access same config
func Test9(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	// Config should still be valid
	if testConfig.DatabaseURL == "" {
		t.Error("DatabaseURL should persist across tests")
	}
}

// Test10: Config is ready for use
func Test10(t *T) {
	if testConfig == nil {
		t.Fatal("testConfig is nil")
	}
	if !testConfig.Initialized || testConfig.DatabaseURL == "" {
		t.Error("config should be fully initialized")
	}
}
`,
		whyItMatters: `TestMain enables expensive setup once instead of repeating it in every test.

**Why TestMain Matters:**
- **Performance:** Setup database once, not per test
- **Resources:** Share connections, configs across tests
- **Cleanup:** Guarantee teardown even if tests panic
- **Environment:** Set env vars, configure logging for all tests

**Without TestMain (slow):**
\`\`\`go
func TestUserCreate(t *testing.T) {
    db := setupDatabase()  // Slow
    defer db.Close()
    // test...
}

func TestUserUpdate(t *testing.T) {
    db := setupDatabase()  // Slow (repeated)
    defer db.Close()
    // test...
}
// Tests take 5 seconds
\`\`\`

**With TestMain (fast):**
\`\`\`go
var db *sql.DB

func TestMain(m *testing.M) {
    db = setupDatabase()  // Once
    code := m.Run()
    db.Close()
    os.Exit(code)
}

func TestUserCreate(t *testing.T) {
    // Use shared db
}

func TestUserUpdate(t *testing.T) {
    // Use shared db
}
// Tests take 1 second
\`\`\`

**Production Benefits:**
- **Database Tests:** Connect once, rollback transactions per test
- **Config Tests:** Initialize config once, share across tests
- **Mock Services:** Start mock server once, use in all tests
- **CI Speed:** Faster tests mean faster deployments

**Common Patterns:**
\`\`\`go
// Config-based setup
var cfg *Config

func TestMain(m *testing.M) {
    cfg = &Config{
        DBHost: "localhost",
        DBPort: 5432,
        Debug:  true,
    }

    code := m.Run()

    cfg = nil
    os.Exit(code)
}
\`\`\`

**Important Notes:**
- TestMain runs once per package
- Defer doesn't work with os.Exit (bypasses it)
- Must manually cleanup before os.Exit
- Setup failures should panic or log.Fatal

The Go standard library uses TestMain extensively - check net/http tests for examples.`,	order: 4,
	translations: {
		ru: {
			title: 'Настройка TestMain',
			description: `Используйте **TestMain** для настройки и очистки тестов.

**Требования:**
1. Реализуйте функцию \`TestMain(m *M)\`
2. Настройте тестовые ресурсы (инициализируйте TestConfig)
3. Запустите тесты с \`m.Run()\`
4. Очистите ресурсы после тестов
5. Сохраните код выхода для проверки

**Пример:**
\`\`\`go
var testConfig *Config
var exitCode int

func TestMain(m *M) {
    // Настройка
    testConfig = &Config{Ready: true}

    // Запуск тестов
    exitCode = m.Run()

    // Очистка
    testConfig = nil
}
\`\`\`

**Почему TestMain:**
- Запускается один раз на пакет, перед всеми тестами
- Идеален для дорогой настройки (БД, сервисы)
- Очистка гарантирована после всех тестов
- Код выхода сообщает CI прошли ли тесты

**Ограничения:**
- Должен вызвать m.Run() для выполнения тестов
- Настройка перед m.Run(), очистка после`,
			hint1: `TestMain запускается перед всеми тестами. Используйте для настройки уровня пакета.`,
			hint2: `m.Run() возвращает код выхода (0 = все прошли). Сохраните его для проверки.`,
			whyItMatters: `TestMain позволяет делать дорогую настройку один раз вместо повторения в каждом тесте.

**Почему TestMain важен:**
- **Производительность:** Настройте базу данных один раз
- **Ресурсы:** Разделяйте соединения между тестами
- **Очистка:** Гарантируйте teardown даже при панике
- **Окружение:** Установите env переменные для всех тестов`,
			solutionCode: `package testmain_test

// TestConfig представляет общую конфигурацию тестов
type TestConfig struct {
	DatabaseURL string
	LogLevel    string
	Initialized bool
	Counter     int
}

var testConfig *TestConfig
var exitCode int  // Отслеживаем код выхода от m.Run()

func TestMain(m *M) {
	// Настройка: Инициализировать конфигурацию
	testConfig = &TestConfig{
		DatabaseURL: "postgres://test@localhost/testdb",
		LogLevel:    "debug",
		Initialized: true,
		Counter:     0,
	}

	// Запустить все тесты
	exitCode = m.Run()

	// Очистка: Освободить ресурсы
	testConfig = nil
}

func TestDatabaseConnection(t *T) {
	// Проверить что TestMain инициализировал конфиг
	if testConfig == nil {
		t.Fatal("testConfig должен быть инициализирован TestMain")
	}

	if !testConfig.Initialized {
		t.Error("testConfig.Initialized должен быть true")
	}

	// Проверить что DatabaseURL установлен
	expected := "postgres://test@localhost/testdb"
	if testConfig.DatabaseURL != expected {
		t.Errorf("DatabaseURL = %q, ожидается %q", testConfig.DatabaseURL, expected)
	}
}`
		},
		uz: {
			title: `TestMain sozlash`,
			description: `Test sozlash va tozalash mantiqasi uchun **TestMain** dan foydalaning.

**Talablar:**
1. \`TestMain(m *M)\` funksiyasini amalga oshiring
2. Test resurslarini sozlang (TestConfig ni boshlang)
3. \`m.Run()\` bilan testlarni ishga tushiring
4. Testlardan keyin resurslarni tozalang
5. Tekshirish uchun chiqish kodini saqlang

**Misol:**
\`\`\`go
var testConfig *Config
var exitCode int

func TestMain(m *M) {
    // Sozlash
    testConfig = &Config{Ready: true}

    // Testlarni ishga tushirish
    exitCode = m.Run()

    // Tozalash
    testConfig = nil
}
\`\`\`

**Cheklovlar:**
- Testlarni bajarish uchun m.Run() ni chaqirishi kerak
- m.Run() dan oldin sozlash, keyin tozalash`,
			hint1: `TestMain barcha testlardan oldin ishlaydi. Paket darajasidagi sozlash uchun foydalaning.`,
			hint2: `m.Run() chiqish kodini qaytaradi (0 = hammasi o'tdi). Tekshirish uchun saqlang.`,
			whyItMatters: `TestMain har bir testda takrorlash o'rniga qimmat sozlashni bir marta amalga oshirishga imkon beradi.`,
			solutionCode: `package testmain_test

// TestConfig umumiy test konfiguratsiyasini ifodalaydi
type TestConfig struct {
	DatabaseURL string
	LogLevel    string
	Initialized bool
	Counter     int
}

var testConfig *TestConfig
var exitCode int  // m.Run() dan chiqish kodini kuzatish

func TestMain(m *M) {
	// Sozlash: Test konfiguratsiyasini boshlash
	testConfig = &TestConfig{
		DatabaseURL: "postgres://test@localhost/testdb",
		LogLevel:    "debug",
		Initialized: true,
		Counter:     0,
	}

	// Barcha testlarni ishga tushirish
	exitCode = m.Run()

	// Tozalash: Resurslarni bo'shatish
	testConfig = nil
}

func TestDatabaseConnection(t *T) {
	// TestMain konfigni boshladi tekshirish
	if testConfig == nil {
		t.Fatal("testConfig TestMain tomonidan boshlanishi kerak")
	}

	if !testConfig.Initialized {
		t.Error("testConfig.Initialized true bo'lishi kerak")
	}

	// DatabaseURL o'rnatilganligini tekshirish
	expected := "postgres://test@localhost/testdb"
	if testConfig.DatabaseURL != expected {
		t.Errorf("DatabaseURL = %q, kutilgan %q", testConfig.DatabaseURL, expected)
	}
}`
		}
	}
};

export default task;
