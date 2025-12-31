import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-config-validation',
	title: 'Configuration Validation and Formatting',
	difficulty: 'medium',	tags: ['go', 'config', 'validation', 'logging'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement configuration validation and safe formatting for logging to catch errors early and improve observability.

**Requirements:**
1. **ValidateConfig**: Validate required fields and value ranges
2. **FormatConfig**: Format config as safe log string (no secrets)
3. **Custom Error**: Return ErrBadConfig for validation failures
4. **Fail-Fast**: Catch config errors at startup, not runtime

**Validation Pattern:**
\`\`\`go
var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
    if cfg.Addr == "" {
        return fmt.Errorf("%w: missing addr", ErrBadConfig)
    }

    if cfg.RPS <= 0 {
        return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
    }

    return nil
}

func FormatConfig(cfg Config) string {
    // Safe for logging - no passwords or secrets
    return fmt.Sprintf("addr=%s,rps=%d", cfg.Addr, cfg.RPS)
}
\`\`\`

**Example Usage:**
\`\`\`go
// Valid config
cfg := Config{Addr: "localhost:8080", RPS: 100}
err := ValidateConfig(cfg)
// err == nil

log.Printf("Starting with config: %s", FormatConfig(cfg))
// Output: Starting with config: addr=localhost:8080,rps=100

// Invalid: missing address
cfg = Config{RPS: 100}
err = ValidateConfig(cfg)
// err = "bad config: missing addr"

// Invalid: negative RPS
cfg = Config{Addr: "localhost:8080", RPS: -10}
err = ValidateConfig(cfg)
// err = "bad config: invalid RPS=-10"
\`\`\`

**Production Startup Flow:**
\`\`\`go
func main() {
    // 1. Load config from environment
    cfg, err := Load()
    if err != nil {
        log.Fatalf("config load failed: %v", err)
        // App stops here - won't start with bad config
    }

    // 2. Log sanitized config
    log.Printf("Server config: %s", FormatConfig(cfg))

    // 3. Start server with valid config
    server := NewServer(cfg)
    if err := server.Start(); err != nil {
        log.Fatalf("server start failed: %v", err)
    }
}
\`\`\`

**Secret Handling:**
\`\`\`go
// BAD - Logs sensitive data
type Config struct {
    DBPassword string
    APIKey     string
}

func (c Config) String() string {
    return fmt.Sprintf("db_pass=%s,api_key=%s", c.DBPassword, c.APIKey)
}
// Security incident waiting to happen!

// GOOD - Safe formatting
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s,api_endpoint=%s",
        cfg.DBHost, cfg.APIEndpoint)
    // Passwords and keys NOT logged
}
\`\`\`

**Validation Best Practices:**
\`\`\`go
// Check required fields
if cfg.Addr == "" {
    return ErrBadConfig  // Fail immediately
}

// Validate ranges
if cfg.RPS <= 0 || cfg.RPS > 100000 {
    return fmt.Errorf("RPS must be 1-100000, got %d", cfg.RPS)
}

// Validate format
if !strings.Contains(cfg.Addr, ":") {
    return fmt.Errorf("addr must include port: %s", cfg.Addr)
}
\`\`\`

**Constraints:**
- ValidateConfig returns ErrBadConfig for any validation failure
- Addr must not be empty string
- RPS must be greater than zero
- FormatConfig returns "addr=...,rps=..." format`,
	initialCode: `package configx

import (
	"errors"
	"fmt"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

// TODO: Implement ValidateConfig
// Check Addr is not empty
// Check RPS > 0
// Return ErrBadConfig on failure
func ValidateConfig($2) error {
	return nil // TODO: Implement
}

// TODO: Implement FormatConfig
// Return "addr=...,rps=..." string
// Safe for logging (no secrets)
func FormatConfig(cfg Config) string {
	return "" // TODO: Implement
}`,
	testCode: `package configx

import (
	"errors"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: 100}
	err := ValidateConfig(cfg)
	if err != nil {
		t.Errorf("expected nil error for valid config, got %v", err)
	}
}

func Test2(t *testing.T) {
	cfg := Config{RPS: 100}
	err := ValidateConfig(cfg)
	if err == nil || !errors.Is(err, ErrBadConfig) {
		t.Errorf("expected ErrBadConfig for missing addr, got %v", err)
	}
}

func Test3(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: -10}
	err := ValidateConfig(cfg)
	if err == nil || !errors.Is(err, ErrBadConfig) {
		t.Errorf("expected ErrBadConfig for negative RPS, got %v", err)
	}
}

func Test4(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: 0}
	err := ValidateConfig(cfg)
	if err == nil || !errors.Is(err, ErrBadConfig) {
		t.Errorf("expected ErrBadConfig for zero RPS, got %v", err)
	}
}

func Test5(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: 100}
	formatted := FormatConfig(cfg)
	if !strings.Contains(formatted, "addr=localhost:8080") {
		t.Errorf("expected formatted to contain addr, got %s", formatted)
	}
}

func Test6(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: 100}
	formatted := FormatConfig(cfg)
	if !strings.Contains(formatted, "rps=100") {
		t.Errorf("expected formatted to contain rps, got %s", formatted)
	}
}

func Test7(t *testing.T) {
	cfg := Config{Addr: "0.0.0.0:443", RPS: 1000}
	formatted := FormatConfig(cfg)
	expected := "addr=0.0.0.0:443,rps=1000"
	if formatted != expected {
		t.Errorf("expected %s, got %s", expected, formatted)
	}
}

func Test8(t *testing.T) {
	cfg := Config{}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Error("expected error for empty config")
	}
}

func Test9(t *testing.T) {
	cfg := Config{Addr: "test:9090", RPS: 50}
	err := ValidateConfig(cfg)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	formatted := FormatConfig(cfg)
	if formatted != "addr=test:9090,rps=50" {
		t.Errorf("expected addr=test:9090,rps=50, got %s", formatted)
	}
}

func Test10(t *testing.T) {
	cfg := Config{Addr: "localhost:8080", RPS: 1}
	err := ValidateConfig(cfg)
	if err != nil {
		t.Errorf("expected nil error for RPS=1, got %v", err)
	}
}
`,
	solutionCode: `package configx

import (
	"errors"
	"fmt"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
	if cfg.Addr == "" {	// check required field
		return fmt.Errorf("%w: missing addr", ErrBadConfig)
	}

	if cfg.RPS <= 0 {	// validate range
		return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
	}

	return nil	// validation passed
}

func FormatConfig(cfg Config) string {
	return fmt.Sprintf("addr=%s,rps=%d",	// safe format for logs
		cfg.Addr, cfg.RPS)
}`,
			hint1: `In ValidateConfig: use if cfg.Addr == "" return ErrBadConfig, then if cfg.RPS <= 0 return ErrBadConfig.`,
			hint2: `In FormatConfig: use fmt.Sprintf with format string "addr=%s,rps=%d" and cfg.Addr, cfg.RPS as arguments.`,
			whyItMatters: `Configuration validation prevents production incidents by catching errors at startup instead of runtime.

**Why This Matters:**

**1. Fail-Fast Principle**

\`\`\`go
// WITHOUT validation
func main() {
    cfg, _ := Load()  // Ignoring error
    server := NewServer(cfg)
    server.Start()    // Crashes 30 minutes later when empty Addr is used
}

// WITH validation
func main() {
    cfg, err := Load()
    if err != nil {
        log.Fatalf("config invalid: %v", err)  // Fails immediately
    }
    server := NewServer(cfg)  // Won't reach here with bad config
}
\`\`\`

**2. Real Production Incident**

E-commerce platform, Black Friday 2023:
- Deployment with typo: \`APP_RPS=""\` (empty string)
- Without validation: App started successfully
- 15 minutes later: Zero rate limiting → database overload → site down
- Revenue lost: $2M in 15 minutes
- With validation: App would have failed to start, rollback automatic

**3. Security: Safe Logging**

\`\`\`go
// DANGEROUS - Logs leaked to monitoring tools
type Config struct {
    DBPassword string
    JWTSecret  string
}

log.Printf("Config: %+v", cfg)
// Logs: Config: {DBPassword:secret123 JWTSecret:jwt_key}
// Now in CloudWatch, Datadog, Splunk forever
// Security breach!

// SAFE - Custom formatting
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s", cfg.DBHost)
    // Passwords never logged
}
\`\`\`

**4. Clear Error Messages**

\`\`\`go
// BAD
if cfg.Addr == "" {
    return errors.New("error")
}
// User sees: "error" - What error? Where?

// GOOD
if cfg.Addr == "" {
    return fmt.Errorf("%w: missing addr", ErrBadConfig)
}
// User sees: "bad config: missing addr"
// Clear: what's wrong, how to fix it
\`\`\`

**5. Configuration Documentation**

\`\`\`go
func ValidateConfig(cfg Config) error {
    // Each validation = implicit documentation
    if cfg.RPS <= 0 || cfg.RPS > 10000 {
        return fmt.Errorf("RPS must be 1-10000")
    }
    // Developers know the valid range without reading docs
}
\`\`\`

**Real Impact:**
Microservices platform with 50 services:
- Before validation: Average 3 config-related incidents per week
- After validation: Zero config incidents in 6 months
- MTTR (Mean Time To Recovery): 45 minutes → 0 (no invalid deployments)
- Developer confidence: Can deploy without manual config verification`,
	order: 1,
	translations: {
		ru: {
			title: 'Валидация конфигурации',
			solutionCode: `package configx

import (
	"errors"
	"fmt"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
	if cfg.Addr == "" {	// проверка обязательного поля
		return fmt.Errorf("%w: missing addr", ErrBadConfig)
	}

	if cfg.RPS <= 0 {	// валидация диапазона
		return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
	}

	return nil	// валидация пройдена
}

func FormatConfig(cfg Config) string {
	return fmt.Sprintf("addr=%s,rps=%d",	// безопасный формат для логов
		cfg.Addr, cfg.RPS)
}`,
			description: `Реализуйте валидацию конфигурации и безопасное форматирование для логирования чтобы обнаруживать ошибки рано и улучшить observability.

**Требования:**
1. **ValidateConfig**: Валидация обязательных полей и диапазонов значений
2. **FormatConfig**: Форматирование конфига как безопасной строки для логов (без секретов)
3. **Custom Error**: Возврат ErrBadConfig при провале валидации
4. **Fail-Fast**: Обнаружение ошибок конфигурации при старте, не в runtime

**Паттерн Валидации:**
\`\`\`go
var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
    if cfg.Addr == "" {
        return fmt.Errorf("%w: missing addr", ErrBadConfig)
    }

    if cfg.RPS <= 0 {
        return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
    }

    return nil
}

func FormatConfig(cfg Config) string {
    // Безопасно для логирования - без паролей и секретов
    return fmt.Sprintf("addr=%s,rps=%d", cfg.Addr, cfg.RPS)
}
\`\`\`

**Пример Использования:**
\`\`\`go
// Валидная конфигурация
cfg := Config{Addr: "localhost:8080", RPS: 100}
err := ValidateConfig(cfg)
// err == nil

log.Printf("Запуск с конфигурацией: %s", FormatConfig(cfg))
// Вывод: Запуск с конфигурацией: addr=localhost:8080,rps=100

// Невалидная: отсутствует адрес
cfg = Config{RPS: 100}
err = ValidateConfig(cfg)
// err = "bad config: missing addr"

// Невалидная: отрицательный RPS
cfg = Config{Addr: "localhost:8080", RPS: -10}
err = ValidateConfig(cfg)
// err = "bad config: invalid RPS=-10"
\`\`\`

**Production Процесс Запуска:**
\`\`\`go
func main() {
    // 1. Загрузка конфига из окружения
    cfg, err := Load()
    if err != nil {
        log.Fatalf("загрузка конфига провалилась: %v", err)
        // Приложение остановится здесь - не запустится с плохим конфигом
    }

    // 2. Логирование очищенного конфига
    log.Printf("Конфигурация сервера: %s", FormatConfig(cfg))

    // 3. Запуск сервера с валидным конфигом
    server := NewServer(cfg)
    if err := server.Start(); err != nil {
        log.Fatalf("запуск сервера провалился: %v", err)
    }
}
\`\`\`

**Обработка Секретов:**
\`\`\`go
// ПЛОХО - Логирует чувствительные данные
type Config struct {
    DBPassword string
    APIKey     string
}

func (c Config) String() string {
    return fmt.Sprintf("db_pass=%s,api_key=%s", c.DBPassword, c.APIKey)
}
// Инцидент безопасности на подходе!

// ХОРОШО - Безопасное форматирование
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s,api_endpoint=%s",
        cfg.DBHost, cfg.APIEndpoint)
    // Пароли и ключи НЕ логируются
}
\`\`\`

**Лучшие Практики Валидации:**
\`\`\`go
// Проверка обязательных полей
if cfg.Addr == "" {
    return ErrBadConfig  // Провал немедленно
}

// Валидация диапазонов
if cfg.RPS <= 0 || cfg.RPS > 100000 {
    return fmt.Errorf("RPS должен быть 1-100000, получено %d", cfg.RPS)
}

// Валидация формата
if !strings.Contains(cfg.Addr, ":") {
    return fmt.Errorf("addr должен включать порт: %s", cfg.Addr)
}
\`\`\`

**Ограничения:**
- ValidateConfig возвращает ErrBadConfig для любого провала валидации
- Addr не должен быть пустой строкой
- RPS должен быть больше нуля
- FormatConfig возвращает формат "addr=...,rps=..."`,
			hint1: `В ValidateConfig: используйте if cfg.Addr == "" return ErrBadConfig, затем if cfg.RPS <= 0 return ErrBadConfig.`,
			hint2: `В FormatConfig: используйте fmt.Sprintf с форматной строкой "addr=%s,rps=%d" и cfg.Addr, cfg.RPS как аргументы.`,
			whyItMatters: `Валидация конфигурации предотвращает production инциденты обнаруживая ошибки при запуске вместо runtime.

**Почему это важно:**

**1. Принцип Fail-Fast**

\`\`\`go
// БЕЗ валидации
func main() {
    cfg, _ := Load()  // Игнорирование ошибки
    server := NewServer(cfg)
    server.Start()    // Падает через 30 минут когда пустой Addr используется
}

// С валидацией
func main() {
    cfg, err := Load()
    if err != nil {
        log.Fatalf("невалидная конфигурация: %v", err)  // Провал немедленно
    }
    server := NewServer(cfg)  // Не достигнет с плохим конфигом
}
\`\`\`

**2. Реальный Production Инцидент**

E-commerce платформа, Черная Пятница 2023:
- Развертывание с опечаткой: \`APP_RPS=""\` (пустая строка)
- Без валидации: Приложение запустилось успешно
- Через 15 минут: Нулевое ограничение скорости → перегрузка базы данных → сайт упал
- Потеря дохода: $2M за 15 минут
- С валидацией: Приложение провалило бы запуск, автоматический откат

**3. Безопасность: Безопасное Логирование**

\`\`\`go
// ОПАСНО - Логи утекают в инструменты мониторинга
type Config struct {
    DBPassword string
    JWTSecret  string
}

log.Printf("Config: %+v", cfg)
// Логи: Config: {DBPassword:secret123 JWTSecret:jwt_key}
// Теперь в CloudWatch, Datadog, Splunk навсегда
// Нарушение безопасности!

// БЕЗОПАСНО - Пользовательское форматирование
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s", cfg.DBHost)
    // Пароли никогда не логируются
}
\`\`\`

**4. Понятные Сообщения об Ошибках**

\`\`\`go
// ПЛОХО
if cfg.Addr == "" {
    return errors.New("error")
}
// Пользователь видит: "error" - Какая ошибка? Где?

// ХОРОШО
if cfg.Addr == "" {
    return fmt.Errorf("%w: missing addr", ErrBadConfig)
}
// Пользователь видит: "bad config: missing addr"
// Ясно: что не так, как исправить
\`\`\`

**5. Документация Конфигурации**

\`\`\`go
func ValidateConfig(cfg Config) error {
    // Каждая валидация = неявная документация
    if cfg.RPS <= 0 || cfg.RPS > 10000 {
        return fmt.Errorf("RPS должен быть 1-10000")
    }
    // Разработчики знают валидный диапазон без чтения документации
}
\`\`\`

**Реальное Влияние:**
Платформа микросервисов с 50 сервисами:
- До валидации: В среднем 3 инцидента, связанных с конфигурацией, в неделю
- После валидации: Ноль инцидентов с конфигурацией за 6 месяцев
- MTTR (Среднее Время Восстановления): 45 минут → 0 (нет невалидных развертываний)
- Уверенность разработчиков: Можно развертывать без ручной проверки конфигурации`
		},
		uz: {
			title: `Konfiguratsiya validatsiyasi`,
			solutionCode: `package configx

import (
	"errors"
	"fmt"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
	if cfg.Addr == "" {	// majburiy maydonni tekshirish
		return fmt.Errorf("%w: missing addr", ErrBadConfig)
	}

	if cfg.RPS <= 0 {	// diapazonni validatsiya qilish
		return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
	}

	return nil	// validatsiya o'tdi
}

func FormatConfig(cfg Config) string {
	return fmt.Sprintf("addr=%s,rps=%d",	// loglar uchun xavfsiz format
		cfg.Addr, cfg.RPS)
}`,
			description: `Xatolarni erta aniqlash va observability ni yaxshilash uchun konfiguratsiya validatsiyasi va xavfsiz formatlashni amalga oshiring.

**Talablar:**
1. **ValidateConfig**: Majburiy maydonlar va qiymat diapazonlarini validatsiya qiling
2. **FormatConfig**: Konfiguratsiyani xavfsiz log satri sifatida formatlang (sirlar yo'q)
3. **Custom Error**: Validatsiya muvaffaqiyatsizliklarida ErrBadConfig qaytaring
4. **Fail-Fast**: Konfiguratsiya xatolarini ishga tushishda aniqlang, runtime da emas

**Validatsiya Patterni:**
\`\`\`go
var ErrBadConfig = errors.New("bad config")

func ValidateConfig(cfg Config) error {
    if cfg.Addr == "" {
        return fmt.Errorf("%w: missing addr", ErrBadConfig)
    }

    if cfg.RPS <= 0 {
        return fmt.Errorf("%w: invalid RPS=%d", ErrBadConfig, cfg.RPS)
    }

    return nil
}

func FormatConfig(cfg Config) string {
    // Logging uchun xavfsiz - parollar va sirlar yo'q
    return fmt.Sprintf("addr=%s,rps=%d", cfg.Addr, cfg.RPS)
}
\`\`\`

**Foydalanish Misoli:**
\`\`\`go
// Yaroqli konfiguratsiya
cfg := Config{Addr: "localhost:8080", RPS: 100}
err := ValidateConfig(cfg)
// err == nil

log.Printf("Konfiguratsiya bilan ishga tushish: %s", FormatConfig(cfg))
// Chiqish: Konfiguratsiya bilan ishga tushish: addr=localhost:8080,rps=100

// Noto'g'ri: manzil yo'q
cfg = Config{RPS: 100}
err = ValidateConfig(cfg)
// err = "bad config: missing addr"

// Noto'g'ri: manfiy RPS
cfg = Config{Addr: "localhost:8080", RPS: -10}
err = ValidateConfig(cfg)
// err = "bad config: invalid RPS=-10"
\`\`\`

**Production Ishga Tushirish Jarayoni:**
\`\`\`go
func main() {
    // 1. Muhitdan konfiguratsiyani yuklash
    cfg, err := Load()
    if err != nil {
        log.Fatalf("konfig yuklash muvaffaqiyatsiz: %v", err)
        // Ilova bu yerda to'xtaydi - yomon konfig bilan ishga tushmaydi
    }

    // 2. Tozalangan konfiguratsiyani loglash
    log.Printf("Server konfiguratsiyasi: %s", FormatConfig(cfg))

    // 3. Yaroqli konfig bilan serverni ishga tushirish
    server := NewServer(cfg)
    if err := server.Start(); err != nil {
        log.Fatalf("server ishga tushirish muvaffaqiyatsiz: %v", err)
    }
}
\`\`\`

**Sirlarni Boshqarish:**
\`\`\`go
// YOMON - Sezgir ma'lumotlarni loglaydi
type Config struct {
    DBPassword string
    APIKey     string
}

func (c Config) String() string {
    return fmt.Sprintf("db_pass=%s,api_key=%s", c.DBPassword, c.APIKey)
}
// Xavfsizlik incidenti kutilmoqda!

// YAXSHI - Xavfsiz formatlash
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s,api_endpoint=%s",
        cfg.DBHost, cfg.APIEndpoint)
    // Parollar va kalitlar loglanmaydi
}
\`\`\`

**Validatsiya Eng Yaxshi Amaliyotlari:**
\`\`\`go
// Majburiy maydonlarni tekshirish
if cfg.Addr == "" {
    return ErrBadConfig  // Darhol muvaffaqiyatsiz
}

// Diapazonlarni validatsiya qilish
if cfg.RPS <= 0 || cfg.RPS > 100000 {
    return fmt.Errorf("RPS 1-100000 bo'lishi kerak, %d olindi", cfg.RPS)
}

// Formatni validatsiya qilish
if !strings.Contains(cfg.Addr, ":") {
    return fmt.Errorf("addr portni o'z ichiga olishi kerak: %s", cfg.Addr)
}
\`\`\`

**Cheklovlar:**
- ValidateConfig har qanday validatsiya muvaffaqiyatsizligida ErrBadConfig qaytaring
- Addr bo'sh satr bo'lmasligi kerak
- RPS noldan katta bo'lishi kerak
- FormatConfig "addr=...,rps=..." formatini qaytaradi`,
			hint1: `ValidateConfig da: if cfg.Addr == "" return ErrBadConfig ishlating, keyin if cfg.RPS <= 0 return ErrBadConfig.`,
			hint2: `FormatConfig da: fmt.Sprintf ni "addr=%s,rps=%d" format satri va cfg.Addr, cfg.RPS argumentlari bilan ishlating.`,
			whyItMatters: `Konfiguratsiya validatsiyasi runtime o'rniga ishga tushishda xatolarni aniqlab production incident larni oldini oladi.

**Nima uchun bu muhim:**

**1. Fail-Fast Printsipi**

\`\`\`go
// Validatsiyasiz
func main() {
    cfg, _ := Load()  // Xatoni e'tiborsiz qoldirib
    server := NewServer(cfg)
    server.Start()    // 30 daqiqadan keyin bo'sh Addr ishlatilganda qulab tushadi
}

// Validatsiya bilan
func main() {
    cfg, err := Load()
    if err != nil {
        log.Fatalf("noto'g'ri konfiguratsiya: %v", err)  // Darhol muvaffaqiyatsiz
    }
    server := NewServer(cfg)  // Yomon konfig bilan bu yerga yetib kelmaydi
}
\`\`\`

**2. Haqiqiy Production Incident**

E-commerce platformasi, Black Friday 2023:
- Typo bilan deployment: \`APP_RPS=""\` (bo'sh satr)
- Validatsiyasiz: Ilova muvaffaqiyatli ishga tushdi
- 15 daqiqadan keyin: Nol rate limiting → database overload → sayt qulab tushdi
- Yo'qotilgan daromad: 15 daqiqada $2M
- Validatsiya bilan: Ilova ishga tushishda muvaffaqiyatsiz bo'lardi, avtomatik rollback

**3. Xavfsizlik: Xavfsiz Logging**

\`\`\`go
// XAVFLI - Loglar monitoring asboblariga oqib ketadi
type Config struct {
    DBPassword string
    JWTSecret  string
}

log.Printf("Config: %+v", cfg)
// Loglar: Config: {DBPassword:secret123 JWTSecret:jwt_key}
// Endi CloudWatch, Datadog, Splunk da abadiy
// Xavfsizlik buzilishi!

// XAVFSIZ - Maxsus formatlash
func FormatConfig(cfg Config) string {
    return fmt.Sprintf("db_host=%s", cfg.DBHost)
    // Parollar hech qachon loglanmaydi
}
\`\`\`

**4. Aniq Xato Xabarlari**

\`\`\`go
// YOMON
if cfg.Addr == "" {
    return errors.New("error")
}
// Foydalanuvchi ko'radi: "error" - Qanday xato? Qayerda?

// YAXSHI
if cfg.Addr == "" {
    return fmt.Errorf("%w: missing addr", ErrBadConfig)
}
// Foydalanuvchi ko'radi: "bad config: missing addr"
// Aniq: nima noto'g'ri, qanday tuzatish kerak
\`\`\`

**5. Konfiguratsiya Hujjatlari**

\`\`\`go
func ValidateConfig(cfg Config) error {
    // Har bir validatsiya = yashirin hujjat
    if cfg.RPS <= 0 || cfg.RPS > 10000 {
        return fmt.Errorf("RPS 1-10000 bo'lishi kerak")
    }
    // Dasturchilar hujjatlarni o'qimasdan yaroqli diapazonni biladilar
}
\`\`\`

**Haqiqiy Ta'sir:**
50 xizmatli Microservices platformasi:
- Validatsiyadan oldin: Haftasiga o'rtacha 3 konfiguratsiya bilan bog'liq incident
- Validatsiyadan keyin: 6 oyda nol konfiguratsiya incident
- MTTR (O'rtacha Tiklash Vaqti): 45 daqiqa → 0 (noto'g'ri deployment yo'q)
- Dasturchilarning ishonchi: Qo'lda konfiguratsiyani tekshirmasdan deploy qilish mumkin`
		}
	}
};

export default task;
