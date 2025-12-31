import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-config-env-parsing',
	title: 'Environment Variable Parsing with Defaults',
	difficulty: 'medium',	tags: ['go', 'config', 'env', '12-factor'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement robust environment variable parsing with default values and type conversion for production config management.

**Requirements:**
1. **ParseEnv**: Parse config from map[string]string without os.Getenv
2. **ApplyDefaults**: Set default values (RPS=100) for missing fields
3. **Type Conversion**: Convert string RPS to int using strconv.Atoi
4. **Error Handling**: Return errors for invalid numeric values

**12-Factor App Pattern:**
\`\`\`go
type Config struct {
    Addr string  // Required: server address
    RPS  int     // Optional: requests per second, default 100
}

func ParseEnv(env map[string]string) (Config, error) {
    cfg := Config{Addr: env["APP_ADDR"]}

    if raw := env["APP_RPS"]; raw != "" {
        rps, err := strconv.Atoi(raw)
        if err != nil {
            return cfg, fmt.Errorf("invalid RPS: %w", err)
        }
        cfg.RPS = rps
    }

    ApplyDefaults(&cfg)
    return cfg, ValidateConfig(cfg)
}

func ApplyDefaults(cfg *Config) {
    if cfg == nil {
        return
    }
    if cfg.RPS == 0 {
        cfg.RPS = 100  // Production default
    }
}
\`\`\`

**Example Usage:**
\`\`\`go
// Development: minimal config
env := map[string]string{
    "APP_ADDR": "localhost:8080",
}
cfg, _ := ParseEnv(env)
// cfg.RPS = 100 (default applied)

// Production: full config
env := map[string]string{
    "APP_ADDR": "0.0.0.0:443",
    "APP_RPS":  "1000",
}
cfg, _ := ParseEnv(env)
// cfg.RPS = 1000 (parsed from env)

// Error handling
env := map[string]string{
    "APP_ADDR": "localhost:8080",
    "APP_RPS":  "invalid",  // Not a number
}
_, err := ParseEnv(env)
// err != nil
\`\`\`

**Real-World Scenario:**
\`\`\`go
// Kubernetes deployment with ConfigMap
// configmap.yaml:
// data:
//   APP_ADDR: "0.0.0.0:8080"
//   APP_RPS: "500"

func main() {
    cfg, err := Load()  // Reads from os.Getenv
    if err != nil {
        log.Fatalf("config error: %v", err)
    }

    server := NewServer(cfg)
    log.Printf("Starting server: %s", FormatConfig(cfg))
    server.Start()
}
\`\`\`

**Constraints:**
- ParseEnv must not call os.Getenv directly
- ApplyDefaults modifies config in-place
- RPS defaults to 100 if zero or empty
- Return errors for invalid number formats`,
	initialCode: `package configx

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

// TODO: Implement Load
// Read APP_ADDR and APP_RPS from environment
// Call ParseEnv with the environment map
func Load() (Config, error) {
	var zero Config
	return zero, nil // TODO: Implement
}

// TODO: Implement ParseEnv
// Parse config from env map
// Convert RPS string to int
// Apply defaults and validate
func ParseEnv(env map[string]string) (Config, error) {
	var zero Config
	return zero, nil // TODO: Implement
}

// TODO: Implement ApplyDefaults
// Set RPS=100 if it's zero
// Modify config in-place
func ApplyDefaults(cfg *Config) {
	// TODO: Implement
}`,
	testCode: `package configx

import (
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	env := map[string]string{"APP_ADDR": "localhost:8080"}
	cfg, err := ParseEnv(env)
	if err != nil {
		t.Errorf("expected nil error for valid config, got %v", err)
	}
	if cfg.Addr != "localhost:8080" {
		t.Errorf("expected addr localhost:8080, got %s", cfg.Addr)
	}
}

func Test2(t *testing.T) {
	env := map[string]string{"APP_ADDR": "localhost:8080", "APP_RPS": "200"}
	cfg, err := ParseEnv(env)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.RPS != 200 {
		t.Errorf("expected RPS 200, got %d", cfg.RPS)
	}
}

func Test3(t *testing.T) {
	env := map[string]string{"APP_ADDR": "localhost:8080"}
	cfg, err := ParseEnv(env)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.RPS != 100 {
		t.Errorf("expected default RPS 100, got %d", cfg.RPS)
	}
}

func Test4(t *testing.T) {
	env := map[string]string{"APP_ADDR": "localhost:8080", "APP_RPS": "invalid"}
	_, err := ParseEnv(env)
	if err == nil {
		t.Error("expected error for invalid RPS")
	}
}

func Test5(t *testing.T) {
	cfg := Config{RPS: 0}
	ApplyDefaults(&cfg)
	if cfg.RPS != 100 {
		t.Errorf("expected RPS default 100, got %d", cfg.RPS)
	}
}

func Test6(t *testing.T) {
	cfg := Config{RPS: 50}
	ApplyDefaults(&cfg)
	if cfg.RPS != 50 {
		t.Errorf("expected RPS unchanged 50, got %d", cfg.RPS)
	}
}

func Test7(t *testing.T) {
	ApplyDefaults(nil)
}

func Test8(t *testing.T) {
	env := map[string]string{"APP_RPS": "100"}
	_, err := ParseEnv(env)
	if err == nil || !errors.Is(err, ErrBadConfig) {
		t.Errorf("expected ErrBadConfig for missing addr, got %v", err)
	}
}

func Test9(t *testing.T) {
	env := map[string]string{"APP_ADDR": "0.0.0.0:443", "APP_RPS": "1000"}
	cfg, err := ParseEnv(env)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.Addr != "0.0.0.0:443" || cfg.RPS != 1000 {
		t.Errorf("expected addr 0.0.0.0:443 and RPS 1000, got %s and %d", cfg.Addr, cfg.RPS)
	}
}

func Test10(t *testing.T) {
	env := map[string]string{}
	_, err := ParseEnv(env)
	if err == nil || !errors.Is(err, ErrBadConfig) {
		t.Errorf("expected ErrBadConfig for empty env, got %v", err)
	}
}
`,
	solutionCode: `package configx

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func Load() (Config, error) {
	env := map[string]string{
		"APP_ADDR": os.Getenv("APP_ADDR"),
		"APP_RPS":  os.Getenv("APP_RPS"),
	}
	return ParseEnv(env)
}

func ParseEnv(env map[string]string) (Config, error) {
	cfg := Config{
		Addr: env["APP_ADDR"],	// read address
	}

	if raw := env["APP_RPS"]; raw != "" {	// if RPS provided
		rps, err := strconv.Atoi(raw)	// convert to int
		if err != nil {
			return cfg, fmt.Errorf("invalid RPS: %w", err)
		}
		cfg.RPS = rps
	}

	ApplyDefaults(&cfg)	// apply defaults
	if err := ValidateConfig(cfg); err != nil {	// validate
		return cfg, err
	}

	return cfg, nil
}

func ApplyDefaults(cfg *Config) {
	if cfg == nil {
		return
	}
	if cfg.RPS == 0 {	// if not set
		cfg.RPS = 100	// use production default
	}
}`,
			hint1: `In ParseEnv: read Addr directly from env, for RPS check if non-empty then use strconv.Atoi, call ApplyDefaults before ValidateConfig.`,
			hint2: `In ApplyDefaults: check nil pointer first, then set cfg.RPS = 100 only if cfg.RPS == 0.`,
			whyItMatters: `Environment-based configuration is the foundation of cloud-native applications and follows the 12-Factor App methodology.

**Why This Matters:**

**1. Cloud-Native Deployment**
Modern platforms (Kubernetes, Docker, AWS ECS) use environment variables for configuration. Without proper parsing, your app crashes on startup:

\`\`\`go
// BAD - Hardcoded config (can't change without rebuild)
cfg := Config{
    Addr: "localhost:8080",
    RPS:  100,
}

// GOOD - Environment-based (configurable per environment)
cfg, err := Load()  // Reads from environment
if err != nil {
    log.Fatalf("invalid config: %v", err)
}
\`\`\`

**2. Real Production: Multi-Environment Setup**

\`\`\`bash
# Development
export APP_ADDR="localhost:8080"
export APP_RPS="10"
go run main.go

# Staging
export APP_ADDR="staging.api.com:443"
export APP_RPS="100"
go run main.go

# Production
export APP_ADDR="0.0.0.0:443"
export APP_RPS="1000"
go run main.go
\`\`\`

Same binary, different configurations. No code changes between environments.

**3. Default Values = Resilient Systems**

Production app with 50 microservices. If every service required explicit RPS config:
- 50 environment variables to manage
- Easy to forget one
- App crashes on missing var

With defaults:
- Missing RPS? Use 100
- App starts successfully
- Production stays online

**4. Type Safety**

\`\`\`go
// Without proper parsing
rps := os.Getenv("APP_RPS")  // Returns ""
// App uses zero RPS → crashes or misbehaves

// With ParseEnv
cfg, err := ParseEnv(env)
if err != nil {
    // Caught at startup, not in production
    log.Fatalf("config error: %v", err)
}
\`\`\`

**Real Impact:**
E-commerce platform with 20 microservices:
- Before: 2 production incidents per month from config errors
- After: Zero config-related incidents
- Deployment time: 30 minutes → 5 minutes (no manual config checks)`,	order: 0,
	translations: {
		ru: {
			title: 'Парсинг переменных окружения',
			solutionCode: `package configx

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func Load() (Config, error) {
	env := map[string]string{
		"APP_ADDR": os.Getenv("APP_ADDR"),
		"APP_RPS":  os.Getenv("APP_RPS"),
	}
	return ParseEnv(env)
}

func ParseEnv(env map[string]string) (Config, error) {
	cfg := Config{
		Addr: env["APP_ADDR"],	// читаем адрес
	}

	if raw := env["APP_RPS"]; raw != "" {	// если RPS указан
		rps, err := strconv.Atoi(raw)	// конвертируем в int
		if err != nil {
			return cfg, fmt.Errorf("invalid RPS: %w", err)
		}
		cfg.RPS = rps
	}

	ApplyDefaults(&cfg)	// применяем дефолты
	if err := ValidateConfig(cfg); err != nil {	// валидируем
		return cfg, err
	}

	return cfg, nil
}

func ApplyDefaults(cfg *Config) {
	if cfg == nil {
		return
	}
	if cfg.RPS == 0 {	// если не задано
		cfg.RPS = 100	// используем production дефолт
	}
}`,
			description: `Реализуйте надежный парсинг переменных окружения с значениями по умолчанию и конвертацией типов для production конфигурации.

**Требования:**
1. **ParseEnv**: Парсинг конфига из map[string]string без os.Getenv
2. **ApplyDefaults**: Установка дефолтных значений (RPS=100) для отсутствующих полей
3. **Конвертация типов**: Преобразование строки RPS в int через strconv.Atoi
4. **Обработка ошибок**: Возврат ошибок при некорректных числовых значениях`,
			hint1: `В ParseEnv: читайте Addr напрямую из env, для RPS проверьте непустоту и используйте strconv.Atoi, вызовите ApplyDefaults перед ValidateConfig.`,
			hint2: `В ApplyDefaults: сначала проверьте nil pointer, затем установите cfg.RPS = 100 только если cfg.RPS == 0.`,
			whyItMatters: `Конфигурация через переменные окружения - основа cloud-native приложений и следует методологии 12-Factor App.

**Почему это важно:**

**1. Cloud-Native Deployment**
Современные платформы (Kubernetes, Docker, AWS ECS) используют переменные окружения для конфигурации. Без правильного парсинга ваше приложение упадет при запуске:

\`\`\`go
// ПЛОХО - Хардкодная конфигурация (нельзя изменить без пересборки)
cfg := Config{
    Addr: "localhost:8080",
    RPS:  100,
}

// ХОРОШО - На основе окружения (настраивается для каждой среды)
cfg, err := Load()  // Читает из окружения
if err != nil {
    log.Fatalf("невалидная конфигурация: %v", err)
}
\`\`\`

**2. Real Production: Мультиокружение**

\`\`\`bash
# Development
export APP_ADDR="localhost:8080"
export APP_RPS="10"
go run main.go

# Staging
export APP_ADDR="staging.api.com:443"
export APP_RPS="100"
go run main.go

# Production
export APP_ADDR="0.0.0.0:443"
export APP_RPS="1000"
go run main.go
\`\`\`

Один и тот же бинарник, разные конфигурации. Никаких изменений кода между окружениями.

**3. Значения по Умолчанию = Устойчивые Системы**

Production приложение с 50 микросервисами. Если каждый сервис требует явной RPS конфигурации:
- 50 переменных окружения для управления
- Легко забыть одну
- Приложение падает при отсутствующей переменной

С дефолтами:
- RPS отсутствует? Используем 100
- Приложение запускается успешно
- Production остается онлайн

**4. Типобезопасность**

\`\`\`go
// Без правильного парсинга
rps := os.Getenv("APP_RPS")  // Возвращает ""
// Приложение использует нулевой RPS → падает или работает некорректно

// С ParseEnv
cfg, err := ParseEnv(env)
if err != nil {
    // Обнаружено при запуске, а не в production
    log.Fatalf("ошибка конфигурации: %v", err)
}
\`\`\`

**Реальное Влияние:**
E-commerce платформа с 20 микросервисами:
- До: 2 production инцидента в месяц из-за ошибок конфигурации
- После: Ноль инцидентов связанных с конфигурацией
- Время развертывания: 30 минут → 5 минут (без ручных проверок конфигурации)`
		},
		uz: {
			title: `Muhit o'zgaruvchilarini tahlil qilish`,
			solutionCode: `package configx

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Addr string
	RPS  int
}

var ErrBadConfig = errors.New("bad config")

func Load() (Config, error) {
	env := map[string]string{
		"APP_ADDR": os.Getenv("APP_ADDR"),
		"APP_RPS":  os.Getenv("APP_RPS"),
	}
	return ParseEnv(env)
}

func ParseEnv(env map[string]string) (Config, error) {
	cfg := Config{
		Addr: env["APP_ADDR"],	// manzilni o'qiymiz
	}

	if raw := env["APP_RPS"]; raw != "" {	// agar RPS berilgan bo'lsa
		rps, err := strconv.Atoi(raw)	// int ga aylantiramiz
		if err != nil {
			return cfg, fmt.Errorf("noto'g'ri RPS: %w", err)
		}
		cfg.RPS = rps
	}

	ApplyDefaults(&cfg)	// standart qiymatlarni qo'llaymiz
	if err := ValidateConfig(cfg); err != nil {	// validatsiya qilamiz
		return cfg, err
	}

	return cfg, nil
}

func ApplyDefaults(cfg *Config) {
	if cfg == nil {
		return
	}
	if cfg.RPS == 0 {	// agar o'rnatilmagan bo'lsa
		cfg.RPS = 100	// production standart qiymatini ishlatamiz
	}
}`,
			description: `Production konfiguratsiya boshqarish uchun standart qiymatlar va tur konversiyasi bilan mustahkam environment o'zgaruvchilarini parsing ni amalga oshiring.

**Talablar:**
1. **ParseEnv**: os.Getenv siz map[string]string dan konfiguratsiyani parsing qiling
2. **ApplyDefaults**: Yo'q maydonlar uchun standart qiymatlarni o'rnating (RPS=100)
3. **Tur Konversiyasi**: strconv.Atoi yordamida satr RPS ni int ga aylantiring
4. **Xatolarni Qayta Ishlash**: Noto'g'ri raqamli qiymatlar uchun xatolar qaytaring

**12-Factor App Pattern:**
\`\`\`go
func ParseEnv(env map[string]string) (Config, error) {
    cfg := Config{Addr: env["APP_ADDR"]}

    if raw := env["APP_RPS"]; raw != "" {
        rps, err := strconv.Atoi(raw)
        if err != nil {
            return cfg, fmt.Errorf("noto'g'ri RPS: %w", err)
        }
        cfg.RPS = rps
    }

    ApplyDefaults(&cfg)
    return cfg, ValidateConfig(cfg)
}
\`\`\`

**Cheklovlar:**
- ParseEnv to'g'ridan-to'g'ri os.Getenv ni chaqirmasligi kerak
- ApplyDefaults konfiguratsiyani joyida o'zgartiradi
- RPS nol yoki bo'sh bo'lsa 100 ga default
- Noto'g'ri raqam formatlari uchun xatolar qaytaring`,
			hint1: `ParseEnv da: Addr ni to'g'ridan-to'g'ri env dan o'qing, RPS uchun bo'sh emasligini tekshiring va strconv.Atoi ishlating, ValidateConfig dan oldin ApplyDefaults ni chaqiring.`,
			hint2: `ApplyDefaults da: avval nil pointer ni tekshiring, keyin cfg.RPS = 100 ni faqat cfg.RPS == 0 bo'lsa o'rnating.`,
			whyItMatters: `Environment-based konfiguratsiya cloud-native ilovalarning asosi va 12-Factor App metodologiyasiga amal qiladi.

**Nima uchun bu muhim:**

**1. Cloud-Native Deployment**
Zamonaviy platformalar (Kubernetes, Docker, AWS ECS) konfiguratsiya uchun environment o'zgaruvchilaridan foydalanadi. To'g'ri parsing bo'lmasa, ilovangiz ishga tushishda qulab tushadi:

\`\`\`go
// YOMON - Hardcoded konfiguratsiya (qayta build qilmasdan o'zgartirib bo'lmaydi)
cfg := Config{
    Addr: "localhost:8080",
    RPS:  100,
}

// YAXSHI - Environment-based (har bir muhit uchun sozlanadi)
cfg, err := Load()  // Environment dan o'qiydi
if err != nil {
    log.Fatalf("noto'g'ri konfiguratsiya: %v", err)
}
\`\`\`

**2. Real Production: Multi-Environment Setup**

\`\`\`bash
# Development
export APP_ADDR="localhost:8080"
export APP_RPS="10"
go run main.go

# Staging
export APP_ADDR="staging.api.com:443"
export APP_RPS="100"
go run main.go

# Production
export APP_ADDR="0.0.0.0:443"
export APP_RPS="1000"
go run main.go
\`\`\`

Bir xil binary, turli konfiguratsiyalar. Muhitlar orasida kod o'zgarishlari yo'q.

**3. Standart Qiymatlar = Barqaror Tizimlar**

50 microservice bilan production ilova. Har bir xizmat aniq RPS konfiguratsiyasini talab qilsa:
- 50 environment o'zgaruvchi boshqarish kerak
- Birini unutish oson
- Ilova yo'q o'zgaruvchida qulab tushadi

Standart qiymatlar bilan:
- RPS yo'q? 100 ishlating
- Ilova muvaffaqiyatli ishga tushadi
- Production onlayn qoladi

**4. Tur Xavfsizligi**

\`\`\`go
// To'g'ri parsing siz
rps := os.Getenv("APP_RPS")  // "" qaytaradi
// Ilova nol RPS ishlatadi → qulab tushadi yoki noto'g'ri ishlaydi

// ParseEnv bilan
cfg, err := ParseEnv(env)
if err != nil {
    // Ishga tushishda aniqlandi, production da emas
    log.Fatalf("konfiguratsiya xatosi: %v", err)
}
\`\`\`

**Haqiqiy Ta'sir:**
20 microservice bilan e-commerce platformasi:
- Oldin: Oyiga 2 production incident konfiguratsiya xatolaridan
- Keyin: Nol konfiguratsiya bilan bog'liq incident
- Deployment vaqti: 30 daqiqa → 5 daqiqa (qo'lda konfiguratsiyani tekshirmasdan)`
		}
	}
};

export default task;
