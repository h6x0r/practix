import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-config-multi-source',
	title: 'Multi-Source Configuration Merge with Priority',
	difficulty: 'hard',
	tags: ['go', 'config', '12-factor', 'yaml'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement configuration loading from multiple sources (defaults, file, environment) with proper precedence for flexible deployment.

**Requirements:**
1. **MergeConfigs**: Merge configs with priority: env > file > defaults
2. **LoadFromFile**: Parse YAML config file with error handling
3. **ApplyEnvOverrides**: Override file config with environment variables
4. **Complete**: Final config must have all required fields set

**Multi-Source Priority Pattern:**
\`\`\`go
type Config struct {
    Host    string
    Port    int
    Timeout int
    Debug   bool
}

// Priority: Environment > File > Defaults
func LoadConfig(filePath string) (Config, error) {
    // 1. Start with defaults
    cfg := DefaultConfig()

    // 2. Override with file (if exists)
    if filePath != "" {
        fileCfg, err := LoadFromFile(filePath)
        if err != nil && !os.IsNotExist(err) {
            return cfg, err  // Fail on parse error, not missing file
        }
        cfg = MergeConfigs(cfg, fileCfg)
    }

    // 3. Override with environment
    cfg = ApplyEnvOverrides(cfg)

    // 4. Validate complete config
    return cfg, ValidateConfig(cfg)
}

func DefaultConfig() Config {
    return Config{
        Host:    "localhost",
        Port:    8080,
        Timeout: 30,
        Debug:   false,
    }
}

func MergeConfigs(base, override Config) Config {
    result := base

    // Only override non-zero values
    if override.Host != "" {
        result.Host = override.Host
    }
    if override.Port != 0 {
        result.Port = override.Port
    }
    if override.Timeout != 0 {
        result.Timeout = override.Timeout
    }
    // bool: always override (can't detect zero value)
    result.Debug = override.Debug

    return result
}

func ApplyEnvOverrides(cfg Config) Config {
    if host := os.Getenv("APP_HOST"); host != "" {
        cfg.Host = host
    }
    if port := os.Getenv("APP_PORT"); port != "" {
        if p, err := strconv.Atoi(port); err == nil {
            cfg.Port = p
        }
    }
    if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
        if t, err := strconv.Atoi(timeout); err == nil {
            cfg.Timeout = t
        }
    }
    if debug := os.Getenv("APP_DEBUG"); debug == "true" {
        cfg.Debug = true
    }
    return cfg
}
\`\`\`

**Example Usage:**
\`\`\`go
// Scenario 1: Development (no file, just defaults)
cfg, _ := LoadConfig("")
// cfg = {Host: "localhost", Port: 8080, Timeout: 30, Debug: false}

// Scenario 2: Production with config file
// config.yaml:
// host: "0.0.0.0"
// port: 443
// timeout: 60

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 443, Timeout: 60, Debug: false}

// Scenario 3: Production with env overrides
// config.yaml: host=0.0.0.0, port=443
// Environment: APP_PORT=8443, APP_DEBUG=true

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 8443, Timeout: 60, Debug: true}
//                        ^^^^              ^^^        ^^^^
//                        from env          from file  from env
\`\`\`

**Real-World Scenario:**
\`\`\`yaml
# base-config.yaml (committed to git)
host: "localhost"
port: 8080
timeout: 30
debug: false

# Docker deployment
# Use file as base, override with env
docker run -e APP_HOST=0.0.0.0 -e APP_PORT=80 myapp

# Kubernetes deployment
# ConfigMap for file, Secrets for sensitive env vars
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 60
---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    env:
    - name: APP_DEBUG
      value: "false"
    - name: APP_TIMEOUT
      value: "120"  # Override for production
\`\`\`

**YAML Parsing:**
\`\`\`go
import "gopkg.in/yaml.v3"

func LoadFromFile(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }

    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return Config{}, fmt.Errorf("parse yaml: %w", err)
    }

    return cfg, nil
}
\`\`\`

**Merge Strategy:**
\`\`\`go
// Use zero-value detection
// String: "" means not set
// Int: 0 means not set
// Bool: no zero-value detection (always override)

// WRONG
func MergeConfigs(base, override Config) Config {
    if override.Port != 0 {  // Problem: can't set port to 0
        base.Port = override.Port
    }
    return base
}

// BETTER - Use pointers for optional fields
type Config struct {
    Port *int  // nil = not set, 0 = explicitly zero
}

// SIMPLE - Document that 0 means "use default"
// For this task, use simple zero-value detection
\`\`\`

**Constraints:**
- LoadConfig must try all three sources: defaults, file, env
- Priority order: Environment > File > Defaults
- Missing file is OK, parse error is not
- Environment variables override everything
- Final config must pass ValidateConfig`,
	initialCode: `package configx

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Host    string \`yaml:"host"\`
	Port    int    \`yaml:"port"\`
	Timeout int    \`yaml:"timeout"\`
	Debug   bool   \`yaml:"debug"\`
}

// TODO: Implement DefaultConfig
// Return Config with default values:
// Host="localhost", Port=8080, Timeout=30, Debug=false
func DefaultConfig() Config {
	return Config{}
}

// TODO: Implement LoadFromFile
// Read YAML file from path
// Unmarshal into Config struct
// Return error if file exists but parse fails
func LoadFromFile(path string) (Config, error) {
	return Config{}, nil
}

// TODO: Implement MergeConfigs
// Merge override into base
// Only override non-zero values (except bool)
// Return merged config
func MergeConfigs(base, override Config) Config {
	return Config{}
}

// TODO: Implement ApplyEnvOverrides
// Read APP_HOST, APP_PORT, APP_TIMEOUT, APP_DEBUG
// Override cfg with environment values
// Convert string to int for Port/Timeout
// Return updated config
func ApplyEnvOverrides(cfg Config) Config {
	return Config{}
}

// TODO: Implement LoadConfig
// Start with DefaultConfig()
// Merge with LoadFromFile (if path provided)
// Apply ApplyEnvOverrides
// Return final config
func LoadConfig(filePath string) (Config, error) {
	return Config{}, nil
}`,
	testCode: `package configx

import (
	"os"
	"testing"
)

func Test1(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Host != "localhost" || cfg.Port != 8080 || cfg.Timeout != 30 || cfg.Debug != false {
		t.Errorf("unexpected default config: %+v", cfg)
	}
}

func Test2(t *testing.T) {
	base := Config{Host: "base", Port: 80, Timeout: 10}
	override := Config{Host: "override", Port: 443}
	result := MergeConfigs(base, override)
	if result.Host != "override" || result.Port != 443 || result.Timeout != 10 {
		t.Errorf("unexpected merge result: %+v", result)
	}
}

func Test3(t *testing.T) {
	base := Config{Host: "base", Port: 80, Timeout: 10}
	override := Config{}
	result := MergeConfigs(base, override)
	if result.Host != "base" || result.Port != 80 || result.Timeout != 10 {
		t.Errorf("expected base preserved, got: %+v", result)
	}
}

func Test4(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.yaml")
	defer os.Remove(f.Name())
	f.WriteString("host: filehost\nport: 9090\n")
	f.Close()
	cfg, err := LoadFromFile(f.Name())
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.Host != "filehost" || cfg.Port != 9090 {
		t.Errorf("unexpected config: %+v", cfg)
	}
}

func Test5(t *testing.T) {
	_, err := LoadFromFile("/nonexistent/path.yaml")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func Test6(t *testing.T) {
	os.Setenv("APP_HOST", "envhost")
	os.Setenv("APP_PORT", "7070")
	defer os.Unsetenv("APP_HOST")
	defer os.Unsetenv("APP_PORT")
	cfg := ApplyEnvOverrides(DefaultConfig())
	if cfg.Host != "envhost" || cfg.Port != 7070 {
		t.Errorf("env override failed: %+v", cfg)
	}
}

func Test7(t *testing.T) {
	os.Setenv("APP_DEBUG", "true")
	defer os.Unsetenv("APP_DEBUG")
	cfg := ApplyEnvOverrides(DefaultConfig())
	if !cfg.Debug {
		t.Error("expected Debug=true from env")
	}
}

func Test8(t *testing.T) {
	os.Unsetenv("APP_HOST")
	os.Unsetenv("APP_PORT")
	os.Unsetenv("APP_TIMEOUT")
	os.Unsetenv("APP_DEBUG")
	cfg, err := LoadConfig("")
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.Host != "localhost" || cfg.Port != 8080 {
		t.Errorf("expected defaults, got: %+v", cfg)
	}
}

func Test9(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.yaml")
	defer os.Remove(f.Name())
	f.WriteString("host: filehost\nport: 9090\ntimeout: 60\n")
	f.Close()
	os.Setenv("APP_PORT", "8443")
	defer os.Unsetenv("APP_PORT")
	cfg, err := LoadConfig(f.Name())
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if cfg.Host != "filehost" || cfg.Port != 8443 || cfg.Timeout != 60 {
		t.Errorf("priority merge failed: %+v", cfg)
	}
}

func Test10(t *testing.T) {
	base := Config{Host: "base", Debug: false}
	override := Config{Debug: true}
	result := MergeConfigs(base, override)
	if !result.Debug {
		t.Error("expected Debug=true after merge")
	}
}
`,
	solutionCode: `package configx

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Host    string \`yaml:"host"\`
	Port    int    \`yaml:"port"\`
	Timeout int    \`yaml:"timeout"\`
	Debug   bool   \`yaml:"debug"\`
}

func DefaultConfig() Config {
	return Config{	// return defaults
		Host:    "localhost",
		Port:    8080,
		Timeout: 30,
		Debug:   false,
	}
}

func LoadFromFile(path string) (Config, error) {
	data, err := os.ReadFile(path)	// read file
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {	// parse YAML
		return Config{}, fmt.Errorf("parse yaml: %w", err)
	}

	return cfg, nil
}

func MergeConfigs(base, override Config) Config {
	result := base	// start with base

	if override.Host != "" {	// override non-empty
		result.Host = override.Host
	}
	if override.Port != 0 {	// override non-zero
		result.Port = override.Port
	}
	if override.Timeout != 0 {
		result.Timeout = override.Timeout
	}
	result.Debug = override.Debug	// always override bool

	return result
}

func ApplyEnvOverrides(cfg Config) Config {
	if host := os.Getenv("APP_HOST"); host != "" {	// check env var
		cfg.Host = host
	}
	if port := os.Getenv("APP_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {	// convert to int
			cfg.Port = p
		}
	}
	if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
		if t, err := strconv.Atoi(timeout); err == nil {
			cfg.Timeout = t
		}
	}
	if debug := os.Getenv("APP_DEBUG"); debug == "true" {	// parse bool
		cfg.Debug = true
	}
	return cfg
}

func LoadConfig(filePath string) (Config, error) {
	cfg := DefaultConfig()	// 1. defaults

	if filePath != "" {	// 2. file (optional)
		fileCfg, err := LoadFromFile(filePath)
		if err != nil && !os.IsNotExist(err) {	// fail on parse error only
			return cfg, err
		}
		if err == nil {	// merge if file loaded
			cfg = MergeConfigs(cfg, fileCfg)
		}
	}

	cfg = ApplyEnvOverrides(cfg)	// 3. environment

	return cfg, nil	// return merged config
}`,
	hint1: `In MergeConfigs: copy base to result, then check each field of override - if non-zero (for Host/Port/Timeout check != "" or != 0), set result field. For Debug, always copy.`,
	hint2: `In LoadConfig: call DefaultConfig(), then LoadFromFile (check os.IsNotExist for missing file), MergeConfigs if file loaded, then ApplyEnvOverrides. Chain them: cfg = MergeConfigs(cfg, fileCfg).`,
	whyItMatters: `Multi-source configuration enables the same application binary to run in different environments without code changes, following 12-Factor App principles.

**Why This Matters:**

**1. One Binary, All Environments**

\`\`\`go
// Same compiled binary runs everywhere:
// - Developer laptop (defaults only)
// - Staging server (config file)
// - Production (config file + env overrides)
// - Kubernetes (ConfigMap + Secrets)

// NO NEED for separate builds per environment
// NO NEED for environment-specific code
\`\`\`

**2. Real Production Setup**

\`\`\`bash
# Development: No config file needed
$ go run main.go
# Uses defaults: localhost:8080, timeout=30s

# Staging: Config file in git
$ cat config.yaml
host: "staging.api.com"
port: 443
timeout: 60

$ go run main.go
# Uses: staging.api.com:443, timeout=60s

# Production: Config file + secrets via env
$ export APP_HOST="api.prod.com"
$ export APP_DEBUG="false"
$ go run main.go
# Uses: api.prod.com:443 (env override), timeout=60s (file)
\`\`\`

**3. Kubernetes Best Practice**

\`\`\`yaml
# ConfigMap: Non-sensitive config
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 120

---
# Secret: Sensitive overrides
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
stringData:
  APP_DEBUG: "false"
  APP_TIMEOUT: "300"  # Override for specific pod

---
# Pod: Mounts both
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    volumeMounts:
    - name: config
      mountPath: /etc/config
    envFrom:
    - secretRef:
        name: app-secrets
  volumes:
  - name: config
    configMap:
      name: app-config

# Result:
# - Base config from ConfigMap (config.yaml)
# - Sensitive values from Secret (env vars)
# - Perfect separation of concerns
\`\`\`

**4. Gradual Migration**

\`\`\`go
// Year 1: Hardcoded config (bad)
cfg := Config{Host: "localhost", Port: 8080}

// Year 2: Add defaults (better)
cfg := DefaultConfig()

// Year 3: Add file support (good)
cfg := LoadFromFile("config.yaml")

// Year 4: Add env overrides (best)
cfg := LoadConfig("config.yaml")  // Defaults + File + Env

// Backward compatible at each step
// No breaking changes
\`\`\`

**5. Priority System Explained**

\`\`\`go
// Why Environment > File > Defaults?

// Defaults: Reasonable values for development
DefaultConfig() // host=localhost, port=8080

// File: Shared team configuration (in git)
config.yaml     // host=staging.api.com, port=443

// Environment: Deployment-specific secrets (not in git)
APP_HOST=prod.api.com  // Override for production

// Priority chain ensures:
// 1. Developers work without config files (defaults)
// 2. Teams share common config (file in git)
// 3. Secrets stay out of git (env vars)
// 4. Production overrides everything (env vars win)
\`\`\`

**6. Security Benefits**

\`\`\`go
// BAD - Secrets in config file (committed to git)
// config.yaml (NEVER DO THIS):
database_password: "prod_secret_123"
api_key: "sk_live_real_key"

// GOOD - Secrets via environment
// config.yaml (safe to commit):
database_host: "db.prod.com"
database_port: 5432

// .env (NOT committed):
export DB_PASSWORD="prod_secret_123"
export API_KEY="sk_live_real_key"

// App code:
cfg := LoadConfig("config.yaml")
cfg = ApplyEnvOverrides(cfg)  // Adds secrets from env
\`\`\`

**Real Impact:**

Microservices platform with 30 services:

Before multi-source config:
- 30 different config files per environment
- 90 config files to maintain (dev/staging/prod)
- Secrets in git → security audit finding
- Config changes require code deployment

After multi-source config:
- 1 config file per service (30 total)
- Secrets in environment variables
- Config changes without code deployment
- Same binary for all environments

Results:
- Configuration errors: 5/month → 0/month
- Security findings: 3 → 0
- Deployment time: 20 min → 5 min
- Developer onboarding: 2 days → 4 hours`,
	order: 3,
	translations: {
		ru: {
			title: 'Конфигурация из нескольких источников',
			solutionCode: `package configx

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Host    string \`yaml:"host"\`
	Port    int    \`yaml:"port"\`
	Timeout int    \`yaml:"timeout"\`
	Debug   bool   \`yaml:"debug"\`
}

func DefaultConfig() Config {
	return Config{	// возвращаем дефолты
		Host:    "localhost",
		Port:    8080,
		Timeout: 30,
		Debug:   false,
	}
}

func LoadFromFile(path string) (Config, error) {
	data, err := os.ReadFile(path)	// читаем файл
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {	// парсим YAML
		return Config{}, fmt.Errorf("parse yaml: %w", err)
	}

	return cfg, nil
}

func MergeConfigs(base, override Config) Config {
	result := base	// начинаем с base

	if override.Host != "" {	// переопределяем непустые
		result.Host = override.Host
	}
	if override.Port != 0 {	// переопределяем ненулевые
		result.Port = override.Port
	}
	if override.Timeout != 0 {
		result.Timeout = override.Timeout
	}
	result.Debug = override.Debug	// всегда переопределяем bool

	return result
}

func ApplyEnvOverrides(cfg Config) Config {
	if host := os.Getenv("APP_HOST"); host != "" {	// проверяем env var
		cfg.Host = host
	}
	if port := os.Getenv("APP_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {	// конвертируем в int
			cfg.Port = p
		}
	}
	if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
		if t, err := strconv.Atoi(timeout); err == nil {
			cfg.Timeout = t
		}
	}
	if debug := os.Getenv("APP_DEBUG"); debug == "true" {	// парсим bool
		cfg.Debug = true
	}
	return cfg
}

func LoadConfig(filePath string) (Config, error) {
	cfg := DefaultConfig()	// 1. дефолты

	if filePath != "" {	// 2. файл (опционально)
		fileCfg, err := LoadFromFile(filePath)
		if err != nil && !os.IsNotExist(err) {	// провал только на ошибке парсинга
			return cfg, err
		}
		if err == nil {	// слияние если файл загружен
			cfg = MergeConfigs(cfg, fileCfg)
		}
	}

	cfg = ApplyEnvOverrides(cfg)	// 3. окружение

	return cfg, nil	// возвращаем слитый конфиг
}`,
			description: `Реализуйте загрузку конфигурации из нескольких источников (дефолты, файл, окружение) с правильным приоритетом для гибкого развертывания.

**Требования:**
1. **MergeConfigs**: Слияние конфигов с приоритетом: env > file > defaults
2. **LoadFromFile**: Парсинг YAML конфиг файла с обработкой ошибок
3. **ApplyEnvOverrides**: Переопределение файловой конфигурации переменными окружения
4. **Complete**: Финальная конфигурация должна иметь все необходимые поля установленными

**Паттерн Мультиисточниковой Приоритетности:**
\`\`\`go
type Config struct {
    Host    string
    Port    int
    Timeout int
    Debug   bool
}

// Приоритет: Environment > File > Defaults
func LoadConfig(filePath string) (Config, error) {
    // 1. Начать с дефолтов
    cfg := DefaultConfig()

    // 2. Переопределить файлом (если существует)
    if filePath != "" {
        fileCfg, err := LoadFromFile(filePath)
        if err != nil && !os.IsNotExist(err) {
            return cfg, err  // Провал при ошибке парсинга, не при отсутствии файла
        }
        cfg = MergeConfigs(cfg, fileCfg)
    }

    // 3. Переопределить окружением
    cfg = ApplyEnvOverrides(cfg)

    // 4. Валидировать полный конфиг
    return cfg, ValidateConfig(cfg)
}

func DefaultConfig() Config {
    return Config{
        Host:    "localhost",
        Port:    8080,
        Timeout: 30,
        Debug:   false,
    }
}

func MergeConfigs(base, override Config) Config {
    result := base

    // Переопределить только ненулевые значения
    if override.Host != "" {
        result.Host = override.Host
    }
    if override.Port != 0 {
        result.Port = override.Port
    }
    if override.Timeout != 0 {
        result.Timeout = override.Timeout
    }
    // bool: всегда переопределить (не может обнаружить нулевое значение)
    result.Debug = override.Debug

    return result
}

func ApplyEnvOverrides(cfg Config) Config {
    if host := os.Getenv("APP_HOST"); host != "" {
        cfg.Host = host
    }
    if port := os.Getenv("APP_PORT"); port != "" {
        if p, err := strconv.Atoi(port); err == nil {
            cfg.Port = p
        }
    }
    if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
        if t, err := strconv.Atoi(timeout); err == nil {
            cfg.Timeout = t
        }
    }
    if debug := os.Getenv("APP_DEBUG"); debug == "true" {
        cfg.Debug = true
    }
    return cfg
}
\`\`\`

**Пример Использования:**
\`\`\`go
// Сценарий 1: Разработка (нет файла, только дефолты)
cfg, _ := LoadConfig("")
// cfg = {Host: "localhost", Port: 8080, Timeout: 30, Debug: false}

// Сценарий 2: Production с конфиг файлом
// config.yaml:
// host: "0.0.0.0"
// port: 443
// timeout: 60

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 443, Timeout: 60, Debug: false}

// Сценарий 3: Production с env переопределениями
// config.yaml: host=0.0.0.0, port=443
// Environment: APP_PORT=8443, APP_DEBUG=true

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 8443, Timeout: 60, Debug: true}
//                        ^^^^              ^^^        ^^^^
//                        из env          из файла    из env
\`\`\`

**Сценарий из Реальной Жизни:**
\`\`\`yaml
# base-config.yaml (закоммичен в git)
host: "localhost"
port: 8080
timeout: 30
debug: false

# Docker развертывание
# Использовать файл как базу, переопределить с env
docker run -e APP_HOST=0.0.0.0 -e APP_PORT=80 myapp

# Kubernetes развертывание
# ConfigMap для файла, Secrets для чувствительных env vars
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 60
---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    env:
    - name: APP_DEBUG
      value: "false"
    - name: APP_TIMEOUT
      value: "120"  # Переопределение для production
\`\`\`

**YAML Парсинг:**
\`\`\`go
import "gopkg.in/yaml.v3"

func LoadFromFile(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }

    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return Config{}, fmt.Errorf("parse yaml: %w", err)
    }

    return cfg, nil
}
\`\`\`

**Стратегия Слияния:**
\`\`\`go
// Использовать обнаружение нулевых значений
// String: "" означает не установлено
// Int: 0 означает не установлено
// Bool: нет обнаружения нулевых значений (всегда переопределять)

// НЕПРАВИЛЬНО
func MergeConfigs(base, override Config) Config {
    if override.Port != 0 {  // Проблема: нельзя установить порт в 0
        base.Port = override.Port
    }
    return base
}

// ЛУЧШЕ - Использовать указатели для опциональных полей
type Config struct {
    Port *int  // nil = не установлено, 0 = явно ноль
}

// ПРОСТОЕ - Документировать что 0 означает "использовать дефолт"
// Для этой задачи используйте простое обнаружение нулевых значений
\`\`\`

**Ограничения:**
- LoadConfig должен пытаться все три источника: дефолты, файл, env
- Порядок приоритета: Environment > File > Defaults
- Отсутствующий файл OK, ошибка парсинга - нет
- Переменные окружения переопределяют всё
- Финальный конфиг должен пройти ValidateConfig`,
			hint1: `В MergeConfigs: скопируйте base в result, затем проверьте каждое поле override - если ненулевое (для Host/Port/Timeout проверьте != "" или != 0), установите поле result. Для Debug всегда копируйте.`,
			hint2: `В LoadConfig: вызовите DefaultConfig(), затем LoadFromFile (проверьте os.IsNotExist для отсутствующего файла), MergeConfigs если файл загружен, затем ApplyEnvOverrides.`,
			whyItMatters: `Мультиисточниковая конфигурация позволяет одному бинарнику приложения работать в разных окружениях без изменений кода, следуя принципам 12-Factor App.

**Почему это важно:**

**1. Один Бинарник, Все Окружения**

\`\`\`go
// Один скомпилированный бинарник работает везде:
// - Ноутбук разработчика (только дефолты)
// - Staging сервер (конфиг файл)
// - Production (конфиг файл + env переопределения)
// - Kubernetes (ConfigMap + Secrets)

// НЕ НУЖНЫ отдельные сборки для каждого окружения
// НЕ НУЖЕН код специфичный для окружения
\`\`\`

**2. Реальная Production Настройка**

\`\`\`bash
# Разработка: Конфиг файл не нужен
$ go run main.go
# Используются дефолты: localhost:8080, timeout=30s

# Staging: Конфиг файл в git
$ cat config.yaml
host: "staging.api.com"
port: 443
timeout: 60

$ go run main.go
# Используется: staging.api.com:443, timeout=60s

# Production: Конфиг файл + секреты через env
$ export APP_HOST="api.prod.com"
$ export APP_DEBUG="false"
$ go run main.go
# Используется: api.prod.com:443 (env переопределение), timeout=60s (файл)
\`\`\`

**3. Kubernetes Лучшая Практика**

\`\`\`yaml
# ConfigMap: Несекретный конфиг
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 120

---
# Secret: Чувствительные переопределения
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
stringData:
  APP_DEBUG: "false"
  APP_TIMEOUT: "300"  # Переопределение для конкретного пода

---
# Pod: Монтирует оба
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    volumeMounts:
    - name: config
      mountPath: /etc/config
    envFrom:
    - secretRef:
        name: app-secrets
  volumes:
  - name: config
    configMap:
      name: app-config

# Результат:
# - Базовый конфиг из ConfigMap (config.yaml)
# - Чувствительные значения из Secret (env vars)
# - Идеальное разделение ответственности
\`\`\`

**4. Постепенная Миграция**

\`\`\`go
// Год 1: Хардкодный конфиг (плохо)
cfg := Config{Host: "localhost", Port: 8080}

// Год 2: Добавить дефолты (лучше)
cfg := DefaultConfig()

// Год 3: Добавить поддержку файлов (хорошо)
cfg := LoadFromFile("config.yaml")

// Год 4: Добавить env переопределения (отлично)
cfg := LoadConfig("config.yaml")  // Дефолты + Файл + Env

// Обратно совместимо на каждом шаге
// Без ломающих изменений
\`\`\`

**5. Объяснение Системы Приоритетов**

\`\`\`go
// Почему Environment > File > Defaults?

// Дефолты: Разумные значения для разработки
DefaultConfig() // host=localhost, port=8080

// Файл: Общая конфигурация команды (в git)
config.yaml     // host=staging.api.com, port=443

// Окружение: Секреты специфичные для развертывания (не в git)
APP_HOST=prod.api.com  // Переопределение для production

// Цепь приоритетов гарантирует:
// 1. Разработчики работают без конфиг файлов (дефолты)
// 2. Команды делятся общим конфигом (файл в git)
// 3. Секреты остаются вне git (env vars)
// 4. Production переопределяет всё (env vars побеждают)
\`\`\`

**6. Преимущества Безопасности**

\`\`\`go
// ПЛОХО - Секреты в конфиг файле (закоммичены в git)
// config.yaml (НИКОГДА ТАК НЕ ДЕЛАЙТЕ):
database_password: "prod_secret_123"
api_key: "sk_live_real_key"

// ХОРОШО - Секреты через окружение
// config.yaml (безопасно коммитить):
database_host: "db.prod.com"
database_port: 5432

// .env (НЕ закоммичен):
export DB_PASSWORD="prod_secret_123"
export API_KEY="sk_live_real_key"

// Код приложения:
cfg := LoadConfig("config.yaml")
cfg = ApplyEnvOverrides(cfg)  // Добавляет секреты из env
\`\`\`

**Реальное Влияние:**

Платформа микросервисов с 30 сервисами:

До мультиисточниковой конфигурации:
- 30 разных конфиг файлов на окружение
- 90 конфиг файлов для поддержки (dev/staging/prod)
- Секреты в git → находка аудита безопасности
- Изменения конфига требуют развертывания кода

После мультиисточниковой конфигурации:
- 1 конфиг файл на сервис (30 всего)
- Секреты в переменных окружения
- Изменения конфига без развертывания кода
- Один бинарник для всех окружений

Результаты:
- Ошибки конфигурации: 5/месяц → 0/месяц
- Находки безопасности: 3 → 0
- Время развертывания: 20 мин → 5 мин
- Онбординг разработчиков: 2 дня → 4 часа`
		},
		uz: {
			title: `Ko'p manbali konfiguratsiya`,
			solutionCode: `package configx

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Host    string \`yaml:"host"\`
	Port    int    \`yaml:"port"\`
	Timeout int    \`yaml:"timeout"\`
	Debug   bool   \`yaml:"debug"\`
}

func DefaultConfig() Config {
	return Config{	// standart qiymatlarni qaytaramiz
		Host:    "localhost",
		Port:    8080,
		Timeout: 30,
		Debug:   false,
	}
}

func LoadFromFile(path string) (Config, error) {
	data, err := os.ReadFile(path)	// faylni o'qiymiz
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {	// YAML ni parse qilamiz
		return Config{}, fmt.Errorf("yaml parse: %w", err)
	}

	return cfg, nil
}

func MergeConfigs(base, override Config) Config {
	result := base	// base dan boshlaymiz

	if override.Host != "" {	// bo'sh bo'lmaganlarni override qilamiz
		result.Host = override.Host
	}
	if override.Port != 0 {	// noldan farqlilarni override qilamiz
		result.Port = override.Port
	}
	if override.Timeout != 0 {
		result.Timeout = override.Timeout
	}
	result.Debug = override.Debug	// har doim bool ni override qilamiz

	return result
}

func ApplyEnvOverrides(cfg Config) Config {
	if host := os.Getenv("APP_HOST"); host != "" {	// env var ni tekshiramiz
		cfg.Host = host
	}
	if port := os.Getenv("APP_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {	// int ga aylantiramiz
			cfg.Port = p
		}
	}
	if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
		if t, err := strconv.Atoi(timeout); err == nil {
			cfg.Timeout = t
		}
	}
	if debug := os.Getenv("APP_DEBUG"); debug == "true" {	// bool ni parse qilamiz
		cfg.Debug = true
	}
	return cfg
}

func LoadConfig(filePath string) (Config, error) {
	cfg := DefaultConfig()	// 1. standart qiymatlar

	if filePath != "" {	// 2. fayl (ixtiyoriy)
		fileCfg, err := LoadFromFile(filePath)
		if err != nil && !os.IsNotExist(err) {	// faqat parse xatosida muvaffaqiyatsiz
			return cfg, err
		}
		if err == nil {	// fayl yuklangan bo'lsa birlashtirish
			cfg = MergeConfigs(cfg, fileCfg)
		}
	}

	cfg = ApplyEnvOverrides(cfg)	// 3. muhit

	return cfg, nil	// birlashtirilgan konfigni qaytarish
}`,
			description: `Moslashuvchan deployment uchun to'g'ri ustunlik bilan bir nechta manbalardan (standart, fayl, muhit) konfiguratsiya yuklashni amalga oshiring.

**Talablar:**
1. **MergeConfigs**: Ustunlik bilan konfiglarni birlashtirish: env > file > defaults
2. **LoadFromFile**: Xatolarni qayta ishlash bilan YAML konfig faylini parse qilish
3. **ApplyEnvOverrides**: Muhit o'zgaruvchilari bilan fayl konfigini override qilish
4. **Complete**: Yakuniy konfig barcha kerakli maydonlarga ega bo'lishi kerak

**Ko'p Manbali Ustunlik Patterni:**
\`\`\`go
type Config struct {
    Host    string
    Port    int
    Timeout int
    Debug   bool
}

// Ustunlik: Environment > File > Defaults
func LoadConfig(filePath string) (Config, error) {
    // 1. Standart qiymatlardan boshlash
    cfg := DefaultConfig()

    // 2. Fayl bilan override qilish (agar mavjud bo'lsa)
    if filePath != "" {
        fileCfg, err := LoadFromFile(filePath)
        if err != nil && !os.IsNotExist(err) {
            return cfg, err  // Parse xatosida muvaffaqiyatsiz, fayl yo'qligida emas
        }
        cfg = MergeConfigs(cfg, fileCfg)
    }

    // 3. Muhit bilan override qilish
    cfg = ApplyEnvOverrides(cfg)

    // 4. To'liq konfigni validatsiya qilish
    return cfg, ValidateConfig(cfg)
}

func DefaultConfig() Config {
    return Config{
        Host:    "localhost",
        Port:    8080,
        Timeout: 30,
        Debug:   false,
    }
}

func MergeConfigs(base, override Config) Config {
    result := base

    // Faqat noldan farqli qiymatlarni override qilish
    if override.Host != "" {
        result.Host = override.Host
    }
    if override.Port != 0 {
        result.Port = override.Port
    }
    if override.Timeout != 0 {
        result.Timeout = override.Timeout
    }
    // bool: har doim override qilish (nol qiymatni aniqlay olmaydi)
    result.Debug = override.Debug

    return result
}

func ApplyEnvOverrides(cfg Config) Config {
    if host := os.Getenv("APP_HOST"); host != "" {
        cfg.Host = host
    }
    if port := os.Getenv("APP_PORT"); port != "" {
        if p, err := strconv.Atoi(port); err == nil {
            cfg.Port = p
        }
    }
    if timeout := os.Getenv("APP_TIMEOUT"); timeout != "" {
        if t, err := strconv.Atoi(timeout); err == nil {
            cfg.Timeout = t
        }
    }
    if debug := os.Getenv("APP_DEBUG"); debug == "true" {
        cfg.Debug = true
    }
    return cfg
}
\`\`\`

**Foydalanish Misoli:**
\`\`\`go
// Stsenari 1: Development (fayl yo'q, faqat standart)
cfg, _ := LoadConfig("")
// cfg = {Host: "localhost", Port: 8080, Timeout: 30, Debug: false}

// Stsenari 2: Production konfig fayl bilan
// config.yaml:
// host: "0.0.0.0"
// port: 443
// timeout: 60

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 443, Timeout: 60, Debug: false}

// Stsenari 3: Production env override bilan
// config.yaml: host=0.0.0.0, port=443
// Environment: APP_PORT=8443, APP_DEBUG=true

cfg, _ := LoadConfig("config.yaml")
// cfg = {Host: "0.0.0.0", Port: 8443, Timeout: 60, Debug: true}
//                        ^^^^              ^^^        ^^^^
//                        env dan         fayldan    env dan
\`\`\`

**Haqiqiy Dunyo Stsenariysi:**
\`\`\`yaml
# base-config.yaml (git ga commit qilingan)
host: "localhost"
port: 8080
timeout: 30
debug: false

# Docker deployment
# Faylni asos sifatida ishlating, env bilan override qiling
docker run -e APP_HOST=0.0.0.0 -e APP_PORT=80 myapp

# Kubernetes deployment
# Fayl uchun ConfigMap, sezgir env vars uchun Secrets
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 60
---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    env:
    - name: APP_DEBUG
      value: "false"
    - name: APP_TIMEOUT
      value: "120"  # Production uchun override
\`\`\`

**YAML Parse qilish:**
\`\`\`go
import "gopkg.in/yaml.v3"

func LoadFromFile(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }

    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return Config{}, fmt.Errorf("yaml parse: %w", err)
    }

    return cfg, nil
}
\`\`\`

**Birlashtirish Strategiyasi:**
\`\`\`go
// Nol qiymat aniqlashdan foydalaning
// String: "" o'rnatilmaganligini bildiradi
// Int: 0 o'rnatilmaganligini bildiradi
// Bool: nol qiymat aniqlash yo'q (har doim override qilish)

// NOTO'G'RI
func MergeConfigs(base, override Config) Config {
    if override.Port != 0 {  // Muammo: portni 0 ga o'rnatib bo'lmaydi
        base.Port = override.Port
    }
    return base
}

// YAXSHIROQ - Ixtiyoriy maydonlar uchun pointerlardan foydalaning
type Config struct {
    Port *int  // nil = o'rnatilmagan, 0 = aniq nol
}

// ODDIY - 0 "standartdan foydalanish" ni bildiradi deb hujjatlang
// Bu vazifa uchun oddiy nol qiymat aniqlashdan foydalaning
\`\`\`

**Cheklovlar:**
- LoadConfig uchta manbani sinab ko'rishi kerak: defaults, file, env
- Ustunlik tartibi: Environment > File > Defaults
- Yo'q fayl OK, parse xatosi emas
- Muhit o'zgaruvchilari hamma narsani override qiladi
- Yakuniy konfig ValidateConfig dan o'tishi kerak`,
			hint1: `MergeConfigs da: base ni result ga nusxalang, keyin override ning har bir maydonini tekshiring - agar noldan farqli bo'lsa (Host/Port/Timeout uchun != "" yoki != 0 tekshiring), result maydonini o'rnating. Debug uchun har doim nusxalang.`,
			hint2: `LoadConfig da: DefaultConfig() ni chaqiring, keyin LoadFromFile (yo'q fayl uchun os.IsNotExist ni tekshiring), fayl yuklangan bo'lsa MergeConfigs, keyin ApplyEnvOverrides.`,
			whyItMatters: `Ko'p manbali konfiguratsiya bir xil ilova ikkiligini kod o'zgarishlarisiz turli muhitlarda ishga tushirishga imkon beradi, 12-Factor App prinsipalariga amal qiladi.

**Nima uchun bu muhim:**

**1. Bir Binary, Barcha Muhitlar**

\`\`\`go
// Bir xil compiled binary hamma joyda ishlaydi:
// - Ishlab chiquvchi noutbuk (faqat standart qiymatlar)
// - Staging server (konfig fayl)
// - Production (konfig fayl + env override)
// - Kubernetes (ConfigMap + Secrets)

// Har bir muhit uchun alohida build kerak EMAS
// Muhitga xos kod kerak EMAS
\`\`\`

**2. Haqiqiy Production Setup**

\`\`\`bash
# Development: Konfig fayl kerak emas
$ go run main.go
# Standart qiymatlar ishlatiladi: localhost:8080, timeout=30s

# Staging: Git da konfig fayl
$ cat config.yaml
host: "staging.api.com"
port: 443
timeout: 60

$ go run main.go
# Ishlatiladi: staging.api.com:443, timeout=60s

# Production: Konfig fayl + env orqali sirlar
$ export APP_HOST="api.prod.com"
$ export APP_DEBUG="false"
$ go run main.go
# Ishlatiladi: api.prod.com:443 (env override), timeout=60s (fayl)
\`\`\`

**3. Kubernetes Eng Yaxshi Amaliyot**

\`\`\`yaml
# ConfigMap: Maxfiy bo'lmagan konfig
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    host: "0.0.0.0"
    port: 8080
    timeout: 120

---
# Secret: Sezgir override lar
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
stringData:
  APP_DEBUG: "false"
  APP_TIMEOUT: "300"  # Muayyan pod uchun override

---
# Pod: Ikkalasini ham mount qiladi
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    volumeMounts:
    - name: config
      mountPath: /etc/config
    envFrom:
    - secretRef:
        name: app-secrets
  volumes:
  - name: config
    configMap:
      name: app-config

# Natija:
# - ConfigMap dan asosiy konfig (config.yaml)
# - Secret dan sezgir qiymatlar (env vars)
# - Mukammal mas'uliyat ajratish
\`\`\`

**4. Bosqichma-bosqich Migratsiya**

\`\`\`go
// Yil 1: Hardcoded konfig (yomon)
cfg := Config{Host: "localhost", Port: 8080}

// Yil 2: Standart qiymatlarni qo'shish (yaxshiroq)
cfg := DefaultConfig()

// Yil 3: Fayl qo'llab-quvvatlashni qo'shish (yaxshi)
cfg := LoadFromFile("config.yaml")

// Yil 4: Env override ni qo'shish (a'lo)
cfg := LoadConfig("config.yaml")  // Standart + Fayl + Env

// Har bir bosqichda orqaga mos keladi
// Buzuvchi o'zgarishlar yo'q
\`\`\`

**5. Ustunlik Tizimi Tushuntirildi**

\`\`\`go
// Nega Environment > File > Defaults?

// Standart qiymatlar: Development uchun oqilona qiymatlar
DefaultConfig() // host=localhost, port=8080

// Fayl: Jamoa konfiguratsiyasi (git da)
config.yaml     // host=staging.api.com, port=443

// Muhit: Deployment-specific sirlar (git da emas)
APP_HOST=prod.api.com  // Production uchun override

// Ustunlik zanjiri ta'minlaydi:
// 1. Ishlab chiquvchilar konfig fayllarsiz ishlaydi (standart)
// 2. Jamoalar umumiy konfigni baham ko'radilar (fayl git da)
// 3. Sirlar git dan tashqarida qoladi (env vars)
// 4. Production hamma narsani override qiladi (env vars yutadi)
\`\`\`

**6. Xavfsizlik Afzalliklari**

\`\`\`go
// YOMON - Konfig faylida sirlar (git ga commit qilingan)
// config.yaml (HECH QACHON BUNI QILMANG):
database_password: "prod_secret_123"
api_key: "sk_live_real_key"

// YAXSHI - Muhit orqali sirlar
// config.yaml (commit qilish xavfsiz):
database_host: "db.prod.com"
database_port: 5432

// .env (commit qilinmagan):
export DB_PASSWORD="prod_secret_123"
export API_KEY="sk_live_real_key"

// Ilova kodi:
cfg := LoadConfig("config.yaml")
cfg = ApplyEnvOverrides(cfg)  // Env dan sirlarni qo'shadi
\`\`\`

**Haqiqiy Ta'sir:**

30 xizmatli Mikroservislar platformasi:

Ko'p manbali konfiguratsiyadan oldin:
- Har muhit uchun 30 ta turli konfig fayl
- Qo'llab-quvvatlash uchun 90 konfig fayl (dev/staging/prod)
- Git da sirlar → xavfsizlik audit topilmasi
- Konfig o'zgarishlari kod deployment ni talab qiladi

Ko'p manbali konfiguratsiyadan keyin:
- Har xizmat uchun 1 konfig fayl (jami 30)
- Muhit o'zgaruvchilarida sirlar
- Kod deployment siz konfig o'zgarishlari
- Barcha muhitlar uchun bitta binary

Natijalar:
- Konfiguratsiya xatolari: 5/oy → 0/oy
- Xavfsizlik topilmalari: 3 → 0
- Deployment vaqti: 20 daqiqa → 5 daqiqa
- Dasturchi onboarding: 2 kun → 4 soat`
		}
	}
};

export default task;
