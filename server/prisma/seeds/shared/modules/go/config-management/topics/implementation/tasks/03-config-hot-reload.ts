import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-config-hot-reload',
	title: 'Configuration Hot Reload with Signal Handling',
	difficulty: 'hard',
	tags: ['go', 'config', 'signals', 'concurrency'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement configuration hot reload without service restart using OS signals and atomic operations for zero-downtime config updates.

**Requirements:**
1. **WatchSignal**: Listen for SIGHUP signal to trigger reload
2. **ReloadConfig**: Atomically reload config from disk
3. **GetConfig**: Thread-safe config access using atomic.Value
4. **Graceful**: Zero downtime, no request failures during reload

**Hot Reload Pattern:**
\`\`\`go
type Manager struct {
    config atomic.Value  // Holds *Config
    path   string
}

func NewManager(path string) (*Manager, error) {
    m := &Manager{path: path}
    cfg, err := loadFromFile(path)
    if err != nil {
        return nil, err
    }
    m.config.Store(cfg)
    return m, nil
}

func (m *Manager) GetConfig() Config {
    return *m.config.Load().(*Config)
}

func (m *Manager) ReloadConfig() error {
    cfg, err := loadFromFile(m.path)
    if err != nil {
        return err
    }
    m.config.Store(cfg)  // Atomic swap
    return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGHUP)
    defer signal.Stop(sigCh)

    for {
        select {
        case <-ctx.Done():
            return
        case <-sigCh:
            if err := m.ReloadConfig(); err != nil {
                log.Printf("reload failed: %v", err)
            } else {
                log.Println("config reloaded")
            }
        }
    }
}
\`\`\`

**Example Usage:**
\`\`\`go
func main() {
    // 1. Initialize manager
    mgr, err := NewManager("config.json")
    if err != nil {
        log.Fatalf("init failed: %v", err)
    }

    // 2. Start signal watcher in background
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    go mgr.WatchSignal(ctx)

    // 3. Use config in handlers
    http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        cfg := mgr.GetConfig()  // Always gets latest
        limiter := rate.NewLimiter(rate.Limit(cfg.RPS), 1)

        if !limiter.Allow() {
            http.Error(w, "rate limit", 429)
            return
        }
        // Process request with current config
    })

    log.Println("Server started. Send SIGHUP to reload config")
    http.ListenAndServe(":8080", nil)
}
\`\`\`

**Real-World Scenario:**
\`\`\`bash
# Terminal 1: Start server
$ go run main.go
Server started. Send SIGHUP to reload config
Config: rps=100

# Terminal 2: Update config file
$ echo '{"rps": 500}' > config.json

# Terminal 3: Send reload signal
$ kill -HUP $(pgrep main)

# Terminal 1: See reload
Config reloaded
Config: rps=500

# No restart needed!
# Active connections continue
# New requests use new config immediately
\`\`\`

**Production Benefits:**
\`\`\`go
// Update rate limits without restart
// Before: 5 minutes downtime per config change
// After: Zero downtime, instant config updates

// Use cases:
// - Adjust rate limits during traffic spikes
// - Enable/disable features via config
// - Update timeouts based on monitoring
// - Change log levels for debugging
\`\`\`

**Thread Safety:**
\`\`\`go
// WRONG - Race condition
var globalConfig Config

func GetConfig() Config {
    return globalConfig  // Read
}

func ReloadConfig() {
    globalConfig = loadNew()  // Write - RACE!
}

// RIGHT - Atomic operations
var globalConfig atomic.Value

func GetConfig() Config {
    return *globalConfig.Load().(*Config)  // Atomic read
}

func ReloadConfig() {
    cfg := loadNew()
    globalConfig.Store(cfg)  // Atomic write
}
\`\`\`

**Constraints:**
- Must use atomic.Value for thread-safe config storage
- WatchSignal must listen for SIGHUP signal
- ReloadConfig validates before applying (fail safe)
- GetConfig returns current config, never stale
- No mutex locks (use atomic operations only)`,
	initialCode: `package configx

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
)

type Config struct {
	RPS int \`json:"rps"\`
}

type Manager struct {
	config atomic.Value // holds *Config
	path   string
}

// TODO: Implement NewManager
// Load initial config from path
// Store in atomic.Value
// Return initialized manager
func NewManager(path string) (*Manager, error) {
	return nil, nil
}

// TODO: Implement GetConfig
// Load config from atomic.Value
// Return current Config (not pointer)
func (m *Manager) GetConfig() Config {
	return Config{}
}

// TODO: Implement ReloadConfig
// Load config from m.path
// Validate before applying
// Store atomically in m.config
func (m *Manager) ReloadConfig() error {
	return nil
}

// TODO: Implement WatchSignal
// Create signal channel for SIGHUP
// Loop: wait for signal or ctx.Done()
// Call ReloadConfig on SIGHUP
func (m *Manager) WatchSignal(ctx context.Context) {
	// TODO: Implement
}

// Helper function (provided)
func loadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}`,
	testCode: `package configx

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, err := NewManager(f.Name())
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if mgr == nil {
		t.Error("expected non-nil manager")
	}
}

func Test2(t *testing.T) {
	_, err := NewManager("/nonexistent/path.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func Test3(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, _ := NewManager(f.Name())
	cfg := mgr.GetConfig()
	if cfg.RPS != 100 {
		t.Errorf("expected RPS 100, got %d", cfg.RPS)
	}
}

func Test4(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, _ := NewManager(f.Name())
	os.WriteFile(f.Name(), []byte(\`{"rps": 200}\`), 0644)
	err := mgr.ReloadConfig()
	if err != nil {
		t.Errorf("expected nil error on reload, got %v", err)
	}
	cfg := mgr.GetConfig()
	if cfg.RPS != 200 {
		t.Errorf("expected RPS 200 after reload, got %d", cfg.RPS)
	}
}

func Test5(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, _ := NewManager(f.Name())
	os.Remove(f.Name())
	err := mgr.ReloadConfig()
	if err == nil {
		t.Error("expected error for deleted file")
	}
}

func Test6(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, _ := NewManager(f.Name())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		mgr.WatchSignal(ctx)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("WatchSignal did not exit on context cancel")
	}
}

func Test7(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 50})
	f.Close()
	mgr, _ := NewManager(f.Name())
	cfg1 := mgr.GetConfig()
	cfg2 := mgr.GetConfig()
	if cfg1.RPS != cfg2.RPS {
		t.Error("GetConfig should return consistent values")
	}
}

func Test8(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr, _ := NewManager(f.Name())
	os.WriteFile(f.Name(), []byte("invalid json"), 0644)
	err := mgr.ReloadConfig()
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
	cfg := mgr.GetConfig()
	if cfg.RPS != 100 {
		t.Errorf("expected original RPS 100 preserved, got %d", cfg.RPS)
	}
}

func Test9(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 0})
	f.Close()
	mgr, _ := NewManager(f.Name())
	cfg := mgr.GetConfig()
	if cfg.RPS != 0 {
		t.Errorf("expected RPS 0, got %d", cfg.RPS)
	}
}

func Test10(t *testing.T) {
	f, _ := os.CreateTemp("", "config*.json")
	defer os.Remove(f.Name())
	json.NewEncoder(f).Encode(Config{RPS: 100})
	f.Close()
	mgr1, _ := NewManager(f.Name())
	mgr2, _ := NewManager(f.Name())
	os.WriteFile(f.Name(), []byte(\`{"rps": 500}\`), 0644)
	mgr1.ReloadConfig()
	if mgr1.GetConfig().RPS != 500 {
		t.Error("mgr1 should have reloaded")
	}
	if mgr2.GetConfig().RPS != 100 {
		t.Error("mgr2 should be independent")
	}
}
`,
	solutionCode: `package configx

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
)

type Config struct {
	RPS int \`json:"rps"\`
}

type Manager struct {
	config atomic.Value // holds *Config
	path   string
}

func NewManager(path string) (*Manager, error) {
	m := &Manager{path: path}	// create manager
	cfg, err := loadFromFile(path)	// load initial config
	if err != nil {
		return nil, err
	}
	m.config.Store(cfg)	// store atomically
	return m, nil
}

func (m *Manager) GetConfig() Config {
	return *m.config.Load().(*Config)	// atomic load, dereference
}

func (m *Manager) ReloadConfig() error {
	cfg, err := loadFromFile(m.path)	// load from disk
	if err != nil {
		return err	// validation failed
	}
	m.config.Store(cfg)	// atomic swap
	return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
	sigCh := make(chan os.Signal, 1)	// buffered signal channel
	signal.Notify(sigCh, syscall.SIGHUP)	// register for SIGHUP
	defer signal.Stop(sigCh)	// cleanup on exit

	for {
		select {
		case <-ctx.Done():	// context cancelled
			return
		case <-sigCh:	// SIGHUP received
			if err := m.ReloadConfig(); err != nil {
				log.Printf("reload failed: %v", err)
			} else {
				log.Println("config reloaded")
			}
		}
	}
}

func loadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}`,
	hint1: `In NewManager: create Manager with path, call loadFromFile, then m.config.Store(cfg). In GetConfig: use m.config.Load().(*Config) and dereference with *.`,
	hint2: `In WatchSignal: create signal channel with make(chan os.Signal, 1), use signal.Notify(ch, syscall.SIGHUP), select on ctx.Done() and signal channel.`,
	whyItMatters: `Hot reload enables zero-downtime configuration updates, critical for production systems that require 99.99% uptime.

**Why This Matters:**

**1. Zero-Downtime Updates**

\`\`\`go
// Without hot reload
// 1. Update config file
// 2. kubectl rollout restart deployment  → 5 min downtime
// 3. Wait for pods to restart
// 4. Hope no errors

// With hot reload
// 1. Update config file
// 2. kill -HUP <pid>  → instant, zero downtime
// 3. Config active immediately
// 4. No connection drops
\`\`\`

**2. Real Production Incident**

Black Friday, e-commerce platform:
- Traffic spike: 1000 → 50,000 req/sec
- Rate limit (RPS=1000) causing legitimate user blocks
- Need to increase to RPS=10,000 immediately

Without hot reload:
- Restart required → 5 minutes downtime
- Lost revenue: $50K/minute × 5 = $250K
- Customers lost to competitors

With hot reload:
- Update config.json: RPS=10000
- kill -HUP → instant
- Zero downtime, zero revenue loss
- Crisis averted in 10 seconds

**3. Thread Safety: Why atomic.Value**

\`\`\`go
// WRONG - Data race
var config Config

// Goroutine 1: Handler
func HandleRequest() {
    rps := config.RPS  // READ
    // Race: config might change mid-read
}

// Goroutine 2: Reload
func Reload() {
    config = newConfig  // WRITE - RACE!
}

// Result: go run -race → DATA RACE DETECTED
// Production: corrupted config, crashes, undefined behavior

// RIGHT - Atomic operations
var config atomic.Value

func HandleRequest() {
    cfg := config.Load().(*Config)  // Atomic read
    rps := cfg.RPS  // Safe: cfg is snapshot
}

func Reload() {
    config.Store(newConfig)  // Atomic write
}

// Result: no races, safe concurrent access
\`\`\`

**4. Signal Handling Pattern**

\`\`\`go
// SIGHUP: reload config (convention)
// SIGTERM: graceful shutdown
// SIGINT: interrupt (Ctrl+C)

// Production deployment:
// 1. kubectl edit configmap → update config
// 2. kubectl exec pod -- kill -HUP 1
// 3. Config reloaded, zero downtime

// Kubernetes ConfigMap + Hot Reload = Perfect match
\`\`\`

**5. Gradual Rollout**

\`\`\`go
// Update config on 1 pod, test, then rollout
// Pod 1: kill -HUP → new config
// Monitor: errors? latency? CPU?
// If good: rollout to remaining pods
// If bad: revert instantly with another SIGHUP
\`\`\`

**Real Impact:**

Payment processing service:
- 100 config changes per month
- Before hot reload:
  - Each change: 5 min downtime
  - 100 × 5 = 500 minutes downtime/month
  - Revenue loss: $500K/month
  - Customer complaints: 50/month

- After hot reload:
  - Each change: 0 downtime
  - Revenue loss: $0
  - Customer complaints: 0
  - Deployment confidence: 100%

**6. Feature Flags Without External Service**

\`\`\`go
type Config struct {
    EnableNewFeature bool \`json:"enable_new_feature"\`
    RPS              int  \`json:"rps"\`
}

// Enable feature for testing
// 1. Update config: enable_new_feature=true
// 2. kill -HUP
// 3. Test in production with 1% traffic
// 4. If issues: disable instantly
// No LaunchDarkly/Split.io needed for simple flags
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Горячая перезагрузка конфигурации',
			solutionCode: `package configx

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
)

type Config struct {
	RPS int \`json:"rps"\`
}

type Manager struct {
	config atomic.Value // хранит *Config
	path   string
}

func NewManager(path string) (*Manager, error) {
	m := &Manager{path: path}	// создаем manager
	cfg, err := loadFromFile(path)	// загружаем начальный конфиг
	if err != nil {
		return nil, err
	}
	m.config.Store(cfg)	// атомарно сохраняем
	return m, nil
}

func (m *Manager) GetConfig() Config {
	return *m.config.Load().(*Config)	// атомарная загрузка, разыменование
}

func (m *Manager) ReloadConfig() error {
	cfg, err := loadFromFile(m.path)	// загружаем с диска
	if err != nil {
		return err	// валидация провалилась
	}
	m.config.Store(cfg)	// атомарная замена
	return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
	sigCh := make(chan os.Signal, 1)	// буферизованный канал сигналов
	signal.Notify(sigCh, syscall.SIGHUP)	// регистрируем SIGHUP
	defer signal.Stop(sigCh)	// очистка при выходе

	for {
		select {
		case <-ctx.Done():	// контекст отменен
			return
		case <-sigCh:	// получен SIGHUP
			if err := m.ReloadConfig(); err != nil {
				log.Printf("перезагрузка провалилась: %v", err)
			} else {
				log.Println("конфиг перезагружен")
			}
		}
	}
}

func loadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}`,
			description: `Реализуйте горячую перезагрузку конфигурации без перезапуска сервиса используя OS сигналы и атомарные операции для обновления конфига без простоя.

**Требования:**
1. **WatchSignal**: Прослушивание SIGHUP сигнала для запуска перезагрузки
2. **ReloadConfig**: Атомарная перезагрузка конфига с диска
3. **GetConfig**: Потокобезопасный доступ к конфигу через atomic.Value
4. **Graceful**: Нулевой простой, отсутствие отказов запросов во время перезагрузки

**Паттерн Горячей Перезагрузки:**
\`\`\`go
type Manager struct {
    config atomic.Value  // Хранит *Config
    path   string
}

func NewManager(path string) (*Manager, error) {
    m := &Manager{path: path}
    cfg, err := loadFromFile(path)
    if err != nil {
        return nil, err
    }
    m.config.Store(cfg)
    return m, nil
}

func (m *Manager) GetConfig() Config {
    return *m.config.Load().(*Config)
}

func (m *Manager) ReloadConfig() error {
    cfg, err := loadFromFile(m.path)
    if err != nil {
        return err
    }
    m.config.Store(cfg)  // Атомарная замена
    return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGHUP)
    defer signal.Stop(sigCh)

    for {
        select {
        case <-ctx.Done():
            return
        case <-sigCh:
            if err := m.ReloadConfig(); err != nil {
                log.Printf("перезагрузка провалилась: %v", err)
            } else {
                log.Println("конфиг перезагружен")
            }
        }
    }
}
\`\`\`

**Пример Использования:**
\`\`\`go
func main() {
    // 1. Инициализация менеджера
    mgr, err := NewManager("config.json")
    if err != nil {
        log.Fatalf("инициализация провалилась: %v", err)
    }

    // 2. Запуск наблюдателя сигналов в фоне
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    go mgr.WatchSignal(ctx)

    // 3. Использование конфига в обработчиках
    http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        cfg := mgr.GetConfig()  // Всегда получает последний
        limiter := rate.NewLimiter(rate.Limit(cfg.RPS), 1)

        if !limiter.Allow() {
            http.Error(w, "rate limit", 429)
            return
        }
        // Обработка запроса с текущим конфигом
    })

    log.Println("Сервер запущен. Отправьте SIGHUP для перезагрузки конфига")
    http.ListenAndServe(":8080", nil)
}
\`\`\`

**Сценарий из Реальной Жизни:**
\`\`\`bash
# Терминал 1: Запуск сервера
$ go run main.go
Сервер запущен. Отправьте SIGHUP для перезагрузки конфига
Config: rps=100

# Терминал 2: Обновление конфиг файла
$ echo '{"rps": 500}' > config.json

# Терминал 3: Отправка сигнала перезагрузки
$ kill -HUP $(pgrep main)

# Терминал 1: Видим перезагрузку
Конфиг перезагружен
Config: rps=500

# Перезапуск не нужен!
# Активные соединения продолжаются
# Новые запросы используют новый конфиг немедленно
\`\`\`

**Преимущества для Production:**
\`\`\`go
// Обновление rate limits без перезапуска
// До: 5 минут простоя на каждое изменение конфига
// После: Нулевой простой, мгновенные обновления конфига

// Сценарии использования:
// - Настройка rate limits во время всплесков трафика
// - Включение/выключение функций через конфиг
// - Обновление таймаутов на основе мониторинга
// - Изменение уровней логирования для отладки
\`\`\`

**Потокобезопасность:**
\`\`\`go
// НЕПРАВИЛЬНО - Состояние гонки
var globalConfig Config

func GetConfig() Config {
    return globalConfig  // Чтение
}

func ReloadConfig() {
    globalConfig = loadNew()  // Запись - ГОНКА!
}

// ПРАВИЛЬНО - Атомарные операции
var globalConfig atomic.Value

func GetConfig() Config {
    return *globalConfig.Load().(*Config)  // Атомарное чтение
}

func ReloadConfig() {
    cfg := loadNew()
    globalConfig.Store(cfg)  // Атомарная запись
}
\`\`\`

**Ограничения:**
- Должен использовать atomic.Value для потокобезопасного хранения конфига
- WatchSignal должен слушать сигнал SIGHUP
- ReloadConfig валидирует перед применением (безопасный отказ)
- GetConfig возвращает текущий конфиг, никогда не устаревший
- Никаких блокировок мьютекса (только атомарные операции)`,
			hint1: `В NewManager: создайте Manager с path, вызовите loadFromFile, затем m.config.Store(cfg). В GetConfig: используйте m.config.Load().(*Config) и разыменуйте с *.`,
			hint2: `В WatchSignal: создайте канал сигналов с make(chan os.Signal, 1), используйте signal.Notify(ch, syscall.SIGHUP), select на ctx.Done() и канале сигналов.`,
			whyItMatters: `Горячая перезагрузка обеспечивает обновление конфигурации без простоя, критично для production систем требующих 99.99% uptime.

**Почему это важно:**

**1. Обновления без Простоя**

\`\`\`go
// Без горячей перезагрузки
// 1. Обновить конфиг файл
// 2. kubectl rollout restart deployment  → 5 мин простоя
// 3. Ждать перезапуска подов
// 4. Надеяться что нет ошибок

// С горячей перезагрузкой
// 1. Обновить конфиг файл
// 2. kill -HUP <pid>  → мгновенно, нулевой простой
// 3. Конфиг активен немедленно
// 4. Никаких разрывов соединений
\`\`\`

**2. Реальный Production Инцидент**

Черная Пятница, e-commerce платформа:
- Всплеск трафика: 1000 → 50,000 req/sec
- Rate limit (RPS=1000) блокирует легитимных пользователей
- Нужно увеличить до RPS=10,000 немедленно

Без горячей перезагрузки:
- Требуется перезапуск → 5 минут простоя
- Потеря дохода: $50K/минута × 5 = $250K
- Клиенты уходят к конкурентам

С горячей перезагрузкой:
- Обновить config.json: RPS=10000
- kill -HUP → мгновенно
- Нулевой простой, нулевая потеря дохода
- Кризис предотвращен за 10 секунд

**3. Потокобезопасность: Почему atomic.Value**

\`\`\`go
// НЕПРАВИЛЬНО - Гонка данных
var config Config

// Горутина 1: Обработчик
func HandleRequest() {
    rps := config.RPS  // ЧТЕНИЕ
    // Гонка: конфиг может измениться во время чтения
}

// Горутина 2: Перезагрузка
func Reload() {
    config = newConfig  // ЗАПИСЬ - ГОНКА!
}

// Результат: go run -race → DATA RACE DETECTED
// Production: поврежденный конфиг, падения, неопределенное поведение

// ПРАВИЛЬНО - Атомарные операции
var config atomic.Value

func HandleRequest() {
    cfg := config.Load().(*Config)  // Атомарное чтение
    rps := cfg.RPS  // Безопасно: cfg это снимок
}

func Reload() {
    config.Store(newConfig)  // Атомарная запись
}

// Результат: никаких гонок, безопасный конкурентный доступ
\`\`\`

**4. Паттерн Обработки Сигналов**

\`\`\`go
// SIGHUP: перезагрузка конфига (соглашение)
// SIGTERM: graceful shutdown
// SIGINT: прерывание (Ctrl+C)

// Production развертывание:
// 1. kubectl edit configmap → обновить конфиг
// 2. kubectl exec pod -- kill -HUP 1
// 3. Конфиг перезагружен, нулевой простой

// Kubernetes ConfigMap + Горячая Перезагрузка = Идеальное сочетание
\`\`\`

**5. Постепенное Развертывание**

\`\`\`go
// Обновить конфиг на 1 поде, протестировать, затем развернуть
// Под 1: kill -HUP → новый конфиг
// Мониторинг: ошибки? задержка? CPU?
// Если хорошо: развертывание на оставшиеся поды
// Если плохо: откат мгновенно с другим SIGHUP
\`\`\`

**Реальное Влияние:**

Сервис обработки платежей:
- 100 изменений конфига в месяц
- До горячей перезагрузки:
  - Каждое изменение: 5 мин простоя
  - 100 × 5 = 500 минут простоя/месяц
  - Потеря дохода: $500K/месяц
  - Жалобы клиентов: 50/месяц

- После горячей перезагрузки:
  - Каждое изменение: 0 простоя
  - Потеря дохода: $0
  - Жалобы клиентов: 0
  - Уверенность в развертывании: 100%

**6. Feature Flags Без Внешнего Сервиса**

\`\`\`go
type Config struct {
    EnableNewFeature bool \`json:"enable_new_feature"\`
    RPS              int  \`json:"rps"\`
}

// Включить функцию для тестирования
// 1. Обновить конфиг: enable_new_feature=true
// 2. kill -HUP
// 3. Тестировать в production с 1% трафика
// 4. Если проблемы: отключить мгновенно
// Не нужны LaunchDarkly/Split.io для простых флагов
\`\`\``
		},
		uz: {
			title: `Konfiguratsiyani hot reload`,
			solutionCode: `package configx

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
)

type Config struct {
	RPS int \`json:"rps"\`
}

type Manager struct {
	config atomic.Value // *Config ni saqlaydi
	path   string
}

func NewManager(path string) (*Manager, error) {
	m := &Manager{path: path}	// manager yaratamiz
	cfg, err := loadFromFile(path)	// boshlang'ich konfigni yuklaymiz
	if err != nil {
		return nil, err
	}
	m.config.Store(cfg)	// atomik saqlash
	return m, nil
}

func (m *Manager) GetConfig() Config {
	return *m.config.Load().(*Config)	// atomik yuklash, dereference
}

func (m *Manager) ReloadConfig() error {
	cfg, err := loadFromFile(m.path)	// diskdan yuklaymiz
	if err != nil {
		return err	// validatsiya muvaffaqiyatsiz
	}
	m.config.Store(cfg)	// atomik almashtirish
	return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
	sigCh := make(chan os.Signal, 1)	// buferlangan signal kanali
	signal.Notify(sigCh, syscall.SIGHUP)	// SIGHUP uchun ro'yxatdan o'tish
	defer signal.Stop(sigCh)	// chiqishda tozalash

	for {
		select {
		case <-ctx.Done():	// kontekst bekor qilindi
			return
		case <-sigCh:	// SIGHUP qabul qilindi
			if err := m.ReloadConfig(); err != nil {
				log.Printf("qayta yuklash muvaffaqiyatsiz: %v", err)
			} else {
				log.Println("konfig qayta yuklandi")
			}
		}
	}
}

func loadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}`,
			description: `Nol to'xtovsiz konfig yangilashlari uchun OS signallar va atomik operatsiyalardan foydalanib xizmat qayta ishga tushirmasdan konfiguratsiya hot reload ni amalga oshiring.

**Talablar:**
1. **WatchSignal**: Qayta yuklashni boshlash uchun SIGHUP signaliga quloq soling
2. **ReloadConfig**: Diskdan konfigni atomik qayta yuklang
3. **GetConfig**: atomic.Value yordamida thread-safe konfig kirishi
4. **Graceful**: Nol to'xtov, qayta yuklash paytida request muvaffaqiyatsizliklari yo'q

**Hot Reload Patterni:**
\`\`\`go
type Manager struct {
    config atomic.Value  // *Config ni saqlaydi
    path   string
}

func NewManager(path string) (*Manager, error) {
    m := &Manager{path: path}
    cfg, err := loadFromFile(path)
    if err != nil {
        return nil, err
    }
    m.config.Store(cfg)
    return m, nil
}

func (m *Manager) GetConfig() Config {
    return *m.config.Load().(*Config)
}

func (m *Manager) ReloadConfig() error {
    cfg, err := loadFromFile(m.path)
    if err != nil {
        return err
    }
    m.config.Store(cfg)  // Atomik almashtirish
    return nil
}

func (m *Manager) WatchSignal(ctx context.Context) {
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGHUP)
    defer signal.Stop(sigCh)

    for {
        select {
        case <-ctx.Done():
            return
        case <-sigCh:
            if err := m.ReloadConfig(); err != nil {
                log.Printf("qayta yuklash muvaffaqiyatsiz: %v", err)
            } else {
                log.Println("konfig qayta yuklandi")
            }
        }
    }
}
\`\`\`

**Foydalanish Misoli:**
\`\`\`go
func main() {
    // 1. Manager ni ishga tushirish
    mgr, err := NewManager("config.json")
    if err != nil {
        log.Fatalf("ishga tushirish muvaffaqiyatsiz: %v", err)
    }

    // 2. Signal kuzatuvchisini fonda ishga tushirish
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    go mgr.WatchSignal(ctx)

    // 3. Handlarlarda konfigdan foydalanish
    http.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        cfg := mgr.GetConfig()  // Har doim eng oxirgisini oladi
        limiter := rate.NewLimiter(rate.Limit(cfg.RPS), 1)

        if !limiter.Allow() {
            http.Error(w, "rate limit", 429)
            return
        }
        // Joriy konfig bilan requestni qayta ishlash
    })

    log.Println("Server ishga tushdi. Konfigni qayta yuklash uchun SIGHUP yuboring")
    http.ListenAndServe(":8080", nil)
}
\`\`\`

**Haqiqiy Dunyo Stsenariysi:**
\`\`\`bash
# Terminal 1: Serverni ishga tushiring
$ go run main.go
Server ishga tushdi. Konfigni qayta yuklash uchun SIGHUP yuboring
Config: rps=100

# Terminal 2: Konfig faylini yangilang
$ echo '{"rps": 500}' > config.json

# Terminal 3: Reload signalini yuboring
$ kill -HUP $(pgrep main)

# Terminal 1: Reload ni ko'ring
Konfig qayta yuklandi
Config: rps=500

# Qayta ishga tushirish kerak emas!
# Faol ulanishlar davom etadi
# Yangi so'rovlar darhol yangi konfigdan foydalanadi
\`\`\`

**Production Foydalari:**
\`\`\`go
// Qayta ishga tushirmasdan rate limit ni yangilash
// Oldin: Har bir konfig o'zgarishi uchun 5 daqiqa to'xtov
// Keyin: Nol to'xtov, bir zumda konfig yangilanishlari

// Foydalanish holatlari:
// - Trafik ko'tarilishi paytida rate limit ni sozlash
// - Konfig orqali xususiyatlarni yoqish/o'chirish
// - Monitoring asosida timeout larni yangilash
// - Debug qilish uchun log darajalarini o'zgartirish
\`\`\`

**Thread Xavfsizligi:**
\`\`\`go
// NOTO'G'RI - Race condition
var globalConfig Config

func GetConfig() Config {
    return globalConfig  // O'qish
}

func ReloadConfig() {
    globalConfig = loadNew()  // Yozish - RACE!
}

// TO'G'RI - Atomik operatsiyalar
var globalConfig atomic.Value

func GetConfig() Config {
    return *globalConfig.Load().(*Config)  // Atomik o'qish
}

func ReloadConfig() {
    cfg := loadNew()
    globalConfig.Store(cfg)  // Atomik yozish
}
\`\`\`

**Cheklovlar:**
- Thread-safe konfig saqlash uchun atomic.Value dan foydalanish kerak
- WatchSignal SIGHUP signaliga quloq solishi kerak
- ReloadConfig qo'llashdan oldin validatsiya qiladi (fail safe)
- GetConfig joriy konfigni qaytaradi, hech qachon eskimagan
- Mutex lock yo'q (faqat atomik operatsiyalar)`,
			hint1: `NewManager da: path bilan Manager yarating, loadFromFile ni chaqiring, keyin m.config.Store(cfg). GetConfig da: m.config.Load().(*Config) ishlating va * bilan dereference qiling.`,
			hint2: `WatchSignal da: make(chan os.Signal, 1) bilan signal kanalini yarating, signal.Notify(ch, syscall.SIGHUP) ishlating, ctx.Done() va signal kanalida select qiling.`,
			whyItMatters: `Hot reload 99.99% uptime talab qiladigan production tizimlar uchun muhim bo'lgan nol to'xtovsiz konfiguratsiya yangilanishlarini yoqadi.

**Nima uchun bu muhim:**

**1. Nol-To'xtov Yangilanishlar**

\`\`\`go
// Hot reload siz
// 1. Konfig faylini yangilash
// 2. kubectl rollout restart deployment  → 5 daqiqa to'xtov
// 3. Podlarning qayta ishga tushishini kutish
// 4. Xatolar yo'qligiga umid qilish

// Hot reload bilan
// 1. Konfig faylini yangilash
// 2. kill -HUP <pid>  → bir zumda, nol to'xtov
// 3. Konfig darhol faol
// 4. Ulanish uzilishlari yo'q
\`\`\`

**2. Haqiqiy Production Incident**

Black Friday, e-commerce platformasi:
- Trafik ko'tarilishi: 1000 → 50,000 req/sec
- Rate limit (RPS=1000) qonuniy foydalanuvchilarni bloklaydi
- Darhol RPS=10,000 ga oshirish kerak

Hot reload siz:
- Qayta ishga tushirish kerak → 5 daqiqa to'xtov
- Daromad yo'qotilishi: $50K/daqiqa × 5 = $250K
- Mijozlar raqobatchilarga ketadi

Hot reload bilan:
- config.json ni yangilash: RPS=10000
- kill -HUP → bir zumda
- Nol to'xtov, nol daromad yo'qotilishi
- Inqiroz 10 soniyada bartaraf etildi

**3. Thread Xavfsizligi: Nega atomic.Value**

\`\`\`go
// NOTO'G'RI - Ma'lumotlar poygasi
var config Config

// Goroutine 1: Handler
func HandleRequest() {
    rps := config.RPS  // O'QISH
    // Race: o'qish paytida konfig o'zgarishi mumkin
}

// Goroutine 2: Reload
func Reload() {
    config = newConfig  // YOZISH - RACE!
}

// Natija: go run -race → DATA RACE DETECTED
// Production: buzilgan konfig, qulab tushishlar, noaniq xatti-harakat

// TO'G'RI - Atomik operatsiyalar
var config atomic.Value

func HandleRequest() {
    cfg := config.Load().(*Config)  // Atomik o'qish
    rps := cfg.RPS  // Xavfsiz: cfg snapshot
}

func Reload() {
    config.Store(newConfig)  // Atomik yozish
}

// Natija: race yo'q, xavfsiz concurrent kirish
\`\`\`

**4. Signal Boshqarish Patterni**

\`\`\`go
// SIGHUP: konfigni qayta yuklash (konvensiya)
// SIGTERM: graceful shutdown
// SIGINT: interrupt (Ctrl+C)

// Production deployment:
// 1. kubectl edit configmap → konfigni yangilash
// 2. kubectl exec pod -- kill -HUP 1
// 3. Konfig qayta yuklandi, nol to'xtov

// Kubernetes ConfigMap + Hot Reload = Mukammal mos kelish
\`\`\`

**5. Bosqichma-bosqich Rollout**

\`\`\`go
// 1 podda konfigni yangilash, test qilish, keyin rollout
// Pod 1: kill -HUP → yangi konfig
// Monitoring: xatolar? kechikish? CPU?
// Agar yaxshi bo'lsa: qolgan podlarga rollout
// Agar yomon bo'lsa: boshqa SIGHUP bilan bir zumda qaytarish
\`\`\`

**Haqiqiy Ta'sir:**

To'lov qayta ishlash xizmati:
- Oyiga 100 konfig o'zgarishi
- Hot reload dan oldin:
  - Har bir o'zgarish: 5 daqiqa to'xtov
  - 100 × 5 = 500 daqiqa to'xtov/oy
  - Daromad yo'qotilishi: $500K/oy
  - Mijozlarning shikoyatlari: 50/oy

- Hot reload dan keyin:
  - Har bir o'zgarish: 0 to'xtov
  - Daromad yo'qotilishi: $0
  - Mijozlarning shikoyatlari: 0
  - Deployment ishonchi: 100%

**6. Tashqi Xizmatsiz Feature Flaglar**

\`\`\`go
type Config struct {
    EnableNewFeature bool \`json:"enable_new_feature"\`
    RPS              int  \`json:"rps"\`
}

// Test qilish uchun xususiyatni yoqish
// 1. Konfigni yangilash: enable_new_feature=true
// 2. kill -HUP
// 3. 1% trafik bilan production da test qilish
// 4. Agar muammolar bo'lsa: bir zumda o'chirish
// Oddiy flaglar uchun LaunchDarkly/Split.io kerak emas
\`\`\``
		}
	}
};

export default task;
