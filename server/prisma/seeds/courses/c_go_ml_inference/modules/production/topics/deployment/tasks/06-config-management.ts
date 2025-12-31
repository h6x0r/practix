import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-config-management',
  title: 'ML Service Configuration',
  difficulty: 'medium',
  tags: ['go', 'ml', 'config', 'environment', 'production'],
  estimatedTime: '25m',
  isPremium: false,
  order: 6,

  description: `
## ML Service Configuration

Build a configuration management system for ML inference services that supports multiple sources, validation, and hot-reloading.

### Requirements

1. **ConfigManager** - Main configuration component:
   - \`NewConfigManager() *ConfigManager\` - Create manager
   - \`Load(sources ...ConfigSource) error\` - Load from sources
   - \`Get(key string) interface{}\` - Get config value
   - \`GetString/GetInt/GetFloat/GetBool/GetDuration\` - Typed getters
   - \`Watch(callback func(Config)) func()\` - Watch for changes

2. **ConfigSource** - Configuration sources:
   - \`EnvSource\` - Environment variables
   - \`FileSource\` - JSON/YAML files
   - \`DefaultSource\` - Default values

3. **ML-Specific Config**:
   - Model path and version
   - Batch size and timeout
   - GPU/CPU device selection
   - Inference concurrency limits
   - Cache TTL settings

4. **Features**:
   - Source priority (later sources override earlier)
   - Validation with required fields
   - Type coercion (string to int, etc.)
   - Hot-reload on file changes

### Example

\`\`\`go
manager := NewConfigManager()

err := manager.Load(
    NewDefaultSource(map[string]interface{}{
        "model.batch_size": 32,
        "model.timeout":    "5s",
    }),
    NewEnvSource("ML_"),
    NewFileSource("config.yaml"),
)

batchSize := manager.GetInt("model.batch_size")
timeout := manager.GetDuration("model.timeout")

unwatch := manager.Watch(func(cfg Config) {
    log.Println("Config updated")
})
defer unwatch()
\`\`\`
`,

  initialCode: `package mlconfig

import (
	"sync"
	"time"
)

type ConfigSource interface {
}

type Config struct {
	values map[string]interface{}
}

type ConfigManager struct {
}

func NewConfigManager() *ConfigManager {
	return nil
}

func (m *ConfigManager) Load(sources ...ConfigSource) error {
	return nil
}

func (m *ConfigManager) Get(key string) interface{} {
	return nil
}

func (m *ConfigManager) GetString(key string) string {
	return ""
}

func (m *ConfigManager) GetInt(key string) int {
	return 0
}

func (m *ConfigManager) GetFloat(key string) float64 {
	return 0
}

func (m *ConfigManager) GetBool(key string) bool {
	return false
}

func (m *ConfigManager) GetDuration(key string) time.Duration {
	return 0
}

func (m *ConfigManager) Watch(callback func(Config)) func() {
	return func() {}
}

type EnvSource struct {
}

func NewEnvSource(prefix string) *EnvSource {
	return nil
}

type DefaultSource struct {
}

func NewDefaultSource(defaults map[string]interface{}) *DefaultSource {
	return nil
}`,

  solutionCode: `package mlconfig

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ConfigSource provides configuration values
type ConfigSource interface {
	Load() (map[string]interface{}, error)
	Name() string
}

// Config holds loaded configuration
type Config struct {
	values map[string]interface{}
}

// ConfigManager manages ML service configuration
type ConfigManager struct {
	mu        sync.RWMutex
	values    map[string]interface{}
	watchers  []func(Config)
	watcherMu sync.Mutex
}

// NewConfigManager creates a new configuration manager
func NewConfigManager() *ConfigManager {
	return &ConfigManager{
		values:   make(map[string]interface{}),
		watchers: make([]func(Config), 0),
	}
}

// Load loads configuration from sources
func (m *ConfigManager) Load(sources ...ConfigSource) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	newValues := make(map[string]interface{})

	for _, source := range sources {
		vals, err := source.Load()
		if err != nil {
			return err
		}
		for k, v := range vals {
			newValues[k] = v
		}
	}

	m.values = newValues
	m.notifyWatchers()

	return nil
}

func (m *ConfigManager) notifyWatchers() {
	m.watcherMu.Lock()
	watchers := make([]func(Config), len(m.watchers))
	copy(watchers, m.watchers)
	m.watcherMu.Unlock()

	cfg := Config{values: m.values}
	for _, w := range watchers {
		go w(cfg)
	}
}

// Get returns a configuration value
func (m *ConfigManager) Get(key string) interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.values[key]
}

// GetString returns a string configuration value
func (m *ConfigManager) GetString(key string) string {
	val := m.Get(key)
	if val == nil {
		return ""
	}
	switch v := val.(type) {
	case string:
		return v
	default:
		return ""
	}
}

// GetInt returns an int configuration value
func (m *ConfigManager) GetInt(key string) int {
	val := m.Get(key)
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		i, _ := strconv.Atoi(v)
		return i
	default:
		return 0
	}
}

// GetFloat returns a float configuration value
func (m *ConfigManager) GetFloat(key string) float64 {
	val := m.Get(key)
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	default:
		return 0
	}
}

// GetBool returns a bool configuration value
func (m *ConfigManager) GetBool(key string) bool {
	val := m.Get(key)
	if val == nil {
		return false
	}
	switch v := val.(type) {
	case bool:
		return v
	case string:
		b, _ := strconv.ParseBool(v)
		return b
	case int:
		return v != 0
	default:
		return false
	}
}

// GetDuration returns a duration configuration value
func (m *ConfigManager) GetDuration(key string) time.Duration {
	val := m.Get(key)
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case time.Duration:
		return v
	case string:
		d, _ := time.ParseDuration(v)
		return d
	case int:
		return time.Duration(v) * time.Millisecond
	case int64:
		return time.Duration(v) * time.Millisecond
	case float64:
		return time.Duration(v) * time.Millisecond
	default:
		return 0
	}
}

// Watch registers a callback for configuration changes
func (m *ConfigManager) Watch(callback func(Config)) func() {
	m.watcherMu.Lock()
	m.watchers = append(m.watchers, callback)
	idx := len(m.watchers) - 1
	m.watcherMu.Unlock()

	return func() {
		m.watcherMu.Lock()
		defer m.watcherMu.Unlock()
		if idx < len(m.watchers) {
			m.watchers = append(m.watchers[:idx], m.watchers[idx+1:]...)
		}
	}
}

// Reload reloads configuration from stored sources
func (m *ConfigManager) Reload(sources ...ConfigSource) error {
	return m.Load(sources...)
}

// EnvSource loads configuration from environment variables
type EnvSource struct {
	prefix string
}

// NewEnvSource creates an environment variable source
func NewEnvSource(prefix string) *EnvSource {
	return &EnvSource{prefix: prefix}
}

func (s *EnvSource) Name() string {
	return "env"
}

func (s *EnvSource) Load() (map[string]interface{}, error) {
	values := make(map[string]interface{})

	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key, value := parts[0], parts[1]
		if !strings.HasPrefix(key, s.prefix) {
			continue
		}

		// Convert ML_MODEL_BATCH_SIZE to model.batch_size
		configKey := strings.TrimPrefix(key, s.prefix)
		configKey = strings.ToLower(configKey)
		configKey = strings.ReplaceAll(configKey, "_", ".")

		values[configKey] = value
	}

	return values, nil
}

// DefaultSource provides default values
type DefaultSource struct {
	defaults map[string]interface{}
}

// NewDefaultSource creates a default value source
func NewDefaultSource(defaults map[string]interface{}) *DefaultSource {
	return &DefaultSource{defaults: defaults}
}

func (s *DefaultSource) Name() string {
	return "defaults"
}

func (s *DefaultSource) Load() (map[string]interface{}, error) {
	values := make(map[string]interface{}, len(s.defaults))
	for k, v := range s.defaults {
		values[k] = v
	}
	return values, nil
}

// FileSource loads configuration from a JSON file
type FileSource struct {
	path string
}

// NewFileSource creates a file source
func NewFileSource(path string) *FileSource {
	return &FileSource{path: path}
}

func (s *FileSource) Name() string {
	return "file:" + s.path
}

func (s *FileSource) Load() (map[string]interface{}, error) {
	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]interface{}), nil
		}
		return nil, err
	}

	var values map[string]interface{}
	if err := json.Unmarshal(data, &values); err != nil {
		return nil, err
	}

	// Flatten nested maps
	return flattenMap(values, ""), nil
}

func flattenMap(m map[string]interface{}, prefix string) map[string]interface{} {
	result := make(map[string]interface{})

	for k, v := range m {
		key := k
		if prefix != "" {
			key = prefix + "." + k
		}

		if nested, ok := v.(map[string]interface{}); ok {
			for nk, nv := range flattenMap(nested, key) {
				result[nk] = nv
			}
		} else {
			result[key] = v
		}
	}

	return result
}

// MLConfig provides typed access to ML configuration
type MLConfig struct {
	ModelPath       string
	ModelVersion    string
	BatchSize       int
	Timeout         time.Duration
	Device          string
	MaxConcurrency  int
	CacheTTL        time.Duration
	EnableProfiling bool
}

// LoadMLConfig loads ML-specific configuration
func LoadMLConfig(manager *ConfigManager) MLConfig {
	return MLConfig{
		ModelPath:       manager.GetString("model.path"),
		ModelVersion:    manager.GetString("model.version"),
		BatchSize:       manager.GetInt("model.batch_size"),
		Timeout:         manager.GetDuration("model.timeout"),
		Device:          manager.GetString("model.device"),
		MaxConcurrency:  manager.GetInt("inference.max_concurrency"),
		CacheTTL:        manager.GetDuration("cache.ttl"),
		EnableProfiling: manager.GetBool("profiling.enabled"),
	}
}
`,

  testCode: `package mlconfig

import (
	"os"
	"testing"
	"time"
)

func TestNewConfigManager(t *testing.T) {
	m := NewConfigManager()
	if m == nil {
		t.Fatal("Expected non-nil manager")
	}
}

func TestLoadDefaults(t *testing.T) {
	m := NewConfigManager()

	err := m.Load(NewDefaultSource(map[string]interface{}{
		"model.batch_size": 32,
		"model.timeout":    "5s",
	}))

	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if m.GetInt("model.batch_size") != 32 {
		t.Error("Expected batch_size 32")
	}

	if m.GetDuration("model.timeout") != 5*time.Second {
		t.Error("Expected timeout 5s")
	}
}

func TestSourcePriority(t *testing.T) {
	m := NewConfigManager()

	err := m.Load(
		NewDefaultSource(map[string]interface{}{
			"value": 1,
		}),
		NewDefaultSource(map[string]interface{}{
			"value": 2,
		}),
	)

	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if m.GetInt("value") != 2 {
		t.Error("Later source should override")
	}
}

func TestEnvSource(t *testing.T) {
	os.Setenv("ML_MODEL_BATCH_SIZE", "64")
	os.Setenv("ML_CACHE_TTL", "10m")
	defer os.Unsetenv("ML_MODEL_BATCH_SIZE")
	defer os.Unsetenv("ML_CACHE_TTL")

	m := NewConfigManager()
	err := m.Load(NewEnvSource("ML_"))

	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if m.GetString("model.batch.size") != "64" {
		t.Errorf("Expected '64', got '%s'", m.GetString("model.batch.size"))
	}
}

func TestGetTyped(t *testing.T) {
	m := NewConfigManager()
	m.Load(NewDefaultSource(map[string]interface{}{
		"string":   "hello",
		"int":      42,
		"float":    3.14,
		"bool":     true,
		"duration": "1h30m",
	}))

	if m.GetString("string") != "hello" {
		t.Error("GetString failed")
	}
	if m.GetInt("int") != 42 {
		t.Error("GetInt failed")
	}
	if m.GetFloat("float") != 3.14 {
		t.Error("GetFloat failed")
	}
	if m.GetBool("bool") != true {
		t.Error("GetBool failed")
	}
	if m.GetDuration("duration") != 90*time.Minute {
		t.Error("GetDuration failed")
	}
}

func TestTypeCoercion(t *testing.T) {
	m := NewConfigManager()
	m.Load(NewDefaultSource(map[string]interface{}{
		"int_from_string":   "42",
		"float_from_string": "3.14",
		"bool_from_string":  "true",
		"int_from_float":    3.7,
	}))

	if m.GetInt("int_from_string") != 42 {
		t.Error("String to int coercion failed")
	}
	if m.GetFloat("float_from_string") != 3.14 {
		t.Error("String to float coercion failed")
	}
	if m.GetBool("bool_from_string") != true {
		t.Error("String to bool coercion failed")
	}
	if m.GetInt("int_from_float") != 3 {
		t.Error("Float to int coercion failed")
	}
}

func TestWatch(t *testing.T) {
	m := NewConfigManager()

	called := make(chan bool, 1)
	unwatch := m.Watch(func(cfg Config) {
		called <- true
	})
	defer unwatch()

	m.Load(NewDefaultSource(map[string]interface{}{"key": "value"}))

	select {
	case <-called:
		// Success
	case <-time.After(time.Second):
		t.Error("Watch callback not called")
	}
}

func TestUnwatch(t *testing.T) {
	m := NewConfigManager()

	callCount := 0
	unwatch := m.Watch(func(cfg Config) {
		callCount++
	})

	m.Load(NewDefaultSource(map[string]interface{}{"key": "1"}))
	time.Sleep(50 * time.Millisecond)

	unwatch()

	m.Load(NewDefaultSource(map[string]interface{}{"key": "2"}))
	time.Sleep(50 * time.Millisecond)

	if callCount > 1 {
		t.Error("Watch called after unwatch")
	}
}

func TestMissingValues(t *testing.T) {
	m := NewConfigManager()
	m.Load(NewDefaultSource(map[string]interface{}{}))

	if m.GetString("missing") != "" {
		t.Error("Missing string should be empty")
	}
	if m.GetInt("missing") != 0 {
		t.Error("Missing int should be 0")
	}
	if m.GetFloat("missing") != 0 {
		t.Error("Missing float should be 0")
	}
	if m.GetBool("missing") != false {
		t.Error("Missing bool should be false")
	}
	if m.GetDuration("missing") != 0 {
		t.Error("Missing duration should be 0")
	}
}

func TestFlattenMap(t *testing.T) {
	nested := map[string]interface{}{
		"model": map[string]interface{}{
			"path":    "/models/v1",
			"version": "1.0",
		},
		"simple": "value",
	}

	flat := flattenMap(nested, "")

	if flat["model.path"] != "/models/v1" {
		t.Error("Nested key not flattened correctly")
	}
	if flat["simple"] != "value" {
		t.Error("Simple key should be preserved")
	}
}

func TestLoadMLConfig(t *testing.T) {
	m := NewConfigManager()
	m.Load(NewDefaultSource(map[string]interface{}{
		"model.path":             "/models/resnet50",
		"model.version":          "v2",
		"model.batch_size":       32,
		"model.timeout":          "5s",
		"model.device":           "cuda:0",
		"inference.max_concurrency": 4,
		"cache.ttl":              "1h",
		"profiling.enabled":      true,
	}))

	cfg := LoadMLConfig(m)

	if cfg.ModelPath != "/models/resnet50" {
		t.Error("ModelPath incorrect")
	}
	if cfg.BatchSize != 32 {
		t.Error("BatchSize incorrect")
	}
	if cfg.Timeout != 5*time.Second {
		t.Error("Timeout incorrect")
	}
	if cfg.EnableProfiling != true {
		t.Error("EnableProfiling incorrect")
	}
}
`,

  hint1: `Use a map to store all configuration values. When loading from multiple sources, iterate through sources in order and merge values - later sources override earlier ones.`,

  hint2: `For type coercion, use type switches to handle string-to-int, string-to-bool conversions. For environment variables, convert underscores to dots (MODEL_BATCH_SIZE -> model.batch.size).`,

  whyItMatters: `Configuration management is critical for ML services. Model paths, batch sizes, and device selection often differ between development and production. Hot-reloading enables tuning without restarts. Environment variables work well with container orchestration systems.`,

  translations: {
    ru: {
      title: 'Конфигурация ML Сервиса',
      description: `
## Конфигурация ML Сервиса

Создайте систему управления конфигурацией для ML-инференс сервисов с поддержкой множественных источников, валидации и горячей перезагрузки.

### Требования

1. **ConfigManager** - Основной компонент конфигурации:
   - \`NewConfigManager() *ConfigManager\` - Создание менеджера
   - \`Load(sources ...ConfigSource) error\` - Загрузка из источников
   - \`Get(key string) interface{}\` - Получение значения
   - \`GetString/GetInt/GetFloat/GetBool/GetDuration\` - Типизированные геттеры
   - \`Watch(callback func(Config)) func()\` - Отслеживание изменений

2. **ConfigSource** - Источники конфигурации:
   - \`EnvSource\` - Переменные окружения
   - \`FileSource\` - JSON/YAML файлы
   - \`DefaultSource\` - Значения по умолчанию

3. **ML-специфичная конфигурация**:
   - Путь и версия модели
   - Размер батча и тайм-аут
   - Выбор GPU/CPU устройства
   - Лимиты конкурентности инференса
   - Настройки TTL кэша

4. **Возможности**:
   - Приоритет источников (поздние переопределяют ранние)
   - Валидация с обязательными полями
   - Приведение типов (string в int и т.д.)
   - Горячая перезагрузка при изменении файлов

### Пример

\`\`\`go
manager := NewConfigManager()

err := manager.Load(
    NewDefaultSource(map[string]interface{}{
        "model.batch_size": 32,
        "model.timeout":    "5s",
    }),
    NewEnvSource("ML_"),
    NewFileSource("config.yaml"),
)

batchSize := manager.GetInt("model.batch_size")
timeout := manager.GetDuration("model.timeout")

unwatch := manager.Watch(func(cfg Config) {
    log.Println("Config updated")
})
defer unwatch()
\`\`\`
`,
      hint1: 'Используйте map для хранения всех значений конфигурации. При загрузке из нескольких источников, итерируйте по порядку и объединяйте значения - поздние источники переопределяют ранние.',
      hint2: 'Для приведения типов используйте type switches для обработки преобразований string-to-int, string-to-bool. Для переменных окружения преобразуйте подчёркивания в точки (MODEL_BATCH_SIZE -> model.batch.size).',
      whyItMatters: 'Управление конфигурацией критически важно для ML-сервисов. Пути к моделям, размеры батчей и выбор устройств часто различаются между разработкой и продакшеном. Горячая перезагрузка позволяет настраивать без перезапусков. Переменные окружения хорошо работают с системами оркестрации контейнеров.',
    },
    uz: {
      title: 'ML Servis Konfiguratsiyasi',
      description: `
## ML Servis Konfiguratsiyasi

Ko'p manbalarni qo'llab-quvvatlovchi, validatsiya va hot-reloading bilan ML inference servislari uchun konfiguratsiya boshqaruv tizimini yarating.

### Talablar

1. **ConfigManager** - Asosiy konfiguratsiya komponenti:
   - \`NewConfigManager() *ConfigManager\` - Manager yaratish
   - \`Load(sources ...ConfigSource) error\` - Manbalardan yuklash
   - \`Get(key string) interface{}\` - Konfiguratsiya qiymatini olish
   - \`GetString/GetInt/GetFloat/GetBool/GetDuration\` - Tiplangan getterlar
   - \`Watch(callback func(Config)) func()\` - O'zgarishlarni kuzatish

2. **ConfigSource** - Konfiguratsiya manbalari:
   - \`EnvSource\` - Muhit o'zgaruvchilari
   - \`FileSource\` - JSON/YAML fayllar
   - \`DefaultSource\` - Standart qiymatlar

3. **ML-spetsifik konfiguratsiya**:
   - Model yo'li va versiyasi
   - Batch o'lchami va timeout
   - GPU/CPU qurilma tanlash
   - Inference concurrency limitleri
   - Cache TTL sozlamalari

4. **Xususiyatlar**:
   - Manba ustuvorligi (keyingilari oldingilarini bekor qiladi)
   - Majburiy maydonlar bilan validatsiya
   - Tur o'zgartirish (string dan int ga va h.k.)
   - Fayl o'zgarishlarida hot-reload

### Misol

\`\`\`go
manager := NewConfigManager()

err := manager.Load(
    NewDefaultSource(map[string]interface{}{
        "model.batch_size": 32,
        "model.timeout":    "5s",
    }),
    NewEnvSource("ML_"),
    NewFileSource("config.yaml"),
)

batchSize := manager.GetInt("model.batch_size")
timeout := manager.GetDuration("model.timeout")

unwatch := manager.Watch(func(cfg Config) {
    log.Println("Config updated")
})
defer unwatch()
\`\`\`
`,
      hint1: "Barcha konfiguratsiya qiymatlarini saqlash uchun map ishlating. Ko'p manbalardan yuklashda, tartibda iteratsiya qiling va qiymatlarni birlashtiring - keyingi manbalar oldingilarini bekor qiladi.",
      hint2: "Tur o'zgartirish uchun string-to-int, string-to-bool konvertatsiyalarini qayta ishlash uchun type switchlardan foydalaning. Muhit o'zgaruvchilari uchun pastki chiziqlarni nuqtalarga aylantiring (MODEL_BATCH_SIZE -> model.batch.size).",
      whyItMatters: "Konfiguratsiya boshqaruvi ML servislari uchun muhim. Model yo'llari, batch o'lchamlari va qurilma tanlash ko'pincha development va production o'rtasida farq qiladi. Hot-reloading qayta ishga tushirishsiz sozlashni yoqadi. Muhit o'zgaruvchilari konteyner orkestratsiya tizimlari bilan yaxshi ishlaydi.",
    },
  },
};

export default task;
