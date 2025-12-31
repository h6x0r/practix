import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-alerting-system',
  title: 'ML Alerting System',
  difficulty: 'medium',
  tags: ['go', 'ml', 'monitoring', 'alerting', 'thresholds'],
  estimatedTime: '25m',
  isPremium: false,
  order: 4,

  description: `
## ML Alerting System

Build an alerting system for ML inference that triggers notifications based on metric thresholds.

### Requirements

1. **AlertManager** - Main alerting component:
   - \`NewAlertManager(config AlertConfig)\` - Create with configuration
   - \`AddRule(rule AlertRule)\` - Register alert rule
   - \`Check(metrics map[string]float64) []Alert\` - Check metrics against rules
   - \`Subscribe(handler AlertHandler)\` - Register alert handler

2. **AlertRule** - Rule definition:
   - \`Name string\` - Rule identifier
   - \`Metric string\` - Metric name to monitor
   - \`Condition Condition\` - Threshold condition (GT, LT, EQ, etc.)
   - \`Threshold float64\` - Threshold value
   - \`Duration time.Duration\` - How long condition must be true
   - \`Severity Severity\` - Alert severity level

3. **Alert States**:
   - \`Pending\` - Condition met but duration not reached
   - \`Firing\` - Alert active
   - \`Resolved\` - Condition no longer met

4. **Built-in Rules** (examples):
   - High latency: P99 > 500ms for 5 minutes
   - Low throughput: RPS < 10 for 2 minutes
   - High error rate: errors > 5% for 1 minute
   - Model drift: drift_score > 0.1

### Example

\`\`\`go
manager := NewAlertManager(AlertConfig{
    CheckInterval: time.Second * 30,
})

manager.AddRule(AlertRule{
    Name:      "high_latency",
    Metric:    "p99_latency_ms",
    Condition: ConditionGT,
    Threshold: 500,
    Duration:  5 * time.Minute,
    Severity:  SeverityWarning,
})

manager.Subscribe(func(alert Alert) {
    log.Printf("[%s] %s: %s", alert.Severity, alert.Name, alert.Message)
})

alerts := manager.Check(map[string]float64{
    "p99_latency_ms": 600,
})
\`\`\`
`,

  initialCode: `package alerting

import (
	"sync"
	"time"
)

type Severity string

)

type Condition string

)

type AlertState string

)

type AlertRule struct {
	Name      string
	Metric    string
	Condition Condition
	Threshold float64
	Duration  time.Duration
	Severity  Severity
}

type Alert struct {
	Name      string
	Metric    string
	Value     float64
	Threshold float64
	State     AlertState
	Severity  Severity
	Message   string
	FiredAt   time.Time
	ResolvedAt time.Time
}

type AlertHandler func(alert Alert)

type AlertConfig struct {
	CheckInterval time.Duration
}

type AlertManager struct {
}

func NewAlertManager(config AlertConfig) *AlertManager {
	return nil
}

func (m *AlertManager) AddRule(rule AlertRule) {
}

func (m *AlertManager) Check(metrics map[string]float64) []Alert {
	return nil
}

func (m *AlertManager) Subscribe(handler AlertHandler) {
}

func evaluateCondition(value float64, condition Condition, threshold float64) bool {
	return false
}`,

  solutionCode: `package alerting

import (
	"fmt"
	"sync"
	"time"
)

// Severity levels for alerts
type Severity string

const (
	SeverityInfo     Severity = "info"
	SeverityWarning  Severity = "warning"
	SeverityCritical Severity = "critical"
)

// Condition types for alert rules
type Condition string

const (
	ConditionGT Condition = "gt"
	ConditionLT Condition = "lt"
	ConditionGE Condition = "ge"
	ConditionLE Condition = "le"
	ConditionEQ Condition = "eq"
)

// AlertState represents current alert state
type AlertState string

const (
	StatePending  AlertState = "pending"
	StateFiring   AlertState = "firing"
	StateResolved AlertState = "resolved"
)

// AlertRule defines when to trigger an alert
type AlertRule struct {
	Name      string
	Metric    string
	Condition Condition
	Threshold float64
	Duration  time.Duration
	Severity  Severity
}

// Alert represents a triggered alert
type Alert struct {
	Name       string
	Metric     string
	Value      float64
	Threshold  float64
	State      AlertState
	Severity   Severity
	Message    string
	FiredAt    time.Time
	ResolvedAt time.Time
}

// AlertHandler handles triggered alerts
type AlertHandler func(alert Alert)

// AlertConfig configures the alert manager
type AlertConfig struct {
	CheckInterval time.Duration
}

// alertState tracks pending/firing state for a rule
type alertState struct {
	pendingSince time.Time
	firing       bool
	firedAt      time.Time
}

// AlertManager manages alert rules and notifications
type AlertManager struct {
	config   AlertConfig
	rules    []AlertRule
	states   map[string]*alertState
	handlers []AlertHandler
	mu       sync.RWMutex
}

// NewAlertManager creates a new alert manager
func NewAlertManager(config AlertConfig) *AlertManager {
	if config.CheckInterval == 0 {
		config.CheckInterval = 30 * time.Second
	}
	return &AlertManager{
		config:   config,
		rules:    make([]AlertRule, 0),
		states:   make(map[string]*alertState),
		handlers: make([]AlertHandler, 0),
	}
}

// AddRule registers a new alert rule
func (m *AlertManager) AddRule(rule AlertRule) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rules = append(m.rules, rule)
	m.states[rule.Name] = &alertState{}
}

// Check evaluates all rules against current metrics
func (m *AlertManager) Check(metrics map[string]float64) []Alert {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	alerts := make([]Alert, 0)

	for _, rule := range m.rules {
		value, exists := metrics[rule.Metric]
		state := m.states[rule.Name]

		if !exists {
			// Metric not present, resolve if firing
			if state.firing {
				alert := m.createAlert(rule, 0, StateResolved, now)
				alerts = append(alerts, alert)
				m.notifyHandlers(alert)
				state.firing = false
				state.pendingSince = time.Time{}
			}
			continue
		}

		conditionMet := evaluateCondition(value, rule.Condition, rule.Threshold)

		if conditionMet {
			if state.pendingSince.IsZero() {
				// Start pending
				state.pendingSince = now
			}

			pendingDuration := now.Sub(state.pendingSince)

			if !state.firing && pendingDuration >= rule.Duration {
				// Transition to firing
				state.firing = true
				state.firedAt = now
				alert := m.createAlert(rule, value, StateFiring, now)
				alerts = append(alerts, alert)
				m.notifyHandlers(alert)
			} else if !state.firing {
				// Still pending
				alert := m.createAlert(rule, value, StatePending, now)
				alerts = append(alerts, alert)
			} else {
				// Already firing, include in results
				alert := m.createAlert(rule, value, StateFiring, state.firedAt)
				alerts = append(alerts, alert)
			}
		} else {
			// Condition not met
			if state.firing {
				// Resolve alert
				alert := m.createAlert(rule, value, StateResolved, now)
				alert.ResolvedAt = now
				alerts = append(alerts, alert)
				m.notifyHandlers(alert)
			}
			state.firing = false
			state.pendingSince = time.Time{}
		}
	}

	return alerts
}

func (m *AlertManager) createAlert(rule AlertRule, value float64, state AlertState, firedAt time.Time) Alert {
	return Alert{
		Name:      rule.Name,
		Metric:    rule.Metric,
		Value:     value,
		Threshold: rule.Threshold,
		State:     state,
		Severity:  rule.Severity,
		Message:   fmt.Sprintf("%s: %s %s %.2f (current: %.2f)", rule.Name, rule.Metric, rule.Condition, rule.Threshold, value),
		FiredAt:   firedAt,
	}
}

// Subscribe registers an alert handler
func (m *AlertManager) Subscribe(handler AlertHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers = append(m.handlers, handler)
}

func (m *AlertManager) notifyHandlers(alert Alert) {
	for _, handler := range m.handlers {
		handler(alert)
	}
}

// evaluateCondition checks if a value meets the condition
func evaluateCondition(value float64, condition Condition, threshold float64) bool {
	switch condition {
	case ConditionGT:
		return value > threshold
	case ConditionLT:
		return value < threshold
	case ConditionGE:
		return value >= threshold
	case ConditionLE:
		return value <= threshold
	case ConditionEQ:
		return value == threshold
	default:
		return false
	}
}

// GetActiveAlerts returns all currently firing alerts
func (m *AlertManager) GetActiveAlerts() []Alert {
	m.mu.RLock()
	defer m.mu.RUnlock()

	alerts := make([]Alert, 0)
	for _, rule := range m.rules {
		state := m.states[rule.Name]
		if state.firing {
			alerts = append(alerts, m.createAlert(rule, 0, StateFiring, state.firedAt))
		}
	}
	return alerts
}
`,

  testCode: `package alerting

import (
	"testing"
	"time"
)

func TestNewAlertManager(t *testing.T) {
	manager := NewAlertManager(AlertConfig{
		CheckInterval: time.Second * 30,
	})

	if manager == nil {
		t.Fatal("Expected non-nil manager")
	}
}

func TestAddRule(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "test_rule",
		Metric:    "latency",
		Condition: ConditionGT,
		Threshold: 100,
		Duration:  0,
		Severity:  SeverityWarning,
	})

	if len(manager.rules) != 1 {
		t.Errorf("Expected 1 rule, got %d", len(manager.rules))
	}
}

func TestCheckFiring(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "high_latency",
		Metric:    "latency",
		Condition: ConditionGT,
		Threshold: 100,
		Duration:  0, // Immediate
		Severity:  SeverityWarning,
	})

	alerts := manager.Check(map[string]float64{
		"latency": 150,
	})

	if len(alerts) != 1 {
		t.Fatalf("Expected 1 alert, got %d", len(alerts))
	}

	if alerts[0].State != StateFiring {
		t.Errorf("Expected firing state, got %s", alerts[0].State)
	}

	if alerts[0].Value != 150 {
		t.Errorf("Expected value 150, got %f", alerts[0].Value)
	}
}

func TestCheckResolved(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "high_latency",
		Metric:    "latency",
		Condition: ConditionGT,
		Threshold: 100,
		Duration:  0,
		Severity:  SeverityWarning,
	})

	// First check - should fire
	manager.Check(map[string]float64{"latency": 150})

	// Second check - should resolve
	alerts := manager.Check(map[string]float64{"latency": 50})

	foundResolved := false
	for _, a := range alerts {
		if a.State == StateResolved {
			foundResolved = true
			break
		}
	}

	if !foundResolved {
		t.Error("Expected resolved alert")
	}
}

func TestCheckPending(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "high_latency",
		Metric:    "latency",
		Condition: ConditionGT,
		Threshold: 100,
		Duration:  time.Hour, // Long duration
		Severity:  SeverityWarning,
	})

	alerts := manager.Check(map[string]float64{
		"latency": 150,
	})

	if len(alerts) != 1 {
		t.Fatalf("Expected 1 alert, got %d", len(alerts))
	}

	if alerts[0].State != StatePending {
		t.Errorf("Expected pending state, got %s", alerts[0].State)
	}
}

func TestConditions(t *testing.T) {
	tests := []struct {
		condition Condition
		value     float64
		threshold float64
		expected  bool
	}{
		{ConditionGT, 10, 5, true},
		{ConditionGT, 5, 10, false},
		{ConditionLT, 5, 10, true},
		{ConditionLT, 10, 5, false},
		{ConditionGE, 10, 10, true},
		{ConditionGE, 9, 10, false},
		{ConditionLE, 10, 10, true},
		{ConditionLE, 11, 10, false},
		{ConditionEQ, 10, 10, true},
		{ConditionEQ, 11, 10, false},
	}

	for _, tt := range tests {
		result := evaluateCondition(tt.value, tt.condition, tt.threshold)
		if result != tt.expected {
			t.Errorf("evaluateCondition(%f, %s, %f) = %v, want %v",
				tt.value, tt.condition, tt.threshold, result, tt.expected)
		}
	}
}

func TestSubscribe(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	var receivedAlert *Alert
	manager.Subscribe(func(alert Alert) {
		receivedAlert = &alert
	})

	manager.AddRule(AlertRule{
		Name:      "test",
		Metric:    "value",
		Condition: ConditionGT,
		Threshold: 0,
		Duration:  0,
		Severity:  SeverityCritical,
	})

	manager.Check(map[string]float64{"value": 10})

	if receivedAlert == nil {
		t.Fatal("Handler not called")
	}

	if receivedAlert.Severity != SeverityCritical {
		t.Errorf("Expected critical severity, got %s", receivedAlert.Severity)
	}
}

func TestMultipleRules(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "rule1",
		Metric:    "metric1",
		Condition: ConditionGT,
		Threshold: 10,
		Duration:  0,
		Severity:  SeverityWarning,
	})

	manager.AddRule(AlertRule{
		Name:      "rule2",
		Metric:    "metric2",
		Condition: ConditionLT,
		Threshold: 5,
		Duration:  0,
		Severity:  SeverityCritical,
	})

	alerts := manager.Check(map[string]float64{
		"metric1": 15, // Should fire
		"metric2": 3,  // Should fire
	})

	if len(alerts) != 2 {
		t.Errorf("Expected 2 alerts, got %d", len(alerts))
	}
}

func TestGetActiveAlerts(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "active_rule",
		Metric:    "latency",
		Condition: ConditionGT,
		Threshold: 100,
		Duration:  0,
		Severity:  SeverityWarning,
	})

	// Fire the alert
	manager.Check(map[string]float64{"latency": 150})

	active := manager.GetActiveAlerts()
	if len(active) != 1 {
		t.Errorf("Expected 1 active alert, got %d", len(active))
	}

	// Resolve the alert
	manager.Check(map[string]float64{"latency": 50})

	active = manager.GetActiveAlerts()
	if len(active) != 0 {
		t.Errorf("Expected 0 active alerts, got %d", len(active))
	}
}

func TestMissingMetric(t *testing.T) {
	manager := NewAlertManager(AlertConfig{})

	manager.AddRule(AlertRule{
		Name:      "test_rule",
		Metric:    "cpu_usage",
		Condition: ConditionGT,
		Threshold: 80,
		Duration:  0,
		Severity:  SeverityWarning,
	})

	// Fire the alert first
	manager.Check(map[string]float64{"cpu_usage": 90})

	// Then check without the metric - should resolve
	alerts := manager.Check(map[string]float64{})

	foundResolved := false
	for _, a := range alerts {
		if a.Name == "test_rule" && a.State == StateResolved {
			foundResolved = true
			break
		}
	}

	if !foundResolved {
		t.Error("Expected alert to be resolved when metric is missing")
	}
}
`,

  hint1: `Track pending state with timestamps. When condition is first met, record the time. On subsequent checks, compare elapsed time against rule duration.`,

  hint2: `Use a map to track state per rule. Each rule needs its own pending timestamp and firing flag to handle transitions correctly.`,

  whyItMatters: `Alerting is essential for ML operations. Detecting high latency, error rates, or model drift early prevents user impact. Duration-based alerting reduces alert fatigue by filtering transient spikes.`,

  translations: {
    ru: {
      title: 'Система Алертинга для ML',
      description: `
## Система Алертинга для ML

Создайте систему алертинга для ML-инференса, которая срабатывает по пороговым значениям метрик.

### Требования

1. **AlertManager** - Основной компонент алертинга:
   - \`NewAlertManager(config AlertConfig)\` - Создание с конфигурацией
   - \`AddRule(rule AlertRule)\` - Регистрация правила
   - \`Check(metrics map[string]float64) []Alert\` - Проверка метрик
   - \`Subscribe(handler AlertHandler)\` - Регистрация обработчика

2. **AlertRule** - Определение правила:
   - \`Name string\` - Идентификатор правила
   - \`Metric string\` - Имя метрики для мониторинга
   - \`Condition Condition\` - Условие порога (GT, LT, EQ и т.д.)
   - \`Threshold float64\` - Пороговое значение
   - \`Duration time.Duration\` - Время удержания условия
   - \`Severity Severity\` - Уровень критичности

3. **Состояния алерта**:
   - \`Pending\` - Условие выполнено, но duration не достигнут
   - \`Firing\` - Алерт активен
   - \`Resolved\` - Условие больше не выполняется

4. **Встроенные правила** (примеры):
   - Высокая латентность: P99 > 500ms в течение 5 минут
   - Низкая пропускная способность: RPS < 10 в течение 2 минут
   - Высокий уровень ошибок: errors > 5% в течение 1 минуты
   - Дрейф модели: drift_score > 0.1

### Пример

\`\`\`go
manager := NewAlertManager(AlertConfig{
    CheckInterval: time.Second * 30,
})

manager.AddRule(AlertRule{
    Name:      "high_latency",
    Metric:    "p99_latency_ms",
    Condition: ConditionGT,
    Threshold: 500,
    Duration:  5 * time.Minute,
    Severity:  SeverityWarning,
})

manager.Subscribe(func(alert Alert) {
    log.Printf("[%s] %s: %s", alert.Severity, alert.Name, alert.Message)
})

alerts := manager.Check(map[string]float64{
    "p99_latency_ms": 600,
})
\`\`\`
`,
      hint1: 'Отслеживайте pending-состояние с временными метками. При первом выполнении условия запишите время. При последующих проверках сравнивайте прошедшее время с duration правила.',
      hint2: 'Используйте map для отслеживания состояния по каждому правилу. Каждому правилу нужен свой pending timestamp и флаг firing для корректной обработки переходов.',
      whyItMatters: 'Алертинг необходим для ML-операций. Раннее обнаружение высокой латентности, ошибок или дрейфа модели предотвращает влияние на пользователей. Алертинг на основе duration снижает усталость от алертов, фильтруя кратковременные всплески.',
    },
    uz: {
      title: 'ML Alerting Tizimi',
      description: `
## ML Alerting Tizimi

Metrika chegaralari asosida bildirishnomalarni ishga tushiruvchi ML inference uchun alerting tizimini yarating.

### Talablar

1. **AlertManager** - Asosiy alerting komponenti:
   - \`NewAlertManager(config AlertConfig)\` - Konfiguratsiya bilan yaratish
   - \`AddRule(rule AlertRule)\` - Alert qoidasini ro'yxatdan o'tkazish
   - \`Check(metrics map[string]float64) []Alert\` - Qoidalarga qarshi metrikalarni tekshirish
   - \`Subscribe(handler AlertHandler)\` - Alert handlerni ro'yxatdan o'tkazish

2. **AlertRule** - Qoida ta'rifi:
   - \`Name string\` - Qoida identifikatori
   - \`Metric string\` - Kuzatiladigan metrika nomi
   - \`Condition Condition\` - Chegara sharti (GT, LT, EQ va h.k.)
   - \`Threshold float64\` - Chegara qiymati
   - \`Duration time.Duration\` - Shart qancha vaqt to'g'ri bo'lishi kerak
   - \`Severity Severity\` - Alert jiddiyligi darajasi

3. **Alert holatlari**:
   - \`Pending\` - Shart bajarildi, lekin duration ga erishilmadi
   - \`Firing\` - Alert faol
   - \`Resolved\` - Shart endi bajarilmaydi

4. **O'rnatilgan qoidalar** (misollar):
   - Yuqori latentlik: P99 > 500ms 5 daqiqa davomida
   - Past o'tkazuvchanlik: RPS < 10 2 daqiqa davomida
   - Yuqori xato darajasi: errors > 5% 1 daqiqa davomida
   - Model drift: drift_score > 0.1

### Misol

\`\`\`go
manager := NewAlertManager(AlertConfig{
    CheckInterval: time.Second * 30,
})

manager.AddRule(AlertRule{
    Name:      "high_latency",
    Metric:    "p99_latency_ms",
    Condition: ConditionGT,
    Threshold: 500,
    Duration:  5 * time.Minute,
    Severity:  SeverityWarning,
})

manager.Subscribe(func(alert Alert) {
    log.Printf("[%s] %s: %s", alert.Severity, alert.Name, alert.Message)
})

alerts := manager.Check(map[string]float64{
    "p99_latency_ms": 600,
})
\`\`\`
`,
      hint1: "Pending holatini vaqt belgilari bilan kuzatib boring. Shart birinchi marta bajarilganda vaqtni yozib oling. Keyingi tekshiruvlarda o'tgan vaqtni qoida duration bilan solishtiring.",
      hint2: "Har bir qoida uchun holatni kuzatish uchun map ishlating. Har bir qoidaga o'zining pending timestamp va firing bayrog'i kerak o'tishlarni to'g'ri boshqarish uchun.",
      whyItMatters: "Alerting ML operatsiyalari uchun muhim. Yuqori latentlik, xatolar yoki model drift ni erta aniqlash foydalanuvchilarga ta'sirni oldini oladi. Duration asosidagi alerting vaqtinchalik o'sishlarni filtrlash orqali alert charchashini kamaytiradi.",
    },
  },
};

export default task;
