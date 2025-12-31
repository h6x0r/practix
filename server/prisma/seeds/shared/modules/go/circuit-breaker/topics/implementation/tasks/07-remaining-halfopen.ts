import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-remaining-halfopen',
	title: 'RemainingHalfOpen Method',
	difficulty: 'medium',	tags: ['go', 'circuit-breaker', 'observability'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RemainingHalfOpen** method to report recovery progress.

**Requirements:**
1. Return number of successful requests remaining before circuit closes
2. Only meaningful in **HalfOpen** state: calculate \`halfMax - halfCount\`
3. Return **0** if state is Closed or Open (not in HalfOpen)
4. Protect with read lock for thread safety
5. Handle edge case: if \`halfCount > halfMax\`, return 0 (prevent negative)

**State-Based Logic:**
\`\`\`go
HalfOpen: return halfMax - halfCount  // Remaining trials
Closed:   return 0                     // Not applicable
Open:     return 0                     // Not applicable
\`\`\`

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 5)  // halfMax = 5

// Circuit opens, then transitions to HalfOpen
breaker.RemainingHalfOpen()  // Returns: 5 (need 5 successes)

// First success
breaker.Do(ctx, successFunc)
breaker.RemainingHalfOpen()  // Returns: 4 (need 4 more)

// Second success
breaker.Do(ctx, successFunc)
breaker.RemainingHalfOpen()  // Returns: 3 (need 3 more)

// Continue until halfCount reaches halfMax...
breaker.Do(ctx, successFunc)  // 3rd success
breaker.Do(ctx, successFunc)  // 4th success
breaker.Do(ctx, successFunc)  // 5th success → Closed!

breaker.RemainingHalfOpen()  // Returns: 0 (now Closed)
\`\`\`

**Constraints:**
- Use read lock (not write lock)
- Return 0 for Closed and Open states
- Calculate: \`halfMax - halfCount\` for HalfOpen
- Handle negative values: return 0 if calculation is negative`,
	initialCode: `package circuitx

// TODO: Implement RemainingHalfOpen method
// Return remaining successful requests needed in HalfOpen state
// Return 0 for other states
func (b *Breaker) RemainingHalfOpen() int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package circuitx

func (b *Breaker) RemainingHalfOpen() int {
	b.mu.Lock()                                // guard read-modify logic under lock
	defer b.mu.Unlock()                        // release lock afterwards
	if b.state != HalfOpen || b.halfMax <= 0 { // only meaningful in half-open state
		return 0
	}
	remaining := b.halfMax - b.halfCount // calculate remaining permitted successes
	if remaining < 0 {                   // prevent negative numbers when counters drift
		return 0
	}
	return remaining
}`,
		testCode: `package circuitx

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// RemainingHalfOpen returns 0 for Closed state
	breaker := New(3, 5*time.Second, 5)
	breaker.state = Closed
	if breaker.RemainingHalfOpen() != 0 {
		t.Errorf("expected 0 for Closed, got %d", breaker.RemainingHalfOpen())
	}
}

func Test2(t *testing.T) {
	// RemainingHalfOpen returns 0 for Open state
	breaker := New(3, 5*time.Second, 5)
	breaker.state = Open
	if breaker.RemainingHalfOpen() != 0 {
		t.Errorf("expected 0 for Open, got %d", breaker.RemainingHalfOpen())
	}
}

func Test3(t *testing.T) {
	// RemainingHalfOpen returns correct value in HalfOpen
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 2
	if breaker.RemainingHalfOpen() != 3 {
		t.Errorf("expected 3, got %d", breaker.RemainingHalfOpen())
	}
}

func Test4(t *testing.T) {
	// RemainingHalfOpen returns halfMax when halfCount is 0
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 0
	if breaker.RemainingHalfOpen() != 5 {
		t.Errorf("expected 5, got %d", breaker.RemainingHalfOpen())
	}
}

func Test5(t *testing.T) {
	// RemainingHalfOpen returns 0 when halfCount equals halfMax
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 5
	if breaker.RemainingHalfOpen() != 0 {
		t.Errorf("expected 0 when at halfMax, got %d", breaker.RemainingHalfOpen())
	}
}

func Test6(t *testing.T) {
	// RemainingHalfOpen returns 0 for negative remaining
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 10
	if breaker.RemainingHalfOpen() != 0 {
		t.Errorf("expected 0 for negative remaining, got %d", breaker.RemainingHalfOpen())
	}
}

func Test7(t *testing.T) {
	// RemainingHalfOpen returns 0 for zero halfMax
	breaker := New(3, 5*time.Second, 0)
	breaker.state = HalfOpen
	if breaker.RemainingHalfOpen() != 0 {
		t.Errorf("expected 0 for zero halfMax, got %d", breaker.RemainingHalfOpen())
	}
}

func Test8(t *testing.T) {
	// RemainingHalfOpen is consistent across calls
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 2
	r1 := breaker.RemainingHalfOpen()
	r2 := breaker.RemainingHalfOpen()
	if r1 != r2 {
		t.Error("expected consistent results")
	}
}

func Test9(t *testing.T) {
	// RemainingHalfOpen is safe for concurrent calls
	breaker := New(3, 5*time.Second, 10)
	breaker.state = HalfOpen
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			_ = breaker.RemainingHalfOpen()
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}

func Test10(t *testing.T) {
	// RemainingHalfOpen decreases after success
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 0
	initial := breaker.RemainingHalfOpen()
	breaker.halfCount = 1
	after := breaker.RemainingHalfOpen()
	if after != initial-1 {
		t.Errorf("expected %d, got %d", initial-1, after)
	}
}
`,
		hint1: `Check if state is HalfOpen. If not, return 0.`,
			hint2: `Calculate remaining as halfMax - halfCount, but ensure it does not go negative.`,
			whyItMatters: `RemainingHalfOpen provides visibility into recovery progress, enabling intelligent monitoring and decision-making.

**Why This Matters:**
- **Recovery Tracking:** Monitor how close circuit is to fully closing
- **Progress Visibility:** Show recovery status in dashboards
- **Smart Retries:** Applications can adjust behavior based on remaining trials
- **Debugging:** Understand why circuit isn't closing (failures before reaching halfMax)

**Real-World Example:**

**Dashboard Widget:**
\`\`\`go
type CircuitBreakerStatus struct {
    Service   string
    State     string
    Remaining int
    HealthPct float64
}

func GetCircuitStatus(name string, breaker *circuitx.Breaker) CircuitBreakerStatus {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    status := CircuitBreakerStatus{
        Service:   name,
        State:     state.String(),
        Remaining: remaining,
    }

    if state == circuitx.HalfOpen && remaining > 0 {
        // Calculate recovery percentage
        // If halfMax = 5, remaining = 3, then 40% recovered
        completed := 5 - remaining  // Assuming halfMax is known
        status.HealthPct = float64(completed) / float64(5) * 100
    }

    return status
}

// Dashboard display:
// Service: Payment Gateway
// State: HalfOpen
// Recovery: 3/5 trials completed (60%)
\`\`\`

**Intelligent Client Behavior:**
\`\`\`go
func ProcessPayment(payment Payment) error {
    remaining := paymentBreaker.RemainingHalfOpen()

    if remaining > 0 {
        // Circuit is in HalfOpen, recovering
        log.WithField("remaining_trials", remaining).
            Info("Payment service recovering, trying request")
    }

    return paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return paymentGateway.Charge(payment)
    })
}
\`\`\`

**Metrics & Alerting:**
\`\`\`go
// Prometheus metric
func RecordCircuitBreakerMetrics() {
    services := map[string]*circuitx.Breaker{
        "database": dbBreaker,
        "cache":    cacheBreaker,
        "payment":  paymentBreaker,
    }

    for name, breaker := range services {
        state := breaker.State()
        remaining := breaker.RemainingHalfOpen()

        // State metric (0=closed, 1=open, 0.5=halfopen)
        stateValue := map[circuitx.State]float64{
            circuitx.Closed:   0.0,
            circuitx.Open:     1.0,
            circuitx.HalfOpen: 0.5,
        }[state]

        circuitStateGauge.WithLabelValues(name).Set(stateValue)

        // Recovery progress metric (only in HalfOpen)
        if state == circuitx.HalfOpen {
            recoveryProgress.WithLabelValues(name).Set(float64(remaining))
        }
    }
}

// Alert: Circuit stuck in HalfOpen
// IF circuit_breaker_state{service="payment"} == 0.5
// AND circuit_breaker_remaining{service="payment"} > 0
// FOR 5 minutes
// THEN alert "Payment circuit stuck in recovery"
\`\`\`

**Testing Recovery Logic:**
\`\`\`go
func TestHalfOpenRecovery(t *testing.T) {
    breaker := New(3, 5*time.Second, 5)

    // Open the circuit
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, failingFunc)
    }

    // Transition to HalfOpen
    time.Sleep(5 * time.Second)
    breaker.Do(ctx, successFunc)  // Triggers transition

    // Verify initial remaining
    if remaining := breaker.RemainingHalfOpen(); remaining != 4 {
        t.Errorf("Expected 4 remaining, got %d", remaining)
    }

    // Execute successful requests
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, successFunc)
    }

    // Should have 1 remaining
    if remaining := breaker.RemainingHalfOpen(); remaining != 1 {
        t.Errorf("Expected 1 remaining, got %d", remaining)
    }

    // Final success closes circuit
    breaker.Do(ctx, successFunc)

    // Should return 0 (not in HalfOpen)
    if remaining := breaker.RemainingHalfOpen(); remaining != 0 {
        t.Errorf("Expected 0 (Closed state), got %d", remaining)
    }
}
\`\`\`

**Progressive UI Feedback:**
\`\`\`go
// Show user recovery status
func GetServiceHealthMessage(breaker *circuitx.Breaker) string {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    switch state {
    case circuitx.Closed:
        return "Service healthy"
    case circuitx.Open:
        return "Service temporarily unavailable"
    case circuitx.HalfOpen:
        if remaining > 0 {
            return fmt.Sprintf(
                "Service recovering... %d more successful requests needed",
                remaining,
            )
        }
        return "Service recovering..."
    default:
        return "Unknown status"
    }
}

// API response:
// {
//   "status": "recovering",
//   "message": "Service recovering... 3 more successful requests needed"
// }
\`\`\`

**Advanced Monitoring:**
\`\`\`go
// Track recovery attempts over time
type RecoveryAttempt struct {
    Timestamp     time.Time
    Remaining     int
    PreviousState circuitx.State
    Success       bool
}

var recoveryLog []RecoveryAttempt

func MonitorRecovery(breaker *circuitx.Breaker) {
    ticker := time.NewTicker(1 * time.Second)
    prevRemaining := -1

    for range ticker.C {
        remaining := breaker.RemainingHalfOpen()
        state := breaker.State()

        // Detect change in remaining count
        if remaining != prevRemaining && state == circuitx.HalfOpen {
            success := remaining < prevRemaining

            recoveryLog = append(recoveryLog, RecoveryAttempt{
                Timestamp:     time.Now(),
                Remaining:     remaining,
                PreviousState: state,
                Success:       success,
            })

            if success {
                log.WithField("remaining", remaining).
                    Info("Recovery progress")
            }
        }

        prevRemaining = remaining
    }
}
\`\`\`

**Why Return 0 for Non-HalfOpen States:**

1. **Closed State:** Circuit is healthy, concept of "remaining trials" doesn't apply
2. **Open State:** Circuit is blocked, no trials happening yet
3. **Consistency:** Returning 0 provides clear "not applicable" signal

**Edge Case Handling:**
\`\`\`go
// Why check for negative?
remaining := b.halfMax - b.halfCount

// Scenario: Race condition or manual state manipulation
// halfMax = 5, halfCount = 6 (shouldn't happen, but defensive)
// remaining = 5 - 6 = -1
// Return 0 instead of -1 to prevent confusion
if remaining < 0 {
    return 0
}
\`\`\`

**Key Insight:**
RemainingHalfOpen is an observability method - it doesn't change behavior, but provides critical insight into the circuit breaker's recovery process. In production systems with multiple circuit breakers protecting different services, this visibility is essential for:

1. **Diagnosing stuck circuits** that aren't closing
2. **Understanding recovery patterns** across services
3. **Building confidence** that services are truly healthy before fully reopening
4. **Providing user feedback** during degraded service periods

Without this method, circuit breakers are "black boxes" - you know they're in HalfOpen, but not how close they are to recovery.`,	order: 6,
	translations: {
		ru: {
			title: 'Таймаут полуоткрытого состояния',
			solutionCode: `package circuitx

func (b *Breaker) RemainingHalfOpen() int {
	b.mu.Lock()                                // защищаем логику чтения-изменения под блокировкой
	defer b.mu.Unlock()                        // освобождаем блокировку после
	if b.state != HalfOpen || b.halfMax <= 0 { // имеет смысл только в состоянии half-open
		return 0
	}
	remaining := b.halfMax - b.halfCount // вычисляем оставшиеся разрешённые успехи
	if remaining < 0 {                   // предотвращаем отрицательные числа при дрейфе счётчиков
		return 0
	}
	return remaining
}`,
			description: `Реализуйте метод **RemainingHalfOpen** для отчёта о прогрессе восстановления.

**Требования:**
1. Верните количество успешных запросов до закрытия цепи
2. Имеет смысл только в **HalfOpen**: вычислите \`halfMax - halfCount\`
3. Верните **0** если состояние Closed или Open
4. Защитите read lock для потокобезопасности
5. Обработайте edge case: если \`halfCount > halfMax\`, верните 0

**Логика по состояниям:**
\`\`\`go
HalfOpen: return halfMax - halfCount  // Осталось попыток
Closed:   return 0                     // Не применимо
Open:     return 0                     // Не применимо
\`\`\`

**Ограничения:**
- Используйте read lock (не write lock)
- Возвращайте 0 для Closed и Open
- Вычисляйте: \`halfMax - halfCount\` для HalfOpen
- Обрабатывайте отрицательные значения: верните 0`,
			hint1: `Проверьте, является ли состояние HalfOpen. Если нет, верните 0.`,
			hint2: `Вычислите remaining как halfMax - halfCount, убедитесь что не отрицательное.`,
			whyItMatters: `RemainingHalfOpen обеспечивает видимость прогресса восстановления, позволяя интеллектуальный мониторинг и принятие решений.

**Почему это важно:**
- **Отслеживание восстановления:** Мониторинг того, насколько близко circuit breaker к полному закрытию
- **Видимость прогресса:** Отображение статуса восстановления в дашбордах
- **Умные повторные попытки:** Приложения могут корректировать поведение на основе оставшихся попыток
- **Отладка:** Понимание, почему circuit breaker не закрывается (сбои до достижения halfMax)

**Пример из реальной практики:**

**Виджет дашборда:**
\`\`\`go
type CircuitBreakerStatus struct {
    Service   string
    State     string
    Remaining int
    HealthPct float64
}

func GetCircuitStatus(name string, breaker *circuitx.Breaker) CircuitBreakerStatus {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    status := CircuitBreakerStatus{
        Service:   name,
        State:     state.String(),
        Remaining: remaining,
    }

    if state == circuitx.HalfOpen && remaining > 0 {
        // Вычисляем процент восстановления
        // Если halfMax = 5, remaining = 3, тогда 40% восстановлено
        completed := 5 - remaining  // Предполагается, что halfMax известен
        status.HealthPct = float64(completed) / float64(5) * 100
    }

    return status
}

// Отображение в дашборде:
// Сервис: Платёжный шлюз
// Состояние: HalfOpen
// Восстановление: 3/5 попыток завершено (60%)
\`\`\`

**Интеллектуальное поведение клиента:**
\`\`\`go
func ProcessPayment(payment Payment) error {
    remaining := paymentBreaker.RemainingHalfOpen()

    if remaining > 0 {
        // Circuit breaker в HalfOpen, восстанавливается
        log.WithField("remaining_trials", remaining).
            Info("Платёжный сервис восстанавливается, пробуем запрос")
    }

    return paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return paymentGateway.Charge(payment)
    })
}
\`\`\`

**Метрики и оповещения:**
\`\`\`go
// Метрика Prometheus
func RecordCircuitBreakerMetrics() {
    services := map[string]*circuitx.Breaker{
        "database": dbBreaker,
        "cache":    cacheBreaker,
        "payment":  paymentBreaker,
    }

    for name, breaker := range services {
        state := breaker.State()
        remaining := breaker.RemainingHalfOpen()

        // Метрика состояния (0=закрыт, 1=открыт, 0.5=полуоткрыт)
        stateValue := map[circuitx.State]float64{
            circuitx.Closed:   0.0,
            circuitx.Open:     1.0,
            circuitx.HalfOpen: 0.5,
        }[state]

        circuitStateGauge.WithLabelValues(name).Set(stateValue)

        // Метрика прогресса восстановления (только в HalfOpen)
        if state == circuitx.HalfOpen {
            recoveryProgress.WithLabelValues(name).Set(float64(remaining))
        }
    }
}

// Оповещение: Circuit застрял в HalfOpen
// ЕСЛИ circuit_breaker_state{service="payment"} == 0.5
// И circuit_breaker_remaining{service="payment"} > 0
// В ТЕЧЕНИЕ 5 минут
// ТО алерт "Payment circuit застрял в восстановлении"
\`\`\`

**Тестирование логики восстановления:**
\`\`\`go
func TestHalfOpenRecovery(t *testing.T) {
    breaker := New(3, 5*time.Second, 5)

    // Открываем circuit
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, failingFunc)
    }

    // Переход в HalfOpen
    time.Sleep(5 * time.Second)
    breaker.Do(ctx, successFunc)  // Триггер перехода

    // Проверяем начальное remaining
    if remaining := breaker.RemainingHalfOpen(); remaining != 4 {
        t.Errorf("Ожидалось 4 remaining, получено %d", remaining)
    }

    // Выполняем успешные запросы
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, successFunc)
    }

    // Должно остаться 1
    if remaining := breaker.RemainingHalfOpen(); remaining != 1 {
        t.Errorf("Ожидалось 1 remaining, получено %d", remaining)
    }

    // Финальный успех закрывает circuit
    breaker.Do(ctx, successFunc)

    // Должно вернуть 0 (не в HalfOpen)
    if remaining := breaker.RemainingHalfOpen(); remaining != 0 {
        t.Errorf("Ожидалось 0 (состояние Closed), получено %d", remaining)
    }
}
\`\`\`

**Прогрессивная обратная связь UI:**
\`\`\`go
// Показываем пользователю статус восстановления
func GetServiceHealthMessage(breaker *circuitx.Breaker) string {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    switch state {
    case circuitx.Closed:
        return "Сервис здоров"
    case circuitx.Open:
        return "Сервис временно недоступен"
    case circuitx.HalfOpen:
        if remaining > 0 {
            return fmt.Sprintf(
                "Сервис восстанавливается... необходимо ещё %d успешных запросов",
                remaining,
            )
        }
        return "Сервис восстанавливается..."
    default:
        return "Неизвестный статус"
    }
}

// API ответ:
// {
//   "status": "recovering",
//   "message": "Сервис восстанавливается... необходимо ещё 3 успешных запросов"
// }
\`\`\`

**Продвинутый мониторинг:**
\`\`\`go
// Отслеживание попыток восстановления во времени
type RecoveryAttempt struct {
    Timestamp     time.Time
    Remaining     int
    PreviousState circuitx.State
    Success       bool
}

var recoveryLog []RecoveryAttempt

func MonitorRecovery(breaker *circuitx.Breaker) {
    ticker := time.NewTicker(1 * time.Second)
    prevRemaining := -1

    for range ticker.C {
        remaining := breaker.RemainingHalfOpen()
        state := breaker.State()

        // Обнаружение изменения в счётчике remaining
        if remaining != prevRemaining && state == circuitx.HalfOpen {
            success := remaining < prevRemaining

            recoveryLog = append(recoveryLog, RecoveryAttempt{
                Timestamp:     time.Now(),
                Remaining:     remaining,
                PreviousState: state,
                Success:       success,
            })

            if success {
                log.WithField("remaining", remaining).
                    Info("Прогресс восстановления")
            }
        }

        prevRemaining = remaining
    }
}
\`\`\`

**Почему возвращать 0 для не-HalfOpen состояний:**

1. **Состояние Closed:** Circuit здоров, концепция "оставшихся попыток" не применима
2. **Состояние Open:** Circuit заблокирован, попытки ещё не происходят
3. **Согласованность:** Возврат 0 даёт чёткий сигнал "не применимо"

**Обработка граничных случаев:**
\`\`\`go
// Почему проверяем на отрицательное значение?
remaining := b.halfMax - b.halfCount

// Сценарий: Состояние гонки или ручная манипуляция состоянием
// halfMax = 5, halfCount = 6 (не должно случиться, но защита)
// remaining = 5 - 6 = -1
// Возвращаем 0 вместо -1, чтобы избежать путаницы
if remaining < 0 {
    return 0
}
\`\`\`

**Ключевое понимание:**
RemainingHalfOpen - это метод наблюдаемости - он не изменяет поведение, но предоставляет критически важную информацию о процессе восстановления circuit breaker. В производственных системах с несколькими circuit breaker'ами, защищающими различные сервисы, эта видимость необходима для:

1. **Диагностики застрявших circuit'ов**, которые не закрываются
2. **Понимания паттернов восстановления** между сервисами
3. **Построения уверенности**, что сервисы действительно здоровы перед полным открытием
4. **Предоставления обратной связи пользователям** во время периодов деградации сервиса

Без этого метода circuit breaker'ы - это "чёрные ящики" - вы знаете, что они в HalfOpen, но не знаете, насколько близко они к восстановлению.`
		},
		uz: {
			title: `Yarim ochiq holat timeoutи`,
			solutionCode: `package circuitx

func (b *Breaker) RemainingHalfOpen() int {
	b.mu.Lock()                                // o'qish-o'zgartirish mantiqini qulf ostida himoya qilamiz
	defer b.mu.Unlock()                        // keyin qulfni bo'shatamiz
	if b.state != HalfOpen || b.halfMax <= 0 { // faqat half-open holatida ma'noli
		return 0
	}
	remaining := b.halfMax - b.halfCount // qolgan ruxsat etilgan muvaffaqiyatlarni hisoblaymiz
	if remaining < 0 {                   // hisoblagichlar surish paytida salbiy raqamlarni oldini olamiz
		return 0
	}
	return remaining
}`,
			description: `Tiklash jarayoni haqida hisobot berish uchun **RemainingHalfOpen** metodini amalga oshiring.

**Talablar:**
1. Zanjir yopilishidan oldin qolgan muvaffaqiyatli so'rovlar sonini qaytaring
2. Faqat **HalfOpen** holatida ma'noli: \`halfMax - halfCount\` ni hisoblang
3. Holat Closed yoki Open bo'lsa **0** qaytaring
4. Thread xavfsizligi uchun read lock bilan himoyalang
5. Edge case: agar \`halfCount > halfMax\` bo'lsa, 0 qaytaring (salbiyni oldini olish)

**Holatga asoslangan mantiq:**
\`\`\`go
HalfOpen: return halfMax - halfCount  // Qolgan sinovlar
Closed:   return 0                     // Qo'llanilmaydi
Open:     return 0                     // Qo'llanilmaydi
\`\`\`

**Cheklovlar:**
- Read lock ishlating (write lock emas)
- Closed va Open uchun 0 qaytaring
- HalfOpen uchun: \`halfMax - halfCount\` ni hisoblang
- Salbiy qiymatlarni qayta ishlang: salbiy bo'lsa 0 qaytaring`,
			hint1: `Holat HalfOpen ekanligini tekshiring. Agar yo'q bo'lsa, 0 qaytaring.`,
			hint2: `remaining ni halfMax - halfCount sifatida hisoblang, salbiy bo'lmasligini ta'minlang.`,
			whyItMatters: `RemainingHalfOpen tiklash jarayoniga ko'rinishni ta'minlaydi, aqlli monitoring va qaror qabul qilishni yoqadi.

**Nima uchun bu muhim:**
- **Tiklanishni kuzatish:** Circuit breakerning to'liq yopilishiga qanchalik yaqinligini monitoring qilish
- **Jarayon ko'rinishi:** Dashboardlarda tiklash holatini ko'rsatish
- **Aqlli qayta urinishlar:** Ilovalar qolgan urinishlarga asoslanib xatti-harakatni o'zgartirishi mumkin
- **Debugging:** Circuit breaker nima uchun yopilmayotganini tushunish (halfMax ga yetishdan oldin xatolar)

**Amaliy hayotdan misol:**

**Dashboard vijet:**
\`\`\`go
type CircuitBreakerStatus struct {
    Service   string
    State     string
    Remaining int
    HealthPct float64
}

func GetCircuitStatus(name string, breaker *circuitx.Breaker) CircuitBreakerStatus {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    status := CircuitBreakerStatus{
        Service:   name,
        State:     state.String(),
        Remaining: remaining,
    }

    if state == circuitx.HalfOpen && remaining > 0 {
        // Tiklash foizini hisoblaymiz
        // Agar halfMax = 5, remaining = 3, unda 40% tiklandi
        completed := 5 - remaining  // halfMax ma'lum deb faraz qilinadi
        status.HealthPct = float64(completed) / float64(5) * 100
    }

    return status
}

// Dashboard ko'rinishi:
// Xizmat: To'lov shlyuzi
// Holat: HalfOpen
// Tiklash: 3/5 urinish tugallandi (60%)
\`\`\`

**Aqlli mijoz xatti-harakati:**
\`\`\`go
func ProcessPayment(payment Payment) error {
    remaining := paymentBreaker.RemainingHalfOpen()

    if remaining > 0 {
        // Circuit breaker HalfOpen da, tiklanmoqda
        log.WithField("remaining_trials", remaining).
            Info("To'lov xizmati tiklanmoqda, so'rovga urinib ko'ramiz")
    }

    return paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return paymentGateway.Charge(payment)
    })
}
\`\`\`

**Metrikalar va ogohlantirishlar:**
\`\`\`go
// Prometheus metrikasi
func RecordCircuitBreakerMetrics() {
    services := map[string]*circuitx.Breaker{
        "database": dbBreaker,
        "cache":    cacheBreaker,
        "payment":  paymentBreaker,
    }

    for name, breaker := range services {
        state := breaker.State()
        remaining := breaker.RemainingHalfOpen()

        // Holat metrikasi (0=yopiq, 1=ochiq, 0.5=yarim ochiq)
        stateValue := map[circuitx.State]float64{
            circuitx.Closed:   0.0,
            circuitx.Open:     1.0,
            circuitx.HalfOpen: 0.5,
        }[state]

        circuitStateGauge.WithLabelValues(name).Set(stateValue)

        // Tiklash jarayoni metrikasi (faqat HalfOpen da)
        if state == circuitx.HalfOpen {
            recoveryProgress.WithLabelValues(name).Set(float64(remaining))
        }
    }
}

// Ogohlantirish: Circuit HalfOpen da qotib qoldi
// AGAR circuit_breaker_state{service="payment"} == 0.5
// VA circuit_breaker_remaining{service="payment"} > 0
// 5 daqiqa davomida
// U HOLDA ogohlantirish "Payment circuit tiklanishda qotib qoldi"
\`\`\`

**Tiklash mantiqini testlash:**
\`\`\`go
func TestHalfOpenRecovery(t *testing.T) {
    breaker := New(3, 5*time.Second, 5)

    // Circuitni ochamiz
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, failingFunc)
    }

    // HalfOpen ga o'tish
    time.Sleep(5 * time.Second)
    breaker.Do(ctx, successFunc)  // O'tish triggerini ishga tushirish

    // Boshlang'ich remaining ni tekshiramiz
    if remaining := breaker.RemainingHalfOpen(); remaining != 4 {
        t.Errorf("4 remaining kutilgan, %d olindi", remaining)
    }

    // Muvaffaqiyatli so'rovlarni bajaramiz
    for i := 0; i < 3; i++ {
        breaker.Do(ctx, successFunc)
    }

    // 1 ta qolishi kerak
    if remaining := breaker.RemainingHalfOpen(); remaining != 1 {
        t.Errorf("1 remaining kutilgan, %d olindi", remaining)
    }

    // Oxirgi muvaffaqiyat circuitni yopadi
    breaker.Do(ctx, successFunc)

    // 0 qaytarishi kerak (HalfOpen da emas)
    if remaining := breaker.RemainingHalfOpen(); remaining != 0 {
        t.Errorf("0 kutilgan (Closed holat), %d olindi", remaining)
    }
}
\`\`\`

**Progressiv UI qaytarilish:**
\`\`\`go
// Foydalanuvchiga tiklash holatini ko'rsatamiz
func GetServiceHealthMessage(breaker *circuitx.Breaker) string {
    state := breaker.State()
    remaining := breaker.RemainingHalfOpen()

    switch state {
    case circuitx.Closed:
        return "Xizmat sog'lom"
    case circuitx.Open:
        return "Xizmat vaqtinchalik mavjud emas"
    case circuitx.HalfOpen:
        if remaining > 0 {
            return fmt.Sprintf(
                "Xizmat tiklanmoqda... yana %d muvaffaqiyatli so'rov kerak",
                remaining,
            )
        }
        return "Xizmat tiklanmoqda..."
    default:
        return "Noma'lum holat"
    }
}

// API javobi:
// {
//   "status": "recovering",
//   "message": "Xizmat tiklanmoqda... yana 3 muvaffaqiyatli so'rov kerak"
// }
\`\`\`

**Rivojlangan monitoring:**
\`\`\`go
// Tiklash urinishlarini vaqt bo'yicha kuzatish
type RecoveryAttempt struct {
    Timestamp     time.Time
    Remaining     int
    PreviousState circuitx.State
    Success       bool
}

var recoveryLog []RecoveryAttempt

func MonitorRecovery(breaker *circuitx.Breaker) {
    ticker := time.NewTicker(1 * time.Second)
    prevRemaining := -1

    for range ticker.C {
        remaining := breaker.RemainingHalfOpen()
        state := breaker.State()

        // Remaining hisoblagichidagi o'zgarishni aniqlash
        if remaining != prevRemaining && state == circuitx.HalfOpen {
            success := remaining < prevRemaining

            recoveryLog = append(recoveryLog, RecoveryAttempt{
                Timestamp:     time.Now(),
                Remaining:     remaining,
                PreviousState: state,
                Success:       success,
            })

            if success {
                log.WithField("remaining", remaining).
                    Info("Tiklash jarayoni")
            }
        }

        prevRemaining = remaining
    }
}
\`\`\`

**Nima uchun HalfOpen bo'lmagan holatlar uchun 0 qaytarish:**

1. **Closed holat:** Circuit sog'lom, "qolgan urinishlar" kontseptsiyasi qo'llanilmaydi
2. **Open holat:** Circuit bloklangan, urinishlar hali sodir bo'lmayapti
3. **Izchillik:** 0 ni qaytarish aniq "qo'llanilmaydi" signalini beradi

**Chegara holatlarini qayta ishlash:**
\`\`\`go
// Nima uchun salbiy qiymatni tekshiramiz?
remaining := b.halfMax - b.halfCount

// Stsenariy: Poyga holati yoki qo'lda holat boshqaruvi
// halfMax = 5, halfCount = 6 (sodir bo'lmasligi kerak, ammo himoya)
// remaining = 5 - 6 = -1
// Chalkashlikni oldini olish uchun -1 o'rniga 0 qaytaramiz
if remaining < 0 {
    return 0
}
\`\`\`

**Asosiy tushuncha:**
RemainingHalfOpen - bu kuzatuv usuli - u xatti-harakatni o'zgartirmaydi, lekin circuit breakerning tiklash jarayoni haqida juda muhim ma'lumot beradi. Turli xizmatlarni himoya qiluvchi bir nechta circuit breakerlarga ega ishlab chiqarish tizimlarida bu ko'rinish quyidagilar uchun zarur:

1. **Yopilmayotgan qotib qolgan circuitlarni diagnostika qilish**
2. **Xizmatlar o'rtasida tiklash patternlarini tushunish**
3. **Ishonch qurish** xizmatlar to'liq ochilishdan oldin haqiqatan sog'lom ekanligiga
4. **Foydalanuvchilarga qaytarilish berish** xizmat degradatsiya davrlari mobaynida

Bu metodsiz circuit breakerlar "qora qutilar" - ular HalfOpen ekanligini bilasiz, lekin tiklanishga qanchalik yaqin ekanligini bilmaysiz.`
		}
	}
};

export default task;
