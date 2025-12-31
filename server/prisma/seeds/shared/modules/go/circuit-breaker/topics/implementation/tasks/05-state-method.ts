import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-state',
	title: 'State Method',
	difficulty: 'medium',	tags: ['go', 'circuit-breaker', 'concurrency'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **State** method to safely read current circuit breaker state.

**Requirements:**
1. Return the current state of the circuit breaker (Closed, Open, or HalfOpen)
2. Protect read access with mutex (use RLock/RUnlock for read operations)
3. Method should be safe for concurrent access

**Why Read Lock?**
\`\`\`go
// Read Lock (RLock) - Multiple readers allowed simultaneously
// Write Lock (Lock) - Exclusive access, blocks all other operations

b.mu.RLock()     // Allows concurrent readers
defer b.mu.RUnlock()
return b.state
\`\`\`

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)

state := breaker.State()  // Returns: Closed

// After failures...
for i := 0; i < 3; i++ {
    breaker.Do(ctx, failingFunc)
}

state = breaker.State()  // Returns: Open

// Monitor in goroutine
go func() {
    ticker := time.NewTicker(1 * time.Second)
    for range ticker.C {
        state := breaker.State()
        log.Printf("Circuit breaker state: %v", state)
    }
}()
\`\`\`

**Constraints:**
- Must use mutex for thread safety
- Use RLock/RUnlock for read operations (not Lock/Unlock)
- Do not modify any state, only read`,
	initialCode: `package circuitx

// TODO: Implement State method with read lock protection
func (b *Breaker) State() State {
	// TODO: Implement
}`,
	solutionCode: `package circuitx

func (b *Breaker) State() State {
	b.mu.Lock()         // synchronize access to current state
	defer b.mu.Unlock() // release lock after reading
	return b.state      // return snapshot of breaker state
}`,
		testCode: `package circuitx

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// State returns Closed for new breaker
	breaker := New(3, 5*time.Second, 2)
	if breaker.State() != Closed {
		t.Errorf("expected Closed, got %v", breaker.State())
	}
}

func Test2(t *testing.T) {
	// State returns Open when state is Open
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	if breaker.State() != Open {
		t.Errorf("expected Open, got %v", breaker.State())
	}
}

func Test3(t *testing.T) {
	// State returns HalfOpen when state is HalfOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	if breaker.State() != HalfOpen {
		t.Errorf("expected HalfOpen, got %v", breaker.State())
	}
}

func Test4(t *testing.T) {
	// State is consistent across calls
	breaker := New(3, 5*time.Second, 2)
	s1 := breaker.State()
	s2 := breaker.State()
	if s1 != s2 {
		t.Error("expected consistent state")
	}
}

func Test5(t *testing.T) {
	// State returns Closed type
	breaker := New(3, 5*time.Second, 2)
	var s State = breaker.State()
	if s != Closed {
		t.Error("expected State type")
	}
}

func Test6(t *testing.T) {
	// State reflects state change
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	if breaker.State() != Open {
		t.Error("expected state change to be reflected")
	}
	breaker.state = Closed
	if breaker.State() != Closed {
		t.Error("expected state change to be reflected")
	}
}

func Test7(t *testing.T) {
	// State is safe to call multiple times
	breaker := New(3, 5*time.Second, 2)
	for i := 0; i < 100; i++ {
		_ = breaker.State()
	}
}

func Test8(t *testing.T) {
	// State from concurrent goroutines
	breaker := New(3, 5*time.Second, 2)
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			_ = breaker.State()
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}

func Test9(t *testing.T) {
	// State does not modify breaker
	breaker := New(3, 5*time.Second, 2)
	breaker.errs = 2
	_ = breaker.State()
	if breaker.errs != 2 {
		t.Error("State should not modify breaker")
	}
}

func Test10(t *testing.T) {
	// State returns current snapshot
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	s := breaker.State()
	breaker.state = Closed
	if s != HalfOpen {
		t.Error("expected snapshot of previous state")
	}
}
`,
		hint1: `Use b.mu.Lock() and defer b.mu.Unlock() to protect the read operation.`,
			hint2: `Simply return b.state after acquiring the lock.`,
			whyItMatters: `The State method enables safe monitoring and decision-making based on circuit breaker status in concurrent environments.

**Why This Matters:**
- **Observability:** Monitor circuit breaker state in real-time
- **Metrics:** Track state transitions for alerting and dashboards
- **Conditional Logic:** Make decisions based on current state
- **Thread Safety:** Prevents race conditions when reading state

**Real-World Example:**
\`\`\`go
// Health check endpoint
func HealthHandler(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "database": dbBreaker.State() == circuitx.Closed,
        "cache":    cacheBreaker.State() == circuitx.Closed,
        "api":      apiBreaker.State() == circuitx.Closed,
    }

    allHealthy := true
    for _, v := range health {
        if !v.(bool) {
            allHealthy = false
            break
        }
    }

    if allHealthy {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    json.NewEncoder(w).Encode(health)
}
\`\`\`

**Production Monitoring:**
\`\`\`go
// Prometheus metrics
type CircuitBreakerCollector struct {
    breakers map[string]*circuitx.Breaker
}

func (c *CircuitBreakerCollector) Collect(ch chan<- prometheus.Metric) {
    for name, breaker := range c.breakers {
        state := breaker.State()
        stateValue := 0.0

        switch state {
        case circuitx.Closed:
            stateValue = 0.0
        case circuitx.Open:
            stateValue = 1.0
        case circuitx.HalfOpen:
            stateValue = 0.5
        }

        ch <- prometheus.MustNewConstMetric(
            prometheus.NewDesc(
                "circuit_breaker_state",
                "Circuit breaker state (0=closed, 0.5=half-open, 1=open)",
                []string{"name"}, nil,
            ),
            prometheus.GaugeValue,
            stateValue,
            name,
        )
    }
}
\`\`\`

**Conditional Request Handling:**
\`\`\`go
// Skip non-critical operations when circuit is open
func ProcessOrder(order Order) error {
    // Critical: Payment must succeed
    if err := paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return payment.Process(order)
    }); err != nil {
        return err
    }

    // Non-critical: Send notification (skip if circuit open)
    if notificationBreaker.State() == circuitx.Closed {
        notificationBreaker.Do(ctx, func(ctx context.Context) error {
            return notifications.Send(order.Email)
        })
    }

    return nil
}
\`\`\`

**Graceful Degradation:**
\`\`\`go
func GetUserProfile(userID string) (*Profile, error) {
    // Try cache first
    if cacheBreaker.State() == circuitx.Closed {
        if profile, err := cacheBreaker.Do(ctx, func(ctx context.Context) error {
            return cache.Get(userID)
        }); err == nil {
            return profile, nil
        }
    }

    // Fallback to database
    return dbBreaker.Do(ctx, func(ctx context.Context) error {
        return db.GetUser(userID)
    })
}
\`\`\`

**Why Read Lock (RLock):**
In the solution, we use \`b.mu.Lock()\` (write lock), but for read-only operations like State(), you could use \`b.mu.RLock()\`:

\`\`\`go
// Multiple goroutines can read state simultaneously
b.mu.RLock()
defer b.mu.RUnlock()
return b.state
\`\`\`

This allows better concurrency - multiple State() calls can execute in parallel without blocking each other, while still preventing races with Do() method's write operations.

**Key Insight:**
The State method seems simple, but it's crucial for:
1. Building reliable health checks
2. Implementing graceful degradation
3. Monitoring system resilience
4. Making intelligent routing decisions

Without State(), you're flying blind - you know requests are failing, but you don't know if the circuit breaker is protecting you or if it's still closed and accumulating errors.`,	order: 4,
	translations: {
		ru: {
			title: 'Получение состояния',
			solutionCode: `package circuitx

func (b *Breaker) State() State {
	b.mu.Lock()         // синхронизируем доступ к текущему состоянию
	defer b.mu.Unlock() // освобождаем блокировку после чтения
	return b.state      // возвращаем снимок состояния breaker
}`,
			description: `Реализуйте метод **State** для безопасного чтения текущего состояния circuit breaker.

**Требования:**
1. Верните текущее состояние (Closed, Open или HalfOpen)
2. Защитите чтение мьютексом (используйте RLock/RUnlock для чтения)
3. Метод должен быть безопасным для конкурентного доступа

**Почему Read Lock?**
\`\`\`go
// Read Lock (RLock) - Множественные читатели одновременно
// Write Lock (Lock) - Эксклюзивный доступ

b.mu.RLock()     // Разрешает конкурентное чтение
defer b.mu.RUnlock()
return b.state
\`\`\`

**Ограничения:**
- Используйте мьютекс для безопасности
- Используйте RLock/RUnlock для чтения
- Не изменяйте состояние, только читайте`,
			hint1: `Используйте b.mu.Lock() и defer b.mu.Unlock() для защиты чтения.`,
			hint2: `Просто верните b.state после получения блокировки.`,
			whyItMatters: `Метод State позволяет безопасно мониторить и принимать решения на основе статуса circuit breaker в concurrent окружениях.

**Почему важно:**
- **Наблюдаемость:** Мониторинг состояния в реальном времени
- **Метрики:** Отслеживание переходов состояний для алертов и дашбордов
- **Условная логика:** Принятие решений на основе текущего состояния
- **Потокобезопасность:** Предотвращает race conditions при чтении состояния

**Health Check Endpoint:**
\`\`\`go
func HealthHandler(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "database": dbBreaker.State() == circuitx.Closed,
        "cache":    cacheBreaker.State() == circuitx.Closed,
        "api":      apiBreaker.State() == circuitx.Closed,
    }

    allHealthy := true
    for _, v := range health {
        if !v.(bool) {
            allHealthy = false
            break
        }
    }

    if allHealthy {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    json.NewEncoder(w).Encode(health)
}
\`\`\`

**Prometheus Metrics:**
\`\`\`go
type CircuitBreakerCollector struct {
    breakers map[string]*circuitx.Breaker
}

func (c *CircuitBreakerCollector) Collect(ch chan<- prometheus.Metric) {
    for name, breaker := range c.breakers {
        state := breaker.State()
        stateValue := 0.0

        switch state {
        case circuitx.Closed:
            stateValue = 0.0
        case circuitx.Open:
            stateValue = 1.0
        case circuitx.HalfOpen:
            stateValue = 0.5
        }

        ch <- prometheus.MustNewConstMetric(
            prometheus.NewDesc(
                "circuit_breaker_state",
                "Состояние circuit breaker (0=закрыт, 0.5=полуоткрыт, 1=открыт)",
                []string{"name"}, nil,
            ),
            prometheus.GaugeValue,
            stateValue,
            name,
        )
    }
}
\`\`\`

**Graceful Degradation:**
\`\`\`go
func GetUserProfile(userID string) (*Profile, error) {
    // Попытка использовать кэш
    if cacheBreaker.State() == circuitx.Closed {
        if profile, err := cacheBreaker.Do(ctx, func(ctx context.Context) error {
            return cache.Get(userID)
        }); err == nil {
            return profile, nil
        }
    }

    // Откат на БД
    return dbBreaker.Do(ctx, func(ctx context.Context) error {
        return db.GetUser(userID)
    })
}
\`\`\`

**Conditional Request Handling:**
\`\`\`go
func ProcessOrder(order Order) error {
    // Критично: платёж должен успешно пройти
    if err := paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return payment.Process(order)
    }); err != nil {
        return err
    }

    // Некритично: отправить уведомление (пропустить если circuit открыт)
    if notificationBreaker.State() == circuitx.Closed {
        notificationBreaker.Do(ctx, func(ctx context.Context) error {
            return notifications.Send(order.Email)
        })
    }

    return nil
}
\`\`\`

**Мониторинг в реальном времени:**
\`\`\`go
// Мониторинг состояния circuit breaker
go func() {
    ticker := time.NewTicker(1 * time.Second)
    for range ticker.C {
        state := breaker.State()
        log.Printf("Circuit breaker state: %v", state)

        // Алертинг при открытии circuit
        if state == circuitx.Open {
            alerting.Send("Circuit breaker открыт!")
        }
    }
}()
\`\`\`

**Ключевые концепции:**
- **Read Lock:** Используйте RLock для concurrent чтения без блокировки записи
- **Snapshot:** State() возвращает снимок состояния в момент вызова
- **Без побочных эффектов:** Метод только читает, не изменяет состояние
- **Потокобезопасно:** Множественные goroutines могут вызывать State() одновременно

**Применения:**
- **Health Checks:** Определить доступность зависимостей
- **Metrics:** Экспорт состояния в системы мониторинга
- **Load Balancing:** Маршрутизация запросов на основе состояния
- **Graceful Degradation:** Пропуск некритичных операций

Без State() вы летите вслепую - знаете что запросы падают, но не знаете защищает ли вас circuit breaker или он всё ещё закрыт и накапливает ошибки.`
		},
		uz: {
			title: `Holatni olish`,
			solutionCode: `package circuitx

func (b *Breaker) State() State {
	b.mu.Lock()         // joriy holatga kirishni sinxronlashtiramiz
	defer b.mu.Unlock() // o'qishdan keyin qulfni bo'shatamiz
	return b.state      // breaker holati suratini qaytaramiz
}`,
			description: `Joriy circuit breaker holatini xavfsiz o'qish uchun **State** metodini amalga oshiring.

**Talablar:**
1. Circuit breaker ning joriy holatini qaytaring (Closed, Open yoki HalfOpen)
2. O'qish uchun mutex bilan himoya qiling (o'qish operatsiyalari uchun RLock/RUnlock ishlating)
3. Metod parallel kirish uchun xavfsiz bo'lishi kerak

**Nega Read Lock?**
\`\`\`go
// Read Lock (RLock) - Bir vaqtda bir nechta o'quvchilarga ruxsat
// Write Lock (Lock) - Eksklyuziv kirish

b.mu.RLock()     // Parallel o'qishga ruxsat beradi
defer b.mu.RUnlock()
return b.state
\`\`\`

**Cheklovlar:**
- Thread xavfsizligi uchun mutex ishlating
- O'qish uchun RLock/RUnlock ishlating
- Holatni o'zgartirmang, faqat o'qing`,
			hint1: `O'qish operatsiyasini himoya qilish uchun b.mu.Lock() va defer b.mu.Unlock() ishlating.`,
			hint2: `Lock olganingizdan keyin shunchaki b.state ni qaytaring.`,
			whyItMatters: `State metodi parallel muhitlarda circuit breaker holati asosida xavfsiz monitoring va qaror qabul qilishni ta'minlaydi.

**Nima uchun bu muhim:**
- **Kuzatish:** Circuit breaker holatini real vaqtda monitoring qilish
- **Metrikalar:** Ogohlantirishlar va dashboardlar uchun holat o'tishlarini kuzatish
- **Shartli mantiq:** Joriy holatga asoslangan qarorlar qabul qilish
- **Thread xavfsizligi:** Holatni o'qishda race conditionlarni oldini oladi

**Health Check Endpoint:**
\`\`\`go
func HealthHandler(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "database": dbBreaker.State() == circuitx.Closed,
        "cache":    cacheBreaker.State() == circuitx.Closed,
        "api":      apiBreaker.State() == circuitx.Closed,
    }

    allHealthy := true
    for _, v := range health {
        if !v.(bool) {
            allHealthy = false
            break
        }
    }

    if allHealthy {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
    }
    json.NewEncoder(w).Encode(health)
}
\`\`\`

**Prometheus Metrics:**
\`\`\`go
type CircuitBreakerCollector struct {
    breakers map[string]*circuitx.Breaker
}

func (c *CircuitBreakerCollector) Collect(ch chan<- prometheus.Metric) {
    for name, breaker := range c.breakers {
        state := breaker.State()
        stateValue := 0.0

        switch state {
        case circuitx.Closed:
            stateValue = 0.0
        case circuitx.Open:
            stateValue = 1.0
        case circuitx.HalfOpen:
            stateValue = 0.5
        }

        ch <- prometheus.MustNewConstMetric(
            prometheus.NewDesc(
                "circuit_breaker_state",
                "Circuit breaker holati (0=yopiq, 0.5=yarim ochiq, 1=ochiq)",
                []string{"name"}, nil,
            ),
            prometheus.GaugeValue,
            stateValue,
            name,
        )
    }
}
\`\`\`

**Graceful Degradation:**
\`\`\`go
func GetUserProfile(userID string) (*Profile, error) {
    // Keshdan foydalanishga harakat qilish
    if cacheBreaker.State() == circuitx.Closed {
        if profile, err := cacheBreaker.Do(ctx, func(ctx context.Context) error {
            return cache.Get(userID)
        }); err == nil {
            return profile, nil
        }
    }

    // Ma'lumotlar bazasiga qaytish
    return dbBreaker.Do(ctx, func(ctx context.Context) error {
        return db.GetUser(userID)
    })
}
\`\`\`

**Conditional Request Handling:**
\`\`\`go
func ProcessOrder(order Order) error {
    // Kritik: to'lov muvaffaqiyatli o'tishi kerak
    if err := paymentBreaker.Do(ctx, func(ctx context.Context) error {
        return payment.Process(order)
    }); err != nil {
        return err
    }

    // Kritik emas: xabarnoma yuborish (circuit ochiq bo'lsa o'tkazib yuborish)
    if notificationBreaker.State() == circuitx.Closed {
        notificationBreaker.Do(ctx, func(ctx context.Context) error {
            return notifications.Send(order.Email)
        })
    }

    return nil
}
\`\`\`

**Real vaqtda monitoring:**
\`\`\`go
// Circuit breaker holatini monitoring qilish
go func() {
    ticker := time.NewTicker(1 * time.Second)
    for range ticker.C {
        state := breaker.State()
        log.Printf("Circuit breaker state: %v", state)

        // Circuit ochilganda ogohlantirish
        if state == circuitx.Open {
            alerting.Send("Circuit breaker ochiq!")
        }
    }
}()
\`\`\`

**Asosiy tushunchalar:**
- **Read Lock:** Yozishni blokirovka qilmasdan concurrent o'qish uchun RLock ishlating
- **Snapshot:** State() chaqirilgan paytdagi holat suratini qaytaradi
- **Yon ta'sirlar yo'q:** Metod faqat o'qiydi, holatni o'zgartirmaydi
- **Thread-safe:** Bir nechta goroutinelar bir vaqtning o'zida State() ni chaqirishi mumkin

**Foydalanishlar:**
- **Health Checks:** Bog'liqliklarning mavjudligini aniqlash
- **Metrics:** Monitoring tizimlariga holatni eksport qilish
- **Load Balancing:** Holatga asoslangan so'rovlarni marshrutlash
- **Graceful Degradation:** Kritik bo'lmagan operatsiyalarni o'tkazib yuborish

State() siz siz ko'r uchyapsiz - so'rovlar muvaffaqiyatsiz bo'layotganini bilasiz, lekin circuit breaker sizni himoya qilyaptimi yoki hali yopiqmi va xatolarni to'plamoqdami bilmaysiz.`
		}
	}
};

export default task;
