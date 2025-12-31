import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-reset',
	title: 'Reset Method',
	difficulty: 'medium',	tags: ['go', 'circuit-breaker', 'admin'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Reset** method to manually close the circuit breaker.

**Requirements:**
1. Force circuit breaker to \`Closed\` state
2. Reset ALL counters: \`errs = 0\`, \`halfCount = 0\`
3. Clear the \`openUntil\` timestamp (set to zero value)
4. Protect mutations with mutex

**When to Use Reset:**
- Manual intervention by operators
- After fixing underlying service issue
- Testing and debugging
- Emergency override during incidents

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)

// Circuit opens after failures
for i := 0; i < 3; i++ {
    breaker.Do(ctx, failingFunc)
}

fmt.Println(breaker.State())  // Open

// Admin fixes the database issue
// Manually reset circuit without waiting for cooldown
breaker.Reset()

fmt.Println(breaker.State())  // Closed
// Circuit immediately accepts requests again
\`\`\`

**Constraints:**
- Must transition to Closed state regardless of current state
- Clear openUntil timestamp (use time.Time{} for zero value)
- Reset all counters to 0
- Use mutex for thread safety`,
	initialCode: `package circuitx

import "time"

// TODO: Implement Reset method
// Force state to Closed and clear all counters
func (b *Breaker) Reset() {
	// TODO: Implement
}`,
	solutionCode: `package circuitx

import "time"

func (b *Breaker) Reset() {
	b.mu.Lock()               // ensure exclusive access while mutating fields
	defer b.mu.Unlock()       // release lock after reset
	b.state = Closed          // close breaker immediately
	b.errs = 0                // clear accumulated error count
	b.halfCount = 0           // reset half-open success counter
	b.openUntil = time.Time{} // clear open-until timestamp
}`,
		testCode: `package circuitx

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Reset sets state to Closed
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.Reset()
	if breaker.state != Closed {
		t.Errorf("expected Closed state, got %v", breaker.state)
	}
}

func Test2(t *testing.T) {
	// Reset clears errs counter
	breaker := New(3, 5*time.Second, 2)
	breaker.errs = 5
	breaker.Reset()
	if breaker.errs != 0 {
		t.Errorf("expected errs 0, got %d", breaker.errs)
	}
}

func Test3(t *testing.T) {
	// Reset clears halfCount counter
	breaker := New(3, 5*time.Second, 2)
	breaker.halfCount = 3
	breaker.Reset()
	if breaker.halfCount != 0 {
		t.Errorf("expected halfCount 0, got %d", breaker.halfCount)
	}
}

func Test4(t *testing.T) {
	// Reset clears openUntil timestamp
	breaker := New(3, 5*time.Second, 2)
	breaker.openUntil = time.Now().Add(1 * time.Hour)
	breaker.Reset()
	if !breaker.openUntil.IsZero() {
		t.Error("expected openUntil to be zero")
	}
}

func Test5(t *testing.T) {
	// Reset from Open state
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.errs = 10
	breaker.openUntil = time.Now().Add(1 * time.Hour)
	breaker.Reset()
	if breaker.state != Closed || breaker.errs != 0 {
		t.Error("Reset should work from Open state")
	}
}

func Test6(t *testing.T) {
	// Reset from HalfOpen state
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	breaker.halfCount = 1
	breaker.Reset()
	if breaker.state != Closed || breaker.halfCount != 0 {
		t.Error("Reset should work from HalfOpen state")
	}
}

func Test7(t *testing.T) {
	// Reset from Closed state
	breaker := New(3, 5*time.Second, 2)
	breaker.errs = 2
	breaker.Reset()
	if breaker.state != Closed || breaker.errs != 0 {
		t.Error("Reset should work from Closed state")
	}
}

func Test8(t *testing.T) {
	// Reset multiple times
	breaker := New(3, 5*time.Second, 2)
	for i := 0; i < 5; i++ {
		breaker.state = Open
		breaker.Reset()
		if breaker.state != Closed {
			t.Errorf("Reset %d failed", i)
		}
	}
}

func Test9(t *testing.T) {
	// Reset is safe for concurrent calls
	breaker := New(3, 5*time.Second, 2)
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			breaker.Reset()
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}

func Test10(t *testing.T) {
	// Reset preserves configuration
	breaker := New(5, 10*time.Second, 3)
	breaker.state = Open
	breaker.Reset()
	if breaker.threshold != 5 || breaker.openDur != 10*time.Second || breaker.halfMax != 3 {
		t.Error("Reset should preserve configuration")
	}
}
`,
		hint1: `Use b.mu.Lock() and defer b.mu.Unlock() to protect the state mutation.`,
			hint2: `Set state to Closed, all counters to 0, and openUntil to time.Time{}.`,
			whyItMatters: `The Reset method provides manual control over circuit breaker state, crucial for operational flexibility and incident response.

**Why This Matters:**
- **Manual Override:** Operators can force circuit closed after fixing issues
- **Incident Response:** Quick recovery without waiting for cooldown
- **Testing:** Reset state between test cases
- **Deployment:** Clear circuit state after rolling out fixes

**Real-World Example:**
\`\`\`go
// Admin API endpoint for manual circuit breaker control
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    service := r.URL.Query().Get("service")

    switch service {
    case "database":
        dbBreaker.Reset()
        log.Info("Database circuit breaker manually reset")
    case "payment":
        paymentBreaker.Reset()
        log.Info("Payment circuit breaker manually reset")
    case "cache":
        cacheBreaker.Reset()
        log.Info("Cache circuit breaker manually reset")
    default:
        http.Error(w, "Unknown service", http.StatusBadRequest)
        return
    }

    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, "Circuit breaker for %s has been reset", service)
}
\`\`\`

**Production Incident Scenario:**

**3:00 AM - Database goes down**
- Circuit opens after threshold failures
- All requests start failing fast with ErrOpen
- On-call engineer gets paged

**3:15 AM - Database team fixes issue**
- Database is healthy again
- But circuit breaker is still Open (waiting for cooldown)
- Users still getting errors!

**3:16 AM - Manual Reset**
\`\`\`go
// Engineer calls admin API
curl -X POST http://admin.example.com/circuit-breaker/reset?service=database

// Or via internal tool
dbBreaker.Reset()
\`\`\`

- Circuit immediately closes
- Traffic resumes instantly
- No waiting for cooldown period
- Users get successful responses

**Without Reset:** Would need to wait full cooldown period (e.g., 60 seconds) even though issue is fixed, causing unnecessary service degradation.

**Advanced Pattern - Graceful Reset:**
\`\`\`go
// Reset with health check validation
func SafeReset(breaker *circuitx.Breaker, healthCheck func() error) error {
    // Verify service is actually healthy before resetting
    if err := healthCheck(); err != nil {
        return fmt.Errorf("health check failed, not resetting: %w", err)
    }

    breaker.Reset()
    log.Info("Circuit breaker reset after successful health check")
    return nil
}

// Usage
SafeReset(dbBreaker, func() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    return db.Ping(ctx)
})
\`\`\`

**Monitoring Integration:**
\`\`\`go
// Log reset events for audit trail
func (b *Breaker) Reset() {
    b.mu.Lock()
    defer b.mu.Unlock()

    oldState := b.state
    b.state = Closed
    b.errs = 0
    b.halfCount = 0
    b.openUntil = time.Time{}

    // Emit metric
    metrics.Counter("circuit_breaker_reset_total").Inc()

    // Log for audit
    log.WithFields(log.Fields{
        "old_state": oldState,
        "new_state": "Closed",
        "event":     "manual_reset",
    }).Info("Circuit breaker manually reset")
}
\`\`\`

**Testing Use Case:**
\`\`\`go
func TestCircuitBreaker(t *testing.T) {
    breaker := New(3, 5*time.Second, 2)

    t.Run("opens after threshold", func(t *testing.T) {
        // Test logic...
        breaker.Reset()  // Clean state for next test
    })

    t.Run("closes after successful trials", func(t *testing.T) {
        // Test logic...
        breaker.Reset()  // Clean state for next test
    })
}
\`\`\`

**Security Consideration:**
\`\`\`go
// Protect Reset endpoint with authentication
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    // Verify admin role
    if !hasAdminRole(r) {
        http.Error(w, "Unauthorized", http.StatusUnauthorized)
        return
    }

    // Log who performed the reset
    user := getUserFromContext(r.Context())
    log.WithField("user", user).Info("Manual circuit reset requested")

    dbBreaker.Reset()
    w.WriteHeader(http.StatusOK)
}
\`\`\`

**Key Insight:**
Reset is a "break glass in case of emergency" feature. It's powerful because it overrides the circuit breaker's automated decision-making, giving humans control when automation isn't responding appropriately to real-world conditions.

**Best Practices:**
1. **Log all resets** for audit trail
2. **Require authentication** for reset endpoints
3. **Validate health** before resetting when possible
4. **Monitor reset frequency** - frequent resets indicate misconfiguration
5. **Document runbooks** explaining when to use Reset`,	order: 5,
	translations: {
		ru: {
			title: 'Сброс Circuit Breaker',
			solutionCode: `package circuitx

import "time"

func (b *Breaker) Reset() {
	b.mu.Lock()               // обеспечиваем эксклюзивный доступ при изменении полей
	defer b.mu.Unlock()       // освобождаем блокировку после сброса
	b.state = Closed          // немедленно закрываем breaker
	b.errs = 0                // очищаем накопленный счётчик ошибок
	b.halfCount = 0           // сбрасываем счётчик успехов half-open
	b.openUntil = time.Time{} // очищаем timestamp open-until
}`,
			description: `Реализуйте метод **Reset** для ручного закрытия circuit breaker.

**Требования:**
1. Принудительно переведите в состояние \`Closed\`
2. Сбросьте ВСЕ счётчики: \`errs = 0\`, \`halfCount = 0\`
3. Очистите timestamp \`openUntil\` (установите в нулевое значение)
4. Защитите мутации мьютексом

**Когда использовать Reset:**
- Ручное вмешательство операторов
- После исправления проблемы с сервисом
- Тестирование и отладка
- Экстренное переопределение при инцидентах

**Ограничения:**
- Должен перейти в Closed независимо от текущего состояния
- Очистить openUntil (используйте time.Time{} для нулевого значения)
- Сбросить все счётчики в 0
- Использовать мьютекс для безопасности`,
			hint1: `Используйте b.mu.Lock() и defer b.mu.Unlock() для защиты мутации.`,
			hint2: `Установите state в Closed, все счётчики в 0, openUntil в time.Time{}.`,
			whyItMatters: `Метод Reset предоставляет ручное управление состоянием circuit breaker, критично для операционной гибкости и реагирования на инциденты.

**Почему это важно:**
- **Ручное переопределение:** Операторы могут принудительно закрыть цепь после исправления проблем
- **Реагирование на инциденты:** Быстрое восстановление без ожидания периода cooldown
- **Тестирование:** Сброс состояния между тестовыми случаями
- **Развёртывание:** Очистка состояния цепи после выкатки исправлений

**Пример из реальной практики:**
\`\`\`go
// Admin API endpoint для ручного управления circuit breaker
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    service := r.URL.Query().Get("service")

    switch service {
    case "database":
        dbBreaker.Reset()
        log.Info("Circuit breaker базы данных вручную сброшен")
    case "payment":
        paymentBreaker.Reset()
        log.Info("Circuit breaker платежей вручную сброшен")
    case "cache":
        cacheBreaker.Reset()
        log.Info("Circuit breaker кэша вручную сброшен")
    default:
        http.Error(w, "Неизвестный сервис", http.StatusBadRequest)
        return
    }

    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, "Circuit breaker для %s был сброшен", service)
}
\`\`\`

**Сценарий производственного инцидента:**

**3:00 AM - База данных выходит из строя**
- Circuit breaker открывается после превышения порога ошибок
- Все запросы начинают быстро завершаться с ошибкой ErrOpen
- Дежурный инженер получает уведомление о проблеме

**3:15 AM - Команда БД устраняет проблему**
- База данных снова работает нормально
- Но circuit breaker всё ещё в состоянии Open (ожидает период cooldown)
- Пользователи продолжают получать ошибки!

**3:16 AM - Ручной сброс**
\`\`\`go
// Инженер вызывает admin API
curl -X POST http://admin.example.com/circuit-breaker/reset?service=database

// Или через внутренний инструмент
dbBreaker.Reset()
\`\`\`

- Circuit breaker немедленно закрывается
- Трафик восстанавливается мгновенно
- Не нужно ждать период cooldown
- Пользователи начинают получать успешные ответы

**Без Reset:** Пришлось бы ждать полный период cooldown (например, 60 секунд), даже если проблема уже устранена, что вызывает ненужную деградацию сервиса.

**Продвинутый паттерн - Безопасный сброс:**
\`\`\`go
// Сброс с валидацией health check
func SafeReset(breaker *circuitx.Breaker, healthCheck func() error) error {
    // Проверяем, что сервис действительно здоров перед сбросом
    if err := healthCheck(); err != nil {
        return fmt.Errorf("health check не прошёл, сброс не выполнен: %w", err)
    }

    breaker.Reset()
    log.Info("Circuit breaker сброшен после успешного health check")
    return nil
}

// Использование
SafeReset(dbBreaker, func() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    return db.Ping(ctx)
})
\`\`\`

**Интеграция с мониторингом:**
\`\`\`go
// Логирование событий сброса для аудита
func (b *Breaker) Reset() {
    b.mu.Lock()
    defer b.mu.Unlock()

    oldState := b.state
    b.state = Closed
    b.errs = 0
    b.halfCount = 0
    b.openUntil = time.Time{}

    // Отправка метрики
    metrics.Counter("circuit_breaker_reset_total").Inc()

    // Логирование для аудита
    log.WithFields(log.Fields{
        "old_state": oldState,
        "new_state": "Closed",
        "event":     "manual_reset",
    }).Info("Circuit breaker вручную сброшен")
}
\`\`\`

**Использование в тестировании:**
\`\`\`go
func TestCircuitBreaker(t *testing.T) {
    breaker := New(3, 5*time.Second, 2)

    t.Run("открывается после превышения порога", func(t *testing.T) {
        // Логика теста...
        breaker.Reset()  // Очистка состояния для следующего теста
    })

    t.Run("закрывается после успешных попыток", func(t *testing.T) {
        // Логика теста...
        breaker.Reset()  // Очистка состояния для следующего теста
    })
}
\`\`\`

**Соображения безопасности:**
\`\`\`go
// Защита endpoint Reset с помощью аутентификации
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    // Проверка роли администратора
    if !hasAdminRole(r) {
        http.Error(w, "Не авторизован", http.StatusUnauthorized)
        return
    }

    // Логирование, кто выполнил сброс
    user := getUserFromContext(r.Context())
    log.WithField("user", user).Info("Запрошен ручной сброс circuit breaker")

    dbBreaker.Reset()
    w.WriteHeader(http.StatusOK)
}
\`\`\`

**Ключевое понимание:**
Reset - это функция "разбить стекло в случае чрезвычайной ситуации". Она мощна, потому что переопределяет автоматическое принятие решений circuit breaker, давая людям контроль, когда автоматизация не реагирует должным образом на реальные условия.

**Лучшие практики:**
1. **Логировать все сбросы** для ведения аудита
2. **Требовать аутентификацию** для endpoint'ов сброса
3. **Валидировать здоровье** перед сбросом, когда это возможно
4. **Мониторить частоту сбросов** - частые сбросы указывают на неправильную конфигурацию
5. **Документировать runbook'и** с объяснением, когда использовать Reset`
		},
		uz: {
			title: `Circuit Breaker ni qayta tiklash`,
			solutionCode: `package circuitx

import "time"

func (b *Breaker) Reset() {
	b.mu.Lock()               // maydonlarni o'zgartirishda eksklyuziv kirishni ta'minlaymiz
	defer b.mu.Unlock()       // qayta o'rnatishdan keyin qulfni bo'shatamiz
	b.state = Closed          // breakerni darhol yopamiz
	b.errs = 0                // to'plangan xato hisobini tozalaymiz
	b.halfCount = 0           // half-open muvaffaqiyat hisoblagichini qayta o'rnatamiz
	b.openUntil = time.Time{} // open-until vaqt belgisini tozalaymiz
}`,
			description: `Circuit breakerni qo'lda yopish uchun **Reset** metodini amalga oshiring.

**Talablar:**
1. Circuit breakerni \`Closed\` holatiga majburlang
2. BARCHA hisoblagichlarni qayta o'rnating: \`errs = 0\`, \`halfCount = 0\`
3. \`openUntil\` vaqt belgisini tozalang (nol qiymatga o'rnating)
4. Mutatsiyalarni mutex bilan himoyalang

**Reset ni qachon ishlatish:**
- Operatorlar tomonidan qo'lda aralashuv
- Asosiy xizmat muammosini tuzatgandan keyin
- Sinov va debug
- Incidentlar paytida favqulodda bekor qilish

**Cheklovlar:**
- Joriy holatdan qat'i nazar Closed holatiga o'tishi kerak
- openUntil vaqt belgisini tozalang (nol qiymat uchun time.Time{} ishlating)
- Barcha hisoblagichlarni 0 ga qayta o'rnating
- Thread xavfsizligi uchun mutex ishlating`,
			hint1: `Holat mutatsiyasini himoya qilish uchun b.mu.Lock() va defer b.mu.Unlock() ishlating.`,
			hint2: `state ni Closed ga, barcha hisoblagichlarni 0 ga, openUntil ni time.Time{} ga o'rnating.`,
			whyItMatters: `Reset metodi circuit breaker holati ustidan qo'lda boshqaruvni ta'minlaydi, operatsion moslashuvchanlik va incidentlarga javob berish uchun juda muhim.

**Nima uchun bu muhim:**
- **Qo'lda bekor qilish:** Operatorlar muammolarni tuzatgandan keyin zanjirni majburiy ravishda yopishi mumkin
- **Incidentlarga javob:** Cooldown davrini kutmasdan tez tiklash
- **Sinov:** Test holatlari orasida holatni qayta o'rnatish
- **Joylashtirish:** Tuzatishlarni joylashtirganidan keyin zanjir holatini tozalash

**Amaliy hayotdan misol:**
\`\`\`go
// Circuit breaker qo'lda boshqarish uchun admin API endpoint
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    service := r.URL.Query().Get("service")

    switch service {
    case "database":
        dbBreaker.Reset()
        log.Info("Ma'lumotlar bazasi circuit breaker qo'lda qayta o'rnatildi")
    case "payment":
        paymentBreaker.Reset()
        log.Info("To'lov circuit breaker qo'lda qayta o'rnatildi")
    case "cache":
        cacheBreaker.Reset()
        log.Info("Kesh circuit breaker qo'lda qayta o'rnatildi")
    default:
        http.Error(w, "Noma'lum xizmat", http.StatusBadRequest)
        return
    }

    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, "%s uchun circuit breaker qayta o'rnatildi", service)
}
\`\`\`

**Ishlab chiqarish incidenti stsenariysi:**

**3:00 AM - Ma'lumotlar bazasi ishdan chiqadi**
- Circuit breaker xatolar chegarasidan oshgandan keyin ochiladi
- Barcha so'rovlar ErrOpen xatosi bilan tezda muvaffaqiyatsiz tugay boshlaydi
- Navbatdagi muhandis muammo haqida xabar oladi

**3:15 AM - Ma'lumotlar bazasi jamoasi muammoni tuzatadi**
- Ma'lumotlar bazasi yana normal ishlayapti
- Ammo circuit breaker hali Open holatida (cooldown davrini kutmoqda)
- Foydalanuvchilar hali ham xatolarni olishda davom etmoqda!

**3:16 AM - Qo'lda qayta o'rnatish**
\`\`\`go
// Muhandis admin API ni chaqiradi
curl -X POST http://admin.example.com/circuit-breaker/reset?service=database

// Yoki ichki vosita orqali
dbBreaker.Reset()
\`\`\`

- Circuit breaker darhol yopiladi
- Trafik bir zumda tiklanadi
- Cooldown davrini kutish shart emas
- Foydalanuvchilar muvaffaqiyatli javoblarni olishni boshlaydi

**Reset siz:** Muammo tuzatilgan bo'lsa ham, to'liq cooldown davrini (masalan, 60 soniya) kutish kerak bo'lardi, bu esa keraksiz xizmat degradatsiyasiga olib keladi.

**Rivojlangan pattern - Xavfsiz qayta o'rnatish:**
\`\`\`go
// Health check validatsiyasi bilan qayta o'rnatish
func SafeReset(breaker *circuitx.Breaker, healthCheck func() error) error {
    // Qayta o'rnatishdan oldin xizmat haqiqatan sog'lomligini tekshiramiz
    if err := healthCheck(); err != nil {
        return fmt.Errorf("health check muvaffaqiyatsiz, qayta o'rnatilmadi: %w", err)
    }

    breaker.Reset()
    log.Info("Circuit breaker muvaffaqiyatli health check dan keyin qayta o'rnatildi")
    return nil
}

// Foydalanish
SafeReset(dbBreaker, func() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    return db.Ping(ctx)
})
\`\`\`

**Monitoring bilan integratsiya:**
\`\`\`go
// Audit uchun qayta o'rnatish hodisalarini log qilish
func (b *Breaker) Reset() {
    b.mu.Lock()
    defer b.mu.Unlock()

    oldState := b.state
    b.state = Closed
    b.errs = 0
    b.halfCount = 0
    b.openUntil = time.Time{}

    // Metrikani yuborish
    metrics.Counter("circuit_breaker_reset_total").Inc()

    // Audit uchun log
    log.WithFields(log.Fields{
        "old_state": oldState,
        "new_state": "Closed",
        "event":     "manual_reset",
    }).Info("Circuit breaker qo'lda qayta o'rnatildi")
}
\`\`\`

**Testda foydalanish holati:**
\`\`\`go
func TestCircuitBreaker(t *testing.T) {
    breaker := New(3, 5*time.Second, 2)

    t.Run("chegara oshgandan keyin ochiladi", func(t *testing.T) {
        // Test mantiq...
        breaker.Reset()  // Keyingi test uchun holatni tozalash
    })

    t.Run("muvaffaqiyatli sinovlardan keyin yopiladi", func(t *testing.T) {
        // Test mantiq...
        breaker.Reset()  // Keyingi test uchun holatni tozalash
    })
}
\`\`\`

**Xavfsizlik fikrlari:**
\`\`\`go
// Reset endpoint ni autentifikatsiya bilan himoyalash
func ResetCircuitHandler(w http.ResponseWriter, r *http.Request) {
    // Admin rolini tekshirish
    if !hasAdminRole(r) {
        http.Error(w, "Ruxsat berilmagan", http.StatusUnauthorized)
        return
    }

    // Kim qayta o'rnatganini log qilish
    user := getUserFromContext(r.Context())
    log.WithField("user", user).Info("Qo'lda circuit qayta o'rnatish so'raldi")

    dbBreaker.Reset()
    w.WriteHeader(http.StatusOK)
}
\`\`\`

**Asosiy tushuncha:**
Reset bu "favqulodda holatlarda oynani sindirish" funksiyasi. U kuchli, chunki circuit breakerning avtomatik qaror qabul qilishini bekor qiladi va avtomatlashtirish real sharoitlarga to'g'ri javob bermayotganda odamlarga nazoratni beradi.

**Eng yaxshi amaliyotlar:**
1. **Barcha qayta o'rnatishlarni log qilish** audit uchun
2. **Qayta o'rnatish endpointlari uchun autentifikatsiya talab qilish**
3. **Iloji bo'lsa qayta o'rnatishdan oldin sog'lomlikni validatsiya qilish**
4. **Qayta o'rnatish chastotasini monitoring qilish** - tez-tez qayta o'rnatish noto'g'ri konfiguratsiyani ko'rsatadi
5. **Runbooklarni hujjatlash** Reset qachon ishlatishni tushuntirish bilan`
		}
	}
};

export default task;
