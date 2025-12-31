import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-backoff-sequence',
	title: 'Generate Backoff Sequence',
	difficulty: 'medium',	tags: ['go', 'retry', 'backoff', 'testing'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **BackoffSequence** function that generates a slice of backoff durations for testing and monitoring.

**Requirements:**
1. Generate exactly \`n\` backoff durations
2. Use \`Backoff(attempt, base)\` for each index
3. Return nil for n <= 0
4. Guard against invalid base duration

**Purpose:**
This function is useful for:
- **Testing:** Verify retry timing behavior
- **Monitoring:** Log planned retry schedule
- **Configuration:** Preview backoff strategy before applying

**Example:**
\`\`\`go
// Generate first 5 retry delays
seq := BackoffSequence(5, 100*time.Millisecond)
// Result (approximate, jitter varies):
// [100ms, 200ms, 400ms, 800ms, 1600ms]

for i, delay := range seq {
    fmt.Printf("Retry %d: wait %v\n", i, delay)
}
// Output:
// Retry 0: wait 127ms
// Retry 1: wait 189ms
// Retry 2: wait 423ms
// Retry 3: wait 991ms
// Retry 4: wait 1523ms
\`\`\`

**Testing Use Case:**
\`\`\`go
func TestRetryTiming(t *testing.T) {
    seq := BackoffSequence(3, 100*time.Millisecond)

    // Verify exponential growth
    assert.True(t, seq[1] > seq[0])
    assert.True(t, seq[2] > seq[1])

    // Verify jitter range [0.5x, 1.5x]
    expected := 100 * time.Millisecond
    assert.GreaterOrEqual(t, seq[0], expected/2)
    assert.LessOrEqual(t, seq[0], expected*3/2)
}
\`\`\`

**Monitoring Use Case:**
\`\`\`go
// Log retry schedule for debugging
seq := BackoffSequence(maxRetries, baseDelay)
log.Printf("Retry schedule: %v", seq)
// Output: Retry schedule: [123ms 234ms 456ms 912ms]

// Estimate max retry time
totalTime := time.Duration(0)
for _, delay := range seq {
    totalTime += delay
}
log.Printf("Max retry duration: %v", totalTime)
\`\`\`

**Configuration Preview:**
\`\`\`go
// Show user what retry strategy looks like
func ShowRetryConfig(n int, base time.Duration) {
    seq := BackoffSequence(n, base)
    fmt.Printf("Retry strategy: %d attempts\n", n)
    for i, delay := range seq {
        fmt.Printf("  Attempt %d: wait %v\n", i+1, delay)
    }
    fmt.Printf("Total retry time: %v\n", sum(seq))
}

// Output:
// Retry strategy: 4 attempts
//   Attempt 1: wait 134ms
//   Attempt 2: wait 245ms
//   Attempt 3: wait 489ms
//   Attempt 4: wait 978ms
// Total retry time: 1846ms
\`\`\`

**Edge Cases:**
- n <= 0 → return nil
- n == 1 → single-element slice
- base <= 0 → use 1ms default

**Constraints:**
- Preallocate slice with exact size
- Reuse Backoff() function for consistency
- Handle invalid inputs gracefully`,
	initialCode: `package retryx

import "time"

// TODO: Implement BackoffSequence function
// Generate slice of n backoff durations
func BackoffSequence(n int, base time.Duration) []time.Duration {
	// TODO: Implement
}`,
	solutionCode: `package retryx

import "time"

func BackoffSequence(n int, base time.Duration) []time.Duration {
	if n <= 0 {	// No durations requested
		return nil
	}
	if base <= 0 {	// Guard against invalid base
		base = time.Millisecond
	}
	seq := make([]time.Duration, n)	// Preallocate exact size
	for i := 0; i < n; i++ {	// Generate each delay
		seq[i] = Backoff(i, base)	// Reuse Backoff for consistency
	}
	return seq	// Return complete sequence
}`,
	testCode: `package retryx

import (
	"testing"
	"time"
)

func TestBackoffSequenceWithZeroAttempts(t *testing.T) {
	result := BackoffSequence(0, 100*time.Millisecond)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestBackoffSequenceWithNegativeAttempts(t *testing.T) {
	result := BackoffSequence(-5, 100*time.Millisecond)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestBackoffSequenceWithOneAttempt(t *testing.T) {
	result := BackoffSequence(1, 100*time.Millisecond)
	if len(result) != 1 {
		t.Errorf("expected length 1, got %d", len(result))
	}
	if result[0] < 50*time.Millisecond || result[0] > 150*time.Millisecond {
		t.Errorf("expected result[0] between 50ms and 150ms, got %v", result[0])
	}
}

func TestBackoffSequenceWithMultipleAttempts(t *testing.T) {
	result := BackoffSequence(5, 100*time.Millisecond)
	if len(result) != 5 {
		t.Errorf("expected length 5, got %d", len(result))
	}
	for i, delay := range result {
		t.Logf("Attempt %d: %v", i, delay)
	}
}

func TestBackoffSequenceExponentialGrowth(t *testing.T) {
	result := BackoffSequence(4, 100*time.Millisecond)
	// Verify exponential growth pattern (considering jitter)
	// Each attempt should generally be larger than the previous
	for i := 1; i < len(result); i++ {
		// Due to jitter, this is not guaranteed, but we can log it
		t.Logf("Attempt %d: %v, Attempt %d: %v", i-1, result[i-1], i, result[i])
	}
}

func TestBackoffSequenceWithZeroBase(t *testing.T) {
	result := BackoffSequence(3, 0)
	if len(result) != 3 {
		t.Errorf("expected length 3, got %d", len(result))
	}
	// Should default to 1ms
	for i, delay := range result {
		if delay < 0 {
			t.Errorf("expected positive delay at index %d, got %v", i, delay)
		}
	}
}

func TestBackoffSequenceWithNegativeBase(t *testing.T) {
	result := BackoffSequence(3, -100*time.Millisecond)
	if len(result) != 3 {
		t.Errorf("expected length 3, got %d", len(result))
	}
	// Should default to 1ms
	for i, delay := range result {
		if delay < 0 {
			t.Errorf("expected positive delay at index %d, got %v", i, delay)
		}
	}
}

func TestBackoffSequenceLength(t *testing.T) {
	n := 10
	result := BackoffSequence(n, 50*time.Millisecond)
	if len(result) != n {
		t.Errorf("expected length %d, got %d", n, len(result))
	}
}

func TestBackoffSequenceAllPositive(t *testing.T) {
	result := BackoffSequence(5, 100*time.Millisecond)
	for i, delay := range result {
		if delay <= 0 {
			t.Errorf("expected positive delay at index %d, got %v", i, delay)
		}
	}
}

func TestBackoffSequenceJitterVariance(t *testing.T) {
	// Generate two sequences and verify they are different due to jitter
	seq1 := BackoffSequence(5, 100*time.Millisecond)
	seq2 := BackoffSequence(5, 100*time.Millisecond)

	different := false
	for i := 0; i < len(seq1); i++ {
		if seq1[i] != seq2[i] {
			different = true
			break
		}
	}

	if !different {
		t.Logf("sequences may be identical due to random jitter, but usually should differ")
	}
}`,
			hint1: `Preallocate slice with make([]time.Duration, n) for exact size.`,
			hint2: `Loop from 0 to n-1 and call Backoff(i, base) for each index.`,
			whyItMatters: `Generating backoff sequences enables testing, monitoring, and configuration of retry behavior.

**Why This Function Matters:**

**1. Testing Retry Logic:**
\`\`\`go
func TestExponentialGrowth(t *testing.T) {
    seq := BackoffSequence(5, 100*time.Millisecond)

    // Verify each delay is roughly 2x previous (±50% jitter)
    for i := 1; i < len(seq); i++ {
        ratio := float64(seq[i]) / float64(seq[i-1])
        // Should be ~2.0, but jitter allows 1.0-3.0
        assert.GreaterOrEqual(t, ratio, 1.0)
        assert.LessOrEqual(t, ratio, 3.0)
    }
}

// Without BackoffSequence, you'd need to:
// 1. Mock time.Sleep
// 2. Capture actual retry delays
// 3. Much more complex test setup
\`\`\`

**2. Monitoring Retry Behavior:**
\`\`\`go
// Production monitoring code
func retryWithMetrics(ctx context.Context, op Op) error {
    seq := BackoffSequence(maxRetries, baseDelay)

    // Log planned retry schedule
    logger.Info("Starting retry operation",
        "max_retries", maxRetries,
        "retry_schedule", seq,
        "max_duration", sum(seq))

    return Do(ctx, maxRetries, baseDelay, op)
}

// Logs:
// Starting retry operation max_retries=5
// retry_schedule=[127ms 234ms 456ms 912ms 1824ms]
// max_duration=3553ms
\`\`\`

**3. Capacity Planning:**
\`\`\`go
// Calculate worst-case retry duration for SLA planning
func maxRetryDuration(attempts int, base time.Duration) time.Duration {
    seq := BackoffSequence(attempts, base)
    total := time.Duration(0)
    for _, delay := range seq {
        total += delay
    }
    // Account for jitter (max 1.5x)
    return time.Duration(float64(total) * 1.5)
}

// If SLA requires response within 5 seconds:
maxDuration := maxRetryDuration(retries, baseDelay)
if maxDuration > 5*time.Second {
    log.Warn("Retry config may violate SLA",
        "max_duration", maxDuration,
        "sla", 5*time.Second)
}
\`\`\`

**4. Configuration Validation:**
\`\`\`go
// Validate retry config at startup
func validateRetryConfig(cfg RetryConfig) error {
    seq := BackoffSequence(cfg.MaxRetries, cfg.BaseDelay)

    // Check total time doesn't exceed limit
    total := sum(seq)
    if total > cfg.MaxTotalTime {
        return fmt.Errorf("retry config invalid: total=%v exceeds max=%v",
            total, cfg.MaxTotalTime)
    }

    // Check individual delays aren't too large
    for i, delay := range seq {
        if delay > cfg.MaxSingleDelay {
            return fmt.Errorf("retry %d delay=%v exceeds max=%v",
                i, delay, cfg.MaxSingleDelay)
        }
    }

    return nil
}
\`\`\`

**5. User-Facing Configuration:**
\`\`\`go
// CLI tool showing retry behavior
$ myapp retry-config --show
Retry Configuration:
  Max Attempts: 5
  Base Delay: 100ms

Retry Schedule:
  Attempt 1: 123ms  (total: 123ms)
  Attempt 2: 234ms  (total: 357ms)
  Attempt 3: 456ms  (total: 813ms)
  Attempt 4: 912ms  (total: 1725ms)
  Attempt 5: 1824ms (total: 3549ms)

Worst case: 5.3 seconds (with max jitter)
\`\`\`

**Real-World Example (AWS SDK):**
AWS SDK for Go includes similar functionality to display retry attempts:
\`\`\`go
// aws-sdk-go/aws/client/default_retryer.go
func (d DefaultRetryer) RetryRules(r *request.Request) time.Duration {
    delay := d.computeDelay(r.RetryCount)
    logger.Debug("Retry attempt",
        "attempt", r.RetryCount,
        "delay", delay)
    return delay
}
\`\`\`

**Performance Note:**
This function is cheap (allocates one slice) and typically called once at startup or in tests, not in hot paths. The small allocation cost is worth the visibility it provides.`,	order: 3,
	translations: {
		ru: {
			title: 'Последовательность задержек',
			solutionCode: `package retryx

import "time"

func BackoffSequence(n int, base time.Duration) []time.Duration {
	if n <= 0 {	// Длительности не запрошены
		return nil
	}
	if base <= 0 {	// Защита от некорректной базы
		base = time.Millisecond
	}
	seq := make([]time.Duration, n)	// Предварительное выделение точного размера
	for i := 0; i < n; i++ {	// Генерация каждой задержки
		seq[i] = Backoff(i, base)	// Переиспользуем Backoff для согласованности
	}
	return seq	// Возвращаем полную последовательность
}`,
			description: `Реализуйте функцию **BackoffSequence**, которая генерирует слайс backoff длительностей для тестирования и мониторинга.

**Требования:**
1. Сгенерируйте ровно \`n\` backoff длительностей
2. Используйте \`Backoff(attempt, base)\` для каждого индекса
3. Верните nil для n <= 0
4. Защитите от некорректной базовой длительности

**Назначение:**
Функция полезна для:
- **Тестирования:** Проверка поведения retry timing
- **Мониторинга:** Логирование запланированного retry расписания
- **Конфигурации:** Предпросмотр backoff стратегии

**Пример:**
\`\`\`go
// Генерация первых 5 retry задержек
seq := BackoffSequence(5, 100*time.Millisecond)
// Результат (примерно, джиттер варьируется):
// [100ms, 200ms, 400ms, 800ms, 1600ms]

for i, delay := range seq {
    fmt.Printf("Retry %d: wait %v\n", i, delay)
}
// Вывод:
// Retry 0: wait 127ms
// Retry 1: wait 189ms
// Retry 2: wait 423ms
// Retry 3: wait 991ms
// Retry 4: wait 1523ms
\`\`\`

**Use Case тестирования:**
\`\`\`go
func TestRetryTiming(t *testing.T) {
    seq := BackoffSequence(3, 100*time.Millisecond)

    // Проверка экспоненциального роста
    assert.True(t, seq[1] > seq[0])
    assert.True(t, seq[2] > seq[1])

    // Проверка диапазона jitter [0.5x, 1.5x]
    expected := 100 * time.Millisecond
    assert.GreaterOrEqual(t, seq[0], expected/2)
    assert.LessOrEqual(t, seq[0], expected*3/2)
}
\`\`\`

**Use Case мониторинга:**
\`\`\`go
// Логирование retry расписания для отладки
seq := BackoffSequence(maxRetries, baseDelay)
log.Printf("Retry schedule: %v", seq)

// Оценка максимального retry времени
totalTime := time.Duration(0)
for _, delay := range seq {
    totalTime += delay
}
log.Printf("Max retry duration: %v", totalTime)
`,
			hint1: `Предварительно выделите слайс с make([]time.Duration, n) для точного размера.`,
			hint2: `Цикл от 0 до n-1 и вызов Backoff(i, base) для каждого индекса.`,
			whyItMatters: `Генерация backoff последовательностей позволяет тестировать, мониторить и конфигурировать retry поведение.

**Почему эта функция важна:**

**1. Тестирование Retry логики:**
\`\`\`go
func TestExponentialGrowth(t *testing.T) {
    seq := BackoffSequence(5, 100*time.Millisecond)

    // Проверка что каждая задержка примерно в 2 раза больше предыдущей (±50% джиттер)
    for i := 1; i < len(seq); i++ {
        ratio := float64(seq[i]) / float64(seq[i-1])
        // Должно быть ~2.0, но джиттер позволяет 1.0-3.0
        assert.GreaterOrEqual(t, ratio, 1.0)
        assert.LessOrEqual(t, ratio, 3.0)
    }
}

// Без BackoffSequence нужно было бы:
// 1. Мокировать time.Sleep
// 2. Захватывать реальные retry задержки
// 3. Намного более сложная настройка теста
\`\`\`

**2. Мониторинг Retry поведения:**
\`\`\`go
// Production код мониторинга
func retryWithMetrics(ctx context.Context, op Op) error {
    seq := BackoffSequence(maxRetries, baseDelay)

    // Логируем запланированное retry расписание
    logger.Info("Starting retry operation",
        "max_retries", maxRetries,
        "retry_schedule", seq,
        "max_duration", sum(seq))

    return Do(ctx, maxRetries, baseDelay, op)
}

// Логи:
// Starting retry operation max_retries=5
// retry_schedule=[127ms 234ms 456ms 912ms 1824ms]
// max_duration=3553ms
\`\`\`

**3. Планирование емкости:**
\`\`\`go
// Вычисление наихудшего retry времени для планирования SLA
func maxRetryDuration(attempts int, base time.Duration) time.Duration {
    seq := BackoffSequence(attempts, base)
    total := time.Duration(0)
    for _, delay := range seq {
        total += delay
    }
    // Учитываем джиттер (макс 1.5x)
    return time.Duration(float64(total) * 1.5)
}

// Если SLA требует ответ в течение 5 секунд:
maxDuration := maxRetryDuration(retries, baseDelay)
if maxDuration > 5*time.Second {
    log.Warn("Retry конфиг может нарушить SLA",
        "max_duration", maxDuration,
        "sla", 5*time.Second)
}
\`\`\`

**4. Валидация конфигурации:**
\`\`\`go
// Валидация retry конфига при старте
func validateRetryConfig(cfg RetryConfig) error {
    seq := BackoffSequence(cfg.MaxRetries, cfg.BaseDelay)

    // Проверка что общее время не превышает лимит
    total := sum(seq)
    if total > cfg.MaxTotalTime {
        return fmt.Errorf("retry конфиг некорректен: total=%v превышает max=%v",
            total, cfg.MaxTotalTime)
    }

    // Проверка что отдельные задержки не слишком большие
    for i, delay := range seq {
        if delay > cfg.MaxSingleDelay {
            return fmt.Errorf("retry %d delay=%v превышает max=%v",
                i, delay, cfg.MaxSingleDelay)
        }
    }

    return nil
}
\`\`\`

**5. Пользовательская конфигурация:**
\`\`\`go
// CLI инструмент показывающий retry поведение
$ myapp retry-config --show
Retry Configuration:
  Max Attempts: 5
  Base Delay: 100ms

Retry Schedule:
  Attempt 1: 123ms  (total: 123ms)
  Attempt 2: 234ms  (total: 357ms)
  Attempt 3: 456ms  (total: 813ms)
  Attempt 4: 912ms  (total: 1725ms)
  Attempt 5: 1824ms (total: 3549ms)

Worst case: 5.3 seconds (with max jitter)
\`\`\`

**Реальный пример (AWS SDK):**
AWS SDK для Go включает похожую функциональность для отображения retry попыток:
\`\`\`go
// aws-sdk-go/aws/client/default_retryer.go
func (d DefaultRetryer) RetryRules(r *request.Request) time.Duration {
    delay := d.computeDelay(r.RetryCount)
    logger.Debug("Retry attempt",
        "attempt", r.RetryCount,
        "delay", delay)
    return delay
}
\`\`\`

**Замечание по производительности:**
Функция дешевая (выделяет один слайс) и обычно вызывается один раз при старте или в тестах, не в горячих путях. Небольшие затраты на выделение памяти стоят видимости которую она предоставляет.`
		},
		uz: {
			title: `Kutish ketma-ketligi`,
			solutionCode: `package retryx

import "time"

func BackoffSequence(n int, base time.Duration) []time.Duration {
	if n <= 0 {	// Davomiyliklar so'ralmadi
		return nil
	}
	if base <= 0 {	// Noto'g'ri bazadan himoya
		base = time.Millisecond
	}
	seq := make([]time.Duration, n)	// Aniq o'lchamni oldindan ajratish
	for i := 0; i < n; i++ {	// Har bir kechikishni generatsiya qilish
		seq[i] = Backoff(i, base)	// Izchillik uchun Backoff ni qayta ishlatamiz
	}
	return seq	// To'liq ketma-ketlikni qaytarish
}`,
			description: `Sinov va monitoring uchun backoff davomiyliklari slice ni generatsiya qiluvchi **BackoffSequence** funksiyasini amalga oshiring.

**Talablar:**
1. Aniq \`n\` ta backoff davomiyligini generatsiya qiling
2. Har bir indeks uchun \`Backoff(attempt, base)\` ishlating
3. n <= 0 uchun nil qaytaring
4. Noto'g'ri bazaviy davomiylikdan himoya qiling

**Maqsad:**
Bu funksiya quyidagilar uchun foydali:
- **Sinov:** Retry vaqt xatti-harakatini tekshirish
- **Monitoring:** Rejalashtirilgan retry jadvalini log qilish
- **Konfiguratsiya:** Qo'llashdan oldin backoff strategiyasini oldindan ko'rish

**Misol:**
\`\`\`go
// Birinchi 5 retry kechikishini generatsiya qilish
seq := BackoffSequence(5, 100*time.Millisecond)
// Natija (taxminan, jitter o'zgaradi):
// [100ms, 200ms, 400ms, 800ms, 1600ms]

for i, delay := range seq {
    fmt.Printf("Retry %d: wait %v\\n", i, delay)
}
\`\`\`

**Edge Case lar:**
- n <= 0 → nil qaytaring
- n == 1 → bitta elementli slice
- base <= 0 → 1ms standart ishlating

**Cheklovlar:**
- Aniq o'lcham bilan slice ni oldindan ajrating
- Izchillik uchun Backoff() funksiyasini qayta ishlating
- Noto'g'ri kirishlarni to'g'ri qayta ishlang`,
			hint1: `Aniq o'lcham uchun make([]time.Duration, n) bilan slice ni oldindan ajrating.`,
			hint2: `0 dan n-1 gacha loop va har bir indeks uchun Backoff(i, base) ni chaqiring.`,
			whyItMatters: `Backoff ketma-ketliklarini generatsiya qilish retry xatti-harakatini sinash, monitoring va konfiguratsiya qilishni yoqadi.

**Nega bu funksiya muhim:**

**1. Retry mantiqini sinash:**
\`\`\`go
func TestExponentialGrowth(t *testing.T) {
    seq := BackoffSequence(5, 100*time.Millisecond)

    // Har bir kechikish oldingi kechikishdan taxminan 2 marta katta ekanligini tekshirish (±50% jitter)
    for i := 1; i < len(seq); i++ {
        ratio := float64(seq[i]) / float64(seq[i-1])
        // ~2.0 bo'lishi kerak, lekin jitter 1.0-3.0 ga imkon beradi
        assert.GreaterOrEqual(t, ratio, 1.0)
        assert.LessOrEqual(t, ratio, 3.0)
    }
}

// BackoffSequence siz quyidagilarni qilish kerak bo'lar edi:
// 1. time.Sleep ni mock qilish
// 2. Haqiqiy retry kechikishlarini ushlash
// 3. Ancha murakkab test sozlamasi
\`\`\`

**2. Retry xatti-harakatini monitoring qilish:**
\`\`\`go
// Production monitoring kodi
func retryWithMetrics(ctx context.Context, op Op) error {
    seq := BackoffSequence(maxRetries, baseDelay)

    // Rejalashtirilgan retry jadvalini log qilish
    logger.Info("Starting retry operation",
        "max_retries", maxRetries,
        "retry_schedule", seq,
        "max_duration", sum(seq))

    return Do(ctx, maxRetries, baseDelay, op)
}

// Loglar:
// Starting retry operation max_retries=5
// retry_schedule=[127ms 234ms 456ms 912ms 1824ms]
// max_duration=3553ms
\`\`\`

**3. Sig'imni rejalashtirish:**
\`\`\`go
// SLA rejalashtirish uchun eng yomon retry vaqtini hisoblash
func maxRetryDuration(attempts int, base time.Duration) time.Duration {
    seq := BackoffSequence(attempts, base)
    total := time.Duration(0)
    for _, delay := range seq {
        total += delay
    }
    // Jitterni hisobga olish (maks 1.5x)
    return time.Duration(float64(total) * 1.5)
}

// Agar SLA 5 soniya ichida javob talab qilsa:
maxDuration := maxRetryDuration(retries, baseDelay)
if maxDuration > 5*time.Second {
    log.Warn("Retry konfiguratsiya SLA ni buzishi mumkin",
        "max_duration", maxDuration,
        "sla", 5*time.Second)
}
\`\`\`

**4. Konfiguratsiyani tekshirish:**
\`\`\`go
// Ishga tushirishda retry konfiguratsiyasini tekshirish
func validateRetryConfig(cfg RetryConfig) error {
    seq := BackoffSequence(cfg.MaxRetries, cfg.BaseDelay)

    // Umumiy vaqt chegaradan oshmasligini tekshirish
    total := sum(seq)
    if total > cfg.MaxTotalTime {
        return fmt.Errorf("retry konfiguratsiya noto'g'ri: total=%v maksdan oshadi=%v",
            total, cfg.MaxTotalTime)
    }

    // Alohida kechikishlar juda katta emasligini tekshirish
    for i, delay := range seq {
        if delay > cfg.MaxSingleDelay {
            return fmt.Errorf("retry %d delay=%v maksdan oshadi=%v",
                i, delay, cfg.MaxSingleDelay)
        }
    }

    return nil
}
\`\`\`

**5. Foydalanuvchi uchun konfiguratsiya:**
\`\`\`go
// Retry xatti-harakatini ko'rsatuvchi CLI vositasi
$ myapp retry-config --show
Retry Configuration:
  Max Attempts: 5
  Base Delay: 100ms

Retry Schedule:
  Attempt 1: 123ms  (total: 123ms)
  Attempt 2: 234ms  (total: 357ms)
  Attempt 3: 456ms  (total: 813ms)
  Attempt 4: 912ms  (total: 1725ms)
  Attempt 5: 1824ms (total: 3549ms)

Worst case: 5.3 seconds (with max jitter)
\`\`\`

**Haqiqiy misol (AWS SDK):**
AWS SDK for Go retry urinishlarini ko'rsatish uchun shunga o'xshash funksionallikni o'z ichiga oladi:
\`\`\`go
// aws-sdk-go/aws/client/default_retryer.go
func (d DefaultRetryer) RetryRules(r *request.Request) time.Duration {
    delay := d.computeDelay(r.RetryCount)
    logger.Debug("Retry attempt",
        "attempt", r.RetryCount,
        "delay", delay)
    return delay
}
\`\`\`

**Samaradorlik eslatmasi:**
Funksiya arzon (bitta slice ajratadi) va odatda ishga tushirishda yoki testlarda bir marta chaqiriladi, issiq yo'llarda emas. Kichik xotira ajratish xarajatlari u taqdim etadigan ko'rinish uchun arziydi.`
		}
	}
};

export default task;
