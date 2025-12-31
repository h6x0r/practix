import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-do',
	title: 'Retry Operation with Backoff',
	difficulty: 'medium',	tags: ['go', 'retry', 'context', 'backoff'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Do** function that retries an operation up to n times with exponential backoff.

**Requirements:**
1. Retry operation up to \`n\` times
2. Return immediately on success (err == nil)
3. Apply exponential backoff between retries using \`Backoff()\`
4. Respect context cancellation (\`ctx.Done()\`)
5. Use \`SleepContext()\` for context-aware waiting
6. Return last error when all retries exhausted

**Type Definition:**
\`\`\`go
type Op func(context.Context) error
\`\`\`

**Retry Logic:**
\`\`\`
Attempt 1 → fail → wait Backoff(0, base)
Attempt 2 → fail → wait Backoff(1, base)
Attempt 3 → success → return nil
\`\`\`

**Example:**
\`\`\`go
attempts := 0
op := func(ctx context.Context) error {
    attempts++
    if attempts < 3 {
        return errors.New("temporary error")
    }
    return nil  // Success on 3rd try
}

err := Do(ctx, 5, 100*time.Millisecond, op)
// Tries 3 times, succeeds, returns nil
// Total time: ~100ms + ~200ms = ~300ms
\`\`\`

**Edge Cases:**
- Context canceled during wait → return ctx.Err()
- Context canceled before retry → return ctx.Err()
- Operation succeeds immediately → return nil (no retries)
- All retries fail → return last error

**Constraints:**
- Check context before each attempt
- Don't sleep after last failed attempt
- Handle nil operation gracefully`,
	initialCode: `package retryx

import (
	"context"
	"time"
)

type Op func(context.Context) error

// TODO: Implement Do function
// Retry operation n times with backoff
func Do($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package retryx

import (
	"context"
	"time"
)

type Op func(context.Context) error

func Do(ctx context.Context, n int, base time.Duration, op Op) error {
	if n <= 0 {	// No attempts requested
		return nil
	}
	var lastErr error	// Track last error for final return
	for attempt := 0; attempt < n; attempt++ {	// Iterate through allowed attempts
		if ctx.Err() != nil {	// Context already canceled
			return ctx.Err()
		}
		if err := op(ctx); err == nil {	// Operation succeeded
			return nil	// Return immediately on success
		} else {
			lastErr = err	// Remember error for retry/return
		}
		if attempt == n-1 {	// Last attempt, don't sleep
			break
		}
		if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Wait with context respect
			return err	// Context canceled during sleep
		}
	}
	return lastErr	// All retries exhausted, return last error
}`,
	testCode: `package retryx

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestDoSuccessOnFirstAttempt(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return nil
	}
	err := Do(ctx, 5, 100*time.Millisecond, op)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 1 {
		t.Errorf("expected 1 call, got %d", callCount)
	}
}

func TestDoSuccessOnThirdAttempt(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		if callCount < 3 {
			return errors.New("temporary error")
		}
		return nil
	}
	err := Do(ctx, 5, 10*time.Millisecond, op)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestDoAllRetriesFail(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("persistent error")
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return expectedErr
	}
	err := Do(ctx, 3, 10*time.Millisecond, op)
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

func TestDoWithZeroAttempts(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("should not be called")
	}
	err := Do(ctx, 0, 100*time.Millisecond, op)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 0 {
		t.Errorf("expected 0 calls, got %d", callCount)
	}
}

func TestDoWithNegativeAttempts(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("should not be called")
	}
	err := Do(ctx, -5, 100*time.Millisecond, op)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if callCount != 0 {
		t.Errorf("expected 0 calls, got %d", callCount)
	}
}

func TestDoWithCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	op := func(ctx context.Context) error {
		return errors.New("should fail")
	}
	err := Do(ctx, 5, 100*time.Millisecond, op)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestDoContextCanceledDuringRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("temporary error")
	}
	err := Do(ctx, 10, 100*time.Millisecond, op)
	if err == nil {
		t.Errorf("expected error, got nil")
	}
	if callCount > 3 {
		t.Errorf("expected fewer calls due to timeout, got %d", callCount)
	}
}

func TestDoNoSleepAfterLastAttempt(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	op := func(ctx context.Context) error {
		return errors.New("fail")
	}
	Do(ctx, 3, 50*time.Millisecond, op)
	elapsed := time.Since(start)
	// Should sleep only between first-second and second-third attempts
	// Not after the third (last) attempt
	if elapsed > 200*time.Millisecond {
		t.Errorf("slept after last attempt, elapsed: %v", elapsed)
	}
}

func TestDoRespectsBackoff(t *testing.T) {
	ctx := context.Background()
	start := time.Now()
	callCount := 0
	op := func(ctx context.Context) error {
		callCount++
		return errors.New("fail")
	}
	Do(ctx, 3, 50*time.Millisecond, op)
	elapsed := time.Since(start)
	// Should have exponential backoff between retries
	// At least some delay should occur
	if elapsed < 50*time.Millisecond {
		t.Errorf("expected some backoff delay, got %v", elapsed)
	}
}

func TestDoWithNilOperation(t *testing.T) {
	ctx := context.Background()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for nil operation")
		}
	}()
	Do(ctx, 3, 100*time.Millisecond, nil)
}`,
			hint1: `Check ctx.Err() before each attempt to detect early cancellation.`,
			hint2: `Use SleepContext to wait between retries while respecting context.`,
			whyItMatters: `Retry logic with backoff is fundamental to building resilient distributed systems.

**Why Retry Operations:**
Distributed systems are inherently unreliable:
- **Network blips:** Temporary packet loss (1-5% of requests)
- **Service restarts:** Rolling deployments cause brief unavailability
- **Rate limits:** APIs throttle excessive requests
- **Database locks:** Concurrent transactions can deadlock

**Without Retries:**
\`\`\`go
// 5% network error rate = 95% success
// 10 API calls = 0.95^10 = 59.9% overall success
// Nearly half of all multi-step operations fail!
\`\`\`

**With Retries (3 attempts):**
\`\`\`go
// Single request success = 1 - 0.05^3 = 99.9875%
// 10 API calls = 0.999875^10 = 98.75% overall success
// 40x improvement in reliability!
\`\`\`

**Production Pattern:**
\`\`\`go
// Without retry
resp, err := http.Get(url)
if err != nil {
    return err  // Fails 5% of the time
}

// With retry
var resp *http.Response
err := Do(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
    r, err := http.Get(url)
    if err == nil {
        resp = r
    }
    return err
})
// Fails 0.0125% of the time
\`\`\`

**Real-World Examples:**
- **AWS SDK:** Retries all API calls (default: 3 attempts)
- **gRPC:** Built-in retry policy configuration
- **Kubernetes:** Retries failed API server requests
- **Database drivers:** Connection retry on temporary failures

**Context Integration:**
The \`ctx\` parameter is crucial:
\`\`\`go
// Request timeout: 5 seconds
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// Will stop retrying when timeout exceeded
err := Do(ctx, 10, 100*time.Millisecond, fetchData)
\`\`\`

**Cost-Benefit:**
- Small investment: ~20 lines of code
- Huge payoff: 40x reliability improvement
- Industry standard: Every production system uses retries`,	order: 1,
	translations: {
		ru: {
			title: 'Обёртка повторных попыток',
			solutionCode: `package retryx

import (
	"context"
	"time"
)

type Op func(context.Context) error

func Do(ctx context.Context, n int, base time.Duration, op Op) error {
	if n <= 0 {	// Попытки не запрошены
		return nil
	}
	var lastErr error	// Отслеживаем последнюю ошибку для возврата
	for attempt := 0; attempt < n; attempt++ {	// Проходим по разрешённым попыткам
		if ctx.Err() != nil {	// Контекст уже отменён
			return ctx.Err()
		}
		if err := op(ctx); err == nil {	// Операция успешна
			return nil	// Возвращаем сразу при успехе
		} else {
			lastErr = err	// Запоминаем ошибку для retry/возврата
		}
		if attempt == n-1 {	// Последняя попытка, не спим
			break
		}
		if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Ждём с учётом контекста
			return err	// Контекст отменён во время сна
		}
	}
	return lastErr	// Все retry исчерпаны, возвращаем последнюю ошибку
}`,
			description: `Реализуйте функцию **Do**, которая повторяет операцию до n раз с экспоненциальной задержкой.

**Требования:**
1. Повторите операцию до \`n\` раз
2. Верните результат сразу при успехе (err == nil)
3. Примените экспоненциальную задержку между retry используя \`Backoff()\`
4. Учитывайте отмену контекста (\`ctx.Done()\`)
5. Используйте \`SleepContext()\` для ожидания с учетом контекста
6. Верните последнюю ошибку когда все retry исчерпаны

**Определение типа:**
\`\`\`go
type Op func(context.Context) error
\`\`\`

**Логика Retry:**
\`\`\`
Попытка 1 → fail → ждем Backoff(0, base)
Попытка 2 → fail → ждем Backoff(1, base)
Попытка 3 → success → return nil
\`\`\`

**Пример:**
\`\`\`go
attempts := 0
op := func(ctx context.Context) error {
    attempts++
    if attempts < 3 {
        return errors.New("temporary error")
    }
    return nil  // Успех на 3-й попытке
}

err := Do(ctx, 5, 100*time.Millisecond, op)
// Пробует 3 раза, успех, возвращает nil
// Общее время: ~100мс + ~200мс = ~300мс
\`\`\`

**Edge Cases:**
- Контекст отменен во время ожидания → вернуть ctx.Err()
- Контекст отменен перед retry → вернуть ctx.Err()
- Операция успешна сразу → вернуть nil (без retry)
- Все retry упали → вернуть последнюю ошибку`,
			hint1: `Проверяйте ctx.Err() перед каждой попыткой для раннего обнаружения отмены.`,
			hint2: `Используйте SleepContext для ожидания между retry с учетом контекста.`,
			whyItMatters: `Retry логика с backoff - основа построения устойчивых распределенных систем.

**Почему Retry важен:**
Распределенные системы ненадежны по своей природе:
- **Сетевые сбои:** Временная потеря пакетов (1-5% запросов)
- **Перезапуск сервисов:** Rolling deployment вызывает кратковременную недоступность
- **Rate limits:** API ограничивают чрезмерные запросы
- **Блокировки БД:** Конкурентные транзакции могут deadlock

**Без Retry:**
\`\`\`go
// 5% сетевых ошибок = 95% успех
// 10 API вызовов = 0.95^10 = 59.9% общий успех
// Почти половина всех многошаговых операций падает!
\`\`\`

**С Retry (3 попытки):**
\`\`\`go
// Успех одного запроса = 1 - 0.05^3 = 99.9875%
// 10 API вызовов = 0.999875^10 = 98.75% общий успех
// Улучшение надежности в 40 раз!
\`\`\`

**Продакшен паттерн:**
\`\`\`go
// Без retry
resp, err := http.Get(url)
if err != nil {
    return err  // Падает 5% времени
}

// С retry
var resp *http.Response
err := Do(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
    r, err := http.Get(url)
    if err == nil {
        resp = r
    }
    return err
})
// Падает 0.0125% времени
\`\`\`

**Практические примеры:**
- **AWS SDK:** Retry всех API вызовов (default: 3 попытки)
- **gRPC:** Встроенная конфигурация retry policy
- **Kubernetes:** Retry упавших API server запросов
- **Database драйверы:** Connection retry при временных сбоях

**Интеграция с Context:**
Параметр \`ctx\` критически важен:
\`\`\`go
// Timeout запроса: 5 секунд
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// Остановит retry когда timeout превышен
err := Do(ctx, 10, 100*time.Millisecond, fetchData)
\`\`\`

**Соотношение затрат и выгод:**
- Небольшие затраты: ~20 строк кода
- Огромная отдача: улучшение надежности в 40 раз
- Индустриальный стандарт: Каждая production система использует retry`
		},
		uz: {
			title: `Qayta urinish wrapper`,
			solutionCode: `package retryx

import (
	"context"
	"time"
)

type Op func(context.Context) error

func Do(ctx context.Context, n int, base time.Duration, op Op) error {
	if n <= 0 {	// Urinishlar so'ralmadi
		return nil
	}
	var lastErr error	// Yakuniy qaytarish uchun oxirgi xatoni kuzatamiz
	for attempt := 0; attempt < n; attempt++ {	// Ruxsat etilgan urinishlarni aylanamiz
		if ctx.Err() != nil {	// Kontekst allaqachon bekor qilindi
			return ctx.Err()
		}
		if err := op(ctx); err == nil {	// Operatsiya muvaffaqiyatli
			return nil	// Muvaffaqiyatda darhol qaytamiz
		} else {
			lastErr = err	// Retry/qaytarish uchun xatoni eslab qolamiz
		}
		if attempt == n-1 {	// Oxirgi urinish, uxlamaymiz
			break
		}
		if err := SleepContext(ctx, Backoff(attempt, base)); err != nil {	// Kontekstni hisobga olib kutamiz
			return err	// Uxlash paytida kontekst bekor qilindi
		}
	}
	return lastErr	// Barcha retrylar tugadi, oxirgi xatoni qaytaramiz
}`,
			description: `Operatsiyani eksponensial backoff bilan n martagacha qayta urinib ko'radigan **Do** funksiyasini amalga oshiring.

**Talablar:**
1. Operatsiyani \`n\` martagacha qayta urining
2. Muvaffaqiyatda darhol qaytaring (err == nil)
3. Retrylar orasida \`Backoff()\` ishlatib eksponensial backoff qo'llang
4. Kontekst bekor qilishni hurmat qiling (\`ctx.Done()\`)
5. Kontekstni hisobga olgan kutish uchun \`SleepContext()\` ishlating
6. Barcha retrylar tugaganda oxirgi xatoni qaytaring

**Tip ta'rifi:**
\`\`\`go
type Op func(context.Context) error
\`\`\`

**Retry mantiqi:**
\`\`\`
Urinish 1 → muvaffaqiyatsiz → Backoff(0, base) kuting
Urinish 2 → muvaffaqiyatsiz → Backoff(1, base) kuting
Urinish 3 → muvaffaqiyat → nil qaytaring
\`\`\`

**Edge Case lar:**
- Kutish paytida kontekst bekor qilindi → ctx.Err() qaytaring
- Retrydan oldin kontekst bekor qilindi → ctx.Err() qaytaring
- Operatsiya darhol muvaffaqiyatli → nil qaytaring (retrylar yo'q)
- Barcha retrylar muvaffaqiyatsiz → oxirgi xatoni qaytaring

**Cheklovlar:**
- Har bir urinishdan oldin kontekstni tekshiring
- Oxirgi muvaffaqiyatsiz urinishdan keyin uxlamang
- nil operatsiyani to'g'ri qayta ishlang`,
			hint1: `Erta bekor qilishni aniqlash uchun har bir urinishdan oldin ctx.Err() ni tekshiring.`,
			hint2: `Kontekstni hisobga olib retrylar orasida kutish uchun SleepContext ishlating.`,
			whyItMatters: `Backoff bilan retry mantiqi chidamli taqsimlangan tizimlar qurishning asosidir.

**Nega Retry Operatsiyalari:**
Taqsimlangan tizimlar tabiatan ishonchsizdir:
- **Tarmoq uzilishlari:** Vaqtinchalik paket yo'qolishi (so'rovlarning 1-5%)
- **Xizmat qayta ishga tushishi:** Rolling deployment qisqa muddatli mavjud emaslikni keltirib chiqaradi
- **Rate limitlar:** APIlar ortiqcha so'rovlarni cheklaydi
- **Database qulflari:** Parallel tranzaksiyalar deadlock bo'lishi mumkin

**Retrylarsiz:**
\`\`\`go
// 5% tarmoq xato darajasi = 95% muvaffaqiyat
// 10 API chaqiruv = 0.95^10 = 59.9% umumiy muvaffaqiyat
// Deyarli barcha ko'p bosqichli operatsiyalarning yarmi muvaffaqiyatsiz!
\`\`\`

**Retry bilan (3 urinish):**
\`\`\`go
// Bitta so'rov muvaffaqiyati = 1 - 0.05^3 = 99.9875%
// 10 API chaqiruv = 0.999875^10 = 98.75% umumiy muvaffaqiyat
// Ishonchlilikni 40 marta yaxshilash!
\`\`\`

**Ishlab chiqarish patterni:**
\`\`\`go
// Retrysiz
resp, err := http.Get(url)
if err != nil {
    return err  // Vaqtning 5% muvaffaqiyatsiz
}

// Retry bilan
var resp *http.Response
err := Do(ctx, 3, 100*time.Millisecond, func(ctx context.Context) error {
    r, err := http.Get(url)
    if err == nil {
        resp = r
    }
    return err
})
// Vaqtning 0.0125% muvaffaqiyatsiz
\`\`\`

**Amaliy misollar:**
- **AWS SDK:** Barcha API chaqiruvlarini retry qiladi (standart: 3 urinish)
- **gRPC:** O'rnatilgan retry policy konfiguratsiyasi
- **Kubernetes:** Muvaffaqiyatsiz API server so'rovlarini retry qiladi
- **Database drayverlar:** Vaqtinchalik nosozliklarda ulanish retry

**Context Integratsiyasi:**
\`ctx\` parametri juda muhim:
\`\`\`go
// So'rov timeout: 5 soniya
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// Timeout oshganda retryni to'xtatadi
err := Do(ctx, 10, 100*time.Millisecond, fetchData)
\`\`\`

**Xarajat-Foyda Nisbati:**
- Kichik sarmoya: ~20 qator kod
- Katta foyda: 40 marta ishonchlilik yaxshilanishi
- Sanoat standarti: Har bir production tizim retry ishlatadi`
		}
	}
};

export default task;
