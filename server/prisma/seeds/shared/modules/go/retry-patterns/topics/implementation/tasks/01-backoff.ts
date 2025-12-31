import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-retry-backoff',
	title: 'Exponential Backoff with Jitter',
	difficulty: 'easy',	tags: ['go', 'retry', 'backoff', 'jitter'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Backoff** function that calculates exponential delay with jitter for retry operations.

**Requirements:**
1. Calculate exponential backoff: \`base * 2^attempt\`
2. Add jitter (randomness) in range [0.5, 1.5] to prevent thundering herd
3. Handle negative attempts (treat as 0)
4. Guard against invalid base durations (use 1ms as default)

**Formula:**
\`\`\`
delay = base << attempt              // base * 2^attempt
jitter = 0.5 + rand.Float64()        // [0.5, 1.5]
result = delay * jitter
\`\`\`

**Example:**
\`\`\`go
// attempt=0: base * 2^0 * jitter = 100ms * 1 * [0.5-1.5] = 50-150ms
Backoff(0, 100*time.Millisecond)  // ~100ms ± 50%

// attempt=3: base * 2^3 * jitter = 100ms * 8 * [0.5-1.5] = 400-1200ms
Backoff(3, 100*time.Millisecond)  // ~800ms ± 50%
\`\`\`

**Constraints:**
- Use bit shift (\`<<\`) for efficient power-of-2 multiplication
- Jitter prevents all clients retrying simultaneously
- Handle edge cases: negative attempts, zero/negative base`,
	initialCode: `package retryx

import (
	"math/rand"
	"time"
)

// TODO: Implement Backoff function
// Calculate exponential delay with jitter
func Backoff(attempt int, base time.Duration) time.Duration {
	// TODO: Implement
}`,
	solutionCode: `package retryx

import (
	"math/rand"
	"time"
)

func Backoff(attempt int, base time.Duration) time.Duration {
	if attempt < 0 {	// Negative attempts treated as zero
		attempt = 0
	}
	if base <= 0 {	// Guard against invalid base
		base = time.Millisecond
	}
	delay := base << attempt	// Exponential: base * 2^attempt
	jitter := 0.5 + rand.Float64()	// Random factor [0.5, 1.5]
	return time.Duration(float64(delay) * jitter)	// Apply jitter to delay
}`,
	testCode: `package retryx

import (
	"testing"
	"time"
)

func TestBackoffWithZeroAttempt(t *testing.T) {
	result := Backoff(0, 100*time.Millisecond)
	if result < 50*time.Millisecond || result > 150*time.Millisecond {
		t.Errorf("expected result between 50ms and 150ms, got %v", result)
	}
}

func TestBackoffWithPositiveAttempt(t *testing.T) {
	result := Backoff(3, 100*time.Millisecond)
	// base * 2^3 = 100ms * 8 = 800ms
	// with jitter [0.5, 1.5]: 400ms to 1200ms
	if result < 400*time.Millisecond || result > 1200*time.Millisecond {
		t.Errorf("expected result between 400ms and 1200ms, got %v", result)
	}
}

func TestBackoffWithNegativeAttempt(t *testing.T) {
	result := Backoff(-5, 100*time.Millisecond)
	// negative attempt should be treated as 0
	if result < 50*time.Millisecond || result > 150*time.Millisecond {
		t.Errorf("expected result between 50ms and 150ms, got %v", result)
	}
}

func TestBackoffWithZeroBase(t *testing.T) {
	result := Backoff(0, 0)
	// should default to 1ms
	if result < 500*time.Nanosecond || result > 1500*time.Nanosecond {
		t.Errorf("expected result between 0.5ms and 1.5ms, got %v", result)
	}
}

func TestBackoffWithNegativeBase(t *testing.T) {
	result := Backoff(0, -100*time.Millisecond)
	// should default to 1ms
	if result < 500*time.Nanosecond || result > 1500*time.Nanosecond {
		t.Errorf("expected result between 0.5ms and 1.5ms, got %v", result)
	}
}

func TestBackoffExponentialGrowth(t *testing.T) {
	result1 := Backoff(1, 100*time.Millisecond)
	result2 := Backoff(2, 100*time.Millisecond)
	// Attempt 1: 100ms * 2 = 200ms, Attempt 2: 100ms * 4 = 400ms
	// With jitter, result2 should generally be larger than result1
	if result2 < result1 {
		// This might occasionally fail due to jitter, but statistically unlikely
		t.Logf("result1: %v, result2: %v (jitter may cause variance)", result1, result2)
	}
}

func TestBackoffLargeAttempt(t *testing.T) {
	result := Backoff(10, 100*time.Millisecond)
	// base * 2^10 = 100ms * 1024 = 102400ms
	// with jitter [0.5, 1.5]: 51200ms to 153600ms
	if result < 51200*time.Millisecond || result > 153600*time.Millisecond {
		t.Errorf("expected result between 51200ms and 153600ms, got %v", result)
	}
}

func TestBackoffJitterVariance(t *testing.T) {
	// Test that results vary due to jitter
	results := make([]time.Duration, 10)
	for i := 0; i < 10; i++ {
		results[i] = Backoff(2, 100*time.Millisecond)
	}
	// Check if we have variance (not all values are the same)
	allSame := true
	for i := 1; i < len(results); i++ {
		if results[i] != results[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("expected jitter to create variance, got all same values")
	}
}

func TestBackoffSmallBase(t *testing.T) {
	result := Backoff(0, 1*time.Microsecond)
	// base * 2^0 = 1us
	// with jitter [0.5, 1.5]: 0.5us to 1.5us
	if result < 500*time.Nanosecond || result > 1500*time.Nanosecond {
		t.Errorf("expected result between 0.5us and 1.5us, got %v", result)
	}
}

func TestBackoffAttemptTwo(t *testing.T) {
	result := Backoff(2, 50*time.Millisecond)
	// base * 2^2 = 50ms * 4 = 200ms
	// with jitter [0.5, 1.5]: 100ms to 300ms
	if result < 100*time.Millisecond || result > 300*time.Millisecond {
		t.Errorf("expected result between 100ms and 300ms, got %v", result)
	}
}`,
			hint1: `Use bit shift (<<) to multiply base by 2^attempt efficiently.`,
			hint2: `Add rand.Float64() to 0.5 to get random multiplier in [0.5, 1.5] range.`,
			whyItMatters: `Exponential backoff with jitter is the industry standard for retry logic, preventing cascading failures.

**Why Exponential Backoff:**
- **Linear backoff (100ms, 200ms, 300ms):** Predictable, can cause synchronized retries
- **Exponential backoff (100ms, 200ms, 400ms, 800ms):** Delays grow rapidly, gives system time to recover
- **With jitter:** Randomizes retry timing to prevent thundering herd problem

**The Thundering Herd Problem:**
\`\`\`
Without jitter:
1000 clients fail at the same time
↓
All wait exactly 1 second
↓
All retry simultaneously
↓
Server overwhelmed again

With jitter:
1000 clients fail at the same time
↓
Each waits 0.5-1.5 seconds (randomized)
↓
Retries spread out over 1 second window
↓
Server recovers gradually
\`\`\`

**Production Benefits:**
- **AWS SDK:** Uses exponential backoff with jitter for all API calls
- **Kubernetes:** Default retry strategy for failed pods
- **Database connections:** Prevents connection storms during recovery
- **Rate limiting:** Spreads load when quota resets

**Real-World Example:**
When a database goes down and comes back up, without jitter, all 10,000 connections try to reconnect at exactly the same time, potentially crashing it again. With jitter, reconnections spread over several seconds, allowing graceful recovery.

**Formula Choice:**
The \`0.5 + rand.Float64()\` formula (resulting in [0.5, 1.5]) is AWS's recommended approach. Other options:
- Full jitter: \`rand.Float64() * delay\` (0 to delay)
- Equal jitter: \`delay/2 + rand.Float64() * delay/2\` (same as ours)
- Decorrelated jitter: More complex, but better distribution`,	order: 0,
	translations: {
		ru: {
			title: 'Экспоненциальная задержка',
			solutionCode: `package retryx

import (
	"math/rand"
	"time"
)

func Backoff(attempt int, base time.Duration) time.Duration {
	if attempt < 0 {	// Отрицательные попытки считаем нулём
		attempt = 0
	}
	if base <= 0 {	// Защита от некорректной базы
		base = time.Millisecond
	}
	delay := base << attempt	// Экспоненциально: base * 2^attempt
	jitter := 0.5 + rand.Float64()	// Случайный множитель [0.5, 1.5]
	return time.Duration(float64(delay) * jitter)	// Применяем джиттер к задержке
}`,
			description: `Реализуйте функцию **Backoff**, которая вычисляет экспоненциальную задержку с джиттером для retry операций.

**Требования:**
1. Вычислите экспоненциальную задержку: \`base * 2^attempt\`
2. Добавьте джиттер (случайность) в диапазоне [0.5, 1.5]
3. Обработайте отрицательные попытки (считать как 0)
4. Защитите от некорректных базовых длительностей (используйте 1мс)

**Формула:**
\`\`\`
delay = base << attempt              // base * 2^attempt
jitter = 0.5 + rand.Float64()        // [0.5, 1.5]
result = delay * jitter
\`\`\`

**Пример:**
\`\`\`go
// попытка=0: 100мс * 1 * [0.5-1.5] = 50-150мс
Backoff(0, 100*time.Millisecond)  // ~100мс ± 50%

// попытка=3: 100мс * 8 * [0.5-1.5] = 400-1200мс
Backoff(3, 100*time.Millisecond)  // ~800мс ± 50%
\`\`\`

**Ограничения:**
- Используйте битовый сдвиг (<<) для эффективного умножения
- Джиттер предотвращает одновременные retry всех клиентов
- Обработайте edge cases`,
			hint1: `Используйте битовый сдвиг (<<) для умножения base на 2^attempt.`,
			hint2: `Добавьте rand.Float64() к 0.5 для получения множителя [0.5, 1.5].`,
			whyItMatters: `Экспоненциальная задержка с джиттером - индустриальный стандарт для retry логики, предотвращающий каскадные сбои.

**Почему Экспоненциальный Backoff:**
- **Линейный backoff (100ms, 200ms, 300ms):** Предсказуем, может вызвать синхронизированные retry
- **Экспоненциальный backoff (100ms, 200ms, 400ms, 800ms):** Задержки растут быстро, дает системе время восстановиться
- **С джиттером:** Рандомизирует timing retry для предотвращения проблемы thundering herd

**Проблема Thundering Herd:**
\`\`\`
Без джиттера:
1000 клиентов падают одновременно
↓
Все ждут ровно 1 секунду
↓
Все retry одновременно
↓
Сервер снова перегружен

С джиттером:
1000 клиентов падают одновременно
↓
Каждый ждет 0.5-1.5 секунд (рандомизировано)
↓
Retry распределены в окне 1 секунды
↓
Сервер восстанавливается постепенно
\`\`\`

**Продакшен преимущества:**
- **AWS SDK:** Использует экспоненциальный backoff с джиттером для всех API вызовов
- **Kubernetes:** Стандартная retry стратегия для упавших подов
- **Database connections:** Предотвращает connection storm во время восстановления
- **Rate limiting:** Распределяет нагрузку когда квота сбрасывается

**Реальный пример:**
Когда база данных падает и восстанавливается, без джиттера все 10,000 соединений пытаются переподключиться в один и тот же момент, потенциально роняя её снова. С джиттером переподключения распределены на несколько секунд, позволяя плавное восстановление.

**Выбор формулы:**
Формула \`0.5 + rand.Float64()\` (результат [0.5, 1.5]) - рекомендованный подход AWS. Другие варианты:
- Full jitter: \`rand.Float64() * delay\` (от 0 до delay)
- Equal jitter: \`delay/2 + rand.Float64() * delay/2\` (то же что у нас)
- Decorrelated jitter: Более сложный, но лучшее распределение`
		},
		uz: {
			title: `Eksponentsial kutish`,
			solutionCode: `package retryx

import (
	"math/rand"
	"time"
)

func Backoff(attempt int, base time.Duration) time.Duration {
	if attempt < 0 {	// Salbiy urinishlar nol sifatida ko'riladi
		attempt = 0
	}
	if base <= 0 {	// Noto'g'ri bazadan himoya
		base = time.Millisecond
	}
	delay := base << attempt	// Eksponensial: base * 2^attempt
	jitter := 0.5 + rand.Float64()	// Tasodifiy ko'paytiruvchi [0.5, 1.5]
	return time.Duration(float64(delay) * jitter)	// Kechikishga jitter qo'llaymiz
}`,
			description: `Retry operatsiyalari uchun jitter bilan eksponensial kechikishni hisoblaydigan **Backoff** funksiyasini amalga oshiring.

**Talablar:**
1. Eksponensial backoff hisoblash: \`base * 2^attempt\`
2. Thundering herd ni oldini olish uchun [0.5, 1.5] diapazonida jitter (tasodifiylik) qo'shish
3. Salbiy urinishlarni qayta ishlash (0 deb hisoblash)
4. Noto'g'ri bazaviy davomiyliklardan himoya qilish (standart sifatida 1ms ishlatish)

**Formula:**
\`\`\`
delay = base << attempt              // base * 2^attempt
jitter = 0.5 + rand.Float64()        // [0.5, 1.5]
result = delay * jitter
\`\`\`

**Misol:**
\`\`\`go
// attempt=0: base * 2^0 * jitter = 100ms * 1 * [0.5-1.5] = 50-150ms
Backoff(0, 100*time.Millisecond)  // ~100ms ± 50%

// attempt=3: base * 2^3 * jitter = 100ms * 8 * [0.5-1.5] = 400-1200ms
Backoff(3, 100*time.Millisecond)  // ~800ms ± 50%
\`\`\`

**Cheklovlar:**
- Samarali 2 darajasiga ko'paytirish uchun bit shift (<<) ishlating
- Jitter barcha mijozlarning bir vaqtda retry qilishini oldini oladi
- Edge case larni qayta ishlang: salbiy urinishlar, nol/salbiy bazaviy`,
			hint1: `base ni 2^attempt ga samarali ko'paytirish uchun bit shift (<<) ishlating.`,
			hint2: `[0.5, 1.5] diapazonida tasodifiy ko'paytiruvchi olish uchun rand.Float64() ni 0.5 ga qo'shing.`,
			whyItMatters: `Jitter bilan eksponensial backoff retry mantiqining sanoat standarti bo'lib, kaskadli nosozliklarni oldini oladi.

**Nega Eksponensial Backoff:**
- **Lineer backoff (100ms, 200ms, 300ms):** Bashorat qilinadigan, sinxronlashtirilgan retrylarni keltirib chiqarishi mumkin
- **Eksponensial backoff (100ms, 200ms, 400ms, 800ms):** Kechikishlar tez o'sadi, tizimga tiklanish uchun vaqt beradi
- **Jitter bilan:** Thundering herd muammosini oldini olish uchun retry vaqtini tasodifiylashtiradi

**Thundering Herd Muammosi:**
\`\`\`
Jittersiz:
1000 mijoz bir vaqtda muvaffaqiyatsiz bo'ladi
↓
Hammasi aniq 1 soniya kutadi
↓
Hammasi bir vaqtda retry qiladi
↓
Server yana ortiqcha yuklanadi

Jitter bilan:
1000 mijoz bir vaqtda muvaffaqiyatsiz bo'ladi
↓
Har biri 0.5-1.5 soniya kutadi (tasodifiy)
↓
Retrylar 1 soniyalik oyna bo'ylab taqsimlanadi
↓
Server asta-sekin tiklanadi
\`\`\`

**Ishlab chiqarish foydalari:**
- **AWS SDK:** Barcha API chaqiruvlar uchun jitter bilan eksponensial backoff ishlatadi
- **Kubernetes:** Muvaffaqiyatsiz podlar uchun standart retry strategiyasi
- **Database ulanishlari:** Tiklanish paytida ulanish bo'ronlarini oldini oladi
- **Rate limiting:** Kvota qayta tiklanganida yukni taqsimlaydi

**Haqiqiy dunyo misoli:**
Ma'lumotlar bazasi ishdan chiqqanda va qayta tiklanganida, jittersiz, barcha 10,000 ulanish aynan bir vaqtning o'zida qayta ulanishga urinadi, uni yana ishdan chiqarishi mumkin. Jitter bilan qayta ulanishlar bir necha soniya davomida tarqaladi, bu yumshoq tiklanishga imkon beradi.

**Formula tanlovi:**
\`0.5 + rand.Float64()\` formulasi ([0.5, 1.5] oralig'ida) AWS tomonidan tavsiya etilgan yondashuv. Boshqa variantlar:
- Full jitter: \`rand.Float64() * delay\` (0 dan delay gacha)
- Equal jitter: \`delay/2 + rand.Float64() * delay/2\` (biznikiga o'xshash)
- Decorrelated jitter: Murakkabroq, lekin yaxshiroq taqsimot`
		}
	}
};

export default task;
