import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-do-halfopen',
	title: 'Do Method - HalfOpen State',
	difficulty: 'hard',	tags: ['go', 'circuit-breaker', 'recovery'],
	estimatedTime: '40m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Do** method behavior for **HalfOpen** state - the most complex state!

**Requirements:**
1. On **success**: increment \`halfCount\` counter
2. If \`halfCount >= halfMax\`: circuit has recovered, transition to \`Closed\`
3. When transitioning to Closed: reset ALL counters (\`errs\`, \`halfCount\`)
4. On **failure**: immediately call \`tripToOpen()\` to re-open circuit
5. HalfOpen allows LIMITED trial requests to test recovery

**State Machine - HalfOpen:**
\`\`\`
HalfOpen State (Testing Recovery):
  Success:
    → Increment halfCount
    → If halfCount >= halfMax:
        ✓ Recovery confirmed → Closed
        ✓ Reset errs = 0, halfCount = 0
    → Else: Stay HalfOpen (more trials needed)

  Failure:
    ✗ Recovery failed → Open (via tripToOpen)
    ✗ Restart cooldown period
\`\`\`

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)  // halfMax = 2

// Circuit opens after failures
// ... wait 5 seconds ...
// Circuit transitions to HalfOpen on first request

// Trial 1: Success
breaker.Do(ctx, successFunc)  // halfCount = 1, stay HalfOpen

// Trial 2: Success
breaker.Do(ctx, successFunc)  // halfCount = 2 >= halfMax
// → Transition to Closed! Circuit fully recovered.

// Example 2: Failure in HalfOpen
breaker.Do(ctx, failingFunc)  // Immediate trip to Open
// → Back to cooldown period
\`\`\`

**Constraints:**
- Success in HalfOpen increments \`halfCount\`
- Must check \`halfCount >= halfMax\` to close circuit
- Failure immediately reopens (call \`tripToOpen()\`)
- Reset ALL counters when transitioning to Closed`,
	initialCode: `package circuitx

import (
	"context"
	"time"
)

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()
	now := time.Now()
	switch b.state {
	case Open:
		if now.After(b.openUntil) {
			b.state = HalfOpen
			b.halfCount = 0
		} else {
			b.mu.Unlock()
			return ErrOpen
		}
	}
	b.mu.Unlock()

	err := f(ctx)

	b.mu.Lock()
	defer b.mu.Unlock()

	if err == nil {
		switch b.state {
		case Closed:
			b.errs = 0
		case HalfOpen:
			// TODO: Increment halfCount
			// TODO: Check if halfCount >= halfMax
			// TODO: If yes, transition to Closed and reset counters
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++
		if b.errs >= b.threshold {
			b.tripToOpen()
		}
	case HalfOpen:
		// TODO: On failure in HalfOpen, immediately trip to Open
	}
	return err
}`,
	solutionCode: `package circuitx

import (
	"context"
	"time"
)

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()       // inspect and possibly mutate state under lock
	now := time.Now() // snapshot current time for threshold checks
	switch b.state {
	case Open:
		if now.After(b.openUntil) { // transition from open to half-open when cooldown finished
			b.state = HalfOpen // allow limited trial requests
			b.halfCount = 0    // reset success counter for half-open phase
		} else {
			b.mu.Unlock()
			return ErrOpen // deny requests while breaker remains open
		}
	}
	b.mu.Unlock() // release lock before invoking user function

	err := f(ctx) // execute protected operation with provided context

	b.mu.Lock()         // reacquire lock to update state counters based on outcome
	defer b.mu.Unlock() // ensure lock released before returning

	if err == nil { // handle successful invocation
		switch b.state {
		case Closed:
			b.errs = 0 // reset consecutive error counter
		case HalfOpen:
			b.halfCount++                 // track successes allowed in half-open state
			if b.halfCount >= b.halfMax { // promote breaker to closed after threshold successes
				b.state = Closed    // circuit has recovered
				b.errs = 0          // reset error counter for closed state
				b.halfCount = 0     // reset half-open counter
			}
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++                   // increment error counter in closed state
		if b.errs >= b.threshold { // exceed threshold -> open breaker
			b.tripToOpen() // move to open state and schedule reopen time
		}
	case HalfOpen:
		b.tripToOpen() // failure in half-open immediately reopens breaker
	}
	return err // propagate original error to caller
}`,
		testCode: `package circuitx

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Do increments halfCount on success in HalfOpen
	breaker := New(3, 5*time.Second, 3)
	breaker.state = HalfOpen
	breaker.halfCount = 0
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.halfCount != 1 {
		t.Errorf("expected halfCount 1, got %d", breaker.halfCount)
	}
}

func Test2(t *testing.T) {
	// Do transitions to Closed at halfMax
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	breaker.halfCount = 1
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.state != Closed {
		t.Errorf("expected Closed state at halfMax, got %v", breaker.state)
	}
}

func Test3(t *testing.T) {
	// Do resets counters on transition to Closed
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	breaker.halfCount = 1
	breaker.errs = 5
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.errs != 0 || breaker.halfCount != 0 {
		t.Error("expected counters reset on Closed transition")
	}
}

func Test4(t *testing.T) {
	// Do trips to Open on failure in HalfOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	breaker.halfCount = 1
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return errors.New("failure")
	})
	if breaker.state != Open {
		t.Errorf("expected Open state on HalfOpen failure, got %v", breaker.state)
	}
}

func Test5(t *testing.T) {
	// Do stays HalfOpen below halfMax
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	breaker.halfCount = 0
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.state != HalfOpen {
		t.Errorf("expected HalfOpen below halfMax, got %v", breaker.state)
	}
}

func Test6(t *testing.T) {
	// Do executes function in HalfOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	executed := false
	breaker.Do(context.Background(), func(ctx context.Context) error {
		executed = true
		return nil
	})
	if !executed {
		t.Error("function should execute in HalfOpen")
	}
}

func Test7(t *testing.T) {
	// Do returns nil on success in HalfOpen
	breaker := New(3, 5*time.Second, 5)
	breaker.state = HalfOpen
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

func Test8(t *testing.T) {
	// Do returns error on failure in HalfOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	testErr := errors.New("test error")
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return testErr
	})
	if err != testErr {
		t.Errorf("expected test error, got %v", err)
	}
}

func Test9(t *testing.T) {
	// Do multiple successes lead to Closed
	breaker := New(3, 5*time.Second, 3)
	breaker.state = HalfOpen
	for i := 0; i < 3; i++ {
		breaker.Do(context.Background(), func(ctx context.Context) error {
			return nil
		})
	}
	if breaker.state != Closed {
		t.Errorf("expected Closed after 3 successes, got %v", breaker.state)
	}
}

func Test10(t *testing.T) {
	// Do failure resets openUntil
	breaker := New(3, 5*time.Second, 2)
	breaker.state = HalfOpen
	before := time.Now()
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return errors.New("failure")
	})
	if breaker.openUntil.Before(before) {
		t.Error("expected openUntil to be updated after failure")
	}
}
`,
		hint1: `On success in HalfOpen: increment b.halfCount, then check if b.halfCount >= b.halfMax.`,
			hint2: `If threshold reached: set state to Closed and reset both b.errs = 0 and b.halfCount = 0.`,
			whyItMatters: `HalfOpen is the critical recovery phase where the circuit breaker tests if a failing service has recovered.

**Why HalfOpen is Complex:**
- **Optimistic Testing:** Allows LIMITED requests to probe service health
- **Quick Failure:** One failure means service still unhealthy → back to Open
- **Gradual Recovery:** Requires multiple successes to confirm recovery
- **Balance:** Too few trials = false negatives, too many = overload during recovery

**Real-World Example:**
\`\`\`go
// Payment gateway circuit breaker
paymentBreaker := New(5, 60*time.Second, 3)  // halfMax=3

// 9:00 AM - Payment gateway goes down
// Circuit opens after 5 failures

// 9:01 AM - Circuit enters HalfOpen
// Trial 1: payment.Process() → Success! (halfCount=1)
// Trial 2: payment.Process() → Success! (halfCount=2)
// Trial 3: payment.Process() → Success! (halfCount=3)
// → Circuit CLOSES! Gateway confirmed healthy

// Alternative scenario:
// Trial 1: payment.Process() → Success! (halfCount=1)
// Trial 2: payment.Process() → FAILURE!
// → Circuit reopens immediately, wait another 60s
\`\`\`

**Production Patterns:**

**HalfMax Tuning:**
\`\`\`go
// Critical service - require strong recovery signal
criticalBreaker := New(3, 30*time.Second, 5)  // 5 successes needed

// Best-effort service - quick recovery
cacheBreaker := New(3, 10*time.Second, 1)  // 1 success enough

// Payment/financial - moderate confidence
paymentBreaker := New(5, 60*time.Second, 3)  // 3 successes
\`\`\`

**Why This Matters:**

1. **Prevents Thundering Herd:** HalfOpen limits concurrent recovery attempts
   1.1. Without HalfOpen: All requests hit recovering service → overload
   1.2. With HalfOpen: Gradual traffic increase → safe recovery

2. **False Positive Prevention:** Single success might be luck, halfMax confirms recovery
   2.1. halfMax=1: Aggressive recovery, risk of flapping
   2.2. halfMax=3-5: Confident recovery, stable state transitions

3. **Service Protection:** Failing request in HalfOpen immediately reopens
   3.1. Service still struggling? → Don't overwhelm it
   3.2. Give it more time to recover (new cooldown)

4. **Metrics & Monitoring:**
   4.1. Count HalfOpen → Open transitions (flapping indicator)
   4.2. Track halfCount progress (recovery health)
   4.3. Monitor time-in-HalfOpen (recovery duration)

**Common Mistakes:**

1. **Not resetting counters on Closed transition:**
   \`\`\`go
   // WRONG
   if b.halfCount >= b.halfMax {
       b.state = Closed  // Forgot to reset!
   }

   // CORRECT
   if b.halfCount >= b.halfMax {
       b.state = Closed
       b.errs = 0        // Clean slate
       b.halfCount = 0   // Clean slate
   }
   \`\`\`

2. **Not failing fast on error:**
   \`\`\`go
   // WRONG - incrementing counter
   case HalfOpen:
       b.halfErrCount++

   // CORRECT - immediate trip
   case HalfOpen:
       b.tripToOpen()  // Service still broken!
   \`\`\`

3. **Wrong halfMax value:**
   3.1. Too low: Circuit flaps (opens/closes repeatedly)
   3.2. Too high: Takes too long to recover
   3.3. Sweet spot: 2-5 for most services

**Real Production Story:**
A microservices team used halfMax=1, causing circuit to close after single success. When database recovered but was still slow, circuit would:
- Close after 1 fast query
- Immediately open after first slow query
- Enter HalfOpen after cooldown
- Repeat...

This "flapping" prevented full recovery. Setting halfMax=5 required consistent performance before closing, solving the issue.`,	order: 3,
	translations: {
		ru: {
			title: 'Выполнение в полуоткрытом состоянии',
			solutionCode: `package circuitx

import (
	"context"
	"time"
)

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()       // проверяем и возможно изменяем состояние под блокировкой
	now := time.Now() // фиксируем текущее время для проверки порогов
	switch b.state {
	case Open:
		if now.After(b.openUntil) { // переход из open в half-open когда cooldown завершён
			b.state = HalfOpen // разрешаем ограниченные пробные запросы
			b.halfCount = 0    // сбрасываем счётчик успехов для фазы half-open
		} else {
			b.mu.Unlock()
			return ErrOpen // отклоняем запросы пока breaker остаётся открытым
		}
	}
	b.mu.Unlock() // освобождаем блокировку перед вызовом пользовательской функции

	err := f(ctx) // выполняем защищённую операцию с переданным контекстом

	b.mu.Lock()         // снова получаем блокировку для обновления счётчиков
	defer b.mu.Unlock() // гарантируем освобождение блокировки перед возвратом

	if err == nil { // обрабатываем успешный вызов
		switch b.state {
		case Closed:
			b.errs = 0 // сбрасываем счётчик последовательных ошибок
		case HalfOpen:
			b.halfCount++                 // отслеживаем успехи в half-open состоянии
			if b.halfCount >= b.halfMax { // повышаем breaker до closed после порога успехов
				b.state = Closed    // цепь восстановилась
				b.errs = 0          // сбрасываем счётчик ошибок для closed состояния
				b.halfCount = 0     // сбрасываем счётчик half-open
			}
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++                   // увеличиваем счётчик ошибок в closed состоянии
		if b.errs >= b.threshold { // превышен порог -> открываем breaker
			b.tripToOpen() // переходим в open и планируем время переоткрытия
		}
	case HalfOpen:
		b.tripToOpen() // ошибка в half-open немедленно открывает breaker
	}
	return err // передаём оригинальную ошибку вызывающему
}`,
			description: `Реализуйте поведение метода **Do** для состояния **HalfOpen** - самое сложное состояние!

**Требования:**
1. При **успехе**: увеличить счётчик \`halfCount\`
2. Если \`halfCount >= halfMax\`: восстановление подтверждено, переход в \`Closed\`
3. При переходе в Closed: сбросить ВСЕ счётчики (\`errs\`, \`halfCount\`)
4. При **ошибке**: немедленно вызвать \`tripToOpen()\`
5. HalfOpen позволяет ОГРАНИЧЕННОЕ число пробных запросов

**Машина состояний - HalfOpen:**
\`\`\`
Успех:
  → Увеличить halfCount
  → Если halfCount >= halfMax:
      ✓ Восстановление → Closed
      ✓ Сброс errs = 0, halfCount = 0
  → Иначе: HalfOpen (нужно больше проб)

Ошибка:
  ✗ Восстановление провалено → Open (tripToOpen)
  ✗ Перезапуск cooldown
\`\`\`

**Ограничения:**
- Успех увеличивает \`halfCount\`
- Проверяйте \`halfCount >= halfMax\` для закрытия
- Ошибка немедленно открывает (tripToOpen)
- Сбросьте ВСЕ счётчики при переходе в Closed`,
			hint1: `При успехе в HalfOpen: увеличьте b.halfCount, затем проверьте порог.`,
			hint2: `Если достигнут: state = Closed, сбросьте errs и halfCount.`,
			whyItMatters: `HalfOpen - критическая фаза восстановления для тестирования здоровья сервиса.

**Почему HalfOpen сложен:**
- **Оптимистичное тестирование:** Ограниченные запросы для проверки
- **Быстрый отказ:** Одна ошибка = сервис всё ещё нездоров
- **Постепенное восстановление:** Нужно несколько успехов
- **Баланс:** Мало проб = ложные негативы, много = перегрузка

**Пример из реальной практики:**
\`\`\`go
// Circuit breaker платёжного шлюза
paymentBreaker := New(5, 60*time.Second, 3)  // halfMax=3

// 9:00 AM - Платёжный шлюз падает
// Цепь открывается после 5 ошибок

// 9:01 AM - Цепь входит в HalfOpen
// Проба 1: payment.Process() → Успех! (halfCount=1)
// Проба 2: payment.Process() → Успех! (halfCount=2)
// Проба 3: payment.Process() → Успех! (halfCount=3)
// → Цепь ЗАКРЫВАЕТСЯ! Шлюз подтверждён как здоровый

// Альтернативный сценарий:
// Проба 1: payment.Process() → Успех! (halfCount=1)
// Проба 2: payment.Process() → ОШИБКА!
// → Цепь немедленно открывается, ждём ещё 60s
\`\`\`

**Продакшен паттерны:**

**Настройка HalfMax:**
\`\`\`go
// Критичный сервис - требуем сильный сигнал восстановления
criticalBreaker := New(3, 30*time.Second, 5)  // нужно 5 успехов

// Best-effort сервис - быстрое восстановление
cacheBreaker := New(3, 10*time.Second, 1)  // достаточно 1 успеха

// Платёжный/финансовый - умеренная уверенность
paymentBreaker := New(5, 60*time.Second, 3)  // 3 успеха
\`\`\`

**Почему это важно:**

1. **Предотвращает Thundering Herd:** HalfOpen ограничивает конкурентные попытки восстановления
   1.1. Без HalfOpen: Все запросы бьют восстанавливающийся сервис → перегрузка
   1.2. С HalfOpen: Постепенное увеличение трафика → безопасное восстановление

2. **Предотвращение ложных срабатываний:** Один успех может быть удачей, halfMax подтверждает восстановление
   2.1. halfMax=1: Агрессивное восстановление, риск флаппинга
   2.2. halfMax=3-5: Уверенное восстановление, стабильные переходы состояний

3. **Защита сервиса:** Неудачный запрос в HalfOpen немедленно открывает
   3.1. Сервис всё ещё борется? → Не перегружайте его
   3.2. Дайте ему больше времени на восстановление (новый cooldown)

**Частые ошибки:**

1. **Не сбрасывать счётчики при переходе в Closed:**
   \`\`\`go
   // НЕПРАВИЛЬНО
   if b.halfCount >= b.halfMax {
       b.state = Closed  // Забыли сбросить!
   }

   // ПРАВИЛЬНО
   if b.halfCount >= b.halfMax {
       b.state = Closed
       b.errs = 0        // Чистый лист
       b.halfCount = 0   // Чистый лист
   }
   \`\`\`

2. **Неправильное значение halfMax:**
   2.1. Слишком низко: Цепь флаппит (открывается/закрывается повторно)
   2.2. Слишком высоко: Слишком долго восстанавливается
   2.3. Оптимально: 2-5 для большинства сервисов`
		},
		uz: {
			title: `Yarim ochiq holatda bajarish`,
			solutionCode: `package circuitx

import (
	"context"
	"time"
)

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()       // qulf ostida holatni tekshiramiz va ehtimol o'zgartiramiz
	now := time.Now() // chegara tekshiruvlari uchun joriy vaqtni oldindan olamiz
	switch b.state {
	case Open:
		if now.After(b.openUntil) { // cooldown tugaganda open dan half-open ga o'tish
			b.state = HalfOpen // cheklangan sinov so'rovlariga ruxsat beramiz
			b.halfCount = 0    // half-open bosqichi uchun muvaffaqiyat hisoblagichini qayta o'rnatamiz
		} else {
			b.mu.Unlock()
			return ErrOpen // breaker ochiq qolgan paytda so'rovlarni rad etamiz
		}
	}
	b.mu.Unlock() // foydalanuvchi funksiyasini chaqirishdan oldin qulfni bo'shatamiz

	err := f(ctx) // berilgan kontekst bilan himoyalangan operatsiyani bajaramiz

	b.mu.Lock()         // natijaga asoslangan hisoblagichlarni yangilash uchun qulfni qayta olamiz
	defer b.mu.Unlock() // qaytishdan oldin qulf bo'shatilishini ta'minlaymiz

	if err == nil { // muvaffaqiyatli chaqiruvni qayta ishlaymiz
		switch b.state {
		case Closed:
			b.errs = 0 // ketma-ket xato hisoblagichini qayta o'rnatamiz
		case HalfOpen:
			b.halfCount++                 // half-open holatida muvaffaqiyatlarni kuzatamiz
			if b.halfCount >= b.halfMax { // chegara muvaffaqiyatlaridan keyin breakerni closed ga ko'taramiz
				b.state = Closed    // zanjir tiklandi
				b.errs = 0          // closed holati uchun xato hisoblagichini qayta o'rnatamiz
				b.halfCount = 0     // half-open hisoblagichini qayta o'rnatamiz
			}
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++                   // closed holatida xato hisoblagichini oshiramiz
		if b.errs >= b.threshold { // chegaradan oshdi -> breakerni ochamiz
			b.tripToOpen() // open holatiga o'tamiz va qayta ochilish vaqtini rejalashtiramiz
		}
	case HalfOpen:
		b.tripToOpen() // half-open dagi nosozlik breakerni darhol qayta ochadi
	}
	return err // asl xatoni chaqiruvchiga uzatamiz
}`,
			description: `**HalfOpen** holati uchun **Do** metodi xatti-harakatini amalga oshiring - eng murakkab holat!

**Talablar:**
1. **Muvaffaqiyatda**: \`halfCount\` hisoblagichini oshiring
2. Agar \`halfCount >= halfMax\`: tiklash tasdiqlandi, \`Closed\` ga o'ting
3. Closed ga o'tishda: BARCHA hisoblagichlarni qayta o'rnating (\`errs\`, \`halfCount\`)
4. **Muvaffaqiyatsizlikda**: darhol \`tripToOpen()\` ni chaqiring
5. HalfOpen tiklanishni sinash uchun CHEKLANGAN sinov so'rovlariga ruxsat beradi

**Holat mashinasi - HalfOpen:**
\`\`\`
Muvaffaqiyat:
  → halfCount ni oshirish
  → Agar halfCount >= halfMax:
      ✓ Tiklash → Closed
      ✓ errs = 0, halfCount = 0 qayta o'rnatish
  → Aks holda: HalfOpen (ko'proq sinov kerak)

Muvaffaqiyatsizlik:
  ✗ Tiklash muvaffaqiyatsiz → Open (tripToOpen)
  ✗ Cooldown ni qayta boshlash
\`\`\`

**Cheklovlar:**
- Muvaffaqiyat \`halfCount\` ni oshiradi
- Zanjirni yopish uchun \`halfCount >= halfMax\` ni tekshiring
- Muvaffaqiyatsizlik darhol ochadi (tripToOpen)
- Closed ga o'tishda BARCHA hisoblagichlarni qayta o'rnating`,
			hint1: `HalfOpen da muvaffaqiyatda: b.halfCount ni oshiring, keyin chegarani tekshiring.`,
			hint2: `Agar yetilgan bo'lsa: state = Closed, errs va halfCount ni qayta o'rnating.`,
			whyItMatters: `HalfOpen - circuit breaker muvaffaqiyatsiz xizmat tiklanganligini sinab ko'radigan muhim tiklash bosqichi.

**Nima uchun HalfOpen murakkab:**
- **Optimistik sinov:** Xizmat salomatligini tekshirish uchun CHEKLANGAN so'rovlar
- **Tez muvaffaqiyatsizlik:** Bitta muvaffaqiyatsizlik = xizmat hali sog'lom emas
- **Bosqichma-bosqich tiklash:** Tiklanishni tasdiqlash uchun bir nechta muvaffaqiyat kerak
- **Muvozanat:** Kam sinov = yolg'on salbiy, ko'p = tiklash paytida ortiqcha yuklash

**Amaliy misoldan:**
\`\`\`go
// To'lov shlyuzi circuit breaker
paymentBreaker := New(5, 60*time.Second, 3)  // halfMax=3

// 9:00 AM - To'lov shlyuzi ishdan chiqadi
// 5 ta muvaffaqiyatsizlikdan keyin zanjir ochiladi

// 9:01 AM - Zanjir HalfOpen ga kiradi
// Sinov 1: payment.Process() → Muvaffaqiyat! (halfCount=1)
// Sinov 2: payment.Process() → Muvaffaqiyat! (halfCount=2)
// Sinov 3: payment.Process() → Muvaffaqiyat! (halfCount=3)
// → Zanjir YOPILADI! Shlyuz sog'lom deb tasdiqlandi

// Muqobil stsenariy:
// Sinov 1: payment.Process() → Muvaffaqiyat! (halfCount=1)
// Sinov 2: payment.Process() → MUVAFFAQIYATSIZLIK!
// → Zanjir darhol qayta ochiladi, yana 60s kutamiz
\`\`\`

**Ishlab chiqarish patternlari:**

**HalfMax sozlash:**
\`\`\`go
// Muhim xizmat - kuchli tiklash signalini talab qilamiz
criticalBreaker := New(3, 30*time.Second, 5)  // 5 ta muvaffaqiyat kerak

// Best-effort xizmat - tez tiklash
cacheBreaker := New(3, 10*time.Second, 1)  // 1 ta muvaffaqiyat yetarli

// To'lov/moliyaviy - o'rtacha ishonch
paymentBreaker := New(5, 60*time.Second, 3)  // 3 ta muvaffaqiyat
\`\`\`

**Nima uchun bu muhim:**

1. **Thundering Herd ni oldini oladi:** HalfOpen parallel tiklash urinishlarini cheklaydi
   1.1. HalfOpen siz: Barcha so'rovlar tiklanayotgan xizmatga uriladi → ortiqcha yuklash
   1.2. HalfOpen bilan: Bosqichli trafik oshishi → xavfsiz tiklash

2. **Yolg'on ijobiy natijalarni oldini oladi:** Bitta muvaffaqiyat omad bo'lishi mumkin, halfMax tiklanishni tasdiqlaydi
   2.1. halfMax=1: Agressiv tiklash, flapping xavfi
   2.2. halfMax=3-5: Ishonchli tiklash, barqaror holat o'tishlari

3. **Xizmatni himoya qilish:** HalfOpen da muvaffaqiyatsiz so'rov darhol ochadi
   3.1. Xizmat hali kurashayaptimi? → Uni ortiqcha yuklamang
   3.2. Unga tiklanish uchun ko'proq vaqt bering (yangi cooldown)

**Keng tarqalgan xatolar:**

1. **Closed ga o'tishda hisoblagichlarni qayta o'rnatmaslik:**
   \`\`\`go
   // NOTO'G'RI
   if b.halfCount >= b.halfMax {
       b.state = Closed  // Qayta o'rnatishni unutdik!
   }

   // TO'G'RI
   if b.halfCount >= b.halfMax {
       b.state = Closed
       b.errs = 0        // Toza sahifa
       b.halfCount = 0   // Toza sahifa
   }
   \`\`\`

2. **Noto'g'ri halfMax qiymati:**
   2.1. Juda past: Zanjir flapping (qayta-qayta ochiladi/yopiladi)
   2.2. Juda yuqori: Juda uzoq tiklanadi
   2.3. Eng yaxshi: Ko'pchilik xizmatlar uchun 2-5`
		}
	}
};

export default task;
