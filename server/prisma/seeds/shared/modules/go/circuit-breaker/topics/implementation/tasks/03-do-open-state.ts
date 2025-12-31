import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-do-open',
	title: 'Do Method - Open State',
	difficulty: 'medium',	tags: ['go', 'circuit-breaker', 'fail-fast'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Do** method behavior for **Open** state.

**Requirements:**
1. Check if current time is after \`openUntil\` timestamp
2. If cooldown expired: transition to \`HalfOpen\` state and reset \`halfCount\`
3. If still cooling down: return \`ErrOpen\` immediately without calling function
4. Do NOT execute the function \`f\` while in Open state (fail-fast)

**State Machine - Open:**
\`\`\`
Open State (Circuit is Open):
  If time.Now().After(openUntil):
    → Transition to HalfOpen
    → Reset halfCount = 0
    → Allow request to proceed
  Else:
    → Return ErrOpen immediately
    → DO NOT call function f
\`\`\`

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)

// Circuit trips to Open
for i := 0; i < 3; i++ {
    breaker.Do(ctx, failingFunc)
}

// Now circuit is Open - requests rejected immediately
err := breaker.Do(ctx, anyFunc)
// err == ErrOpen (function NOT called)

// Wait for cooldown
time.Sleep(5 * time.Second)

// Next request transitions to HalfOpen
err = breaker.Do(ctx, anyFunc)
// State is now HalfOpen, function IS called
\`\`\`

**Constraints:**
- Must check time BEFORE releasing mutex
- Return ErrOpen if cooldown not expired
- Transition to HalfOpen only after cooldown expires`,
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
		// TODO: Check if openUntil has passed
		// If yes: transition to HalfOpen and reset halfCount
		// If no: unlock and return ErrOpen
	}
	b.mu.Unlock()

	err := f(ctx)

	b.mu.Lock()
	defer b.mu.Unlock()

	if err == nil {
		switch b.state {
		case Closed:
			b.errs = 0
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++
		if b.errs >= b.threshold {
			b.tripToOpen()
		}
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
				b.state = Closed
				b.errs = 0
				b.halfCount = 0
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
	// Do returns ErrOpen in Open state
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(5 * time.Second)
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if err != ErrOpen {
		t.Errorf("expected ErrOpen, got %v", err)
	}
}

func Test2(t *testing.T) {
	// Do does not execute function in Open state
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(5 * time.Second)
	executed := false
	breaker.Do(context.Background(), func(ctx context.Context) error {
		executed = true
		return nil
	})
	if executed {
		t.Error("function should not be executed in Open state")
	}
}

func Test3(t *testing.T) {
	// Do transitions to HalfOpen after cooldown
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(-1 * time.Second)
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.state != Closed && breaker.state != HalfOpen {
		t.Errorf("expected HalfOpen or Closed state after cooldown")
	}
}

func Test4(t *testing.T) {
	// Do resets halfCount on transition to HalfOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(-1 * time.Second)
	breaker.halfCount = 5
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	// halfCount should be reset on transition or on success
}

func Test5(t *testing.T) {
	// Do stays Open during cooldown
	breaker := New(3, 1*time.Hour, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(1 * time.Hour)
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.state != Open {
		t.Errorf("expected Open state during cooldown, got %v", breaker.state)
	}
}

func Test6(t *testing.T) {
	// Do executes function after cooldown expires
	breaker := New(3, 1*time.Millisecond, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(-1 * time.Millisecond)
	executed := false
	breaker.Do(context.Background(), func(ctx context.Context) error {
		executed = true
		return nil
	})
	if !executed {
		t.Error("function should execute after cooldown")
	}
}

func Test7(t *testing.T) {
	// Do returns ErrOpen immediately in Open state
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(5 * time.Second)
	start := time.Now()
	breaker.Do(context.Background(), func(ctx context.Context) error {
		time.Sleep(1 * time.Second)
		return nil
	})
	elapsed := time.Since(start)
	if elapsed > 100*time.Millisecond {
		t.Error("Open state should return immediately")
	}
}

func Test8(t *testing.T) {
	// Do multiple Open calls all return ErrOpen
	breaker := New(3, 5*time.Second, 2)
	breaker.state = Open
	breaker.openUntil = time.Now().Add(5 * time.Second)
	for i := 0; i < 3; i++ {
		err := breaker.Do(context.Background(), func(ctx context.Context) error {
			return nil
		})
		if err != ErrOpen {
			t.Errorf("call %d: expected ErrOpen, got %v", i, err)
		}
	}
}

func Test9(t *testing.T) {
	// Do transitions after exact cooldown
	breaker := New(3, 1*time.Nanosecond, 2)
	breaker.state = Open
	breaker.openUntil = time.Now()
	time.Sleep(1 * time.Millisecond)
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
}

func Test10(t *testing.T) {
	// Do trips to Open after failures propagates to Open behavior
	breaker := New(3, 100*time.Millisecond, 2)
	for i := 0; i < 3; i++ {
		breaker.Do(context.Background(), func(ctx context.Context) error {
			return errors.New("failure")
		})
	}
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if err != ErrOpen {
		t.Errorf("expected ErrOpen after tripping, got %v", err)
	}
}
`,
		hint1: `Check if now.After(b.openUntil) to determine if cooldown period has expired.`,
			hint2: `If cooldown expired: set state to HalfOpen and halfCount to 0. Otherwise: unlock mutex and return ErrOpen.`,
			whyItMatters: `The Open state implements fail-fast behavior to protect failing services from being overwhelmed with requests.

**Why This Matters:**
- **Fail-Fast:** Immediately reject requests without waiting for timeout
- **Resource Protection:** Prevents wasting threads/connections on failing service
- **User Experience:** Fast error response instead of slow timeout
- **Service Recovery:** Gives failing service time to recover

**Real-World Example:**
\`\`\`go
// Database connection pool exhausted
dbBreaker := New(5, 30*time.Second, 3)

// Circuit opens after 5 failures
for i := 0; i < 5; i++ {
    dbBreaker.Do(ctx, queryDB) // Connection timeout (30s each)
}

// Now circuit is OPEN - subsequent requests fail immediately
err := dbBreaker.Do(ctx, queryDB)
// Returns ErrOpen in microseconds instead of 30s timeout!

// After 30 seconds, circuit tries HalfOpen
// Allows limited trial requests to test if DB recovered
\`\`\`

**Production Impact:**

**Without Circuit Breaker:**
- 1000 requests × 30s timeout = 30,000 seconds of blocked threads
- Thread pool exhaustion
- Service becomes unresponsive
- Cascading failure to other services

**With Circuit Breaker (Open State):**
- First 5 requests fail (150s total)
- Circuit opens
- Next 995 requests fail instantly (microseconds)
- Thread pool available for other work
- Service remains responsive
- After 30s cooldown, try recovery

**Key Insight:**
The Open state is not just about "blocking requests" - it's about:
1. **Instant failure** instead of slow timeout
2. **Resource preservation** (threads, connections, memory)
3. **Graceful degradation** with clear error (ErrOpen)
4. **Automatic recovery** attempt after cooldown

**Common Use Cases:**
- External API down (avoid 60s timeouts)
- Database connection pool exhausted
- Downstream microservice overloaded
- Payment gateway maintenance window`,	order: 2,
	translations: {
		ru: {
			title: 'Выполнение в открытом состоянии',
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
				b.state = Closed
				b.errs = 0
				b.halfCount = 0
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
			description: `Реализуйте поведение метода **Do** для состояния **Open**.

**Требования:**
1. Проверьте, истекло ли время \`openUntil\`
2. Если cooldown истёк: переход в \`HalfOpen\`, сброс \`halfCount\`
3. Если ещё охлаждается: вернуть \`ErrOpen\` без вызова функции
4. НЕ выполнять функцию \`f\` в Open (fail-fast)

**Машина состояний - Open:**
\`\`\`
Если time.Now().After(openUntil):
  → HalfOpen, halfCount = 0, разрешить запрос
Иначе:
  → Вернуть ErrOpen, НЕ вызывать f
\`\`\`

**Ограничения:**
- Проверяйте время ДО освобождения мьютекса
- Возвращайте ErrOpen если cooldown не истёк
- Переходите в HalfOpen только после истечения`,
			hint1: `Проверьте now.After(b.openUntil) для определения истечения cooldown.`,
			hint2: `Если истёк: state = HalfOpen, halfCount = 0. Иначе: unlock и return ErrOpen.`,
			whyItMatters: `Open состояние реализует fail-fast для защиты падающих сервисов от перегрузки.

**Почему важно:**
- **Fail-Fast:** Мгновенный отказ вместо ожидания timeout
- **Защита ресурсов:** Не тратим потоки на падающий сервис
- **UX:** Быстрый ответ об ошибке вместо медленного timeout
- **Восстановление:** Даёт сервису время восстановиться

**Пример из реальной практики:**
\`\`\`go
// Пул соединений с БД исчерпан
dbBreaker := New(5, 30*time.Second, 3)

// Цепь открывается после 5 ошибок
for i := 0; i < 5; i++ {
    dbBreaker.Do(ctx, queryDB) // Таймаут соединения (по 30s каждый)
}

// Теперь цепь ОТКРЫТА - последующие запросы падают мгновенно
err := dbBreaker.Do(ctx, queryDB)
// Возвращает ErrOpen за микросекунды вместо 30s timeout!

// Через 30 секунд цепь пытается HalfOpen
// Разрешает ограниченные пробные запросы для проверки восстановления БД
\`\`\`

**Продакшен паттерн:**

Без Circuit Breaker:
- 1000 запросов × 30s timeout = 30,000 секунд заблокированных потоков
- Исчерпание пула потоков
- Сервис становится неотзывчивым
- Каскадный сбой на другие сервисы

С Circuit Breaker (Open State):
- Первые 5 запросов падают (150s всего)
- Цепь открывается
- Следующие 995 запросов падают мгновенно (микросекунды)
- Пул потоков доступен для другой работы
- Сервис остаётся отзывчивым
- После 30s cooldown пытаемся восстановиться

**Практические преимущества:**
Open состояние - это не просто "блокировка запросов", это:
1. **Мгновенный сбой** вместо медленного timeout
2. **Сохранение ресурсов** (потоки, соединения, память)
3. **Плавная деградация** с понятной ошибкой (ErrOpen)
4. **Автоматическая попытка восстановления** после cooldown`
		},
		uz: {
			title: `Ochiq holatda bajarish`,
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
				b.state = Closed
				b.errs = 0
				b.halfCount = 0
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
			description: `**Open** holati uchun **Do** metodi xatti-harakatini amalga oshiring.

**Talablar:**
1. Joriy vaqt \`openUntil\` vaqt belgisidan keyin ekanligini tekshiring
2. Agar cooldown tugagan bo'lsa: \`HalfOpen\` holatiga o'ting va \`halfCount\` ni qayta o'rnating
3. Agar hali sovutilayotgan bo'lsa: funksiyani chaqirmasdan darhol \`ErrOpen\` qaytaring
4. Open holatida \`f\` funksiyasini BAJARMANG (fail-fast)

**Holat mashinasi - Open:**
\`\`\`
Agar time.Now().After(openUntil):
  → HalfOpen ga o'tish, halfCount = 0, so'rovga ruxsat
Aks holda:
  → Darhol ErrOpen qaytarish, f ni CHAQIRMANG
\`\`\`

**Cheklovlar:**
- Mutex ni bo'shatishdan OLDIN vaqtni tekshiring
- Cooldown tugamagan bo'lsa ErrOpen qaytaring
- Faqat cooldown tugaganidan keyin HalfOpen ga o'ting`,
			hint1: `Cooldown davri tugaganligini aniqlash uchun now.After(b.openUntil) ni tekshiring.`,
			hint2: `Agar tugagan bo'lsa: state = HalfOpen, halfCount = 0. Aks holda: unlock va ErrOpen qaytaring.`,
			whyItMatters: `Open holati muvaffaqiyatsiz xizmatlarni so'rovlar bilan ortiqcha yuklanishdan himoya qilish uchun fail-fast xatti-harakatini amalga oshiradi.

**Nima uchun bu muhim:**
- **Fail-Fast:** Timeout ni kutmasdan darhol so'rovlarni rad etish
- **Resurslarni himoya qilish:** Muvaffaqiyatsiz xizmatga threadlar/ulanishlarni sarflamaydi
- **UX:** Sekin timeout o'rniga tez xato javobi
- **Xizmat tiklash:** Muvaffaqiyatsiz xizmatga tiklanish uchun vaqt beradi

**Amaliy misoldan:**
\`\`\`go
// Ma'lumotlar bazasi ulanishlar puli tugagan
dbBreaker := New(5, 30*time.Second, 3)

// 5 ta muvaffaqiyatsizlikdan keyin zanjir ochiladi
for i := 0; i < 5; i++ {
    dbBreaker.Do(ctx, queryDB) // Ulanish timeout (har biri 30s)
}

// Endi zanjir OCHIQ - keyingi so'rovlar darhol muvaffaqiyatsiz bo'ladi
err := dbBreaker.Do(ctx, queryDB)
// 30s timeout o'rniga mikrosekundlarda ErrOpen qaytaradi!

// 30 soniyadan keyin zanjir HalfOpen ga harakat qiladi
// DB tiklanganligini tekshirish uchun cheklangan sinov so'rovlariga ruxsat beradi
\`\`\`

**Ishlab chiqarish patterni:**

Circuit Breaker siz:
- 1000 so'rov × 30s timeout = 30,000 soniya bloklangan threadlar
- Thread pool tugashi
- Xizmat javob bermaydigan holga keladi
- Boshqa xizmatlarga kaskadli nosozlik

Circuit Breaker bilan (Open State):
- Birinchi 5 so'rov muvaffaqiyatsiz (jami 150s)
- Zanjir ochiladi
- Keyingi 995 so'rov darhol muvaffaqiyatsiz (mikrosekundlar)
- Thread pool boshqa ish uchun mavjud
- Xizmat javob berishda qoladi
- 30s cooldown dan keyin tiklanishga harakat qilamiz

**Amaliy foydalari:**
Open holati shunchaki "so'rovlarni bloklash" emas, bu:
1. **Darhol muvaffaqiyatsizlik** sekin timeout o'rniga
2. **Resurslarni saqlash** (threadlar, ulanishlar, xotira)
3. **Yumshoq degradatsiya** aniq xato bilan (ErrOpen)
4. **Avtomatik tiklash urinishi** cooldown dan keyin`
		}
	}
};

export default task;
