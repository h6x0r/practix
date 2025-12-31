import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-do-closed',
	title: 'Do Method - Closed State',
	difficulty: 'medium',	tags: ['go', 'circuit-breaker', 'state-machine'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Do** method behavior for **Closed** state.

**Requirements:**
1. Execute the provided function \`f\` with the context
2. On success: reset error counter (\`errs = 0\`)
3. On failure: increment error counter
4. When \`errs >= threshold\`: transition to Open state using \`tripToOpen()\`
5. Protect state mutations with mutex

**State Machine - Closed:**
\`\`\`
Closed State:
  Success → Reset error counter → Stay Closed
  Failure → Increment errors → Check threshold
    If errors >= threshold → Open
    Else → Stay Closed
\`\`\`

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)

// First 2 failures - stay closed
breaker.Do(ctx, failingFunc)  // errs = 1
breaker.Do(ctx, failingFunc)  // errs = 2

// 3rd failure - trip to open
breaker.Do(ctx, failingFunc)  // errs = 3 → Open state

// Success resets counter
breaker.Do(ctx, successFunc)  // errs = 0
\`\`\`

**Constraints:**
- Must use mutex for thread safety
- Call \`tripToOpen()\` when threshold reached
- Return the original error from function`,
	initialCode: `package circuitx

import (
	"context"
)

// TODO: Implement Do method for Closed state
// Handle success: reset error counter
// Handle failure: increment counter, check threshold, call tripToOpen()
func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()
	// TODO: Check if we need to transition states (for other states)
	b.mu.Unlock()

	err := f(ctx)  // Execute function

	b.mu.Lock()
	defer b.mu.Unlock()

	// TODO: Handle success case (err == nil) for Closed state
	if err == nil {
		switch b.state {
		case Closed:
			// TODO: Reset error counter
		}
		return nil
	}

	// TODO: Handle failure case for Closed state
	switch b.state {
	case Closed:
		// TODO: Increment error counter and check threshold
	}
	return err
}

func (b *Breaker) tripToOpen() {
	b.state = Open
	b.openUntil = time.Now().Add(b.openDur)
	b.errs = 0
	b.halfCount = 0
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
			b.state = HalfOpen
			b.halfCount = 0
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
}

func (b *Breaker) tripToOpen() {
	b.state = Open                          // mark breaker as open
	b.openUntil = time.Now().Add(b.openDur) // compute moment to attempt half-open transition
	b.errs = 0                              // reset error counter for next closed phase
	b.halfCount = 0                         // reset half-open success counter
}`,
		testCode: `package circuitx

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Do executes function in Closed state
	breaker := New(3, 5*time.Second, 2)
	executed := false
	breaker.Do(context.Background(), func(ctx context.Context) error {
		executed = true
		return nil
	})
	if !executed {
		t.Error("expected function to be executed")
	}
}

func Test2(t *testing.T) {
	// Do resets errs on success
	breaker := New(3, 5*time.Second, 2)
	breaker.errs = 2
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.errs != 0 {
		t.Errorf("expected errs 0 after success, got %d", breaker.errs)
	}
}

func Test3(t *testing.T) {
	// Do increments errs on failure
	breaker := New(3, 5*time.Second, 2)
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return errors.New("failure")
	})
	if breaker.errs != 1 {
		t.Errorf("expected errs 1 after failure, got %d", breaker.errs)
	}
}

func Test4(t *testing.T) {
	// Do trips to Open at threshold
	breaker := New(3, 5*time.Second, 2)
	for i := 0; i < 3; i++ {
		breaker.Do(context.Background(), func(ctx context.Context) error {
			return errors.New("failure")
		})
	}
	if breaker.state != Open {
		t.Errorf("expected Open state after threshold, got %v", breaker.state)
	}
}

func Test5(t *testing.T) {
	// Do returns nil on success
	breaker := New(3, 5*time.Second, 2)
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

func Test6(t *testing.T) {
	// Do returns error on failure
	breaker := New(3, 5*time.Second, 2)
	testErr := errors.New("test error")
	err := breaker.Do(context.Background(), func(ctx context.Context) error {
		return testErr
	})
	if err != testErr {
		t.Errorf("expected test error, got %v", err)
	}
}

func Test7(t *testing.T) {
	// Do stays Closed below threshold
	breaker := New(3, 5*time.Second, 2)
	for i := 0; i < 2; i++ {
		breaker.Do(context.Background(), func(ctx context.Context) error {
			return errors.New("failure")
		})
	}
	if breaker.state != Closed {
		t.Errorf("expected Closed state below threshold, got %v", breaker.state)
	}
}

func Test8(t *testing.T) {
	// Do success resets after failures
	breaker := New(3, 5*time.Second, 2)
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return errors.New("failure")
	})
	breaker.Do(context.Background(), func(ctx context.Context) error {
		return nil
	})
	if breaker.errs != 0 {
		t.Errorf("expected errs 0 after success, got %d", breaker.errs)
	}
}

func Test9(t *testing.T) {
	// Do passes context to function
	breaker := New(3, 5*time.Second, 2)
	ctx := context.WithValue(context.Background(), "key", "value")
	breaker.Do(ctx, func(c context.Context) error {
		if c.Value("key") != "value" {
			t.Error("context not passed correctly")
		}
		return nil
	})
}

func Test10(t *testing.T) {
	// Do tripToOpen resets counters
	breaker := New(3, 5*time.Second, 2)
	for i := 0; i < 3; i++ {
		breaker.Do(context.Background(), func(ctx context.Context) error {
			return errors.New("failure")
		})
	}
	if breaker.errs != 0 || breaker.halfCount != 0 {
		t.Error("expected counters reset after tripToOpen")
	}
}
`,
		hint1: `In Closed state on success: set b.errs = 0 to reset the error counter.`,
			hint2: `In Closed state on failure: increment b.errs and check if b.errs >= b.threshold, then call b.tripToOpen().`,
			whyItMatters: `The Closed state is the normal operating mode where the circuit breaker tracks failures and protects against cascading failures.

**Why This Matters:**
- **Failure Detection:** Counts consecutive failures to detect service degradation
- **Threshold Protection:** Opens circuit when failure rate exceeds acceptable limits
- **Fast Recovery:** Resets counter on success, allowing services to recover quickly

**Real-World Example:**
\`\`\`go
// Payment service with 3-failure threshold
paymentBreaker := New(3, 30*time.Second, 2)

// Normal operation - all succeed
paymentBreaker.Do(ctx, processPayment) // Success, errs=0
paymentBreaker.Do(ctx, processPayment) // Success, errs=0

// Service starts degrading
paymentBreaker.Do(ctx, processPayment) // Fail, errs=1
paymentBreaker.Do(ctx, processPayment) // Fail, errs=2
paymentBreaker.Do(ctx, processPayment) // Fail, errs=3 → OPEN

// Now circuit is open, protecting payment service from overload
\`\`\`

**Production Patterns:**
1. **Error Counter Reset:** One success resets the counter, assuming transient failures
2. **Threshold Tuning:** Lower threshold = faster protection, higher = more tolerance
3. **Mutex Safety:** Critical for concurrent requests to same service

**Why Reset on Success:**
If errors are spread out over time with successes between them, the service is likely healthy. Only consecutive failures indicate a real problem.

**Real-World Benefits:**

Not resetting the counter on success means even intermittent errors will eventually open the circuit, causing false positives.`,	order: 1,
	translations: {
		ru: {
			title: 'Выполнение в закрытом состоянии',
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
			b.state = HalfOpen
			b.halfCount = 0
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
}

func (b *Breaker) tripToOpen() {
	b.state = Open                          // помечаем breaker как открытый
	b.openUntil = time.Now().Add(b.openDur) // вычисляем момент для попытки half-open перехода
	b.errs = 0                              // сбрасываем счётчик ошибок для следующей closed фазы
	b.halfCount = 0                         // сбрасываем счётчик успехов half-open
}`,
			description: `Реализуйте поведение метода **Do** для состояния **Closed**.

**Требования:**
1. Выполните функцию \`f\` с контекстом
2. При успехе: сбросьте счётчик ошибок (\`errs = 0\`)
3. При ошибке: увеличьте счётчик ошибок
4. Когда \`errs >= threshold\`: переход в Open через \`tripToOpen()\`
5. Защитите мутации состояния мьютексом

**Машина состояний - Closed:**
\`\`\`
Успех → Сброс счётчика → Closed
Ошибка → Увеличение счётчика → Проверка порога
  Если ошибки >= порога → Open
  Иначе → Closed
\`\`\`

**Ограничения:**
- Используйте мьютекс для потокобезопасности
- Вызывайте \`tripToOpen()\` при достижении порога
- Возвращайте оригинальную ошибку`,
			hint1: `В Closed при успехе: установите b.errs = 0.`,
			hint2: `В Closed при ошибке: увеличьте b.errs и проверьте порог.`,
			whyItMatters: `Closed - нормальный режим работы, где circuit breaker отслеживает сбои и защищает от каскадных отказов.

**Почему важно:**
- **Обнаружение сбоев:** Считает последовательные ошибки
- **Защита по порогу:** Открывает цепь при превышении лимита
- **Быстрое восстановление:** Сбрасывает счётчик при успехе

**Пример из реальной практики:**
\`\`\`go
// Платёжный сервис с порогом в 3 ошибки
paymentBreaker := New(3, 30*time.Second, 2)

// Нормальная работа - все успешны
paymentBreaker.Do(ctx, processPayment) // Успех, errs=0
paymentBreaker.Do(ctx, processPayment) // Успех, errs=0

// Сервис начинает деградировать
paymentBreaker.Do(ctx, processPayment) // Ошибка, errs=1
paymentBreaker.Do(ctx, processPayment) // Ошибка, errs=2
paymentBreaker.Do(ctx, processPayment) // Ошибка, errs=3 → OPEN

// Теперь цепь открыта, защищая платёжный сервис от перегрузки
\`\`\`

**Продакшен паттерны:**
1. **Сброс счётчика ошибок:** Один успех сбрасывает счётчик, предполагая временные сбои
2. **Настройка порога:** Низкий порог = быстрая защита, высокий = больше толерантности
3. **Безопасность мьютекса:** Критично для конкурентных запросов к одному сервису

**Практические преимущества:**
Если ошибки распределены во времени с успехами между ними, сервис скорее всего здоров. Только последовательные ошибки указывают на реальную проблему.

Не сбрасывать счётчик при успехе означает, что даже редкие ошибки в конечном итоге откроют цепь, вызывая ложные срабатывания.`
		},
		uz: {
			title: `Yopiq holatda bajarish`,
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
			b.state = HalfOpen
			b.halfCount = 0
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
}

func (b *Breaker) tripToOpen() {
	b.state = Open                          // breakerni ochiq deb belgilaymiz
	b.openUntil = time.Now().Add(b.openDur) // half-open o'tish urinishi uchun vaqtni hisoblaymiz
	b.errs = 0                              // keyingi closed bosqichi uchun xato hisoblagichini qayta o'rnatamiz
	b.halfCount = 0                         // half-open muvaffaqiyat hisoblagichini qayta o'rnatamiz
}`,
			description: `**Closed** holati uchun **Do** metodi xatti-harakatini amalga oshiring.

**Talablar:**
1. Berilgan \`f\` funksiyasini kontekst bilan bajaring
2. Muvaffaqiyatda: xato hisoblagichini qayta o'rnating (\`errs = 0\`)
3. Muvaffaqiyatsizlikda: xato hisoblagichini oshiring
4. \`errs >= threshold\` bo'lganda: \`tripToOpen()\` orqali Open holatiga o'ting
5. Holat mutatsiyalarini mutex bilan himoyalang

**Holat mashinasi - Closed:**
\`\`\`
Muvaffaqiyat → Hisoblagichni qayta o'rnatish → Closed
Muvaffaqiyatsizlik → Hisoblagichni oshirish → Chegarani tekshirish
  Agar xatolar >= chegara → Open
  Aks holda → Closed
\`\`\`

**Cheklovlar:**
- Thread xavfsizligi uchun mutex ishlating
- Chegaraga yetganda \`tripToOpen()\` ni chaqiring
- Asl xatoni qaytaring`,
			hint1: `Closed da muvaffaqiyatda: b.errs = 0 o'rnating.`,
			hint2: `Closed da muvaffaqiyatsizlikda: b.errs ni oshiring va chegarani tekshiring.`,
			whyItMatters: `Closed - circuit breaker nosozliklarni kuzatib, kaskadli nosozliklardan himoya qiluvchi normal ish rejimi.

**Nima uchun bu muhim:**
- **Nosozlikni aniqlash:** Xizmat degradatsiyasini aniqlash uchun ketma-ket nosozliklarni hisoblaydi
- **Chegara himoyasi:** Nosozlik darajasi qabul qilinadigan chegaradan oshganda zanjirni ochadi
- **Tez tiklash:** Muvaffaqiyatda hisoblagichni qayta o'rnatadi

**Amaliy misoldan:**
\`\`\`go
// 3 ta xato chegarasi bilan to'lov xizmati
paymentBreaker := New(3, 30*time.Second, 2)

// Oddiy ish - hammasi muvaffaqiyatli
paymentBreaker.Do(ctx, processPayment) // Muvaffaqiyat, errs=0
paymentBreaker.Do(ctx, processPayment) // Muvaffaqiyat, errs=0

// Xizmat degradatsiya bo'la boshlaydi
paymentBreaker.Do(ctx, processPayment) // Xato, errs=1
paymentBreaker.Do(ctx, processPayment) // Xato, errs=2
paymentBreaker.Do(ctx, processPayment) // Xato, errs=3 → OPEN

// Endi zanjir ochiq, to'lov xizmatini ortiqcha yuklanishdan himoya qiladi
\`\`\`

**Ishlab chiqarish patternlari:**
1. **Xato hisoblagichini qayta o'rnatish:** Bitta muvaffaqiyat hisoblagichni qayta o'rnatadi, vaqtinchalik nosozliklarni taxmin qiladi
2. **Chegarani sozlash:** Past chegara = tez himoya, yuqori = ko'proq bardoshlilik
3. **Mutex xavfsizligi:** Bir xizmatga parallel so'rovlar uchun muhim

**Amaliy foydalari:**
Agar xatolar vaqt bo'yicha muvaffaqiyatlar bilan tarqalgan bo'lsa, xizmat sog'lom bo'lishi mumkin. Faqat ketma-ket xatolar haqiqiy muammoni ko'rsatadi.

Muvaffaqiyatda hisoblagichni qayta o'rnatmaslik, hatto vaqti-vaqti bilan xatolar ham oxir-oqibatda zanjirni ochishini anglatadi, bu yolg'on ijobiy natijalarni keltirib chiqaradi.`
		}
	}
};

export default task;
