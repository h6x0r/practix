import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-circuit-breaker-new',
	title: 'Circuit Breaker Constructor',
	difficulty: 'easy',	tags: ['go', 'circuit-breaker', 'resilience'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **New** constructor for the circuit breaker.

**Requirements:**
1. Create constructor that accepts \`threshold\`, \`openDur\`, and \`halfMax\` parameters
2. Initialize breaker in \`Closed\` state
3. Store all configuration parameters in the Breaker struct
4. Return pointer to initialized Breaker

**Example:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)
// threshold=3 failures triggers Open state
// openDur=5s is the cooldown duration
// halfMax=2 successful requests in HalfOpen closes the circuit
\`\`\`

**Constraints:**
- Initial state must be Closed
- All counters (errs, halfCount) start at 0
- openDur will be reused each time circuit opens`,
	initialCode: `package circuitx

import "time"

// TODO: Implement New constructor
func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	// TODO: Implement
}`,
	solutionCode: `package circuitx

import "time"

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{         // initialize breaker in closed state with provided parameters
		state:     Closed,   // start in closed state (accepting requests)
		threshold: threshold, // number of failures before opening circuit
		openDur:   openDur,  // duration to keep circuit open before trying half-open
		halfMax:   halfMax,  // successful requests needed in half-open to close circuit
	}
}`,
		testCode: `package circuitx

import (
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// New returns non-nil breaker
	breaker := New(3, 5*time.Second, 2)
	if breaker == nil {
		t.Error("expected non-nil breaker")
	}
}

func Test2(t *testing.T) {
	// New initializes state to Closed
	breaker := New(3, 5*time.Second, 2)
	if breaker.state != Closed {
		t.Errorf("expected Closed state, got %v", breaker.state)
	}
}

func Test3(t *testing.T) {
	// New sets threshold correctly
	breaker := New(5, 5*time.Second, 2)
	if breaker.threshold != 5 {
		t.Errorf("expected threshold 5, got %d", breaker.threshold)
	}
}

func Test4(t *testing.T) {
	// New sets openDur correctly
	breaker := New(3, 10*time.Second, 2)
	if breaker.openDur != 10*time.Second {
		t.Errorf("expected openDur 10s, got %v", breaker.openDur)
	}
}

func Test5(t *testing.T) {
	// New sets halfMax correctly
	breaker := New(3, 5*time.Second, 4)
	if breaker.halfMax != 4 {
		t.Errorf("expected halfMax 4, got %d", breaker.halfMax)
	}
}

func Test6(t *testing.T) {
	// New initializes errs to 0
	breaker := New(3, 5*time.Second, 2)
	if breaker.errs != 0 {
		t.Errorf("expected errs 0, got %d", breaker.errs)
	}
}

func Test7(t *testing.T) {
	// New initializes halfCount to 0
	breaker := New(3, 5*time.Second, 2)
	if breaker.halfCount != 0 {
		t.Errorf("expected halfCount 0, got %d", breaker.halfCount)
	}
}

func Test8(t *testing.T) {
	// New with different parameters
	breaker := New(10, 30*time.Second, 5)
	if breaker.threshold != 10 || breaker.openDur != 30*time.Second || breaker.halfMax != 5 {
		t.Error("expected correct parameters")
	}
}

func Test9(t *testing.T) {
	// New with minimum values
	breaker := New(1, 1*time.Millisecond, 1)
	if breaker.threshold != 1 || breaker.halfMax != 1 {
		t.Error("expected minimum values")
	}
}

func Test10(t *testing.T) {
	// New returns pointer
	breaker := New(3, 5*time.Second, 2)
	var _ *Breaker = breaker
}
`,
		hint1: `Initialize the Breaker struct with all configuration fields set.`,
			hint2: `Set state to Closed and leave counters (errs, halfCount) at zero.`,
			whyItMatters: `The circuit breaker constructor sets up the initial state and configuration for protecting services from cascading failures.

**Why This Matters:**
- **Fail-fast:** Prevents overwhelming failing services with repeated requests
- **Self-healing:** Automatically attempts recovery after cooldown period
- **Configuration:** Tunable parameters adapt to different service characteristics

**Real-World Example:**
\`\`\`go
// API with strict SLA - open quickly
apiBreaker := New(2, 30*time.Second, 1)

// Database with retry tolerance - allow more failures
dbBreaker := New(5, 10*time.Second, 3)

// External service - long cooldown
extBreaker := New(3, 60*time.Second, 2)
\`\`\`

**Production Pattern:**
Different services need different thresholds:
- **threshold:** How many failures before opening (lower = more sensitive)
- **openDur:** How long to wait before retry (longer = more conservative)
- **halfMax:** How many successes to close circuit (higher = more confident)

**Real-World Benefits:**
Without proper initialization, a circuit breaker can't protect your service from cascade failures that bring down entire systems.`,	order: 0,
	translations: {
		ru: {
			title: 'Конструктор Circuit Breaker',
			solutionCode: `package circuitx

import "time"

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{         // инициализируем breaker в закрытом состоянии с указанными параметрами
		state:     Closed,   // начинаем в закрытом состоянии (принимаем запросы)
		threshold: threshold, // количество сбоев перед открытием цепи
		openDur:   openDur,  // длительность ожидания перед попыткой half-open
		halfMax:   halfMax,  // успешных запросов для закрытия цепи в half-open
	}
}`,
			description: `Реализуйте конструктор **New** для circuit breaker.

**Требования:**
1. Создайте конструктор с параметрами \`threshold\`, \`openDur\` и \`halfMax\`
2. Инициализируйте breaker в состоянии \`Closed\`
3. Сохраните все параметры конфигурации в структуре Breaker
4. Верните указатель на инициализированный Breaker

**Пример:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)
// threshold=3 сбоя переводят в Open
// openDur=5с это период охлаждения
// halfMax=2 успешных запроса в HalfOpen закрывают цепь
\`\`\`

**Ограничения:**
- Начальное состояние должно быть Closed
- Все счётчики начинаются с 0
- openDur переиспользуется при каждом открытии`,
			hint1: `Инициализируйте структуру Breaker со всеми полями конфигурации.`,
			hint2: `Установите state в Closed, счётчики оставьте нулевыми.`,
			whyItMatters: `Конструктор circuit breaker настраивает начальное состояние для защиты от каскадных сбоев.

**Почему важно:**
- **Fail-fast:** Предотвращает перегрузку падающих сервисов
- **Самовосстановление:** Автоматически пытается восстановиться
- **Конфигурация:** Настраиваемые параметры под разные сервисы

**Пример из реальной практики:**
\`\`\`go
// API со строгим SLA - открываем быстро
apiBreaker := New(2, 30*time.Second, 1)

// База данных с толерантностью к retry - разрешаем больше ошибок
dbBreaker := New(5, 10*time.Second, 3)

// Внешний сервис - долгий cooldown
extBreaker := New(3, 60*time.Second, 2)
\`\`\`

**Продакшен паттерн:**
Разным сервисам нужны разные пороги:
- **threshold:** Сколько ошибок до открытия (ниже = более чувствительный)
- **openDur:** Как долго ждать перед retry (дольше = более консервативный)
- **halfMax:** Сколько успехов для закрытия цепи (выше = более уверенный)

**Практические преимущества:**
Без правильной инициализации circuit breaker не сможет защитить ваш сервис от каскадных сбоев, которые могут обрушить всю систему.`
		},
		uz: {
			title: `Circuit Breaker konstruktori`,
			solutionCode: `package circuitx

import "time"

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{         // breakerni yopiq holatda berilgan parametrlar bilan ishga tushiramiz
		state:     Closed,   // yopiq holatda boshlaymiz (so'rovlarni qabul qiladi)
		threshold: threshold, // zanjirni ochishdan oldingi nosozliklar soni
		openDur:   openDur,  // half-open ga o'tishdan oldin kutish davomiyligi
		halfMax:   halfMax,  // half-open da zanjirni yopish uchun kerak bo'lgan muvaffaqiyatli so'rovlar
	}
}`,
			description: `Circuit breaker uchun **New** konstruktorini amalga oshiring.

**Talablar:**
1. \`threshold\`, \`openDur\` va \`halfMax\` parametrlarini qabul qiluvchi konstruktor yarating
2. Breaker ni \`Closed\` holatida ishga tushiring
3. Barcha konfiguratsiya parametrlarini Breaker strukturasida saqlang
4. Ishga tushirilgan Breaker ga pointer qaytaring

**Misol:**
\`\`\`go
breaker := New(3, 5*time.Second, 2)
// threshold=3 muvaffaqiyatsizlik Open holatini yoqadi
// openDur=5s sovutish davomiyligi
// halfMax=2 muvaffaqiyatli so'rov HalfOpen da zanjirni yopadi
\`\`\`

**Cheklovlar:**
- Boshlang'ich holat Closed bo'lishi kerak
- Barcha hisoblagichlar (errs, halfCount) 0 dan boshlanadi
- openDur har safar zanjir ochilganda qayta ishlatiladi`,
			hint1: `Breaker strukturasini barcha konfiguratsiya maydonlari bilan ishga tushiring.`,
			hint2: `state ni Closed ga o'rnating, hisoblagichlarni nolda qoldiring.`,
			whyItMatters: `Circuit breaker konstruktori kaskadli nosozliklardan himoya qilish uchun boshlang'ich holat va konfiguratsiyani o'rnatadi.

**Nima uchun bu muhim:**
- **Fail-fast:** Muvaffaqiyatsiz xizmatlarga takroriy so'rovlar bilan ortiqcha yuklanishni oldini oladi
- **O'z-o'zini tiklash:** Sovutish davridan keyin avtomatik tiklanishga harakat qiladi
- **Konfiguratsiya:** Sozlanuvchi parametrlar turli xizmat xususiyatlariga moslashadi

**Amaliy misoldan:**
\`\`\`go
// Qattiq SLA bilan API - tez ochilish
apiBreaker := New(2, 30*time.Second, 1)

// Retry bardoshli ma'lumotlar bazasi - ko'proq xatolarga ruxsat
dbBreaker := New(5, 10*time.Second, 3)

// Tashqi xizmat - uzoq cooldown
extBreaker := New(3, 60*time.Second, 2)
\`\`\`

**Ishlab chiqarish patterni:**
Turli xizmatlarga turli chegaralar kerak:
- **threshold:** Ochilishdan oldin nechta xato (pastroq = sezgirroq)
- **openDur:** Retry dan oldin qancha kutish (uzoqroq = konservativroq)
- **halfMax:** Zanjirni yopish uchun nechta muvaffaqiyat (yuqoriroq = ishonchli)

**Amaliy foydalari:**
To'g'ri initsializatsiyasiz circuit breaker sizning xizmatingizni butun tizimni qulatishi mumkin bo'lgan kaskadli nosozliklardan himoya qila olmaydi.`
		}
	}
};

export default task;
