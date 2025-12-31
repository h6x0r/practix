import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-square-stage',
	title: 'SquareStage',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'stage', 'context'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **SquareStage** that returns a pipeline Stage function for squaring numbers with context cancellation.

**Type Definition:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Requirements:**
1. Create function \`SquareStage(workers int) Stage\`
2. Handle workers <= 0 (default to 1 worker)
3. Return a Stage function that takes (ctx, in) and returns output channel
4. Launch 'workers' goroutines that respect context cancellation
5. Each worker uses nested select for ctx.Done() checks
6. Use sync.WaitGroup to coordinate workers
7. Close output when all workers finish
8. Return output channel immediately

**Example:**
\`\`\`go
ctx := context.Background()
stage := SquareStage(2) // Create stage with 2 workers

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Output: 1 4 9 16 25 (order may vary)

// With cancellation
ctx, cancel := context.WithCancel(context.Background())
stage = SquareStage(3)
in = GenWithContext(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out = stage(ctx, in)

count := 0
for v := range out {
    fmt.Println(v)
    count++
    if count == 3 {
        cancel() // Cancel after 3 results
    }
}
// Output: First 3 squared values (then stops)
\`\`\`

**Constraints:**
- Must return Stage function type
- Must use nested select for proper cancellation
- Must check ctx.Done() on both receive and send
- Must use sync.WaitGroup`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

type Stage func(context.Context, <-chan int) <-chan int

// TODO: Implement SquareStage
func SquareStage(workers int) Stage {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Stage func(context.Context, <-chan int) <-chan int

func SquareStage(workers int) Stage {
	if workers <= 0 {                                           // Validate worker count
		workers = 1                                         // Default to 1
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Return Stage function
		out := make(chan int)                               // Create output
		var wg sync.WaitGroup                               // WaitGroup for workers
		wg.Add(workers)                                     // Add worker count
		for i := 0; i < workers; i++ {                      // Launch workers
			go func() {                                 // Worker goroutine
				defer wg.Done()                     // Mark done
				for {                               // Infinite loop
					select {
					case <-ctx.Done():          // Context cancelled
						return              // Exit worker
					case v, ok := <-in:         // Read from input
						if !ok {            // Channel closed
							return      // Exit worker
						}
						select {
						case <-ctx.Done():  // Check again before send
							return      // Exit worker
						case out <- v * v:  // Send squared value
						}
					}
				}
			}()
		}
		go func() {                                         // Closer goroutine
			wg.Wait()                                   // Wait for workers
			close(out)                                  // Close output
		}()
		return out                                          // Return immediately
	}
}`,
			hint1: `Return a function that matches the Stage signature: func(context.Context, <-chan int) <-chan int.`,
			hint2: `Use nested select: outer select reads from in or ctx.Done(), inner select sends to out or ctx.Done(). This ensures cancellation is respected at both receive and send points.`,
			testCode: `package concurrency

import (
	"context"
	"sort"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	stage := SquareStage(1)
	in := make(chan int)
	close(in)
	out := stage(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from closed input, got %d", count)
	}
}

func Test2(t *testing.T) {
	stage := SquareStage(1)
	in := make(chan int, 1)
	in <- 5
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 25 {
		t.Errorf("expected 25, got %d", v)
	}
}

func Test3(t *testing.T) {
	stage := SquareStage(3)
	in := make(chan int, 5)
	for _, v := range []int{1, 2, 3, 4, 5} {
		in <- v
	}
	close(in)
	out := stage(context.Background(), in)
	results := make([]int, 0, 5)
	for v := range out {
		results = append(results, v)
	}
	sort.Ints(results)
	expected := []int{1, 4, 9, 16, 25}
	for i, v := range results {
		if v != expected[i] {
			t.Errorf("expected %v, got %v", expected, results)
			break
		}
	}
}

func Test4(t *testing.T) {
	stage := SquareStage(0)
	in := make(chan int, 3)
	in <- 2
	in <- 3
	in <- 4
	close(in)
	out := stage(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 values with 0 workers, got %d", count)
	}
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stage := SquareStage(2)
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)
	out := stage(ctx, in)
	count := 0
	for range out {
		count++
	}
	if count > 5 {
		t.Errorf("expected <= 5 values with cancelled context, got %d", count)
	}
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	stage := SquareStage(2)
	in := make(chan int, 3)
	in <- 3
	in <- 4
	in <- 5
	close(in)
	out := stage(ctx, in)
	for range out {
	}
}

func Test7(t *testing.T) {
	stage := SquareStage(-5)
	in := make(chan int, 1)
	in <- 7
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 49 {
		t.Errorf("expected 49, got %d", v)
	}
}

func Test8(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		stage := SquareStage(2)
		in := make(chan int)
		close(in)
		_ = stage(context.Background(), in)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("SquareStage should return immediately")
	}
}

func Test9(t *testing.T) {
	stage := SquareStage(1)
	in := make(chan int, 1)
	in <- 0
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 0 {
		t.Errorf("expected 0, got %d", v)
	}
}

func Test10(t *testing.T) {
	stage := SquareStage(1)
	in := make(chan int, 1)
	in <- -4
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 16 {
		t.Errorf("expected 16, got %d", v)
	}
}
`,
	whyItMatters: `SquareStage demonstrates composable pipeline stages with context cancellation, enabling building complex data processing pipelines from reusable components.

**Why Stage Pattern:**
- **Composability:** Chain stages to build complex pipelines
- **Reusability:** Same stage can be used in different pipelines
- **Cancellation:** Context propagates through all stages
- **Flexibility:** Easily add, remove, or reorder stages

**Production Pattern:**
\`\`\`go
// Type definition
type Stage func(context.Context, <-chan int) <-chan int

// Transform stage with error handling
func TransformStage(transform func(int) (int, error), workers int) Stage {
    if workers <= 0 {
        workers = 1
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        result, err := transform(v)
                        if err != nil {
                            continue // Skip invalid values
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Database enrichment stage
func EnrichFromDB(db *sql.DB, workers int) Stage {
    if workers <= 0 {
        workers = 10
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Enrich with DB data
                        var enrichedValue int
                        err := db.QueryRowContext(ctx, "SELECT value FROM data WHERE id = ?", id).Scan(&enrichedValue)
                        if err != nil {
                            continue
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- enrichedValue:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// API call stage with rate limiting
func APICallStage(client *http.Client, workers int, rateLimit time.Duration) Stage {
    if workers <= 0 {
        workers = 5
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        limiter := time.NewTicker(rateLimit)
        defer limiter.Stop()

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Rate limiting
                        select {
                        case <-ctx.Done():
                            return
                        case <-limiter.C:
                        }

                        // Make API call
                        resp, err := client.Get(fmt.Sprintf("http://api/data/%d", id))
                        if err != nil {
                            continue
                        }
                        defer resp.Body.Close()

                        var result int
                        json.NewDecoder(resp.Body).Decode(&result)

                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Composing multiple stages
func ProcessPipeline() {
    ctx := context.Background()

    // Create stages
    square := SquareStage(3)
    multiply := MultiplyStage(2)
    filter := FilterStage(func(n int) bool { return n > 10 })

    // Build pipeline
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := filter(ctx, multiply(ctx, square(ctx, source)))

    for v := range result {
        fmt.Println(v) // Only values > 10
    }
}

// Pipeline with timeout
func ProcessWithTimeout(timeout time.Duration) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    stage1 := SquareStage(2)
    stage2 := MultiplyStage(3)

    source := GenWithContext(ctx, 1, 2, 3, 4, 5)
    result := stage2(ctx, stage1(ctx, source))

    for v := range result {
        fmt.Println(v)
    }
}
\`\`\`

**Real-World Benefits:**
- **Modularity:** Each stage is independent and testable
- **Performance Tuning:** Adjust workers per stage based on bottlenecks
- **Error Isolation:** Errors in one stage don't affect others
- **Pipeline Flexibility:** Easy to add caching, logging, metrics stages

**Stage Design Patterns:**
- **Transform:** Modify each value (square, multiply, format)
- **Filter:** Select subset of values (predicate, validation)
- **Enrich:** Add data from external sources (DB, API, cache)
- **Aggregate:** Combine multiple values (sum, count, average)
- **Split:** Route to different outputs (fan-out by condition)

**Worker Tuning by Stage Type:**
- **CPU-Intensive (square, hash):** workers = NumCPU()
- **I/O-Bound (DB, API):** workers = 10-100
- **Memory-Intensive:** workers = based on available memory
- **Rate-Limited (API):** workers = rate limit / throughput

Without composable stages, building pipelines requires writing monolithic code that's hard to test, modify, and reuse across different data processing scenarios.`,	order: 3,
	translations: {
		ru: {
			title: 'Этап возведения в квадрат в pipeline',
			description: `Реализуйте **SquareStage**, который возвращает функцию Stage pipeline для возведения в квадрат с отменой контекста.

**Определение типа:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Требования:**
1. Создайте функцию \`SquareStage(workers int) Stage\`
2. Обработайте workers <= 0 (по умолчанию 1 рабочий)
3. Верните функцию Stage которая принимает (ctx, in) и возвращает выходной канал
4. Запустите 'workers' горутин которые учитывают отмену контекста
5. Каждый рабочий использует вложенный select для проверок ctx.Done()
6. Используйте sync.WaitGroup для координации рабочих
7. Закройте выход когда все рабочие закончат
8. Верните выходной канал немедленно

**Пример:**
\`\`\`go
ctx := context.Background()
stage := SquareStage(2)

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Вывод: 1 4 9 16 25
\`\`\`

**Ограничения:**
- Должен возвращать функцию типа Stage
- Должен использовать вложенный select для правильной отмены
- Должен проверять ctx.Done() при чтении и отправке
- Должен использовать sync.WaitGroup`,
			hint1: `Верните функцию соответствующую сигнатуре Stage: func(context.Context, <-chan int) <-chan int.`,
			hint2: `Используйте вложенный select: внешний select читает из in или ctx.Done(), внутренний select отправляет в out или ctx.Done().`,
			whyItMatters: `SquareStage демонстрирует компонуемые стадии pipeline с отменой контекста, позволяя строить сложные конвейеры обработки данных из переиспользуемых компонентов.

**Зачем паттерн Stage:**
- **Компонуемость:** Соединение стадий для построения сложных конвейеров
- **Переиспользуемость:** Одна и та же стадия в разных конвейерах
- **Отмена:** Контекст распространяется через все стадии
- **Гибкость:** Легко добавлять, удалять или переупорядочивать стадии

**Продакшен паттерн:**
\`\`\`go
// Определение типа
type Stage func(context.Context, <-chan int) <-chan int

// Стадия трансформации с обработкой ошибок
func TransformStage(transform func(int) (int, error), workers int) Stage {
    if workers <= 0 {
        workers = 1
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        result, err := transform(v)
                        if err != nil {
                            continue // Пропускаем невалидные значения
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Стадия обогащения из БД
func EnrichFromDB(db *sql.DB, workers int) Stage {
    if workers <= 0 {
        workers = 10
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Обогащение данными из БД
                        var enrichedValue int
                        err := db.QueryRowContext(ctx, "SELECT value FROM data WHERE id = ?", id).Scan(&enrichedValue)
                        if err != nil {
                            continue
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- enrichedValue:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Стадия API вызовов с ограничением частоты
func APICallStage(client *http.Client, workers int, rateLimit time.Duration) Stage {
    if workers <= 0 {
        workers = 5
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        limiter := time.NewTicker(rateLimit)
        defer limiter.Stop()

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Ограничение частоты
                        select {
                        case <-ctx.Done():
                            return
                        case <-limiter.C:
                        }

                        // Выполнение API вызова
                        resp, err := client.Get(fmt.Sprintf("http://api/data/%d", id))
                        if err != nil {
                            continue
                        }
                        defer resp.Body.Close()

                        var result int
                        json.NewDecoder(resp.Body).Decode(&result)

                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Композиция нескольких стадий
func ProcessPipeline() {
    ctx := context.Background()

    // Создание стадий
    square := SquareStage(3)
    multiply := MultiplyStage(2)
    filter := FilterStage(func(n int) bool { return n > 10 })

    // Построение конвейера
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := filter(ctx, multiply(ctx, square(ctx, source)))

    for v := range result {
        fmt.Println(v) // Только значения > 10
    }
}

// Конвейер с таймаутом
func ProcessWithTimeout(timeout time.Duration) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    stage1 := SquareStage(2)
    stage2 := MultiplyStage(3)

    source := GenWithContext(ctx, 1, 2, 3, 4, 5)
    result := stage2(ctx, stage1(ctx, source))

    for v := range result {
        fmt.Println(v)
    }
}
\`\`\`

**Практические преимущества:**
- **Модульность:** Каждая стадия независима и тестируема
- **Настройка производительности:** Настройка workers для каждой стадии на основе узких мест
- **Изоляция ошибок:** Ошибки в одной стадии не влияют на другие
- **Гибкость конвейера:** Легко добавлять стадии кеширования, логирования, метрик

**Паттерны проектирования стадий:**
- **Transform:** Изменение каждого значения (квадрат, умножение, форматирование)
- **Filter:** Выбор подмножества значений (предикат, валидация)
- **Enrich:** Добавление данных из внешних источников (БД, API, кэш)
- **Aggregate:** Объединение нескольких значений (сумма, счетчик, среднее)
- **Split:** Маршрутизация к разным выходам (fan-out по условию)

**Настройка Workers по типу стадии:**
- **CPU-Intensive (квадрат, хэш):** workers = NumCPU()
- **I/O-Bound (БД, API):** workers = 10-100
- **Memory-Intensive:** workers = на основе доступной памяти
- **Rate-Limited (API):** workers = rate limit / throughput

Без компонуемых стадий построение конвейеров требует написания монолитного кода который сложно тестировать, изменять и переиспользовать в разных сценариях обработки данных.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Stage func(context.Context, <-chan int) <-chan int

func SquareStage(workers int) Stage {
	if workers <= 0 {                                           // Проверяем количество рабочих
		workers = 1                                         // По умолчанию 1
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Возвращаем функцию Stage
		out := make(chan int)                               // Создаём выход
		var wg sync.WaitGroup                               // WaitGroup для рабочих
		wg.Add(workers)                                     // Добавляем количество рабочих
		for i := 0; i < workers; i++ {                      // Запускаем рабочих
			go func() {                                 // Горутина рабочего
				defer wg.Done()                     // Отмечаем завершение
				for {                               // Бесконечный цикл
					select {
					case <-ctx.Done():          // Контекст отменён
						return              // Выходим из рабочего
					case v, ok := <-in:         // Читаем из входа
						if !ok {            // Канал закрыт
							return      // Выходим из рабочего
						}
						select {
						case <-ctx.Done():  // Проверяем снова перед отправкой
							return      // Выходим из рабочего
						case out <- v * v:  // Отправляем квадрат значения
						}
					}
				}
			}()
		}
		go func() {                                         // Горутина закрывателя
			wg.Wait()                                   // Ждём рабочих
			close(out)                                  // Закрываем выход
		}()
		return out                                          // Возвращаем немедленно
	}
}`
		},
		uz: {
			title: 'Pipeline da kvadratga ko\'tarish bosqichi',
			description: `Kontekst bekor qilish bilan raqamlarni kvadratga ko'tarish uchun pipeline Stage funksiyasini qaytaruvchi **SquareStage** ni amalga oshiring.

**Tur ta'rifi:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Talablar:**
1. \`SquareStage(workers int) Stage\` funksiyasini yarating
2. workers <= 0 ni ishlang (standart 1 worker)
3. (ctx, in) qabul qiladigan va chiqish kanalini qaytaruvchi Stage funksiyasini qaytaring
4. Kontekst bekor qilinishini hisobga oladigan 'workers' goroutinelarni ishga tushiring
5. Har bir worker ctx.Done() tekshiruvlari uchun ichki selectdan foydalanadi
6. Workerlarni muvofiqlashtirish uchun sync.WaitGroup dan foydalaning
7. Barcha workerlar tugaganda chiqishni yoping
8. Chiqish kanalini darhol qaytaring

**Misol:**
\`\`\`go
ctx := context.Background()
stage := SquareStage(2)

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Natija: 1 4 9 16 25
\`\`\`

**Cheklovlar:**
- Stage turi funksiyasini qaytarishi kerak
- To'g'ri bekor qilish uchun ichki selectdan foydalanishi kerak
- O'qish va yuborishda ctx.Done() ni tekshirishi kerak
- sync.WaitGroup dan foydalanishi kerak`,
			hint1: `Stage imzosiga mos funksiyani qaytaring: func(context.Context, <-chan int) <-chan int.`,
			hint2: `Ichki selectdan foydalaning: tashqi select in yoki ctx.Done() dan o'qiydi, ichki select out ga yoki ctx.Done() ga yuboradi.`,
			whyItMatters: `SquareStage kontekst bekor qilish bilan tuzilishi mumkin bo'lgan pipeline bosqichlarini namoyish etadi, qayta ishlatilishi mumkin komponentlardan murakkab ma'lumotlarni qayta ishlash pipelinelarini qurishga imkon beradi.

**Nega Stage patterni kerak:**
- **Kompozitsiyalik:** Murakkab pipelinelar qurish uchun bosqichlarni bog'lash
- **Qayta ishlatish:** Turli pipelinelarda bir xil bosqich
- **Bekor qilish:** Kontekst barcha bosqichlar orqali tarqaladi
- **Moslashuvchanlik:** Bosqichlarni osongina qo'shish, o'chirish yoki qayta tartibga solish

**Ishlab chiqarish patterni:**
\`\`\`go
// Tur ta'rifi
type Stage func(context.Context, <-chan int) <-chan int

// Xatolarni qayta ishlash bilan transformatsiya bosqichi
func TransformStage(transform func(int) (int, error), workers int) Stage {
    if workers <= 0 {
        workers = 1
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        result, err := transform(v)
                        if err != nil {
                            continue // Yaroqsiz qiymatlarni o'tkazib yuborish
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Ma'lumotlar bazasidan boyitish bosqichi
func EnrichFromDB(db *sql.DB, workers int) Stage {
    if workers <= 0 {
        workers = 10
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Ma'lumotlar bazasidan boyitish
                        var enrichedValue int
                        err := db.QueryRowContext(ctx, "SELECT value FROM data WHERE id = ?", id).Scan(&enrichedValue)
                        if err != nil {
                            continue
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- enrichedValue:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Tezlik cheklash bilan API chaqiruv bosqichi
func APICallStage(client *http.Client, workers int, rateLimit time.Duration) Stage {
    if workers <= 0 {
        workers = 5
    }
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        var wg sync.WaitGroup
        wg.Add(workers)

        limiter := time.NewTicker(rateLimit)
        defer limiter.Stop()

        for i := 0; i < workers; i++ {
            go func() {
                defer wg.Done()
                for {
                    select {
                    case <-ctx.Done():
                        return
                    case id, ok := <-in:
                        if !ok {
                            return
                        }
                        // Tezlik cheklash
                        select {
                        case <-ctx.Done():
                            return
                        case <-limiter.C:
                        }

                        // API chaqiruvini amalga oshirish
                        resp, err := client.Get(fmt.Sprintf("http://api/data/%d", id))
                        if err != nil {
                            continue
                        }
                        defer resp.Body.Close()

                        var result int
                        json.NewDecoder(resp.Body).Decode(&result)

                        select {
                        case <-ctx.Done():
                            return
                        case out <- result:
                        }
                    }
                }
            }()
        }

        go func() {
            wg.Wait()
            close(out)
        }()
        return out
    }
}

// Bir nechta bosqichlarni kompozitsiyalash
func ProcessPipeline() {
    ctx := context.Background()

    // Bosqichlarni yaratish
    square := SquareStage(3)
    multiply := MultiplyStage(2)
    filter := FilterStage(func(n int) bool { return n > 10 })

    // Pipelineni qurish
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := filter(ctx, multiply(ctx, square(ctx, source)))

    for v := range result {
        fmt.Println(v) // Faqat > 10 qiymatlar
    }
}

// Timeout bilan pipeline
func ProcessWithTimeout(timeout time.Duration) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    stage1 := SquareStage(2)
    stage2 := MultiplyStage(3)

    source := GenWithContext(ctx, 1, 2, 3, 4, 5)
    result := stage2(ctx, stage1(ctx, source))

    for v := range result {
        fmt.Println(v)
    }
}
\`\`\`

**Amaliy foydalari:**
- **Modullik:** Har bir bosqich mustaqil va testlanishi mumkin
- **Unumdorlikni sozlash:** Tor joylar asosida har bir bosqich uchun workerlarni sozlash
- **Xatolarni ajratish:** Bir bosqichdagi xatolar boshqalarga ta'sir qilmaydi
- **Pipeline moslashuvchanligi:** Keshlash, logging, metrikalar bosqichlarini osongina qo'shish

**Bosqichlarni loyihalash patternlari:**
- **Transform:** Har bir qiymatni o'zgartirish (kvadrat, ko'paytirish, formatlash)
- **Filter:** Qiymatlar kichik to'plamini tanlash (predikat, validatsiya)
- **Enrich:** Tashqi manbalardan ma'lumotlar qo'shish (DB, API, kesh)
- **Aggregate:** Bir nechta qiymatlarni birlashtirish (yig'indi, hisoblagich, o'rtacha)
- **Split:** Turli chiqishlarga yo'naltirish (shart bo'yicha fan-out)

**Bosqich turi bo'yicha Workerlarni sozlash:**
- **CPU-Intensive (kvadrat, xesh):** workers = NumCPU()
- **I/O-Bound (DB, API):** workers = 10-100
- **Memory-Intensive:** workers = mavjud xotiraga asoslangan
- **Rate-Limited (API):** workers = tezlik limiti / o'tkazuvchanlik

Kompozitsiyalanadigan bosqichlar bo'lmasa, pipelinelar qurish turli ma'lumotlarni qayta ishlash stsenariylarda testlash, o'zgartirish va qayta ishlatish qiyin bo'lgan monolit kodini yozishni talab qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type Stage func(context.Context, <-chan int) <-chan int

func SquareStage(workers int) Stage {
	if workers <= 0 {                                           // Workerlar sonini tekshiramiz
		workers = 1                                         // Standart 1
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Stage funksiyasini qaytaramiz
		out := make(chan int)                               // Chiqish yaratamiz
		var wg sync.WaitGroup                               // Workerlar uchun WaitGroup
		wg.Add(workers)                                     // Workerlar sonini qo'shamiz
		for i := 0; i < workers; i++ {                      // Workerlarni ishga tushiramiz
			go func() {                                 // Worker goroutinesi
				defer wg.Done()                     // Tugaganini belgilaymiz
				for {                               // Cheksiz tsikl
					select {
					case <-ctx.Done():          // Kontekst bekor qilindi
						return              // Workerdan chiqamiz
					case v, ok := <-in:         // Kirishdan o'qiymiz
						if !ok {            // Kanal yopilgan
							return      // Workerdan chiqamiz
						}
						select {
						case <-ctx.Done():  // Yuborishdan oldin yana tekshiramiz
							return      // Workerdan chiqamiz
						case out <- v * v:  // Kvadrat qiymatni yuboramiz
						}
					}
				}
			}()
		}
		go func() {                                         // Yopuvchi goroutine
			wg.Wait()                                   // Workerlarni kutamiz
			close(out)                                  // Chiqishni yopamiz
		}()
		return out                                          // Darhol qaytaramiz
	}
}`
		}
	}
};

export default task;
