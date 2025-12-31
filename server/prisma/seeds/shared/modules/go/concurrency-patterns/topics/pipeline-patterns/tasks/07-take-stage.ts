import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-take-stage',
	title: 'TakeStage',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'stage', 'limit'],
	estimatedTime: '35m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **TakeStage** that returns a pipeline Stage function for taking only the first N values from a channel.

**Type Definition:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Requirements:**
1. Create function \`TakeStage(n int) Stage\`
2. Return a Stage function that takes (ctx, in) and returns output channel
3. Handle n <= 0 (drain input without sending anything)
4. Send only first n values, then stop
5. After sending n values, drain remaining input (use \`for range in {}\`)
6. Use nested select for ctx.Done() checks
7. Track count of sent values
8. Close output channel properly

**Example:**
\`\`\`go
ctx := context.Background()

// Take first 3 values
take3 := TakeStage(3)
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := take3(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Output: 1 2 3

// Take 0 values (drain without output)
take0 := TakeStage(0)
in = Gen(1, 2, 3, 4, 5)
out = take0(ctx, in)
for v := range out {
    fmt.Println(v)
}
// No output

// Take more than available
take100 := TakeStage(100)
in = Gen(1, 2, 3)
out = take100(ctx, in)
for v := range out {
    fmt.Println(v)
}
// Output: 1 2 3 (all available values)
\`\`\`

**Constraints:**
- Must handle n <= 0 by draining input
- Must drain remaining input after taking n values
- Must use nested select for cancellation
- Must track sent count accurately`,
	initialCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

// TODO: Implement TakeStage
func TakeStage(n int) Stage {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func TakeStage(n int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Return Stage function
		out := make(chan int)                               // Create output channel
		go func() {                                         // Launch goroutine
			defer close(out)                            // Always close output
			if n <= 0 {                                 // Handle n <= 0
				for range in {                      // Drain input
				}
				return                              // Exit
			}
			sent := 0                                   // Track sent count
			for {                                       // Infinite loop
				select {
				case <-ctx.Done():                  // Context cancelled
					return                      // Exit goroutine
				case v, ok := <-in:                 // Read from input
					if !ok {                    // Channel closed
						return              // Exit goroutine
					}
					if sent < n {               // Still need to send
						select {
						case <-ctx.Done():  // Check before send
							return      // Exit goroutine
						case out <- v:      // Send value
							sent++      // Increment counter
						}
					}
					if sent >= n {              // Reached limit
						for range in {      // Drain remaining
						}
						return              // Exit goroutine
					}
				}
			}
		}()
		return out                                          // Return immediately
	}
}`,
	testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func TestTakeStage1(t *testing.T) {
	// Test taking first 3 values from a stream
	ctx := context.Background()
	take3 := TakeStage(3)
	in := make(chan int)

	go func() {
		for i := 1; i <= 10; i++ {
			in <- i
		}
		close(in)
	}()

	out := take3(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 3 {
		t.Errorf("expected 3 values, got %d", len(result))
	}
	if result[0] != 1 || result[1] != 2 || result[2] != 3 {
		t.Errorf("expected [1 2 3], got %v", result)
	}
}

func TestTakeStage2(t *testing.T) {
	// Test taking 0 values (should drain input without output)
	ctx := context.Background()
	take0 := TakeStage(0)
	in := make(chan int)

	go func() {
		for i := 1; i <= 5; i++ {
			in <- i
		}
		close(in)
	}()

	out := take0(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestTakeStage3(t *testing.T) {
	// Test taking negative values (should drain input without output)
	ctx := context.Background()
	takeNeg := TakeStage(-5)
	in := make(chan int)

	go func() {
		for i := 1; i <= 5; i++ {
			in <- i
		}
		close(in)
	}()

	out := takeNeg(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestTakeStage4(t *testing.T) {
	// Test taking more than available
	ctx := context.Background()
	take100 := TakeStage(100)
	in := make(chan int)

	go func() {
		for i := 1; i <= 3; i++ {
			in <- i
		}
		close(in)
	}()

	out := take100(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 3 {
		t.Errorf("expected 3 values, got %d", len(result))
	}
}

func TestTakeStage5(t *testing.T) {
	// Test with context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	take10 := TakeStage(10)
	in := make(chan int)

	go func() {
		for i := 1; i <= 100; i++ {
			select {
			case in <- i:
			case <-time.After(10 * time.Millisecond):
				return
			}
		}
		close(in)
	}()

	out := take10(ctx, in)
	cancel() // Cancel immediately

	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	// Should stop early due to cancellation
	if len(result) > 10 {
		t.Errorf("expected at most 10 values, got %d", len(result))
	}
}

func TestTakeStage6(t *testing.T) {
	// Test taking exactly 1 value
	ctx := context.Background()
	take1 := TakeStage(1)
	in := make(chan int)

	go func() {
		for i := 1; i <= 10; i++ {
			in <- i
		}
		close(in)
	}()

	out := take1(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 1 {
		t.Errorf("expected 1 value, got %d", len(result))
	}
	if result[0] != 1 {
		t.Errorf("expected 1, got %v", result[0])
	}
}

func TestTakeStage7(t *testing.T) {
	// Test with empty input channel
	ctx := context.Background()
	take5 := TakeStage(5)
	in := make(chan int)
	close(in) // Close immediately

	out := take5(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 0 {
		t.Errorf("expected 0 values, got %d", len(result))
	}
}

func TestTakeStage8(t *testing.T) {
	// Test that stage drains remaining input after taking n values
	ctx := context.Background()
	take2 := TakeStage(2)
	in := make(chan int, 10)

	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)

	out := take2(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 2 {
		t.Errorf("expected 2 values, got %d", len(result))
	}
	// Verify channel was drained (goroutine should exit properly)
}

func TestTakeStage9(t *testing.T) {
	// Test taking large number of values
	ctx := context.Background()
	take1000 := TakeStage(1000)
	in := make(chan int)

	go func() {
		for i := 1; i <= 2000; i++ {
			in <- i
		}
		close(in)
	}()

	out := take1000(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	if len(result) != 1000 {
		t.Errorf("expected 1000 values, got %d", len(result))
	}
	if result[0] != 1 || result[999] != 1000 {
		t.Errorf("expected first=1 and last=1000, got first=%d last=%d", result[0], result[999])
	}
}

func TestTakeStage10(t *testing.T) {
	// Test context cancellation during drain phase
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	take2 := TakeStage(2)
	in := make(chan int)

	go func() {
		for i := 1; i <= 1000; i++ {
			select {
			case in <- i:
				time.Sleep(1 * time.Millisecond)
			case <-ctx.Done():
				close(in)
				return
			}
		}
		close(in)
	}()

	out := take2(ctx, in)
	result := []int{}
	for v := range out {
		result = append(result, v)
	}

	// Should get 2 values or less if context cancelled
	if len(result) > 2 {
		t.Errorf("expected at most 2 values, got %d", len(result))
	}
}`,
			hint1: `Handle n <= 0 as a special case: use "for range in {}" to drain the input without sending anything, then return.`,
			hint2: `Keep a "sent" counter. After each successful send, increment it. When sent >= n, use "for range in {}" to drain remaining input, then return.`,
			whyItMatters: `TakeStage enables limiting pipeline output, useful for previews, pagination, sampling, and preventing unbounded data processing.

**Why Take/Limit:**
- **Resource Control:** Prevent processing too much data
- **Pagination:** Implement page-based data loading
- **Preview:** Show first N results before full processing
- **Testing:** Test pipelines with small subsets
- **Rate Limiting:** Control throughput by limiting batch sizes

**Production Pattern:**
\`\`\`go
// Pagination stage
func PaginateStage(page, pageSize int) Stage {
    skip := page * pageSize
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            // Skip to page
            skipped := 0
            for skipped < skip {
                select {
                case <-ctx.Done():
                    return
                case _, ok := <-in:
                    if !ok {
                        return
                    }
                    skipped++
                }
            }

            // Take page
            taken := 0
            for taken < pageSize {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                        taken++
                    }
                }
            }

            // Drain remaining
            for range in {
            }
        }()
        return out
    }
}

// Head (first N) and Tail (last N) stages
func HeadStage(n int) Stage {
    return TakeStage(n) // Just an alias
}

func TailStage(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            buffer := make([]int, 0, n)

            // Collect all values, keeping only last N
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        // Send last N
                        for _, val := range buffer {
                            select {
                            case <-ctx.Done():
                                return
                            case out <- val:
                            }
                        }
                        return
                    }
                    buffer = append(buffer, v)
                    if len(buffer) > n {
                        buffer = buffer[1:] // Keep last N
                    }
                }
            }
        }()
        return out
    }
}

// Top N by score
func TopNStage(n int, score func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            type scored struct {
                value int
                score int
            }
            var items []scored

            // Collect all with scores
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        // Sort and send top N
                        sort.Slice(items, func(i, j int) bool {
                            return items[i].score > items[j].score
                        })

                        limit := n
                        if len(items) < limit {
                            limit = len(items)
                        }

                        for i := 0; i < limit; i++ {
                            select {
                            case <-ctx.Done():
                                return
                            case out <- items[i].value:
                            }
                        }
                        return
                    }
                    items = append(items, scored{v, score(v)})
                }
            }
        }()
        return out
    }
}

// Sample every Nth value
func SampleEveryNth(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    count++
                    if count%n == 0 {
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                        }
                    }
                }
            }
        }()
        return out
    }
}

// Rate limited take (take N per time window)
func RateLimitedTake(n int, window time.Duration) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            ticker := time.NewTicker(window)
            defer ticker.Stop()

            for {
                sent := 0
                for sent < n {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                            sent++
                        }
                    }
                }

                // Wait for next window
                select {
                case <-ctx.Done():
                    return
                case <-ticker.C:
                }
            }
        }()
        return out
    }
}

// Preview pipeline (take first N, show progress)
func PreviewData() {
    ctx := context.Background()

    square := SquareStage(3)
    multiply := MultiplyStage(2)
    take5 := TakeStage(5)

    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := take5(ctx, multiply(ctx, square(ctx, source)))

    fmt.Println("Preview (first 5 results):")
    for v := range result {
        fmt.Println(v)
    }
}

// Paginated API
func FetchPage(page, pageSize int) []int {
    ctx := context.Background()

    paginate := PaginateStage(page, pageSize)
    source := GenFromDB(db, "SELECT id FROM items ORDER BY created_at")
    results := paginate(ctx, source)

    var items []int
    for item := range results {
        items = append(items, item)
    }
    return items
}

// A/B testing with sampling
func ABTest() {
    ctx := context.Background()

    sample10Percent := SampleEveryNth(10) // Take 1 in 10
    process := ProcessStage()

    allUsers := GenFromDB(db, "SELECT id FROM users")
    testGroup := sample10Percent(ctx, allUsers)

    for userID := range testGroup {
        applyExperiment(userID)
    }
}
\`\`\`

**Real-World Benefits:**
- **Cost Control:** Don't process more than needed
- **Quick Feedback:** Get results faster with preview
- **Testing:** Test with small samples before full run
- **UI Responsiveness:** Load data in chunks

**Common Use Cases:**
- **Pagination:** Load N records per page
- **Preview:** Show first N search results
- **Sampling:** Take random or systematic sample
- **Rate Limiting:** Process N items per time window
- **Top N:** Find best N results
- **Circuit Breaker:** Stop after N failures

**Pattern Variations:**
- **Head:** Take first N (TakeStage)
- **Tail:** Take last N (requires buffering)
- **Skip + Take:** Pagination (skip M, take N)
- **Top N:** Take N highest/lowest by score
- **Sample:** Take every Nth or random N
- **Time-Based:** Take all within time window

**Performance Considerations:**
- **Drain Input:** Always drain to prevent goroutine leaks
- **Early Exit:** Stop processing ASAP after reaching limit
- **Memory:** Tail/TopN stages need O(N) memory
- **Cancellation:** Respect context for long-running sources

Without TakeStage, pipelines would process all data even when only a subset is needed, wasting CPU, memory, and time.`,	order: 6,
	translations: {
		ru: {
			title: 'Ограничение количества элементов в pipeline',
			description: `Реализуйте **TakeStage**, который возвращает функцию Stage pipeline для взятия только первых N значений из канала.

**Определение типа:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Требования:**
1. Создайте функцию \`TakeStage(n int) Stage\`
2. Верните функцию Stage которая принимает (ctx, in) и возвращает выходной канал
3. Обработайте n <= 0 (сливайте вход без отправки)
4. Отправьте только первые n значений, затем остановитесь
5. После отправки n значений, слейте оставшийся вход (используйте \`for range in {}\`)
6. Используйте вложенный select для проверок ctx.Done()
7. Отслеживайте количество отправленных значений
8. Правильно закрывайте выходной канал

**Пример:**
\`\`\`go
ctx := context.Background()

take3 := TakeStage(3)
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := take3(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Вывод: 1 2 3
\`\`\`

**Ограничения:**
- Должен обрабатывать n <= 0 сливая вход
- Должен сливать оставшийся вход после взятия n значений
- Должен использовать вложенный select для отмены
- Должен точно отслеживать счётчик отправленных`,
			hint1: `Обработайте n <= 0 как специальный случай: используйте "for range in {}" для слива входа без отправки, затем return.`,
			hint2: `Держите счётчик "sent". После каждой успешной отправки увеличивайте его. Когда sent >= n, используйте "for range in {}" для слива оставшегося входа, затем return.`,
			whyItMatters: `TakeStage обеспечивает ограничение вывода pipeline, полезно для превью, пагинации, сэмплирования и предотвращения неограниченной обработки данных.

**Почему Take/Limit важен:**
- **Контроль ресурсов:** Предотвращение обработки слишком большого объёма данных
- **Пагинация:** Реализация постраничной загрузки данных
- **Превью:** Показ первых N результатов до полной обработки
- **Тестирование:** Тестирование pipeline на малых подмножествах
- **Rate Limiting:** Контроль пропускной способности ограничением размера батчей

**Production паттерны:**
\`\`\`go
// Стадия пагинации
func PaginateStage(page, pageSize int) Stage {
    skip := page * pageSize
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            // Пропускаем до страницы
            skipped := 0
            for skipped < skip {
                select {
                case <-ctx.Done():
                    return
                case _, ok := <-in:
                    if !ok {
                        return
                    }
                    skipped++
                }
            }

            // Берём страницу
            taken := 0
            for taken < pageSize {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                        taken++
                    }
                }
            }

            // Дренируем остальное
            for range in {
            }
        }()
        return out
    }
}

// Head (первые N) и Tail (последние N)
func HeadStage(n int) Stage {
    return TakeStage(n)
}

func TailStage(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            buffer := make([]int, 0, n)

            // Собираем все значения, храним только последние N
            for v := range in {
                buffer = append(buffer, v)
                if len(buffer) > n {
                    buffer = buffer[1:]
                }
            }

            // Отправляем последние N
            for _, val := range buffer {
                out <- val
            }
        }()
        return out
    }
}

// Top N по score
func TopNStage(n int, score func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            type scored struct {
                value int
                score int
            }
            var items []scored

            // Собираем все со счётом
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        // Сортируем и отправляем топ N
                        sort.Slice(items, func(i, j int) bool {
                            return items[i].score > items[j].score
                        })

                        limit := n
                        if len(items) < limit {
                            limit = len(items)
                        }

                        for i := 0; i < limit; i++ {
                            select {
                            case <-ctx.Done():
                                return
                            case out <- items[i].value:
                            }
                        }
                        return
                    }
                    items = append(items, scored{v, score(v)})
                }
            }
        }()
        return out
    }
}

// Сэмплирование каждого N-го значения
func SampleEveryNth(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    count++
                    if count%n == 0 {
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                        }
                    }
                }
            }
        }()
        return out
    }
}

// Take с ограничением скорости (N за временное окно)
func RateLimitedTake(n int, window time.Duration) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            ticker := time.NewTicker(window)
            defer ticker.Stop()

            for {
                sent := 0
                for sent < n {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                            sent++
                        }
                    }
                }

                // Ждём следующее окно
                select {
                case <-ctx.Done():
                    return
                case <-ticker.C:
                }
            }
        }()
        return out
    }
}

// Превью данных
func PreviewData() {
    ctx := context.Background()

    square := SquareStage(3)
    multiply := MultiplyStage(2)
    take5 := TakeStage(5)

    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := take5(ctx, multiply(ctx, square(ctx, source)))

    fmt.Println("Превью (первые 5 результатов):")
    for v := range result {
        fmt.Println(v)
    }
}

// Постраничный API
func FetchPage(page, pageSize int) []int {
    ctx := context.Background()

    paginate := PaginateStage(page, pageSize)
    source := GenFromDB(db, "SELECT id FROM items ORDER BY created_at")
    results := paginate(ctx, source)

    var items []int
    for item := range results {
        items = append(items, item)
    }
    return items
}

// A/B тестирование с сэмплированием
func ABTest() {
    ctx := context.Background()

    sample10Percent := SampleEveryNth(10) // Берём 1 из 10
    process := ProcessStage()

    allUsers := GenFromDB(db, "SELECT id FROM users")
    testGroup := sample10Percent(ctx, allUsers)

    for userID := range testGroup {
        applyExperiment(userID)
    }
}
\`\`\`

**Реальные преимущества:**
- **Контроль затрат:** Не обрабатывать больше чем нужно
- **Быстрая обратная связь:** Получение результатов быстрее с превью
- **Тестирование:** Тестирование на малых выборках перед полным запуском
- **Отзывчивость UI:** Загрузка данных порциями

**Типичные сценарии использования:**
- **Пагинация:** Загрузка N записей на страницу
- **Превью:** Показ первых N результатов поиска
- **Сэмплирование:** Случайная или систематическая выборка
- **Rate Limiting:** Обработка N элементов за временное окно
- **Top N:** Поиск лучших N результатов
- **Circuit Breaker:** Остановка после N ошибок

**Вариации паттерна:**
- **Head:** Первые N элементов (TakeStage)
- **Tail:** Последние N элементов (требует буферизацию)
- **Skip + Take:** Пагинация (пропустить M, взять N)
- **Top N:** Взять N самых высоких/низких по счёту
- **Sample:** Каждый N-й или случайные N элементов
- **Time-Based:** Все в пределах временного окна

**Соображения производительности:**
- **Сливайте вход:** Всегда сливайте для предотвращения утечек горутин
- **Ранний выход:** Останавливайте обработку ASAP после достижения лимита
- **Память:** Tail/TopN стадии требуют O(N) памяти
- **Отмена:** Учитывайте контекст для долгих источников

Без TakeStage pipeline обрабатывали бы все данные даже когда нужно только подмножество, растрачивая CPU, память и время.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func TakeStage(n int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Возвращаем функцию Stage
		out := make(chan int)                               // Создаём выходной канал
		go func() {                                         // Запускаем горутину
			defer close(out)                            // Всегда закрываем выход
			if n <= 0 {                                 // Обрабатываем n <= 0
				for range in {                      // Сливаем вход
				}
				return                              // Выходим
			}
			sent := 0                                   // Отслеживаем счётчик отправленных
			for {                                       // Бесконечный цикл
				select {
				case <-ctx.Done():                  // Контекст отменён
					return                      // Выходим из горутины
				case v, ok := <-in:                 // Читаем из входа
					if !ok {                    // Канал закрыт
						return              // Выходим из горутины
					}
					if sent < n {               // Ещё нужно отправить
						select {
						case <-ctx.Done():  // Проверяем перед отправкой
							return      // Выходим из горутины
						case out <- v:      // Отправляем значение
							sent++      // Увеличиваем счётчик
						}
					}
					if sent >= n {              // Достигли лимита
						for range in {      // Сливаем оставшиеся
						}
						return              // Выходим из горутины
					}
				}
			}
		}()
		return out                                          // Возвращаем немедленно
	}
}`
		},
		uz: {
			title: 'Pipeline da elementlar sonini cheklash',
			description: `Kanaldan faqat birinchi N qiymatlarni olish uchun pipeline Stage funksiyasini qaytaruvchi **TakeStage** ni amalga oshiring.

**Tur ta'rifi:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Talablar:**
1. \`TakeStage(n int) Stage\` funksiyasini yarating
2. (ctx, in) qabul qiladigan va chiqish kanalini qaytaruvchi Stage funksiyasini qaytaring
3. n <= 0 ni ishlang (hech narsa yubormasdan kirishni to'kib tashlang)
4. Faqat birinchi n qiymatlarni yuboring, keyin to'xtating
5. n qiymatlarni yuborilgandan keyin qolgan kirishni to'kib tashlang ('for range in {}' dan foydalaning)
6. ctx.Done() tekshiruvlari uchun ichki selectdan foydalaning
7. Yuborilgan qiymatlar sonini kuzating
8. Chiqish kanalini to'g'ri yoping

**Misol:**
\`\`\`go
ctx := context.Background()

take3 := TakeStage(3)
in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := take3(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Natija: 1 2 3
\`\`\`

**Cheklovlar:**
- Kirishni to'kib tashlash orqali n <= 0 ni ishlashi kerak
- n qiymatlarni olgandan keyin qolgan kirishni to'kib tashlashi kerak
- Bekor qilish uchun ichki selectdan foydalanishi kerak
- Yuborilganlar hisoblagichini to'g'ri kuzatishi kerak`,
			hint1: `n <= 0 ni maxsus holat sifatida ishlang: hech narsa yubormasdan kirishni to'kib tashlash uchun "for range in {}" dan foydalaning, keyin return.`,
			hint2: `"sent" hisoblagichini saqlang. Har bir muvaffaqiyatli yuborishdan keyin uni oshiring. sent >= n bo'lganda, qolgan kirishni to'kib tashlash uchun "for range in {}" dan foydalaning, keyin return.`,
			whyItMatters: `TakeStage pipeline chiqishini cheklashni ta'minlaydi, preview, pagination, sampling va cheksiz ma'lumotlarni qayta ishlashning oldini olish uchun foydali.

**Nima uchun Take/Limit muhim:**
- **Resurslarni boshqarish:** Juda ko'p ma'lumotlarni qayta ishlashning oldini olish
- **Pagination:** Sahifalarga bo'lingan ma'lumotlarni yuklashni amalga oshirish
- **Preview:** To'liq qayta ishlashdan oldin birinchi N natijalarni ko'rsatish
- **Test qilish:** Kichik to'plamlarda pipelineni test qilish
- **Rate Limiting:** Batch hajmini cheklash orqali o'tkazish qobiliyatini boshqarish

**Production patternlar:**
\`\`\`go
// Pagination bosqichi
func PaginateStage(page, pageSize int) Stage {
    skip := page * pageSize
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            // Sahifagacha o'tkazib yuborish
            skipped := 0
            for skipped < skip {
                select {
                case <-ctx.Done():
                    return
                case _, ok := <-in:
                    if !ok {
                        return
                    }
                    skipped++
                }
            }

            // Sahifani olish
            taken := 0
            for taken < pageSize {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                        taken++
                    }
                }
            }

            // Qolganini drenaj qilish
            for range in {
            }
        }()
        return out
    }
}

// Head (birinchi N) va Tail (oxirgi N)
func HeadStage(n int) Stage {
    return TakeStage(n)
}

func TailStage(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            buffer := make([]int, 0, n)

            // Barcha qiymatlarni yig'amiz, faqat oxirgi N ni saqlaymiz
            for v := range in {
                buffer = append(buffer, v)
                if len(buffer) > n {
                    buffer = buffer[1:]
                }
            }

            // Oxirgi N ni yuboramiz
            for _, val := range buffer {
                out <- val
            }
        }()
        return out
    }
}

// Top N skor bo'yicha
func TopNStage(n int, score func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)

            type scored struct {
                value int
                score int
            }
            var items []scored

            // Barcha qiymatlarni skorlar bilan yig'amiz
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        // Saralab top N ni yuboramiz
                        sort.Slice(items, func(i, j int) bool {
                            return items[i].score > items[j].score
                        })

                        limit := n
                        if len(items) < limit {
                            limit = len(items)
                        }

                        for i := 0; i < limit; i++ {
                            select {
                            case <-ctx.Done():
                                return
                            case out <- items[i].value:
                            }
                        }
                        return
                    }
                    items = append(items, scored{v, score(v)})
                }
            }
        }()
        return out
    }
}

// Har N-chi qiymatni namuna olish
func SampleEveryNth(n int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            count := 0

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    count++
                    if count%n == 0 {
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                        }
                    }
                }
            }
        }()
        return out
    }
}

// Tezlik cheklangan take (vaqt oynasida N)
func RateLimitedTake(n int, window time.Duration) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            ticker := time.NewTicker(window)
            defer ticker.Stop()

            for {
                sent := 0
                for sent < n {
                    select {
                    case <-ctx.Done():
                        return
                    case v, ok := <-in:
                        if !ok {
                            return
                        }
                        select {
                        case <-ctx.Done():
                            return
                        case out <- v:
                            sent++
                        }
                    }
                }

                // Keyingi oynani kutamiz
                select {
                case <-ctx.Done():
                    return
                case <-ticker.C:
                }
            }
        }()
        return out
    }
}

// Ma'lumotlarni preview qilish
func PreviewData() {
    ctx := context.Background()

    square := SquareStage(3)
    multiply := MultiplyStage(2)
    take5 := TakeStage(5)

    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result := take5(ctx, multiply(ctx, square(ctx, source)))

    fmt.Println("Preview (birinchi 5 natija):")
    for v := range result {
        fmt.Println(v)
    }
}

// Sahifalangan API
func FetchPage(page, pageSize int) []int {
    ctx := context.Background()

    paginate := PaginateStage(page, pageSize)
    source := GenFromDB(db, "SELECT id FROM items ORDER BY created_at")
    results := paginate(ctx, source)

    var items []int
    for item := range results {
        items = append(items, item)
    }
    return items
}

// Namuna olish bilan A/B test
func ABTest() {
    ctx := context.Background()

    sample10Percent := SampleEveryNth(10) // 10 tadan 1 ni olamiz
    process := ProcessStage()

    allUsers := GenFromDB(db, "SELECT id FROM users")
    testGroup := sample10Percent(ctx, allUsers)

    for userID := range testGroup {
        applyExperiment(userID)
    }
}
\`\`\`

**Haqiqiy foydalari:**
- **Xarajatlarni boshqarish:** Kerakdan ko'proq qayta ishlamaslik
- **Tez fikr-mulohaza:** Preview bilan natijalarni tezroq olish
- **Test qilish:** To'liq ishga tushirishdan oldin kichik namunalarda test qilish
- **UI tezkor javobi:** Ma'lumotlarni qismlarda yuklash

**Umumiy foydalanish stsenariylari:**
- **Pagination:** Sahifa uchun N yozuvni yuklash
- **Preview:** Qidiruv natijalaridan birinchi N ni ko'rsatish
- **Sampling:** Tasodifiy yoki muntazam namuna olish
- **Rate Limiting:** Vaqt oynasida N elementni qayta ishlash
- **Top N:** Eng yaxshi N natijalarni topish
- **Circuit Breaker:** N xatodan keyin to'xtatish

**Pattern variatsiyalari:**
- **Head:** Birinchi N element (TakeStage)
- **Tail:** Oxirgi N element (buferlash talab qiladi)
- **Skip + Take:** Pagination (M ni o'tkazib, N ni olish)
- **Top N:** Skor bo'yicha eng yuqori/past N ni olish
- **Sample:** Har N-chi yoki tasodifiy N elementlar
- **Time-Based:** Vaqt oynasidagi barcha elementlar

**Samaradorlik mulohazalari:**
- **Kirishni to'kish:** Goroutina oqishini oldini olish uchun har doim to'king
- **Erta chiqish:** Limitga yetgandan keyin ASAP qayta ishlashni to'xtating
- **Xotira:** Tail/TopN bosqichlari O(N) xotira talab qiladi
- **Bekor qilish:** Uzoq davom etadigan manbalar uchun kontekstni hurmat qiling

TakeStage bo'lmasa, pipelinelar faqat qism kerak bo'lganda ham barcha ma'lumotlarni qayta ishlar edi, CPU, xotira va vaqtni isrof qilardi.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func TakeStage(n int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Stage funksiyasini qaytaramiz
		out := make(chan int)                               // Chiqish kanalini yaratamiz
		go func() {                                         // Goroutine ishga tushiramiz
			defer close(out)                            // Har doim chiqishni yopamiz
			if n <= 0 {                                 // n <= 0 ni ishlaymiz
				for range in {                      // Kirishni to'kamiz
				}
				return                              // Chiqamiz
			}
			sent := 0                                   // Yuborilganlar hisoblagichini kuzatamiz
			for {                                       // Cheksiz tsikl
				select {
				case <-ctx.Done():                  // Kontekst bekor qilindi
					return                      // Goroutinedan chiqamiz
				case v, ok := <-in:                 // Kirishdan o'qiymiz
					if !ok {                    // Kanal yopilgan
						return              // Goroutinedan chiqamiz
					}
					if sent < n {               // Hali yuborish kerak
						select {
						case <-ctx.Done():  // Yuborishdan oldin tekshiramiz
							return      // Goroutinedan chiqamiz
						case out <- v:      // Qiymatni yuboramiz
							sent++      // Hisoblagichni oshiramiz
						}
					}
					if sent >= n {              // Limitga yetdik
						for range in {      // Qolganlarini to'kamiz
						}
						return              // Goroutinedan chiqamiz
					}
				}
			}
		}()
		return out                                          // Darhol qaytaramiz
	}
}`
		}
	}
};

export default task;
