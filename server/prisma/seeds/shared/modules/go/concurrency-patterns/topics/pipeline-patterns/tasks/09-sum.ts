import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-sum',
	title: 'Sum',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'reduce', 'sink'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Sum** that collects all values from a channel and returns their sum (sink/reduce operation).

**Requirements:**
1. Create function \`Sum(in <-chan int) int\`
2. Read all values from the input channel
3. Accumulate sum of all values
4. Return total sum when channel closes
5. Handle empty channel (return 0)
6. This is a blocking operation (waits for channel to close)

**Example:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
total := Sum(in)
fmt.Println(total)
// Output: 15

in = Gen(10, 20, 30)
total = Sum(in)
fmt.Println(total)
// Output: 60

// Empty channel
in = Gen()
total = Sum(in)
fmt.Println(total)
// Output: 0

// Complete pipeline with sum
ctx := context.Background()
source := Gen(1, 2, 3, 4, 5)
squared := SquareStage(2)(ctx, source)
doubled := MultiplyStage(2)(ctx, squared)
total = Sum(doubled)
fmt.Println(total)
// Output: 110 (sum of [2, 8, 18, 32, 50])
\`\`\`

**Constraints:**
- Must use range loop to read from channel
- Must accumulate sum correctly
- Must return when channel closes
- This is a blocking/synchronous operation`,
	initialCode: `package concurrency

// TODO: Implement Sum
func Sum(in <-chan int) int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package concurrency

func Sum(in <-chan int) int {
	total := 0                                                  // Initialize accumulator
	for v := range in {                                         // Range over channel
		total += v                                          // Add to sum
	}
	return total                                                // Return final sum
}`,
	testCode: `package concurrency

import (
	"testing"
)

func TestSum1(t *testing.T) {
	// Test summing positive numbers
	in := make(chan int)
	go func() {
		for i := 1; i <= 5; i++ {
			in <- i
		}
		close(in)
	}()

	result := Sum(in)
	expected := 15 // 1+2+3+4+5
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum2(t *testing.T) {
	// Test summing with zero
	in := make(chan int)
	go func() {
		in <- 0
		in <- 0
		in <- 0
		close(in)
	}()

	result := Sum(in)
	expected := 0
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum3(t *testing.T) {
	// Test summing negative numbers
	in := make(chan int)
	go func() {
		in <- -1
		in <- -2
		in <- -3
		close(in)
	}()

	result := Sum(in)
	expected := -6
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum4(t *testing.T) {
	// Test summing mixed positive and negative
	in := make(chan int)
	go func() {
		in <- 10
		in <- -5
		in <- 3
		in <- -2
		close(in)
	}()

	result := Sum(in)
	expected := 6
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum5(t *testing.T) {
	// Test empty channel
	in := make(chan int)
	close(in)

	result := Sum(in)
	expected := 0
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum6(t *testing.T) {
	// Test single value
	in := make(chan int)
	go func() {
		in <- 42
		close(in)
	}()

	result := Sum(in)
	expected := 42
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum7(t *testing.T) {
	// Test large numbers
	in := make(chan int)
	go func() {
		in <- 1000000
		in <- 2000000
		in <- 3000000
		close(in)
	}()

	result := Sum(in)
	expected := 6000000
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum8(t *testing.T) {
	// Test many small values
	in := make(chan int)
	go func() {
		for i := 0; i < 100; i++ {
			in <- 1
		}
		close(in)
	}()

	result := Sum(in)
	expected := 100
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum9(t *testing.T) {
	// Test with buffered channel
	in := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)

	result := Sum(in)
	expected := 55 // 1+2+...+10
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}

func TestSum10(t *testing.T) {
	// Test sum of alternating values
	in := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			if i%2 == 0 {
				in <- i
			} else {
				in <- -i
			}
		}
		close(in)
	}()

	result := Sum(in)
	expected := -5 // 0-1+2-3+4-5+6-7+8-9
	if result != expected {
		t.Errorf("expected %d, got %d", expected, result)
	}
}`,
			hint1: `Initialize total to 0, then use "for v := range in" to read all values from the channel until it closes.`,
			hint2: `The range loop automatically exits when the channel is closed, so you just need to accumulate the sum in the loop body.`,
			whyItMatters: `Sum demonstrates sink/reduce operations in pipelines, collecting and aggregating stream data into a final result.

**Why Sink/Reduce:**
- **Aggregation:** Combine stream into single value
- **Pipeline Termination:** End processing with final result
- **Data Collection:** Gather all processed results
- **Metrics:** Calculate totals, averages, counts

**Production Pattern:**
\`\`\`go
// Generic reduce function
func Reduce(in <-chan int, initial int, reducer func(acc, val int) int) int {
    result := initial
    for v := range in {
        result = reducer(result, v)
    }
    return result
}

// Various reduction operations
func Sum(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + val
    })
}

func Product(in <-chan int) int {
    return Reduce(in, 1, func(acc, val int) int {
        return acc * val
    })
}

func Max(in <-chan int) int {
    return Reduce(in, math.MinInt, func(acc, val int) int {
        if val > acc {
            return val
        }
        return acc
    })
}

func Min(in <-chan int) int {
    return Reduce(in, math.MaxInt, func(acc, val int) int {
        if val < acc {
            return val
        }
        return acc
    })
}

func Count(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + 1
    })
}

// Average with two accumulators
func Average(in <-chan int) float64 {
    sum := 0
    count := 0
    for v := range in {
        sum += v
        count++
    }
    if count == 0 {
        return 0
    }
    return float64(sum) / float64(count)
}

// Collect all values into slice
func Collect(in <-chan int) []int {
    var results []int
    for v := range in {
        results = append(results, v)
    }
    return results
}

// Collect with limit
func CollectN(in <-chan int, n int) []int {
    results := make([]int, 0, n)
    for v := range in {
        results = append(results, v)
        if len(results) >= n {
            break
        }
    }
    // Drain remaining
    for range in {
    }
    return results
}

// Group by predicate
func GroupBy(in <-chan int, keyFunc func(int) string) map[string][]int {
    groups := make(map[string][]int)
    for v := range in {
        key := keyFunc(v)
        groups[key] = append(groups[key], v)
    }
    return groups
}

// Frequency count
func Frequency(in <-chan int) map[int]int {
    freq := make(map[int]int)
    for v := range in {
        freq[v]++
    }
    return freq
}

// Statistics collection
type Stats struct {
    Sum   int
    Count int
    Min   int
    Max   int
    Avg   float64
}

func ComputeStats(in <-chan int) Stats {
    stats := Stats{
        Min: math.MaxInt,
        Max: math.MinInt,
    }

    for v := range in {
        stats.Sum += v
        stats.Count++
        if v < stats.Min {
            stats.Min = v
        }
        if v > stats.Max {
            stats.Max = v
        }
    }

    if stats.Count > 0 {
        stats.Avg = float64(stats.Sum) / float64(stats.Count)
    }
    return stats
}

// Partition by predicate
func Partition(in <-chan int, predicate func(int) bool) (matched []int, unmatched []int) {
    for v := range in {
        if predicate(v) {
            matched = append(matched, v)
        } else {
            unmatched = append(unmatched, v)
        }
    }
    return matched, unmatched
}

// First N values
func First(in <-chan int) (int, bool) {
    v, ok := <-in
    // Drain remaining
    for range in {
    }
    return v, ok
}

// Last value
func Last(in <-chan int) (int, bool) {
    var last int
    var hasValue bool
    for v := range in {
        last = v
        hasValue = true
    }
    return last, hasValue
}

// Any matches predicate
func Any(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if predicate(v) {
            // Drain remaining
            for range in {
            }
            return true
        }
    }
    return false
}

// All match predicate
func All(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if !predicate(v) {
            // Drain remaining
            for range in {
            }
            return false
        }
    }
    return true
}

// Complete pipeline examples
func CalculateTotal() {
    ctx := context.Background()

    // Build pipeline
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    filtered := FilterStage(func(n int) bool { return n%2 == 0 })(ctx, source)
    squared := SquareStage(2)(ctx, filtered)
    doubled := MultiplyStage(2)(ctx, squared)

    // Compute final result
    total := Sum(doubled)
    fmt.Printf("Total: %d\n", total)
}

func ProcessAndAnalyze() {
    ctx := context.Background()

    // Process data
    source := GenFromDB(db, "SELECT value FROM metrics")
    transformed := MultiplyStage(100)(ctx, source)
    filtered := FilterStage(func(n int) bool { return n > 0 })(ctx, transformed)

    // Collect statistics
    stats := ComputeStats(filtered)
    fmt.Printf("Stats: %+v\n", stats)
}

func BatchProcess() {
    ctx := context.Background()

    // Process in batches
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    batch1 := TakeStage(5)(ctx, source)
    batch2 := TakeStage(5)(ctx, source)

    sum1 := Sum(batch1)
    sum2 := Sum(batch2)

    fmt.Printf("Batch 1: %d, Batch 2: %d\n", sum1, sum2)
}

// Parallel reduce
func ParallelSum(ctx context.Context, in <-chan int, workers int) int {
    // Fan-out
    var channels []<-chan int
    for i := 0; i < workers; i++ {
        channels = append(channels, in)
    }

    // Fan-in
    merged := FanIn(ctx, channels...)

    // Reduce
    return Sum(merged)
}

// Reduce with timeout
func SumWithTimeout(in <-chan int, timeout time.Duration) (int, error) {
    done := make(chan int)

    go func() {
        done <- Sum(in)
    }()

    select {
    case result := <-done:
        return result, nil
    case <-time.After(timeout):
        return 0, fmt.Errorf("sum timeout after %v", timeout)
    }
}
\`\`\`

**Real-World Benefits:**
- **Final Results:** Get single value from stream processing
- **Metrics:** Calculate totals, averages, counts in real-time
- **Validation:** Check if any/all values meet criteria
- **Data Collection:** Gather processed results for further use

**Common Sink Operations:**
- **Aggregation:** Sum, Product, Average, Min, Max
- **Counting:** Count, Frequency, GroupBy
- **Collection:** Collect, CollectN, Partition
- **Testing:** Any, All, First, Last
- **Statistics:** ComputeStats (multiple metrics at once)

**Sink Patterns:**
- **Reduce:** Combine all values into one (Sum, Product, Max)
- **Collect:** Gather all values into collection (Collect, GroupBy)
- **Test:** Boolean result based on values (Any, All)
- **Extract:** Get specific value(s) (First, Last)
- **Aggregate:** Compute multiple metrics (ComputeStats)

**Performance Considerations:**
- **Blocking:** Sink operations block until channel closes
- **Memory:** Collect operations may use O(N) memory
- **Streaming:** Reduce operations use O(1) memory
- **Parallel:** Can parallelize some reductions

**Pipeline Termination:**
1. **Source** generates data
2. **Stages** transform data
3. **Sink** collects final result
4. **Return** result to caller

Without proper sink operations, you'd need to manually range over channels and accumulate results, making final result collection repetitive and error-prone.`,	order: 8,
	translations: {
		ru: {
			title: 'Агрегация значений из канала',
			description: `Реализуйте **Sum**, который собирает все значения из канала и возвращает их сумму (операция sink/reduce).

**Требования:**
1. Создайте функцию \`Sum(in <-chan int) int\`
2. Прочитайте все значения из входного канала
3. Накапливайте сумму всех значений
4. Верните общую сумму когда канал закроется
5. Обработайте пустой канал (верните 0)
6. Это блокирующая операция (ждёт закрытия канала)

**Пример:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
total := Sum(in)
fmt.Println(total)
// Вывод: 15

in = Gen()
total = Sum(in)
fmt.Println(total)
// Вывод: 0
\`\`\`

**Ограничения:**
- Должен использовать range цикл для чтения из канала
- Должен правильно накапливать сумму
- Должен возвращаться когда канал закрывается
- Это блокирующая/синхронная операция`,
			hint1: `Инициализируйте total в 0, затем используйте "for v := range in" для чтения всех значений из канала пока он не закроется.`,
			hint2: `Range цикл автоматически завершается когда канал закрыт, поэтому вам нужно только накапливать сумму в теле цикла.`,
			whyItMatters: `Sum демонстрирует операции sink/reduce в конвейерах, собирая и агрегируя потоковые данные в финальный результат.

**Почему Sink/Reduce важны:**
- **Агрегация:** Объединение потока в одно значение
- **Завершение Pipeline:** Завершение обработки с финальным результатом
- **Сбор данных:** Собрание всех обработанных результатов
- **Метрики:** Расчёт итогов, средних, счётчиков

**Production паттерны:**
\`\`\`go
// Универсальная функция reduce
func Reduce(in <-chan int, initial int, reducer func(acc, val int) int) int {
    result := initial
    for v := range in {
        result = reducer(result, v)
    }
    return result
}

// Различные операции редукции
func Sum(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + val
    })
}

func Product(in <-chan int) int {
    return Reduce(in, 1, func(acc, val int) int {
        return acc * val
    })
}

func Max(in <-chan int) int {
    return Reduce(in, math.MinInt, func(acc, val int) int {
        if val > acc {
            return val
        }
        return acc
    })
}

func Min(in <-chan int) int {
    return Reduce(in, math.MaxInt, func(acc, val int) int {
        if val < acc {
            return val
        }
        return acc
    })
}

func Count(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + 1
    })
}

// Среднее значение с двумя аккумуляторами
func Average(in <-chan int) float64 {
    sum := 0
    count := 0
    for v := range in {
        sum += v
        count++
    }
    if count == 0 {
        return 0
    }
    return float64(sum) / float64(count)
}

// Сбор всех значений в slice
func Collect(in <-chan int) []int {
    var results []int
    for v := range in {
        results = append(results, v)
    }
    return results
}

// Сбор с ограничением
func CollectN(in <-chan int, n int) []int {
    results := make([]int, 0, n)
    for v := range in {
        results = append(results, v)
        if len(results) >= n {
            break
        }
    }
    // Опустошаем остальное
    for range in {
    }
    return results
}

// Группировка по предикату
func GroupBy(in <-chan int, keyFunc func(int) string) map[string][]int {
    groups := make(map[string][]int)
    for v := range in {
        key := keyFunc(v)
        groups[key] = append(groups[key], v)
    }
    return groups
}

// Частотный анализ
func Frequency(in <-chan int) map[int]int {
    freq := make(map[int]int)
    for v := range in {
        freq[v]++
    }
    return freq
}

// Сбор статистики
type Stats struct {
    Sum   int
    Count int
    Min   int
    Max   int
    Avg   float64
}

func ComputeStats(in <-chan int) Stats {
    stats := Stats{
        Min: math.MaxInt,
        Max: math.MinInt,
    }

    for v := range in {
        stats.Sum += v
        stats.Count++
        if v < stats.Min {
            stats.Min = v
        }
        if v > stats.Max {
            stats.Max = v
        }
    }

    if stats.Count > 0 {
        stats.Avg = float64(stats.Sum) / float64(stats.Count)
    }
    return stats
}

// Разделение по предикату
func Partition(in <-chan int, predicate func(int) bool) (matched []int, unmatched []int) {
    for v := range in {
        if predicate(v) {
            matched = append(matched, v)
        } else {
            unmatched = append(unmatched, v)
        }
    }
    return matched, unmatched
}

// Первое значение
func First(in <-chan int) (int, bool) {
    v, ok := <-in
    // Опустошаем остальное
    for range in {
    }
    return v, ok
}

// Последнее значение
func Last(in <-chan int) (int, bool) {
    var last int
    var hasValue bool
    for v := range in {
        last = v
        hasValue = true
    }
    return last, hasValue
}

// Есть ли хоть одно соответствие
func Any(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if predicate(v) {
            // Опустошаем остальное
            for range in {
            }
            return true
        }
    }
    return false
}

// Все соответствуют предикату
func All(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if !predicate(v) {
            // Опустошаем остальное
            for range in {
            }
            return false
        }
    }
    return true
}

// Полные примеры pipeline
func CalculateTotal() {
    ctx := context.Background()

    // Строим pipeline
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    filtered := FilterStage(func(n int) bool { return n%2 == 0 })(ctx, source)
    squared := SquareStage(2)(ctx, filtered)
    doubled := MultiplyStage(2)(ctx, squared)

    // Вычисляем финальный результат
    total := Sum(doubled)
    fmt.Printf("Total: %d\n", total)
}

func ProcessAndAnalyze() {
    ctx := context.Background()

    // Обрабатываем данные
    source := GenFromDB(db, "SELECT value FROM metrics")
    transformed := MultiplyStage(100)(ctx, source)
    filtered := FilterStage(func(n int) bool { return n > 0 })(ctx, transformed)

    // Собираем статистику
    stats := ComputeStats(filtered)
    fmt.Printf("Stats: %+v\n", stats)
}

func BatchProcess() {
    ctx := context.Background()

    // Обработка пакетами
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    batch1 := TakeStage(5)(ctx, source)
    batch2 := TakeStage(5)(ctx, source)

    sum1 := Sum(batch1)
    sum2 := Sum(batch2)

    fmt.Printf("Batch 1: %d, Batch 2: %d\n", sum1, sum2)
}

// Параллельная редукция
func ParallelSum(ctx context.Context, in <-chan int, workers int) int {
    // Fan-out
    var channels []<-chan int
    for i := 0; i < workers; i++ {
        channels = append(channels, in)
    }

    // Fan-in
    merged := FanIn(ctx, channels...)

    // Reduce
    return Sum(merged)
}

// Редукция с тайм-аутом
func SumWithTimeout(in <-chan int, timeout time.Duration) (int, error) {
    done := make(chan int)

    go func() {
        done <- Sum(in)
    }()

    select {
    case result := <-done:
        return result, nil
    case <-time.After(timeout):
        return 0, fmt.Errorf("sum timeout after %v", timeout)
    }
}
\`\`\`

**Реальные преимущества:**
- **Финальные результаты:** Получение одного значения из потоковой обработки
- **Метрики:** Вычисление итогов, средних, счётчиков в реальном времени
- **Валидация:** Проверка соответствия любого/всех значений критериям
- **Сбор данных:** Сбор обработанных результатов для дальнейшего использования

**Типичные операции sink:**
- **Агрегация:** Sum, Product, Average, Min, Max
- **Подсчёт:** Count, Frequency, GroupBy
- **Сбор:** Collect, CollectN, Partition
- **Тестирование:** Any, All, First, Last
- **Статистика:** ComputeStats (множественные метрики за раз)

**Паттерны sink:**
- **Reduce:** Объединение всех значений в одно (Sum, Product, Max)
- **Collect:** Сбор всех значений в коллекцию (Collect, GroupBy)
- **Test:** Булев результат на основе значений (Any, All)
- **Extract:** Получение конкретного значения(й) (First, Last)
- **Aggregate:** Вычисление множественных метрик (ComputeStats)

**Соображения производительности:**
- **Блокировка:** Sink операции блокируются пока канал не закроется
- **Память:** Collect операции могут использовать O(N) памяти
- **Потоковость:** Reduce операции используют O(1) памяти
- **Параллелизм:** Некоторые редукции можно распараллелить

**Завершение Pipeline:**
1. **Source** генерирует данные
2. **Stages** трансформируют данные
3. **Sink** собирает финальный результат
4. **Return** возвращает результат вызывающему

Без правильных sink операций вам придётся вручную перебирать каналы и накапливать результаты, что делает сбор финальных результатов повторяющимся и подверженным ошибкам.`,
			solutionCode: `package concurrency

func Sum(in <-chan int) int {
	total := 0                                                  // Инициализируем аккумулятор
	for v := range in {                                         // Итерируемся по каналу
		total += v                                          // Добавляем к сумме
	}
	return total                                                // Возвращаем финальную сумму
}`
		},
		uz: {
			title: 'Kanaldan qiymatlarni agregatsiya qilish',
			description: `Kanaldan barcha qiymatlarni yig'adigan va ularning yig'indisini qaytaruvchi **Sum** ni amalga oshiring (sink/reduce operatsiyasi).

**Talablar:**
1. \`Sum(in <-chan int) int\` funksiyasini yarating
2. Kirish kanalidan barcha qiymatlarni o'qing
3. Barcha qiymatlar yig'indisini to'plang
4. Kanal yopilganda umumiy yig'indini qaytaring
5. Bo'sh kanalni ishlang (0 qaytaring)
6. Bu bloklovchi operatsiya (kanalning yopilishini kutadi)

**Misol:**
\`\`\`go
in := Gen(1, 2, 3, 4, 5)
total := Sum(in)
fmt.Println(total)
// Natija: 15

in = Gen()
total = Sum(in)
fmt.Println(total)
// Natija: 0
\`\`\`

**Cheklovlar:**
- Kanaldan o'qish uchun range tsiklidan foydalanishi kerak
- Yig'indini to'g'ri to'plashi kerak
- Kanal yopilganda qaytishi kerak
- Bu bloklovchi/sinxron operatsiya`,
			hint1: `total ni 0 ga ishga tushiring, keyin kanal yopilguncha barcha qiymatlarni o'qish uchun "for v := range in" dan foydalaning.`,
			hint2: `Range tsikli kanal yopilganda avtomatik ravishda chiqadi, shuning uchun siz faqat tsikl tanasida yig'indini to'plashingiz kerak.`,
			whyItMatters: `Sum pipelinelarda sink/reduce operatsiyalarini namoyish etadi, oqim ma'lumotlarini yig'ish va agregatsiya qilishni yakuniy natijaga aylantiradi.

**Nima uchun Sink/Reduce muhim:**
- **Agregatsiya:** Oqimni bitta qiymatga birlashtirish
- **Pipelineni tugatish:** Qayta ishlashni yakuniy natija bilan tugatish
- **Ma'lumotlarni yig'ish:** Barcha qayta ishlangan natijalarni yig'ish
- **Metrikalar:** Jami, o'rtacha, hisoblagichlarni hisoblash

**Production patternlar:**
\`\`\`go
// Universal reduce funksiyasi
func Reduce(in <-chan int, initial int, reducer func(acc, val int) int) int {
    result := initial
    for v := range in {
        result = reducer(result, v)
    }
    return result
}

// Turli xil reduktsiya operatsiyalari
func Sum(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + val
    })
}

func Product(in <-chan int) int {
    return Reduce(in, 1, func(acc, val int) int {
        return acc * val
    })
}

func Max(in <-chan int) int {
    return Reduce(in, math.MinInt, func(acc, val int) int {
        if val > acc {
            return val
        }
        return acc
    })
}

func Min(in <-chan int) int {
    return Reduce(in, math.MaxInt, func(acc, val int) int {
        if val < acc {
            return val
        }
        return acc
    })
}

func Count(in <-chan int) int {
    return Reduce(in, 0, func(acc, val int) int {
        return acc + 1
    })
}

// Ikki akkumulyator bilan o'rtacha qiymat
func Average(in <-chan int) float64 {
    sum := 0
    count := 0
    for v := range in {
        sum += v
        count++
    }
    if count == 0 {
        return 0
    }
    return float64(sum) / float64(count)
}

// Barcha qiymatlarni slice ga yig'ish
func Collect(in <-chan int) []int {
    var results []int
    for v := range in {
        results = append(results, v)
    }
    return results
}

// Cheklov bilan yig'ish
func CollectN(in <-chan int, n int) []int {
    results := make([]int, 0, n)
    for v := range in {
        results = append(results, v)
        if len(results) >= n {
            break
        }
    }
    // Qolganini bo'shatish
    for range in {
    }
    return results
}

// Predikat bo'yicha guruhlash
func GroupBy(in <-chan int, keyFunc func(int) string) map[string][]int {
    groups := make(map[string][]int)
    for v := range in {
        key := keyFunc(v)
        groups[key] = append(groups[key], v)
    }
    return groups
}

// Chastota tahlili
func Frequency(in <-chan int) map[int]int {
    freq := make(map[int]int)
    for v := range in {
        freq[v]++
    }
    return freq
}

// Statistikani yig'ish
type Stats struct {
    Sum   int
    Count int
    Min   int
    Max   int
    Avg   float64
}

func ComputeStats(in <-chan int) Stats {
    stats := Stats{
        Min: math.MaxInt,
        Max: math.MinInt,
    }

    for v := range in {
        stats.Sum += v
        stats.Count++
        if v < stats.Min {
            stats.Min = v
        }
        if v > stats.Max {
            stats.Max = v
        }
    }

    if stats.Count > 0 {
        stats.Avg = float64(stats.Sum) / float64(stats.Count)
    }
    return stats
}

// Predikat bo'yicha ajratish
func Partition(in <-chan int, predicate func(int) bool) (matched []int, unmatched []int) {
    for v := range in {
        if predicate(v) {
            matched = append(matched, v)
        } else {
            unmatched = append(unmatched, v)
        }
    }
    return matched, unmatched
}

// Birinchi qiymat
func First(in <-chan int) (int, bool) {
    v, ok := <-in
    // Qolganini bo'shatish
    for range in {
    }
    return v, ok
}

// Oxirgi qiymat
func Last(in <-chan int) (int, bool) {
    var last int
    var hasValue bool
    for v := range in {
        last = v
        hasValue = true
    }
    return last, hasValue
}

// Hech bo'lsa bitta mos keladimi
func Any(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if predicate(v) {
            // Qolganini bo'shatish
            for range in {
            }
            return true
        }
    }
    return false
}

// Hammasi predikatga mos keladimi
func All(in <-chan int, predicate func(int) bool) bool {
    for v := range in {
        if !predicate(v) {
            // Qolganini bo'shatish
            for range in {
            }
            return false
        }
    }
    return true
}

// To'liq pipeline misollari
func CalculateTotal() {
    ctx := context.Background()

    // Pipeline quramiz
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    filtered := FilterStage(func(n int) bool { return n%2 == 0 })(ctx, source)
    squared := SquareStage(2)(ctx, filtered)
    doubled := MultiplyStage(2)(ctx, squared)

    // Yakuniy natijani hisoblaymiz
    total := Sum(doubled)
    fmt.Printf("Total: %d\n", total)
}

func ProcessAndAnalyze() {
    ctx := context.Background()

    // Ma'lumotlarni qayta ishlaymiz
    source := GenFromDB(db, "SELECT value FROM metrics")
    transformed := MultiplyStage(100)(ctx, source)
    filtered := FilterStage(func(n int) bool { return n > 0 })(ctx, transformed)

    // Statistikani yig'amiz
    stats := ComputeStats(filtered)
    fmt.Printf("Stats: %+v\n", stats)
}

func BatchProcess() {
    ctx := context.Background()

    // Paketlar bilan qayta ishlash
    source := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    batch1 := TakeStage(5)(ctx, source)
    batch2 := TakeStage(5)(ctx, source)

    sum1 := Sum(batch1)
    sum2 := Sum(batch2)

    fmt.Printf("Batch 1: %d, Batch 2: %d\n", sum1, sum2)
}

// Parallel reduktsiya
func ParallelSum(ctx context.Context, in <-chan int, workers int) int {
    // Fan-out
    var channels []<-chan int
    for i := 0; i < workers; i++ {
        channels = append(channels, in)
    }

    // Fan-in
    merged := FanIn(ctx, channels...)

    // Reduce
    return Sum(merged)
}

// Timeout bilan reduktsiya
func SumWithTimeout(in <-chan int, timeout time.Duration) (int, error) {
    done := make(chan int)

    go func() {
        done <- Sum(in)
    }()

    select {
    case result := <-done:
        return result, nil
    case <-time.After(timeout):
        return 0, fmt.Errorf("sum timeout after %v", timeout)
    }
}
\`\`\`

**Haqiqiy foydalari:**
- **Yakuniy natijalar:** Oqim qayta ishlashdan bitta qiymat olish
- **Metrikalar:** Real vaqtda jami, o'rtacha, hisoblagichlarni hisoblash
- **Validatsiya:** Har qanday/barcha qiymatlar mezonlarga mos kelishini tekshirish
- **Ma'lumot yig'ish:** Keyingi foydalanish uchun qayta ishlangan natijalarni yig'ish

**Umumiy sink operatsiyalari:**
- **Agregatsiya:** Sum, Product, Average, Min, Max
- **Hisoblash:** Count, Frequency, GroupBy
- **Yig'ish:** Collect, CollectN, Partition
- **Sinov:** Any, All, First, Last
- **Statistika:** ComputeStats (bir vaqtning o'zida bir nechta metrikalar)

**Sink patternlari:**
- **Reduce:** Barcha qiymatlarni bittaga birlashtirish (Sum, Product, Max)
- **Collect:** Barcha qiymatlarni kolleksiyaga yig'ish (Collect, GroupBy)
- **Test:** Qiymatlarga asoslangan boolean natija (Any, All)
- **Extract:** Ma'lum qiymat(lar)ni olish (First, Last)
- **Aggregate:** Bir nechta metrikalarni hisoblash (ComputeStats)

**Unumdorlik mulohazalari:**
- **Bloklash:** Sink operatsiyalari kanal yopilguncha bloklaydi
- **Xotira:** Collect operatsiyalari O(N) xotirani ishlatishi mumkin
- **Oqim:** Reduce operatsiyalari O(1) xotirani ishlatadi
- **Parallellik:** Ba'zi reduktsiyalarni parallellashtirishingiz mumkin

**Pipeline yakunlanishi:**
1. **Source** ma'lumotlar generatsiya qiladi
2. **Stages** ma'lumotlarni transformatsiya qiladi
3. **Sink** yakuniy natijani yig'adi
4. **Return** natijani chaqiruvchiga qaytaradi

To'g'ri sink operatsiyalarisiz qo'lda kanallarni aylanib o'tishingiz va natijalarni to'plashingiz kerak bo'ladi, bu yakuniy natijalarni yig'ishni takrorlanuvchi va xatolarga moyil qiladi.`,
			solutionCode: `package concurrency

func Sum(in <-chan int) int {
	total := 0                                                  // Akkumulyatorni ishga tushiramiz
	for v := range in {                                         // Kanal bo'ylab iteratsiya qilamiz
		total += v                                          // Yig'indiga qo'shamiz
	}
	return total                                                // Yakuniy yig'indini qaytaramiz
}`
		}
	}
};

export default task;
