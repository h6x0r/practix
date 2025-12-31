import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-filter-stage',
	title: 'FilterStage',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'stage', 'filter'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **FilterStage** that returns a pipeline Stage function for filtering values based on a predicate function.

**Type Definition:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Requirements:**
1. Create function \`FilterStage(predicate func(int) bool) Stage\`
2. Handle nil predicate (default to always return true)
3. Return a Stage function that takes (ctx, in) and returns output channel
4. Launch single goroutine that respects context cancellation
5. Test each value with predicate, only send if predicate returns true
6. Skip values that don't pass predicate (use continue)
7. Use nested select for ctx.Done() checks
8. Close output channel properly

**Example:**
\`\`\`go
ctx := context.Background()

// Filter even numbers
isEven := FilterStage(func(n int) bool {
    return n%2 == 0
})

in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := isEven(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Output: 2 4 6 8 10

// Filter values greater than 5
greaterThan5 := FilterStage(func(n int) bool {
    return n > 5
})

in = Gen(1, 3, 5, 7, 9, 11)
out = greaterThan5(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Output: 7 9 11

// Nil predicate passes all values
all := FilterStage(nil)
in = Gen(1, 2, 3)
out = all(ctx, in)
// Output: 1 2 3
\`\`\`

**Constraints:**
- Must handle nil predicate gracefully
- Must use continue to skip filtered values
- Must use nested select for cancellation
- Single goroutine is sufficient`,
	initialCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

// TODO: Implement FilterStage
func FilterStage(predicate func(int) bool) Stage {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func FilterStage(predicate func(int) bool) Stage {
	if predicate == nil {                                       // Handle nil predicate
		predicate = func(int) bool { return true }          // Default: pass all
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Return Stage function
		out := make(chan int)                               // Create output channel
		go func() {                                         // Launch goroutine
			defer close(out)                            // Always close output
			for {                                       // Infinite loop
				select {
				case <-ctx.Done():                  // Context cancelled
					return                      // Exit goroutine
				case v, ok := <-in:                 // Read from input
					if !ok {                    // Channel closed
						return              // Exit goroutine
					}
					if !predicate(v) {          // Test predicate
						continue            // Skip this value
					}
					select {
					case <-ctx.Done():          // Check before send
						return              // Exit goroutine
					case out <- v:              // Send filtered value
					}
				}
			}
		}()
		return out                                          // Return immediately
	}
}`,
			hint1: `Check if predicate is nil at the start. If nil, set it to a function that always returns true: func(int) bool { return true }.`,
			hint2: `After reading a value, test it with predicate(v). If it returns false, use continue to skip to the next iteration without sending.`,
			testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	stage := FilterStage(func(n int) bool { return true })
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
	isEven := FilterStage(func(n int) bool { return n%2 == 0 })
	in := make(chan int, 6)
	for i := 1; i <= 6; i++ {
		in <- i
	}
	close(in)
	out := isEven(context.Background(), in)
	results := make([]int, 0, 3)
	for v := range out {
		results = append(results, v)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 even numbers, got %d", len(results))
	}
}

func Test3(t *testing.T) {
	greaterThan5 := FilterStage(func(n int) bool { return n > 5 })
	in := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)
	out := greaterThan5(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 5 {
		t.Errorf("expected 5 values > 5, got %d", count)
	}
}

func Test4(t *testing.T) {
	stage := FilterStage(nil)
	in := make(chan int, 3)
	in <- 1
	in <- 2
	in <- 3
	close(in)
	out := stage(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 3 {
		t.Errorf("expected 3 values with nil predicate (pass all), got %d", count)
	}
}

func Test5(t *testing.T) {
	rejectAll := FilterStage(func(n int) bool { return false })
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)
	out := rejectAll(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values when rejecting all, got %d", count)
	}
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stage := FilterStage(func(n int) bool { return true })
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

func Test7(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	stage := FilterStage(func(n int) bool { return n%2 == 1 })
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)
	out := stage(ctx, in)
	for range out {
	}
}

func Test8(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		stage := FilterStage(func(n int) bool { return true })
		in := make(chan int)
		close(in)
		_ = stage(context.Background(), in)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("FilterStage should return immediately")
	}
}

func Test9(t *testing.T) {
	positive := FilterStage(func(n int) bool { return n > 0 })
	in := make(chan int, 5)
	in <- -2
	in <- -1
	in <- 0
	in <- 1
	in <- 2
	close(in)
	out := positive(context.Background(), in)
	results := make([]int, 0, 2)
	for v := range out {
		results = append(results, v)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 positive numbers, got %d", len(results))
	}
}

func Test10(t *testing.T) {
	divisibleBy3 := FilterStage(func(n int) bool { return n%3 == 0 })
	in := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		in <- i
	}
	close(in)
	out := divisibleBy3(context.Background(), in)
	expected := []int{3, 6, 9}
	results := make([]int, 0, 3)
	for v := range out {
		results = append(results, v)
	}
	for i, v := range results {
		if v != expected[i] {
			t.Errorf("at index %d: expected %d, got %d", i, expected[i], v)
		}
	}
}
`,
	whyItMatters: `FilterStage enables selective data processing in pipelines, allowing you to discard invalid, unwanted, or irrelevant data early in the pipeline.

**Why Filtering:**
- **Data Quality:** Remove invalid or malformed data
- **Performance:** Process only relevant data downstream
- **Business Logic:** Apply business rules to select data
- **Resource Efficiency:** Reduce memory and CPU on unwanted data

**Production Pattern:**
\`\`\`go
// Validation filter
func ValidEmailStage() Stage {
    emailRegex := regexp.MustCompile(\`^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{2,}$\`)
    return FilterStage(func(email int) bool {
        // In real code, this would be string validation
        return emailRegex.MatchString(fmt.Sprintf("user%d@test.com", email))
    })
}

// Range filter
func RangeFilterStage(min, max int) Stage {
    return FilterStage(func(v int) bool {
        return v >= min && v <= max
    })
}

// Whitelist filter
func WhitelistStage(allowed map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return allowed[v]
    })
}

// Blacklist filter
func BlacklistStage(blocked map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return !blocked[v]
    })
}

// Deduplication filter
func DeduplicateStage() Stage {
    seen := make(map[int]bool)
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        if seen[v] {
            return false // Already seen, filter out
        }
        seen[v] = true
        return true
    })
}

// Rate-based filter (sample every Nth item)
func SampleStage(n int) Stage {
    count := 0
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        count++
        return count%n == 0 // Keep every Nth item
    })
}

// Business logic filters
func ActiveUsersStage(db *sql.DB) Stage {
    return FilterStage(func(userID int) bool {
        var active bool
        db.QueryRow("SELECT active FROM users WHERE id = ?", userID).Scan(&active)
        return active
    })
}

func PremiumUsersStage(cache *Cache) Stage {
    return FilterStage(func(userID int) bool {
        tier := cache.GetUserTier(userID)
        return tier == "premium" || tier == "enterprise"
    })
}

// Data quality pipeline
func CleanDataPipeline() {
    ctx := context.Background()

    // Create filters
    removeNegative := FilterStage(func(n int) bool { return n >= 0 })
    removeZeros := FilterStage(func(n int) bool { return n != 0 })
    keepInRange := RangeFilterStage(10, 1000)

    // Process data
    source := Gen(-5, 0, 10, 50, 100, 500, 2000)
    result := keepInRange(ctx, removeZeros(ctx, removeNegative(ctx, source)))

    for v := range result {
        fmt.Println(v) // Only valid values: 10, 50, 100, 500
    }
}

// User processing pipeline
func ProcessUsers(db *sql.DB) {
    ctx := context.Background()

    // Filters
    validEmails := ValidEmailStage()
    activeUsers := ActiveUsersStage(db)
    premiumUsers := PremiumUsersStage(cache)
    dedupe := DeduplicateStage()

    // Pipeline
    userIDs := GenFromDB(db, "SELECT id FROM users")
    processed := premiumUsers(ctx,
        activeUsers(ctx,
            validEmails(ctx,
                dedupe(ctx, userIDs))))

    for userID := range processed {
        // Process only valid, active, premium, unique users
        processUser(userID)
    }
}

// Conditional filtering
func DynamicFilterStage(getFilter func() func(int) bool) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            filter := getFilter() // Get current filter

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    if !filter(v) {
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Complex filter composition
func ComplexFilterStage(filters ...func(int) bool) Stage {
    return FilterStage(func(v int) bool {
        for _, filter := range filters {
            if !filter(v) {
                return false // All filters must pass
            }
        }
        return true
    })
}
\`\`\`

**Real-World Benefits:**
- **Early Filtering:** Remove bad data before expensive operations
- **Resource Savings:** Don't process data that will be discarded anyway
- **Pipeline Efficiency:** Reduce data volume flowing through pipeline
- **Clean Architecture:** Separate filtering logic from processing logic

**Common Filter Patterns:**
- **Validation:** Email, phone, format validation
- **Range:** Min/max, date ranges, numeric bounds
- **Membership:** Whitelist, blacklist, set membership
- **State:** Active/inactive, enabled/disabled
- **Quality:** Remove nulls, empty strings, malformed data
- **Deduplication:** Remove duplicates based on ID or hash
- **Sampling:** Keep every Nth item, random sampling

**Performance Considerations:**
- **Filter Early:** Apply filters as early as possible in pipeline
- **Cheap First:** Run fast filters before expensive ones
- **Combine Filters:** Merge multiple simple filters into one
- **Cache Results:** Cache expensive filter lookups

**Filter Composition:**
- **AND:** All filters must pass (use ComplexFilterStage)
- **OR:** Any filter must pass (combine predicates)
- **NOT:** Invert filter logic
- **Sequential:** Apply filters in order for efficiency

Without FilterStage, you'd need to include filtering logic in every processing stage, making code harder to maintain and test.`,	order: 5,
	translations: {
		ru: {
			title: 'Этап фильтрации данных в pipeline',
			description: `Реализуйте **FilterStage**, который возвращает функцию Stage pipeline для фильтрации значений на основе функции-предиката.

**Определение типа:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Требования:**
1. Создайте функцию \`FilterStage(predicate func(int) bool) Stage\`
2. Обработайте nil predicate (по умолчанию всегда возвращать true)
3. Верните функцию Stage которая принимает (ctx, in) и возвращает выходной канал
4. Запустите одну горутину которая учитывает отмену контекста
5. Тестируйте каждое значение предикатом, отправляйте только если predicate возвращает true
6. Пропускайте значения которые не проходят предикат (используйте continue)
7. Используйте вложенный select для проверок ctx.Done()
8. Правильно закрывайте выходной канал

**Пример:**
\`\`\`go
ctx := context.Background()

isEven := FilterStage(func(n int) bool {
    return n%2 == 0
})

in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := isEven(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Вывод: 2 4 6 8 10
\`\`\`

**Ограничения:**
- Должен корректно обрабатывать nil predicate
- Должен использовать continue для пропуска отфильтрованных значений
- Должен использовать вложенный select для отмены
- Достаточно одной горутины`,
			hint1: `Проверьте predicate на nil в начале. Если nil, установите его в функцию которая всегда возвращает true.`,
			hint2: `После чтения значения протестируйте его с predicate(v). Если возвращается false, используйте continue для пропуска.`,
			whyItMatters: `FilterStage обеспечивает выборочную обработку данных в конвейерах, позволяя отбрасывать недействительные, нежелательные или нерелевантные данные рано в конвейере.

**Почему фильтрация критична:**
- **Качество данных:** Удаление невалидных или неправильных данных
- **Производительность:** Обработка только релевантных данных downstream
- **Бизнес-логика:** Применение бизнес-правил для выбора данных
- **Эффективность ресурсов:** Снижение нагрузки на память и CPU от ненужных данных

**Production паттерны:**
\`\`\`go
// Валидационный фильтр
func ValidEmailStage() Stage {
    return FilterStage(func(email int) bool {
        // В реальном коде валидация email строк
        return email > 0 && email < 10000
    })
}

// Фильтр по диапазону
func RangeFilterStage(min, max int) Stage {
    return FilterStage(func(v int) bool {
        return v >= min && v <= max
    })
}

// Whitelist фильтр
func WhitelistStage(allowed map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return allowed[v]
    })
}

// Blacklist фильтр
func BlacklistStage(blocked map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return !blocked[v]
    })
}

// Дедупликация
func DeduplicateStage() Stage {
    seen := make(map[int]bool)
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        if seen[v] {
            return false // Уже видели, отфильтровываем
        }
        seen[v] = true
        return true
    })
}

// Фильтр по частоте (каждый N-й элемент)
func SampleStage(n int) Stage {
    count := 0
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        count++
        return count%n == 0 // Оставляем каждый N-й элемент
    })
}

// Фильтры бизнес-логики
func ActiveUsersStage(db *sql.DB) Stage {
    return FilterStage(func(userID int) bool {
        var active bool
        db.QueryRow("SELECT active FROM users WHERE id = ?", userID).Scan(&active)
        return active
    })
}

func PremiumUsersStage(cache *Cache) Stage {
    return FilterStage(func(userID int) bool {
        tier := cache.GetUserTier(userID)
        return tier == "premium" || tier == "enterprise"
    })
}

// Pipeline очистки данных
func CleanDataPipeline() {
    ctx := context.Background()

    // Создаём фильтры
    removeNegative := FilterStage(func(n int) bool { return n >= 0 })
    removeZeros := FilterStage(func(n int) bool { return n != 0 })
    keepInRange := RangeFilterStage(10, 1000)

    // Обрабатываем данные
    source := Gen(-5, 0, 10, 50, 100, 500, 2000)
    result := keepInRange(ctx, removeZeros(ctx, removeNegative(ctx, source)))

    for v := range result {
        fmt.Println(v) // Только валидные: 10, 50, 100, 500
    }
}

// Pipeline обработки пользователей
func ProcessUsers(db *sql.DB) {
    ctx := context.Background()

    // Фильтры
    validEmails := ValidEmailStage()
    activeUsers := ActiveUsersStage(db)
    premiumUsers := PremiumUsersStage(cache)
    dedupe := DeduplicateStage()

    // Pipeline
    userIDs := GenFromDB(db, "SELECT id FROM users")
    processed := premiumUsers(ctx,
        activeUsers(ctx,
            validEmails(ctx,
                dedupe(ctx, userIDs))))

    for userID := range processed {
        // Обрабатываем только валидных, активных, премиум, уникальных пользователей
        processUser(userID)
    }
}

// Динамическая фильтрация
func DynamicFilterStage(getFilter func() func(int) bool) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            filter := getFilter() // Получаем текущий фильтр

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    if !filter(v) {
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Композиция сложных фильтров
func ComplexFilterStage(filters ...func(int) bool) Stage {
    return FilterStage(func(v int) bool {
        for _, filter := range filters {
            if !filter(v) {
                return false // Все фильтры должны пройти
            }
        }
        return true
    })
}
\`\`\`

**Реальные преимущества:**
- **Ранняя фильтрация:** Удаление плохих данных до дорогих операций
- **Экономия ресурсов:** Не обрабатывать данные которые будут отброшены
- **Эффективность pipeline:** Уменьшение объёма данных текущих через pipeline
- **Чистая архитектура:** Разделение логики фильтрации и обработки

**Общие паттерны фильтров:**
- **Валидация:** Email, телефон, валидация формата
- **Диапазон:** Min/max, диапазоны дат, числовые границы
- **Членство:** Whitelist, blacklist, принадлежность множеству
- **Состояние:** Active/inactive, enabled/disabled
- **Качество:** Удаление nulls, пустых строк, неправильных данных
- **Дедупликация:** Удаление дубликатов по ID или hash
- **Сэмплирование:** Каждый N-й элемент, случайная выборка

**Соображения производительности:**
- **Фильтруйте рано:** Применяйте фильтры как можно раньше в pipeline
- **Дешёвые первыми:** Запускайте быстрые фильтры перед дорогими
- **Объединяйте фильтры:** Сливайте множество простых фильтров в один
- **Кешируйте результаты:** Кешируйте дорогие поиски фильтров

**Композиция фильтров:**
- **AND:** Все фильтры должны пройти (используйте ComplexFilterStage)
- **OR:** Любой фильтр должен пройти (комбинируйте предикаты)
- **NOT:** Инверсия логики фильтра
- **Последовательно:** Применение фильтров по порядку для эффективности

Без FilterStage нужно включать логику фильтрации в каждую стадию обработки, делая код сложнее для поддержки и тестирования.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func FilterStage(predicate func(int) bool) Stage {
	if predicate == nil {                                       // Обрабатываем nil предикат
		predicate = func(int) bool { return true }          // По умолчанию: пропускаем всё
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Возвращаем функцию Stage
		out := make(chan int)                               // Создаём выходной канал
		go func() {                                         // Запускаем горутину
			defer close(out)                            // Всегда закрываем выход
			for {                                       // Бесконечный цикл
				select {
				case <-ctx.Done():                  // Контекст отменён
					return                      // Выходим из горутины
				case v, ok := <-in:                 // Читаем из входа
					if !ok {                    // Канал закрыт
						return              // Выходим из горутины
					}
					if !predicate(v) {          // Тестируем предикат
						continue            // Пропускаем это значение
					}
					select {
					case <-ctx.Done():          // Проверяем перед отправкой
						return              // Выходим из горутины
					case out <- v:              // Отправляем отфильтрованное значение
					}
				}
			}
		}()
		return out                                          // Возвращаем немедленно
	}
}`
		},
		uz: {
			title: 'Pipeline da ma\'lumotlarni filtrlash bosqichi',
			description: `Predikat funksiyasi asosida qiymatlarni filtrlash uchun pipeline Stage funksiyasini qaytaruvchi **FilterStage** ni amalga oshiring.

**Tur ta'rifi:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Talablar:**
1. \`FilterStage(predicate func(int) bool) Stage\` funksiyasini yarating
2. nil predikatni ishlang (standart har doim true qaytaring)
3. (ctx, in) qabul qiladigan va chiqish kanalini qaytaruvchi Stage funksiyasini qaytaring
4. Kontekst bekor qilinishini hisobga oladigan bitta goroutine ishga tushiring
5. Har bir qiymatni predikat bilan tekshiring, faqat predikat true qaytarsa yuboring
6. Predikatdan o'tmagan qiymatlarni o'tkazib yuboring (continue dan foydalaning)
7. ctx.Done() tekshiruvlari uchun ichki selectdan foydalaning
8. Chiqish kanalini to'g'ri yoping

**Misol:**
\`\`\`go
ctx := context.Background()

isEven := FilterStage(func(n int) bool {
    return n%2 == 0
})

in := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
out := isEven(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Natija: 2 4 6 8 10
\`\`\`

**Cheklovlar:**
- nil predikatni to'g'ri ishlashi kerak
- Filtrlangan qiymatlarni o'tkazib yuborish uchun continue dan foydalanishi kerak
- Bekor qilish uchun ichki selectdan foydalanishi kerak
- Bitta goroutine yetarli`,
			hint1: `Boshida predikatni nil ga tekshiring. Agar nil bo'lsa, uni har doim true qaytaradigan funksiyaga o'rnating.`,
			hint2: `Qiymatni o'qiganingizdan keyin uni predikat(v) bilan tekshiring. Agar false qaytarsa, o'tkazib yuborish uchun continue dan foydalaning.`,
			whyItMatters: `FilterStage pipelinelarda tanlab ma'lumotlarni qayta ishlashni ta'minlaydi, yaroqsiz, keraksiz yoki ahamiyatsiz ma'lumotlarni pipelineda erta tashlab yuborishga imkon beradi.

**Nima uchun filtrlash muhim:**
- **Ma'lumot sifati:** Noto'g'ri yoki yaroqsiz ma'lumotlarni olib tashlash
- **Samaradorlik:** Faqat tegishli ma'lumotlarni downstream qayta ishlash
- **Biznes mantiq:** Ma'lumotlarni tanlash uchun biznes qoidalarini qo'llash
- **Resurs samaradorligi:** Keraksiz ma'lumotlardan xotira va CPU yukini kamaytirish

**Production patternlar:**
\`\`\`go
// Validatsiya filtri
func ValidEmailStage() Stage {
    return FilterStage(func(email int) bool {
        // Haqiqiy kodda email stringlarni validatsiya qilish
        return email > 0 && email < 10000
    })
}

// Diapazon filtri
func RangeFilterStage(min, max int) Stage {
    return FilterStage(func(v int) bool {
        return v >= min && v <= max
    })
}

// Whitelist filtri
func WhitelistStage(allowed map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return allowed[v]
    })
}

// Blacklist filtri
func BlacklistStage(blocked map[int]bool) Stage {
    return FilterStage(func(v int) bool {
        return !blocked[v]
    })
}

// Deduplikatsiya
func DeduplicateStage() Stage {
    seen := make(map[int]bool)
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        if seen[v] {
            return false // Allaqachon ko'rgan, filtrlaymiz
        }
        seen[v] = true
        return true
    })
}

// Chastota asosidagi filtr (har N-chi element)
func SampleStage(n int) Stage {
    count := 0
    var mu sync.Mutex

    return FilterStage(func(v int) bool {
        mu.Lock()
        defer mu.Unlock()

        count++
        return count%n == 0 // Har N-chi elementni saqlaymiz
    })
}

// Biznes mantiq filtrlari
func ActiveUsersStage(db *sql.DB) Stage {
    return FilterStage(func(userID int) bool {
        var active bool
        db.QueryRow("SELECT active FROM users WHERE id = ?", userID).Scan(&active)
        return active
    })
}

func PremiumUsersStage(cache *Cache) Stage {
    return FilterStage(func(userID int) bool {
        tier := cache.GetUserTier(userID)
        return tier == "premium" || tier == "enterprise"
    })
}

// Ma'lumotlarni tozalash pipelinei
func CleanDataPipeline() {
    ctx := context.Background()

    // Filtrlarni yaratamiz
    removeNegative := FilterStage(func(n int) bool { return n >= 0 })
    removeZeros := FilterStage(func(n int) bool { return n != 0 })
    keepInRange := RangeFilterStage(10, 1000)

    // Ma'lumotlarni qayta ishlaymiz
    source := Gen(-5, 0, 10, 50, 100, 500, 2000)
    result := keepInRange(ctx, removeZeros(ctx, removeNegative(ctx, source)))

    for v := range result {
        fmt.Println(v) // Faqat yaroqlilar: 10, 50, 100, 500
    }
}

// Foydalanuvchilarni qayta ishlash pipelinei
func ProcessUsers(db *sql.DB) {
    ctx := context.Background()

    // Filtrlar
    validEmails := ValidEmailStage()
    activeUsers := ActiveUsersStage(db)
    premiumUsers := PremiumUsersStage(cache)
    dedupe := DeduplicateStage()

    // Pipeline
    userIDs := GenFromDB(db, "SELECT id FROM users")
    processed := premiumUsers(ctx,
        activeUsers(ctx,
            validEmails(ctx,
                dedupe(ctx, userIDs))))

    for userID := range processed {
        // Faqat yaroqli, faol, premium, noyob foydalanuvchilarni qayta ishlaymiz
        processUser(userID)
    }
}

// Dinamik filtrlash
func DynamicFilterStage(getFilter func() func(int) bool) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            filter := getFilter() // Joriy filtrni olamiz

            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    if !filter(v) {
                        continue
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- v:
                    }
                }
            }
        }()
        return out
    }
}

// Murakkab filtrlarni kompozitsiya qilish
func ComplexFilterStage(filters ...func(int) bool) Stage {
    return FilterStage(func(v int) bool {
        for _, filter := range filters {
            if !filter(v) {
                return false // Barcha filtrlar o'tishi kerak
            }
        }
        return true
    })
}
\`\`\`

**Haqiqiy foydalari:**
- **Erta filtrlash:** Qimmat operatsiyalardan oldin yomon ma'lumotlarni olib tashlash
- **Resurslarni tejash:** Tashlab yuborilishi kerak bo'lgan ma'lumotlarni qayta ishlamaslik
- **Pipeline samaradorligi:** Pipeline orqali oqadigan ma'lumotlar hajmini kamaytirish
- **Toza arxitektura:** Filtrlash va qayta ishlash mantiqini ajratish

**Umumiy filtr patternlari:**
- **Validatsiya:** Email, telefon, format validatsiyasi
- **Diapazon:** Min/max, sana diapazoni, raqamli chegaralar
- **A'zolik:** Whitelist, blacklist, to'plamga tegishlilik
- **Holat:** Active/inactive, enabled/disabled
- **Sifat:** Nulls, bo'sh stringlar, noto'g'ri ma'lumotlarni olib tashlash
- **Deduplikatsiya:** ID yoki hash bo'yicha dublikatlarni olib tashlash
- **Namuna olish:** Har N-chi element, tasodifiy namuna olish

**Samaradorlik mulohazalari:**
- **Erta filtrlash:** Filtrlarni pipelineda imkon qadar erta qo'llang
- **Arzonlarni birinchi:** Tez filtrlarni qimmatlardan oldin ishga tushiring
- **Filtrlarni birlashtiring:** Ko'p oddiy filtrlarni biriga birlashiring
- **Natijalarni keshlang:** Qimmat filtr qidiruvlarini keshlang

**Filtrlarni kompozitsiya qilish:**
- **AND:** Barcha filtrlar o'tishi kerak (ComplexFilterStage dan foydalaning)
- **OR:** Har qanday filtr o'tishi kerak (predikatlarni birlashiring)
- **NOT:** Filtr mantiqini inversiya qilish
- **Ketma-ket:** Samaradorlik uchun filtrlarni tartib bilan qo'llash

FilterStage bo'lmasa, har bir qayta ishlash bosqichida filtrlash mantiqini kiritish kerak bo'lib, kodni saqlash va test qilishni qiyinlashtiradi.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func FilterStage(predicate func(int) bool) Stage {
	if predicate == nil {                                       // nil predikatni ishlaymiz
		predicate = func(int) bool { return true }          // Standart: hammasini o'tkazamiz
	}
	return func(ctx context.Context, in <-chan int) <-chan int { // Stage funksiyasini qaytaramiz
		out := make(chan int)                               // Chiqish kanalini yaratamiz
		go func() {                                         // Goroutine ishga tushiramiz
			defer close(out)                            // Har doim chiqishni yopamiz
			for {                                       // Cheksiz tsikl
				select {
				case <-ctx.Done():                  // Kontekst bekor qilindi
					return                      // Goroutinedan chiqamiz
				case v, ok := <-in:                 // Kirishdan o'qiymiz
					if !ok {                    // Kanal yopilgan
						return              // Goroutinedan chiqamiz
					}
					if !predicate(v) {          // Predikatni tekshiramiz
						continue            // Bu qiymatni o'tkazib yuboramiz
					}
					select {
					case <-ctx.Done():          // Yuborishdan oldin tekshiramiz
						return              // Goroutinedan chiqamiz
					case out <- v:              // Filtrlangan qiymatni yuboramiz
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
