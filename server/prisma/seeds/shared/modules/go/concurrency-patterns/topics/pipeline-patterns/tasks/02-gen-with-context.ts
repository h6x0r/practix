import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-gen-with-context',
	title: 'GenWithContext',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'pipeline', 'context', 'cancellation'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **GenWithContext** that creates a cancellable source channel respecting context cancellation.

**Requirements:**
1. Create function \`GenWithContext(ctx context.Context, nums ...int) <-chan int\`
2. Handle nil or already-cancelled context (close channel immediately)
3. Use unbuffered channel for proper cancellation
4. Launch goroutine that respects context cancellation
5. Use select to check ctx.Done() before each send
6. Close channel when done or cancelled
7. Return channel immediately

**Example:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)

for v := range ch {
    fmt.Println(v)
    if v == 3 {
        cancel() // Cancel after receiving 3
    }
}
// Output: 1 2 3 (stops early due to cancellation)

// Already cancelled context
ctx, cancel = context.WithCancel(context.Background())
cancel()
ch = GenWithContext(ctx, 1, 2, 3)
for v := range ch {
    fmt.Println(v)
}
// No output (context already cancelled)
\`\`\`

**Constraints:**
- Must check ctx.Err() before starting
- Must use select with ctx.Done() for each send
- Must use unbuffered channel
- Must close channel properly`,
	initialCode: `package concurrency

import "context"

// TODO: Implement GenWithContext
func GenWithContext(ctx context.Context, nums ...int) <-chan int {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "context"

func GenWithContext(ctx context.Context, nums ...int) <-chan int {
	out := make(chan int)                                       // Create unbuffered channel
	if ctx != nil && ctx.Err() != nil {                         // Check if already cancelled
		close(out)                                          // Close immediately
		return out                                          // Return closed channel
	}
	go func() {                                                 // Launch goroutine
		defer close(out)                                    // Always close channel
		for _, n := range nums {                            // Iterate over numbers
			select {
			case <-ctx.Done():                          // Context cancelled
				return                              // Exit early
			case out <- n:                              // Send value
			}
		}
	}()
	return out                                                  // Return immediately
}`,
			hint1: `Check if ctx != nil && ctx.Err() != nil at the start. If true, immediately close and return the channel.`,
			hint2: `In the goroutine, use select { case <-ctx.Done(): return; case out <- n: } to respect cancellation on each send.`,
			testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	ch := GenWithContext(context.Background())
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from empty Gen, got %d", count)
	}
}

func Test2(t *testing.T) {
	ch := GenWithContext(context.Background(), 42)
	v, ok := <-ch
	if !ok || v != 42 {
		t.Errorf("expected 42, got %d (ok=%v)", v, ok)
	}
}

func Test3(t *testing.T) {
	expected := []int{1, 2, 3, 4, 5}
	ch := GenWithContext(context.Background(), expected...)
	result := make([]int, 0, len(expected))
	for v := range ch {
		result = append(result, v)
	}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
}

func Test4(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ch := GenWithContext(ctx, 1, 2, 3, 4, 5)
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from cancelled context, got %d", count)
	}
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := GenWithContext(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	received := 0
	for range ch {
		received++
		if received == 3 {
			cancel()
			break
		}
	}
	time.Sleep(10 * time.Millisecond)
	remaining := 0
	for range ch {
		remaining++
	}
	if received+remaining >= 10 {
		t.Log("Cancellation may not have stopped all sends")
	}
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	ch := GenWithContext(ctx, 1, 2, 3)
	for v := range ch {
		if v < 1 || v > 3 {
			t.Errorf("unexpected value: %d", v)
		}
	}
}

func Test7(t *testing.T) {
	ch := GenWithContext(nil, 1, 2, 3)
	_, ok := <-ch
	if ok {
		t.Log("nil context handling may vary")
	}
}

func Test8(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		_ = GenWithContext(context.Background(), 1, 2, 3, 4, 5)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("GenWithContext should return immediately")
	}
}

func Test9(t *testing.T) {
	ch := GenWithContext(context.Background(), 1, 2, 3)
	for v := range ch {
		_ = v
	}
	_, ok := <-ch
	if ok {
		t.Error("expected channel to be closed after iteration")
	}
}

func Test10(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch1 := GenWithContext(ctx, 1, 2, 3)
	ch2 := GenWithContext(ctx, 10, 20, 30)
	v1, _ := <-ch1
	v2, _ := <-ch2
	if v1 != 1 || v2 != 10 {
		t.Errorf("expected independent channels: got %d and %d", v1, v2)
	}
}
`,
	whyItMatters: `GenWithContext enables graceful shutdown of pipeline sources, preventing goroutine leaks and wasted work when pipelines are cancelled.

**Why Context in Generators:**
- **Graceful Shutdown:** Stop generating when no longer needed
- **Resource Cleanup:** Prevent goroutine leaks
- **Responsive Cancellation:** React immediately to cancellation
- **Backpressure:** Avoid generating data that won't be consumed

**Production Pattern:**
\`\`\`go
// Database streaming with cancellation
func GenFromDBWithContext(ctx context.Context, db *sql.DB, query string) <-chan User {
    out := make(chan User)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        rows, err := db.QueryContext(ctx, query)
        if err != nil {
            return
        }
        defer rows.Close()

        for rows.Next() {
            var user User
            if err := rows.Scan(&user.ID, &user.Name); err != nil {
                continue
            }

            select {
            case <-ctx.Done():
                return
            case out <- user:
            }
        }
    }()
    return out
}

// File reading with cancellation
func GenLinesWithContext(ctx context.Context, filename string) <-chan string {
    out := make(chan string)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        file, err := os.Open(filename)
        if err != nil {
            return
        }
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            select {
            case <-ctx.Done():
                return
            case out <- scanner.Text():
            }
        }
    }()
    return out
}

// API polling with cancellation
func GenEventsWithContext(ctx context.Context, url string, interval time.Duration) <-chan Event {
    out := make(chan Event)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                resp, err := http.Get(url)
                if err != nil {
                    continue
                }

                var event Event
                json.NewDecoder(resp.Body).Decode(&event)
                resp.Body.Close()

                select {
                case <-ctx.Done():
                    return
                case out <- event:
                }
            }
        }
    }()
    return out
}

// Range generator with timeout
func GenRangeWithTimeout(start, end int, timeout time.Duration) <-chan int {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    out := make(chan int)

    go func() {
        defer close(out)
        defer cancel()

        for i := start; i < end; i++ {
            select {
            case <-ctx.Done():
                return
            case out <- i:
            }
        }
    }()
    return out
}

// Multiple source cancellation
func ProcessWithCancellation() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    source1 := GenWithContext(ctx, 1, 2, 3)
    source2 := GenWithContext(ctx, 10, 20, 30)
    source3 := GenWithContext(ctx, 100, 200, 300)

    // Merge sources
    merged := FanIn(ctx, source1, source2, source3)

    // Cancel after 100ms
    time.AfterFunc(100*time.Millisecond, cancel)

    for v := range merged {
        fmt.Println(v)
    }
}
\`\`\`

**Real-World Benefits:**
- **No Goroutine Leaks:** Goroutines exit cleanly on cancellation
- **Fast Shutdown:** Services can shut down immediately
- **Resource Efficiency:** Stop work that won't be used
- **Better Testing:** Tests can cancel long-running generators

**Cancellation Patterns:**
- **Manual Cancel:** User clicks "stop" button
- **Timeout:** Operation takes too long
- **First Result:** Cancel after getting first match
- **Error Occurred:** Stop generation on downstream error

**Common Scenarios:**
- **Search:** Cancel after finding first match
- **Streaming:** User navigates away from page
- **Batch Processing:** Job cancelled by admin
- **Health Checks:** Service shutdown initiated

Without GenWithContext, goroutines continue generating data that will never be consumed, wasting CPU and memory until completion.`,	order: 1,
	translations: {
		ru: {
			title: 'Генератор с контекстом',
			description: `Реализуйте **GenWithContext**, который создаёт отменяемый исходный канал учитывающий отмену контекста.

**Требования:**
1. Создайте функцию \`GenWithContext(ctx context.Context, nums ...int) <-chan int\`
2. Обработайте nil или уже отменённый контекст (закройте канал немедленно)
3. Используйте небуферизованный канал для правильной отмены
4. Запустите горутину которая учитывает отмену контекста
5. Используйте select для проверки ctx.Done() перед каждой отправкой
6. Закройте канал когда закончено или отменено
7. Верните канал немедленно

**Пример:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)

for v := range ch {
    fmt.Println(v)
    if v == 3 {
        cancel()
    }
}
// Вывод: 1 2 3 (останавливается рано из-за отмены)
\`\`\`

**Ограничения:**
- Должен проверять ctx.Err() перед началом
- Должен использовать select с ctx.Done() для каждой отправки
- Должен использовать небуферизованный канал
- Должен правильно закрывать канал`,
			hint1: `Проверьте ctx != nil && ctx.Err() != nil в начале. Если true, немедленно закройте и верните канал.`,
			hint2: `В горутине используйте select { case <-ctx.Done(): return; case out <- n: } для учёта отмены при каждой отправке.`,
			whyItMatters: `GenWithContext обеспечивает graceful shutdown источников pipeline, предотвращая утечки горутин и напрасную работу при отмене конвейеров.

**Зачем Context в генераторах:**
- **Graceful Shutdown:** Остановка генерации когда больше не нужна
- **Очистка ресурсов:** Предотвращение утечек горутин
- **Отзывчивая отмена:** Немедленная реакция на отмену
- **Обратное давление:** Избегание генерации данных которые не будут использованы

**Продакшен паттерн:**
\`\`\`go
// Потоковая передача из базы данных с отменой
func GenFromDBWithContext(ctx context.Context, db *sql.DB, query string) <-chan User {
    out := make(chan User)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        rows, err := db.QueryContext(ctx, query)
        if err != nil {
            return
        }
        defer rows.Close()

        for rows.Next() {
            var user User
            if err := rows.Scan(&user.ID, &user.Name); err != nil {
                continue
            }

            select {
            case <-ctx.Done():
                return
            case out <- user:
            }
        }
    }()
    return out
}

// Чтение файла с отменой
func GenLinesWithContext(ctx context.Context, filename string) <-chan string {
    out := make(chan string)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        file, err := os.Open(filename)
        if err != nil {
            return
        }
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            select {
            case <-ctx.Done():
                return
            case out <- scanner.Text():
            }
        }
    }()
    return out
}

// Опрос API с отменой
func GenEventsWithContext(ctx context.Context, url string, interval time.Duration) <-chan Event {
    out := make(chan Event)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                resp, err := http.Get(url)
                if err != nil {
                    continue
                }

                var event Event
                json.NewDecoder(resp.Body).Decode(&event)
                resp.Body.Close()

                select {
                case <-ctx.Done():
                    return
                case out <- event:
                }
            }
        }
    }()
    return out
}

// Генератор диапазона с таймаутом
func GenRangeWithTimeout(start, end int, timeout time.Duration) <-chan int {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    out := make(chan int)

    go func() {
        defer close(out)
        defer cancel()

        for i := start; i < end; i++ {
            select {
            case <-ctx.Done():
                return
            case out <- i:
            }
        }
    }()
    return out
}

// Отмена нескольких источников
func ProcessWithCancellation() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    source1 := GenWithContext(ctx, 1, 2, 3)
    source2 := GenWithContext(ctx, 10, 20, 30)
    source3 := GenWithContext(ctx, 100, 200, 300)

    // Объединение источников
    merged := FanIn(ctx, source1, source2, source3)

    // Отмена через 100мс
    time.AfterFunc(100*time.Millisecond, cancel)

    for v := range merged {
        fmt.Println(v)
    }
}
\`\`\`

**Практические преимущества:**
- **Нет утечек горутин:** Горутины завершаются чисто при отмене
- **Быстрое завершение:** Сервисы могут завершиться немедленно
- **Эффективность ресурсов:** Остановка работы которая не будет использована
- **Лучшее тестирование:** Тесты могут отменять долгие генераторы

**Паттерны отмены:**
- **Ручная отмена:** Пользователь нажимает кнопку "стоп"
- **Таймаут:** Операция занимает слишком много времени
- **Первый результат:** Отмена после получения первого совпадения
- **Произошла ошибка:** Остановка генерации при ошибке в следующих этапах

**Обычные сценарии:**
- **Поиск:** Отмена после нахождения первого совпадения
- **Потоковая передача:** Пользователь покидает страницу
- **Пакетная обработка:** Задание отменено администратором
- **Проверки работоспособности:** Инициировано завершение сервиса

Без GenWithContext горутины продолжают генерировать данные которые никогда не будут использованы, расходуя CPU и память до завершения.`,
			solutionCode: `package concurrency

import "context"

func GenWithContext(ctx context.Context, nums ...int) <-chan int {
	out := make(chan int)                                       // Создаём небуферизованный канал
	if ctx != nil && ctx.Err() != nil {                         // Проверяем отменён ли контекст
		close(out)                                          // Закрываем немедленно
		return out                                          // Возвращаем закрытый канал
	}
	go func() {                                                 // Запускаем горутину
		defer close(out)                                    // Всегда закрываем канал
		for _, n := range nums {                            // Итерируемся по числам
			select {
			case <-ctx.Done():                          // Контекст отменён
				return                              // Выходим рано
			case out <- n:                              // Отправляем значение
			}
		}
	}()
	return out                                                  // Возвращаем немедленно
}`
		},
		uz: {
			title: 'Kontekst bilan generator',
			description: `Kontekst bekor qilinishini hisobga oladigan bekor qilinadigan manba kanalini yaratadigan **GenWithContext** ni amalga oshiring.

**Talablar:**
1. \`GenWithContext(ctx context.Context, nums ...int) <-chan int\` funksiyasini yarating
2. nil yoki allaqachon bekor qilingan kontekstni ishlang (kanalni darhol yoping)
3. To'g'ri bekor qilish uchun buferlanmagan kanaldan foydalaning
4. Kontekst bekor qilinishini hisobga oladigan goroutine ishga tushiring
5. Har bir yuborishdan oldin ctx.Done() ni tekshirish uchun select dan foydalaning
6. Tugatilganda yoki bekor qilinganda kanalni yoping
7. Kanalni darhol qaytaring

**Misol:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
ch := GenWithContext(ctx, 1, 2, 3, 4, 5)

for v := range ch {
    fmt.Println(v)
    if v == 3 {
        cancel()
    }
}
// Natija: 1 2 3 (bekor qilish tufayli erta to'xtaydi)
\`\`\`

**Cheklovlar:**
- Boshlanishdan oldin ctx.Err() ni tekshirishi kerak
- Har bir yuborish uchun ctx.Done() bilan select dan foydalanishi kerak
- Buferlanmagan kanaldan foydalanishi kerak
- Kanalni to'g'ri yopishi kerak`,
			hint1: `Boshida ctx != nil && ctx.Err() != nil ni tekshiring. Agar true bo'lsa, darhol yoping va kanalni qaytaring.`,
			hint2: `Goroutineda har bir yuborishda bekor qilishni hurmat qilish uchun select { case <-ctx.Done(): return; case out <- n: } dan foydalaning.`,
			whyItMatters: `GenWithContext pipeline manbalarining graceful shutdown ni ta'minlaydi, pipelinelar bekor qilinganda goroutine oqishlari va behuda ishlarning oldini oladi.

**Nega generatorlarda Context kerak:**
- **Graceful Shutdown:** Endi kerak bo'lmaganda generatsiyani to'xtatish
- **Resurslarni tozalash:** Goroutine oqishlarining oldini olish
- **Javobgar bekor qilish:** Bekor qilishga darhol javob berish
- **Orqaga bosim:** Ishlatilmaydigan ma'lumotlarni yaratishdan qochish

**Ishlab chiqarish patterni:**
\`\`\`go
// Ma'lumotlar bazasidan bekor qilish bilan streaming
func GenFromDBWithContext(ctx context.Context, db *sql.DB, query string) <-chan User {
    out := make(chan User)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        rows, err := db.QueryContext(ctx, query)
        if err != nil {
            return
        }
        defer rows.Close()

        for rows.Next() {
            var user User
            if err := rows.Scan(&user.ID, &user.Name); err != nil {
                continue
            }

            select {
            case <-ctx.Done():
                return
            case out <- user:
            }
        }
    }()
    return out
}

// Faylni bekor qilish bilan o'qish
func GenLinesWithContext(ctx context.Context, filename string) <-chan string {
    out := make(chan string)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        file, err := os.Open(filename)
        if err != nil {
            return
        }
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            select {
            case <-ctx.Done():
                return
            case out <- scanner.Text():
            }
        }
    }()
    return out
}

// API pollingni bekor qilish bilan
func GenEventsWithContext(ctx context.Context, url string, interval time.Duration) <-chan Event {
    out := make(chan Event)
    if ctx != nil && ctx.Err() != nil {
        close(out)
        return out
    }

    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                resp, err := http.Get(url)
                if err != nil {
                    continue
                }

                var event Event
                json.NewDecoder(resp.Body).Decode(&event)
                resp.Body.Close()

                select {
                case <-ctx.Done():
                    return
                case out <- event:
                }
            }
        }
    }()
    return out
}

// Timeout bilan diapazon generatori
func GenRangeWithTimeout(start, end int, timeout time.Duration) <-chan int {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    out := make(chan int)

    go func() {
        defer close(out)
        defer cancel()

        for i := start; i < end; i++ {
            select {
            case <-ctx.Done():
                return
            case out <- i:
            }
        }
    }()
    return out
}

// Bir nechta manbalarni bekor qilish
func ProcessWithCancellation() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    source1 := GenWithContext(ctx, 1, 2, 3)
    source2 := GenWithContext(ctx, 10, 20, 30)
    source3 := GenWithContext(ctx, 100, 200, 300)

    // Manbalarni birlashtirish
    merged := FanIn(ctx, source1, source2, source3)

    // 100ms dan keyin bekor qilish
    time.AfterFunc(100*time.Millisecond, cancel)

    for v := range merged {
        fmt.Println(v)
    }
}
\`\`\`

**Amaliy foydalari:**
- **Goroutine oqishlari yo'q:** Goroutinelar bekor qilinganda toza ravishda tugaydi
- **Tez tugash:** Xizmatlar darhol tugashi mumkin
- **Resurs samaradorligi:** Ishlatilmaydigan ishni to'xtatish
- **Yaxshiroq testlash:** Testlar uzoq davom etadigan generatorlarni bekor qilishi mumkin

**Bekor qilish patternlari:**
- **Qo'lda bekor qilish:** Foydalanuvchi "to'xtatish" tugmasini bosadi
- **Timeout:** Operatsiya juda ko'p vaqt oladi
- **Birinchi natija:** Birinchi moslikni olgandan keyin bekor qilish
- **Xato yuz berdi:** Keyingi bosqichlarda xato bo'lganda generatsiyani to'xtatish

**Umumiy stsenariylar:**
- **Qidiruv:** Birinchi moslikni topgandan keyin bekor qilish
- **Streaming:** Foydalanuvchi sahifani tark etadi
- **Batch qayta ishlash:** Vazifa administrator tomonidan bekor qilindi
- **Sog'liqni tekshirish:** Xizmatni to'xtatish boshlandi

GenWithContext bo'lmasa, goroutinelar hech qachon ishlatilmaydigan ma'lumotlarni yaratishda davom etadi, tugallanguncha CPU va xotirani isrof qiladi.`,
			solutionCode: `package concurrency

import "context"

func GenWithContext(ctx context.Context, nums ...int) <-chan int {
	out := make(chan int)                                       // Buferlanmagan kanal yaratamiz
	if ctx != nil && ctx.Err() != nil {                         // Kontekst bekor qilinganmi tekshiramiz
		close(out)                                          // Darhol yopamiz
		return out                                          // Yopilgan kanalni qaytaramiz
	}
	go func() {                                                 // Goroutine ishga tushiramiz
		defer close(out)                                    // Har doim kanalni yopamiz
		for _, n := range nums {                            // Raqamlar bo'ylab iteratsiya qilamiz
			select {
			case <-ctx.Done():                          // Kontekst bekor qilindi
				return                              // Erta chiqamiz
			case out <- n:                              // Qiymatni yuboramiz
			}
		}
	}()
	return out                                                  // Darhol qaytaramiz
}`
		}
	}
};

export default task;
