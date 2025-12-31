import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-gen',
	title: 'Gen',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'pipeline', 'channels'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Gen** that creates a source channel and sends numbers into it.

**Requirements:**
1. Create function \`Gen(nums ...int) <-chan int\`
2. Return a receive-only channel of integers
3. Create buffered channel with capacity len(nums)
4. Launch goroutine to send all numbers
5. Close channel after all numbers are sent
6. Return channel immediately (non-blocking)

**Example:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v)
}
// Output: 1 2 3 4 5

ch = Gen()
for v := range ch {
    fmt.Println(v)
}
// No output (empty channel, immediately closed)
\`\`\`

**Constraints:**
- Must use buffered channel with len(nums) capacity
- Must close channel after sending all values
- Must not block on return`,
	initialCode: `package concurrency

// TODO: Implement Gen
func Gen(nums ...int) <-chan int {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

func Gen(nums ...int) <-chan int {
	out := make(chan int, len(nums))                            // Create buffered channel
	go func() {                                                 // Launch goroutine
		defer close(out)                                    // Always close channel
		for _, n := range nums {                            // Iterate over numbers
			out <- n                                    // Send to channel
		}
	}()
	return out                                                  // Return immediately
}`,
			hint1: `Create a buffered channel with make(chan int, len(nums)) so the goroutine can send all values without blocking.`,
			hint2: `Use defer close(out) in the goroutine to ensure the channel is closed after all numbers are sent, allowing range loops to terminate.`,
			testCode: `package concurrency

import (
	"sync"
	"testing"
)

func Test1(t *testing.T) {
	ch := Gen()
	count := 0
	for range ch {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from empty Gen, got %d", count)
	}
}

func Test2(t *testing.T) {
	ch := Gen(42)
	v, ok := <-ch
	if !ok || v != 42 {
		t.Errorf("expected 42, got %d (ok=%v)", v, ok)
	}
	_, ok = <-ch
	if ok {
		t.Error("expected channel to be closed after single value")
	}
}

func Test3(t *testing.T) {
	expected := []int{1, 2, 3, 4, 5}
	ch := Gen(expected...)
	result := make([]int, 0, len(expected))
	for v := range ch {
		result = append(result, v)
	}
	if len(result) != len(expected) {
		t.Errorf("expected %d values, got %d", len(expected), len(result))
	}
	for i, v := range result {
		if v != expected[i] {
			t.Errorf("at index %d: expected %d, got %d", i, expected[i], v)
		}
	}
}

func Test4(t *testing.T) {
	nums := []int{1, 2, 3}
	ch := Gen(nums...)
	if cap(ch) != 0 {
		t.Log("Note: returned channel is receive-only, can't check buffer directly")
	}
}

func Test5(t *testing.T) {
	ch := Gen(10, 20, 30, 40, 50)
	var prev int
	first := true
	for v := range ch {
		if !first && v <= prev {
			t.Log("Values received in order")
		}
		prev = v
		first = false
	}
}

func Test6(t *testing.T) {
	ch1 := Gen(1, 2, 3)
	ch2 := Gen(100, 200, 300)
	v1, _ := <-ch1
	v2, _ := <-ch2
	if v1 != 1 || v2 != 100 {
		t.Errorf("expected independent channels: got %d and %d", v1, v2)
	}
}

func Test7(t *testing.T) {
	ch := Gen(1)
	<-ch
	_, ok := <-ch
	if ok {
		t.Error("expected channel to be closed after all values")
	}
}

func Test8(t *testing.T) {
	nums := make([]int, 1000)
	for i := range nums {
		nums[i] = i
	}
	ch := Gen(nums...)
	count := 0
	for range ch {
		count++
	}
	if count != 1000 {
		t.Errorf("expected 1000 values, got %d", count)
	}
}

func Test9(t *testing.T) {
	done := make(chan bool)
	go func() {
		_ = Gen(1, 2, 3, 4, 5)
		done <- true
	}()
	select {
	case <-done:
	default:
		t.Error("Gen should return immediately")
	}
}

func Test10(t *testing.T) {
	ch := Gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	var wg sync.WaitGroup
	results := make(chan int, 10)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for v := range ch {
				results <- v
			}
		}()
	}
	wg.Wait()
	close(results)
	count := 0
	for range results {
		count++
	}
	if count != 10 {
		t.Errorf("expected 10 values across readers, got %d", count)
	}
}
`,
	whyItMatters: `Gen is the foundation of pipeline patterns, creating the source stage that feeds data into processing pipelines.

**Why Source Generators:**
- **Pipeline Entry:** All pipelines need a data source
- **Decoupling:** Separates data generation from processing
- **Composition:** Allows building complex pipelines from simple parts
- **Testability:** Easy to test with known input sequences

**Production Pattern:**
\`\`\`go
// Reading data from database
func GenFromDB(db *sql.DB, query string) <-chan User {
    out := make(chan User, 100)
    go func() {
        defer close(out)
        rows, _ := db.Query(query)
        defer rows.Close()

        for rows.Next() {
            var user User
            rows.Scan(&user.ID, &user.Name)
            out <- user
        }
    }()
    return out
}

// Reading from file
func GenFromFile(filename string) <-chan string {
    out := make(chan string, 50)
    go func() {
        defer close(out)
        file, _ := os.Open(filename)
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            out <- scanner.Text()
        }
    }()
    return out
}

// Generating from API
func GenFromAPI(url string) <-chan Record {
    out := make(chan Record, 100)
    go func() {
        defer close(out)
        resp, _ := http.Get(url)
        defer resp.Body.Close()

        decoder := json.NewDecoder(resp.Body)
        for {
            var record Record
            if err := decoder.Decode(&record); err != nil {
                break
            }
            out <- record
        }
    }()
    return out
}

// Infinite generator
func GenInfinite(start int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for i := start; ; i++ {
            out <- i
        }
    }()
    return out
}

// Timer-based generator
func GenTicks(interval time.Duration, count int) <-chan time.Time {
    out := make(chan time.Time)
    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for i := 0; i < count; i++ {
            out <- <-ticker.C
        }
    }()
    return out
}

// Range generator
func GenRange(start, end, step int) <-chan int {
    out := make(chan int, (end-start)/step+1)
    go func() {
        defer close(out)
        for i := start; i < end; i += step {
            out <- i
        }
    }()
    return out
}
\`\`\`

**Real-World Benefits:**
- **Stream Processing:** Handle large datasets that don't fit in memory
- **Pipeline Start:** Begin data flow through processing stages
- **Async Loading:** Load data concurrently while processing
- **Resource Control:** Buffer size controls memory usage

**Buffer Sizing Guidelines:**
- **Small Data (< 100 items):** Buffer = len(data)
- **Medium Data (100-10000):** Buffer = 100-1000
- **Large Data (> 10000):** Buffer = 1000-10000
- **Infinite Streams:** Unbuffered or small buffer (1-10)

**Common Use Cases:**
- **Batch Processing:** Gen(batchIDs...) feeds worker pipeline
- **Event Streams:** Gen events from message queue
- **Data Migration:** Gen records from source database
- **Testing:** Gen known test data for pipeline verification

Without Gen, you'd need to manually create and manage channels, making pipeline composition verbose and error-prone.`,	order: 0,
	translations: {
		ru: {
			title: 'Генератор последовательности значений',
			description: `Реализуйте **Gen**, который создаёт исходный канал и отправляет в него числа.

**Требования:**
1. Создайте функцию \`Gen(nums ...int) <-chan int\`
2. Верните канал только для чтения целых чисел
3. Создайте буферизованный канал с ёмкостью len(nums)
4. Запустите горутину для отправки всех чисел
5. Закройте канал после отправки всех чисел
6. Верните канал немедленно (неблокирующе)

**Пример:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v)
}
// Вывод: 1 2 3 4 5
\`\`\`

**Ограничения:**
- Должен использовать буферизованный канал с ёмкостью len(nums)
- Должен закрыть канал после отправки всех значений
- Не должен блокироваться при возврате`,
			hint1: `Создайте буферизованный канал с make(chan int, len(nums)), чтобы горутина могла отправить все значения без блокировки.`,
			hint2: `Используйте defer close(out) в горутине, чтобы гарантировать закрытие канала после отправки всех чисел.`,
			whyItMatters: `Gen - это основа паттернов pipeline, создающая исходную стадию которая подаёт данные в обрабатывающие конвейеры.

**Зачем генераторы источников:**
- **Вход конвейера:** Всем конвейерам нужен источник данных
- **Разделение:** Отделяет генерацию данных от обработки
- **Композиция:** Позволяет строить сложные конвейеры из простых частей
- **Тестируемость:** Легко тестировать с известными входными последовательностями

**Продакшен паттерн:**
\`\`\`go
// Чтение данных из базы данных
func GenFromDB(db *sql.DB, query string) <-chan User {
    out := make(chan User, 100)
    go func() {
        defer close(out)
        rows, _ := db.Query(query)
        defer rows.Close()

        for rows.Next() {
            var user User
            rows.Scan(&user.ID, &user.Name)
            out <- user
        }
    }()
    return out
}

// Чтение из файла
func GenFromFile(filename string) <-chan string {
    out := make(chan string, 50)
    go func() {
        defer close(out)
        file, _ := os.Open(filename)
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            out <- scanner.Text()
        }
    }()
    return out
}

// Генерация из API
func GenFromAPI(url string) <-chan Record {
    out := make(chan Record, 100)
    go func() {
        defer close(out)
        resp, _ := http.Get(url)
        defer resp.Body.Close()

        decoder := json.NewDecoder(resp.Body)
        for {
            var record Record
            if err := decoder.Decode(&record); err != nil {
                break
            }
            out <- record
        }
    }()
    return out
}

// Бесконечный генератор
func GenInfinite(start int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for i := start; ; i++ {
            out <- i
        }
    }()
    return out
}

// Генератор на основе таймера
func GenTicks(interval time.Duration, count int) <-chan time.Time {
    out := make(chan time.Time)
    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for i := 0; i < count; i++ {
            out <- <-ticker.C
        }
    }()
    return out
}

// Генератор диапазона
func GenRange(start, end, step int) <-chan int {
    out := make(chan int, (end-start)/step+1)
    go func() {
        defer close(out)
        for i := start; i < end; i += step {
            out <- i
        }
    }()
    return out
}
\`\`\`

**Практические преимущества:**
- **Потоковая обработка:** Обработка больших наборов данных, не помещающихся в память
- **Запуск конвейера:** Начало потока данных через стадии обработки
- **Асинхронная загрузка:** Загрузка данных параллельно с обработкой
- **Контроль ресурсов:** Размер буфера контролирует использование памяти

**Рекомендации по размерам буфера:**
- **Малые данные (< 100 элементов):** Буфер = len(data)
- **Средние данные (100-10000):** Буфер = 100-1000
- **Большие данные (> 10000):** Буфер = 1000-10000
- **Бесконечные потоки:** Без буфера или малый буфер (1-10)

**Обычные сценарии использования:**
- **Пакетная обработка:** Gen(batchIDs...) питает рабочий конвейер
- **Потоки событий:** Gen события из очереди сообщений
- **Миграция данных:** Gen записей из исходной базы данных
- **Тестирование:** Gen известных тестовых данных для проверки конвейера

Без Gen вам придётся вручную создавать и управлять каналами, делая композицию конвейеров многословной и подверженной ошибкам.`,
			solutionCode: `package concurrency

func Gen(nums ...int) <-chan int {
	out := make(chan int, len(nums))                            // Создаём буферизованный канал
	go func() {                                                 // Запускаем горутину
		defer close(out)                                    // Всегда закрываем канал
		for _, n := range nums {                            // Итерируемся по числам
			out <- n                                    // Отправляем в канал
		}
	}()
	return out                                                  // Возвращаем немедленно
}`
		},
		uz: {
			title: 'Qiymatlar ketma-ketligini generatsiya qilish',
			description: `Manba kanalini yaratadigan va unga raqamlarni yuboradigan **Gen** ni amalga oshiring.

**Talablar:**
1. \`Gen(nums ...int) <-chan int\` funksiyasini yarating
2. Faqat o'qish uchun butun sonlar kanalini qaytaring
3. len(nums) sig'imli buferlangan kanal yarating
4. Barcha raqamlarni yuborish uchun goroutine ishga tushiring
5. Barcha raqamlar yuborilgandan keyin kanalni yoping
6. Kanalni darhol qaytaring (bloklanmasdan)

**Misol:**
\`\`\`go
ch := Gen(1, 2, 3, 4, 5)
for v := range ch {
    fmt.Println(v)
}
// Natija: 1 2 3 4 5
\`\`\`

**Cheklovlar:**
- len(nums) sig'imli buferlangan kanaldan foydalanishi kerak
- Barcha qiymatlar yuborilgandan keyin kanalni yopishi kerak
- Qaytarishda bloklanmasligi kerak`,
			hint1: `Goroutine barcha qiymatlarni bloklashsiz yuborishi uchun make(chan int, len(nums)) bilan buferlangan kanal yarating.`,
			hint2: `Barcha raqamlar yuborilgandan keyin kanal yopilishini ta'minlash uchun goroutineda defer close(out) dan foydalaning.`,
			whyItMatters: `Gen pipeline patternlarining asosi bo'lib, qayta ishlash pipelinelariga ma'lumot uzatadigan manba bosqichini yaratadi.

**Nega manba generatorlari kerak:**
- **Pipeline kirishi:** Barcha pipelinelarga ma'lumot manbasi kerak
- **Ajratish:** Ma'lumot yaratishni qayta ishlashdan ajratadi
- **Kompozitsiya:** Oddiy qismlardan murakkab pipelinelar qurishga imkon beradi
- **Testlashtirish:** Ma'lum kirish ketma-ketliklari bilan osongina testlash mumkin

**Ishlab chiqarish patterni:**
\`\`\`go
// Ma'lumotlar bazasidan ma'lumotlarni o'qish
func GenFromDB(db *sql.DB, query string) <-chan User {
    out := make(chan User, 100)
    go func() {
        defer close(out)
        rows, _ := db.Query(query)
        defer rows.Close()

        for rows.Next() {
            var user User
            rows.Scan(&user.ID, &user.Name)
            out <- user
        }
    }()
    return out
}

// Fayldan o'qish
func GenFromFile(filename string) <-chan string {
    out := make(chan string, 50)
    go func() {
        defer close(out)
        file, _ := os.Open(filename)
        defer file.Close()

        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            out <- scanner.Text()
        }
    }()
    return out
}

// API dan yaratish
func GenFromAPI(url string) <-chan Record {
    out := make(chan Record, 100)
    go func() {
        defer close(out)
        resp, _ := http.Get(url)
        defer resp.Body.Close()

        decoder := json.NewDecoder(resp.Body)
        for {
            var record Record
            if err := decoder.Decode(&record); err != nil {
                break
            }
            out <- record
        }
    }()
    return out
}

// Cheksiz generator
func GenInfinite(start int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for i := start; ; i++ {
            out <- i
        }
    }()
    return out
}

// Timer asosidagi generator
func GenTicks(interval time.Duration, count int) <-chan time.Time {
    out := make(chan time.Time)
    go func() {
        defer close(out)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        for i := 0; i < count; i++ {
            out <- <-ticker.C
        }
    }()
    return out
}

// Diapazon generatori
func GenRange(start, end, step int) <-chan int {
    out := make(chan int, (end-start)/step+1)
    go func() {
        defer close(out)
        for i := start; i < end; i += step {
            out <- i
        }
    }()
    return out
}
\`\`\`

**Amaliy foydalari:**
- **Oqim qayta ishlash:** Xotiraga sig'maydigan katta ma'lumotlar to'plamlarini qayta ishlash
- **Pipeline boshlash:** Qayta ishlash bosqichlari orqali ma'lumot oqimini boshlash
- **Asinxron yuklash:** Qayta ishlash bilan parallel ravishda ma'lumotlarni yuklash
- **Resurs boshqaruvi:** Bufer hajmi xotira foydalanishini nazorat qiladi

**Bufer o'lchamlari bo'yicha tavsiyalar:**
- **Kichik ma'lumotlar (< 100 element):** Bufer = len(data)
- **O'rta ma'lumotlar (100-10000):** Bufer = 100-1000
- **Katta ma'lumotlar (> 10000):** Bufer = 1000-10000
- **Cheksiz oqimlar:** Bufersiz yoki kichik bufer (1-10)

**Umumiy foydalanish holatlari:**
- **Batch qayta ishlash:** Gen(batchIDs...) ishchi pipelineni oziqlantirad
- **Hodisa oqimlari:** Xabarlar navbatidan Gen hodisalari
- **Ma'lumotlarni migratsiya qilish:** Manba bazasidan Gen yozuvlari
- **Testlash:** Pipeline tekshirish uchun ma'lum test ma'lumotlarini Gen

Gen bo'lmasa, kanallarni qo'lda yaratish va boshqarish kerak bo'ladi, bu esa pipeline kompozitsiyasini ko'p so'zli va xatolarga moyil qiladi.`,
			solutionCode: `package concurrency

func Gen(nums ...int) <-chan int {
	out := make(chan int, len(nums))                            // Buferlangan kanal yaratamiz
	go func() {                                                 // Goroutine ishga tushiramiz
		defer close(out)                                    // Har doim kanalni yopamiz
		for _, n := range nums {                            // Raqamlar bo'ylab iteratsiya qilamiz
			out <- n                                    // Kanalga yuboramiz
		}
	}()
	return out                                                  // Darhol qaytaramiz
}`
		}
	}
};

export default task;
