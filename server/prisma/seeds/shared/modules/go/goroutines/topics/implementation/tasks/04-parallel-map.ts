import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-goroutines-parallel-map',
	title: 'Parallel Map with Goroutines',
	difficulty: 'hard',
	tags: ['go', 'goroutines', 'concurrency', 'parallelism'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a parallel map function that applies a transformation to slice elements concurrently while preserving order.

**Requirements:**
1. **ParallelMap**: Transform slice elements using goroutines
2. **Order Preservation**: Return results in original input order
3. **Concurrency Control**: Limit number of concurrent goroutines
4. **Error Handling**: Support functions that return errors

**Implementation Pattern:**
\`\`\`go
func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
    n := len(items)
    results := make([]R, n)
    errors := make([]error, n)

    // Channel to control concurrency
    sem := make(chan struct{}, concurrency)
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, val T) {
            defer wg.Done()
            sem <- struct{}{}        // acquire semaphore
            defer func() { <-sem }() // release semaphore

            result, err := fn(val)
            results[idx] = result
            errors[idx] = err
        }(i, item)
    }

    wg.Wait()

    // Check for errors
    for i, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("error at index %d: %w", i, err)
        }
    }

    return results, nil
}
\`\`\`

**Example Usage:**
\`\`\`go
// Process images in parallel
images := []string{"img1.jpg", "img2.jpg", "img3.jpg"}
results, err := ParallelMap(images, func(path string) (Image, error) {
    return processImage(path)
}, 3) // max 3 concurrent processes

// API calls with rate limiting
urls := []string{"api/user/1", "api/user/2", "api/user/3"}
users, err := ParallelMap(urls, fetchUser, 5) // max 5 concurrent requests

// Data transformation
numbers := []int{1, 2, 3, 4, 5}
squared, _ := ParallelMap(numbers, func(n int) (int, error) {
    return n * n, nil
}, 10)
// Result: [1, 4, 9, 16, 25] (order preserved)
\`\`\`

**Constraints:**
- Must preserve input order in output
- Must limit concurrent goroutines to concurrency parameter
- Must handle errors from transformation function
- Must use sync.WaitGroup for coordination`,
	initialCode: `package goroutinesx

import (
	"fmt"
	"sync"
)

// TODO: Implement ParallelMap
// Create result and error slices
// Launch goroutines with concurrency control
// Apply fn to each item and store result at original index
// Wait for all goroutines and check errors
func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
	// TODO: Implement
}`,
	solutionCode: `package goroutinesx

import (
	"fmt"
	"sync"
)

func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
	if len(items) == 0 {
		return []R{}, nil	// handle empty input
	}

	if concurrency <= 0 {
		concurrency = 1	// ensure valid concurrency
	}

	n := len(items)
	results := make([]R, n)	// pre-allocate results slice
	errors := make([]error, n)	// track errors per index

	// Semaphore channel to control concurrency
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)	// increment wait group
		go func(idx int, val T) {
			defer wg.Done()	// decrement when done

			sem <- struct{}{}	// acquire semaphore slot
			defer func() { <-sem }()	// release semaphore slot

			result, err := fn(val)	// apply transformation
			results[idx] = result	// store at original index
			errors[idx] = err	// store error if any
		}(i, item)	// pass by value to avoid closure issues
	}

	wg.Wait()	// wait for all goroutines

	// Check if any errors occurred
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("error at index %d: %w", i, err)
		}
	}

	return results, nil
}`,
	testCode: `package goroutinesx

import (
	"errors"
	"testing"
)

func Test1(t *testing.T) {
	// ParallelMap transforms elements
	items := []int{1, 2, 3}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n * 2, nil
	}, 3)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
}

func Test2(t *testing.T) {
	// ParallelMap preserves order
	items := []int{1, 2, 3, 4, 5}
	results, _ := ParallelMap(items, func(n int) (int, error) {
		return n * n, nil
	}, 2)

	expected := []int{1, 4, 9, 16, 25}
	for i, v := range expected {
		if results[i] != v {
			t.Errorf("at index %d: expected %d, got %d", i, v, results[i])
		}
	}
}

func Test3(t *testing.T) {
	// ParallelMap returns error from function
	items := []int{1, 2, 3}
	_, err := ParallelMap(items, func(n int) (int, error) {
		if n == 2 {
			return 0, errors.New("test error")
		}
		return n, nil
	}, 3)

	if err == nil {
		t.Error("expected error")
	}
}

func Test4(t *testing.T) {
	// ParallelMap handles empty slice
	items := []int{}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n, nil
	}, 3)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func Test5(t *testing.T) {
	// ParallelMap with concurrency 1
	items := []int{1, 2, 3}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n + 1, nil
	}, 1)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if results[0] != 2 || results[1] != 3 || results[2] != 4 {
		t.Errorf("unexpected results: %v", results)
	}
}

func Test6(t *testing.T) {
	// ParallelMap with high concurrency
	items := []int{1, 2, 3, 4, 5}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n * 10, nil
	}, 100)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(results) != 5 {
		t.Errorf("expected 5 results")
	}
}

func Test7(t *testing.T) {
	// ParallelMap with single element
	items := []int{42}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n * 2, nil
	}, 3)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(results) != 1 || results[0] != 84 {
		t.Errorf("expected [84], got %v", results)
	}
}

func Test8(t *testing.T) {
	// ParallelMap with string transformation
	items := []string{"a", "b", "c"}
	results, err := ParallelMap(items, func(s string) (string, error) {
		return s + s, nil
	}, 3)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if results[0] != "aa" || results[1] != "bb" || results[2] != "cc" {
		t.Errorf("unexpected results: %v", results)
	}
}

func Test9(t *testing.T) {
	// ParallelMap returns nil on error
	items := []int{1, 2, 3}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return 0, errors.New("error")
	}, 3)

	if err == nil {
		t.Error("expected error")
	}
	if results != nil {
		t.Error("expected nil results on error")
	}
}

func Test10(t *testing.T) {
	// ParallelMap handles zero/negative concurrency
	items := []int{1, 2, 3}
	results, err := ParallelMap(items, func(n int) (int, error) {
		return n, nil
	}, 0)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results")
	}
}
`,
	hint1: `Create results and errors slices with length len(items). Use buffered channel as semaphore: sem := make(chan struct{}, concurrency).`,
	hint2: `In goroutine: acquire semaphore with sem <- struct{}{}, defer release with <-sem. Store result at results[idx] to preserve order. Use closure parameters (idx int, val T) to avoid race conditions.`,
	whyItMatters: `Parallel map is fundamental for high-performance data processing. It enables efficient CPU utilization for I/O-bound tasks (API calls, file operations) and CPU-bound tasks (image processing, calculations). The semaphore pattern prevents resource exhaustion while maximizing throughput.`,
	order: 3,
	translations: {
		ru: {
			title: 'Параллельный Map',
			solutionCode: `package goroutinesx

import (
	"fmt"
	"sync"
)

func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
	if len(items) == 0 {
		return []R{}, nil	// обрабатываем пустой вход
	}

	if concurrency <= 0 {
		concurrency = 1	// гарантируем валидную concurrency
	}

	n := len(items)
	results := make([]R, n)	// предварительно выделяем слайс результатов
	errors := make([]error, n)	// отслеживаем ошибки по индексу

	// Канал-семафор для контроля concurrency
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)	// увеличиваем wait group
		go func(idx int, val T) {
			defer wg.Done()	// уменьшаем при завершении

			sem <- struct{}{}	// захватываем слот семафора
			defer func() { <-sem }()	// освобождаем слот семафора

			result, err := fn(val)	// применяем преобразование
			results[idx] = result	// сохраняем по оригинальному индексу
			errors[idx] = err	// сохраняем ошибку если есть
		}(i, item)	// передаём по значению чтобы избежать проблем с замыканием
	}

	wg.Wait()	// ждём все горутины

	// Проверяем наличие ошибок
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("error at index %d: %w", i, err)
		}
	}

	return results, nil
}`,
			description: `Реализуйте функцию параллельного map которая применяет преобразование к элементам слайса конкурентно сохраняя порядок.

**Требования:**
1. **ParallelMap**: Преобразуйте элементы слайса используя горутины
2. **Сохранение порядка**: Верните результаты в исходном порядке входных данных
3. **Управление конкурентностью**: Ограничьте количество конкурентных горутин
4. **Обработка ошибок**: Поддержите функции которые возвращают ошибки

**Паттерн реализации:**
\`\`\`go
func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
    n := len(items)
    results := make([]R, n)
    errors := make([]error, n)

    // Канал для контроля конкурентности
    sem := make(chan struct{}, concurrency)
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, val T) {
            defer wg.Done()
            sem <- struct{}{}        // захват семафора
            defer func() { <-sem }() // освобождение семафора

            result, err := fn(val)
            results[idx] = result
            errors[idx] = err
        }(i, item)
    }

    wg.Wait()

    // Проверка ошибок
    for i, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("error at index %d: %w", i, err)
        }
    }

    return results, nil
}
\`\`\`

**Пример использования:**
\`\`\`go
// Обработка изображений параллельно
images := []string{"img1.jpg", "img2.jpg", "img3.jpg"}
results, err := ParallelMap(images, func(path string) (Image, error) {
    return processImage(path)
}, 3) // максимум 3 конкурентных процесса

// API вызовы с ограничением скорости
urls := []string{"api/user/1", "api/user/2", "api/user/3"}
users, err := ParallelMap(urls, fetchUser, 5) // максимум 5 конкурентных запросов

// Преобразование данных
numbers := []int{1, 2, 3, 4, 5}
squared, _ := ParallelMap(numbers, func(n int) (int, error) {
    return n * n, nil
}, 10)
// Результат: [1, 4, 9, 16, 25] (порядок сохранён)
\`\`\`

**Ограничения:**
- Должен сохранять порядок входных данных в выходных
- Должен ограничивать конкурентные горутины параметром concurrency
- Должен обрабатывать ошибки от функции преобразования
- Должен использовать sync.WaitGroup для координации`,
			hint1: `Создайте слайсы results и errors с длиной len(items). Используйте буферизованный канал как семафор: sem := make(chan struct{}, concurrency).`,
			hint2: `В горутине: захватывайте семафор через sem <- struct{}{}, освобождайте через defer <-sem. Сохраняйте результат в results[idx] для сохранения порядка. Используйте параметры замыкания (idx int, val T) чтобы избежать race conditions.`,
			whyItMatters: `Параллельный map фундаментален для высокопроизводительной обработки данных. Он обеспечивает эффективное использование CPU для I/O-bound задач (API вызовы, файловые операции) и CPU-bound задач (обработка изображений, вычисления). Паттерн семафора предотвращает исчерпание ресурсов при максимизации пропускной способности.`
		},
		uz: {
			title: `Parallel Map`,
			solutionCode: `package goroutinesx

import (
	"fmt"
	"sync"
)

func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
	if len(items) == 0 {
		return []R{}, nil	// bo'sh kirishni qayta ishlaymiz
	}

	if concurrency <= 0 {
		concurrency = 1	// to'g'ri concurrency ni ta'minlaymiz
	}

	n := len(items)
	results := make([]R, n)	// natijalar slice ini oldindan ajratamiz
	errors := make([]error, n)	// indeks bo'yicha xatolarni kuzatamiz

	// Concurrency ni boshqarish uchun semafor kanali
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)	// wait group ni oshiramiz
		go func(idx int, val T) {
			defer wg.Done()	// tugaganda kamaytiramiz

			sem <- struct{}{}	// semafor slotini egallymiz
			defer func() { <-sem }()	// semafor slotini bo'shatamiz

			result, err := fn(val)	// transformatsiyani qo'llaymiz
			results[idx] = result	// asl indeksga saqlaymiz
			errors[idx] = err	// xato bo'lsa saqlaymiz
		}(i, item)	// yopilish muammolaridan qochish uchun qiymat bo'yicha uzatamiz
	}

	wg.Wait()	// barcha goroutinelarni kutamiz

	// Xatolar mavjudligini tekshiramiz
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("error at index %d: %w", i, err)
		}
	}

	return results, nil
}`,
			description: `Tartibni saqlab slice elementlariga transformatsiyani concurrent qo'llaydigan parallel map funktsiyasini amalga oshiring.

**Talablar:**
1. **ParallelMap**: Goroutinelar yordamida slice elementlarini transformatsiya qilish
2. **Tartibni saqlash**: Natijalarni asl kirish tartibida qaytarish
3. **Concurrency boshqaruvi**: Concurrent goroutinelar sonini cheklash
4. **Xatolarni qayta ishlash**: Xato qaytaruvchi funktsiyalarni qo'llab-quvvatlash

**Amalga oshirish patterni:**
\`\`\`go
func ParallelMap[T, R any](items []T, fn func(T) (R, error), concurrency int) ([]R, error) {
    n := len(items)
    results := make([]R, n)
    errors := make([]error, n)

    // Concurrency ni boshqarish uchun kanal
    sem := make(chan struct{}, concurrency)
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, val T) {
            defer wg.Done()
            sem <- struct{}{}        // semaforni egallash
            defer func() { <-sem }() // semaforni bo'shatish

            result, err := fn(val)
            results[idx] = result
            errors[idx] = err
        }(i, item)
    }

    wg.Wait()

    // Xatolarni tekshirish
    for i, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("error at index %d: %w", i, err)
        }
    }

    return results, nil
}
\`\`\`

**Misol:**
\`\`\`go
// Rasmlarni parallel qayta ishlash
images := []string{"img1.jpg", "img2.jpg", "img3.jpg"}
results, err := ParallelMap(images, func(path string) (Image, error) {
    return processImage(path)
}, 3) // maksimal 3 ta concurrent jarayon

// Rate limiting bilan API chaqiruvlar
urls := []string{"api/user/1", "api/user/2", "api/user/3"}
users, err := ParallelMap(urls, fetchUser, 5) // maksimal 5 ta concurrent so'rov

// Ma'lumotlarni transformatsiya qilish
numbers := []int{1, 2, 3, 4, 5}
squared, _ := ParallelMap(numbers, func(n int) (int, error) {
    return n * n, nil
}, 10)
// Natija: [1, 4, 9, 16, 25] (tartib saqlangan)
\`\`\`

**Cheklovlar:**
- Chiqishda kirish tartibini saqlash kerak
- Concurrent goroutinelarni concurrency parametriga cheklash kerak
- Transformatsiya funktsiyasidan xatolarni qayta ishlash kerak
- Koordinatsiya uchun sync.WaitGroup ishlatish kerak`,
			hint1: `results va errors slice larini len(items) uzunlikda yarating. Buferli kanalni semafor sifatida ishlating: sem := make(chan struct{}, concurrency).`,
			hint2: `Goroutine da: semaforni sem <- struct{}{} orqali egalang, defer <-sem orqali bo'shating. Tartibni saqlash uchun natijani results[idx] ga saqlang. Race conditionlardan qochish uchun yopilish parametrlaridan foydalaning (idx int, val T).`,
			whyItMatters: `Parallel map yuqori unumdorlikdagi ma'lumotlarni qayta ishlash uchun asosiy hisoblanadi. U I/O-bound vazifalar (API chaqiruvlar, fayl operatsiyalari) va CPU-bound vazifalar (rasm qayta ishlash, hisoblashlar) uchun samarali CPU foydalanishni ta'minlaydi. Semafor pattern resurslar tugashini oldini olgan holda throughput ni maksimallashtiradi.`
		}
	}
};

export default task;
