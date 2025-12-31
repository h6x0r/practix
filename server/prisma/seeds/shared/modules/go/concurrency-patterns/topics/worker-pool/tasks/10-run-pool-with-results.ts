import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-pool-results',
	title: 'Run Pool With Results',
	difficulty: 'hard',	tags: ['go', 'concurrency', 'worker-pool', 'generics', 'results'],
	estimatedTime: '45m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunPoolWithResults** that creates a worker pool processing jobs that return results, collecting all results into a slice.

**Requirements:**
1. Create generic function \`RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error)\`
2. Define \`type ResultJob[T any] func(context.Context) (T, error)\`
3. Handle nil context (return nil, nil)
4. Handle workers <= 0 (set to 1)
5. Create cancellable context for fail-fast behavior
6. Collect results from all jobs into slice
7. Use mutex to protect shared results slice
8. On first error, cancel context and stop processing
9. Return collected results and first error
10. Return nil results if any error occurred

**Type Definition:**
\`\`\`go
type ResultJob[T any] func(context.Context) (T, error)
\`\`\`

**Example:**
\`\`\`go
jobs := make(chan ResultJob[int], 5)

go func() {
    jobs <- func(ctx context.Context) (int, error) {
        return 1, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 2, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 3, nil
    }
    close(jobs)
}()

results, err := RunPoolWithResults(ctx, jobs, 2)
// results = [1, 2, 3], err = nil

// With error
jobs2 := make(chan ResultJob[string], 3)
go func() {
    jobs2 <- func(ctx context.Context) (string, error) {
        return "a", nil
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        return "", errors.New("failed")
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        time.Sleep(time.Second) // might not execute
        return "c", nil
    }
    close(jobs2)
}()

results2, err2 := RunPoolWithResults(ctx, jobs2, 2)
// results2 = nil, err2 = "failed"
\`\`\`

**Constraints:**
- Must use Go generics (type parameter T)
- Must be thread-safe (use mutex for results)
- Must fail-fast (cancel on first error)
- Must return copy of results slice (not original)`,
	initialCode: `package concurrency

import (
	"context"
	"sync"
)

// TODO: Define ResultJob type
type ResultJob[T any] func(context.Context) (T, error)

// TODO: Implement RunPoolWithResults
func RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error) {
	// TODO: Implement
}`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"sort"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	results, err := RunPoolWithResults[int](nil, nil, 1)
	if results != nil || err != nil {
		t.Errorf("expected nil, nil for nil context, got %v, %v", results, err)
	}
}

func Test2(t *testing.T) {
	jobs := make(chan ResultJob[int])
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 1)
	if err != nil {
		t.Errorf("expected nil error for closed channel, got %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %v", results)
	}
}

func Test3(t *testing.T) {
	jobs := make(chan ResultJob[int], 1)
	jobs <- func(ctx context.Context) (int, error) { return 42, nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 1)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if len(results) != 1 || results[0] != 42 {
		t.Errorf("expected [42], got %v", results)
	}
}

func Test4(t *testing.T) {
	jobs := make(chan ResultJob[int], 5)
	for i := 1; i <= 5; i++ {
		v := i
		jobs <- func(ctx context.Context) (int, error) { return v, nil }
	}
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 3)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}
	sort.Ints(results)
	for i, v := range results {
		if v != i+1 {
			t.Errorf("expected %d at position %d, got %d", i+1, i, v)
		}
	}
}

func Test5(t *testing.T) {
	jobs := make(chan ResultJob[string], 2)
	expectedErr := errors.New("test error")
	jobs <- func(ctx context.Context) (string, error) { return "", expectedErr }
	jobs <- func(ctx context.Context) (string, error) { return "ok", nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 1)
	if err != expectedErr {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if results != nil {
		t.Errorf("expected nil results on error, got %v", results)
	}
}

func Test6(t *testing.T) {
	jobs := make(chan ResultJob[int], 1)
	jobs <- func(ctx context.Context) (int, error) { return 1, nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 0)
	if err != nil {
		t.Errorf("expected nil error with workers=0, got %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

func Test7(t *testing.T) {
	jobs := make(chan ResultJob[int], 1)
	jobs <- func(ctx context.Context) (int, error) { return 1, nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, -5)
	if err != nil {
		t.Errorf("expected nil error with negative workers, got %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

func Test8(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	jobs := make(chan ResultJob[int], 1)
	jobs <- func(ctx context.Context) (int, error) { return 1, nil }
	close(jobs)
	results, err := RunPoolWithResults(ctx, jobs, 1)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results on cancel, got %v", results)
	}
}

func Test9(t *testing.T) {
	jobs := make(chan ResultJob[int], 2)
	jobs <- nil
	jobs <- func(ctx context.Context) (int, error) { return 42, nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 1)
	if err != nil {
		t.Errorf("expected nil error when skipping nil job, got %v", err)
	}
	if len(results) != 1 || results[0] != 42 {
		t.Errorf("expected [42], got %v", results)
	}
}

func Test10(t *testing.T) {
	jobs := make(chan ResultJob[string], 3)
	jobs <- func(ctx context.Context) (string, error) { return "a", nil }
	jobs <- func(ctx context.Context) (string, error) { return "b", nil }
	jobs <- func(ctx context.Context) (string, error) { return "c", nil }
	close(jobs)
	results, err := RunPoolWithResults(context.Background(), jobs, 2)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
	sort.Strings(results)
	expected := []string{"a", "b", "c"}
	for i, v := range results {
		if v != expected[i] {
			t.Errorf("expected %s at position %d, got %s", expected[i], i, v)
		}
	}
}
`,
	solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type ResultJob[T any] func(context.Context) (T, error)

func RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error) {
	if ctx == nil {                                                 // Handle nil context
		return nil, nil                                         // Return nil for safety
	}
	if workers <= 0 {                                               // Handle invalid workers count
		workers = 1                                             // Set minimum workers
	}
	ctx, cancel := context.WithCancel(ctx)                          // Create cancellable context
	defer cancel()                                                  // Always cancel to free resources
	var (
		wg       sync.WaitGroup                                 // Track all workers
		mu       sync.Mutex                                     // Protect results slice
		once     sync.Once                                      // Capture first error only
		firstErr error                                          // Store first error
		results  []T                                            // Collect results
	)
	worker := func() {                                              // Worker function
		defer wg.Done()                                         // Decrement counter when done
		for {                                                   // Worker loop
			select {                                        // Check context or receive job
			case <-ctx.Done():                              // Context cancelled
				return                                  // Exit worker
			case job, ok := <-jobs:                         // Receive job from channel
				if !ok {                                // Channel closed
					return                          // Exit worker
				}
				if job == nil {                         // Skip nil job
					continue                        // Next iteration
				}
				res, err := job(ctx)                    // Execute job
				if err != nil {                         // Job returned error
					once.Do(func() {                // Execute once only
						firstErr = err          // Store first error
						cancel()                // Cancel all workers
					})
					return                          // Exit worker
				}
				mu.Lock()                               // Lock results
				results = append(results, res)          // Add result
				mu.Unlock()                             // Unlock results
			}
		}
	}
	wg.Add(workers)                                                 // Add all workers to wait group
	for i := 0; i < workers; i++ {                                  // Create workers
		go worker()                                             // Launch worker goroutine
	}
	wg.Wait()                                                       // Wait for all workers to finish
	if firstErr != nil {                                            // Check if error occurred
		return nil, firstErr                                    // Return nil results and error
	}
	if err := ctx.Err(); err != nil {                               // Check context state
		return nil, err                                         // Return nil results and error
	}
	mu.Lock()                                                       // Lock for reading results
	defer mu.Unlock()                                               // Unlock when done
	copied := make([]T, len(results))                               // Create result copy
	copy(copied, results)                                           // Copy results
	return copied, nil                                              // Return results copy
}`,
			hint1: `Use Go generics: func RunPoolWithResults[T any](...) ([]T, error). Protect results slice with sync.Mutex when appending.`,
			hint2: `On error, use once.Do(func() { firstErr = err; cancel() }) to fail-fast. Return nil results on error. Make copy of results slice before returning: copied := make([]T, len(results)); copy(copied, results).`,
			whyItMatters: `RunPoolWithResults enables concurrent result collection, essential for parallel data fetching, processing, and aggregation operations.

**Why Collect Results:**
- **Parallel Fetching:** Gather data from multiple sources
- **Aggregation:** Combine results from concurrent operations
- **Type Safety:** Generic type ensures compile-time safety
- **Performance:** Process and collect simultaneously

**Production Pattern:**
\`\`\`go
// Fetch multiple users concurrently
func FetchUsers(ctx context.Context, userIDs []string) ([]*User, error) {
    jobs := make(chan ResultJob[*User], len(userIDs))

    go func() {
        defer close(jobs)
        for _, id := range userIDs {
            userID := id
            jobs <- func(ctx context.Context) (*User, error) {
                return fetchUser(ctx, userID)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}

// Parallel API calls
func FetchMultipleEndpoints(ctx context.Context, urls []string) ([]Response, error) {
    jobs := make(chan ResultJob[Response], len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            endpoint := url
            jobs <- func(ctx context.Context) (Response, error) {
                return httpGet(ctx, endpoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}

// Database queries
func QueryMultipleTables(ctx context.Context, queries []Query) ([]QueryResult, error) {
    jobs := make(chan ResultJob[QueryResult], len(queries))

    go func() {
        defer close(jobs)
        for _, q := range queries {
            query := q
            jobs <- func(ctx context.Context) (QueryResult, error) {
                return db.Execute(ctx, query)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}

// File processing
func ProcessFiles(ctx context.Context, filePaths []string) ([]FileData, error) {
    jobs := make(chan ResultJob[FileData], len(filePaths))

    go func() {
        defer close(jobs)
        for _, path := range filePaths {
            filePath := path
            jobs <- func(ctx context.Context) (FileData, error) {
                data, err := os.ReadFile(filePath)
                if err != nil {
                    return FileData{}, err
                }
                return FileData{Path: filePath, Content: data}, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}

// Image thumbnail generation
func GenerateThumbnails(ctx context.Context, images []string) ([]Thumbnail, error) {
    jobs := make(chan ResultJob[Thumbnail], len(images))

    go func() {
        defer close(jobs)
        for _, img := range images {
            imagePath := img
            jobs <- func(ctx context.Context) (Thumbnail, error) {
                return createThumbnail(ctx, imagePath)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 4)
}

// Price aggregation from multiple sources
func GetBestPrice(ctx context.Context, productID string, sources []PriceSource) ([]PriceInfo, error) {
    jobs := make(chan ResultJob[PriceInfo], len(sources))

    go func() {
        defer close(jobs)
        for _, source := range sources {
            s := source
            jobs <- func(ctx context.Context) (PriceInfo, error) {
                return s.GetPrice(ctx, productID)
            }
        }
    }()

    prices, err := RunPoolWithResults(ctx, jobs, len(sources))
    if err != nil {
        return nil, err
    }

    // Sort by price
    sort.Slice(prices, func(i, j int) bool {
        return prices[i].Price < prices[j].Price
    })

    return prices, nil
}

// Validation with detailed results
type ValidationResult struct {
    ItemID string
    Valid  bool
    Errors []string
}

func ValidateItems(ctx context.Context, items []Item) ([]ValidationResult, error) {
    jobs := make(chan ResultJob[ValidationResult], len(items))

    go func() {
        defer close(jobs)
        for _, item := range items {
            i := item
            jobs <- func(ctx context.Context) (ValidationResult, error) {
                errors := validate(i)
                return ValidationResult{
                    ItemID: i.ID,
                    Valid:  len(errors) == 0,
                    Errors: errors,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}

// Search across multiple indices
func SearchAll(ctx context.Context, query string, indices []SearchIndex) ([]SearchResult, error) {
    jobs := make(chan ResultJob[SearchResult], len(indices))

    go func() {
        defer close(jobs)
        for _, idx := range indices {
            index := idx
            jobs <- func(ctx context.Context) (SearchResult, error) {
                results, err := index.Search(ctx, query)
                if err != nil {
                    return SearchResult{}, err
                }
                return SearchResult{
                    Index:   index.Name,
                    Results: results,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(indices))
}

// Metric calculation
func CalculateMetrics(ctx context.Context, dataPoints []DataPoint) ([]Metric, error) {
    jobs := make(chan ResultJob[Metric], len(dataPoints))

    go func() {
        defer close(jobs)
        for _, dp := range dataPoints {
            dataPoint := dp
            jobs <- func(ctx context.Context) (Metric, error) {
                return computeMetric(ctx, dataPoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}

// DNS lookups
func BulkDNSLookup(ctx context.Context, domains []string) ([]DNSRecord, error) {
    jobs := make(chan ResultJob[DNSRecord], len(domains))

    go func() {
        defer close(jobs)
        for _, domain := range domains {
            d := domain
            jobs <- func(ctx context.Context) (DNSRecord, error) {
                ips, err := net.LookupIP(d)
                if err != nil {
                    return DNSRecord{}, err
                }
                return DNSRecord{
                    Domain: d,
                    IPs:    ips,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 20)
}

// Aggregate data from microservices
type ServiceData struct {
    ServiceName string
    Data        interface{}
}

func AggregateFromServices(ctx context.Context, services []Service) ([]ServiceData, error) {
    jobs := make(chan ResultJob[ServiceData], len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            service := svc
            jobs <- func(ctx context.Context) (ServiceData, error) {
                data, err := service.FetchData(ctx)
                if err != nil {
                    return ServiceData{}, err
                }
                return ServiceData{
                    ServiceName: service.Name,
                    Data:        data,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(services))
}

// Transform data with results
func TransformRecords(ctx context.Context, records []Record) ([]TransformedRecord, error) {
    jobs := make(chan ResultJob[TransformedRecord], len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) (TransformedRecord, error) {
                return transform(ctx, r)
            }
        }
    }()

    results, err := RunPoolWithResults(ctx, jobs, 10)
    if err != nil {
        log.Printf("Transformation failed: %v", err)
        return nil, err
    }

    log.Printf("Successfully transformed %d records", len(results))
    return results, nil
}
\`\`\`

**Real-World Benefits:**
- **Performance:** Collect results in parallel
- **Type Safety:** Compile-time type checking with generics
- **Fail-Fast:** Stop on first error
- **Clean API:** Simple function signature

**Generic Benefits:**
\`\`\`go
// Works with any type
users, err := RunPoolWithResults[*User](ctx, userJobs, 10)
images, err := RunPoolWithResults[Image](ctx, imageJobs, 5)
numbers, err := RunPoolWithResults[int](ctx, numberJobs, 20)
\`\`\`

**Thread Safety:**
The mutex protects the shared results slice:
\`\`\`go
mu.Lock()
results = append(results, res)
mu.Unlock()
\`\`\`

Without the mutex, concurrent appends would cause data races.

**Fail-Fast Behavior:**
\`\`\`go
if err != nil {
    once.Do(func() {
        firstErr = err
        cancel()  // Stop all workers
    })
    return  // Exit worker
}
\`\`\`

This ensures quick error reporting and resource cleanup.

**Result Copy:**
\`\`\`go
copied := make([]T, len(results))
copy(copied, results)
return copied, nil
\`\`\`

Returning a copy prevents external modification of internal state.

**When to Use:**
- Fetching data from multiple sources
- Parallel API calls with result collection
- Database query aggregation
- File processing with results
- Any concurrent operation that returns values

**Performance Characteristics:**
- **Memory:** O(n) for results slice
- **Time:** Parallel processing, limited by slowest job
- **Concurrency:** Controlled by workers parameter

This pattern is fundamental for building high-performance data fetching and processing systems.`,	order: 9,
	translations: {
		ru: {
			title: 'Пул воркеров со сбором результатов',
			description: `Реализуйте **RunPoolWithResults**, который создаёт пул воркеров обрабатывающих задачи возвращающие результаты, собирая все результаты в слайс.

**Требования:**
1. Создайте generic функцию \`RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error)\`
2. Определите \`type ResultJob[T any] func(context.Context) (T, error)\`
3. Обработайте nil context (верните nil, nil)
4. Обработайте workers <= 0 (установите в 1)
5. Создайте отменяемый контекст для fail-fast поведения
6. Собирайте результаты всех задач в слайс
7. Используйте mutex для защиты общего слайса результатов
8. При первой ошибке отмените контекст и прекратите обработку
9. Верните собранные результаты и первую ошибку
10. Верните nil результаты если произошла любая ошибка

**Определение типа:**
\`\`\`go
type ResultJob[T any] func(context.Context) (T, error)
\`\`\`

**Пример:**
\`\`\`go
jobs := make(chan ResultJob[int], 5)

go func() {
    jobs <- func(ctx context.Context) (int, error) {
        return 1, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 2, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 3, nil
    }
    close(jobs)
}()

results, err := RunPoolWithResults(ctx, jobs, 2)
// results = [1, 2, 3], err = nil

// С ошибкой
jobs2 := make(chan ResultJob[string], 3)
go func() {
    jobs2 <- func(ctx context.Context) (string, error) {
        return "a", nil
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        return "", errors.New("failed")
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        time.Sleep(time.Second) // может не выполниться
        return "c", nil
    }
    close(jobs2)
}()

results2, err2 := RunPoolWithResults(ctx, jobs2, 2)
// results2 = nil, err2 = "failed"
`,
			hint1: `Используйте Go generics: func RunPoolWithResults[T any](...) ([]T, error). Защитите слайс результатов sync.Mutex при добавлении.`,
			hint2: `При ошибке используйте once.Do(func() { firstErr = err; cancel() }) для fail-fast. Возвращайте nil результаты при ошибке. Сделайте копию слайса результатов перед возвратом: copied := make([]T, len(results)); copy(copied, results).`,
			whyItMatters: `RunPoolWithResults обеспечивает конкурентный сбор результатов с использованием Go generics, критически важен для параллельной выборки данных, обработки и операций агрегации с типобезопасностью.

**Почему Собирать Результаты:**
- **Параллельная Выборка:** Сбор данных из множества источников одновременно
- **Агрегация:** Комбинирование результатов конкурентных операций
- **Безопасность Типов:** Generic тип обеспечивает compile-time безопасность
- **Производительность:** Обработка и сбор происходят параллельно
- **Fail-Fast:** Быстрая остановка при первой ошибке

**Реальные Паттерны:**

**Параллельная Выборка Пользователей:**
\`\`\`go
func FetchUsers(ctx context.Context, userIDs []string) ([]*User, error) {
    jobs := make(chan ResultJob[*User], len(userIDs))

    go func() {
        defer close(jobs)
        for _, id := range userIDs {
            userID := id
            jobs <- func(ctx context.Context) (*User, error) {
                return fetchUser(ctx, userID)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}
\`\`\`

**Параллельные API Вызовы:**
\`\`\`go
func FetchMultipleEndpoints(ctx context.Context, urls []string) ([]Response, error) {
    jobs := make(chan ResultJob[Response], len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            endpoint := url
            jobs <- func(ctx context.Context) (Response, error) {
                return httpGet(ctx, endpoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}
\`\`\`

**Database Запросы:**
\`\`\`go
func QueryMultipleTables(ctx context.Context, queries []Query) ([]QueryResult, error) {
    jobs := make(chan ResultJob[QueryResult], len(queries))

    go func() {
        defer close(jobs)
        for _, q := range queries {
            query := q
            jobs <- func(ctx context.Context) (QueryResult, error) {
                return db.Execute(ctx, query)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}
\`\`\`

**Обработка Файлов:**
\`\`\`go
func ProcessFiles(ctx context.Context, filePaths []string) ([]FileData, error) {
    jobs := make(chan ResultJob[FileData], len(filePaths))

    go func() {
        defer close(jobs)
        for _, path := range filePaths {
            filePath := path
            jobs <- func(ctx context.Context) (FileData, error) {
                data, err := os.ReadFile(filePath)
                if err != nil {
                    return FileData{}, err
                }
                return FileData{Path: filePath, Content: data}, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}
\`\`\`

**Генерация Thumbnail Изображений:**
\`\`\`go
func GenerateThumbnails(ctx context.Context, images []string) ([]Thumbnail, error) {
    jobs := make(chan ResultJob[Thumbnail], len(images))

    go func() {
        defer close(jobs)
        for _, img := range images {
            imagePath := img
            jobs <- func(ctx context.Context) (Thumbnail, error) {
                return createThumbnail(ctx, imagePath)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 4)
}
\`\`\`

**Агрегация Цен из Множества Источников:**
\`\`\`go
func GetBestPrice(ctx context.Context, productID string, sources []PriceSource) ([]PriceInfo, error) {
    jobs := make(chan ResultJob[PriceInfo], len(sources))

    go func() {
        defer close(jobs)
        for _, source := range sources {
            s := source
            jobs <- func(ctx context.Context) (PriceInfo, error) {
                return s.GetPrice(ctx, productID)
            }
        }
    }()

    prices, err := RunPoolWithResults(ctx, jobs, len(sources))
    if err != nil {
        return nil, err
    }

    // Сортировка по цене
    sort.Slice(prices, func(i, j int) bool {
        return prices[i].Price < prices[j].Price
    })

    return prices, nil
}
\`\`\`

**Валидация с Детальными Результатами:**
\`\`\`go
type ValidationResult struct {
    ItemID string
    Valid  bool
    Errors []string
}

func ValidateItems(ctx context.Context, items []Item) ([]ValidationResult, error) {
    jobs := make(chan ResultJob[ValidationResult], len(items))

    go func() {
        defer close(jobs)
        for _, item := range items {
            i := item
            jobs <- func(ctx context.Context) (ValidationResult, error) {
                errors := validate(i)
                return ValidationResult{
                    ItemID: i.ID,
                    Valid:  len(errors) == 0,
                    Errors: errors,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}
\`\`\`

**Поиск Через Множество Индексов:**
\`\`\`go
func SearchAll(ctx context.Context, query string, indices []SearchIndex) ([]SearchResult, error) {
    jobs := make(chan ResultJob[SearchResult], len(indices))

    go func() {
        defer close(jobs)
        for _, idx := range indices {
            index := idx
            jobs <- func(ctx context.Context) (SearchResult, error) {
                results, err := index.Search(ctx, query)
                if err != nil {
                    return SearchResult{}, err
                }
                return SearchResult{
                    Index:   index.Name,
                    Results: results,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(indices))
}
\`\`\`

**Вычисление Метрик:**
\`\`\`go
func CalculateMetrics(ctx context.Context, dataPoints []DataPoint) ([]Metric, error) {
    jobs := make(chan ResultJob[Metric], len(dataPoints))

    go func() {
        defer close(jobs)
        for _, dp := range dataPoints {
            dataPoint := dp
            jobs <- func(ctx context.Context) (Metric, error) {
                return computeMetric(ctx, dataPoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}
\`\`\`

**DNS Lookups:**
\`\`\`go
func BulkDNSLookup(ctx context.Context, domains []string) ([]DNSRecord, error) {
    jobs := make(chan ResultJob[DNSRecord], len(domains))

    go func() {
        defer close(jobs)
        for _, domain := range domains {
            d := domain
            jobs <- func(ctx context.Context) (DNSRecord, error) {
                ips, err := net.LookupIP(d)
                if err != nil {
                    return DNSRecord{}, err
                }
                return DNSRecord{
                    Domain: d,
                    IPs:    ips,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 20)
}
\`\`\`

**Агрегация Данных из Микросервисов:**
\`\`\`go
type ServiceData struct {
    ServiceName string
    Data        interface{}
}

func AggregateFromServices(ctx context.Context, services []Service) ([]ServiceData, error) {
    jobs := make(chan ResultJob[ServiceData], len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            service := svc
            jobs <- func(ctx context.Context) (ServiceData, error) {
                data, err := service.FetchData(ctx)
                if err != nil {
                    return ServiceData{}, err
                }
                return ServiceData{
                    ServiceName: service.Name,
                    Data:        data,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(services))
}
\`\`\`

**Трансформация Данных с Результатами:**
\`\`\`go
func TransformRecords(ctx context.Context, records []Record) ([]TransformedRecord, error) {
    jobs := make(chan ResultJob[TransformedRecord], len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) (TransformedRecord, error) {
                return transform(ctx, r)
            }
        }
    }()

    results, err := RunPoolWithResults(ctx, jobs, 10)
    if err != nil {
        log.Printf("Трансформация провалилась: %v", err)
        return nil, err
    }

    log.Printf("Успешно трансформировано %d записей", len(results))
    return results, nil
}
\`\`\`

**Реальные Преимущества:**
- **Производительность:** Сбор результатов параллельно
- **Безопасность Типов:** Compile-time проверка с generics
- **Fail-Fast:** Остановка при первой ошибке
- **Чистый API:** Простая сигнатура функции

**Generic Преимущества:**
\`\`\`go
// Работает с любым типом
users, err := RunPoolWithResults[*User](ctx, userJobs, 10)
images, err := RunPoolWithResults[Image](ctx, imageJobs, 5)
numbers, err := RunPoolWithResults[int](ctx, numberJobs, 20)
\`\`\`

**Потокобезопасность:**
Mutex защищает общий слайс результатов:
\`\`\`go
mu.Lock()
results = append(results, res)
mu.Unlock()
\`\`\`

Без mutex конкурентные добавления вызывали бы data race.

**Fail-Fast Поведение:**
\`\`\`go
if err != nil {
    once.Do(func() {
        firstErr = err
        cancel()  // Остановить всех воркеров
    })
    return  // Выйти из воркера
}
\`\`\`

Это обеспечивает быстрое сообщение об ошибке и очистку ресурсов.

**Копия Результатов:**
\`\`\`go
copied := make([]T, len(results))
copy(copied, results)
return copied, nil
\`\`\`

Возврат копии предотвращает внешнюю модификацию внутреннего состояния.

**Когда Использовать:**
- Выборка данных из множества источников
- Параллельные API вызовы со сбором результатов
- Агрегация database запросов
- Обработка файлов с результатами
- Любая конкурентная операция возвращающая значения

**Характеристики Производительности:**
- **Память:** O(n) для слайса результатов
- **Время:** Параллельная обработка, ограничена самой медленной задачей
- **Конкурентность:** Контролируется параметром workers

Этот паттерн фундаментален для построения высокопроизводительных систем выборки и обработки данных.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type ResultJob[T any] func(context.Context) (T, error)

func RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error) {
	if ctx == nil {                                                 // Обработка nil контекста
		return nil, nil                                         // Возврат nil для безопасности
	}
	if workers <= 0 {                                               // Обработка неверного количества воркеров
		workers = 1                                             // Установка минимального количества воркеров
	}
	ctx, cancel := context.WithCancel(ctx)                          // Создание отменяемого контекста
	defer cancel()                                                  // Всегда отменять для освобождения ресурсов
	var (
		wg       sync.WaitGroup                                 // Отслеживание всех воркеров
		mu       sync.Mutex                                     // Защита слайса результатов
		once     sync.Once                                      // Захват только первой ошибки
		firstErr error                                          // Хранение первой ошибки
		results  []T                                            // Сбор результатов
	)
	worker := func() {                                              // Функция воркера
		defer wg.Done()                                         // Уменьшение счётчика при завершении
		for {                                                   // Цикл воркера
			select {                                        // Проверка контекста или получение задачи
			case <-ctx.Done():                              // Контекст отменён
				return                                  // Выход из воркера
			case job, ok := <-jobs:                         // Получение задачи из канала
				if !ok {                                // Канал закрыт
					return                          // Выход из воркера
				}
				if job == nil {                         // Пропуск nil задачи
					continue                        // Следующая итерация
				}
				res, err := job(ctx)                    // Выполнение задачи
				if err != nil {                         // Задача вернула ошибку
					once.Do(func() {                // Выполнить только один раз
						firstErr = err          // Сохранить первую ошибку
						cancel()                // Отменить всех воркеров
					})
					return                          // Выход из воркера
				}
				mu.Lock()                               // Блокировка результатов
				results = append(results, res)          // Добавление результата
				mu.Unlock()                             // Разблокировка результатов
			}
		}
	}
	wg.Add(workers)                                                 // Добавление всех воркеров в wait group
	for i := 0; i < workers; i++ {                                  // Создание воркеров
		go worker()                                             // Запуск горутины воркера
	}
	wg.Wait()                                                       // Ожидание завершения всех воркеров
	if firstErr != nil {                                            // Проверка возникновения ошибки
		return nil, firstErr                                    // Возврат nil результатов и ошибки
	}
	if err := ctx.Err(); err != nil {                               // Проверка состояния контекста
		return nil, err                                         // Возврат nil результатов и ошибки
	}
	mu.Lock()                                                       // Блокировка для чтения результатов
	defer mu.Unlock()                                               // Разблокировка при завершении
	copied := make([]T, len(results))                               // Создание копии результатов
	copy(copied, results)                                           // Копирование результатов
	return copied, nil                                              // Возврат копии результатов
}`
		},
		uz: {
			title: 'Natijalarni yig\'adigan worker puli',
			description: `Natijalarni qaytaradigan vazifalarni qayta ishlaydigan, barcha natijalarni slicega yig'adigan worker pulini yaratadigan **RunPoolWithResults** ni amalga oshiring.

**Talablar:**
1. Generic funksiya yarating 'RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error)'
2. 'type ResultJob[T any] func(context.Context) (T, error)' ni aniqlang
3. nil kontekstni ishlang (nil, nil qaytaring)
4. workers <= 0 ni ishlang (1 ga o'rnating)
5. Fail-fast xatti-harakati uchun bekor qilinadigan kontekst yarating
6. Barcha vazifalar natijalarini slicega yig'ing
7. Umumiy natijalar sliceini himoya qilish uchun mutexdan foydalaning
8. Birinchi xatoda kontekstni bekor qiling va qayta ishlashni to'xtating
9. Yig'ilgan natijalar va birinchi xatoni qaytaring
10. Agar biron xato yuz bergan bo'lsa nil natijalarni qaytaring

**Tur ta'rifi:**
\`\`\`go
type ResultJob[T any] func(context.Context) (T, error)
\`\`\`

**Misol:**
\`\`\`go
jobs := make(chan ResultJob[int], 5)

go func() {
    jobs <- func(ctx context.Context) (int, error) {
        return 1, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 2, nil
    }
    jobs <- func(ctx context.Context) (int, error) {
        return 3, nil
    }
    close(jobs)
}()

results, err := RunPoolWithResults(ctx, jobs, 2)
// results = [1, 2, 3], err = nil

// Xato bilan
jobs2 := make(chan ResultJob[string], 3)
go func() {
    jobs2 <- func(ctx context.Context) (string, error) {
        return "a", nil
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        return "", errors.New("failed")
    }
    jobs2 <- func(ctx context.Context) (string, error) {
        time.Sleep(time.Second) // bajarilmasligi mumkin
        return "c", nil
    }
    close(jobs2)
}()

results2, err2 := RunPoolWithResults(ctx, jobs2, 2)
// results2 = nil, err2 = "failed"
`,
			hint1: `Go genericsdan foydalaning: func RunPoolWithResults[T any](...) ([]T, error). Qo'shishda natijalar sliceini sync.Mutex bilan himoya qiling.`,
			hint2: `Xatoda fail-fast uchun once.Do(func() { firstErr = err; cancel() }) dan foydalaning. Xatoda nil natijalarni qaytaring. Qaytarishdan oldin natijalar sliceining nusxasini yarating: copied := make([]T, len(results)); copy(copied, results).`,
			whyItMatters: `RunPoolWithResults Go genericsdan foydalangan holda parallel natijalarni yig'ishni ta'minlaydi, tur xavfsizligi bilan parallel ma'lumotlarni olish, qayta ishlash va agregatsiya operatsiyalari uchun juda muhim.

**Nima Uchun Natijalarni Yig'ish:**
- **Parallel Olish:** Ko'plab manbalardan bir vaqtning o'zida ma'lumotlarni yig'ish
- **Agregatsiya:** Parallel operatsiyalar natijalarini birlashtirish
- **Tur Xavfsizligi:** Generic tur compile-time xavfsizlikni ta'minlaydi
- **Samaradorlik:** Qayta ishlash va yig'ish parallel amalga oshiriladi
- **Fail-Fast:** Birinchi xatoda tez to'xtatish

**Haqiqiy Patternlar:**

**Foydalanuvchilarni Parallel Olish:**
\`\`\`go
func FetchUsers(ctx context.Context, userIDs []string) ([]*User, error) {
    jobs := make(chan ResultJob[*User], len(userIDs))

    go func() {
        defer close(jobs)
        for _, id := range userIDs {
            userID := id
            jobs <- func(ctx context.Context) (*User, error) {
                return fetchUser(ctx, userID)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}
\`\`\`

**Parallel API Chaqiruvlari:**
\`\`\`go
func FetchMultipleEndpoints(ctx context.Context, urls []string) ([]Response, error) {
    jobs := make(chan ResultJob[Response], len(urls))

    go func() {
        defer close(jobs)
        for _, url := range urls {
            endpoint := url
            jobs <- func(ctx context.Context) (Response, error) {
                return httpGet(ctx, endpoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}
\`\`\`

**Database So'rovlari:**
\`\`\`go
func QueryMultipleTables(ctx context.Context, queries []Query) ([]QueryResult, error) {
    jobs := make(chan ResultJob[QueryResult], len(queries))

    go func() {
        defer close(jobs)
        for _, q := range queries {
            query := q
            jobs <- func(ctx context.Context) (QueryResult, error) {
                return db.Execute(ctx, query)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 5)
}
\`\`\`

**Fayllarni Qayta Ishlash:**
\`\`\`go
func ProcessFiles(ctx context.Context, filePaths []string) ([]FileData, error) {
    jobs := make(chan ResultJob[FileData], len(filePaths))

    go func() {
        defer close(jobs)
        for _, path := range filePaths {
            filePath := path
            jobs <- func(ctx context.Context) (FileData, error) {
                data, err := os.ReadFile(filePath)
                if err != nil {
                    return FileData{}, err
                }
                return FileData{Path: filePath, Content: data}, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}
\`\`\`

**Rasm Thumbnail larini Yaratish:**
\`\`\`go
func GenerateThumbnails(ctx context.Context, images []string) ([]Thumbnail, error) {
    jobs := make(chan ResultJob[Thumbnail], len(images))

    go func() {
        defer close(jobs)
        for _, img := range images {
            imagePath := img
            jobs <- func(ctx context.Context) (Thumbnail, error) {
                return createThumbnail(ctx, imagePath)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 4)
}
\`\`\`

**Ko'plab Manbalardan Narxlarni Agregatsiya:**
\`\`\`go
func GetBestPrice(ctx context.Context, productID string, sources []PriceSource) ([]PriceInfo, error) {
    jobs := make(chan ResultJob[PriceInfo], len(sources))

    go func() {
        defer close(jobs)
        for _, source := range sources {
            s := source
            jobs <- func(ctx context.Context) (PriceInfo, error) {
                return s.GetPrice(ctx, productID)
            }
        }
    }()

    prices, err := RunPoolWithResults(ctx, jobs, len(sources))
    if err != nil {
        return nil, err
    }

    // Narx bo'yicha saralash
    sort.Slice(prices, func(i, j int) bool {
        return prices[i].Price < prices[j].Price
    })

    return prices, nil
}
\`\`\`

**Batafsil Natijalar bilan Validatsiya:**
\`\`\`go
type ValidationResult struct {
    ItemID string
    Valid  bool
    Errors []string
}

func ValidateItems(ctx context.Context, items []Item) ([]ValidationResult, error) {
    jobs := make(chan ResultJob[ValidationResult], len(items))

    go func() {
        defer close(jobs)
        for _, item := range items {
            i := item
            jobs <- func(ctx context.Context) (ValidationResult, error) {
                errors := validate(i)
                return ValidationResult{
                    ItemID: i.ID,
                    Valid:  len(errors) == 0,
                    Errors: errors,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 10)
}
\`\`\`

**Ko'plab Indekslar Bo'yicha Qidiruv:**
\`\`\`go
func SearchAll(ctx context.Context, query string, indices []SearchIndex) ([]SearchResult, error) {
    jobs := make(chan ResultJob[SearchResult], len(indices))

    go func() {
        defer close(jobs)
        for _, idx := range indices {
            index := idx
            jobs <- func(ctx context.Context) (SearchResult, error) {
                results, err := index.Search(ctx, query)
                if err != nil {
                    return SearchResult{}, err
                }
                return SearchResult{
                    Index:   index.Name,
                    Results: results,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(indices))
}
\`\`\`

**Metrikalarni Hisoblash:**
\`\`\`go
func CalculateMetrics(ctx context.Context, dataPoints []DataPoint) ([]Metric, error) {
    jobs := make(chan ResultJob[Metric], len(dataPoints))

    go func() {
        defer close(jobs)
        for _, dp := range dataPoints {
            dataPoint := dp
            jobs <- func(ctx context.Context) (Metric, error) {
                return computeMetric(ctx, dataPoint)
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, runtime.NumCPU())
}
\`\`\`

**DNS Lookuplar:**
\`\`\`go
func BulkDNSLookup(ctx context.Context, domains []string) ([]DNSRecord, error) {
    jobs := make(chan ResultJob[DNSRecord], len(domains))

    go func() {
        defer close(jobs)
        for _, domain := range domains {
            d := domain
            jobs <- func(ctx context.Context) (DNSRecord, error) {
                ips, err := net.LookupIP(d)
                if err != nil {
                    return DNSRecord{}, err
                }
                return DNSRecord{
                    Domain: d,
                    IPs:    ips,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, 20)
}
\`\`\`

**Mikroservislardan Ma'lumotlarni Agregatsiya:**
\`\`\`go
type ServiceData struct {
    ServiceName string
    Data        interface{}
}

func AggregateFromServices(ctx context.Context, services []Service) ([]ServiceData, error) {
    jobs := make(chan ResultJob[ServiceData], len(services))

    go func() {
        defer close(jobs)
        for _, svc := range services {
            service := svc
            jobs <- func(ctx context.Context) (ServiceData, error) {
                data, err := service.FetchData(ctx)
                if err != nil {
                    return ServiceData{}, err
                }
                return ServiceData{
                    ServiceName: service.Name,
                    Data:        data,
                }, nil
            }
        }
    }()

    return RunPoolWithResults(ctx, jobs, len(services))
}
\`\`\`

**Ma'lumotlarni Natijalar Bilan Transformatsiya:**
\`\`\`go
func TransformRecords(ctx context.Context, records []Record) ([]TransformedRecord, error) {
    jobs := make(chan ResultJob[TransformedRecord], len(records))

    go func() {
        defer close(jobs)
        for _, record := range records {
            r := record
            jobs <- func(ctx context.Context) (TransformedRecord, error) {
                return transform(ctx, r)
            }
        }
    }()

    results, err := RunPoolWithResults(ctx, jobs, 10)
    if err != nil {
        log.Printf("Transformatsiya muvaffaqiyatsiz: %v", err)
        return nil, err
    }

    log.Printf("Muvaffaqiyatli %d yozuvlar transformatsiya qilindi", len(results))
    return results, nil
}
\`\`\`

**Haqiqiy Foydalari:**
- **Samaradorlik:** Natijalarni parallel yig'ish
- **Tur Xavfsizligi:** Generics bilan compile-time tekshirish
- **Fail-Fast:** Birinchi xatoda to'xtatish
- **Toza API:** Oddiy funksiya imzosi

**Generic Afzalliklari:**
\`\`\`go
// Har qanday tur bilan ishlaydi
users, err := RunPoolWithResults[*User](ctx, userJobs, 10)
images, err := RunPoolWithResults[Image](ctx, imageJobs, 5)
numbers, err := RunPoolWithResults[int](ctx, numberJobs, 20)
\`\`\`

**Thread Xavfsizligi:**
Mutex umumiy natijalar sliceini himoya qiladi:
\`\`\`go
mu.Lock()
results = append(results, res)
mu.Unlock()
\`\`\`

Mutexsiz parallel qo'shishlar data race ni keltirib chiqaradi.

**Fail-Fast Xatti-Harakati:**
\`\`\`go
if err != nil {
    once.Do(func() {
        firstErr = err
        cancel()  // Barcha workerlarni to'xtating
    })
    return  // Workerdan chiqing
}
\`\`\`

Bu tez xato xabari va resurslarni tozalashni ta'minlaydi.

**Natijalar Nusxasi:**
\`\`\`go
copied := make([]T, len(results))
copy(copied, results)
return copied, nil
\`\`\`

Nusxa qaytarish ichki holatning tashqi o'zgartirilishidan himoya qiladi.

**Qachon Ishlatish:**
- Ko'plab manbalardan ma'lumotlarni olish
- Natijalarni yig'ish bilan parallel API chaqiruvlari
- Database so'rovlari agregatsiyasi
- Natijalar bilan fayllarni qayta ishlash
- Qiymatlarni qaytaradigan har qanday parallel operatsiya

**Samaradorlik Xarakteristikalari:**
- **Xotira:** Natijalar slicesi uchun O(n)
- **Vaqt:** Parallel qayta ishlash, eng sekin vazifa bilan cheklangan
- **Parallellik:** workers parametri bilan boshqariladi

Bu pattern yuqori samarali ma'lumotlarni olish va qayta ishlash tizimlarini qurish uchun asosiy hisoblanadi.`,
			solutionCode: `package concurrency

import (
	"context"
	"sync"
)

type ResultJob[T any] func(context.Context) (T, error)

func RunPoolWithResults[T any](ctx context.Context, jobs <-chan ResultJob[T], workers int) ([]T, error) {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return nil, nil                                         // Xavfsizlik uchun nil qaytarish
	}
	if workers <= 0 {                                               // Noto'g'ri workerlar sonini ishlash
		workers = 1                                             // Minimal workerlar sonini o'rnatish
	}
	ctx, cancel := context.WithCancel(ctx)                          // Bekor qilinadigan kontekst yaratish
	defer cancel()                                                  // Resurslarni ozod qilish uchun doim bekor qilish
	var (
		wg       sync.WaitGroup                                 // Barcha workerlarni kuzatish
		mu       sync.Mutex                                     // Natijalar sliceini himoya qilish
		once     sync.Once                                      // Faqat birinchi xatoni ushlash
		firstErr error                                          // Birinchi xatoni saqlash
		results  []T                                            // Natijalarni yig'ish
	)
	worker := func() {                                              // Worker funksiyasi
		defer wg.Done()                                         // Tugaganda hisoblagichni kamaytirish
		for {                                                   // Worker tsikli
			select {                                        // Kontekstni tekshirish yoki vazifani qabul qilish
			case <-ctx.Done():                              // Kontekst bekor qilindi
				return                                  // Workerdan chiqish
			case job, ok := <-jobs:                         // Kanaldan vazifani qabul qilish
				if !ok {                                // Kanal yopildi
					return                          // Workerdan chiqish
				}
				if job == nil {                         // nil vazifani o'tkazib yuborish
					continue                        // Keyingi iteratsiya
				}
				res, err := job(ctx)                    // Vazifani bajarish
				if err != nil {                         // Vazifa xato qaytardi
					once.Do(func() {                // Faqat bir marta bajarish
						firstErr = err          // Birinchi xatoni saqlash
						cancel()                // Barcha workerlarni bekor qilish
					})
					return                          // Workerdan chiqish
				}
				mu.Lock()                               // Natijalarni blokirovka qilish
				results = append(results, res)          // Natijani qo'shish
				mu.Unlock()                             // Natijalarni blokdan chiqarish
			}
		}
	}
	wg.Add(workers)                                                 // Barcha workerlarni wait groupga qo'shish
	for i := 0; i < workers; i++ {                                  // Workerlarni yaratish
		go worker()                                             // Worker goroutinasini ishga tushirish
	}
	wg.Wait()                                                       // Barcha workerlar tugashini kutish
	if firstErr != nil {                                            // Xato yuz berganligini tekshirish
		return nil, firstErr                                    // nil natijalar va xatoni qaytarish
	}
	if err := ctx.Err(); err != nil {                               // Kontekst holatini tekshirish
		return nil, err                                         // nil natijalar va xatoni qaytarish
	}
	mu.Lock()                                                       // Natijalarni o'qish uchun blokirovka qilish
	defer mu.Unlock()                                               // Tugaganda blokdan chiqarish
	copied := make([]T, len(results))                               // Natijalar nusxasini yaratish
	copy(copied, results)                                           // Natijalarni nusxalash
	return copied, nil                                              // Natijalar nusxasini qaytarish
}`
		}
	}
};

export default task;
