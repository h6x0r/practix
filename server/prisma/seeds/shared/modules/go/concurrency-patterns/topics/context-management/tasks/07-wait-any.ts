import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-wait-any',
	title: 'Wait Any',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'race'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **WaitAny** that waits for the first signal channel to close, returning its index.

**Requirements:**
1. Create function \`WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error)\`
2. Handle nil context (use Background)
3. Handle empty signals (return -1, nil)
4. Launch goroutine for each signal to wait
5. Return index of first signal received
6. Cancel waiting for other signals when first arrives
7. Return -1 and context error if context canceled first
8. Use buffered result channel to prevent goroutine leaks

**Example:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig2) // First to close
}()
go func() {
    time.Sleep(300 * time.Millisecond)
    close(sig3)
}()

idx, err := WaitAny(context.Background(), sig1, sig2, sig3)
// idx = 1, err = nil (sig2 was first)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

idx, err = WaitAny(ctx, sig1, sig2, sig3)
// idx = -1, err = context.DeadlineExceeded
\`\`\`

**Constraints:**
- Must return index of first signal
- Must cancel other waiters
- Must handle context cancellation
- Must prevent goroutine leaks`,
	initialCode: `package concurrency

import (
	"context"
)

// TODO: Implement WaitAny
func WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error) {
	var zero int
	return zero, nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

func WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error) {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	if len(signals) == 0 {                                      // No signals to wait for
		return -1, nil                                      // Return -1 for no signals
	}
	ctxAny, cancel := context.WithCancel(ctx)                   // Create cancelable context
	result := make(chan int, 1)                                 // Buffered channel for result
	for idx, ch := range signals {                              // Iterate over all signals
		idx := idx                                          // Capture loop variable
		go func(ch <-chan struct{}) {                       // Wait for each signal in goroutine
			select {
			case <-ctxAny.Done():                       // Context canceled (another won)
			case <-ch:                                  // This signal received
				select {
				case result <- idx:                 // Try to send index
					cancel()                    // Cancel other waiters
				default:                            // Result already sent
				}
			}
		}(ch)
	}
	select {
	case <-ctxAny.Done():                                       // Context canceled
		if err := ctx.Err(); err != nil {                   // Parent context error
			return -1, err                              // Return parent error
		}
		return -1, context.Canceled                         // Child canceled (race lost)
	case idx := <-result:                                       // First signal index
		return idx, nil                                     // Return winning index
	}
}`,
			hint1: `Create context.WithCancel to cancel all waiters when first signal arrives. Use buffered channel for result to prevent blocking.`,
			hint2: `Launch goroutine for each signal. First one to send to result channel cancels context. Use select default to avoid blocking if result already sent.`,
			whyItMatters: `WaitAny implements the first-responder pattern, essential for racing multiple operations and using whichever completes first.

**Why Wait Any:**
- **Performance:** Use fastest response available
- **Redundancy:** Query multiple sources, use first
- **Failover:** Try multiple backends, use first success
- **Load Balancing:** Distribute load across services

**Production Pattern:**
\`\`\`go
// Query multiple replicas, use first response
func QueryReplicas(ctx context.Context, query string, replicas []Database) (Result, error) {
    signals := make([]<-chan struct{}, len(replicas))
    results := make([]Result, len(replicas))

    for i, db := range replicas {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, database Database, d chan struct{}) {
            defer close(d)
            results[idx] = database.Query(query)
        }(i, db, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Result{}, err
    }

    return results[idx], nil
}

// Call multiple APIs, use fastest
func FetchFromMultipleSources(ctx context.Context, sources []string) (Data, error) {
    signals := make([]<-chan struct{}, len(sources))
    dataSlice := make([]Data, len(sources))

    for i, source := range sources {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            dataSlice[idx] = fetch(url)
        }(i, source, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Data{}, err
    }

    return dataSlice[idx], nil
}

// Try multiple cache servers
func GetFromCache(ctx context.Context, key string, caches []*Cache) (Value, error) {
    signals := make([]<-chan struct{}, len(caches))
    values := make([]Value, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, c *Cache, d chan struct{}) {
            defer close(d)
            values[idx] = c.Get(key)
        }(i, cache, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Value{}, err
    }

    return values[idx], nil
}

// First service to become ready
func WaitForAnyServiceReady(ctx context.Context, services []*Service) (*Service, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, r chan struct{}) {
            for !s.IsReady() {
                time.Sleep(100 * time.Millisecond)
            }
            close(r)
        }(svc, ready)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return services[idx], nil
}

// Race multiple download sources
func DownloadFromMirrors(ctx context.Context, mirrors []string) ([]byte, error) {
    signals := make([]<-chan struct{}, len(mirrors))
    downloads := make([][]byte, len(mirrors))

    for i, mirror := range mirrors {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            downloads[idx] = download(url)
        }(i, mirror, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return downloads[idx], nil
}

// First successful health check
func WaitForAnyHealthy(ctx context.Context, services []HealthChecker) (int, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(idx int, s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()

            for range ticker.C {
                if s.IsHealthy() {
                    close(h)
                    return
                }
            }
        }(i, svc, healthy)
    }

    return WaitAny(ctx, signals...)
}

// Fastest authentication method
func AuthenticateAny(ctx context.Context, credentials Creds, methods []AuthMethod) (Token, error) {
    signals := make([]<-chan struct{}, len(methods))
    tokens := make([]Token, len(methods))

    for i, method := range methods {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, m AuthMethod, d chan struct{}) {
            defer close(d)
            tokens[idx] = m.Authenticate(credentials)
        }(i, method, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Token{}, err
    }

    return tokens[idx], nil
}

// First available worker
func SubmitToFirstAvailableWorker(ctx context.Context, task Task, workers []*Worker) error {
    signals := make([]<-chan struct{}, len(workers))

    for i, worker := range workers {
        available := make(chan struct{})
        signals[i] = available

        go func(w *Worker, a chan struct{}) {
            for !w.IsAvailable() {
                time.Sleep(10 * time.Millisecond)
            }
            close(a)
        }(worker, available)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return err
    }

    return workers[idx].Submit(task)
}
\`\`\`

**Real-World Benefits:**
- **Latency Reduction:** Use fastest response
- **High Availability:** Failover to working replicas
- **Load Distribution:** Spread load across backends
- **Resilience:** Continue if some services fail

**Common Use Cases:**
- **Database Replicas:** Query multiple, use first
- **API Redundancy:** Call backups if primary slow
- **Cache Layers:** Try L1, L2, L3 caches in parallel
- **CDN Mirrors:** Download from multiple mirrors
- **Service Discovery:** Use first available instance
- **Health Checks:** First service to become healthy

**vs WaitAll:**
- **WaitAny:** Need ANY operation to complete (first wins)
- **WaitAll:** Need ALL operations to complete
- Use WaitAny for redundancy and speed
- Use WaitAll for completeness

**Performance Pattern:**
- Query 3 replicas in parallel
- Use response from fastest (usually closest)
- Ignore other 2 responses
- Typical speedup: 2-3x

Without WaitAny, implementing redundant queries and failover patterns requires complex racing logic prone to goroutine leaks.`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	idx, err := WaitAny(context.Background())
	if idx != -1 { t.Errorf("expected -1 for no signals, got %d", idx) }
	if err != nil { t.Errorf("expected nil error, got %v", err) }
}

func Test2(t *testing.T) {
	sig := make(chan struct{})
	close(sig)
	idx, err := WaitAny(context.Background(), sig)
	if idx != 0 { t.Errorf("expected 0 for single closed signal, got %d", idx) }
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test3(t *testing.T) {
	sig1, sig2, sig3 := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() { time.Sleep(50*time.Millisecond); close(sig2) }()
	idx, err := WaitAny(context.Background(), sig1, sig2, sig3)
	if idx != 1 { t.Errorf("expected 1 (sig2 first), got %d", idx) }
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test4(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	sig := make(chan struct{})
	idx, err := WaitAny(ctx, sig)
	if idx != -1 { t.Errorf("expected -1 for timeout, got %d", idx) }
	if !errors.Is(err, context.DeadlineExceeded) { t.Errorf("expected DeadlineExceeded, got %v", err) }
}

func Test5(t *testing.T) {
	sig := make(chan struct{})
	close(sig)
	idx, err := WaitAny(nil, sig)
	if idx != 0 { t.Errorf("expected 0 for nil context with closed signal, got %d", idx) }
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	sig := make(chan struct{})
	idx, err := WaitAny(ctx, sig)
	if idx != -1 { t.Errorf("expected -1 for canceled context, got %d", idx) }
	if !errors.Is(err, context.Canceled) { t.Errorf("expected Canceled, got %v", err) }
}

func Test7(t *testing.T) {
	sig1, sig2 := make(chan struct{}), make(chan struct{})
	close(sig1)
	idx, _ := WaitAny(context.Background(), sig1, sig2)
	if idx != 0 { t.Errorf("expected 0 (sig1), got %d", idx) }
}

func Test8(t *testing.T) {
	sig1, sig2 := make(chan struct{}), make(chan struct{})
	close(sig2)
	idx, _ := WaitAny(context.Background(), sig1, sig2)
	if idx != 1 { t.Errorf("expected 1 (sig2), got %d", idx) }
}

func Test9(t *testing.T) {
	signals := make([]<-chan struct{}, 10)
	for i := range signals {
		ch := make(chan struct{})
		signals[i] = ch
		if i == 5 {
			close(ch)
		}
	}
	idx, err := WaitAny(context.Background(), signals...)
	if idx != 5 { t.Errorf("expected 5, got %d", idx) }
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test10(t *testing.T) {
	sig := make(chan struct{})
	go func() { time.Sleep(10*time.Millisecond); close(sig) }()
	start := time.Now()
	_, _ = WaitAny(context.Background(), sig)
	if time.Since(start) > 50*time.Millisecond { t.Error("should return quickly after signal") }
}
`,
	order: 6,
	translations: {
		ru: {
			title: 'Получение первого результата из нескольких горутин',
			description: `Реализуйте **WaitAny**, который ждёт закрытия первого сигнального канала, возвращая его индекс.

**Требования:**
1. Создайте функцию \`WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error)\`
2. Обработайте nil context (используйте Background)
3. Обработайте пустые signals (верните -1, nil)
4. Запустите горутину для каждого сигнала для ожидания
5. Верните индекс первого полученного сигнала
6. Отмените ожидание других сигналов когда первый прибыл
7. Верните -1 и ошибку контекста если контекст отменён первым
8. Используйте буферизованный канал результата для предотвращения утечек горутин

**Пример:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig2) // Первый закрывается
}()
go func() {
    time.Sleep(300 * time.Millisecond)
    close(sig3)
}()

idx, err := WaitAny(context.Background(), sig1, sig2, sig3)
// idx = 1, err = nil (sig2 был первым)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

idx, err = WaitAny(ctx, sig1, sig2, sig3)
// idx = -1, err = context.DeadlineExceeded
\`\`\`

**Ограничения:**
- Должен возвращать индекс первого сигнала
- Должен отменять других ожидающих
- Должен обрабатывать отмену контекста
- Должен предотвращать утечки горутин`,
			hint1: `Создайте context.WithCancel чтобы отменить всех ожидающих когда прибудет первый сигнал. Используйте буферизованный канал для результата чтобы избежать блокировки.`,
			hint2: `Запустите горутину для каждого сигнала. Первый кто отправит в result канал отменяет контекст. Используйте select default чтобы избежать блокировки если result уже отправлен.`,
			whyItMatters: `WaitAny реализует паттерн first-responder, необходим для гонки множества операций и использования той которая завершится первой.

**Почему Wait Any критичен:**
- **Производительность:** Использование самого быстрого ответа
- **Избыточность:** Запрос множества источников, использование первого
- **Failover:** Попробовать множество backends, использовать первый успешный
- **Load Balancing:** Распределение нагрузки по сервисам

**Production паттерны:**
\`\`\`go
// Запрос множества реплик, использование первого ответа
func QueryReplicas(ctx context.Context, query string, replicas []Database) (Result, error) {
    signals := make([]<-chan struct{}, len(replicas))
    results := make([]Result, len(replicas))

    for i, db := range replicas {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, database Database, d chan struct{}) {
            defer close(d)
            results[idx] = database.Query(query)
        }(i, db, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Result{}, err
    }

    return results[idx], nil
}

// Вызов множества API, использование самого быстрого
func FetchFromMultipleSources(ctx context.Context, sources []string) (Data, error) {
    signals := make([]<-chan struct{}, len(sources))
    dataSlice := make([]Data, len(sources))

    for i, source := range sources {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            dataSlice[idx] = fetch(url)
        }(i, source, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Data{}, err
    }

    return dataSlice[idx], nil
}

// Попытка множества кеш серверов
func GetFromCache(ctx context.Context, key string, caches []*Cache) (Value, error) {
    signals := make([]<-chan struct{}, len(caches))
    values := make([]Value, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, c *Cache, d chan struct{}) {
            defer close(d)
            values[idx] = c.Get(key)
        }(i, cache, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Value{}, err
    }

    return values[idx], nil
}

// Первый готовый сервис
func WaitForAnyServiceReady(ctx context.Context, services []*Service) (*Service, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, r chan struct{}) {
            for !s.IsReady() {
                time.Sleep(100 * time.Millisecond)
            }
            close(r)
        }(svc, ready)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return services[idx], nil
}

// Гонка множества зеркал загрузки
func DownloadFromMirrors(ctx context.Context, mirrors []string) ([]byte, error) {
    signals := make([]<-chan struct{}, len(mirrors))
    downloads := make([][]byte, len(mirrors))

    for i, mirror := range mirrors {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            downloads[idx] = download(url)
        }(i, mirror, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return downloads[idx], nil
}

// Первая успешная проверка здоровья
func WaitForAnyHealthy(ctx context.Context, services []HealthChecker) (int, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(idx int, s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()

            for range ticker.C {
                if s.IsHealthy() {
                    close(h)
                    return
                }
            }
        }(i, svc, healthy)
    }

    return WaitAny(ctx, signals...)
}

// Самый быстрый метод аутентификации
func AuthenticateAny(ctx context.Context, credentials Creds, methods []AuthMethod) (Token, error) {
    signals := make([]<-chan struct{}, len(methods))
    tokens := make([]Token, len(methods))

    for i, method := range methods {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, m AuthMethod, d chan struct{}) {
            defer close(d)
            tokens[idx] = m.Authenticate(credentials)
        }(i, method, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Token{}, err
    }

    return tokens[idx], nil
}

// Первый доступный воркер
func SubmitToFirstAvailableWorker(ctx context.Context, task Task, workers []*Worker) error {
    signals := make([]<-chan struct{}, len(workers))

    for i, worker := range workers {
        available := make(chan struct{})
        signals[i] = available

        go func(w *Worker, a chan struct{}) {
            for !w.IsAvailable() {
                time.Sleep(10 * time.Millisecond)
            }
            close(a)
        }(worker, available)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return err
    }

    return workers[idx].Submit(task)
}
\`\`\`

**Реальные преимущества:**
- **Снижение латентности:** Использование самого быстрого ответа
- **Высокая доступность:** Failover на работающие реплики
- **Распределение нагрузки:** Распространение нагрузки по backends
- **Устойчивость:** Продолжение работы если некоторые сервисы недоступны

**Типичные сценарии использования:**
- **Реплики БД:** Запрос нескольких, использование первой
- **API избыточность:** Вызов резервных если основной медленный
- **Уровни кеша:** Параллельная попытка L1, L2, L3 кешей
- **CDN зеркала:** Загрузка с множества зеркал
- **Service Discovery:** Использование первого доступного экземпляра
- **Health Checks:** Первый сервис который станет здоровым

**WaitAny vs WaitAll:**
- **WaitAny:** Нужна ЛЮБАЯ операция для завершения (первая побеждает)
- **WaitAll:** Нужны ВСЕ операции для завершения
- Используйте WaitAny для избыточности и скорости
- Используйте WaitAll для полноты данных

**Паттерн производительности:**
- Запросить 3 реплики параллельно
- Использовать ответ от самой быстрой (обычно ближайшей)
- Игнорировать другие 2 ответа
- Типичное ускорение: 2-3x в латентности

Без WaitAny реализация избыточных запросов и failover паттернов требует сложной логики гонок подверженной утечкам горутин. Паттерн критичен для построения отказоустойчивых высокопроизводительных распределённых систем.`,
			solutionCode: `package concurrency

import (
	"context"
)

func WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error) {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	if len(signals) == 0 {                                      // Нет сигналов для ожидания
		return -1, nil                                      // Возвращаем -1 для отсутствия сигналов
	}
	ctxAny, cancel := context.WithCancel(ctx)                   // Создаём отменяемый контекст
	result := make(chan int, 1)                                 // Буферизованный канал для результата
	for idx, ch := range signals {                              // Итерируем по всем сигналам
		idx := idx                                          // Захватываем переменную цикла
		go func(ch <-chan struct{}) {                       // Ждём каждый сигнал в горутине
			select {
			case <-ctxAny.Done():                       // Контекст отменён (другой победил)
			case <-ch:                                  // Этот сигнал получен
				select {
				case result <- idx:                 // Пробуем отправить индекс
					cancel()                    // Отменяем других ожидающих
				default:                            // Результат уже отправлен
				}
			}
		}(ch)
	}
	select {
	case <-ctxAny.Done():                                       // Контекст отменён
		if err := ctx.Err(); err != nil {                   // Ошибка родительского контекста
			return -1, err                              // Возвращаем ошибку родителя
		}
		return -1, context.Canceled                         // Дочерний отменён (проиграл гонку)
	case idx := <-result:                                       // Индекс первого сигнала
		return idx, nil                                     // Возвращаем победивший индекс
	}
}`
		},
		uz: {
			title: 'Birinchi goroutinadan natija olish',
			description: `Birinchi signal kanalining yopilishini kutadigan va uning indeksini qaytaradigan **WaitAny** ni amalga oshiring.

**Talablar:**
1. \`WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error)\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. Bo'sh signallarni ishlang (-1, nil qaytaring)
4. Har bir signal uchun kutish uchun goroutina ishga tushiring
5. Birinchi olingan signalning indeksini qaytaring
6. Birinchi signal kelganda boshqa signallarni kutishni bekor qiling
7. Agar kontekst birinchi bo'lib bekor qilinsa -1 va kontekst xatosini qaytaring
8. Goroutina oqishini oldini olish uchun buferli natija kanalidan foydalaning

**Misol:**
\`\`\`go
sig1 := make(chan struct{})
sig2 := make(chan struct{})
sig3 := make(chan struct{})

go func() {
    time.Sleep(200 * time.Millisecond)
    close(sig1)
}()
go func() {
    time.Sleep(100 * time.Millisecond)
    close(sig2) // Birinchi yopiladi
}()
go func() {
    time.Sleep(300 * time.Millisecond)
    close(sig3)
}()

idx, err := WaitAny(context.Background(), sig1, sig2, sig3)
// idx = 1, err = nil (sig2 birinchi edi)

ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
defer cancel()

idx, err = WaitAny(ctx, sig1, sig2, sig3)
// idx = -1, err = context.DeadlineExceeded
\`\`\`

**Cheklovlar:**
- Birinchi signalning indeksini qaytarishi kerak
- Boshqa kutuvchilarni bekor qilishi kerak
- Kontekst bekor qilinishini ishlashi kerak
- Goroutina oqishini oldini olishi kerak`,
			hint1: `Birinchi signal kelganda barcha kutuvchilarni bekor qilish uchun context.WithCancel yarating. Bloklashni oldini olish uchun natija uchun buferli kanaldan foydalaning.`,
			hint2: `Har bir signal uchun goroutina ishga tushiring. result kanaliga birinchi yuborgan kontekstni bekor qiladi. Agar natija allaqachon yuborilgan bo'lsa bloklashni oldini olish uchun select default dan foydalaning.`,
			whyItMatters: `WaitAny first-responder patternini amalga oshiradi, ko'p operatsiyalarni poyga qilish va birinchi tugaganini ishlatish uchun zarur.

**Nima uchun Wait Any muhim:**
- **Samaradorlik:** Eng tez javobdan foydalanish
- **Ortiqcha:** Ko'p manbalarni so'rash, birinchisidan foydalanish
- **Failover:** Ko'p backendlarni sinab ko'rish, birinchi muvaffaqiyatdan foydalanish
- **Yuk balansi:** Xizmatlar bo'ylab yukni taqsimlash

**Production patternlar:**
\`\`\`go
// Ko'p replikalarni so'rash, birinchi javobdan foydalanish
func QueryReplicas(ctx context.Context, query string, replicas []Database) (Result, error) {
    signals := make([]<-chan struct{}, len(replicas))
    results := make([]Result, len(replicas))

    for i, db := range replicas {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, database Database, d chan struct{}) {
            defer close(d)
            results[idx] = database.Query(query)
        }(i, db, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Result{}, err
    }

    return results[idx], nil
}

// Ko'p APIlarni chaqirish, eng tezidan foydalanish
func FetchFromMultipleSources(ctx context.Context, sources []string) (Data, error) {
    signals := make([]<-chan struct{}, len(sources))
    dataSlice := make([]Data, len(sources))

    for i, source := range sources {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            dataSlice[idx] = fetch(url)
        }(i, source, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Data{}, err
    }

    return dataSlice[idx], nil
}

// Ko'p kesh serverlarni sinash
func GetFromCache(ctx context.Context, key string, caches []*Cache) (Value, error) {
    signals := make([]<-chan struct{}, len(caches))
    values := make([]Value, len(caches))

    for i, cache := range caches {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, c *Cache, d chan struct{}) {
            defer close(d)
            values[idx] = c.Get(key)
        }(i, cache, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Value{}, err
    }

    return values[idx], nil
}

// Birinchi tayyor xizmat
func WaitForAnyServiceReady(ctx context.Context, services []*Service) (*Service, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        ready := make(chan struct{})
        signals[i] = ready

        go func(s *Service, r chan struct{}) {
            for !s.IsReady() {
                time.Sleep(100 * time.Millisecond)
            }
            close(r)
        }(svc, ready)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return services[idx], nil
}

// Ko'p yuklab olish oynalarini poyga qilish
func DownloadFromMirrors(ctx context.Context, mirrors []string) ([]byte, error) {
    signals := make([]<-chan struct{}, len(mirrors))
    downloads := make([][]byte, len(mirrors))

    for i, mirror := range mirrors {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, url string, d chan struct{}) {
            defer close(d)
            downloads[idx] = download(url)
        }(i, mirror, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return nil, err
    }

    return downloads[idx], nil
}

// Birinchi muvaffaqiyatli sog'liqni tekshirish
func WaitForAnyHealthy(ctx context.Context, services []HealthChecker) (int, error) {
    signals := make([]<-chan struct{}, len(services))

    for i, svc := range services {
        healthy := make(chan struct{})
        signals[i] = healthy

        go func(idx int, s HealthChecker, h chan struct{}) {
            ticker := time.NewTicker(time.Second)
            defer ticker.Stop()

            for range ticker.C {
                if s.IsHealthy() {
                    close(h)
                    return
                }
            }
        }(i, svc, healthy)
    }

    return WaitAny(ctx, signals...)
}

// Eng tez autentifikatsiya usuli
func AuthenticateAny(ctx context.Context, credentials Creds, methods []AuthMethod) (Token, error) {
    signals := make([]<-chan struct{}, len(methods))
    tokens := make([]Token, len(methods))

    for i, method := range methods {
        done := make(chan struct{})
        signals[i] = done

        go func(idx int, m AuthMethod, d chan struct{}) {
            defer close(d)
            tokens[idx] = m.Authenticate(credentials)
        }(i, method, done)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return Token{}, err
    }

    return tokens[idx], nil
}

// Birinchi mavjud worker
func SubmitToFirstAvailableWorker(ctx context.Context, task Task, workers []*Worker) error {
    signals := make([]<-chan struct{}, len(workers))

    for i, worker := range workers {
        available := make(chan struct{})
        signals[i] = available

        go func(w *Worker, a chan struct{}) {
            for !w.IsAvailable() {
                time.Sleep(10 * time.Millisecond)
            }
            close(a)
        }(worker, available)
    }

    idx, err := WaitAny(ctx, signals...)
    if err != nil {
        return err
    }

    return workers[idx].Submit(task)
}
\`\`\`

**Haqiqiy foydalari:**
- **Kechikishni kamaytirish:** Eng tez javobdan foydalanish
- **Yuqori mavjudlik:** Ishlaydigan replikalarga failover
- **Yuk taqsimlash:** Backendlar bo'ylab yukni tarqatish
- **Barqarorlik:** Ba'zi xizmatlar ishlamasa ham davom etish

**Umumiy foydalanish stsenariylari:**
- **DB replikalari:** Ko'pini so'rash, birinchisidan foydalanish
- **API ortiqcha:** Asosiy sekin bo'lsa zaxira chaqirish
- **Kesh darajalari:** L1, L2, L3 keshlarni parallel sinash
- **CDN oynalari:** Ko'p oynalardan yuklab olish
- **Service Discovery:** Birinchi mavjud instansiyadan foydalanish
- **Health Checks:** Sog'lom bo'lgan birinchi xizmat

**WaitAny vs WaitAll:**
- **WaitAny:** ISTALGAN operatsiyaning tugashini kutish (birinchisi yutadi)
- **WaitAll:** BARCHA operatsiyalarning tugashini kutish
- Ortiqcha va tezlik uchun WaitAny dan foydalaning
- Ma'lumotlarning to'liqligi uchun WaitAll dan foydalaning

**Samaradorlik patterni:**
- 3 ta replikani parallel so'rash
- Eng tez (odatda eng yaqin) javobdan foydalanish
- Qolgan 2 ta javobni e'tiborsiz qoldirish
- Tipik tezlashtirish: kechikishda 2-3x

WaitAny bo'lmasa, ortiqcha so'rovlar va failover patternlarini amalga oshirish goroutina oqishiga moyil murakkab poyga mantiqini talab qiladi. Pattern ishdan chiqishga bardoshli yuqori samarali taqsimlangan tizimlarni qurish uchun juda muhim.`,
			solutionCode: `package concurrency

import (
	"context"
)

func WaitAny(ctx context.Context, signals ...<-chan struct{}) (int, error) {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	if len(signals) == 0 {                                      // Kutish uchun signallar yo'q
		return -1, nil                                      // Signallar yo'qligi uchun -1 qaytaramiz
	}
	ctxAny, cancel := context.WithCancel(ctx)                   // Bekor qilinadigan kontekst yaratamiz
	result := make(chan int, 1)                                 // Natija uchun buferli kanal
	for idx, ch := range signals {                              // Barcha signallar bo'yicha iteratsiya
		idx := idx                                          // Sikl o'zgaruvchisini ushlash
		go func(ch <-chan struct{}) {                       // Har bir signalni goroutinada kutamiz
			select {
			case <-ctxAny.Done():                       // Kontekst bekor qilindi (boshqasi yutdi)
			case <-ch:                                  // Bu signal olindi
				select {
				case result <- idx:                 // Indeksni yuborishga harakat qilamiz
					cancel()                    // Boshqa kutuvchilarni bekor qilamiz
				default:                            // Natija allaqachon yuborilgan
				}
			}
		}(ch)
	}
	select {
	case <-ctxAny.Done():                                       // Kontekst bekor qilindi
		if err := ctx.Err(); err != nil {                   // Ota kontekst xatosi
			return -1, err                              // Ota xatosini qaytaramiz
		}
		return -1, context.Canceled                         // Bola bekor qilindi (poygani yutqazdi)
	case idx := <-result:                                       // Birinchi signal indeksi
		return idx, nil                                     // G'olib indeksini qaytaramiz
	}
}`
		}
	}
};

export default task;
