import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-notify-cancel',
	title: 'Notify Cancel',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'context', 'notification'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **NotifyCancel** that returns a channel which closes when the context is canceled.

**Requirements:**
1. Create function \`NotifyCancel(ctx context.Context) <-chan struct{}\`
2. Return a receive-only channel
3. Handle nil context (return immediately closed channel)
4. Create goroutine that waits for context cancellation
5. Close the channel when context is done
6. Return the channel immediately (non-blocking)

**Example:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notify := NotifyCancel(ctx)

// Channel is open
select {
case <-notify:
    fmt.Println("Not canceled yet")
default:
    fmt.Println("Still running")
}

cancel() // Cancel context

<-notify // Will receive signal
fmt.Println("Context canceled!")

// Nil context
notify = NotifyCancel(nil)
<-notify // Immediately receives (channel closed)
\`\`\`

**Constraints:**
- Must return receive-only channel
- Must close channel when context is done
- Must handle nil context
- Must not block on return`,
	initialCode: `package concurrency

import (
	"context"
)

// TODO: Implement NotifyCancel
func NotifyCancel(ctx context.Context) <-chan struct{} {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                   // Create notification channel
	go func() {                                                 // Run in background goroutine
		defer close(ch)                                     // Always close channel when done
		if ctx == nil {                                     // Handle nil context
			return                                      // Return immediately (closes channel)
		}
		<-ctx.Done()                                        // Wait for context cancellation
	}()
	return ch                                                   // Return channel immediately
}`,
			hint1: `Create a channel and return it immediately. In a separate goroutine, wait for ctx.Done() and then close the channel.`,
			hint2: `Use defer close(ch) to ensure the channel is always closed. For nil context, return immediately from the goroutine.`,
			whyItMatters: `NotifyCancel converts context cancellation into a channel signal, enabling integration with select statements and existing channel-based code.

**Why Notification Channels:**
- **Select Integration:** Use cancellation in select with other channels
- **Multiple Listeners:** Multiple goroutines can wait on same channel
- **Signal Propagation:** Broadcast cancellation to many receivers
- **Legacy Code:** Integrate context with channel-based APIs

**Production Pattern:**
\`\`\`go
// Combine multiple signals
func WaitForAnySignal(ctx context.Context, signals ...<-chan struct{}) {
    cancel := NotifyCancel(ctx)

    cases := make([]reflect.SelectCase, len(signals)+1)
    cases[0] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(cancel)}

    for i, ch := range signals {
        cases[i+1] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ch)}
    }

    chosen, _, _ := reflect.Select(cases)
    if chosen == 0 {
        log.Println("Context canceled")
    } else {
        log.Printf("Signal %d received", chosen-1)
    }
}

// Worker pool with cancellation
type WorkerPool struct {
    workers int
    jobs    chan Job
}

func (p *WorkerPool) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)

    for i := 0; i < p.workers; i++ {
        go func(id int) {
            for {
                select {
                case <-cancel:
                    log.Printf("Worker %d stopping", id)
                    return
                case job := <-p.jobs:
                    processJob(job)
                }
            }
        }(i)
    }
}

// Rate limiter with cancellation
func RateLimiter(ctx context.Context, rate time.Duration) <-chan struct{} {
    cancel := NotifyCancel(ctx)
    tick := time.NewTicker(rate)
    output := make(chan struct{})

    go func() {
        defer close(output)
        defer tick.Stop()

        for {
            select {
            case <-cancel:
                return
            case <-tick.C:
                select {
                case output <- struct{}{}:
                case <-cancel:
                    return
                }
            }
        }
    }()

    return output
}

// Fan-out pattern with cancellation
func FanOut(ctx context.Context, input <-chan int, workers int) []<-chan int {
    cancel := NotifyCancel(ctx)
    outputs := make([]<-chan int, workers)

    for i := 0; i < workers; i++ {
        ch := make(chan int)
        outputs[i] = ch

        go func(out chan<- int) {
            defer close(out)
            for {
                select {
                case <-cancel:
                    return
                case val, ok := <-input:
                    if !ok {
                        return
                    }
                    out <- process(val)
                }
            }
        }(ch)
    }

    return outputs
}

// Timeout with multiple operations
func MultiOperationTimeout(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    done1 := make(chan struct{})
    done2 := make(chan struct{})
    done3 := make(chan struct{})

    go operation1(done1)
    go operation2(done2)
    go operation3(done3)

    for i := 0; i < 3; i++ {
        select {
        case <-cancel:
            return ctx.Err()
        case <-done1:
            done1 = nil
        case <-done2:
            done2 = nil
        case <-done3:
            done3 = nil
        }
    }

    return nil
}

// Graceful shutdown with phases
func GracefulShutdown(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    // Phase 1: Stop accepting new requests
    select {
    case <-cancel:
        return ctx.Err()
    case <-stopAccepting():
    }

    // Phase 2: Finish current requests
    select {
    case <-cancel:
        return ctx.Err()
    case <-finishCurrent():
    }

    // Phase 3: Cleanup
    select {
    case <-cancel:
        return ctx.Err()
    case <-cleanup():
    }

    return nil
}

// Event broadcaster
type Broadcaster struct {
    listeners []chan<- Event
}

func (b *Broadcaster) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)
    events := getEventStream()

    for {
        select {
        case <-cancel:
            b.closeAll()
            return
        case event := <-events:
            b.broadcast(event)
        }
    }
}
\`\`\`

**Real-World Benefits:**
- **Flexibility:** Use cancellation with any channel-based code
- **Broadcasting:** Signal many goroutines at once
- **Select Friendly:** Works seamlessly with select statements
- **API Bridge:** Connect context to channel-based APIs

**Common Use Cases:**
- **Worker Pools:** Stop all workers simultaneously
- **Pipeline Stages:** Cancel entire pipeline
- **Event Processing:** Stop event handlers
- **Rate Limiting:** Cancel rate-limited operations
- **Fan-Out/Fan-In:** Coordinate parallel workers

**Channel vs Context:**
- **Context:** Hierarchical, request-scoped
- **Channel:** Flexible, can fan-out to multiple listeners
- NotifyCancel bridges both worlds

Without NotifyCancel, integrating context cancellation with channel-based patterns requires verbose, error-prone code.`,
	testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	notify := NotifyCancel(ctx)
	cancel()
	select {
	case <-notify:
	case <-time.After(100*time.Millisecond):
		t.Error("channel should close after cancel")
	}
}

func Test2(t *testing.T) {
	notify := NotifyCancel(nil)
	select {
	case <-notify:
	case <-time.After(100*time.Millisecond):
		t.Error("nil context should return immediately closed channel")
	}
}

func Test3(t *testing.T) {
	ctx := context.Background()
	notify := NotifyCancel(ctx)
	select {
	case <-notify:
		t.Error("channel should not close for Background context")
	case <-time.After(50*time.Millisecond):
	}
}

func Test4(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	_ = NotifyCancel(ctx)
	cancel()
}

func Test5(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	notify := NotifyCancel(ctx)
	select {
	case <-notify:
	case <-time.After(100*time.Millisecond):
		t.Error("channel should close after timeout")
	}
}

func Test6(t *testing.T) {
	ctx := context.Background()
	ch := NotifyCancel(ctx)
	if ch == nil { t.Error("returned channel should not be nil") }
}

func Test7(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	notify1 := NotifyCancel(ctx)
	notify2 := NotifyCancel(ctx)
	cancel()
	<-notify1
	<-notify2
}

func Test8(t *testing.T) {
	parent, cancelParent := context.WithCancel(context.Background())
	child, _ := context.WithCancel(parent)
	notify := NotifyCancel(child)
	cancelParent()
	select {
	case <-notify:
	case <-time.After(100*time.Millisecond):
		t.Error("child context notify should close when parent canceled")
	}
}

func Test9(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	notify := NotifyCancel(ctx)
	time.Sleep(10*time.Millisecond)
	cancel()
	<-notify
}

func Test10(t *testing.T) {
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		_ = NotifyCancel(ctx)
	}
}
`,
	order: 3,
	translations: {
		ru: {
			title: 'Уведомление горутин об отмене операции',
			description: `Реализуйте **NotifyCancel**, который возвращает канал, закрывающийся при отмене контекста.

**Требования:**
1. Создайте функцию \`NotifyCancel(ctx context.Context) <-chan struct{}\`
2. Верните receive-only канал
3. Обработайте nil context (верните сразу закрытый канал)
4. Создайте горутину которая ждёт отмены контекста
5. Закройте канал когда контекст отменён
6. Верните канал немедленно (без блокировки)

**Пример:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notify := NotifyCancel(ctx)

// Канал открыт
select {
case <-notify:
    fmt.Println("Ещё не отменён")
default:
    fmt.Println("Всё ещё работает")
}

cancel() // Отмена контекста

<-notify // Получит сигнал
fmt.Println("Контекст отменён!")

// Nil context
notify = NotifyCancel(nil)
<-notify // Сразу получает (канал закрыт)
\`\`\`

**Ограничения:**
- Должен возвращать receive-only канал
- Должен закрывать канал при отмене контекста
- Должен обрабатывать nil context
- Не должен блокироваться при возврате`,
			hint1: `Создайте канал и верните его сразу. В отдельной горутине ждите ctx.Done() и затем закройте канал.`,
			hint2: `Используйте defer close(ch) чтобы гарантировать закрытие канала. Для nil context сразу вернитесь из горутины.`,
			whyItMatters: `NotifyCancel конвертирует отмену контекста в сигнал канала, позволяя интеграцию с select statements и существующим channel-based кодом.

**Почему Notification Channels:**
- **Select интеграция:** Использование отмены в select с другими каналами
- **Множество слушателей:** Множество горутин могут ждать один канал
- **Распространение сигнала:** Broadcast отмены множеству получателей
- **Legacy код:** Интеграция контекста с channel-based API

**Продакшен паттерн:**
\`\`\`go
// Объединение множества сигналов
func WaitForAnySignal(ctx context.Context, signals ...<-chan struct{}) {
    cancel := NotifyCancel(ctx)

    cases := make([]reflect.SelectCase, len(signals)+1)
    cases[0] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(cancel)}

    for i, ch := range signals {
        cases[i+1] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ch)}
    }

    chosen, _, _ := reflect.Select(cases)
    if chosen == 0 {
        log.Println("Context canceled")
    } else {
        log.Printf("Signal %d received", chosen-1)
    }
}

// Worker pool с отменой
type WorkerPool struct {
    workers int
    jobs    chan Job
}

func (p *WorkerPool) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)

    for i := 0; i < p.workers; i++ {
        go func(id int) {
            for {
                select {
                case <-cancel:
                    log.Printf("Worker %d stopping", id)
                    return
                case job := <-p.jobs:
                    processJob(job)
                }
            }
        }(i)
    }
}

// Rate limiter с отменой
func RateLimiter(ctx context.Context, rate time.Duration) <-chan struct{} {
    cancel := NotifyCancel(ctx)
    tick := time.NewTicker(rate)
    output := make(chan struct{})

    go func() {
        defer close(output)
        defer tick.Stop()

        for {
            select {
            case <-cancel:
                return
            case <-tick.C:
                select {
                case output <- struct{}{}:
                case <-cancel:
                    return
                }
            }
        }
    }()

    return output
}

// Fan-out паттерн с отменой
func FanOut(ctx context.Context, input <-chan int, workers int) []<-chan int {
    cancel := NotifyCancel(ctx)
    outputs := make([]<-chan int, workers)

    for i := 0; i < workers; i++ {
        ch := make(chan int)
        outputs[i] = ch

        go func(out chan<- int) {
            defer close(out)
            for {
                select {
                case <-cancel:
                    return
                case val, ok := <-input:
                    if !ok {
                        return
                    }
                    out <- process(val)
                }
            }
        }(ch)
    }

    return outputs
}

// Timeout с множеством операций
func MultiOperationTimeout(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    done1 := make(chan struct{})
    done2 := make(chan struct{})
    done3 := make(chan struct{})

    go operation1(done1)
    go operation2(done2)
    go operation3(done3)

    for i := 0; i < 3; i++ {
        select {
        case <-cancel:
            return ctx.Err()
        case <-done1:
            done1 = nil
        case <-done2:
            done2 = nil
        case <-done3:
            done3 = nil
        }
    }

    return nil
}

// Graceful shutdown с фазами
func GracefulShutdown(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    // Фаза 1: Остановка приёма новых запросов
    select {
    case <-cancel:
        return ctx.Err()
    case <-stopAccepting():
    }

    // Фаза 2: Завершение текущих запросов
    select {
    case <-cancel:
        return ctx.Err()
    case <-finishCurrent():
    }

    // Фаза 3: Очистка
    select {
    case <-cancel:
        return ctx.Err()
    case <-cleanup():
    }

    return nil
}

// Event broadcaster
type Broadcaster struct {
    listeners []chan<- Event
}

func (b *Broadcaster) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)
    events := getEventStream()

    for {
        select {
        case <-cancel:
            b.closeAll()
            return
        case event := <-events:
            b.broadcast(event)
        }
    }
}
\`\`\`

**Практические преимущества:**
- **Гибкость:** Использование отмены с любым channel-based кодом
- **Broadcasting:** Сигнал множеству горутин одновременно
- **Select дружественность:** Работает бесшовно с select statements
- **API мост:** Соединение контекста с channel-based API

**Обычные случаи использования:**
- **Worker Pools:** Остановка всех workers одновременно
- **Pipeline Stages:** Отмена всего pipeline
- **Event Processing:** Остановка event handlers
- **Rate Limiting:** Отмена rate-limited операций
- **Fan-Out/Fan-In:** Координация параллельных workers

**Channel vs Context:**
- **Context:** Иерархический, request-scoped
- **Channel:** Гибкий, может fan-out к множеству слушателей
- NotifyCancel объединяет оба мира

Без NotifyCancel интеграция отмены контекста с channel-based паттернами требует многословного, подверженного ошибкам кода.`,
			solutionCode: `package concurrency

import (
	"context"
)

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                   // Создаём канал уведомлений
	go func() {                                                 // Запускаем в фоновой горутине
		defer close(ch)                                     // Всегда закрываем канал при завершении
		if ctx == nil {                                     // Обработка nil контекста
			return                                      // Возвращаемся сразу (закрываем канал)
		}
		<-ctx.Done()                                        // Ждём отмены контекста
	}()
	return ch                                                   // Возвращаем канал сразу
}`
		},
		uz: {
			title: 'Goroutinalarga bekor qilish haqida xabar berish',
			description: `Kontekst bekor qilinganda yopiladigan kanalni qaytaradigan **NotifyCancel** ni amalga oshiring.

**Talablar:**
1. \`NotifyCancel(ctx context.Context) <-chan struct{}\` funksiyasini yarating
2. Faqat qabul qiladigan kanalni qaytaring
3. nil kontekstni ishlang (darhol yopilgan kanalni qaytaring)
4. Kontekst bekor qilinishini kutadigan goroutina yarating
5. Kontekst bekor qilinganda kanalni yoping
6. Kanalni darhol qaytaring (bloklanmasdan)

**Misol:**
\`\`\`go
ctx, cancel := context.WithCancel(context.Background())
notify := NotifyCancel(ctx)

// Kanal ochiq
select {
case <-notify:
    fmt.Println("Hali bekor qilinmagan")
default:
    fmt.Println("Hali ishlayapti")
}

cancel() // Kontekstni bekor qilish

<-notify // Signal oladi
fmt.Println("Kontekst bekor qilindi!")

// Nil context
notify = NotifyCancel(nil)
<-notify // Darhol oladi (kanal yopilgan)
\`\`\`

**Cheklovlar:**
- Faqat qabul qiladigan kanalni qaytarishi kerak
- Kontekst bekor qilinganda kanalni yopishi kerak
- nil kontekstni ishlashi kerak
- Qaytishda bloklanmasligi kerak`,
			hint1: `Kanal yarating va uni darhol qaytaring. Alohida goroutinada ctx.Done() ni kuting va keyin kanalni yoping.`,
			hint2: `Kanal har doim yopilishini ta'minlash uchun defer close(ch) dan foydalaning. nil kontekst uchun goroutinadan darhol qaytaring.`,
			whyItMatters: `NotifyCancel kontekst bekor qilinishini kanal signaliga aylantiradi, select statementlar va mavjud kanal asosidagi kod bilan integratsiyani ta'minlaydi.

**Nima uchun Notification Channels:**
- **Select integratsiyasi:** Boshqa kanallar bilan select da bekor qilishdan foydalanish
- **Ko'p tinglovchilar:** Ko'p goroutinalar bir kanalni kutishi mumkin
- **Signal tarqatish:** Ko'p qabul qiluvchilarga bekor qilishni broadcast qilish
- **Legacy kod:** Kontekstni kanal asosidagi API bilan integratsiyalash

**Ishlab chiqarish patterni:**
\`\`\`go
// Ko'p signallarni birlashtirish
func WaitForAnySignal(ctx context.Context, signals ...<-chan struct{}) {
    cancel := NotifyCancel(ctx)

    cases := make([]reflect.SelectCase, len(signals)+1)
    cases[0] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(cancel)}

    for i, ch := range signals {
        cases[i+1] = reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ch)}
    }

    chosen, _, _ := reflect.Select(cases)
    if chosen == 0 {
        log.Println("Context canceled")
    } else {
        log.Printf("Signal %d received", chosen-1)
    }
}

// Bekor qilish bilan Worker pool
type WorkerPool struct {
    workers int
    jobs    chan Job
}

func (p *WorkerPool) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)

    for i := 0; i < p.workers; i++ {
        go func(id int) {
            for {
                select {
                case <-cancel:
                    log.Printf("Worker %d stopping", id)
                    return
                case job := <-p.jobs:
                    processJob(job)
                }
            }
        }(i)
    }
}

// Bekor qilish bilan Rate limiter
func RateLimiter(ctx context.Context, rate time.Duration) <-chan struct{} {
    cancel := NotifyCancel(ctx)
    tick := time.NewTicker(rate)
    output := make(chan struct{})

    go func() {
        defer close(output)
        defer tick.Stop()

        for {
            select {
            case <-cancel:
                return
            case <-tick.C:
                select {
                case output <- struct{}{}:
                case <-cancel:
                    return
                }
            }
        }
    }()

    return output
}

// Bekor qilish bilan Fan-out patterni
func FanOut(ctx context.Context, input <-chan int, workers int) []<-chan int {
    cancel := NotifyCancel(ctx)
    outputs := make([]<-chan int, workers)

    for i := 0; i < workers; i++ {
        ch := make(chan int)
        outputs[i] = ch

        go func(out chan<- int) {
            defer close(out)
            for {
                select {
                case <-cancel:
                    return
                case val, ok := <-input:
                    if !ok {
                        return
                    }
                    out <- process(val)
                }
            }
        }(ch)
    }

    return outputs
}

// Ko'p operatsiyalar bilan Timeout
func MultiOperationTimeout(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    done1 := make(chan struct{})
    done2 := make(chan struct{})
    done3 := make(chan struct{})

    go operation1(done1)
    go operation2(done2)
    go operation3(done3)

    for i := 0; i < 3; i++ {
        select {
        case <-cancel:
            return ctx.Err()
        case <-done1:
            done1 = nil
        case <-done2:
            done2 = nil
        case <-done3:
            done3 = nil
        }
    }

    return nil
}

// Fazalar bilan Graceful shutdown
func GracefulShutdown(ctx context.Context) error {
    cancel := NotifyCancel(ctx)

    // Faza 1: Yangi so'rovlarni qabul qilishni to'xtatish
    select {
    case <-cancel:
        return ctx.Err()
    case <-stopAccepting():
    }

    // Faza 2: Joriy so'rovlarni tugatish
    select {
    case <-cancel:
        return ctx.Err()
    case <-finishCurrent():
    }

    // Faza 3: Tozalash
    select {
    case <-cancel:
        return ctx.Err()
    case <-cleanup():
    }

    return nil
}

// Event broadcaster
type Broadcaster struct {
    listeners []chan<- Event
}

func (b *Broadcaster) Run(ctx context.Context) {
    cancel := NotifyCancel(ctx)
    events := getEventStream()

    for {
        select {
        case <-cancel:
            b.closeAll()
            return
        case event := <-events:
            b.broadcast(event)
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- **Moslashuvchanlik:** Bekor qilishni har qanday kanal asosidagi kod bilan ishlatish
- **Broadcasting:** Ko'p goroutinalarga bir vaqtda signal berish
- **Select mos:** Select statementlar bilan beg'ubor ishlaydi
- **API ko'prigi:** Kontekstni kanal asosidagi API bilan bog'lash

**Oddiy foydalanish holatlari:**
- **Worker Pools:** Barcha workerlarni bir vaqtda to'xtatish
- **Pipeline Stages:** Butun pipelineni bekor qilish
- **Event Processing:** Event handlerlarni to'xtatish
- **Rate Limiting:** Rate-limited operatsiyalarni bekor qilish
- **Fan-Out/Fan-In:** Parallel workerlarni muvofiqlashtirish

**Channel vs Context:**
- **Context:** Ierarxik, request-scoped
- **Channel:** Moslashuvchan, ko'p tinglovchilarga fan-out qilishi mumkin
- NotifyCancel ikkala dunyoni birlashtiradi

NotifyCancel bo'lmasa, kontekst bekor qilinishini kanal asosidagi patternlar bilan integratsiyalash ko'p so'zli va xatolarga moyil kodni talab qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
)

func NotifyCancel(ctx context.Context) <-chan struct{} {
	ch := make(chan struct{})                                   // Bildirishnoma kanali yaratamiz
	go func() {                                                 // Fonda goroutinada ishga tushiramiz
		defer close(ch)                                     // Tugaganda har doim kanalni yopamiz
		if ctx == nil {                                     // nil kontekstni ishlash
			return                                      // Darhol qaytamiz (kanalni yopamiz)
		}
		<-ctx.Done()                                        // Kontekst bekor qilinishini kutamiz
	}()
	return ch                                                   // Kanalni darhol qaytaramiz
}`
		}
	}
};

export default task;
