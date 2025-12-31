import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-goroutines-context-watcher',
	title: 'Context-Aware Channel Watcher',
	difficulty: 'medium',	tags: ['go', 'goroutines', 'context', 'lifecycle'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a goroutine that watches an input channel and forwards values until context is canceled.

**Requirements:**
1. **Watcher**: Create goroutine that proxies channel with context awareness
2. **Context Handling**: Stop forwarding when ctx.Done() is signaled
3. **Clean Shutdown**: Close output channel when done
4. **No Leaks**: Ensure goroutine terminates properly

**Implementation Pattern:**
\`\`\`go
func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
    out := make(chan T)

    go func() {
        defer close(out)
        for {
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
                }
            }
        }
    }()

    return out
}
\`\`\`

**Example Usage:**
\`\`\`go
// Timeout for long-running operation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

dataStream := fetchDataStream()
watched := Watcher(ctx, dataStream)

for data := range watched {
    process(data)
    // Automatically stops after 5 seconds
}
\`\`\`

**Constraints:**
- Must create output channel
- Must handle nil context (use Background)
- Must close output channel on return
- Must check ctx.Done() before sending`,
	initialCode: `package goroutinesx

import (
	"context"
)

// TODO: Implement Watcher
// Create output channel
// Launch goroutine to forward values
// Stop on ctx.Done() or input close
// Close output when done
func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
	// TODO: Implement
}`,
	solutionCode: `package goroutinesx

import (
	"context"
)

func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan T, len(in))	// create output channel
	if in == nil {
		close(out)	// close immediately for nil input
		return out
	}

	go func() {
		defer close(out)	// ensure output closed on return
		for {
			select {
			case <-ctx.Done():	// context canceled
				return
			case v, ok := <-in:
				if !ok {	// input channel closed
					return
				}
				select {
				case <-ctx.Done():	// check again before sending
					return
				case out <- v:	// forward value to output
				}
			}
		}
	}()
	return out
}`,
	testCode: `package goroutinesx

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	// Test forwarding values
	ctx := context.Background()
	in := make(chan int)
	out := Watcher(ctx, in)

	go func() {
		in <- 1
		in <- 2
		in <- 3
		close(in)
	}()

	results := []int{}
	for v := range out {
		results = append(results, v)
	}

	if len(results) != 3 {
		t.Errorf("expected 3 values, got %d", len(results))
	}
}

func Test2(t *testing.T) {
	// Test context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan int)
	out := Watcher(ctx, in)

	go func() {
		in <- 1
		time.Sleep(10 * time.Millisecond)
		cancel()
		time.Sleep(10 * time.Millisecond)
		in <- 2 // should not be received
	}()

	result := <-out
	if result != 1 {
		t.Errorf("expected 1, got %d", result)
	}

	time.Sleep(50 * time.Millisecond)
	select {
	case v, ok := <-out:
		if ok {
			t.Errorf("unexpected value received after cancel: %d", v)
		}
	default:
		t.Error("output channel not closed after cancel")
	}
}

func Test3(t *testing.T) {
	// Test with nil input channel
	ctx := context.Background()
	out := Watcher[int](ctx, nil)

	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected closed channel for nil input")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout waiting for channel close")
	}
}

func Test4(t *testing.T) {
	// Test closing input channel
	ctx := context.Background()
	in := make(chan string)
	out := Watcher(ctx, in)

	close(in)

	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected closed output channel")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout waiting for output close")
	}
}

func Test5(t *testing.T) {
	// Test with timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	in := make(chan int)
	out := Watcher(ctx, in)

	go func() {
		for i := 0; i < 10; i++ {
			in <- i
			time.Sleep(20 * time.Millisecond)
		}
	}()

	count := 0
	for range out {
		count++
	}

	if count >= 10 {
		t.Errorf("expected less than 10 values due to timeout, got %d", count)
	}
}

func Test6(t *testing.T) {
	// Test forwarding strings
	ctx := context.Background()
	in := make(chan string)
	out := Watcher(ctx, in)

	go func() {
		in <- "hello"
		in <- "world"
		close(in)
	}()

	result1 := <-out
	result2 := <-out

	if result1 != "hello" || result2 != "world" {
		t.Errorf("expected 'hello' and 'world', got '%s' and '%s'", result1, result2)
	}
}

func Test7(t *testing.T) {
	// Test empty channel
	ctx := context.Background()
	in := make(chan int)
	out := Watcher(ctx, in)

	close(in)

	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected closed channel")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout")
	}
}

func Test8(t *testing.T) {
	// Test canceled context before start
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	in := make(chan int)
	out := Watcher(ctx, in)

	select {
	case _, ok := <-out:
		if ok {
			t.Error("expected closed channel for canceled context")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout")
	}
}

func Test9(t *testing.T) {
	// Test multiple values
	ctx := context.Background()
	in := make(chan int, 5)
	out := Watcher(ctx, in)

	for i := 0; i < 5; i++ {
		in <- i
	}
	close(in)

	sum := 0
	for v := range out {
		sum += v
	}

	expected := 10 // 0+1+2+3+4
	if sum != expected {
		t.Errorf("expected sum %d, got %d", expected, sum)
	}
}

func Test10(t *testing.T) {
	// Test with nil context
	in := make(chan bool)
	out := Watcher[bool](nil, in)

	go func() {
		in <- true
		close(in)
	}()

	result := <-out
	if !result {
		t.Error("expected true")
	}
}`,
			hint1: `Create output channel and launch a goroutine with defer close(out). Use nested select to check ctx.Done().`,
			hint2: `First select reads from input or ctx.Done(). Second select sends to output or checks ctx.Done() again.`,
			whyItMatters: `Context-aware goroutines are essential for preventing goroutine leaks and enabling graceful shutdown in production systems. This pattern ensures your goroutines respect timeouts and cancellation signals, critical for resource management.`,	order: 0,
	translations: {
		ru: {
			title: 'Наблюдатель с контекстом',
			solutionCode: `package goroutinesx

import (
	"context"
)

func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan T, len(in))	// создаём выходной канал
	if in == nil {
		close(out)	// сразу закрываем для nil входа
		return out
	}

	go func() {
		defer close(out)	// гарантируем закрытие выхода при возврате
		for {
			select {
			case <-ctx.Done():	// контекст отменён
				return
			case v, ok := <-in:
				if !ok {	// входной канал закрыт
					return
				}
				select {
				case <-ctx.Done():	// проверяем снова перед отправкой
					return
				case out <- v:	// пересылаем значение в выход
				}
			}
		}
	}()
	return out
}`,
			description: `Реализуйте горутину которая наблюдает за входным каналом и пересылает значения до отмены контекста.

**Требования:**
1. **Watcher**: Создайте горутину которая проксирует канал с учётом контекста
2. **Обработка контекста**: Прекратите пересылку когда сигнализируется ctx.Done()
3. **Чистое завершение**: Закройте выходной канал когда закончено
4. **Без утечек**: Убедитесь что горутина завершается правильно

**Паттерн реализации:**
\`\`\`go
func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
    out := make(chan T)

    go func() {
        defer close(out)
        for {
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
                }
            }
        }
    }()

    return out
}
\`\`\`

**Пример использования:**
\`\`\`go
// Таймаут для долгой операции
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

dataStream := fetchDataStream()
watched := Watcher(ctx, dataStream)

for data := range watched {
    process(data)
    // Автоматически остановится через 5 секунд
}
\`\`\`

**Ограничения:**
- Должен создать выходной канал
- Должен обрабатывать nil контекст (использовать Background)
- Должен закрывать выходной канал при возврате
- Должен проверять ctx.Done() перед отправкой`,
			hint1: `Создайте выходной канал и запустите горутину с defer close(out). Используйте вложенный select для проверки ctx.Done().`,
			hint2: `Первый select читает из входа или ctx.Done(). Второй select отправляет в выход или снова проверяет ctx.Done().`,
			whyItMatters: `Context-aware горутины критичны для предотвращения утечек горутин и обеспечения graceful shutdown в production системах. Этот паттерн гарантирует что ваши горутины уважают таймауты и сигналы отмены, что критично для управления ресурсами.`
		},
		uz: {
			title: `Kontekst bilan kuzatuvchi`,
			solutionCode: `package goroutinesx

import (
	"context"
)

func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan T, len(in))	// chiqish kanalini yaratamiz
	if in == nil {
		close(out)	// nil kirish uchun darhol yopamiz
		return out
	}

	go func() {
		defer close(out)	// qaytishda chiqishni yopishni kafolatlaymiz
		for {
			select {
			case <-ctx.Done():	// kontekst bekor qilindi
				return
			case v, ok := <-in:
				if !ok {	// kirish kanali yopilgan
					return
				}
				select {
				case <-ctx.Done():	// yuborishdan oldin yana tekshiramiz
					return
				case out <- v:	// qiymatni chiqishga yo'naltiramiz
				}
			}
		}
	}()
	return out
}`,
			description: `Kirish kanalini kuzatib, kontekst bekor qilinguncha qiymatlarni yo'naltiruvchi goroutine ni amalga oshiring.

**Talablar:**
1. **Watcher**: Kontekst xabardorligi bilan kanalni proksi qiluvchi goroutine yaratish
2. **Kontekst qayta ishlash**: ctx.Done() signal bo'lganda yo'naltirishni to'xtatish
3. **Toza yopilish**: Tugaganda chiqish kanalini yopish
4. **Oqishsiz**: Goroutine to'g'ri tugashini ta'minlash

**Amalga oshirish patterni:**
\`\`\`go
func Watcher[T any](ctx context.Context, in <-chan T) <-chan T {
    out := make(chan T)

    go func() {
        defer close(out)
        for {
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
                }
            }
        }
    }()

    return out
}
\`\`\`

**Cheklovlar:**
- Chiqish kanalini yaratish kerak
- nil kontekstni qayta ishlash kerak (Background ishlating)
- Qaytishda chiqish kanalini yopish kerak
- Yuborishdan oldin ctx.Done() ni tekshirish kerak`,
			hint1: `Chiqish kanalini yarating va defer close(out) bilan goroutine ishga tushiring. ctx.Done() ni tekshirish uchun ichki select ishlating.`,
			hint2: `Birinchi select kirish yoki ctx.Done() dan o'qiydi. Ikkinchi select chiqishga yuboradi yoki ctx.Done() ni qayta tekshiradi.`,
			whyItMatters: `Kontekstni hisobga olgan goroutinelar production tizimlarida goroutine oqishlarini oldini olish va graceful shutdown ni yoqish uchun muhimdir. Bu pattern goroutinelaringiz timeout va bekor qilish signallarini hurmat qilishini ta'minlaydi, resurs boshqaruvi uchun muhim.`
		}
	}
};

export default task;
