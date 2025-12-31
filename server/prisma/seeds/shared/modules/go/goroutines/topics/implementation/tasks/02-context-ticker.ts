import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-goroutines-context-ticker',
	title: 'Context-Aware Ticker',
	difficulty: 'medium',	tags: ['go', 'goroutines', 'context', 'time'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement a ticker that sends periodic ticks until context is canceled, with proper cleanup.

**Requirements:**
1. **ContextTicker**: Create channel that emits ticks at regular intervals
2. **Ticker Cleanup**: Stop ticker when context is canceled
3. **Channel Management**: Close output channel on context cancellation
4. **Resource Cleanup**: Prevent ticker resource leak

**Implementation Pattern:**
\`\`\`go
func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
    out := make(chan time.Time)
    ticker := time.NewTicker(d)

    go func() {
        defer ticker.Stop()  // Critical: stop ticker
        defer close(out)

        for {
            select {
            case <-ctx.Done():
                return
            case t := <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- t:
                }
            }
        }
    }()

    return out
}
\`\`\`

**Example Usage:**
\`\`\`go
// Health check every 30 seconds
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

ticks := ContextTicker(ctx, 30*time.Second)
for t := range ticks {
    healthCheck()
    log.Printf("Health check at %v", t)
}

// Metrics collection
ctx := context.Background()
ticks := ContextTicker(ctx, 1*time.Minute)
for range ticks {
    collectMetrics()
    pushToPrometheus()
}
\`\`\`

**Constraints:**
- Must use time.NewTicker
- Must call ticker.Stop() in defer
- Must close output channel
- Must handle context cancellation`,
	initialCode: `package goroutinesx

import (
	"context"
	"time"
)

// TODO: Implement ContextTicker
// Create time.NewTicker with duration d
// Forward ticks to output channel
// Stop ticker and close channel on ctx.Done()
func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
	// TODO: Implement
}`,
	solutionCode: `package goroutinesx

import (
	"context"
	"time"
)

func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
	out := make(chan time.Time)
	ticker := time.NewTicker(d)	// create ticker

	go func() {
		defer ticker.Stop()	// critical: stop ticker to free resources
		defer close(out)	// close output channel
		for {
			select {
			case <-ctx.Done():	// context canceled
				return
			case t := <-ticker.C:	// tick received
				select {
				case <-ctx.Done():	// check again before sending
					return
				case out <- t:	// forward tick
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
	// ContextTicker returns a channel
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch := ContextTicker(ctx, 100*time.Millisecond)
	if ch == nil {
		t.Error("expected non-nil channel")
	}
}

func Test2(t *testing.T) {
	// ContextTicker sends ticks
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch := ContextTicker(ctx, 50*time.Millisecond)

	select {
	case <-ch:
		// Got a tick
	case <-time.After(200*time.Millisecond):
		t.Error("expected to receive tick")
	}
}

func Test3(t *testing.T) {
	// ContextTicker closes channel on context cancel
	ctx, cancel := context.WithCancel(context.Background())
	ch := ContextTicker(ctx, 50*time.Millisecond)

	// Receive one tick first
	<-ch

	cancel()

	// Channel should close
	select {
	case _, ok := <-ch:
		if ok {
			// Might get one more tick before closure
		}
	case <-time.After(200*time.Millisecond):
		t.Error("expected channel to close")
	}
}

func Test4(t *testing.T) {
	// ContextTicker with cancelled context closes immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ch := ContextTicker(ctx, 50*time.Millisecond)

	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected closed channel")
		}
	case <-time.After(200*time.Millisecond):
		t.Error("expected channel to close immediately")
	}
}

func Test5(t *testing.T) {
	// ContextTicker sends multiple ticks
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch := ContextTicker(ctx, 30*time.Millisecond)

	count := 0
	timeout := time.After(200*time.Millisecond)

	for count < 3 {
		select {
		case <-ch:
			count++
		case <-timeout:
			t.Error("expected 3 ticks")
			return
		}
	}
}

func Test6(t *testing.T) {
	// ContextTicker with timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	ch := ContextTicker(ctx, 30*time.Millisecond)

	// Should receive some ticks then close
	time.Sleep(200*time.Millisecond)

	select {
	case _, ok := <-ch:
		if ok {
			// OK - might still have a tick
		}
	case <-time.After(100*time.Millisecond):
		// Channel might already be closed
	}
}

func Test7(t *testing.T) {
	// ContextTicker tick type is time.Time
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch := ContextTicker(ctx, 50*time.Millisecond)

	select {
	case tick := <-ch:
		if tick.IsZero() {
			t.Error("expected non-zero time")
		}
	case <-time.After(200*time.Millisecond):
		t.Error("expected tick")
	}
}

func Test8(t *testing.T) {
	// ContextTicker with short duration
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ch := ContextTicker(ctx, 10*time.Millisecond)

	select {
	case <-ch:
		// Fast tick received
	case <-time.After(100*time.Millisecond):
		t.Error("expected quick tick")
	}
}

func Test9(t *testing.T) {
	// ContextTicker allows range iteration
	ctx, cancel := context.WithCancel(context.Background())

	ch := ContextTicker(ctx, 30*time.Millisecond)

	go func() {
		time.Sleep(150*time.Millisecond)
		cancel()
	}()

	count := 0
	for range ch {
		count++
		if count > 10 {
			break
		}
	}
}

func Test10(t *testing.T) {
	// ContextTicker respects parent context
	parent, parentCancel := context.WithCancel(context.Background())
	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	ch := ContextTicker(ctx, 50*time.Millisecond)

	<-ch // Get one tick

	parentCancel()

	time.Sleep(100*time.Millisecond)

	select {
	case _, ok := <-ch:
		if ok {
			// Might get buffered tick
		}
	case <-time.After(100*time.Millisecond):
		// OK - channel should close
	}
}
`,
		hint1: `Create ticker with time.NewTicker(d). Use defer ticker.Stop() and defer close(out).`,
			hint2: `Use nested select: outer reads from ticker.C or ctx.Done(), inner sends to out or checks ctx.Done().`,
			whyItMatters: `Context-aware tickers prevent resource leaks in long-running services. Forgetting to stop tickers is a common cause of memory leaks in production Go applications. This pattern ensures proper cleanup.`,	order: 1,
	translations: {
		ru: {
			title: 'Тикер с контекстом',
			solutionCode: `package goroutinesx

import (
	"context"
	"time"
)

func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
	out := make(chan time.Time)
	ticker := time.NewTicker(d)	// создаём ticker

	go func() {
		defer ticker.Stop()	// критично: останавливаем ticker для освобождения ресурсов
		defer close(out)	// закрываем выходной канал
		for {
			select {
			case <-ctx.Done():	// контекст отменён
				return
			case t := <-ticker.C:	// получен тик
				select {
				case <-ctx.Done():	// проверяем снова перед отправкой
					return
				case out <- t:	// пересылаем тик
				}
			}
		}
	}()

	return out
}`,
			description: `Реализуйте ticker который отправляет периодические тики до отмены контекста с правильной очисткой.

**Требования:**
1. **ContextTicker**: Создайте канал который посылает тики с регулярными интервалами
2. **Очистка тикера**: Остановите ticker когда контекст отменён
3. **Управление каналами**: Закройте выходной канал при отмене контекста
4. **Очистка ресурсов**: Предотвратите утечку ресурсов ticker

**Паттерн реализации:**
\`\`\`go
func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
    out := make(chan time.Time)
    ticker := time.NewTicker(d)

    go func() {
        defer ticker.Stop()  // Критично: остановить ticker
        defer close(out)

        for {
            select {
            case <-ctx.Done():
                return
            case t := <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- t:
                }
            }
        }
    }()

    return out
}
\`\`\`

**Пример использования:**
\`\`\`go
// Проверка здоровья каждые 30 секунд
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

ticks := ContextTicker(ctx, 30*time.Second)
for t := range ticks {
    healthCheck()
    log.Printf("Health check at %v", t)
}

// Сбор метрик
ctx := context.Background()
ticks := ContextTicker(ctx, 1*time.Minute)
for range ticks {
    collectMetrics()
    pushToPrometheus()
}
\`\`\`

**Ограничения:**
- Должен использовать time.NewTicker
- Должен вызвать ticker.Stop() в defer
- Должен закрыть выходной канал
- Должен обрабатывать отмену контекста`,
			hint1: `Создайте ticker с time.NewTicker(d). Используйте defer ticker.Stop() и defer close(out).`,
			hint2: `Используйте вложенный select: внешний читает из ticker.C или ctx.Done(), внутренний отправляет в out или проверяет ctx.Done().`,
			whyItMatters: `Context-aware ticker'ы предотвращают утечки ресурсов в долгоживущих сервисах. Забывание остановки ticker'ов - частая причина утечек памяти в production Go приложениях. Этот паттерн обеспечивает правильную очистку.`
		},
		uz: {
			title: `Kontekst bilan tiker`,
			solutionCode: `package goroutinesx

import (
	"context"
	"time"
)

func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
	out := make(chan time.Time)
	ticker := time.NewTicker(d)	// ticker yaratamiz

	go func() {
		defer ticker.Stop()	// muhim: resurslarni bo'shatish uchun tickerni to'xtatamiz
		defer close(out)	// chiqish kanalini yopamiz
		for {
			select {
			case <-ctx.Done():	// kontekst bekor qilindi
				return
			case t := <-ticker.C:	// tick qabul qilindi
				select {
				case <-ctx.Done():	// yuborishdan oldin yana tekshiramiz
					return
				case out <- t:	// tickni yo'naltiramiz
				}
			}
		}
	}()

	return out
}`,
			description: `Kontekst bekor qilinguncha to'g'ri tozalash bilan davriy ticklar yuboradigan ticker ni amalga oshiring.

**Talablar:**
1. **ContextTicker**: Muntazam intervallarda tick yuboradigan kanal yaratish
2. **Ticker tozalash**: Kontekst bekor qilinganda tickerni to'xtatish
3. **Kanal boshqaruvi**: Kontekst bekor qilishda chiqish kanalini yopish
4. **Resurs tozalash**: Ticker resurs oqishini oldini olish

**Amalga oshirish patterni:**
\`\`\`go
func ContextTicker(ctx context.Context, d time.Duration) <-chan time.Time {
    out := make(chan time.Time)
    ticker := time.NewTicker(d)

    go func() {
        defer ticker.Stop()  // Muhim: tickerni to'xtatish
        defer close(out)

        for {
            select {
            case <-ctx.Done():
                return
            case t := <-ticker.C:
                select {
                case <-ctx.Done():
                    return
                case out <- t:
                }
            }
        }
    }()

    return out
}
\`\`\`

**Cheklovlar:**
- time.NewTicker ishlatish kerak
- defer da ticker.Stop() ni chaqirish kerak
- Chiqish kanalini yopish kerak
- Kontekst bekor qilishni qayta ishlash kerak`,
			hint1: `time.NewTicker(d) bilan ticker yarating. defer ticker.Stop() va defer close(out) ishlating.`,
			hint2: `Ichki select ishlating: tashqi ticker.C yoki ctx.Done() dan o'qiydi, ichki out ga yuboradi yoki ctx.Done() ni tekshiradi.`,
			whyItMatters: `Kontekstni hisobga olgan tickerlar uzoq ishlaydigan xizmatlarda resurs oqishlarini oldini oladi. Tickerlarni to'xtatishni unutish production Go ilovalarida memory leak larning keng tarqalgan sababidir. Bu pattern to'g'ri tozalashni ta'minlaydi.`
		}
	}
};

export default task;
