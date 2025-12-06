
// @ts-ignore
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// --- REAL GO COURSE CONTENT DERIVED FROM FILES ---

const GO_COURSE = {
  slug: 'c_go',
  title: 'Go Language Mastery',
  description: 'Master Go from syntax to high-performance concurrency patterns and system design.',
  category: 'language',
  icon: '🐹',
  estimatedTime: '56h',
};

const MODULES = [
  {
    title: 'Concurrency & Synchronization',
    description: 'Deep dive into Goroutines, Channels, and the sync package.',
    section: 'core',
    order: 1,
    topics: [
      {
        title: 'Channels Patterns',
        description: 'Advanced communication patterns using channels.',
        difficulty: 'hard',
        estimatedTime: '4h',
        order: 1,
        tasks: [
          {
            slug: 'go-fan-in',
            title: 'Fan-In Pattern',
            difficulty: 'medium',
            tags: ['go', 'concurrency', 'channels'],
            estimatedTime: '30m',
            isPremium: false,
            description: `Implement the **Fan-In** pattern.
            
The function \`FanIn\` should accept a variadic number of read-only channels and return a single channel that emits data from all input channels.

### Requirements:
1. The output channel must close **only** when all input channels are closed.
2. It must be thread-safe.
3. Ensure no goroutines leak after completion.

|||Example
Input: ch1 yields [1, 2], ch2 yields [3, 4]
Output: [1, 3, 2, 4] (Order may vary)
`,
            initialCode: `package channelsx

import (
	"context"
)

// FanIn merges multiple input channels into a single output channel.
func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	out := make(chan T)
	// TODO: Implement fan-in logic
	return out
}`,
            solutionCode: `package channelsx

import (
	"context"
	"sync"
)

func FanIn[T any](ctx context.Context, ins ...<-chan T) <-chan T {
	if ctx == nil {
		ctx = context.Background()
	}
	out := make(chan T)
	var wg sync.WaitGroup
	forward := func(in <-chan T) {
		defer wg.Done()
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
	}
	for _, in := range ins {
		if in == nil {
			continue
		}
		wg.Add(1)
		go forward(in)
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}`,
            hints: [
              "Use a `sync.WaitGroup` to track active input channels.",
              "Launch a goroutine for each input channel that forwards data to `out`."
            ]
          },
          {
            slug: 'go-worker-pool',
            title: 'Worker Pool',
            difficulty: 'hard',
            tags: ['go', 'concurrency', 'patterns'],
            estimatedTime: '45m',
            isPremium: true,
            description: `Implement a **Worker Pool** to process jobs concurrently with a limited number of workers.

### Requirements:
1. \`RunWorkerPool\` accepts a context, an input channel, a worker count, and a handler function.
2. Spawn exactly \`workers\` number of goroutines.
3. If any handler returns an error, the first error should be returned by \`RunWorkerPool\`, and the context should be canceled to stop other workers.
`,
            initialCode: `package channelsx

import (
	"context"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	// TODO: Implement worker pool
	return nil
}`,
            solutionCode: `package channelsx

import (
	"context"
	"sync"
)

type Handler[T any] func(context.Context, T) error

func RunWorkerPool[T any](ctx context.Context, in <-chan T, workers int, h Handler[T]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if h == nil || workers <= 0 || in == nil {
		return nil
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	recordErr := func(err error) {
		if err != nil {
			once.Do(func() {
				firstErr = err
				cancel()
			})
		}
	}

	worker := func() {
		defer wg.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-in:
				if !ok {
					return
				}
				if err := h(ctx, v); err != nil {
					recordErr(err)
				}
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()
	}
	wg.Wait()
	return firstErr
}`,
            hints: [
              "Use `sync.Once` to capture only the first error.",
              "Call `cancel()` inside the error handler to signal other workers to stop."
            ]
          }
        ]
      }
    ]
  },
  {
    title: 'Resilience & Reliability',
    description: 'Building robust systems that handle failures gracefully.',
    section: 'core',
    order: 2,
    topics: [
      {
        title: 'Caching Strategies',
        description: 'In-memory storage mechanisms and expiration policies.',
        difficulty: 'medium',
        estimatedTime: '2h',
        order: 1,
        tasks: [
          {
            slug: 'go-ttl-cache',
            title: 'TTL Cache',
            difficulty: 'medium',
            tags: ['go', 'cache', 'system-design'],
            estimatedTime: '45m',
            isPremium: false,
            description: `Implement a Thread-Safe **TTL (Time-To-Live) Cache**.

### Requirements:
1. ` + "`NewTTLCache(ttl)`" + ` initializes the cache.
2. ` + "`Set(key, value)`" + ` adds items.
3. ` + "`Get(key)`" + ` retrieves items.
4. **Important**: If an item's TTL has expired, ` + "`Get`" + ` must not return it and should remove it from the map (Lazy Expiration).
5. It must be safe for concurrent use.
`,
            initialCode: `package cache

import (
	"sync"
	"time"
)

type TTLCache struct {
	// TODO: Add fields
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{}
}

func (c *TTLCache) Set(key string, v any) {
    // Implement
}

func (c *TTLCache) Get(key string) (any, bool) {
    // Implement
    return nil, false
}`,
            solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	expire := time.Time{}
	if c.ttl > 0 {
		expire = time.Now().Add(c.ttl)
	}
	c.m[key] = entry{v: v, exp: expire}
}

func (c *TTLCache) Get(key string) (any, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	ent, ok := c.m[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
			delete(c.m, key)
		}
		return nil, false
	}
	return ent.v, true
}`,
            hints: [
              "Use `sync.RWMutex`. Use RLock for reading the entry, but upgrade to Lock if you need to delete an expired entry.",
              "Be careful of race conditions when upgrading the lock. Re-check the condition after acquiring the Write lock."
            ]
          }
        ]
      },
      {
        title: 'Fault Tolerance',
        description: 'Patterns like Circuit Breaker and Retries.',
        difficulty: 'hard',
        estimatedTime: '3h',
        order: 2,
        tasks: [
          {
            slug: 'go-circuit-breaker',
            title: 'Circuit Breaker',
            difficulty: 'hard',
            tags: ['go', 'resiliency'],
            estimatedTime: '1h',
            isPremium: true,
            description: `Implement a **Circuit Breaker** state machine.

It should have 3 states:
1. **Closed**: Requests pass through. Errors count towards threshold.
2. **Open**: Requests fail immediately with ` + "`ErrOpen`" + `.
3. **Half-Open**: After a cooldown, allow limited requests to test if the service is back.

### Logic:
- **Closed** -> **Open**: When failure count >= threshold.
- **Open** -> **Half-Open**: After ` + "`openDur`" + ` time passes.
- **Half-Open** -> **Closed**: After ` + "`halfMax`" + ` consecutive successes.
- **Half-Open** -> **Open**: If any request fails.
`,
            initialCode: `package circuitx

import (
	"context"
	"errors"
	"sync"
	"time"
)

var ErrOpen = errors.New("circuit open")

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

type Breaker struct {
	// TODO: Add fields
}

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{
        // Init
	}
}

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	// TODO: Implement state machine
    return nil
}`,
            solutionCode: `package circuitx

import (
	"context"
	"errors"
	"sync"
	"time"
)

var ErrOpen = errors.New("circuit open")

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

type Breaker struct {
	mu        sync.Mutex
	state     State
	errs      int
	threshold int
	openUntil time.Time
	openDur   time.Duration
	halfMax   int
	halfCount int
}

func New(threshold int, openDur time.Duration, halfMax int) *Breaker {
	return &Breaker{
		state:     Closed,
		threshold: threshold,
		openDur:   openDur,
		halfMax:   halfMax,
	}
}

func (b *Breaker) Do(ctx context.Context, f func(context.Context) error) error {
	b.mu.Lock()
	now := time.Now()
	switch b.state {
	case Open:
		if now.After(b.openUntil) {
			b.state = HalfOpen
			b.halfCount = 0
		} else {
			b.mu.Unlock()
			return ErrOpen
		}
	}
	b.mu.Unlock()

	err := f(ctx)

	b.mu.Lock()
	defer b.mu.Unlock()

	if err == nil {
		switch b.state {
		case Closed:
			b.errs = 0
		case HalfOpen:
			b.halfCount++
			if b.halfCount >= b.halfMax {
				b.state = Closed
				b.errs = 0
				b.halfCount = 0
			}
		}
		return nil
	}

	switch b.state {
	case Closed:
		b.errs++
		if b.errs >= b.threshold {
			b.tripToOpen()
		}
	case HalfOpen:
		b.tripToOpen()
	}
	return err
}

func (b *Breaker) tripToOpen() {
	b.state = Open
	b.openUntil = time.Now().Add(b.openDur)
	b.errs = 0
	b.halfCount = 0
}`,
            hints: ["Use `sync.Mutex` to protect all state transitions.", "Check the state *before* calling the function `f`, and update state *after* `f` returns."]
          }
        ]
      }
    ]
  },
  {
    title: 'Functional Programming',
    description: 'Utilizing Go Generics for functional patterns.',
    section: 'frameworks',
    order: 3,
    topics: [
      {
        title: 'Generics',
        description: 'Map, Filter, and Reduce implementation.',
        difficulty: 'medium',
        estimatedTime: '1.5h',
        order: 1,
        tasks: [
          {
            slug: 'go-generic-map',
            title: 'Generic Map',
            difficulty: 'easy',
            tags: ['go', 'generics'],
            estimatedTime: '15m',
            isPremium: false,
            description: `Implement a generic \`Map\` function.
            
It should take a slice of type \`T\` and a function that transforms \`T\` to \`R\`, returning a slice of \`R\`.`,
            initialCode: `package genericsx

func Map[T any, R any](in []T, f func(T) R) []R {
	// Implement
    return nil
}`,
            solutionCode: `package genericsx

func Map[T any, R any](in []T, f func(T) R) []R {
	out := make([]R, len(in))
	for i, v := range in {
		out[i] = f(v)
	}
	return out
}`,
            hints: ["Pre-allocate the result slice using `make([]R, len(in))` for better performance."]
          }
        ]
      }
    ]
  }
];

async function main() {
  console.log('🌱 Starting seed...');

  // 1. Create or Update Course
  const course = await prisma.course.upsert({
    where: { slug: GO_COURSE.slug },
    update: GO_COURSE,
    create: GO_COURSE,
  });
  console.log(`📚 Course: ${course.title}`);

  // 2. Create Modules & Topics & Tasks
  for (const mod of MODULES) {
    const module = await prisma.module.create({
      data: {
        title: mod.title,
        description: mod.description,
        order: mod.order,
        section: mod.section,
        courseId: course.id,
      },
    });
    console.log(`  📦 Module: ${module.title}`);

    for (const top of mod.topics) {
      const topic = await prisma.topic.create({
        data: {
          title: top.title,
          description: top.description,
          difficulty: top.difficulty,
          estimatedTime: top.estimatedTime,
          order: top.order,
          moduleId: module.id,
        },
      });
      console.log(`    🏷️ Topic: ${topic.title}`);

      for (const t of top.tasks) {
        // Upsert tasks
        const task = await prisma.task.upsert({
          where: { slug: t.slug },
          update: {
            topicId: topic.id,
            order: top.tasks.indexOf(t) + 1,
            initialCode: t.initialCode,
            solutionCode: t.solutionCode,
            description: t.description,
            hints: t.hints
          },
          create: {
            ...t,
            topicId: topic.id,
            order: top.tasks.indexOf(t) + 1,
          },
        });
        console.log(`      📝 Task: ${task.title}`);
      }
    }
  }

  console.log('✅ Seeding finished.');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
