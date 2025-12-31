import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-connection-pool',
  title: 'Model Connection Pool',
  difficulty: 'hard',
  tags: ['go', 'ml', 'pool', 'concurrency', 'resources'],
  estimatedTime: '35m',
  isPremium: true,
  order: 5,

  description: `
## Model Connection Pool

Implement a connection pool for ML model instances to efficiently manage GPU/CPU resources and handle concurrent inference requests.

### Requirements

1. **ModelPool** - Pool management:
   - \`NewModelPool(config PoolConfig, factory ModelFactory)\` - Create pool
   - \`Acquire(ctx context.Context) (Model, error)\` - Get model from pool
   - \`Release(model Model)\` - Return model to pool
   - \`Close()\` - Shutdown pool and release resources

2. **PoolConfig** - Configuration:
   - \`MinSize int\` - Minimum pool size (pre-warmed)
   - \`MaxSize int\` - Maximum pool size
   - \`MaxIdleTime time.Duration\` - Idle timeout before cleanup
   - \`MaxLifetime time.Duration\` - Maximum model lifetime
   - \`AcquireTimeout time.Duration\` - Max wait time for model

3. **ModelFactory** - Creates model instances:
   - \`Create() (Model, error)\` - Create new model
   - \`Validate(model Model) error\` - Check model health
   - \`Destroy(model Model) error\` - Clean up model

4. **Pool Features**:
   - Pre-warming: create MinSize models on startup
   - Lazy creation: create up to MaxSize on demand
   - Health checks: validate models before use
   - Automatic cleanup: remove idle/expired models
   - Metrics: pool size, wait times, utilization

### Example

\`\`\`go
factory := &ONNXModelFactory{modelPath: "model.onnx"}

pool := NewModelPool(PoolConfig{
    MinSize:        2,
    MaxSize:        10,
    MaxIdleTime:    5 * time.Minute,
    AcquireTimeout: 30 * time.Second,
}, factory)

model, err := pool.Acquire(ctx)
if err != nil {
    return err
}
defer pool.Release(model)

result := model.Predict(features)
\`\`\`
`,

  initialCode: `package modelpool

import (
	"context"
	"sync"
	"time"
)

type Model interface {
}

type ModelFactory interface {
}

type PoolConfig struct {
	MinSize        int
	MaxSize        int
	MaxIdleTime    time.Duration
	MaxLifetime    time.Duration
	AcquireTimeout time.Duration
}

type PoolStats struct {
	CurrentSize   int
	IdleCount     int
	InUse         int
	TotalCreated  int64
	TotalDestroyed int64
	WaitCount     int64
	WaitDuration  time.Duration
}

type ModelPool struct {
}

func NewModelPool(config PoolConfig, factory ModelFactory) (*ModelPool, error) {
	return nil, nil
}

func (p *ModelPool) Acquire(ctx context.Context) (Model, error) {
	return nil, nil
}

func (p *ModelPool) Release(model Model) {
}

func (p *ModelPool) Stats() PoolStats {
	return PoolStats{}
}

func (p *ModelPool) Close() error {
	return nil
}

type pooledModel struct {
}`,

  solutionCode: `package modelpool

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Model represents an ML model instance
type Model interface {
	Predict(input []float32) ([]float32, error)
	Close() error
}

// ModelFactory creates and manages model instances
type ModelFactory interface {
	Create() (Model, error)
	Validate(model Model) error
	Destroy(model Model) error
}

// PoolConfig configures the model pool
type PoolConfig struct {
	MinSize        int
	MaxSize        int
	MaxIdleTime    time.Duration
	MaxLifetime    time.Duration
	AcquireTimeout time.Duration
}

// PoolStats holds pool statistics
type PoolStats struct {
	CurrentSize    int
	IdleCount      int
	InUse          int
	TotalCreated   int64
	TotalDestroyed int64
	WaitCount      int64
	WaitDuration   time.Duration
}

// pooledModel wraps a model with pool metadata
type pooledModel struct {
	model      Model
	createdAt  time.Time
	lastUsedAt time.Time
}

// ModelPool manages a pool of model instances
type ModelPool struct {
	config  PoolConfig
	factory ModelFactory

	mu       sync.Mutex
	idle     []*pooledModel
	inUse    map[*pooledModel]bool
	closed   bool
	cond     *sync.Cond

	totalCreated   int64
	totalDestroyed int64
	waitCount      int64
	totalWaitTime  int64

	cleanupDone chan struct{}
}

// NewModelPool creates a new model pool
func NewModelPool(config PoolConfig, factory ModelFactory) (*ModelPool, error) {
	if config.MinSize < 0 {
		config.MinSize = 0
	}
	if config.MaxSize <= 0 {
		config.MaxSize = 10
	}
	if config.MinSize > config.MaxSize {
		config.MinSize = config.MaxSize
	}
	if config.MaxIdleTime <= 0 {
		config.MaxIdleTime = 5 * time.Minute
	}
	if config.AcquireTimeout <= 0 {
		config.AcquireTimeout = 30 * time.Second
	}

	p := &ModelPool{
		config:      config,
		factory:     factory,
		idle:        make([]*pooledModel, 0, config.MaxSize),
		inUse:       make(map[*pooledModel]bool),
		cleanupDone: make(chan struct{}),
	}
	p.cond = sync.NewCond(&p.mu)

	// Pre-warm pool
	for i := 0; i < config.MinSize; i++ {
		model, err := factory.Create()
		if err != nil {
			p.Close()
			return nil, err
		}
		pm := &pooledModel{
			model:      model,
			createdAt:  time.Now(),
			lastUsedAt: time.Now(),
		}
		p.idle = append(p.idle, pm)
		atomic.AddInt64(&p.totalCreated, 1)
	}

	// Start cleanup goroutine
	go p.cleanupLoop()

	return p, nil
}

func (p *ModelPool) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.cleanup()
		case <-p.cleanupDone:
			return
		}
	}
}

func (p *ModelPool) cleanup() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	now := time.Now()
	newIdle := make([]*pooledModel, 0, len(p.idle))

	for _, pm := range p.idle {
		shouldRemove := false

		// Check idle timeout (keep minimum)
		if len(newIdle)+len(p.inUse) >= p.config.MinSize {
			if now.Sub(pm.lastUsedAt) > p.config.MaxIdleTime {
				shouldRemove = true
			}
		}

		// Check lifetime
		if p.config.MaxLifetime > 0 && now.Sub(pm.createdAt) > p.config.MaxLifetime {
			shouldRemove = true
		}

		if shouldRemove {
			p.factory.Destroy(pm.model)
			atomic.AddInt64(&p.totalDestroyed, 1)
		} else {
			newIdle = append(newIdle, pm)
		}
	}

	p.idle = newIdle
}

// Acquire gets a model from the pool
func (p *ModelPool) Acquire(ctx context.Context) (Model, error) {
	p.mu.Lock()

	if p.closed {
		p.mu.Unlock()
		return nil, errors.New("pool is closed")
	}

	startWait := time.Now()
	atomic.AddInt64(&p.waitCount, 1)

	// Try to get from idle
	for {
		// Check context
		select {
		case <-ctx.Done():
			p.mu.Unlock()
			return nil, ctx.Err()
		default:
		}

		// Try idle pool
		if len(p.idle) > 0 {
			pm := p.idle[len(p.idle)-1]
			p.idle = p.idle[:len(p.idle)-1]

			// Validate model
			if err := p.factory.Validate(pm.model); err != nil {
				p.factory.Destroy(pm.model)
				atomic.AddInt64(&p.totalDestroyed, 1)
				continue // Try next
			}

			pm.lastUsedAt = time.Now()
			p.inUse[pm] = true
			p.mu.Unlock()

			atomic.AddInt64(&p.totalWaitTime, int64(time.Since(startWait)))
			return &pooledModelWrapper{pm: pm, pool: p}, nil
		}

		// Try to create new
		if len(p.idle)+len(p.inUse) < p.config.MaxSize {
			p.mu.Unlock()

			model, err := p.factory.Create()
			if err != nil {
				return nil, err
			}

			atomic.AddInt64(&p.totalCreated, 1)

			pm := &pooledModel{
				model:      model,
				createdAt:  time.Now(),
				lastUsedAt: time.Now(),
			}

			p.mu.Lock()
			p.inUse[pm] = true
			p.mu.Unlock()

			atomic.AddInt64(&p.totalWaitTime, int64(time.Since(startWait)))
			return &pooledModelWrapper{pm: pm, pool: p}, nil
		}

		// Wait for release
		deadline, hasDeadline := ctx.Deadline()
		if !hasDeadline {
			deadline = time.Now().Add(p.config.AcquireTimeout)
		}

		if time.Now().After(deadline) {
			p.mu.Unlock()
			return nil, errors.New("acquire timeout")
		}

		// Wait with timeout
		done := make(chan struct{})
		go func() {
			time.Sleep(100 * time.Millisecond)
			p.cond.Broadcast()
			close(done)
		}()

		p.cond.Wait()
		<-done
	}
}

// pooledModelWrapper wraps model for automatic release tracking
type pooledModelWrapper struct {
	pm   *pooledModel
	pool *ModelPool
}

func (w *pooledModelWrapper) Predict(input []float32) ([]float32, error) {
	return w.pm.model.Predict(input)
}

func (w *pooledModelWrapper) Close() error {
	// Don't close, just release back to pool
	w.pool.Release(w.pm.model)
	return nil
}

// Release returns a model to the pool
func (p *ModelPool) Release(model Model) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Find the pooled model
	var pm *pooledModel
	for candidate := range p.inUse {
		if wrapper, ok := model.(*pooledModelWrapper); ok {
			if wrapper.pm == candidate {
				pm = candidate
				break
			}
		}
		if candidate.model == model {
			pm = candidate
			break
		}
	}

	if pm == nil {
		return
	}

	delete(p.inUse, pm)

	if p.closed {
		p.factory.Destroy(pm.model)
		atomic.AddInt64(&p.totalDestroyed, 1)
		return
	}

	pm.lastUsedAt = time.Now()
	p.idle = append(p.idle, pm)
	p.cond.Signal()
}

// Stats returns current pool statistics
func (p *ModelPool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	return PoolStats{
		CurrentSize:    len(p.idle) + len(p.inUse),
		IdleCount:      len(p.idle),
		InUse:          len(p.inUse),
		TotalCreated:   atomic.LoadInt64(&p.totalCreated),
		TotalDestroyed: atomic.LoadInt64(&p.totalDestroyed),
		WaitCount:      atomic.LoadInt64(&p.waitCount),
		WaitDuration:   time.Duration(atomic.LoadInt64(&p.totalWaitTime)),
	}
}

// Close shuts down the pool
func (p *ModelPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}

	p.closed = true
	close(p.cleanupDone)

	// Destroy all idle models
	for _, pm := range p.idle {
		p.factory.Destroy(pm.model)
		atomic.AddInt64(&p.totalDestroyed, 1)
	}
	p.idle = nil

	// Wake up waiting acquires
	p.cond.Broadcast()

	return nil
}
`,

  testCode: `package modelpool

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type mockModel struct {
	id     int
	closed bool
}

func (m *mockModel) Predict(input []float32) ([]float32, error) {
	return input, nil
}

func (m *mockModel) Close() error {
	m.closed = true
	return nil
}

type mockFactory struct {
	counter   int32
	failNext  bool
	validateFail bool
}

func (f *mockFactory) Create() (Model, error) {
	if f.failNext {
		return nil, errors.New("create failed")
	}
	id := atomic.AddInt32(&f.counter, 1)
	return &mockModel{id: int(id)}, nil
}

func (f *mockFactory) Validate(model Model) error {
	if f.validateFail {
		return errors.New("validation failed")
	}
	return nil
}

func (f *mockFactory) Destroy(model Model) error {
	if m, ok := model.(*mockModel); ok {
		m.closed = true
	}
	return nil
}

func TestNewModelPool(t *testing.T) {
	factory := &mockFactory{}
	pool, err := NewModelPool(PoolConfig{
		MinSize: 2,
		MaxSize: 5,
	}, factory)

	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()

	stats := pool.Stats()
	if stats.CurrentSize != 2 {
		t.Errorf("Expected 2 pre-warmed models, got %d", stats.CurrentSize)
	}
	if stats.IdleCount != 2 {
		t.Errorf("Expected 2 idle models, got %d", stats.IdleCount)
	}
}

func TestAcquireRelease(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 1,
		MaxSize: 5,
	}, factory)
	defer pool.Close()

	model, err := pool.Acquire(context.Background())
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	stats := pool.Stats()
	if stats.InUse != 1 {
		t.Errorf("Expected 1 in use, got %d", stats.InUse)
	}

	pool.Release(model)

	stats = pool.Stats()
	if stats.InUse != 0 {
		t.Errorf("Expected 0 in use after release, got %d", stats.InUse)
	}
}

func TestPoolGrowth(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 0,
		MaxSize: 3,
	}, factory)
	defer pool.Close()

	models := make([]Model, 3)
	for i := 0; i < 3; i++ {
		m, err := pool.Acquire(context.Background())
		if err != nil {
			t.Fatalf("Acquire %d failed: %v", i, err)
		}
		models[i] = m
	}

	stats := pool.Stats()
	if stats.CurrentSize != 3 {
		t.Errorf("Expected pool size 3, got %d", stats.CurrentSize)
	}

	for _, m := range models {
		pool.Release(m)
	}
}

func TestAcquireTimeout(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize:        1,
		MaxSize:        1,
		AcquireTimeout: 100 * time.Millisecond,
	}, factory)
	defer pool.Close()

	// Acquire the only model
	model, _ := pool.Acquire(context.Background())

	// Try to acquire another - should timeout
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := pool.Acquire(ctx)
	if err == nil {
		t.Error("Expected timeout error")
	}

	pool.Release(model)
}

func TestConcurrentAccess(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 2,
		MaxSize: 5,
	}, factory)
	defer pool.Close()

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				model, err := pool.Acquire(context.Background())
				if err != nil {
					errors <- err
					return
				}
				// Simulate work
				time.Sleep(time.Millisecond)
				pool.Release(model)
			}
		}()
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Concurrent error: %v", err)
	}
}

func TestPoolClose(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 2,
		MaxSize: 5,
	}, factory)

	pool.Close()

	_, err := pool.Acquire(context.Background())
	if err == nil {
		t.Error("Expected error on closed pool")
	}
}

func TestModelValidation(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 1,
		MaxSize: 5,
	}, factory)
	defer pool.Close()

	// Acquire and release
	model, _ := pool.Acquire(context.Background())
	pool.Release(model)

	// Make validation fail
	factory.validateFail = true

	// Acquire again - should get new model after validation fails
	model2, err := pool.Acquire(context.Background())
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	pool.Release(model2)

	// Check that old model was destroyed
	stats := pool.Stats()
	if stats.TotalDestroyed < 1 {
		t.Error("Expected at least 1 destroyed model")
	}
}

func TestStats(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 1,
		MaxSize: 3,
	}, factory)
	defer pool.Close()

	stats := pool.Stats()
	if stats.TotalCreated != 1 {
		t.Errorf("Expected 1 created, got %d", stats.TotalCreated)
	}

	model, _ := pool.Acquire(context.Background())
	stats = pool.Stats()
	if stats.WaitCount != 1 {
		t.Errorf("Expected wait count 1, got %d", stats.WaitCount)
	}

	pool.Release(model)
}

func TestReleaseAfterClose(t *testing.T) {
	factory := &mockFactory{}
	pool, _ := NewModelPool(PoolConfig{
		MinSize: 1,
		MaxSize: 3,
	}, factory)

	model, _ := pool.Acquire(context.Background())
	pool.Close()

	// Should not panic
	pool.Release(model)

	stats := pool.Stats()
	if stats.TotalDestroyed < 1 {
		t.Error("Expected at least 1 destroyed model")
	}
}

func TestDefaultConfigValues(t *testing.T) {
	factory := &mockFactory{}
	pool, err := NewModelPool(PoolConfig{
		// All defaults
	}, factory)

	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()

	// Should use default MaxSize of 10
	models := make([]Model, 0)
	for i := 0; i < 5; i++ {
		m, err := pool.Acquire(context.Background())
		if err != nil {
			t.Fatalf("Acquire failed: %v", err)
		}
		models = append(models, m)
	}

	stats := pool.Stats()
	if stats.CurrentSize != 5 {
		t.Errorf("Expected pool size 5, got %d", stats.CurrentSize)
	}

	for _, m := range models {
		pool.Release(m)
	}
}
`,

  hint1: `Use a slice for idle models (LIFO for cache locality) and a map for in-use tracking. sync.Cond is perfect for waiting when pool is exhausted.`,

  hint2: `Wrap models with metadata (createdAt, lastUsedAt) for lifecycle management. Run a background goroutine to periodically clean up expired models.`,

  whyItMatters: `Model pools prevent expensive model loading on every request. They manage GPU memory efficiently by limiting concurrent model instances. Pre-warming ensures low latency for initial requests. Connection pooling patterns from database drivers apply directly to ML model management.`,

  translations: {
    ru: {
      title: 'Пул Соединений для Моделей',
      description: `
## Пул Соединений для Моделей

Реализуйте пул соединений для экземпляров ML-моделей для эффективного управления GPU/CPU ресурсами и обработки параллельных запросов инференса.

### Требования

1. **ModelPool** - Управление пулом:
   - \`NewModelPool(config PoolConfig, factory ModelFactory)\` - Создание пула
   - \`Acquire(ctx context.Context) (Model, error)\` - Получение модели из пула
   - \`Release(model Model)\` - Возврат модели в пул
   - \`Close()\` - Остановка пула и освобождение ресурсов

2. **PoolConfig** - Конфигурация:
   - \`MinSize int\` - Минимальный размер пула (прогретый)
   - \`MaxSize int\` - Максимальный размер пула
   - \`MaxIdleTime time.Duration\` - Тайм-аут простоя
   - \`MaxLifetime time.Duration\` - Максимальное время жизни модели
   - \`AcquireTimeout time.Duration\` - Максимальное время ожидания

3. **ModelFactory** - Создание экземпляров модели:
   - \`Create() (Model, error)\` - Создание новой модели
   - \`Validate(model Model) error\` - Проверка здоровья модели
   - \`Destroy(model Model) error\` - Очистка модели

4. **Возможности пула**:
   - Прогрев: создание MinSize моделей при запуске
   - Ленивое создание: создание до MaxSize по требованию
   - Проверка здоровья: валидация моделей перед использованием
   - Автоматическая очистка: удаление простаивающих/истекших моделей
   - Метрики: размер пула, время ожидания, утилизация

### Пример

\`\`\`go
factory := &ONNXModelFactory{modelPath: "model.onnx"}

pool := NewModelPool(PoolConfig{
    MinSize:        2,
    MaxSize:        10,
    MaxIdleTime:    5 * time.Minute,
    AcquireTimeout: 30 * time.Second,
}, factory)

model, err := pool.Acquire(ctx)
if err != nil {
    return err
}
defer pool.Release(model)

result := model.Predict(features)
\`\`\`
`,
      hint1: 'Используйте слайс для idle моделей (LIFO для локальности кэша) и map для отслеживания используемых. sync.Cond идеален для ожидания при исчерпании пула.',
      hint2: 'Оберните модели метаданными (createdAt, lastUsedAt) для управления жизненным циклом. Запустите фоновую горутину для периодической очистки истекших моделей.',
      whyItMatters: 'Пулы моделей предотвращают дорогую загрузку модели на каждый запрос. Они эффективно управляют памятью GPU, ограничивая количество параллельных экземпляров. Прогрев обеспечивает низкую латентность для начальных запросов. Паттерны connection pooling из драйверов БД применимы к управлению ML-моделями.',
    },
    uz: {
      title: 'Model Connection Pool',
      description: `
## Model Connection Pool

GPU/CPU resurslarini samarali boshqarish va parallel inference so'rovlarini qayta ishlash uchun ML model nusxalari uchun connection pool yarating.

### Talablar

1. **ModelPool** - Pool boshqaruvi:
   - \`NewModelPool(config PoolConfig, factory ModelFactory)\` - Pool yaratish
   - \`Acquire(ctx context.Context) (Model, error)\` - Pooldan model olish
   - \`Release(model Model)\` - Modelni poolga qaytarish
   - \`Close()\` - Poolni to'xtatish va resurslarni ozod qilish

2. **PoolConfig** - Konfiguratsiya:
   - \`MinSize int\` - Minimal pool hajmi (oldindan isitilgan)
   - \`MaxSize int\` - Maksimal pool hajmi
   - \`MaxIdleTime time.Duration\` - Bo'sh turish timeout
   - \`MaxLifetime time.Duration\` - Maksimal model umri
   - \`AcquireTimeout time.Duration\` - Maksimal kutish vaqti

3. **ModelFactory** - Model nusxalarini yaratish:
   - \`Create() (Model, error)\` - Yangi model yaratish
   - \`Validate(model Model) error\` - Model sog'ligini tekshirish
   - \`Destroy(model Model) error\` - Modelni tozalash

4. **Pool xususiyatlari**:
   - Oldindan isitish: ishga tushirishda MinSize model yaratish
   - Lazy yaratish: talab bo'yicha MaxSize gacha yaratish
   - Sog'lik tekshiruvi: foydalanishdan oldin modellarni tekshirish
   - Avtomatik tozalash: bo'sh/muddati o'tgan modellarni o'chirish
   - Metrikalar: pool hajmi, kutish vaqtlari, foydalanish

### Misol

\`\`\`go
factory := &ONNXModelFactory{modelPath: "model.onnx"}

pool := NewModelPool(PoolConfig{
    MinSize:        2,
    MaxSize:        10,
    MaxIdleTime:    5 * time.Minute,
    AcquireTimeout: 30 * time.Second,
}, factory)

model, err := pool.Acquire(ctx)
if err != nil {
    return err
}
defer pool.Release(model)

result := model.Predict(features)
\`\`\`
`,
      hint1: "Idle modellar uchun slice (kesh lokaliti uchun LIFO) va foydalanilayotganlarni kuzatish uchun map ishlating. Pool tugaganda kutish uchun sync.Cond ideal.",
      hint2: "Modellarni hayot sikli boshqaruvi uchun metadata (createdAt, lastUsedAt) bilan o'rang. Muddati o'tgan modellarni davriy tozalash uchun fon goroutineni ishga tushiring.",
      whyItMatters: "Model poollari har bir so'rovda qimmat model yuklashni oldini oladi. Ular parallel nusxalarni cheklash orqali GPU xotirasini samarali boshqaradi. Oldindan isitish dastlabki so'rovlar uchun past kechikishni ta'minlaydi. DB drayverlaridagi connection pooling patternlari ML model boshqaruviga to'g'ridan-to'g'ri qo'llaniladi.",
    },
  },
};

export default task;
