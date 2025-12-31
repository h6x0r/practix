import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-concurrency-heartbeat',
	title: 'Heartbeat',
	difficulty: 'hard',	tags: ['go', 'concurrency', 'context', 'monitoring', 'ticker'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Heartbeat** that periodically calls send function until context is canceled, useful for health monitoring and keepalive signals.

**Requirements:**
1. Create function \`Heartbeat(ctx context.Context, interval time.Duration, send func()) error\`
2. Handle nil context (use Background)
3. Handle interval <= 0 (use 1 second default)
4. Create ticker for periodic sends
5. Call send() immediately once before loop
6. Call send() on each tick
7. Stop ticker on exit using defer
8. Return nil when context is canceled (not an error)
9. Use select to wait for ctx.Done() or ticker

**Example:**
\`\`\`go
beats := 0
ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
defer cancel()

err := Heartbeat(ctx, 100*time.Millisecond, func() {
    beats++
    fmt.Println("beat")
})
// err = nil, beats = 3 (immediate + 2 ticks)

// Zero interval defaults to 1 second
err = Heartbeat(ctx, 0, func() {
    sendHealthCheck()
})

// Nil context uses Background
err = Heartbeat(nil, time.Second, func() {
    keepAlive()
})
\`\`\`

**Constraints:**
- Must use time.NewTicker
- Must call send() immediately once
- Must defer ticker.Stop()
- Must return nil on cancellation`,
	initialCode: `package concurrency

import (
	"context"
	"time"
)

// TODO: Implement Heartbeat
func Heartbeat(ctx context.Context, interval time.Duration, send func()) error {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
	"time"
)

func Heartbeat(ctx context.Context, interval time.Duration, send func()) error {
	if ctx == nil {                                             // Handle nil context
		ctx = context.Background()                          // Use Background as fallback
	}
	if interval <= 0 {                                          // Invalid interval
		interval = time.Second                              // Default to 1 second
	}
	ticker := time.NewTicker(interval)                          // Create ticker
	defer ticker.Stop()                                         // Always stop ticker on exit
	send()                                                      // Send immediately once
	for {                                                       // Loop until canceled
		select {
		case <-ctx.Done():                                  // Context canceled
			return nil                                  // Return success (not error)
		case <-ticker.C:                                    // Tick received
			send()                                      // Send heartbeat
		}
	}
}`,
			hint1: `Create time.NewTicker and defer ticker.Stop(). Call send() once before entering the for loop.`,
			hint2: `Use infinite for loop with select between ctx.Done() and ticker.C. Return nil (not error) when context is canceled.`,
			whyItMatters: `Heartbeat enables liveness monitoring and keepalive mechanisms, essential for detecting failures and maintaining connections in distributed systems.

**Why Heartbeat:**
- **Liveness Detection:** Prove process is still alive
- **Connection Keepalive:** Prevent connection timeouts
- **Health Monitoring:** Regular health check signals
- **Coordination:** Synchronize distributed components
- **Failure Detection:** Detect when service stops sending

**Production Pattern:**
\`\`\`go
// Service health heartbeat
func ServiceHealthHeartbeat(ctx context.Context, registry *Registry, serviceID string) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        registry.UpdateHealth(serviceID, Healthy)
        log.Printf("Service %s: heartbeat sent", serviceID)
    })
}

// WebSocket keepalive
func WebSocketKeepalive(ctx context.Context, conn *websocket.Conn) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        conn.WriteMessage(websocket.PingMessage, []byte{})
    })
}

// Leader election heartbeat
type Leader struct {
    id string
    store *LeaderStore
}

func (l *Leader) MaintainLeadership(ctx context.Context) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        l.store.RenewLease(l.id)
    })
}

// Database connection keepalive
func KeepDatabaseAlive(ctx context.Context, db *sql.DB) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        if err := db.Ping(); err != nil {
            log.Printf("Database ping failed: %v", err)
        }
    })
}

// Distributed lock renewal
func RenewDistributedLock(ctx context.Context, lock *Lock) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        if err := lock.Renew(); err != nil {
            log.Printf("Lock renewal failed: %v", err)
        }
    })
}

// Metrics reporting
func ReportMetrics(ctx context.Context, reporter *MetricsReporter) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        metrics := collectMetrics()
        reporter.Send(metrics)
    })
}

// Session keepalive
func MaintainSession(ctx context.Context, session *Session) error {
    return Heartbeat(ctx, 5*time.Minute, func() {
        session.Touch()
    })
}

// Cache warmup refresh
func RefreshCache(ctx context.Context, cache *Cache) error {
    return Heartbeat(ctx, 10*time.Minute, func() {
        cache.Refresh()
    })
}

// Progress reporting
func ReportProgress(ctx context.Context, job *Job) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        progress := job.GetProgress()
        job.ReportProgress(progress)
    })
}

// Monitoring system heartbeat
type Monitor struct {
    endpoint string
    hostname string
}

func (m *Monitor) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        status := SystemStatus{
            Hostname:  m.hostname,
            Timestamp: time.Now(),
            Status:    "alive",
        }
        sendToMonitoring(m.endpoint, status)
    })
}

// Worker registration renewal
func MaintainWorkerRegistration(ctx context.Context, workerID string, registry *WorkerRegistry) error {
    return Heartbeat(ctx, 15*time.Second, func() {
        registry.Renew(workerID)
    })
}

// Distributed consensus heartbeat
func ConsensusHeartbeat(ctx context.Context, node *ConsensusNode) error {
    return Heartbeat(ctx, 1*time.Second, func() {
        node.SendHeartbeat()
    })
}

// Load balancer health check
func LoadBalancerHeartbeat(ctx context.Context, lb *LoadBalancer, serverID string) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        lb.MarkHealthy(serverID)
    })
}

// Scheduled task coordinator
func CoordinatorHeartbeat(ctx context.Context, coordinator *TaskCoordinator) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        coordinator.AnnounceAlive()
    })
}

// Client connection heartbeat
type Client struct {
    conn net.Conn
}

func (c *Client) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 20*time.Second, func() {
        c.conn.Write([]byte("PING\n"))
    })
}

// Service mesh sidecar heartbeat
func SidecarHeartbeat(ctx context.Context, mesh *ServiceMesh, serviceID string) error {
    return Heartbeat(ctx, 3*time.Second, func() {
        mesh.UpdateService(serviceID, ServiceInfo{
            LastSeen: time.Now(),
            Status:   "healthy",
        })
    })
}
\`\`\`

**Real-World Benefits:**
- **Failure Detection:** Know immediately when service stops
- **Connection Stability:** Prevent idle connection drops
- **Distributed Coordination:** Synchronize system state
- **Observability:** Track service liveness over time

**Common Use Cases:**
- **Service Discovery:** Register presence periodically
- **Health Checks:** Send regular health signals
- **Connection Keepalive:** Prevent TCP/WebSocket timeouts
- **Leader Election:** Maintain leadership claim
- **Distributed Locks:** Renew lock ownership
- **Session Management:** Keep sessions alive
- **Metrics Collection:** Report metrics periodically
- **Progress Tracking:** Update job progress

**Heartbeat Intervals:**
- **Critical Systems:** 1-5 seconds (fast failure detection)
- **Service Health:** 10-30 seconds (balanced)
- **Connection Keepalive:** 30-60 seconds (prevent timeout)
- **Session Management:** 5-15 minutes (reduce overhead)
- **Metrics Reporting:** 1-5 minutes (aggregated data)

**Best Practices:**
- **Immediate First Beat:** Call send() before loop starts
- **Graceful Shutdown:** Return nil on cancellation
- **Error Handling:** Log but don't fail on send errors
- **Monitoring:** Track heartbeat failures
- **Backpressure:** Don't let failed sends block ticker

**Failure Detection:**
- **Missed Heartbeats:** 3-5 consecutive misses = failure
- **Timeout:** 2-3x heartbeat interval
- **Recovery:** Resume heartbeats when service recovers

**Anti-Patterns:**
- **No Cleanup:** Always defer ticker.Stop()
- **Blocking Send:** Send should be fast, delegate slow work
- **Ignoring Context:** Always respect cancellation
- **Too Frequent:** Balance detection speed vs overhead

Without Heartbeat, implementing periodic health signals with proper cancellation and cleanup requires repetitive boilerplate in every service.`,
	testCode: `package concurrency

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	err := Heartbeat(ctx, 50*time.Millisecond, func() { atomic.AddInt64(&count, 1) })
	if err != nil { t.Errorf("expected nil, got %v", err) }
	if count < 3 { t.Errorf("expected at least 3 beats, got %d", count) }
}

func Test2(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, 20*time.Millisecond, func() { atomic.AddInt64(&count, 1) })
	if count < 1 { t.Error("expected at least 1 beat (immediate)") }
}

func Test3(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_ = Heartbeat(nil, 30*time.Millisecond, func() { atomic.AddInt64(&count, 1) })
}

func Test4(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, 0, func() { atomic.AddInt64(&count, 1) })
	if count < 1 { t.Error("expected at least 1 beat with zero interval") }
}

func Test5(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, -1*time.Second, func() { atomic.AddInt64(&count, 1) })
	if count < 1 { t.Error("expected at least 1 beat with negative interval") }
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := Heartbeat(ctx, 100*time.Millisecond, func() {})
	if err != nil { t.Errorf("expected nil, got %v", err) }
}

func Test7(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 70*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, 20*time.Millisecond, func() { atomic.AddInt64(&count, 1) })
	if count < 2 { t.Errorf("expected at least 2 beats, got %d", count) }
}

func Test8(t *testing.T) {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, 10*time.Millisecond, func() {})
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond { t.Error("should run for at least 40ms") }
}

func Test9(t *testing.T) {
	var firstCall int64 = -1
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_ = Heartbeat(ctx, 50*time.Millisecond, func() {
		if atomic.LoadInt64(&firstCall) < 0 {
			atomic.StoreInt64(&firstCall, time.Since(start).Nanoseconds())
		}
	})
	if firstCall > 20*time.Millisecond.Nanoseconds() { t.Error("first beat should be immediate") }
}

func Test10(t *testing.T) {
	var count int64
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_ = Heartbeat(ctx, 30*time.Millisecond, func() { atomic.AddInt64(&count, 1) })
	if count < 5 { t.Errorf("expected at least 5 beats, got %d", count) }
}
`,
	order: 9,
	translations: {
		ru: {
			title: 'Механизм heartbeat',
			description: `Реализуйте **Heartbeat**, который периодически вызывает функцию send пока контекст не отменён, полезен для мониторинга здоровья и keepalive сигналов.

**Требования:**
1. Создайте функцию \`Heartbeat(ctx context.Context, interval time.Duration, send func()) error\`
2. Обработайте nil context (используйте Background)
3. Обработайте interval <= 0 (используйте 1 секунду по умолчанию)
4. Создайте ticker для периодических отправок
5. Вызовите send() сразу один раз перед циклом
6. Вызывайте send() на каждом тике
7. Остановите ticker при выходе используя defer
8. Верните nil когда контекст отменён (не ошибку)
9. Используйте select для ожидания ctx.Done() или ticker

**Пример:**
\`\`\`go
beats := 0
ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
defer cancel()

err := Heartbeat(ctx, 100*time.Millisecond, func() {
    beats++
    fmt.Println("beat")
})
// err = nil, beats = 3 (немедленный + 2 тика)

// Нулевой interval по умолчанию 1 секунда
err = Heartbeat(ctx, 0, func() {
    sendHealthCheck()
})

// Nil context использует Background
err = Heartbeat(nil, time.Second, func() {
    keepAlive()
})
\`\`\`

**Ограничения:**
- Должен использовать time.NewTicker
- Должен вызвать send() сразу один раз
- Должен использовать defer ticker.Stop()
- Должен возвращать nil при отмене`,
			hint1: `Создайте time.NewTicker и defer ticker.Stop(). Вызовите send() один раз перед входом в for цикл.`,
			hint2: `Используйте бесконечный for цикл с select между ctx.Done() и ticker.C. Верните nil (не ошибку) когда контекст отменён.`,
			whyItMatters: `Heartbeat позволяет мониторинг liveness и механизмы keepalive, необходим для обнаружения отказов и поддержания соединений в распределённых системах.

**Почему Heartbeat:**
- **Обнаружение Liveness:** Доказать что процесс всё ещё жив
- **Connection Keepalive:** Предотвратить таймауты соединений
- **Мониторинг здоровья:** Регулярные сигналы health check
- **Координация:** Синхронизация распределённых компонентов
- **Обнаружение отказов:** Обнаружить когда сервис перестаёт отправлять

**Production Pattern:**
\`\`\`go
// Service health heartbeat
func ServiceHealthHeartbeat(ctx context.Context, registry *Registry, serviceID string) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        registry.UpdateHealth(serviceID, Healthy)
        log.Printf("Service %s: heartbeat sent", serviceID)
    })
}

// WebSocket keepalive
func WebSocketKeepalive(ctx context.Context, conn *websocket.Conn) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        conn.WriteMessage(websocket.PingMessage, []byte{})
    })
}

// Leader election heartbeat
type Leader struct {
    id string
    store *LeaderStore
}

func (l *Leader) MaintainLeadership(ctx context.Context) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        l.store.RenewLease(l.id)
    })
}

// Database connection keepalive
func KeepDatabaseAlive(ctx context.Context, db *sql.DB) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        if err := db.Ping(); err != nil {
            log.Printf("Database ping failed: %v", err)
        }
    })
}

// Distributed lock renewal
func RenewDistributedLock(ctx context.Context, lock *Lock) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        if err := lock.Renew(); err != nil {
            log.Printf("Lock renewal failed: %v", err)
        }
    })
}

// Metrics reporting
func ReportMetrics(ctx context.Context, reporter *MetricsReporter) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        metrics := collectMetrics()
        reporter.Send(metrics)
    })
}

// Session keepalive
func MaintainSession(ctx context.Context, session *Session) error {
    return Heartbeat(ctx, 5*time.Minute, func() {
        session.Touch()
    })
}

// Cache warmup refresh
func RefreshCache(ctx context.Context, cache *Cache) error {
    return Heartbeat(ctx, 10*time.Minute, func() {
        cache.Refresh()
    })
}

// Progress reporting
func ReportProgress(ctx context.Context, job *Job) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        progress := job.GetProgress()
        job.ReportProgress(progress)
    })
}

// Monitoring system heartbeat
type Monitor struct {
    endpoint string
    hostname string
}

func (m *Monitor) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        status := SystemStatus{
            Hostname:  m.hostname,
            Timestamp: time.Now(),
            Status:    "alive",
        }
        sendToMonitoring(m.endpoint, status)
    })
}

// Worker registration renewal
func MaintainWorkerRegistration(ctx context.Context, workerID string, registry *WorkerRegistry) error {
    return Heartbeat(ctx, 15*time.Second, func() {
        registry.Renew(workerID)
    })
}

// Distributed consensus heartbeat
func ConsensusHeartbeat(ctx context.Context, node *ConsensusNode) error {
    return Heartbeat(ctx, 1*time.Second, func() {
        node.SendHeartbeat()
    })
}

// Load balancer health check
func LoadBalancerHeartbeat(ctx context.Context, lb *LoadBalancer, serverID string) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        lb.MarkHealthy(serverID)
    })
}

// Client connection heartbeat
type Client struct {
    conn net.Conn
}

func (c *Client) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 20*time.Second, func() {
        c.conn.Write([]byte("PING\\n"))
    })
}

// Service mesh sidecar heartbeat
func SidecarHeartbeat(ctx context.Context, mesh *ServiceMesh, serviceID string) error {
    return Heartbeat(ctx, 3*time.Second, func() {
        mesh.UpdateService(serviceID, ServiceInfo{
            LastSeen: time.Now(),
            Status:   "healthy",
        })
    })
}
\`\`\`

**Практические преимущества:**
- **Обнаружение отказов:** Мгновенно узнаёте когда сервис останавливается
- **Стабильность соединений:** Предотвращение разрыва idle соединений
- **Распределённая координация:** Синхронизация состояния системы
- **Наблюдаемость:** Отслеживание liveness сервиса во времени

**Типичные сценарии использования:**
- **Service Discovery:** Периодическая регистрация присутствия
- **Health Checks:** Отправка регулярных сигналов здоровья
- **Connection Keepalive:** Предотвращение таймаутов TCP/WebSocket
- **Leader Election:** Поддержание заявки на лидерство
- **Distributed Locks:** Обновление владения блокировкой
- **Session Management:** Поддержание сессий активными
- **Metrics Collection:** Периодический отчёт метрик
- **Progress Tracking:** Обновление прогресса задач

**Интервалы Heartbeat:**
- **Критические системы:** 1-5 секунд (быстрое обнаружение отказов)
- **Service Health:** 10-30 секунд (сбалансированно)
- **Connection Keepalive:** 30-60 секунд (предотвращение таймаутов)
- **Session Management:** 5-15 минут (снижение накладных расходов)
- **Metrics Reporting:** 1-5 минут (агрегированные данные)

**Лучшие практики:**
- **Немедленный первый бит:** Вызов send() до начала цикла
- **Graceful Shutdown:** Возврат nil при отмене
- **Обработка ошибок:** Логирование, но не падение при ошибках отправки
- **Мониторинг:** Отслеживание пропущенных heartbeat
- **Backpressure:** Не блокировать ticker при неудачных отправках

**Обнаружение отказов:**
- **Пропущенные Heartbeat:** 3-5 пропусков подряд = отказ
- **Таймаут:** 2-3x интервала heartbeat
- **Восстановление:** Возобновление heartbeat при восстановлении сервиса

**Антипаттерны:**
- **Отсутствие очистки:** Всегда defer ticker.Stop()
- **Блокирующий Send:** Send должен быть быстрым, делегируйте медленную работу
- **Игнорирование Context:** Всегда уважайте отмену
- **Слишком частый:** Баланс между скоростью обнаружения и накладными расходами

Без Heartbeat реализация периодических сигналов здоровья с правильной отменой и очисткой требует повторяющегося boilerplate в каждом сервисе.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func Heartbeat(ctx context.Context, interval time.Duration, send func()) error {
	if ctx == nil {                                             // Обработка nil контекста
		ctx = context.Background()                          // Используем Background как fallback
	}
	if interval <= 0 {                                          // Неверный интервал
		interval = time.Second                              // По умолчанию 1 секунда
	}
	ticker := time.NewTicker(interval)                          // Создаём тикер
	defer ticker.Stop()                                         // Всегда останавливаем тикер при выходе
	send()                                                      // Отправляем сразу один раз
	for {                                                       // Цикл до отмены
		select {
		case <-ctx.Done():                                  // Контекст отменён
			return nil                                  // Возвращаем успех (не ошибку)
		case <-ticker.C:                                    // Получен тик
			send()                                      // Отправляем heartbeat
		}
	}
}`
		},
		uz: {
			title: 'Heartbeat mexanizmi',
			description: `Kontekst bekor qilinmaguncha davriy ravishda send funksiyasini chaqiradigan **Heartbeat** ni amalga oshiring, health monitoring va keepalive signallari uchun foydali.

**Talablar:**
1. \`Heartbeat(ctx context.Context, interval time.Duration, send func()) error\` funksiyasini yarating
2. nil kontekstni ishlang (Background dan foydalaning)
3. interval <= 0 ni ishlang (sukut bo'yicha 1 soniya ishlatiladi)
4. Davriy yuborish uchun ticker yarating
5. Sikl oldidan darhol bir marta send() ni chaqiring
6. Har bir tickda send() ni chaqiring
7. Chiqishda defer dan foydalanib tickerni to'xtating
8. Kontekst bekor qilinganda nil qaytaring (xato emas)
9. ctx.Done() yoki tickerni kutish uchun select dan foydalaning

**Misol:**
\`\`\`go
beats := 0
ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
defer cancel()

err := Heartbeat(ctx, 100*time.Millisecond, func() {
    beats++
    fmt.Println("beat")
})
// err = nil, beats = 3 (darhol + 2 tick)

// Nol interval sukut bo'yicha 1 soniya
err = Heartbeat(ctx, 0, func() {
    sendHealthCheck()
})

// Nil kontekst Background dan foydalanadi
err = Heartbeat(nil, time.Second, func() {
    keepAlive()
})
\`\`\`

**Cheklovlar:**
- time.NewTicker dan foydalanishi kerak
- Darhol bir marta send() ni chaqirishi kerak
- defer ticker.Stop() dan foydalanishi kerak
- Bekor qilinganda nil qaytarishi kerak`,
			hint1: `time.NewTicker yarating va defer ticker.Stop() dan foydalaning. for siklga kirishdan oldin bir marta send() ni chaqiring.`,
			hint2: `ctx.Done() va ticker.C o'rtasida select bilan cheksiz for siklidan foydalaning. Kontekst bekor qilinganda nil (xato emas) qaytaring.`,
			whyItMatters: `Heartbeat liveness monitoringini va keepalive mexanizmlarini yoqadi, taqsimlangan tizimlarda nosozliklarni aniqlash va ulanishlarni saqlash uchun zarur.

**Nima uchun Heartbeat:**
- **Liveness aniqlash:** Jarayon hali ham tirikligini isbotlash
- **Connection Keepalive:** Ulanish timeoutlarini oldini olish
- **Health monitoring:** Muntazam health check signallari
- **Koordinatsiya:** Taqsimlangan komponentlarni sinxronlashtirish
- **Nosozliklarni aniqlash:** Xizmat yuborishni to'xtatganda aniqlash

**Production Pattern:**
\`\`\`go
// Service health heartbeat
func ServiceHealthHeartbeat(ctx context.Context, registry *Registry, serviceID string) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        registry.UpdateHealth(serviceID, Healthy)
        log.Printf("Service %s: heartbeat sent", serviceID)
    })
}

// WebSocket keepalive
func WebSocketKeepalive(ctx context.Context, conn *websocket.Conn) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        conn.WriteMessage(websocket.PingMessage, []byte{})
    })
}

// Leader election heartbeat
type Leader struct {
    id string
    store *LeaderStore
}

func (l *Leader) MaintainLeadership(ctx context.Context) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        l.store.RenewLease(l.id)
    })
}

// Database connection keepalive
func KeepDatabaseAlive(ctx context.Context, db *sql.DB) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        if err := db.Ping(); err != nil {
            log.Printf("Database ping failed: %v", err)
        }
    })
}

// Distributed lock yangilash
func RenewDistributedLock(ctx context.Context, lock *Lock) error {
    return Heartbeat(ctx, 10*time.Second, func() {
        if err := lock.Renew(); err != nil {
            log.Printf("Lock renewal failed: %v", err)
        }
    })
}

// Metrikalar hisoboti
func ReportMetrics(ctx context.Context, reporter *MetricsReporter) error {
    return Heartbeat(ctx, 1*time.Minute, func() {
        metrics := collectMetrics()
        reporter.Send(metrics)
    })
}

// Sessiya keepalive
func MaintainSession(ctx context.Context, session *Session) error {
    return Heartbeat(ctx, 5*time.Minute, func() {
        session.Touch()
    })
}

// Kesh yangilash
func RefreshCache(ctx context.Context, cache *Cache) error {
    return Heartbeat(ctx, 10*time.Minute, func() {
        cache.Refresh()
    })
}

// Progress hisoboti
func ReportProgress(ctx context.Context, job *Job) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        progress := job.GetProgress()
        job.ReportProgress(progress)
    })
}

// Monitoring tizimi heartbeat
type Monitor struct {
    endpoint string
    hostname string
}

func (m *Monitor) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 30*time.Second, func() {
        status := SystemStatus{
            Hostname:  m.hostname,
            Timestamp: time.Now(),
            Status:    "alive",
        }
        sendToMonitoring(m.endpoint, status)
    })
}

// Worker ro'yxatdan o'tishni yangilash
func MaintainWorkerRegistration(ctx context.Context, workerID string, registry *WorkerRegistry) error {
    return Heartbeat(ctx, 15*time.Second, func() {
        registry.Renew(workerID)
    })
}

// Taqsimlangan konsensus heartbeat
func ConsensusHeartbeat(ctx context.Context, node *ConsensusNode) error {
    return Heartbeat(ctx, 1*time.Second, func() {
        node.SendHeartbeat()
    })
}

// Load balancer health check
func LoadBalancerHeartbeat(ctx context.Context, lb *LoadBalancer, serverID string) error {
    return Heartbeat(ctx, 5*time.Second, func() {
        lb.MarkHealthy(serverID)
    })
}

// Klient ulanishi heartbeat
type Client struct {
    conn net.Conn
}

func (c *Client) SendHeartbeats(ctx context.Context) error {
    return Heartbeat(ctx, 20*time.Second, func() {
        c.conn.Write([]byte("PING\\n"))
    })
}

// Service mesh sidecar heartbeat
func SidecarHeartbeat(ctx context.Context, mesh *ServiceMesh, serviceID string) error {
    return Heartbeat(ctx, 3*time.Second, func() {
        mesh.UpdateService(serviceID, ServiceInfo{
            LastSeen: time.Now(),
            Status:   "healthy",
        })
    })
}
\`\`\`

**Haqiqiy dunyo foydalari:**
- **Nosozliklarni aniqlash:** Xizmat to'xtaganda darhol bilasiz
- **Ulanish barqarorligi:** Idle ulanishlar uzilishini oldini olish
- **Taqsimlangan koordinatsiya:** Tizim holatini sinxronlashtirish
- **Kuzatuvchanlik:** Xizmat livnessini vaqt o'tishi bilan kuzatish

**Odatiy foydalanish holatlari:**
- **Service Discovery:** Mavjudlikni davriy ro'yxatdan o'tkazish
- **Health Checks:** Muntazam salomatlik signallari yuborish
- **Connection Keepalive:** TCP/WebSocket timeoutlarini oldini olish
- **Leader Election:** Liderlik da'vosini saqlash
- **Distributed Locks:** Qulf egaligini yangilash
- **Session Management:** Sessiyalarni faol saqlash
- **Metrics Collection:** Metrikalarni davriy hisobot qilish
- **Progress Tracking:** Vazifa progressini yangilash

**Heartbeat intervallari:**
- **Kritik tizimlar:** 1-5 soniya (tez nosozliklarni aniqlash)
- **Service Health:** 10-30 soniya (muvozanatlangan)
- **Connection Keepalive:** 30-60 soniya (timeoutlarni oldini olish)
- **Session Management:** 5-15 daqiqa (overheadni kamaytirish)
- **Metrics Reporting:** 1-5 daqiqa (agregatlangan ma'lumotlar)

**Eng yaxshi amaliyotlar:**
- **Darhol birinchi beat:** Sikl boshlanishidan oldin send() ni chaqirish
- **Graceful Shutdown:** Bekor qilishda nil qaytarish
- **Xatolarni ishlash:** Yuborish xatolarida log yozish, lekin tushmang
- **Monitoring:** Yo'qolgan heartbeatlarni kuzatish
- **Backpressure:** Muvaffaqiyatsiz yuborishlarda tickerni bloklamang

**Nosozliklarni aniqlash:**
- **O'tkazib yuborilgan Heartbeatlar:** Ketma-ket 3-5 ta o'tkazib yuborish = nosozlik
- **Timeout:** Heartbeat intervalining 2-3x
- **Qayta tiklash:** Xizmat qayta tiklanganda heartbeatlarni davom ettirish

**Antipatternlar:**
- **Tozalash yo'q:** Har doim defer ticker.Stop()
- **Blokirovka qiluvchi Send:** Send tez bo'lishi kerak, sekin ishni delegat qiling
- **Contextni e'tiborsiz qoldirish:** Har doim bekor qilishni hurmat qiling
- **Juda tez-tez:** Aniqlash tezligi va overhead o'rtasidagi muvozanat

Heartbeatsiz to'g'ri bekor qilish va tozalash bilan davriy salomatlik signallarini amalga oshirish har bir xizmatda takrorlanuvchi boilerplate talab qiladi.`,
			solutionCode: `package concurrency

import (
	"context"
	"time"
)

func Heartbeat(ctx context.Context, interval time.Duration, send func()) error {
	if ctx == nil {                                             // nil kontekstni ishlash
		ctx = context.Background()                          // Fallback sifatida Background ishlatamiz
	}
	if interval <= 0 {                                          // Noto'g'ri interval
		interval = time.Second                              // Sukut bo'yicha 1 soniya
	}
	ticker := time.NewTicker(interval)                          // Ticker yaratamiz
	defer ticker.Stop()                                         // Chiqishda har doim tickerni to'xtatamiz
	send()                                                      // Darhol bir marta yuboramiz
	for {                                                       // Bekor qilinguncha sikl
		select {
		case <-ctx.Done():                                  // Kontekst bekor qilindi
			return nil                                  // Muvaffaqiyatni qaytaramiz (xato emas)
		case <-ticker.C:                                    // Tick olindi
			send()                                      // Heartbeat yuboramiz
		}
	}
}`
		}
	}
};

export default task;
