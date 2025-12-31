import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-graceful-shutdown',
	title: 'Graceful Shutdown',
	difficulty: 'medium',
	tags: ['go', 'ml', 'deployment', 'shutdown'],
	estimatedTime: '25m',
	isPremium: true,
	order: 1,
	description: `# Graceful Shutdown

Implement graceful shutdown for ML inference servers.

## Task

Build a graceful shutdown handler that:
- Catches shutdown signals (SIGTERM, SIGINT)
- Stops accepting new requests
- Waits for in-flight requests to complete
- Releases resources properly

## Example

\`\`\`go
server := NewInferenceServer(model)
server.Start(":8080")
// On SIGTERM: stops accepting, waits for requests, exits
\`\`\``,

	initialCode: `package main

import (
	"fmt"
	"net/http"
)

// GracefulServer implements graceful shutdown
type GracefulServer struct {
	// Your fields here
}

// NewGracefulServer creates a graceful server
func NewGracefulServer(handler http.Handler) *GracefulServer {
	// Your code here
	return nil
}

// Start starts the server
func (s *GracefulServer) Start(addr string) error {
	// Your code here
	return nil
}

// Shutdown initiates graceful shutdown
func (s *GracefulServer) Shutdown() error {
	// Your code here
	return nil
}

func main() {
	fmt.Println("Graceful Shutdown")
}`,

	solutionCode: `package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// ServerState represents server lifecycle state
type ServerState int32

const (
	StateStarting ServerState = iota
	StateRunning
	StateShuttingDown
	StateStopped
)

// GracefulServer implements graceful shutdown
type GracefulServer struct {
	server          *http.Server
	handler         http.Handler
	state           int32
	shutdownTimeout time.Duration
	drainTimeout    time.Duration

	activeRequests  int64
	shutdownChan    chan struct{}
	doneChan        chan struct{}

	onShutdown []func()
	mu         sync.Mutex
}

// NewGracefulServer creates a graceful server
func NewGracefulServer(handler http.Handler) *GracefulServer {
	return &GracefulServer{
		handler:         handler,
		shutdownTimeout: 30 * time.Second,
		drainTimeout:    5 * time.Second,
		shutdownChan:    make(chan struct{}),
		doneChan:        make(chan struct{}),
	}
}

// SetShutdownTimeout sets the shutdown timeout
func (s *GracefulServer) SetShutdownTimeout(d time.Duration) {
	s.shutdownTimeout = d
}

// SetDrainTimeout sets the connection drain timeout
func (s *GracefulServer) SetDrainTimeout(d time.Duration) {
	s.drainTimeout = d
}

// OnShutdown registers a shutdown callback
func (s *GracefulServer) OnShutdown(fn func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onShutdown = append(s.onShutdown, fn)
}

// requestTracker wraps handler to track active requests
func (s *GracefulServer) requestTracker(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if shutting down
		if atomic.LoadInt32(&s.state) >= int32(StateShuttingDown) {
			http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
			return
		}

		atomic.AddInt64(&s.activeRequests, 1)
		defer atomic.AddInt64(&s.activeRequests, -1)

		next.ServeHTTP(w, r)
	})
}

// Start starts the server
func (s *GracefulServer) Start(addr string) error {
	s.server = &http.Server{
		Addr:    addr,
		Handler: s.requestTracker(s.handler),
	}

	atomic.StoreInt32(&s.state, int32(StateRunning))

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		select {
		case sig := <-sigChan:
			log.Printf("Received signal: %v", sig)
			s.Shutdown()
		case <-s.shutdownChan:
			// Shutdown called directly
		}
	}()

	log.Printf("Starting server on %s", addr)
	err := s.server.ListenAndServe()

	if err == http.ErrServerClosed {
		<-s.doneChan
		return nil
	}

	return err
}

// Shutdown initiates graceful shutdown
func (s *GracefulServer) Shutdown() error {
	// Only shutdown once
	if !atomic.CompareAndSwapInt32(&s.state, int32(StateRunning), int32(StateShuttingDown)) {
		return nil
	}

	log.Println("Starting graceful shutdown...")
	close(s.shutdownChan)

	// Run shutdown callbacks
	s.mu.Lock()
	callbacks := s.onShutdown
	s.mu.Unlock()

	for _, fn := range callbacks {
		fn()
	}

	// Wait for active requests to drain
	drainStart := time.Now()
	for atomic.LoadInt64(&s.activeRequests) > 0 {
		if time.Since(drainStart) > s.drainTimeout {
			log.Printf("Drain timeout, %d requests still active", atomic.LoadInt64(&s.activeRequests))
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), s.shutdownTimeout)
	defer cancel()

	err := s.server.Shutdown(ctx)

	atomic.StoreInt32(&s.state, int32(StateStopped))
	close(s.doneChan)

	log.Println("Server shutdown complete")
	return err
}

// ActiveRequests returns count of active requests
func (s *GracefulServer) ActiveRequests() int64 {
	return atomic.LoadInt64(&s.activeRequests)
}

// State returns current server state
func (s *GracefulServer) State() ServerState {
	return ServerState(atomic.LoadInt32(&s.state))
}

// IsHealthy returns true if server is accepting requests
func (s *GracefulServer) IsHealthy() bool {
	return s.State() == StateRunning
}

// ResourceManager manages server resources
type ResourceManager struct {
	resources []func() error
	mu        sync.Mutex
}

func NewResourceManager() *ResourceManager {
	return &ResourceManager{
		resources: make([]func() error, 0),
	}
}

// Register registers a cleanup function
func (rm *ResourceManager) Register(cleanup func() error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.resources = append(rm.resources, cleanup)
}

// Cleanup runs all cleanup functions
func (rm *ResourceManager) Cleanup() []error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	var errors []error
	// Cleanup in reverse order (LIFO)
	for i := len(rm.resources) - 1; i >= 0; i-- {
		if err := rm.resources[i](); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}

func main() {
	// Create resource manager
	resources := NewResourceManager()

	// Mock model cleanup
	resources.Register(func() error {
		log.Println("Cleaning up model resources...")
		time.Sleep(100 * time.Millisecond)
		return nil
	})

	// Create handler
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond) // Simulate inference
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)
	server.SetShutdownTimeout(10 * time.Second)
	server.SetDrainTimeout(5 * time.Second)

	// Register resource cleanup on shutdown
	server.OnShutdown(func() {
		errors := resources.Cleanup()
		for _, err := range errors {
			log.Printf("Cleanup error: %v", err)
		}
	})

	log.Fatal(server.Start(":8080"))
}`,

	testCode: `package main

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestGracefulServer(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)

	if server.State() != StateStarting {
		t.Error("Initial state should be Starting")
	}
}

func TestRequestTracking(t *testing.T) {
	var requestsHandled int64

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&requestsHandled, 1)
		time.Sleep(50 * time.Millisecond)
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)
	wrappedHandler := server.requestTracker(handler)

	// Start a request
	go func() {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("GET", "/", nil)
		wrappedHandler.ServeHTTP(w, r)
	}()

	// Check active requests
	time.Sleep(10 * time.Millisecond)
	active := server.ActiveRequests()

	if active < 0 || active > 1 {
		t.Errorf("Active requests should be 0 or 1, got %d", active)
	}
}

func TestShutdownCallbacks(t *testing.T) {
	callbackCalled := false

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)
	server.OnShutdown(func() {
		callbackCalled = true
	})

	// Manually set state to running and shutdown
	atomic.StoreInt32(&server.state, int32(StateRunning))
	server.Shutdown()

	if !callbackCalled {
		t.Error("Shutdown callback should be called")
	}
}

func TestIsHealthy(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)

	if server.IsHealthy() {
		t.Error("Should not be healthy before running")
	}

	atomic.StoreInt32(&server.state, int32(StateRunning))

	if !server.IsHealthy() {
		t.Error("Should be healthy when running")
	}

	atomic.StoreInt32(&server.state, int32(StateShuttingDown))

	if server.IsHealthy() {
		t.Error("Should not be healthy when shutting down")
	}
}

func TestResourceManager(t *testing.T) {
	rm := NewResourceManager()

	order := make([]int, 0)
	rm.Register(func() error {
		order = append(order, 1)
		return nil
	})
	rm.Register(func() error {
		order = append(order, 2)
		return nil
	})

	rm.Cleanup()

	// Should cleanup in reverse order (LIFO)
	if len(order) != 2 || order[0] != 2 || order[1] != 1 {
		t.Errorf("Expected [2, 1], got %v", order)
	}
}

func TestRejectDuringShutdown(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)
	wrappedHandler := server.requestTracker(handler)

	// Set state to shutting down
	atomic.StoreInt32(&server.state, int32(StateShuttingDown))

	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)
	wrappedHandler.ServeHTTP(w, r)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("Expected 503 during shutdown, got %d", w.Code)
	}
}

func TestSetTimeouts(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)
	server.SetShutdownTimeout(60 * time.Second)
	server.SetDrainTimeout(10 * time.Second)

	if server.shutdownTimeout != 60*time.Second {
		t.Error("Shutdown timeout not set correctly")
	}
	if server.drainTimeout != 10*time.Second {
		t.Error("Drain timeout not set correctly")
	}
}

func TestServerState(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	server := NewGracefulServer(handler)

	if server.State() != StateStarting {
		t.Error("Initial state should be Starting")
	}

	atomic.StoreInt32(&server.state, int32(StateRunning))
	if server.State() != StateRunning {
		t.Error("State should be Running")
	}

	atomic.StoreInt32(&server.state, int32(StateStopped))
	if server.State() != StateStopped {
		t.Error("State should be Stopped")
	}
}

func TestDoubleShutdown(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	callCount := 0
	server := NewGracefulServer(handler)
	server.OnShutdown(func() {
		callCount++
	})

	atomic.StoreInt32(&server.state, int32(StateRunning))
	server.Shutdown()
	server.Shutdown() // Second call should be ignored

	if callCount != 1 {
		t.Errorf("Shutdown callback should only be called once, got %d", callCount)
	}
}

func TestResourceManagerEmpty(t *testing.T) {
	rm := NewResourceManager()
	errors := rm.Cleanup()

	if len(errors) != 0 {
		t.Error("Empty resource manager should return no errors")
	}
}`,

	hint1: 'Use context.WithTimeout for controlled shutdown',
	hint2: 'Track active requests with atomic counters',

	whyItMatters: `Graceful shutdown ensures reliable deployments:

- **Zero downtime**: Complete in-flight requests before stopping
- **Data integrity**: Properly release resources and flush data
- **Kubernetes compatibility**: Support SIGTERM for pod termination
- **Rolling updates**: Enable seamless service updates

Graceful shutdown is essential for production ML services.`,

	translations: {
		ru: {
			title: 'Graceful shutdown',
			description: `# Graceful shutdown

Реализуйте graceful shutdown для серверов ML инференса.

## Задача

Создайте обработчик graceful shutdown:
- Отлов сигналов завершения (SIGTERM, SIGINT)
- Прекращение приема новых запросов
- Ожидание завершения текущих запросов
- Корректное освобождение ресурсов

## Пример

\`\`\`go
server := NewInferenceServer(model)
server.Start(":8080")
// On SIGTERM: stops accepting, waits for requests, exits
\`\`\``,
			hint1: 'Используйте context.WithTimeout для контролируемого завершения',
			hint2: 'Отслеживайте активные запросы с атомарными счетчиками',
			whyItMatters: `Graceful shutdown обеспечивает надежные деплои:

- **Нулевой простой**: Завершение текущих запросов перед остановкой
- **Целостность данных**: Корректное освобождение ресурсов и сброс данных
- **Совместимость с Kubernetes**: Поддержка SIGTERM для завершения подов
- **Rolling updates**: Бесшовные обновления сервисов`,
		},
		uz: {
			title: 'Graceful shutdown',
			description: `# Graceful shutdown

ML inference serverlari uchun graceful shutdown ni amalga oshiring.

## Topshiriq

Graceful shutdown handlerini yarating:
- Shutdown signallarini ushlash (SIGTERM, SIGINT)
- Yangi so'rovlarni qabul qilishni to'xtatish
- Joriy so'rovlar tugashini kutish
- Resurslarni to'g'ri chiqarish

## Misol

\`\`\`go
server := NewInferenceServer(model)
server.Start(":8080")
// On SIGTERM: stops accepting, waits for requests, exits
\`\`\``,
			hint1: "Boshqariladigan shutdown uchun context.WithTimeout dan foydalaning",
			hint2: "Faol so'rovlarni atomik hisoblagichlar bilan kuzatib boring",
			whyItMatters: `Graceful shutdown ishonchli deploylarni ta'minlaydi:

- **Nol downtime**: To'xtatishdan oldin joriy so'rovlarni yakunlash
- **Ma'lumotlar yaxlitligi**: Resurslarni to'g'ri chiqarish va ma'lumotlarni flush qilish
- **Kubernetes muvofiqligi**: Pod tugatish uchun SIGTERM ni qo'llab-quvvatlash
- **Rolling updates**: Uzluksiz xizmat yangilanishlarini ta'minlash`,
		},
	},
};

export default task;
