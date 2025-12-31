import { Task } from '../../../../../../../types';

const task: Task = {
  slug: 'goml-distributed-tracing',
  title: 'Distributed Tracing for ML',
  difficulty: 'hard',
  tags: ['go', 'ml', 'tracing', 'observability', 'spans'],
  estimatedTime: '35m',
  isPremium: true,
  order: 5,

  description: `
## Distributed Tracing for ML

Implement distributed tracing for ML inference pipelines to track requests across preprocessing, inference, and postprocessing stages.

### Requirements

1. **Tracer** - Main tracing component:
   - \`NewTracer(serviceName string)\` - Create tracer for a service
   - \`StartSpan(ctx context.Context, name string) (context.Context, Span)\` - Start new span
   - \`Extract(headers map[string]string) context.Context\` - Extract trace from headers
   - \`Inject(ctx context.Context) map[string]string\` - Inject trace into headers

2. **Span** - Individual trace segment:
   - \`SetTag(key string, value interface{})\` - Add metadata
   - \`SetStatus(status SpanStatus)\` - Set success/error status
   - \`LogEvent(message string)\` - Log event within span
   - \`End()\` - Complete the span

3. **Trace Context**:
   - TraceID - Unique identifier for entire request
   - SpanID - Unique identifier for current operation
   - ParentSpanID - Link to parent span
   - Baggage - Request-scoped key-value data

4. **ML-Specific Spans**:
   - \`preprocessing\` - Feature transformation time
   - \`inference\` - Model execution time
   - \`postprocessing\` - Result formatting time
   - \`model_load\` - Model loading time (if lazy)

### Example

\`\`\`go
tracer := NewTracer("inference-service")

func handleRequest(w http.ResponseWriter, r *http.Request) {
    ctx := tracer.Extract(r.Header)

    ctx, span := tracer.StartSpan(ctx, "handle_inference")
    defer span.End()

    // Preprocessing
    ctx, prepSpan := tracer.StartSpan(ctx, "preprocessing")
    features := preprocess(input)
    prepSpan.SetTag("feature_count", len(features))
    prepSpan.End()

    // Inference
    ctx, infSpan := tracer.StartSpan(ctx, "inference")
    result := model.Predict(features)
    infSpan.SetTag("model_name", "resnet50")
    infSpan.End()
}
\`\`\`
`,

  initialCode: `package tracing

import (
	"context"
	"sync"
	"time"
)

type SpanStatus string

)

type Span interface {
}

type SpanContext struct {
	TraceID      string
	SpanID       string
	ParentSpanID string
	Baggage      map[string]string
}

type Tracer struct {
}

func NewTracer(serviceName string) *Tracer {
	return nil
}

func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, Span) {
	return ctx, nil
}

func (t *Tracer) Extract(headers map[string]string) context.Context {
	return context.Background()
}

func (t *Tracer) Inject(ctx context.Context) map[string]string {
	return nil
}

type spanImpl struct {
}

func generateID() string {
	return ""
}`,

  solutionCode: `package tracing

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

// SpanStatus represents span completion status
type SpanStatus string

const (
	StatusOK    SpanStatus = "ok"
	StatusError SpanStatus = "error"
)

// Span represents a single operation in a trace
type Span interface {
	SetTag(key string, value interface{})
	SetStatus(status SpanStatus)
	LogEvent(message string)
	End()
	TraceID() string
	SpanID() string
}

// SpanContext holds trace propagation data
type SpanContext struct {
	TraceID      string
	SpanID       string
	ParentSpanID string
	Baggage      map[string]string
}

type contextKey string

const spanContextKey contextKey = "span_context"

// SpanEvent represents a log event within a span
type SpanEvent struct {
	Timestamp time.Time
	Message   string
}

// FinishedSpan holds completed span data
type FinishedSpan struct {
	Name         string
	ServiceName  string
	TraceID      string
	SpanID       string
	ParentSpanID string
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	Tags         map[string]interface{}
	Events       []SpanEvent
	Status       SpanStatus
}

// SpanCollector receives finished spans
type SpanCollector interface {
	Collect(span FinishedSpan)
}

// Tracer creates and manages spans
type Tracer struct {
	serviceName string
	collector   SpanCollector
	mu          sync.RWMutex
}

// NewTracer creates a new tracer
func NewTracer(serviceName string) *Tracer {
	return &Tracer{
		serviceName: serviceName,
	}
}

// SetCollector sets the span collector
func (t *Tracer) SetCollector(collector SpanCollector) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.collector = collector
}

// StartSpan starts a new span
func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, Span) {
	var parentCtx *SpanContext
	if sc, ok := ctx.Value(spanContextKey).(*SpanContext); ok {
		parentCtx = sc
	}

	spanID := generateID()
	traceID := spanID
	parentSpanID := ""

	if parentCtx != nil {
		traceID = parentCtx.TraceID
		parentSpanID = parentCtx.SpanID
	}

	newCtx := &SpanContext{
		TraceID:      traceID,
		SpanID:       spanID,
		ParentSpanID: parentSpanID,
		Baggage:      make(map[string]string),
	}

	if parentCtx != nil && parentCtx.Baggage != nil {
		for k, v := range parentCtx.Baggage {
			newCtx.Baggage[k] = v
		}
	}

	span := &spanImpl{
		name:         name,
		serviceName:  t.serviceName,
		traceID:      traceID,
		spanID:       spanID,
		parentSpanID: parentSpanID,
		startTime:    time.Now(),
		tags:         make(map[string]interface{}),
		events:       make([]SpanEvent, 0),
		status:       StatusOK,
		tracer:       t,
	}

	return context.WithValue(ctx, spanContextKey, newCtx), span
}

// Extract extracts trace context from headers
func (t *Tracer) Extract(headers map[string]string) context.Context {
	traceID := headers["X-Trace-ID"]
	spanID := headers["X-Span-ID"]

	if traceID == "" || spanID == "" {
		return context.Background()
	}

	ctx := &SpanContext{
		TraceID:      traceID,
		SpanID:       spanID,
		ParentSpanID: headers["X-Parent-Span-ID"],
		Baggage:      make(map[string]string),
	}

	// Extract baggage
	for k, v := range headers {
		if len(k) > 8 && k[:8] == "X-Baggage-" {
			ctx.Baggage[k[10:]] = v
		}
	}

	return context.WithValue(context.Background(), spanContextKey, ctx)
}

// Inject injects trace context into headers
func (t *Tracer) Inject(ctx context.Context) map[string]string {
	headers := make(map[string]string)

	sc, ok := ctx.Value(spanContextKey).(*SpanContext)
	if !ok {
		return headers
	}

	headers["X-Trace-ID"] = sc.TraceID
	headers["X-Span-ID"] = sc.SpanID
	if sc.ParentSpanID != "" {
		headers["X-Parent-Span-ID"] = sc.ParentSpanID
	}

	for k, v := range sc.Baggage {
		headers["X-Baggage-"+k] = v
	}

	return headers
}

// GetSpanContext returns the current span context from context
func GetSpanContext(ctx context.Context) *SpanContext {
	if sc, ok := ctx.Value(spanContextKey).(*SpanContext); ok {
		return sc
	}
	return nil
}

// spanImpl implements the Span interface
type spanImpl struct {
	name         string
	serviceName  string
	traceID      string
	spanID       string
	parentSpanID string
	startTime    time.Time
	endTime      time.Time
	tags         map[string]interface{}
	events       []SpanEvent
	status       SpanStatus
	tracer       *Tracer
	mu           sync.Mutex
	finished     bool
}

func (s *spanImpl) SetTag(key string, value interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tags[key] = value
}

func (s *spanImpl) SetStatus(status SpanStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status = status
}

func (s *spanImpl) LogEvent(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, SpanEvent{
		Timestamp: time.Now(),
		Message:   message,
	})
}

func (s *spanImpl) End() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.finished {
		return
	}

	s.finished = true
	s.endTime = time.Now()

	s.tracer.mu.RLock()
	collector := s.tracer.collector
	s.tracer.mu.RUnlock()

	if collector != nil {
		collector.Collect(FinishedSpan{
			Name:         s.name,
			ServiceName:  s.serviceName,
			TraceID:      s.traceID,
			SpanID:       s.spanID,
			ParentSpanID: s.parentSpanID,
			StartTime:    s.startTime,
			EndTime:      s.endTime,
			Duration:     s.endTime.Sub(s.startTime),
			Tags:         s.tags,
			Events:       s.events,
			Status:       s.status,
		})
	}
}

func (s *spanImpl) TraceID() string {
	return s.traceID
}

func (s *spanImpl) SpanID() string {
	return s.spanID
}

// generateID generates a random trace/span ID
func generateID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}
`,

  testCode: `package tracing

import (
	"context"
	"sync"
	"testing"
)

type mockCollector struct {
	spans []FinishedSpan
	mu    sync.Mutex
}

func (m *mockCollector) Collect(span FinishedSpan) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.spans = append(m.spans, span)
}

func TestNewTracer(t *testing.T) {
	tracer := NewTracer("test-service")
	if tracer == nil {
		t.Fatal("Expected non-nil tracer")
	}
	if tracer.serviceName != "test-service" {
		t.Errorf("Expected service name 'test-service', got '%s'", tracer.serviceName)
	}
}

func TestStartSpan(t *testing.T) {
	tracer := NewTracer("test-service")
	ctx := context.Background()

	ctx, span := tracer.StartSpan(ctx, "test-operation")
	if span == nil {
		t.Fatal("Expected non-nil span")
	}

	if span.TraceID() == "" {
		t.Error("Expected non-empty trace ID")
	}

	if span.SpanID() == "" {
		t.Error("Expected non-empty span ID")
	}

	span.End()
}

func TestNestedSpans(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	ctx := context.Background()

	ctx, parentSpan := tracer.StartSpan(ctx, "parent")
	parentTraceID := parentSpan.TraceID()
	parentSpanID := parentSpan.SpanID()

	ctx, childSpan := tracer.StartSpan(ctx, "child")
	childTraceID := childSpan.TraceID()

	if childTraceID != parentTraceID {
		t.Error("Child should have same trace ID as parent")
	}

	childSpan.End()
	parentSpan.End()

	if len(collector.spans) != 2 {
		t.Fatalf("Expected 2 spans, got %d", len(collector.spans))
	}

	// Find child span
	var childFinished *FinishedSpan
	for i := range collector.spans {
		if collector.spans[i].Name == "child" {
			childFinished = &collector.spans[i]
			break
		}
	}

	if childFinished == nil {
		t.Fatal("Child span not found")
	}

	if childFinished.ParentSpanID != parentSpanID {
		t.Errorf("Child parent span ID should be %s, got %s", parentSpanID, childFinished.ParentSpanID)
	}
}

func TestSpanTags(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	_, span := tracer.StartSpan(context.Background(), "test")
	span.SetTag("model", "resnet50")
	span.SetTag("batch_size", 32)
	span.End()

	if len(collector.spans) != 1 {
		t.Fatal("Expected 1 span")
	}

	if collector.spans[0].Tags["model"] != "resnet50" {
		t.Error("Expected model tag")
	}
	if collector.spans[0].Tags["batch_size"] != 32 {
		t.Error("Expected batch_size tag")
	}
}

func TestSpanStatus(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	_, span := tracer.StartSpan(context.Background(), "test")
	span.SetStatus(StatusError)
	span.End()

	if collector.spans[0].Status != StatusError {
		t.Error("Expected error status")
	}
}

func TestSpanEvents(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	_, span := tracer.StartSpan(context.Background(), "test")
	span.LogEvent("started processing")
	span.LogEvent("completed processing")
	span.End()

	if len(collector.spans[0].Events) != 2 {
		t.Errorf("Expected 2 events, got %d", len(collector.spans[0].Events))
	}
}

func TestExtractInject(t *testing.T) {
	tracer := NewTracer("test-service")

	ctx, span := tracer.StartSpan(context.Background(), "test")
	traceID := span.TraceID()
	spanID := span.SpanID()

	headers := tracer.Inject(ctx)

	if headers["X-Trace-ID"] != traceID {
		t.Error("Trace ID not injected correctly")
	}
	if headers["X-Span-ID"] != spanID {
		t.Error("Span ID not injected correctly")
	}

	// Extract in another service
	tracer2 := NewTracer("service-2")
	ctx2 := tracer2.Extract(headers)

	ctx2, span2 := tracer2.StartSpan(ctx2, "downstream")
	if span2.TraceID() != traceID {
		t.Error("Trace ID not preserved across services")
	}
	span2.End()
	span.End()
}

func TestGenerateID(t *testing.T) {
	id1 := generateID()
	id2 := generateID()

	if id1 == "" {
		t.Error("ID should not be empty")
	}
	if id1 == id2 {
		t.Error("IDs should be unique")
	}
	if len(id1) != 32 {
		t.Errorf("Expected 32 char ID, got %d", len(id1))
	}
}

func TestSpanDuration(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	_, span := tracer.StartSpan(context.Background(), "test")
	span.End()

	if collector.spans[0].Duration < 0 {
		t.Error("Duration should be positive")
	}
}

func TestSetCollector(t *testing.T) {
	tracer := NewTracer("test-service")
	collector := &mockCollector{spans: make([]FinishedSpan, 0)}
	tracer.SetCollector(collector)

	_, span := tracer.StartSpan(context.Background(), "test")
	span.End()

	if len(collector.spans) != 1 {
		t.Fatalf("Expected 1 span collected, got %d", len(collector.spans))
	}
}
`,

  hint1: `Use context.Context to propagate trace information. Store SpanContext in context using context.WithValue and retrieve with context.Value.`,

  hint2: `For header propagation, use standard header names like X-Trace-ID, X-Span-ID. When starting a child span, inherit TraceID from parent but generate new SpanID.`,

  whyItMatters: `Distributed tracing is essential for understanding ML inference latency. It reveals which stage (preprocessing, inference, postprocessing) causes delays and helps identify bottlenecks across microservices in production ML systems.`,

  translations: {
    ru: {
      title: 'Распределённая Трассировка для ML',
      description: `
## Распределённая Трассировка для ML

Реализуйте распределённую трассировку для ML-инференс пайплайнов для отслеживания запросов через этапы предобработки, инференса и постобработки.

### Требования

1. **Tracer** - Основной компонент трассировки:
   - \`NewTracer(serviceName string)\` - Создание трейсера для сервиса
   - \`StartSpan(ctx context.Context, name string) (context.Context, Span)\` - Начало нового span
   - \`Extract(headers map[string]string) context.Context\` - Извлечение трейса из заголовков
   - \`Inject(ctx context.Context) map[string]string\` - Внедрение трейса в заголовки

2. **Span** - Отдельный сегмент трассировки:
   - \`SetTag(key string, value interface{})\` - Добавление метаданных
   - \`SetStatus(status SpanStatus)\` - Установка статуса успех/ошибка
   - \`LogEvent(message string)\` - Логирование события внутри span
   - \`End()\` - Завершение span

3. **Контекст трассировки**:
   - TraceID - Уникальный идентификатор всего запроса
   - SpanID - Уникальный идентификатор текущей операции
   - ParentSpanID - Ссылка на родительский span
   - Baggage - Request-scoped key-value данные

4. **ML-специфичные Spans**:
   - \`preprocessing\` - Время трансформации признаков
   - \`inference\` - Время выполнения модели
   - \`postprocessing\` - Время форматирования результата
   - \`model_load\` - Время загрузки модели (если lazy)

### Пример

\`\`\`go
tracer := NewTracer("inference-service")

func handleRequest(w http.ResponseWriter, r *http.Request) {
    ctx := tracer.Extract(r.Header)

    ctx, span := tracer.StartSpan(ctx, "handle_inference")
    defer span.End()

    // Preprocessing
    ctx, prepSpan := tracer.StartSpan(ctx, "preprocessing")
    features := preprocess(input)
    prepSpan.SetTag("feature_count", len(features))
    prepSpan.End()

    // Inference
    ctx, infSpan := tracer.StartSpan(ctx, "inference")
    result := model.Predict(features)
    infSpan.SetTag("model_name", "resnet50")
    infSpan.End()
}
\`\`\`
`,
      hint1: 'Используйте context.Context для распространения информации о трассировке. Храните SpanContext в контексте с помощью context.WithValue и извлекайте с помощью context.Value.',
      hint2: 'Для распространения заголовков используйте стандартные имена: X-Trace-ID, X-Span-ID. При создании дочернего span наследуйте TraceID от родителя, но генерируйте новый SpanID.',
      whyItMatters: 'Распределённая трассировка необходима для понимания латентности ML-инференса. Она показывает, какой этап (предобработка, инференс, постобработка) вызывает задержки и помогает выявить узкие места в микросервисных ML-системах.',
    },
    uz: {
      title: 'ML uchun Distributed Tracing',
      description: `
## ML uchun Distributed Tracing

Preprocessing, inference va postprocessing bosqichlari orqali so'rovlarni kuzatish uchun ML inference pipelinelari uchun distributed tracingni amalga oshiring.

### Talablar

1. **Tracer** - Asosiy tracing komponenti:
   - \`NewTracer(serviceName string)\` - Servis uchun tracer yaratish
   - \`StartSpan(ctx context.Context, name string) (context.Context, Span)\` - Yangi span boshlash
   - \`Extract(headers map[string]string) context.Context\` - Headerlardan trace olish
   - \`Inject(ctx context.Context) map[string]string\` - Trace ni headerlarga qo'yish

2. **Span** - Alohida trace segmenti:
   - \`SetTag(key string, value interface{})\` - Metadata qo'shish
   - \`SetStatus(status SpanStatus)\` - Muvaffaqiyat/xato holatini o'rnatish
   - \`LogEvent(message string)\` - Span ichida voqea qayd qilish
   - \`End()\` - Spanni yakunlash

3. **Trace konteksti**:
   - TraceID - Butun so'rov uchun yagona identifikator
   - SpanID - Joriy operatsiya uchun yagona identifikator
   - ParentSpanID - Ota spanga havola
   - Baggage - Request-scoped key-value ma'lumotlar

4. **ML-spetsifik Spanlar**:
   - \`preprocessing\` - Feature transformatsiya vaqti
   - \`inference\` - Model bajarish vaqti
   - \`postprocessing\` - Natija formatlash vaqti
   - \`model_load\` - Model yuklash vaqti (agar lazy bo'lsa)

### Misol

\`\`\`go
tracer := NewTracer("inference-service")

func handleRequest(w http.ResponseWriter, r *http.Request) {
    ctx := tracer.Extract(r.Header)

    ctx, span := tracer.StartSpan(ctx, "handle_inference")
    defer span.End()

    // Preprocessing
    ctx, prepSpan := tracer.StartSpan(ctx, "preprocessing")
    features := preprocess(input)
    prepSpan.SetTag("feature_count", len(features))
    prepSpan.End()

    // Inference
    ctx, infSpan := tracer.StartSpan(ctx, "inference")
    result := model.Predict(features)
    infSpan.SetTag("model_name", "resnet50")
    infSpan.End()
}
\`\`\`
`,
      hint1: "Trace ma'lumotlarini tarqatish uchun context.Context dan foydalaning. SpanContext ni context.WithValue yordamida kontekstda saqlang va context.Value yordamida oling.",
      hint2: "Header tarqatish uchun standart header nomlaridan foydalaning: X-Trace-ID, X-Span-ID. Bola span yaratishda TraceID ni ota-onadan meros qilib oling, lekin yangi SpanID yarating.",
      whyItMatters: "Distributed tracing ML inference latentligini tushunish uchun muhim. U qaysi bosqich (preprocessing, inference, postprocessing) kechikishlarga sabab bo'layotganini ko'rsatadi va mikroservis ML tizimlarida qiyinchiliklarni aniqlashga yordam beradi.",
    },
  },
};

export default task;
