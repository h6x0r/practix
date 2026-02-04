import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-metrics-prometheus-endpoint',
	title: 'Prometheus Metrics HTTP Endpoint',
	difficulty: 'easy',	tags: ['go', 'metrics', 'prometheus', 'observability'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement an HTTP handler that exposes Prometheus-compatible metrics endpoint.

**Requirements:**
1. **Handler**: Return http.Handler that writes Prometheus-formatted metrics
2. **Response Format**: Include HELP and TYPE comments for clarity
3. **Status Code**: Always return 200 OK with valid metrics output

**Prometheus Format:**
\`\`\`go
// Example output format
# HELP demo_up 1 if service is up
# TYPE demo_up gauge
demo_up 1
\`\`\`

**Implementation Pattern:**
\`\`\`go
func Handler() http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Write status code
        // Write HELP and TYPE comments
        // Write metric value
    })
}
\`\`\`

**Key Concepts:**
- Use http.HandlerFunc to create handler from function
- Write headers and metric lines using fmt.Fprint
- Prometheus expects specific text format with newlines
- HELP line documents what the metric measures
- TYPE line specifies metric type (gauge, counter, histogram, summary)

**Example Usage:**
\`\`\`go
// In your main application
http.Handle("/metrics", Handler())
http.ListenAndServe(":8080", nil)

// Prometheus scrapes this endpoint periodically
// curl http://localhost:8080/metrics
// Output:
// # HELP demo_up 1 if up
// # TYPE demo_up gauge
// demo_up 1
\`\`\`

**Constraints:**
- Must return http.Handler interface
- Response must include newlines after each line
- Metric name should be "demo_up" with value 1
- Include both HELP and TYPE comments`,
	initialCode: `package metricsx

import (
	"fmt"
	"net/http"
)

// TODO: Implement Handler
// Return http.Handler that writes Prometheus metrics
// Format: HELP line, TYPE line, metric line
// Use http.HandlerFunc and fmt.Fprint
func Handler() http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package metricsx

import (
	"fmt"
	"net/http"
)

func Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)                                      // signal successful response
		fmt.Fprint(w, "# HELP demo_up 1 if up\\n")                         // document metric purpose
		fmt.Fprint(w, "# TYPE demo_up gauge\\n")                           // specify metric type for prometheus
		fmt.Fprint(w, "demo_up 1\\n")                                      // write actual metric value
	})
}`,
	testCode: `package metricsx

import (
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Basic handler returns 200 OK
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func Test2(t *testing.T) {
	// Response contains HELP line
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body, _ := ioutil.ReadAll(w.Body)
	if !strings.Contains(string(body), "# HELP demo_up") {
		t.Error("response missing HELP comment")
	}
}

func Test3(t *testing.T) {
	// Response contains TYPE line
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body, _ := ioutil.ReadAll(w.Body)
	if !strings.Contains(string(body), "# TYPE demo_up gauge") {
		t.Error("response missing TYPE comment")
	}
}

func Test4(t *testing.T) {
	// Response contains metric value line
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body, _ := ioutil.ReadAll(w.Body)
	if !strings.Contains(string(body), "demo_up 1") {
		t.Error("response missing metric value")
	}
}

func Test5(t *testing.T) {
	// Metric name is demo_up
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body := w.Body.String()
	lines := strings.Split(body, "\\n")
	found := false
	for _, line := range lines {
		if strings.HasPrefix(line, "demo_up ") {
			found = true
			break
		}
	}
	if !found {
		t.Error("metric name demo_up not found")
	}
}

func Test6(t *testing.T) {
	// Metric value is 1
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body := w.Body.String()
	if !strings.Contains(body, "demo_up 1") {
		t.Error("metric value should be 1")
	}
}

func Test7(t *testing.T) {
	// Lines end with newlines
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body := w.Body.String()
	lines := strings.Split(body, "\\n")
	if len(lines) < 3 {
		t.Errorf("expected at least 3 lines, got %d", len(lines))
	}
}

func Test8(t *testing.T) {
	// Response is proper Prometheus text format
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	body := w.Body.String()
	// Check order: HELP, TYPE, value
	helpIdx := strings.Index(body, "# HELP")
	typeIdx := strings.Index(body, "# TYPE")
	valueIdx := strings.Index(body, "demo_up 1")
	if helpIdx == -1 || typeIdx == -1 || valueIdx == -1 {
		t.Error("missing required prometheus format components")
	}
	if helpIdx > typeIdx || typeIdx > valueIdx {
		t.Error("prometheus format order should be HELP, TYPE, value")
	}
}

func Test9(t *testing.T) {
	// Handler implements http.Handler interface
	handler := Handler()
	var _ http.Handler = handler // compile-time check
	if handler == nil {
		t.Error("handler should not be nil")
	}
}

func Test10(t *testing.T) {
	// Multiple requests return consistent content
	handler := Handler()
	for i := 0; i < 3; i++ {
		req := httptest.NewRequest("GET", "/metrics", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		body := w.Body.String()
		if !strings.Contains(body, "demo_up 1") {
			t.Errorf("request %d: missing demo_up 1", i+1)
		}
	}
}
`,
			hint1: `Use http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {...}) to create a handler.`,
			hint2: `Write each line with fmt.Fprint(w, "text\\n") - include newline at the end of each line.`,
			whyItMatters: `Prometheus metrics are the industry standard for monitoring distributed systems and production applications.

**Why This Matters:**

**1. Production Observability**
Without metrics endpoints, you're flying blind in production - you don't know if your service is running or not. Metrics show you what's happening inside your application in real-time:

\`\`\`go
// Basic monitoring setup
package main

import (
    "log"
    "net/http"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
    // Enable Prometheus metrics endpoint
    http.Handle("/metrics", promhttp.Handler())

    // Your business logic
    http.HandleFunc("/api/orders", handleOrders)

    log.Fatal(http.ListenAndServe(":8080", nil))
}
// Now Prometheus can scrape metrics from :8080/metrics
// Grafana displays real-time dashboards
// Alert Manager notifies of issues immediately
\`\`\`

**Real-world scenario:** A microservice suddenly stops working. Without a metrics endpoint, this goes undetected for hours. With metrics, an alert fires immediately.

**2. Production Debugging and Troubleshooting**
Metrics endpoints enable real-time debugging and understanding of production issues:

\`\`\`go
// Exposing business metrics
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // Count number of requests
    requestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "api_requests_total",
            Help: "Total number of API requests",
        },
        []string{"endpoint", "status"},
    )

    // Measure response times
    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "api_request_duration_seconds",
            Help: "Duration of API requests",
            Buckets: prometheus.DefBuckets,
        },
        []string{"endpoint"},
    )
)

func TrackRequest(endpoint string, duration float64, statusCode int) {
    requestsTotal.WithLabelValues(endpoint, fmt.Sprint(statusCode)).Inc()
    requestDuration.WithLabelValues(endpoint).Observe(duration)
}
\`\`\`

**Production usage:**
- Hit the /metrics endpoint - see all real-time metrics
- See which endpoints are slow
- Know which statuses (200, 500, 503) are occurring most
- Detect memory leaks, goroutine leaks

**3. Alerting and Automated Response**
Metrics enable alerting rules for automatic production issue detection:

\`\`\`yaml
# Prometheus alerting rules
groups:
  - name: service_alerts
    rules:
      # Alert when service is down
      - alert: ServiceDown
        expr: up{job="my-service"} == 0
        for: 1m
        annotations:
          summary: "Service {{ $labels.instance }} is down"

      # Alert when error rate is high
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Error rate above 5%"

      # P99 latency too high
      - alert: HighLatency
        expr: histogram_quantile(0.99, api_request_duration_seconds) > 1.0
        for: 5m
        annotations:
          summary: "P99 latency above 1 second"
\`\`\`

**Production impact:**
- **Automatic detection:** Issues found in minutes, not hours
- **Proactive monitoring:** Problems solved before affecting customers
- **24/7 watch:** On-call team notified immediately via push notifications

**4. Capacity Planning and Scaling**
Metrics show when to scale:

\`\`\`go
// Track resource usage
var (
    goroutinesGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_goroutines_count",
        Help: "Current number of goroutines",
    })

    memoryGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_memory_bytes",
        Help: "Current memory usage",
    })
)

// Periodic updates
go func() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        goroutinesGauge.Set(float64(runtime.NumGoroutine()))

        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        memoryGauge.Set(float64(m.Alloc))
    }
}()
\`\`\`

**Real scenario:** Traffic is slowly increasing. Metrics show:
- CPU climbed from 70% to 85%
- Memory increased from 60% to 80%
- Response times grew from 100ms to 300ms

**Conclusion:** Need to add more servers within 2 weeks - before a major outage.

**5. SLA/SLO Monitoring**
Metrics are required for meeting production SLAs (Service Level Agreements):

\`\`\`go
// Enable SLI (Service Level Indicator) metrics
var (
    // For 99.9% uptime SLA
    uptimeGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "service_uptime_ratio",
        Help: "Service uptime ratio (0-1)",
    })

    // For <200ms latency SLO
    latencyP95 = promauto.NewSummary(prometheus.SummaryOpts{
        Name: "api_latency_p95_seconds",
        Help: "95th percentile API latency",
        Objectives: map[float64]float64{0.95: 0.01},
    })
)
\`\`\`

**Business outcomes:**
- **Customer confidence:** Show SLAs with visual dashboards
- **Contract obligations:** SLA violations are immediately visible
- **Performance history:** See improvements or degradations over time

**6. Cost Optimization**
Metrics show where money is being wasted:

\`\`\`go
// Database connection pool monitoring
var dbPoolMetrics = promauto.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "db_pool_connections",
        Help: "Database connection pool state",
    },
    []string{"state"}, // active, idle, waiting
)

func UpdateDBPoolMetrics(pool *sql.DB) {
    stats := pool.Stats()
    dbPoolMetrics.WithLabelValues("active").Set(float64(stats.InUse))
    dbPoolMetrics.WithLabelValues("idle").Set(float64(stats.Idle))
    dbPoolMetrics.WithLabelValues("waiting").Set(float64(stats.WaitCount))
}
\`\`\`

**Optimization result:**
- Database connection pool reduced from 100 to 20 (many idle connections)
- Cloud server costs decreased by 40%
- All performance SLAs still met

**7. Why Prometheus Format?**
Prometheus is the de facto standard for cloud-native monitoring:
- Used by Kubernetes, Docker, many cloud platforms
- Simple text format (easy to debug with curl)
- Powerful query language (PromQL)
- Built-in alerting and visualization
- Excellent ecosystem (Grafana integration)

**8. Production Example**
E-commerce company tracks these metrics:
\`\`\`go
# Orders processed
orders_total{status="completed"} 45234
orders_total{status="failed"} 127

# Payment processing
payment_duration_seconds{provider="stripe",quantile="0.99"} 0.234
payment_errors_total{provider="stripe"} 5

# Inventory
inventory_items{warehouse="us-east"} 12450
inventory_items{warehouse="eu-west"} 8903

# Cache performance
cache_hits_total 1250000
cache_misses_total 45000
\`\`\`

**9. Debugging Production Issues**
When users report "site is slow":
\`\`\`bash
# Check Prometheus metrics
curl http://api.example.com/metrics | grep http_duration

# See:
http_duration_seconds{endpoint="/checkout",quantile="0.99"} 5.2

# Found it! Checkout endpoint is slow
# Dive deeper:
db_query_duration_seconds{query="get_cart"} 4.8

# Root cause: database query is slow
# Fix the query, deploy, verify metrics improve
\`\`\`

**10. Metric Types Matter**
- **Gauge**: Current value (memory usage, active connections)
- **Counter**: Ever-increasing count (total requests, errors)
- **Histogram**: Distribution of values (response times)
- **Summary**: Similar to histogram with quantiles

**Real-World Impact:**
In production practice, exposing Prometheus metrics:
- **Downtime reduced by 70-85%** (issues found faster)
- **MTTR (Mean Time To Resolution) improved by 80%** (faster debugging)
- **Customer complaints decreased by 60%** (proactive issue resolution)
- **Operational costs reduced by 30-40%** (resource optimization)

**Final Conclusions:**
A Prometheus metrics endpoint is not just a simple "/metrics" URL - it's the gateway to an entire observability platform. It transforms running production services blindly into a fully monitored, alerted, and optimized system. In modern production deployments, this is a **mandatory minimum requirement**, not a luxury.

**Production Best Practices:**
1. Always expose /metrics endpoint
2. Include service health gauge (app_up)
3. Track request counts and errors
4. Monitor resource usage (memory, CPU, goroutines)
5. Use labels for dimensionality (method, status, endpoint)
6. Keep metric names consistent across services
7. Document metrics with HELP comments
8. Set up alerting rules for critical metrics
9. Regularly review and optimize based on metrics data

Without metrics, you're debugging by guessing. With metrics, you have data-driven insights into your production systems.`,	order: 0,
	translations: {
		ru: {
			title: 'HTTP эндпоинт Prometheus метрик',
			solutionCode: `package metricsx

import (
	"fmt"
	"net/http"
)

func Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)                                      // сигнализируем успешный ответ
		fmt.Fprint(w, "# HELP demo_up 1 if up\\n")                         // документируем назначение метрики
		fmt.Fprint(w, "# TYPE demo_up gauge\\n")                           // указываем тип метрики для prometheus
		fmt.Fprint(w, "demo_up 1\\n")                                      // пишем фактическое значение метрики
	})
}`,
			description: `Реализуйте HTTP обработчик, который отдает Prometheus-совместимые метрики.

**Требования:**
1. **Handler**: Вернуть http.Handler, который пишет метрики в формате Prometheus
2. **Формат ответа**: Включить HELP и TYPE комментарии для ясности
3. **Код статуса**: Всегда возвращать 200 OK с валидным выводом метрик

**Формат Prometheus:**
\`\`\`go
// Пример формата вывода
# HELP demo_up 1 if service is up
# TYPE demo_up gauge
demo_up 1
\`\`\`

**Ключевые концепции:**
- Используйте http.HandlerFunc для создания обработчика из функции
- Пишите заголовки и строки метрик через fmt.Fprint
- Prometheus ожидает специфический текстовый формат с переводами строк
- Строка HELP документирует, что измеряет метрика
- Строка TYPE указывает тип метрики (gauge, counter, histogram, summary)

**Ограничения:**
- Должен возвращать интерфейс http.Handler
- Ответ должен включать переводы строк после каждой линии
- Имя метрики должно быть "demo_up" со значением 1
- Включить как HELP, так и TYPE комментарии`,
			hint1: `Используйте http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {...}) для создания обработчика.`,
			hint2: `Пишите каждую строку с fmt.Fprint(w, "text\\n") - включайте перевод строки в конце каждой линии.`,
			whyItMatters: `Prometheus метрики - индустриальный стандарт для мониторинга распределенных систем и production приложений.

**Почему это важно:**

**1. Production Observability**
Без endpoint'а метрик вы летите вслепую в production - вы не знаете, работает ли ваш сервис или нет. Метрики показывают вам, что происходит внутри вашего приложения в реальном времени:

\`\`\`go
// Базовая настройка мониторинга
package main

import (
    "log"
    "net/http"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
    // Включаем endpoint метрик Prometheus
    http.Handle("/metrics", promhttp.Handler())

    // Ваша бизнес-логика
    http.HandleFunc("/api/orders", handleOrders)

    log.Fatal(http.ListenAndServe(":8080", nil))
}
// Теперь Prometheus может скрейпить метрики с :8080/metrics
// Grafana отображает реал-тайм дашборды
// Alert Manager немедленно уведомляет о проблемах
\`\`\`

**Реальный сценарий:** Микросервис внезапно перестает работать. Без endpoint метрик это остается незамеченным часами. С метриками alert срабатывает немедленно.

**2. Production отладка и устранение неполадок**
Endpoint метрик обеспечивает отладку в реальном времени и понимание production проблем:

\`\`\`go
// Раскрытие бизнес-метрик
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // Подсчет количества запросов
    requestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "api_requests_total",
            Help: "Общее количество API запросов",
        },
        []string{"endpoint", "status"},
    )

    // Измерение времени ответа
    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "api_request_duration_seconds",
            Help: "Длительность API запросов",
            Buckets: prometheus.DefBuckets,
        },
        []string{"endpoint"},
    )
)

func TrackRequest(endpoint string, duration float64, statusCode int) {
    requestsTotal.WithLabelValues(endpoint, fmt.Sprint(statusCode)).Inc()
    requestDuration.WithLabelValues(endpoint).Observe(duration)
}
\`\`\`

**Production использование:**
- Обратитесь к /metrics endpoint - увидите все метрики в реальном времени
- Посмотрите, какие endpoint'ы медленные
- Узнайте, какие статусы (200, 500, 503) встречаются чаще всего
- Обнаружьте утечки памяти, утечки горутин

**3. Alerting и автоматическое реагирование**
Метрики обеспечивают правила алертинга для автоматического обнаружения production проблем:

\`\`\`yaml
# Правила алертинга Prometheus
groups:
  - name: service_alerts
    rules:
      # Alert когда сервис недоступен
      - alert: ServiceDown
        expr: up{job="my-service"} == 0
        for: 1m
        annotations:
          summary: "Сервис {{ $labels.instance }} недоступен"

      # Alert когда уровень ошибок высокий
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Уровень ошибок выше 5%"

      # P99 latency слишком высокая
      - alert: HighLatency
        expr: histogram_quantile(0.99, api_request_duration_seconds) > 1.0
        for: 5m
        annotations:
          summary: "P99 latency выше 1 секунды"
\`\`\`

**Production воздействие:**
- **Автоматическое обнаружение:** Проблемы обнаруживаются за минуты, а не часы
- **Проактивный мониторинг:** Проблемы решаются до воздействия на клиентов
- **24/7 наблюдение:** Дежурная команда уведомляется немедленно через push-уведомления

**4. Планирование мощностей и масштабирование**
Метрики показывают, когда нужно масштабироваться:

\`\`\`go
// Отслеживание использования ресурсов
var (
    goroutinesGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_goroutines_count",
        Help: "Текущее количество горутин",
    })

    memoryGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_memory_bytes",
        Help: "Текущее использование памяти",
    })
)

// Периодические обновления
go func() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        goroutinesGauge.Set(float64(runtime.NumGoroutine()))

        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        memoryGauge.Set(float64(m.Alloc))
    }
}()
\`\`\`

**Реальный сценарий:** Трафик медленно увеличивается. Метрики показывают:
- CPU вырос с 70% до 85%
- Память увеличилась с 60% до 80%
- Время ответа выросло со 100мс до 300мс

**Вывод:** Нужно добавить больше серверов в течение 2 недель - до крупного сбоя.

**5. SLA/SLO мониторинг**
Метрики требуются для соблюдения production SLA (Service Level Agreements):

\`\`\`go
// Включение SLI (Service Level Indicator) метрик
var (
    // Для 99.9% uptime SLA
    uptimeGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "service_uptime_ratio",
        Help: "Коэффициент uptime сервиса (0-1)",
    })

    // Для <200мс latency SLO
    latencyP95 = promauto.NewSummary(prometheus.SummaryOpts{
        Name: "api_latency_p95_seconds",
        Help: "95-й процентиль API latency",
        Objectives: map[float64]float64{0.95: 0.01},
    })
)
\`\`\`

**Бизнес-результаты:**
- **Доверие клиентов:** Показывайте SLA с визуальными дашбордами
- **Контрактные обязательства:** Нарушения SLA сразу видны
- **История производительности:** Видьте улучшения или ухудшения со временем

**6. Оптимизация затрат**
Метрики показывают, где тратятся деньги впустую:

\`\`\`go
// Мониторинг пула соединений с БД
var dbPoolMetrics = promauto.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "db_pool_connections",
        Help: "Состояние пула соединений с БД",
    },
    []string{"state"}, // active, idle, waiting
)

func UpdateDBPoolMetrics(pool *sql.DB) {
    stats := pool.Stats()
    dbPoolMetrics.WithLabelValues("active").Set(float64(stats.InUse))
    dbPoolMetrics.WithLabelValues("idle").Set(float64(stats.Idle))
    dbPoolMetrics.WithLabelValues("waiting").Set(float64(stats.WaitCount))
}
\`\`\`

**Результат оптимизации:**
- Пул соединений с БД сокращен со 100 до 20 (много idle соединений)
- Расходы на облачный сервер снизились на 40%
- Все SLA производительности по-прежнему выполняются

**7. Почему формат Prometheus?**
Prometheus - это де-факто стандарт для cloud-native мониторинга:
- Используется Kubernetes, Docker, многими облачными платформами
- Простой текстовый формат (легко отлаживать с curl)
- Мощный язык запросов (PromQL)
- Встроенный алертинг и визуализация
- Отличная экосистема (интеграция с Grafana)

**8. Production пример**
E-commerce компания отслеживает эти метрики:
\`\`\`go
# Обработанные заказы
orders_total{status="completed"} 45234
orders_total{status="failed"} 127

# Обработка платежей
payment_duration_seconds{provider="stripe",quantile="0.99"} 0.234
payment_errors_total{provider="stripe"} 5

# Инвентарь
inventory_items{warehouse="us-east"} 12450
inventory_items{warehouse="eu-west"} 8903

# Производительность кеша
cache_hits_total 1250000
cache_misses_total 45000
\`\`\`

**9. Отладка Production проблем**
Когда пользователи жалуются "сайт медленный":
\`\`\`bash
# Проверяем Prometheus метрики
curl http://api.example.com/metrics | grep http_duration

# Видим:
http_duration_seconds{endpoint="/checkout",quantile="0.99"} 5.2

# Нашли! Checkout endpoint медленный
# Углубляемся:
db_query_duration_seconds{query="get_cart"} 4.8

# Root cause: запрос к базе данных медленный
# Исправляем запрос, деплоим, проверяем улучшение метрик
\`\`\`

**10. Типы метрик имеют значение**
- **Gauge**: Текущее значение (использование памяти, активные соединения)
- **Counter**: Постоянно растущий счетчик (всего запросов, ошибок)
- **Histogram**: Распределение значений (время ответа)
- **Summary**: Похоже на histogram с квантилями

**Реальное воздействие:**
На практике в production раскрытие Prometheus метрик:
- **Downtime снижен на 70-85%** (проблемы обнаруживаются быстрее)
- **MTTR (Mean Time To Resolution) улучшен на 80%** (более быстрая отладка)
- **Жалобы клиентов снизились на 60%** (проактивное решение проблем)
- **Операционные затраты снижены на 30-40%** (оптимизация ресурсов)

**Итоговые выводы:**
Prometheus metrics endpoint - это не просто URL "/metrics" - это шлюз к целой платформе observability. Он превращает слепое управление production сервисами в полностью мониторимую, алертируемую и оптимизированную систему. В современных production развертываниях это **обязательное минимальное требование**, а не роскошь.

**Production лучшие практики:**
1. Всегда раскрывайте /metrics endpoint
2. Включайте gauge здоровья сервиса (app_up)
3. Отслеживайте счетчики запросов и ошибок
4. Мониторьте использование ресурсов (память, CPU, горутины)
5. Используйте labels для размерности (method, status, endpoint)
6. Держите имена метрик согласованными между сервисами
7. Документируйте метрики с HELP комментариями
8. Настройте правила алертинга для критических метрик
9. Регулярно просматривайте и оптимизируйте на основе данных метрик

Без метрик вы отлаживаете угадыванием. С метриками у вас есть data-driven инсайты в ваши production системы.`
		},
		uz: {
			title: `Prometheus metriklar HTTP endpoint`,
			solutionCode: `package metricsx

import (
	"fmt"
	"net/http"
)

func Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)                                      // muvaffaqiyatli javobni signal qilamiz
		fmt.Fprint(w, "# HELP demo_up 1 if up\\n")                         // metrika maqsadini hujjatlaymiz
		fmt.Fprint(w, "# TYPE demo_up gauge\\n")                           // prometheus uchun metrika turini belgilaymiz
		fmt.Fprint(w, "demo_up 1\\n")                                      // haqiqiy metrika qiymatini yozamiz
	})
}`,
			description: `Prometheus-mos metrikalari endpoint ni ochib beruvchi HTTP handler ni amalga oshiring.

**Talablar:**
1. **Handler**: Prometheus formatida metrikalar yozadigan http.Handler qaytarish
2. **Javob formati**: Aniqlik uchun HELP va TYPE izohlarini qo'shish
3. **Holat kodi**: Har doim to'g'ri metrika chiqishi bilan 200 OK qaytarish

**Prometheus formati:**
\`\`\`go
// Chiqish formati misoli
# HELP demo_up 1 if service is up
# TYPE demo_up gauge
demo_up 1
\`\`\`

**Asosiy tushunchalar:**
- Funksiyadan handler yaratish uchun http.HandlerFunc ishlating
- fmt.Fprint yordamida sarlavhalar va metrika qatorlarini yozing
- Prometheus yangi qatorlar bilan maxsus matn formatini kutadi
- HELP qatori metrika nimani o'lchashinini hujjatlashtiradi
- TYPE qatori metrika turini belgilaydi (gauge, counter, histogram, summary)

**Cheklovlar:**
- http.Handler interfeysini qaytarish kerak
- Javob har bir qatordan keyin yangi qatorlarni o'z ichiga olishi kerak
- Metrika nomi "demo_up" qiymati 1 bo'lishi kerak
- HELP va TYPE izohlarini ikkalasini ham qo'shing`,
			hint1: `Handler yaratish uchun http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {...}) ishlating.`,
			hint2: `Har bir qatorni fmt.Fprint(w, "text\\n") bilan yozing - har bir qator oxirida yangi qator qo'shing.`,
			whyItMatters: `Prometheus metrikalari taqsimlangan tizimlar va production ilovalarni monitoring qilish uchun sanoat standartidir.

**Nima uchun bu muhim:**

**1. Production Observability (Ko'rinish)**
Metrikalar endpointsiz production da "ko'r holda" ishlamoqdasiz - xizmat ishlab turibdi yoki yo'qligini bilmaysiz:

\`\`\`go
// Asosiy monitoring sozlash
package main

import (
    "log"
    "net/http"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
    // Prometheus metrikalar endpointini yoqing
    http.Handle("/metrics", promhttp.Handler())

    // O'z biznes logikangiz
    http.HandleFunc("/api/orders", handleOrders)

    log.Fatal(http.ListenAndServe(":8080", nil))
}
// Endi Prometheus :8080/metrics dan metrikalarni olishi mumkin
// Grafana real-time dashboardlarni ko'rsatadi
// Alert Manager muammolarni darhol xabar qiladi
\`\`\`

**Real-world stsenariy:** Mikroxizmat suddenly ishlamay qoladi. Prometheus metrikalari endpointsiz, bu soatlab aniqlanmaydi. Metrikalar bilan, alert darhol ishga tushadi.

**2. Production Debugging va Troubleshooting**
Metrikalar endpointlari real-time debugging va production muammolarini tushunish imkonini beradi:

\`\`\`go
// Biznes metrikalarini ochib berish
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // So'rovlar sonini sanash
    requestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "api_requests_total",
            Help: "Umumiy API so'rovlari soni",
        },
        []string{"endpoint", "status"},
    )

    // Javob vaqtlarini o'lchash
    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "api_request_duration_seconds",
            Help: "API so'rovlarining davomiyligi",
            Buckets: prometheus.DefBuckets,
        },
        []string{"endpoint"},
    )
)

func TrackRequest(endpoint string, duration float64, statusCode int) {
    requestsTotal.WithLabelValues(endpoint, fmt.Sprint(statusCode)).Inc()
    requestDuration.WithLabelValues(endpoint).Observe(duration)
}
\`\`\`

**Production foydalanish:**
- \`/metrics\` endpointiga kiring - barcha real-time metrikalarni ko'rasiz
- Qaysi endpointlar sekin ekanligini ko'ring
- Qaysi statuslar (200, 500, 503) ko'p ketayotganini bilib oling
- Memory leaklarini, goroutine leaklarini aniqlang

**3. Alerting va Automated Response**
Metrikalar production muammolarini avtomatik aniqlash uchun alertlar sozlaydi:

\`\`\`yaml
# Prometheus alerting qoidalari
groups:
  - name: service_alerts
    rules:
      # Xizmat ishlamayotganda alert
      - alert: ServiceDown
        expr: up{job="my-service"} == 0
        for: 1m
        annotations:
          summary: "Xizmat {{ $labels.instance }} ishlamayapti"

      # Xato darajasi yuqori bo'lganda alert
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Xato darajasi 5% dan yuqori"

      # P99 latency juda yuqori
      - alert: HighLatency
        expr: histogram_quantile(0.99, api_request_duration_seconds) > 1.0
        for: 5m
        annotations:
          summary: "P99 latency 1 soniyadan yuqori"
\`\`\`

**Production ta'siri:**
- **Avtomatik aniqlash:** Muammolar daqiqalar ichida topiladi, soatlar emas
- **Proaktiv monitoring:** Muammolar mijozlarga ta'sir qilishidan oldin hal qilinadi
- **24/7 kuzatuv:** On-call jamoasi pushnotification orqali darhol xabardor qilinadi

**4. Capacity Planning va Scaling**
Metrikalar qachon scale qilish kerakligini ko'rsatadi:

\`\`\`go
// Resurs ishlatilishini kuzatish
var (
    goroutinesGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_goroutines_count",
        Help: "Hozirgi goroutine'lar soni",
    })

    memoryGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "runtime_memory_bytes",
        Help: "Hozirgi memory ishlatilishi",
    })
)

// Davriy yangilanish
go func() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        goroutinesGauge.Set(float64(runtime.NumGoroutine()))

        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        memoryGauge.Set(float64(m.Alloc))
    }
}()
\`\`\`

**Real stsenariy:** Trafik asta-sekin oshmoqda. Metrikalar ko'rsatadi:
- CPU 70% dan 85% ga ko'tarildi
- Memory 60% dan 80% ga oshdi
- Javob vaqtlari 100ms dan 300ms ga o'sdi

**Xulosa:** 2 hafta ichida qo'shimcha serverlar qo'shish kerak - katta nosozlik bo'lishidan oldin.

**5. SLA/SLO Monitoring**
Production SLA'larni (Service Level Agreements) bajarish uchun metrikalar kerak:

\`\`\`go
// SLI (Service Level Indicator) metrikalarini yoqish
var (
    // 99.9% uptime SLA uchun
    uptimeGauge = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "service_uptime_ratio",
        Help: "Xizmat uptime nisbati (0-1)",
    })

    // <200ms latency SLO uchun
    latencyP95 = promauto.NewSummary(prometheus.SummaryOpts{
        Name: "api_latency_p95_seconds",
        Help: "95th percentile API latency",
        Objectives: map[float64]float64{0.95: 0.01},
    })
)
\`\`\`

**Business natijalar:**
- **Mijoz ishonchi:** SLA'larni vizual dashboard bilan ko'rsatish
- **Shartnoma majburiyatlari:** SLA buzilishlari darhol ko'rinadi
- **Performance tarixiy ma'lumotlari:** Vaqt o'tishi bilan yaxshilanish yoki yomonlashishni ko'ring

**6. Cost Optimization (Xarajatlarni Optimallashtirish)**
Metrikalar qayerda pul isrof bo'layotganini ko'rsatadi:

\`\`\`go
// Database connection pool monitoring
var dbPoolMetrics = promauto.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "db_pool_connections",
        Help: "Database connection pool holati",
    },
    []string{"state"}, // active, idle, waiting
)

func UpdateDBPoolMetrics(pool *sql.DB) {
    stats := pool.Stats()
    dbPoolMetrics.WithLabelValues("active").Set(float64(stats.InUse))
    dbPoolMetrics.WithLabelValues("idle").Set(float64(stats.Idle))
    dbPoolMetrics.WithLabelValues("waiting").Set(float64(stats.WaitCount))
}
\`\`\`

**Optimization natijasi:**
- Database connection pool 100 dan 20 ga kamaytirildi (idle ulanishlar ko'p ekan)
- Cloud server xarajatlari 40% kamaydi
- Hali ham barcha performance SLA'lar bajariladi

**7. Nima uchun Prometheus formati?**
Prometheus cloud-native monitoring uchun de facto standartdir:
- Kubernetes, Docker, ko'plab cloud platformalar tomonidan ishlatiladi
- Oddiy matn formati (curl bilan debug qilish oson)
- Kuchli so'rov tili (PromQL)
- O'rnatilgan alerting va vizualizatsiya
- Ajoyib ekotizim (Grafana integratsiyasi)

**8. Production misol**
E-commerce kompaniyasi ushbu metrikalarni kuzatadi:
\`\`\`go
# Qayta ishlangan buyurtmalar
orders_total{status="completed"} 45234
orders_total{status="failed"} 127

# To'lov qayta ishlash
payment_duration_seconds{provider="stripe",quantile="0.99"} 0.234
payment_errors_total{provider="stripe"} 5

# Inventar
inventory_items{warehouse="us-east"} 12450
inventory_items{warehouse="eu-west"} 8903

# Kesh ishlashi
cache_hits_total 1250000
cache_misses_total 45000
\`\`\`

**9. Production muammolarini debug qilish**
Foydalanuvchilar "sayt sekin" deb shikoyat qilganda:
\`\`\`bash
# Prometheus metrikalarini tekshiring
curl http://api.example.com/metrics | grep http_duration

# Ko'ramiz:
http_duration_seconds{endpoint="/checkout",quantile="0.99"} 5.2

# Topdik! Checkout endpoint sekin
# Chuqurroq tekshiramiz:
db_query_duration_seconds{query="get_cart"} 4.8

# Root cause: database so'rovi sekin
# So'rovni tuzatamiz, deploy qilamiz, metrikalar yaxshilanganini tekshiramiz
\`\`\`

**10. Metrika turlari muhim**
- **Gauge**: Hozirgi qiymat (memory ishlatilishi, aktiv ulanishlar)
- **Counter**: Doimo o'sib boradigan hisoblagich (jami so'rovlar, xatolar)
- **Histogram**: Qiymatlarning taqsimlanishi (javob vaqtlari)
- **Summary**: Histogram ga o'xshash, kvantillar bilan

**Real-world ta'sir:**
Production amaliyotida Prometheus metrikalarini ochib berish:
- **Downtime 70-85% kamayadi** (muammolar tezroq topiladi)
- **MTTR (Mean Time To Resolution) 80% yaxshilanadi** (tezroq debugging)
- **Mijoz shikoyatlari 60% pasayadi** (proaktiv muammo hal qilish)
- **Operatsion xarajatlar 30-40% kamayadi** (resurs optimizatsiyasi)

**Yakuniy xulosalar:**
Prometheus metrics endpoint oddiy "/metrics" URL emas - bu butun observability platformasiga eshikdir. U production xizmatlarni ko'r holda ishlatishdan to'liq monitoring, alerting va optimization qilinadigan tizimga o'tkazadi. Zamonaviy production deploymentlarda bu **majburiy minimal talab**, lux emas.

**Production eng yaxshi amaliyotlar:**
1. Har doim /metrics endpointini ochib bering
2. Xizmat sog'lig'i gauge'ini qo'shing (app_up)
3. So'rovlar va xatolar hisoblagichlarini kuzating
4. Resurs ishlatilishini monitoring qiling (memory, CPU, goroutine'lar)
5. O'lchovlilik uchun labellardan foydalaning (method, status, endpoint)
6. Xizmatlar o'rtasida metrika nomlarini izchil saqlang
7. HELP izohlar bilan metrikalarni hujjatlashtiring
8. Muhim metrikalar uchun alerting qoidalarini sozlang
9. Metrikalar ma'lumotlariga asoslanib muntazam ko'rib chiqing va optimallashtiring

Metrikalar bo'lmasa, taxmin qilish orqali debug qilasiz. Metrikalar bilan production tizimlaringizga data-driven tushunchalar olasiz.`
		}
	}
};

export default task;
