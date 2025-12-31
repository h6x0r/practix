import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-health-check',
    title: 'Database Health Check',
    difficulty: 'easy',
    tags: ['go', 'database', 'connection-pool', 'health-check'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a database health check function using PingContext. Health checks are essential for monitoring database availability and should be used in readiness probes for Kubernetes deployments and load balancer health endpoints.

**Requirements:**
- Use db.PingContext with timeout
- Return error if database is unreachable
- Handle context cancellation
- Provide meaningful error messages

**Use Cases:**
- Kubernetes readiness probe
- Load balancer health endpoint
- Startup validation
- Monitoring systems`,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

// TODO: Check if database is healthy and reachable
func HealthCheck(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    panic("TODO: implement with db.PingContext")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func HealthCheck(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Create context with timeout
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Ping database to verify connection
    if err := db.PingContext(ctx); err != nil {
        return fmt.Errorf("database health check failed: %w", err)
    }

    return nil
}`,
	testCode: `package dbx

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// HealthCheck returns nil on successful ping
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing()

	err := HealthCheck(context.Background(), db, 5*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test2(t *testing.T) {
	// HealthCheck returns error on ping failure
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing().WillReturnError(errors.New("connection refused"))

	err := HealthCheck(context.Background(), db, 5*time.Second)
	if err == nil {
		t.Error("expected error")
	}
}

func Test3(t *testing.T) {
	// HealthCheck handles timeout
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing().WillReturnError(context.DeadlineExceeded)

	err := HealthCheck(context.Background(), db, 1*time.Nanosecond)
	if err == nil {
		t.Error("expected error for timeout")
	}
}

func Test4(t *testing.T) {
	// HealthCheck handles cancelled context
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectPing().WillReturnError(context.Canceled)

	err := HealthCheck(ctx, db, 5*time.Second)
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test5(t *testing.T) {
	// HealthCheck with short timeout succeeds
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing()

	err := HealthCheck(context.Background(), db, 100*time.Millisecond)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test6(t *testing.T) {
	// HealthCheck error message contains context
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing().WillReturnError(errors.New("network error"))

	err := HealthCheck(context.Background(), db, 5*time.Second)
	if err == nil {
		t.Error("expected error")
	}
}

func Test7(t *testing.T) {
	// HealthCheck with long timeout succeeds
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing()

	err := HealthCheck(context.Background(), db, 30*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test8(t *testing.T) {
	// HealthCheck handles database down
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing().WillReturnError(errors.New("database is down"))

	err := HealthCheck(context.Background(), db, 5*time.Second)
	if err == nil {
		t.Error("expected error for database down")
	}
}

func Test9(t *testing.T) {
	// HealthCheck can be called multiple times
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing()
	mock.ExpectPing()

	err1 := HealthCheck(context.Background(), db, 5*time.Second)
	err2 := HealthCheck(context.Background(), db, 5*time.Second)

	if err1 != nil || err2 != nil {
		t.Errorf("unexpected errors: %v, %v", err1, err2)
	}
}

func Test10(t *testing.T) {
	// HealthCheck with 1 second timeout
	db, mock, _ := sqlmock.New(sqlmock.MonitorPingsOption(true))
	defer db.Close()

	mock.ExpectPing()

	err := HealthCheck(context.Background(), db, 1*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
`,
    hint1: `Use context.WithTimeout() to create a context with timeout, then pass it to db.PingContext().`,
    hint2: `PingContext verifies the connection to the database is still alive, establishing a connection if necessary. It's lightweight and suitable for frequent health checks.`,
    whyItMatters: `Health checks are critical for production systems to detect database failures early and enable automatic recovery. They allow orchestrators like Kubernetes to restart unhealthy pods and load balancers to route traffic away from degraded instances, improving overall system reliability.

**Production Pattern:**
\`\`\`go
// HTTP endpoint for health check
func healthHandler(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        http.Error(w, "database unavailable", 503)
        return
    }
    w.WriteHeader(200)
}
\`\`\`

**Practical Benefits:**
- Early detection of DB failures
- Integration with Kubernetes liveness/readiness
- Automatic service recovery`,
    order: 2,
    translations: {
        ru: {
            title: 'Проверка здоровья БД',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func HealthCheck(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Создаем контекст с тайм-аутом
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Пингуем базу данных для проверки соединения
    if err := db.PingContext(ctx); err != nil {
        return fmt.Errorf("проверка работоспособности базы данных не удалась: %w", err)
    }

    return nil
}`,
            description: `Реализуйте функцию проверки работоспособности базы данных используя PingContext. Проверки работоспособности необходимы для мониторинга доступности базы данных и должны использоваться в пробах готовности для Kubernetes развертываний и эндпоинтах проверки работоспособности балансировщиков нагрузки.

**Требования:**
- Используйте db.PingContext с тайм-аутом
- Верните ошибку если база данных недоступна
- Обработайте отмену контекста
- Предоставьте осмысленные сообщения об ошибках

**Случаи использования:**
- Проба готовности Kubernetes
- Эндпоинт проверки работоспособности балансировщика нагрузки
- Валидация при запуске
- Системы мониторинга`,
            hint1: `Используйте context.WithTimeout() для создания контекста с тайм-аутом, затем передайте его в db.PingContext().`,
            hint2: `PingContext проверяет что соединение с базой данных все еще живое, устанавливая соединение при необходимости. Он легковесный и подходит для частых проверок работоспособности.`,
            whyItMatters: `Проверки работоспособности критически важны для продакшн систем для раннего обнаружения отказов базы данных и включения автоматического восстановления. Они позволяют оркестраторам типа Kubernetes перезапускать нездоровые поды, а балансировщикам нагрузки направлять трафик от деградировавших экземпляров, улучшая общую надежность системы.

**Продакшен паттерн:**
\`\`\`go
// HTTP эндпоинт для проверки здоровья
func healthHandler(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        http.Error(w, "database unavailable", 503)
        return
    }
    w.WriteHeader(200)
}
\`\`\`

**Практические преимущества:**
- Раннее обнаружение отказов БД
- Интеграция с Kubernetes liveness/readiness
- Автоматическое восстановление сервисов`
        },
        uz: {
            title: 'Database salomatligini tekshirish',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func HealthCheck(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Vaqt tugashi bilan kontekst yaratamiz
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Ulanishni tekshirish uchun ma'lumotlar bazasiga ping yuboramiz
    if err := db.PingContext(ctx); err != nil {
        return fmt.Errorf("ma'lumotlar bazasi salomatligini tekshirish muvaffaqiyatsiz: %w", err)
    }

    return nil
}`,
            description: `PingContext dan foydalanib ma'lumotlar bazasi salomatligini tekshirish funksiyasini amalga oshiring. Salomatlikni tekshirish ma'lumotlar bazasi mavjudligini monitoring qilish uchun zarur va Kubernetes joylashtirmalari uchun tayyorlik problarida va load balancer salomatlik endpointlarida ishlatilishi kerak.

**Talablar:**
- db.PingContext ni vaqt tugashi bilan ishlating
- Agar ma'lumotlar bazasi mavjud bo'lmasa xato qaytaring
- Kontekstni bekor qilishni boshqaring
- Ma'noli xato xabarlarini taqdim eting

**Foydalanish holatlari:**
- Kubernetes tayyorlik probasi
- Load balancer salomatlik endpointi
- Ishga tushirishda tekshirish
- Monitoring tizimlari`,
            hint1: `Vaqt tugashi bilan kontekst yaratish uchun context.WithTimeout() dan foydalaning, keyin uni db.PingContext() ga o'tkazing.`,
            hint2: `PingContext ma'lumotlar bazasiga ulanish hali ham tirikligini tekshiradi, kerak bo'lsa ulanishni o'rnatadi. U yengil va tez-tez salomatlikni tekshirish uchun mos keladi.`,
            whyItMatters: `Salomatlikni tekshirish ma'lumotlar bazasi nosozliklarini erta aniqlash va avtomatik tiklanishni yoqish uchun ishlab chiqarish tizimlari uchun juda muhimdir. Ular Kubernetes kabi orkestratorlarga nosog'lom podlarni qayta ishga tushirish va load balancerlarga trafikni yomonlashgan instansiyalardan yo'naltirish imkonini beradi, bu umumiy tizim ishonchliligini yaxshilaydi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Salomatlikni tekshirish uchun HTTP endpointi
func healthHandler(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        http.Error(w, "database unavailable", 503)
        return
    }
    w.WriteHeader(200)
}
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar bazasi nosozliklarini erta aniqlash
- Kubernetes liveness/readiness bilan integratsiya
- Servislarni avtomatik tiklanish`
        }
    }
};

export default task;
