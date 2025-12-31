import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-graceful-shutdown',
    title: 'Graceful Shutdown',
    difficulty: 'medium',
    tags: ['go', 'database', 'connection-pool', 'shutdown'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a graceful shutdown function for database connections. Proper shutdown ensures all in-flight queries complete, connections are closed cleanly, and no data is lost or corrupted during application termination.

**Requirements:**
- Wait for ongoing operations with timeout
- Close database connections using db.Close()
- Handle shutdown signals properly
- Return error if shutdown times out

**Process:**
1. Stop accepting new requests
2. Wait for in-flight operations
3. Close database connections
4. Exit cleanly or timeout`,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

// TODO: Gracefully shutdown database connections
func GracefulShutdown(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    panic("TODO: implement graceful shutdown with db.Close()")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func GracefulShutdown(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Create shutdown context with timeout
    shutdownCtx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Channel to signal completion
    done := make(chan error, 1)

    // Close database in goroutine
    go func() {
        // db.Close() waits for all connections to be returned to pool
        // or closed by their connection lifetime settings
        done <- db.Close()
    }()

    // Wait for shutdown or timeout
    select {
    case err := <-done:
        if err != nil {
            return fmt.Errorf("error closing database: %w", err)
        }
        return nil

    case <-shutdownCtx.Done():
        return fmt.Errorf("database shutdown timed out after %v", timeout)
    }
}`,
	testCode: `package dbx

import (
	"context"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// GracefulShutdown returns nil on successful close
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 5*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test2(t *testing.T) {
	// GracefulShutdown closes the database
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	GracefulShutdown(context.Background(), db, 5*time.Second)

	// Attempting to use closed db should fail
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("expectations were not met: %v", err)
	}
}

func Test3(t *testing.T) {
	// GracefulShutdown handles cancelled context
	db, mock, _ := sqlmock.New()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectClose()

	_ = GracefulShutdown(ctx, db, 5*time.Second)
}

func Test4(t *testing.T) {
	// GracefulShutdown with short timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 100*time.Millisecond)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test5(t *testing.T) {
	// GracefulShutdown with long timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 30*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test6(t *testing.T) {
	// GracefulShutdown with 1 second timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 1*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test7(t *testing.T) {
	// GracefulShutdown expectations met
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	GracefulShutdown(context.Background(), db, 5*time.Second)

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("close was not called: %v", err)
	}
}

func Test8(t *testing.T) {
	// GracefulShutdown with typical Kubernetes timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 25*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test9(t *testing.T) {
	// GracefulShutdown with milliseconds timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 500*time.Millisecond)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test10(t *testing.T) {
	// GracefulShutdown with 10 seconds timeout
	db, mock, _ := sqlmock.New()

	mock.ExpectClose()

	err := GracefulShutdown(context.Background(), db, 10*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
`,
    hint1: `Use db.Close() to close the database. This method blocks until all connections are returned to the pool or closed.`,
    hint2: `Wrap db.Close() in a goroutine and use a select statement with context timeout to prevent indefinite blocking during shutdown.`,
    whyItMatters: `Graceful shutdown is critical for data integrity and user experience. Without it, in-flight transactions may be aborted, data may be lost, and connections may be left hanging. Kubernetes and other orchestrators send SIGTERM before SIGKILL, giving your application time to shut down cleanly - but only if you handle it properly.

**Production Pattern:**
\`\`\`go
// Handling SIGTERM in Kubernetes
func main() {
    shutdown := make(chan os.Signal, 1)
    signal.Notify(shutdown, syscall.SIGTERM)

    <-shutdown
    log.Println("Shutting down gracefully...")

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    db.Close() // Waits for transactions to complete
}
\`\`\`

**Practical Benefits:**
- Protects data integrity
- Zero request loss during deployment
- Proper Kubernetes integration`,
    order: 4,
    translations: {
        ru: {
            title: 'Корректное завершение',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func GracefulShutdown(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Создаем контекст завершения с тайм-аутом
    shutdownCtx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Канал для сигнализации завершения
    done := make(chan error, 1)

    // Закрываем базу данных в горутине
    go func() {
        // db.Close() ждет пока все соединения вернутся в пул
        // или закроются по настройкам времени жизни соединения
        done <- db.Close()
    }()

    // Ждем завершения или тайм-аута
    select {
    case err := <-done:
        if err != nil {
            return fmt.Errorf("ошибка закрытия базы данных: %w", err)
        }
        return nil

    case <-shutdownCtx.Done():
        return fmt.Errorf("завершение работы базы данных истекло после %v", timeout)
    }
}`,
            description: `Реализуйте функцию корректного завершения работы для соединений с базой данных. Правильное завершение гарантирует, что все выполняющиеся запросы завершатся, соединения будут закрыты чисто, и никакие данные не будут потеряны или повреждены во время завершения приложения.

**Требования:**
- Ждите текущих операций с тайм-аутом
- Закрывайте соединения с базой данных используя db.Close()
- Правильно обрабатывайте сигналы завершения
- Верните ошибку если завершение истекло по тайм-ауту

**Процесс:**
1. Прекратить принимать новые запросы
2. Дождаться выполняющихся операций
3. Закрыть соединения с базой данных
4. Выйти чисто или по тайм-ауту`,
            hint1: `Используйте db.Close() для закрытия базы данных. Этот метод блокируется пока все соединения не вернутся в пул или не закроются.`,
            hint2: `Оберните db.Close() в горутину и используйте select с тайм-аутом контекста чтобы предотвратить бесконечную блокировку во время завершения.`,
            whyItMatters: `Корректное завершение критически важно для целостности данных и пользовательского опыта. Без него выполняющиеся транзакции могут быть прерваны, данные могут быть потеряны, а соединения могут остаться висящими. Kubernetes и другие оркестраторы отправляют SIGTERM перед SIGKILL, давая приложению время для чистого завершения - но только если вы обрабатываете это правильно.

**Продакшен паттерн:**
\`\`\`go
// Обработка SIGTERM в Kubernetes
func main() {
    shutdown := make(chan os.Signal, 1)
    signal.Notify(shutdown, syscall.SIGTERM)

    <-shutdown
    log.Println("Shutting down gracefully...")

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    db.Close() // Ждет завершения транзакций
}
\`\`\`

**Практические преимущества:**
- Защита целостности данных
- Нулевые потери запросов при деплое
- Корректная работа с Kubernetes`
        },
        uz: {
            title: 'Graceful shutdown',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

func GracefulShutdown(ctx context.Context, db *sql.DB, timeout time.Duration) error {
    // Vaqt tugashi bilan tugatish kontekstini yaratamiz
    shutdownCtx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Tugashni signalizatsiya qilish uchun kanal
    done := make(chan error, 1)

    // Ma'lumotlar bazasini goroutine da yopamiz
    go func() {
        // db.Close() barcha ulanishlar poolga qaytguncha
        // yoki ulanish umr sozlamalari bo'yicha yopilguncha kutadi
        done <- db.Close()
    }()

    // Tugatish yoki vaqt tugashini kutamiz
    select {
    case err := <-done:
        if err != nil {
            return fmt.Errorf("ma'lumotlar bazasini yopishda xato: %w", err)
        }
        return nil

    case <-shutdownCtx.Done():
        return fmt.Errorf("ma'lumotlar bazasini tugatish %v dan keyin vaqt tugadi", timeout)
    }
}`,
            description: `Ma'lumotlar bazasi ulanishlari uchun to'g'ri tugatish funksiyasini amalga oshiring. To'g'ri tugatish barcha bajarilayotgan so'rovlar tugashini, ulanishlarning toza yopilishini va dastur tugashi paytida hech qanday ma'lumot yo'qolmasligi yoki buzilmasligini ta'minlaydi.

**Talablar:**
- Vaqt tugashi bilan davom etayotgan operatsiyalarni kuting
- db.Close() dan foydalanib ma'lumotlar bazasi ulanishlarini yoping
- Tugatish signallarini to'g'ri boshqaring
- Agar tugatish vaqti tugasa xato qaytaring

**Jarayon:**
1. Yangi so'rovlarni qabul qilishni to'xtating
2. Bajarilayotgan operatsiyalarni kuting
3. Ma'lumotlar bazasi ulanishlarini yoping
4. Toza chiqing yoki vaqt tugashi`,
            hint1: `Ma'lumotlar bazasini yopish uchun db.Close() dan foydalaning. Bu metod barcha ulanishlar poolga qaytguncha yoki yopilguncha bloklanadi.`,
            hint2: `db.Close() ni goroutine ga o'rang va tugatish paytida cheksiz blokirovkani oldini olish uchun kontekst vaqt tugashi bilan select dan foydalaning.`,
            whyItMatters: `To'g'ri tugatish ma'lumotlar yaxlitligi va foydalanuvchi tajribasi uchun juda muhimdir. Busiz, bajarilayotgan tranzaksiyalar bekor qilinishi, ma'lumotlar yo'qolishi va ulanishlar osilgan holda qolishi mumkin. Kubernetes va boshqa orkestratorlar SIGKILL dan oldin SIGTERM yuboradi va dasturingizga toza tugatish uchun vaqt beradi - lekin faqat agar uni to'g'ri boshqarsangiz.

**Ishlab chiqarish patterni:**
\`\`\`go
// Kubernetesda SIGTERM ni boshqarish
func main() {
    shutdown := make(chan os.Signal, 1)
    signal.Notify(shutdown, syscall.SIGTERM)

    <-shutdown
    log.Println("Shutting down gracefully...")

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    db.Close() // Tranzaksiyalar tugashini kutadi
}
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar yaxlitligini himoya qilish
- Deploy paytida so'rovlar yo'qotilmaydi
- Kubernetes bilan to'g'ri ishlash`
        }
    }
};

export default task;
