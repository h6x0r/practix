import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-context-timeout',
    title: 'Query with Context Timeout',
    difficulty: 'medium',
    tags: ['go', 'database', 'sql', 'context', 'timeout'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that queries users with a context timeout. The query should be cancelled if it takes longer than the specified duration, preventing long-running queries from blocking resources.

**Requirements:**
- Create context with timeout using context.WithTimeout
- Pass context to QueryContext
- Handle context.DeadlineExceeded error
- Properly cancel context with defer cancel()

**Type Definition:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

// TODO: Query users with a timeout
func QueryUsersWithTimeout(ctx context.Context, db *sql.DB, timeout time.Duration) ([]User, error) {
    panic("TODO: implement with context.WithTimeout")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUsersWithTimeout(ctx context.Context, db *sql.DB, timeout time.Duration) ([]User, error) {
    // Create context with timeout
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Execute query with timeout context
    query := "SELECT id, name, email FROM users"
    rows, err := db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Iterate through results
    var users []User
    for rows.Next() {
        var user User
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }

    // Check for errors during iteration
    if err := rows.Err(); err != nil {
        return nil, err
    }

    return users, nil
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
	// QueryUsersWithTimeout returns users successfully
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Alice", "alice@test.com").
		AddRow(2, "Bob", "bob@test.com")
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, err := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 2 {
		t.Errorf("expected 2 users, got %d", len(users))
	}
}

func Test2(t *testing.T) {
	// QueryUsersWithTimeout returns empty slice for no results
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"})
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, err := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 0 {
		t.Errorf("expected 0 users, got %d", len(users))
	}
}

func Test3(t *testing.T) {
	// QueryUsersWithTimeout handles timeout exceeded
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WillReturnError(context.DeadlineExceeded)

	_, err := QueryUsersWithTimeout(context.Background(), db, 1*time.Nanosecond)
	if err == nil {
		t.Error("expected error for timeout")
	}
}

func Test4(t *testing.T) {
	// QueryUsersWithTimeout handles cancelled context
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectQuery("SELECT").WillReturnError(context.Canceled)

	_, err := QueryUsersWithTimeout(ctx, db, 5*time.Second)
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test5(t *testing.T) {
	// QueryUsersWithTimeout handles query error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WillReturnError(errors.New("query error"))

	_, err := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if err == nil {
		t.Error("expected error")
	}
}

func Test6(t *testing.T) {
	// QueryUsersWithTimeout returns nil on error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WillReturnError(errors.New("error"))

	users, _ := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if users != nil {
		t.Error("expected nil users on error")
	}
}

func Test7(t *testing.T) {
	// QueryUsersWithTimeout handles single user
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Solo", "solo@test.com")
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, err := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 1 || users[0].Name != "Solo" {
		t.Errorf("expected 1 user named Solo, got %v", users)
	}
}

func Test8(t *testing.T) {
	// QueryUsersWithTimeout populates all fields
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(42, "Test User", "test@example.com")
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, _ := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if len(users) != 1 {
		t.Fatal("expected 1 user")
	}
	if users[0].ID != 42 || users[0].Email != "test@example.com" {
		t.Errorf("fields not populated correctly: %+v", users[0])
	}
}

func Test9(t *testing.T) {
	// QueryUsersWithTimeout preserves order
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "First", "first@test.com").
		AddRow(2, "Second", "second@test.com").
		AddRow(3, "Third", "third@test.com")
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, _ := QueryUsersWithTimeout(context.Background(), db, 5*time.Second)
	if len(users) != 3 {
		t.Fatal("expected 3 users")
	}
	if users[0].Name != "First" || users[2].Name != "Third" {
		t.Error("order not preserved")
	}
}

func Test10(t *testing.T) {
	// QueryUsersWithTimeout works with short timeout
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Fast", "fast@test.com")
	mock.ExpectQuery("SELECT").WillReturnRows(rows)

	users, err := QueryUsersWithTimeout(context.Background(), db, 100*time.Millisecond)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 1 {
		t.Errorf("expected 1 user, got %d", len(users))
	}
}
`,
    hint1: `Use context.WithTimeout(ctx, duration) to create a new context that will be cancelled after the timeout. Always defer cancel().`,
    hint2: `When the timeout is exceeded, QueryContext will return context.DeadlineExceeded error. You can check for this specific error if needed.`,
    whyItMatters: `Context timeouts are essential for preventing slow queries from blocking application resources. They enable you to implement graceful degradation, set SLAs, and prevent cascade failures in distributed systems.

**Production Pattern:**
\`\`\`go
// Protection from slow queries with timeout
ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
defer cancel()

users, err := QueryUsers(ctx, db)
if err == context.DeadlineExceeded {
    // Query took too long
    return cachedUsers, nil // Fallback
}
\`\`\`

**Practical Benefits:**
- Prevents cascade failures
- Enforces response time SLAs
- Automatic resource cleanup`,
    order: 5,
    translations: {
        ru: {
            title: 'Запрос с таймаутом контекста',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUsersWithTimeout(ctx context.Context, db *sql.DB, timeout time.Duration) ([]User, error) {
    // Создаем контекст с тайм-аутом
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Выполняем запрос с контекстом тайм-аута
    query := "SELECT id, name, email FROM users"
    rows, err := db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Итерируем по результатам
    var users []User
    for rows.Next() {
        var user User
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }

    // Проверяем ошибки во время итерации
    if err := rows.Err(); err != nil {
        return nil, err
    }

    return users, nil
}`,
            description: `Реализуйте функцию, которая запрашивает пользователей с тайм-аутом контекста. Запрос должен быть отменен, если он выполняется дольше указанной длительности, предотвращая блокировку ресурсов долгими запросами.

**Требования:**
- Создайте контекст с тайм-аутом используя context.WithTimeout
- Передайте контекст в QueryContext
- Обработайте ошибку context.DeadlineExceeded
- Правильно отмените контекст с defer cancel()

**Определение типа:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Используйте context.WithTimeout(ctx, duration) для создания нового контекста, который будет отменен после тайм-аута. Всегда используйте defer cancel().`,
            hint2: `Когда тайм-аут превышен, QueryContext вернет ошибку context.DeadlineExceeded. При необходимости можно проверить эту конкретную ошибку.`,
            whyItMatters: `Тайм-ауты контекста необходимы для предотвращения блокировки ресурсов приложения медленными запросами. Они позволяют реализовать graceful degradation, устанавливать SLA и предотвращать каскадные сбои в распределенных системах.

**Продакшен паттерн:**
\`\`\`go
// Защита от медленных запросов с тайм-аутом
ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
defer cancel()

users, err := QueryUsers(ctx, db)
if err == context.DeadlineExceeded {
    // Запрос занял слишком много времени
    return cachedUsers, nil // Резервный вариант
}
\`\`\`

**Практические преимущества:**
- Предотвращение каскадных сбоев
- Соблюдение SLA времени отклика
- Автоматическое освобождение ресурсов`
        },
        uz: {
            title: 'Kontekst timeout bilan so\'rov',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUsersWithTimeout(ctx context.Context, db *sql.DB, timeout time.Duration) ([]User, error) {
    // Vaqt tugashi bilan kontekst yaratamiz
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    // Vaqt tugashi konteksti bilan so'rovni bajaramiz
    query := "SELECT id, name, email FROM users"
    rows, err := db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Natijalar bo'yicha iteratsiya qilamiz
    var users []User
    for rows.Next() {
        var user User
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }

    // Iteratsiya davomida xatolarni tekshiramiz
    if err := rows.Err(); err != nil {
        return nil, err
    }

    return users, nil
}`,
            description: `Kontekst vaqt tugashi bilan foydalanuvchilarni so'raydigan funksiyani amalga oshiring. Agar so'rov belgilangan muddatdan uzoqroq davom etsa, u bekor qilinishi kerak, bu uzoq davom etadigan so'rovlarning resurslarni blokirovka qilishining oldini oladi.

**Talablar:**
- context.WithTimeout dan foydalanib vaqt tugashi bilan kontekst yarating
- Kontekstni QueryContext ga o'tkazing
- context.DeadlineExceeded xatosini boshqaring
- Kontekstni defer cancel() bilan to'g'ri bekor qiling

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Vaqt tugashidan keyin bekor qilinadigan yangi kontekst yaratish uchun context.WithTimeout(ctx, duration) dan foydalaning. Har doim defer cancel() dan foydalaning.`,
            hint2: `Vaqt tugashi oshib ketganda, QueryContext context.DeadlineExceeded xatosini qaytaradi. Agar kerak bo'lsa, bu aniq xatoni tekshirishingiz mumkin.`,
            whyItMatters: `Kontekst vaqt tugashi sekin so'rovlarning ilova resurslarini blokirovka qilishining oldini olish uchun zarur. Ular graceful degradation ni amalga oshirish, SLA larni o'rnatish va taqsimlangan tizimlarda kaskad nosozliklarni oldini olish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Vaqt tugashi bilan sekin so'rovlardan himoya
ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
defer cancel()

users, err := QueryUsers(ctx, db)
if err == context.DeadlineExceeded {
    // So'rov juda ko'p vaqt oldi
    return cachedUsers, nil // Zaxira variant
}
\`\`\`

**Amaliy foydalari:**
- Kaskad nosozliklarni oldini olish
- Javob vaqti SLA ni bajarish
- Avtomatik resurslarni ozod qilish`
        }
    }
};

export default task;
