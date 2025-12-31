import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-query-row',
    title: 'Query Single Row',
    difficulty: 'easy',
    tags: ['go', 'database', 'sql', 'query'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that queries a single user from the database using QueryRowContext. You must handle the Scan operation and return appropriate errors for missing rows or scan failures.

**Requirements:**
- Use db.QueryRowContext with context
- Scan into User struct fields
- Return sql.ErrNoRows if user not found
- Handle scan errors properly

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
)

type User struct {
    ID    int64
    Name  string
    Email string
}

// TODO: Query a single user by ID
func QueryUser(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    panic("TODO: implement QueryRowContext with Scan")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUser(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Query single row and scan into user struct
    var user User
    query := "SELECT id, name, email FROM users WHERE id = ?"

    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }

    return &user, nil
}`,
	testCode: `package dbx

import (
	"context"
	"database/sql"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// QueryUser returns user when found
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "John", "john@example.com")
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUser(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Name != "John" {
		t.Errorf("expected name John, got %s", user.Name)
	}
}

func Test2(t *testing.T) {
	// QueryUser returns error when not found
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs(999).WillReturnError(sql.ErrNoRows)

	_, err := QueryUser(context.Background(), db, 999)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Errorf("expected sql.ErrNoRows, got %v", err)
	}
}

func Test3(t *testing.T) {
	// QueryUser populates all User fields
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(42, "Alice", "alice@test.com")
	mock.ExpectQuery("SELECT").WithArgs(42).WillReturnRows(rows)

	user, err := QueryUser(context.Background(), db, 42)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.ID != 42 {
		t.Errorf("expected ID 42, got %d", user.ID)
	}
	if user.Email != "alice@test.com" {
		t.Errorf("expected email alice@test.com, got %s", user.Email)
	}
}

func Test4(t *testing.T) {
	// QueryUser handles context cancellation
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnError(context.Canceled)

	_, err := QueryUser(ctx, db, 1)
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test5(t *testing.T) {
	// QueryUser uses QueryRowContext correctly
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(5, "Test", "test@example.com")
	mock.ExpectQuery("SELECT id, name, email FROM users WHERE id").
		WithArgs(5).WillReturnRows(rows)

	user, _ := QueryUser(context.Background(), db, 5)
	if user == nil {
		t.Error("expected user, got nil")
	}
}

func Test6(t *testing.T) {
	// QueryUser returns nil user on error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnError(sql.ErrNoRows)

	user, _ := QueryUser(context.Background(), db, 1)
	if user != nil {
		t.Error("expected nil user on error")
	}
}

func Test7(t *testing.T) {
	// QueryUser handles scan error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnError(errors.New("scan error"))

	_, err := QueryUser(context.Background(), db, 1)
	if err == nil {
		t.Error("expected error")
	}
}

func Test8(t *testing.T) {
	// QueryUser works with different IDs
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(100, "User100", "user100@test.com")
	mock.ExpectQuery("SELECT").WithArgs(100).WillReturnRows(rows)

	user, err := QueryUser(context.Background(), db, 100)
	if err != nil || user.ID != 100 {
		t.Errorf("expected ID 100, got %v, err: %v", user, err)
	}
}

func Test9(t *testing.T) {
	// QueryUser returns valid pointer
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Ptr", "ptr@test.com")
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, _ := QueryUser(context.Background(), db, 1)
	if user == nil {
		t.Error("expected non-nil pointer")
	}
}

func Test10(t *testing.T) {
	// QueryUser handles empty string fields
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "", "")
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUser(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Name != "" || user.Email != "" {
		t.Error("expected empty strings")
	}
}
`,
    hint1: `Use db.QueryRowContext() which returns a *Row. Call .Scan() on it to populate the User struct fields.`,
    hint2: `QueryRow always returns a non-nil Row. The error, if any, is deferred until Scan is called. sql.ErrNoRows is returned when no row matches.`,
    whyItMatters: `QueryRowContext is the foundation of database operations in Go. Understanding how to query a single row with context cancellation support is essential for building timeout-aware and cancellable database operations.

**Production Pattern:**
\`\`\`go
// Query with timeout to protect from slow queries
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

user, err := QueryUser(ctx, db, userID)
if err != nil {
    return fmt.Errorf("failed to get user: %w", err)
}
\`\`\`

**Practical Benefits:**
- Automatic cancellation on timeout
- Prevents resource blocking
- Better fault tolerance in distributed systems`,
    order: 0,
    translations: {
        ru: {
            title: 'Запрос одной строки',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUser(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Запрашиваем одну строку и сканируем в структуру пользователя
    var user User
    query := "SELECT id, name, email FROM users WHERE id = ?"

    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }

    return &user, nil
}`,
            description: `Реализуйте функцию, которая запрашивает одного пользователя из базы данных с помощью QueryRowContext. Необходимо обработать операцию Scan и вернуть соответствующие ошибки для отсутствующих строк или ошибок сканирования.

**Требования:**
- Используйте db.QueryRowContext с контекстом
- Сканируйте в поля структуры User
- Верните sql.ErrNoRows если пользователь не найден
- Правильно обрабатывайте ошибки сканирования

**Определение типа:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Используйте db.QueryRowContext(), который возвращает *Row. Вызовите .Scan() на нем для заполнения полей структуры User.`,
            hint2: `QueryRow всегда возвращает ненулевой Row. Ошибка, если она есть, откладывается до вызова Scan. sql.ErrNoRows возвращается когда нет совпадающей строки.`,
            whyItMatters: `QueryRowContext является основой операций с базой данных в Go. Понимание того, как запрашивать одну строку с поддержкой отмены контекста, необходимо для создания операций с базой данных, учитывающих тайм-ауты и отмену.

**Продакшен паттерн:**
\`\`\`go
// Запрос с тайм-аутом для защиты от медленных запросов
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

user, err := QueryUser(ctx, db, userID)
if err != nil {
    return fmt.Errorf("failed to get user: %w", err)
}
\`\`\`

**Практические преимущества:**
- Автоматическая отмена при превышении времени
- Предотвращение блокировки ресурсов
- Лучшая отказоустойчивость в распределенных системах`
        },
        uz: {
            title: 'Bitta qator so\'rovi',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID    int64
    Name  string
    Email string
}

func QueryUser(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Bitta qatorni so'rab, foydalanuvchi strukturasiga skanlaymiz
    var user User
    query := "SELECT id, name, email FROM users WHERE id = ?"

    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }

    return &user, nil
}`,
            description: `QueryRowContext yordamida ma'lumotlar bazasidan bitta foydalanuvchini so'raydigan funksiyani amalga oshiring. Scan operatsiyasini boshqarishingiz va yo'qolgan qatorlar yoki skanlash xatolari uchun tegishli xatolarni qaytarishingiz kerak.

**Talablar:**
- db.QueryRowContext ni kontekst bilan ishlating
- User strukturasi maydonlariga skanlang
- Agar foydalanuvchi topilmasa sql.ErrNoRows qaytaring
- Skanlash xatolarini to'g'ri boshqaring

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `db.QueryRowContext() dan foydalaning, u *Row qaytaradi. User strukturasi maydonlarini to'ldirish uchun unda .Scan() ni chaqiring.`,
            hint2: `QueryRow har doim null bo'lmagan Row qaytaradi. Agar xato bo'lsa, u Scan chaqirilgunga qadar kechiktiriladi. Hech qanday qator mos kelmasa sql.ErrNoRows qaytariladi.`,
            whyItMatters: `QueryRowContext Go da ma'lumotlar bazasi operatsiyalarining asosi hisoblanadi. Kontekstni bekor qilishni qo'llab-quvvatlaydigan bitta qatorni so'rashni tushunish, vaqt tugashi va bekor qilinadigan ma'lumotlar bazasi operatsiyalarini yaratish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`go
// Sekin so'rovlardan himoya qilish uchun vaqt tugashi bilan so'rov
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

user, err := QueryUser(ctx, db, userID)
if err != nil {
    return fmt.Errorf("failed to get user: %w", err)
}
\`\`\`

**Amaliy foydalari:**
- Vaqt oshganda avtomatik bekor qilish
- Resurslar blokirovkasining oldini olish
- Taqsimlangan tizimlarda yaxshiroq bardoshlilik`
        }
    }
};

export default task;
