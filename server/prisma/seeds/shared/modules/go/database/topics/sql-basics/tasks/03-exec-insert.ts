import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-exec-insert',
    title: 'Execute Insert Statement',
    difficulty: 'easy',
    tags: ['go', 'database', 'sql', 'insert'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that inserts a new user into the database using ExecContext. You must retrieve the last inserted ID from the Result and return it.

**Requirements:**
- Use db.ExecContext with context
- Execute INSERT statement
- Call result.LastInsertId()
- Handle errors from both Exec and LastInsertId

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

// TODO: Insert a user and return the auto-generated ID
func InsertUser(ctx context.Context, db *sql.DB, name, email string) (int64, error) {
    panic("TODO: implement ExecContext with LastInsertId")
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

func InsertUser(ctx context.Context, db *sql.DB, name, email string) (int64, error) {
    // Execute INSERT statement
    query := "INSERT INTO users (name, email) VALUES (?, ?)"
    result, err := db.ExecContext(ctx, query, name, email)
    if err != nil {
        return 0, err
    }

    // Get the last inserted ID
    id, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    return id, nil
}`,
	testCode: `package dbx

import (
	"context"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// InsertUser returns last insert ID
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Alice", "alice@test.com").
		WillReturnResult(sqlmock.NewResult(42, 1))

	id, err := InsertUser(context.Background(), db, "Alice", "alice@test.com")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if id != 42 {
		t.Errorf("expected id 42, got %d", id)
	}
}

func Test2(t *testing.T) {
	// InsertUser handles exec error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Bob", "bob@test.com").
		WillReturnError(errors.New("exec error"))

	_, err := InsertUser(context.Background(), db, "Bob", "bob@test.com")
	if err == nil {
		t.Error("expected error")
	}
}

func Test3(t *testing.T) {
	// InsertUser returns 0 on error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Error", "error@test.com").
		WillReturnError(errors.New("db error"))

	id, _ := InsertUser(context.Background(), db, "Error", "error@test.com")
	if id != 0 {
		t.Errorf("expected 0 on error, got %d", id)
	}
}

func Test4(t *testing.T) {
	// InsertUser works with different IDs
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Test", "test@test.com").
		WillReturnResult(sqlmock.NewResult(999, 1))

	id, err := InsertUser(context.Background(), db, "Test", "test@test.com")
	if err != nil || id != 999 {
		t.Errorf("expected id 999, got %d, err: %v", id, err)
	}
}

func Test5(t *testing.T) {
	// InsertUser handles context cancellation
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectExec("INSERT").WithArgs("Cancel", "cancel@test.com").
		WillReturnError(context.Canceled)

	_, err := InsertUser(ctx, db, "Cancel", "cancel@test.com")
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test6(t *testing.T) {
	// InsertUser uses correct query format
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT INTO users").WithArgs("Format", "format@test.com").
		WillReturnResult(sqlmock.NewResult(1, 1))

	_, err := InsertUser(context.Background(), db, "Format", "format@test.com")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test7(t *testing.T) {
	// InsertUser with empty name
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("", "empty@test.com").
		WillReturnResult(sqlmock.NewResult(5, 1))

	id, err := InsertUser(context.Background(), db, "", "empty@test.com")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if id != 5 {
		t.Errorf("expected id 5, got %d", id)
	}
}

func Test8(t *testing.T) {
	// InsertUser returns positive ID
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Positive", "pos@test.com").
		WillReturnResult(sqlmock.NewResult(1, 1))

	id, err := InsertUser(context.Background(), db, "Positive", "pos@test.com")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if id <= 0 {
		t.Errorf("expected positive id, got %d", id)
	}
}

func Test9(t *testing.T) {
	// InsertUser handles large ID
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Large", "large@test.com").
		WillReturnResult(sqlmock.NewResult(9999999, 1))

	id, err := InsertUser(context.Background(), db, "Large", "large@test.com")
	if err != nil || id != 9999999 {
		t.Errorf("expected id 9999999, got %d", id)
	}
}

func Test10(t *testing.T) {
	// InsertUser with special characters in email
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectExec("INSERT").WithArgs("Special", "user+tag@example.com").
		WillReturnResult(sqlmock.NewResult(10, 1))

	id, err := InsertUser(context.Background(), db, "Special", "user+tag@example.com")
	if err != nil || id != 10 {
		t.Errorf("expected id 10, got %d, err: %v", id, err)
	}
}
`,
    hint1: `Use db.ExecContext() for statements that don't return rows (INSERT, UPDATE, DELETE). It returns sql.Result.`,
    hint2: `Call result.LastInsertId() to get the auto-generated ID. Note that not all databases support this (PostgreSQL requires RETURNING clause).`,
    whyItMatters: `ExecContext is essential for data modification operations. Understanding how to retrieve auto-generated IDs is critical for workflows where you need to reference newly created records immediately.

**Production Pattern:**
\`\`\`go
// Insert and get ID for related operations
result, err := db.ExecContext(ctx, "INSERT INTO users (...) VALUES (?)", data)
if err != nil {
    return 0, err
}

userID, err := result.LastInsertId()
// Use userID to create related records
_, err = db.ExecContext(ctx, "INSERT INTO profiles (user_id, ...) VALUES (?)", userID)
\`\`\`

**Practical Benefits:**
- Immediate access to new record ID
- Support for related inserts
- Atomic data creation operations`,
    order: 2,
    translations: {
        ru: {
            title: 'Выполнение INSERT',
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

func InsertUser(ctx context.Context, db *sql.DB, name, email string) (int64, error) {
    // Выполняем INSERT запрос
    query := "INSERT INTO users (name, email) VALUES (?, ?)"
    result, err := db.ExecContext(ctx, query, name, email)
    if err != nil {
        return 0, err
    }

    // Получаем ID последней вставленной записи
    id, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    return id, nil
}`,
            description: `Реализуйте функцию, которая вставляет нового пользователя в базу данных с помощью ExecContext. Необходимо получить ID последней вставленной записи из Result и вернуть его.

**Требования:**
- Используйте db.ExecContext с контекстом
- Выполните INSERT запрос
- Вызовите result.LastInsertId()
- Обработайте ошибки от Exec и LastInsertId

**Определение типа:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Используйте db.ExecContext() для запросов, которые не возвращают строки (INSERT, UPDATE, DELETE). Он возвращает sql.Result.`,
            hint2: `Вызовите result.LastInsertId() чтобы получить автогенерируемый ID. Обратите внимание, что не все базы данных поддерживают это (PostgreSQL требует RETURNING клаузу).`,
            whyItMatters: `ExecContext необходим для операций модификации данных. Понимание того, как получать автогенерируемые ID, критически важно для рабочих процессов, где нужно ссылаться на только что созданные записи.

**Продакшен паттерн:**
\`\`\`go
// Вставка и получение ID для связанных операций
result, err := db.ExecContext(ctx, "INSERT INTO users (...) VALUES (?)", data)
if err != nil {
    return 0, err
}

userID, err := result.LastInsertId()
// Используем userID для создания связанных записей
_, err = db.ExecContext(ctx, "INSERT INTO profiles (user_id, ...) VALUES (?)", userID)
\`\`\`

**Практические преимущества:**
- Немедленный доступ к ID новой записи
- Поддержка связанных вставок
- Атомарные операции создания данных`
        },
        uz: {
            title: 'INSERT bajarish',
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

func InsertUser(ctx context.Context, db *sql.DB, name, email string) (int64, error) {
    // INSERT so'rovini bajaramiz
    query := "INSERT INTO users (name, email) VALUES (?, ?)"
    result, err := db.ExecContext(ctx, query, name, email)
    if err != nil {
        return 0, err
    }

    // Oxirgi qo'shilgan yozuvning ID sini olamiz
    id, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    return id, nil
}`,
            description: `ExecContext yordamida ma'lumotlar bazasiga yangi foydalanuvchini qo'shadigan funksiyani amalga oshiring. Result dan oxirgi qo'shilgan ID ni olishingiz va uni qaytarishingiz kerak.

**Talablar:**
- db.ExecContext ni kontekst bilan ishlating
- INSERT so'rovini bajaring
- result.LastInsertId() ni chaqiring
- Exec va LastInsertId dan xatolarni boshqaring

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Qatorlarni qaytarmaydigan so'rovlar (INSERT, UPDATE, DELETE) uchun db.ExecContext() dan foydalaning. U sql.Result qaytaradi.`,
            hint2: `Avtomatik yaratilgan ID ni olish uchun result.LastInsertId() ni chaqiring. Barcha ma'lumotlar bazalari buni qo'llab-quvvatlamaydi (PostgreSQL RETURNING bandini talab qiladi).`,
            whyItMatters: `ExecContext ma'lumotlarni o'zgartirish operatsiyalari uchun zarur. Avtomatik yaratilgan ID larni qanday olishni tushunish, yangi yaratilgan yozuvlarga darhol murojaat qilish kerak bo'lgan ish oqimlari uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`go
// Bog'langan operatsiyalar uchun qo'shish va ID olish
result, err := db.ExecContext(ctx, "INSERT INTO users (...) VALUES (?)", data)
if err != nil {
    return 0, err
}

userID, err := result.LastInsertId()
// Bog'langan yozuvlarni yaratish uchun userID dan foydalanamiz
_, err = db.ExecContext(ctx, "INSERT INTO profiles (user_id, ...) VALUES (?)", userID)
\`\`\`

**Amaliy foydalari:**
- Yangi yozuv ID siga darhol kirish
- Bog'langan qo'shishlarni qo'llab-quvvatlash
- Atom ma'lumot yaratish operatsiyalari`
        }
    }
};

export default task;
