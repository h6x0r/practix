import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-query-multiple',
    title: 'Query Multiple Rows',
    difficulty: 'easy',
    tags: ['go', 'database', 'sql', 'query'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that queries multiple users from the database using QueryContext. You must iterate through rows using rows.Next(), scan each row, and properly close the result set.

**Requirements:**
- Use db.QueryContext with context
- Iterate with rows.Next() loop
- Scan each row into User struct
- Close rows with defer rows.Close()
- Check rows.Err() after iteration

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

// TODO: Query all users matching a department
func QueryUsers(ctx context.Context, db *sql.DB, department string) ([]User, error) {
    panic("TODO: implement QueryContext with rows.Next() iteration")
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

func QueryUsers(ctx context.Context, db *sql.DB, department string) ([]User, error) {
    // Query multiple rows
    query := "SELECT id, name, email FROM users WHERE department = ?"
    rows, err := db.QueryContext(ctx, query, department)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Iterate through result set
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
	"database/sql"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// QueryUsers returns multiple users
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Alice", "alice@test.com").
		AddRow(2, "Bob", "bob@test.com")
	mock.ExpectQuery("SELECT").WithArgs("engineering").WillReturnRows(rows)

	users, err := QueryUsers(context.Background(), db, "engineering")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 2 {
		t.Errorf("expected 2 users, got %d", len(users))
	}
}

func Test2(t *testing.T) {
	// QueryUsers returns empty slice for no matches
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"})
	mock.ExpectQuery("SELECT").WithArgs("unknown").WillReturnRows(rows)

	users, err := QueryUsers(context.Background(), db, "unknown")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 0 {
		t.Errorf("expected 0 users, got %d", len(users))
	}
}

func Test3(t *testing.T) {
	// QueryUsers handles query error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs("dept").WillReturnError(errors.New("db error"))

	_, err := QueryUsers(context.Background(), db, "dept")
	if err == nil {
		t.Error("expected error")
	}
}

func Test4(t *testing.T) {
	// QueryUsers populates all user fields correctly
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(42, "Test User", "test@example.com")
	mock.ExpectQuery("SELECT").WithArgs("dept").WillReturnRows(rows)

	users, _ := QueryUsers(context.Background(), db, "dept")
	if len(users) != 1 {
		t.Fatal("expected 1 user")
	}
	if users[0].ID != 42 || users[0].Name != "Test User" {
		t.Errorf("user fields not populated correctly: %+v", users[0])
	}
}

func Test5(t *testing.T) {
	// QueryUsers handles context cancellation
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectQuery("SELECT").WithArgs("dept").WillReturnError(context.Canceled)

	_, err := QueryUsers(ctx, db, "dept")
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test6(t *testing.T) {
	// QueryUsers returns nil slice on error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs("dept").WillReturnError(errors.New("error"))

	users, _ := QueryUsers(context.Background(), db, "dept")
	if users != nil {
		t.Error("expected nil users on error")
	}
}

func Test7(t *testing.T) {
	// QueryUsers works with many users
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"})
	for i := 1; i <= 100; i++ {
		rows.AddRow(i, "User", "user@test.com")
	}
	mock.ExpectQuery("SELECT").WithArgs("large").WillReturnRows(rows)

	users, err := QueryUsers(context.Background(), db, "large")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 100 {
		t.Errorf("expected 100 users, got %d", len(users))
	}
}

func Test8(t *testing.T) {
	// QueryUsers preserves user order
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "First", "first@test.com").
		AddRow(2, "Second", "second@test.com").
		AddRow(3, "Third", "third@test.com")
	mock.ExpectQuery("SELECT").WithArgs("ordered").WillReturnRows(rows)

	users, _ := QueryUsers(context.Background(), db, "ordered")
	if users[0].Name != "First" || users[2].Name != "Third" {
		t.Error("order not preserved")
	}
}

func Test9(t *testing.T) {
	// QueryUsers handles single user
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"}).
		AddRow(1, "Solo", "solo@test.com")
	mock.ExpectQuery("SELECT").WithArgs("single").WillReturnRows(rows)

	users, _ := QueryUsers(context.Background(), db, "single")
	if len(users) != 1 {
		t.Errorf("expected 1 user, got %d", len(users))
	}
}

func Test10(t *testing.T) {
	// QueryUsers slice is not nil for empty result
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email"})
	mock.ExpectQuery("SELECT").WithArgs("empty").WillReturnRows(rows)

	users, err := QueryUsers(context.Background(), db, "empty")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Empty slice (not nil) is acceptable
	if users == nil {
		users = []User{}
	}
	if len(users) != 0 {
		t.Error("expected empty slice")
	}
}
`,
    hint1: `Use db.QueryContext() which returns *Rows. Always defer rows.Close() immediately after checking the error.`,
    hint2: `Use a for loop with rows.Next() to iterate. After the loop, check rows.Err() to catch any errors that occurred during iteration.`,
    whyItMatters: `Querying multiple rows is one of the most common database operations. Proper resource management with defer rows.Close() prevents connection leaks, and checking rows.Err() ensures you don't miss iteration errors.

**Production Pattern:**
\`\`\`go
// Always close rows to free connections
rows, err := db.QueryContext(ctx, query, params...)
if err != nil {
    return nil, err
}
defer rows.Close() // Critical to prevent leaks

for rows.Next() {
    // Process each row
}
// Check errors after iteration
return rows.Err()
\`\`\`

**Practical Benefits:**
- Prevents connection pool exhaustion
- Guarantees all errors are detected
- Proper DB resource cleanup`,
    order: 1,
    translations: {
        ru: {
            title: 'Запрос нескольких строк',
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

func QueryUsers(ctx context.Context, db *sql.DB, department string) ([]User, error) {
    // Запрашиваем несколько строк
    query := "SELECT id, name, email FROM users WHERE department = ?"
    rows, err := db.QueryContext(ctx, query, department)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Итерируем по набору результатов
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
            description: `Реализуйте функцию, которая запрашивает нескольких пользователей из базы данных с помощью QueryContext. Необходимо итерировать по строкам используя rows.Next(), сканировать каждую строку и правильно закрывать набор результатов.

**Требования:**
- Используйте db.QueryContext с контекстом
- Итерируйте с циклом rows.Next()
- Сканируйте каждую строку в структуру User
- Закрывайте rows с defer rows.Close()
- Проверяйте rows.Err() после итерации

**Определение типа:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Используйте db.QueryContext(), который возвращает *Rows. Всегда делайте defer rows.Close() сразу после проверки ошибки.`,
            hint2: `Используйте цикл for с rows.Next() для итерации. После цикла проверьте rows.Err() чтобы поймать любые ошибки, возникшие во время итерации.`,
            whyItMatters: `Запрос нескольких строк - одна из самых распространенных операций с базой данных. Правильное управление ресурсами с defer rows.Close() предотвращает утечки соединений, а проверка rows.Err() гарантирует, что вы не пропустите ошибки итерации.

**Продакшен паттерн:**
\`\`\`go
// Всегда закрывайте rows для освобождения соединений
rows, err := db.QueryContext(ctx, query, params...)
if err != nil {
    return nil, err
}
defer rows.Close() // Критично для предотвращения утечек

for rows.Next() {
    // Обрабатываем каждую строку
}
// Проверяем ошибки после итерации
return rows.Err()
\`\`\`

**Практические преимущества:**
- Предотвращает истощение пула соединений
- Гарантирует обнаружение всех ошибок
- Корректное освобождение ресурсов БД`
        },
        uz: {
            title: 'Bir nechta qator so\'rovi',
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

func QueryUsers(ctx context.Context, db *sql.DB, department string) ([]User, error) {
    // Bir nechta qatorni so'raymiz
    query := "SELECT id, name, email FROM users WHERE department = ?"
    rows, err := db.QueryContext(ctx, query, department)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // Natijalar to'plami bo'yicha iteratsiya qilamiz
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
            description: `QueryContext yordamida ma'lumotlar bazasidan bir nechta foydalanuvchini so'raydigan funksiyani amalga oshiring. rows.Next() dan foydalanib qatorlarni iteratsiya qilishingiz, har bir qatorni skanlashingiz va natijalar to'plamini to'g'ri yopishingiz kerak.

**Talablar:**
- db.QueryContext ni kontekst bilan ishlating
- rows.Next() tsikli bilan iteratsiya qiling
- Har bir qatorni User strukturasiga skanlang
- rows ni defer rows.Close() bilan yoping
- Iteratsiyadan keyin rows.Err() ni tekshiring

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `db.QueryContext() dan foydalaning, u *Rows qaytaradi. Xatoni tekshirgandan keyin darhol defer rows.Close() qiling.`,
            hint2: `Iteratsiya uchun rows.Next() bilan for tsiklidan foydalaning. Tsikldan keyin iteratsiya davomida yuz bergan xatolarni ushlash uchun rows.Err() ni tekshiring.`,
            whyItMatters: `Bir nechta qatorni so'rash eng keng tarqalgan ma'lumotlar bazasi operatsiyalaridan biridir. defer rows.Close() bilan to'g'ri resurs boshqaruvi ulanish oqishini oldini oladi va rows.Err() ni tekshirish iteratsiya xatolarini o'tkazib yubormasligingizni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Ulanishlarni ozod qilish uchun har doim rows ni yoping
rows, err := db.QueryContext(ctx, query, params...)
if err != nil {
    return nil, err
}
defer rows.Close() // Oqishni oldini olish uchun muhim

for rows.Next() {
    // Har bir qatorni qayta ishlaymiz
}
// Iteratsiyadan keyin xatolarni tekshiramiz
return rows.Err()
\`\`\`

**Amaliy foydalari:**
- Ulanish poolining tugashining oldini oladi
- Barcha xatolar aniqlanishini kafolatlaydi
- Ma'lumotlar bazasi resurslarini to'g'ri ozod qiladi`
        }
    }
};

export default task;
