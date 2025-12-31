import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-prepared-statement',
    title: 'Prepared Statement',
    difficulty: 'medium',
    tags: ['go', 'database', 'sql', 'prepared-statement'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that uses a prepared statement to query multiple users efficiently. Prepared statements are compiled once and reused, improving performance for repeated queries.

**Requirements:**
- Use db.PrepareContext to create statement
- Defer stmt.Close() for cleanup
- Execute statement multiple times with different parameters
- Collect all results into a slice

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

// TODO: Query multiple users by IDs using a prepared statement
func QueryUsersByIDs(ctx context.Context, db *sql.DB, ids []int64) ([]User, error) {
    panic("TODO: implement PrepareContext and reuse the statement")
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

func QueryUsersByIDs(ctx context.Context, db *sql.DB, ids []int64) ([]User, error) {
    // Prepare statement once
    query := "SELECT id, name, email FROM users WHERE id = ?"
    stmt, err := db.PrepareContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer stmt.Close()

    // Execute statement multiple times
    var users []User
    for _, id := range ids {
        var user User
        err := stmt.QueryRowContext(ctx, id).Scan(&user.ID, &user.Name, &user.Email)
        if err != nil {
            if err == sql.ErrNoRows {
                continue // Skip missing users
            }
            return nil, err
        }
        users = append(users, user)
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
	// QueryUsersByIDs returns users for valid IDs
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").ExpectQuery().WithArgs(1).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "email"}).
			AddRow(1, "Alice", "alice@test.com"))

	users, err := QueryUsersByIDs(context.Background(), db, []int64{1})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 1 {
		t.Errorf("expected 1 user, got %d", len(users))
	}
}

func Test2(t *testing.T) {
	// QueryUsersByIDs returns empty for no IDs
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT")

	users, err := QueryUsersByIDs(context.Background(), db, []int64{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 0 {
		t.Errorf("expected 0 users, got %d", len(users))
	}
}

func Test3(t *testing.T) {
	// QueryUsersByIDs skips missing users
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").ExpectQuery().WithArgs(999).
		WillReturnError(sql.ErrNoRows)

	users, err := QueryUsersByIDs(context.Background(), db, []int64{999})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(users) != 0 {
		t.Errorf("expected 0 users for missing ID, got %d", len(users))
	}
}

func Test4(t *testing.T) {
	// QueryUsersByIDs handles multiple IDs
	db, mock, _ := sqlmock.New()
	defer db.Close()

	prep := mock.ExpectPrepare("SELECT")
	prep.ExpectQuery().WithArgs(1).WillReturnRows(
		sqlmock.NewRows([]string{"id", "name", "email"}).AddRow(1, "A", "a@test.com"))
	prep.ExpectQuery().WithArgs(2).WillReturnRows(
		sqlmock.NewRows([]string{"id", "name", "email"}).AddRow(2, "B", "b@test.com"))

	users, _ := QueryUsersByIDs(context.Background(), db, []int64{1, 2})
	if len(users) != 2 {
		t.Errorf("expected 2 users, got %d", len(users))
	}
}

func Test5(t *testing.T) {
	// QueryUsersByIDs handles prepare error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").WillReturnError(errors.New("prepare error"))

	_, err := QueryUsersByIDs(context.Background(), db, []int64{1})
	if err == nil {
		t.Error("expected error")
	}
}

func Test6(t *testing.T) {
	// QueryUsersByIDs returns nil on prepare error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").WillReturnError(errors.New("error"))

	users, _ := QueryUsersByIDs(context.Background(), db, []int64{1})
	if users != nil {
		t.Error("expected nil users on error")
	}
}

func Test7(t *testing.T) {
	// QueryUsersByIDs populates user fields
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").ExpectQuery().WithArgs(42).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "email"}).
			AddRow(42, "Test", "test@example.com"))

	users, _ := QueryUsersByIDs(context.Background(), db, []int64{42})
	if len(users) != 1 || users[0].ID != 42 || users[0].Email != "test@example.com" {
		t.Errorf("user fields not populated correctly: %+v", users)
	}
}

func Test8(t *testing.T) {
	// QueryUsersByIDs handles context cancellation
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectPrepare("SELECT").WillReturnError(context.Canceled)

	_, err := QueryUsersByIDs(ctx, db, []int64{1})
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test9(t *testing.T) {
	// QueryUsersByIDs preserves order
	db, mock, _ := sqlmock.New()
	defer db.Close()

	prep := mock.ExpectPrepare("SELECT")
	prep.ExpectQuery().WithArgs(1).WillReturnRows(
		sqlmock.NewRows([]string{"id", "name", "email"}).AddRow(1, "First", "first@test.com"))
	prep.ExpectQuery().WithArgs(2).WillReturnRows(
		sqlmock.NewRows([]string{"id", "name", "email"}).AddRow(2, "Second", "second@test.com"))

	users, _ := QueryUsersByIDs(context.Background(), db, []int64{1, 2})
	if len(users) >= 2 && users[0].Name != "First" {
		t.Error("order not preserved")
	}
}

func Test10(t *testing.T) {
	// QueryUsersByIDs handles query error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectPrepare("SELECT").ExpectQuery().WithArgs(1).
		WillReturnError(errors.New("query error"))

	_, err := QueryUsersByIDs(context.Background(), db, []int64{1})
	if err == nil {
		t.Error("expected error")
	}
}
`,
    hint1: `Use db.PrepareContext() to compile the SQL statement once. It returns *Stmt which you should close with defer.`,
    hint2: `The prepared statement can be executed multiple times using stmt.QueryRowContext() or stmt.QueryContext() with different parameters.`,
    whyItMatters: `Prepared statements offer significant performance benefits when executing the same query multiple times with different parameters. They also provide protection against SQL injection and reduce parsing overhead on the database server.

**Production Pattern:**
\`\`\`go
// Prepare once, execute many times
stmt, err := db.PrepareContext(ctx, "SELECT * FROM users WHERE id = ?")
defer stmt.Close()

for _, id := range userIDs {
    // Reuse compiled query
    var user User
    stmt.QueryRowContext(ctx, id).Scan(&user)
}
\`\`\`

**Practical Benefits:**
- Up to 50% performance improvement on repeated queries
- Automatic SQL injection protection
- Reduces DB server load`,
    order: 3,
    translations: {
        ru: {
            title: 'Подготовленный запрос',
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

func QueryUsersByIDs(ctx context.Context, db *sql.DB, ids []int64) ([]User, error) {
    // Подготавливаем запрос один раз
    query := "SELECT id, name, email FROM users WHERE id = ?"
    stmt, err := db.PrepareContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer stmt.Close()

    // Выполняем запрос несколько раз
    var users []User
    for _, id := range ids {
        var user User
        err := stmt.QueryRowContext(ctx, id).Scan(&user.ID, &user.Name, &user.Email)
        if err != nil {
            if err == sql.ErrNoRows {
                continue // Пропускаем отсутствующих пользователей
            }
            return nil, err
        }
        users = append(users, user)
    }

    return users, nil
}`,
            description: `Реализуйте функцию, которая использует подготовленный запрос для эффективного получения нескольких пользователей. Подготовленные запросы компилируются один раз и переиспользуются, улучшая производительность для повторяющихся запросов.

**Требования:**
- Используйте db.PrepareContext для создания запроса
- Используйте defer stmt.Close() для очистки
- Выполните запрос несколько раз с разными параметрами
- Соберите все результаты в срез

**Определение типа:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `Используйте db.PrepareContext() для компиляции SQL запроса один раз. Он возвращает *Stmt, который следует закрыть с помощью defer.`,
            hint2: `Подготовленный запрос может быть выполнен несколько раз используя stmt.QueryRowContext() или stmt.QueryContext() с разными параметрами.`,
            whyItMatters: `Подготовленные запросы предлагают значительные преимущества производительности при выполнении одного и того же запроса несколько раз с разными параметрами. Они также обеспечивают защиту от SQL инъекций и уменьшают накладные расходы на парсинг на сервере базы данных.

**Продакшен паттерн:**
\`\`\`go
// Подготовка один раз, выполнение много раз
stmt, err := db.PrepareContext(ctx, "SELECT * FROM users WHERE id = ?")
defer stmt.Close()

for _, id := range userIDs {
    // Повторное использование скомпилированного запроса
    var user User
    stmt.QueryRowContext(ctx, id).Scan(&user)
}
\`\`\`

**Практические преимущества:**
- До 50% улучшение производительности при повторных запросах
- Автоматическая защита от SQL инъекций
- Снижение нагрузки на сервер БД`
        },
        uz: {
            title: 'Prepared statement',
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

func QueryUsersByIDs(ctx context.Context, db *sql.DB, ids []int64) ([]User, error) {
    // So'rovni bir marta tayyorlaymiz
    query := "SELECT id, name, email FROM users WHERE id = ?"
    stmt, err := db.PrepareContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer stmt.Close()

    // So'rovni bir necha marta bajaramiz
    var users []User
    for _, id := range ids {
        var user User
        err := stmt.QueryRowContext(ctx, id).Scan(&user.ID, &user.Name, &user.Email)
        if err != nil {
            if err == sql.ErrNoRows {
                continue // Yo'qolgan foydalanuvchilarni o'tkazib yuboramiz
            }
            return nil, err
        }
        users = append(users, user)
    }

    return users, nil
}`,
            description: `Bir nechta foydalanuvchilarni samarali olish uchun tayyorlangan so'rovdan foydalanadigan funksiyani amalga oshiring. Tayyorlangan so'rovlar bir marta kompilyatsiya qilinadi va qayta ishlatiladi, takroriy so'rovlar uchun unumdorlikni yaxshilaydi.

**Talablar:**
- So'rovni yaratish uchun db.PrepareContext dan foydalaning
- Tozalash uchun defer stmt.Close() dan foydalaning
- So'rovni turli parametrlar bilan bir necha marta bajaring
- Barcha natijalarni slaysga to'plang

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID    int64
    Name  string
    Email string
}
\`\`\``,
            hint1: `SQL so'rovini bir marta kompilyatsiya qilish uchun db.PrepareContext() dan foydalaning. U *Stmt qaytaradi, uni defer bilan yopishingiz kerak.`,
            hint2: `Tayyorlangan so'rov turli parametrlar bilan stmt.QueryRowContext() yoki stmt.QueryContext() yordamida bir necha marta bajarilishi mumkin.`,
            whyItMatters: `Tayyorlangan so'rovlar bir xil so'rovni turli parametrlar bilan bir necha marta bajarishda sezilarli unumdorlik afzalliklarini taklif qiladi. Ular SQL in'eksiyasidan himoya qiladi va ma'lumotlar bazasi serverida tahlil qilish yukini kamaytiradi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Bir marta tayyorlash, ko'p marta bajarish
stmt, err := db.PrepareContext(ctx, "SELECT * FROM users WHERE id = ?")
defer stmt.Close()

for _, id := range userIDs {
    // Kompilyatsiya qilingan so'rovni qayta ishlatish
    var user User
    stmt.QueryRowContext(ctx, id).Scan(&user)
}
\`\`\`

**Amaliy foydalari:**
- Takroriy so'rovlarda 50% gacha unumdorlik yaxshilanishi
- Avtomatik SQL in'eksiyasidan himoya
- Ma'lumotlar bazasi serveridagi yukni kamaytirish`
        }
    }
};

export default task;
