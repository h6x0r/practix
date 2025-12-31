import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-null-handling',
    title: 'Handle NULL Values',
    difficulty: 'medium',
    tags: ['go', 'database', 'sql', 'null'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that queries a user with nullable fields from the database. You must use sql.NullString and sql.NullInt64 to handle NULL values correctly and convert them to pointer types.

**Requirements:**
- Use sql.NullString for nullable string fields
- Use sql.NullInt64 for nullable integer fields
- Convert null types to pointers (*string, *int64)
- Set pointers to nil when value is NULL

**Type Definition:**
\`\`\`go
type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}
\`\`\``,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}

// TODO: Query user with nullable fields
func QueryUserWithNulls(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    panic("TODO: implement with sql.NullString and sql.NullInt64")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}

func QueryUserWithNulls(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Scan into null types
    var user User
    var email sql.NullString
    var age sql.NullInt64

    query := "SELECT id, name, email, age FROM users WHERE id = ?"
    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &email, &age)
    if err != nil {
        return nil, err
    }

    // Convert null types to pointers
    if email.Valid {
        user.Email = &email.String
    }

    if age.Valid {
        user.Age = &age.Int64
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
	// QueryUserWithNulls returns user with all null fields as nil
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "John", nil, nil)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Email != nil || user.Age != nil {
		t.Errorf("expected nil for null fields, got Email=%v, Age=%v", user.Email, user.Age)
	}
}

func Test2(t *testing.T) {
	// QueryUserWithNulls returns user with valid email
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "John", "john@test.com", nil)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Email == nil || *user.Email != "john@test.com" {
		t.Errorf("expected email john@test.com, got %v", user.Email)
	}
}

func Test3(t *testing.T) {
	// QueryUserWithNulls returns user with valid age
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "John", nil, 25)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Age == nil || *user.Age != 25 {
		t.Errorf("expected age 25, got %v", user.Age)
	}
}

func Test4(t *testing.T) {
	// QueryUserWithNulls returns user with both fields valid
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "John", "john@test.com", 30)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Email == nil || user.Age == nil {
		t.Errorf("expected both fields to be non-nil")
	}
}

func Test5(t *testing.T) {
	// QueryUserWithNulls returns error when not found
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs(999).WillReturnError(sql.ErrNoRows)

	_, err := QueryUserWithNulls(context.Background(), db, 999)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Errorf("expected sql.ErrNoRows, got %v", err)
	}
}

func Test6(t *testing.T) {
	// QueryUserWithNulls returns nil user on error
	db, mock, _ := sqlmock.New()
	defer db.Close()

	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnError(errors.New("db error"))

	user, _ := QueryUserWithNulls(context.Background(), db, 1)
	if user != nil {
		t.Error("expected nil user on error")
	}
}

func Test7(t *testing.T) {
	// QueryUserWithNulls handles context cancellation
	db, mock, _ := sqlmock.New()
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnError(context.Canceled)

	_, err := QueryUserWithNulls(ctx, db, 1)
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func Test8(t *testing.T) {
	// QueryUserWithNulls populates ID and Name correctly
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(42, "Alice", nil, nil)
	mock.ExpectQuery("SELECT").WithArgs(42).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 42)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.ID != 42 || user.Name != "Alice" {
		t.Errorf("expected ID=42 Name=Alice, got ID=%d Name=%s", user.ID, user.Name)
	}
}

func Test9(t *testing.T) {
	// QueryUserWithNulls handles empty string as valid
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "Test", "", nil)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Email == nil || *user.Email != "" {
		t.Errorf("expected empty string, got %v", user.Email)
	}
}

func Test10(t *testing.T) {
	// QueryUserWithNulls handles zero age as valid
	db, mock, _ := sqlmock.New()
	defer db.Close()

	rows := sqlmock.NewRows([]string{"id", "name", "email", "age"}).
		AddRow(1, "Baby", nil, 0)
	mock.ExpectQuery("SELECT").WithArgs(1).WillReturnRows(rows)

	user, err := QueryUserWithNulls(context.Background(), db, 1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if user.Age == nil || *user.Age != 0 {
		t.Errorf("expected age 0, got %v", user.Age)
	}
}
`,
    hint1: `Use sql.NullString and sql.NullInt64 types for scanning. These types have Valid and Value fields.`,
    hint2: `After scanning, check the Valid field. If true, assign the value to the pointer. If false, leave the pointer as nil.`,
    whyItMatters: `NULL handling is critical in real-world applications where optional fields are common. Using sql.Null* types prevents runtime panics and allows you to distinguish between zero values and NULL values correctly.

**Production Pattern:**
\`\`\`go
// Safe NULL handling
var email sql.NullString
db.QueryRowContext(ctx, query).Scan(&id, &email)

// Distinguish empty string from NULL
if email.Valid {
    user.Email = &email.String // Has value
} else {
    user.Email = nil // NULL in DB
}
\`\`\`

**Practical Benefits:**
- Prevents runtime panics
- Correct handling of optional fields
- Distinguishes empty values from NULL`,
    order: 4,
    translations: {
        ru: {
            title: 'Обработка NULL значений',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}

func QueryUserWithNulls(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Сканируем в null типы
    var user User
    var email sql.NullString
    var age sql.NullInt64

    query := "SELECT id, name, email, age FROM users WHERE id = ?"
    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &email, &age)
    if err != nil {
        return nil, err
    }

    // Конвертируем null типы в указатели
    if email.Valid {
        user.Email = &email.String
    }

    if age.Valid {
        user.Age = &age.Int64
    }

    return &user, nil
}`,
            description: `Реализуйте функцию, которая запрашивает пользователя с nullable полями из базы данных. Необходимо использовать sql.NullString и sql.NullInt64 для правильной обработки NULL значений и конвертации их в типы указателей.

**Требования:**
- Используйте sql.NullString для nullable строковых полей
- Используйте sql.NullInt64 для nullable целочисленных полей
- Конвертируйте null типы в указатели (*string, *int64)
- Устанавливайте указатели в nil когда значение NULL

**Определение типа:**
\`\`\`go
type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}
\`\`\``,
            hint1: `Используйте типы sql.NullString и sql.NullInt64 для сканирования. Эти типы имеют поля Valid и Value.`,
            hint2: `После сканирования проверьте поле Valid. Если true, присвойте значение указателю. Если false, оставьте указатель как nil.`,
            whyItMatters: `Обработка NULL критически важна в реальных приложениях, где опциональные поля распространены. Использование sql.Null* типов предотвращает паники во время выполнения и позволяет правильно различать нулевые значения и NULL значения.

**Продакшен паттерн:**
\`\`\`go
// Безопасная обработка NULL значений
var email sql.NullString
db.QueryRowContext(ctx, query).Scan(&id, &email)

// Различаем пустую строку и NULL
if email.Valid {
    user.Email = &email.String // Имеет значение
} else {
    user.Email = nil // NULL в БД
}
\`\`\`

**Практические преимущества:**
- Предотвращение runtime паник
- Корректная обработка опциональных полей
- Различение пустых значений и NULL`
        },
        uz: {
            title: 'NULL qiymatlarni ishlash',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}

func QueryUserWithNulls(ctx context.Context, db *sql.DB, id int64) (*User, error) {
    // Null turlarga skanlaymiz
    var user User
    var email sql.NullString
    var age sql.NullInt64

    query := "SELECT id, name, email, age FROM users WHERE id = ?"
    err := db.QueryRowContext(ctx, query, id).Scan(&user.ID, &user.Name, &email, &age)
    if err != nil {
        return nil, err
    }

    // Null turlarni ko'rsatkichlarga aylantiramiz
    if email.Valid {
        user.Email = &email.String
    }

    if age.Valid {
        user.Age = &age.Int64
    }

    return &user, nil
}`,
            description: `Ma'lumotlar bazasidan nullable maydonlari bo'lgan foydalanuvchini so'raydigan funksiyani amalga oshiring. NULL qiymatlarni to'g'ri boshqarish va ularni ko'rsatkich turlariga aylantirish uchun sql.NullString va sql.NullInt64 dan foydalanishingiz kerak.

**Talablar:**
- Nullable string maydonlar uchun sql.NullString dan foydalaning
- Nullable integer maydonlar uchun sql.NullInt64 dan foydalaning
- Null turlarni ko'rsatkichlarga (*string, *int64) aylantiring
- Qiymat NULL bo'lganda ko'rsatkichlarni nil ga o'rnating

**Tur ta'rifi:**
\`\`\`go
type User struct {
    ID       int64
    Name     string
    Email    *string  // nullable
    Age      *int64   // nullable
}
\`\`\``,
            hint1: `Skanlash uchun sql.NullString va sql.NullInt64 turlaridan foydalaning. Bu turlar Valid va Value maydonlariga ega.`,
            hint2: `Skanlashdan keyin Valid maydonini tekshiring. Agar true bo'lsa, qiymatni ko'rsatkichga tayinlang. Agar false bo'lsa, ko'rsatkichni nil qoldiring.`,
            whyItMatters: `NULL boshqarish ixtiyoriy maydonlar keng tarqalgan haqiqiy ilovalarda juda muhimdir. sql.Null* turlaridan foydalanish bajarilish vaqtida paniklarni oldini oladi va nol qiymatlar va NULL qiymatlarni to'g'ri ajratish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`go
// NULL qiymatlarni xavfsiz boshqarish
var email sql.NullString
db.QueryRowContext(ctx, query).Scan(&id, &email)

// Bo'sh qator va NULL ni farqlaymiz
if email.Valid {
    user.Email = &email.String // Qiymatga ega
} else {
    user.Email = nil // Ma'lumotlar bazasida NULL
}
\`\`\`

**Amaliy foydalari:**
- Runtime paniklarni oldini olish
- Ixtiyoriy maydonlarni to'g'ri boshqarish
- Bo'sh qiymatlar va NULL ni farqlash`
        }
    }
};

export default task;
