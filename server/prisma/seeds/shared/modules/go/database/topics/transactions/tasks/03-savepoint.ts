import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-savepoint',
    title: 'Savepoint Simulation',
    difficulty: 'hard',
    tags: ['go', 'database', 'transaction', 'savepoint'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a transaction with savepoint support for partial rollback. While Go's database/sql doesn't have built-in savepoint support, you can execute SAVEPOINT and ROLLBACK TO SAVEPOINT SQL commands directly to achieve nested transaction-like behavior.

**Requirements:**
- Create savepoint using raw SQL
- Rollback to savepoint on specific errors
- Continue transaction after partial rollback
- Commit main transaction if successful

**Savepoint Commands:**
- Create: \`SAVEPOINT name\`
- Rollback: \`ROLLBACK TO SAVEPOINT name\`
- Release: \`RELEASE SAVEPOINT name\``,
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

type Address struct {
    UserID  int64
    Street  string
    City    string
}

// TODO: Create user with optional address using savepoints
// If address insert fails, rollback address but keep user
func CreateUserWithAddress(ctx context.Context, db *sql.DB, user User, address *Address) (int64, error) {
    panic("TODO: implement with SAVEPOINT")
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

type Address struct {
    UserID  int64
    Street  string
    City    string
}

func CreateUserWithAddress(ctx context.Context, db *sql.DB, user User, address *Address) (int64, error) {
    // Begin transaction
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Insert user
    result, err := tx.ExecContext(ctx,
        "INSERT INTO users (name, email) VALUES (?, ?)",
        user.Name, user.Email)
    if err != nil {
        return 0, err
    }

    userID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // If address provided, try to insert it with savepoint
    if address != nil {
        // Create savepoint
        _, err := tx.ExecContext(ctx, "SAVEPOINT address_insert")
        if err != nil {
            return 0, err
        }

        // Try to insert address
        _, err = tx.ExecContext(ctx,
            "INSERT INTO addresses (user_id, street, city) VALUES (?, ?, ?)",
            userID, address.Street, address.City)

        if err != nil {
            // Rollback to savepoint on error
            tx.ExecContext(ctx, "ROLLBACK TO SAVEPOINT address_insert")
            // Continue without address
        } else {
            // Release savepoint on success
            tx.ExecContext(ctx, "RELEASE SAVEPOINT address_insert")
        }
    }

    // Commit transaction
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return userID, nil
}`,
    testCode: `package dbx

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}

	_, err = db.Exec(\`CREATE TABLE users (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT,
		email TEXT
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	_, err = db.Exec(\`CREATE TABLE addresses (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id INTEGER,
		street TEXT,
		city TEXT
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	return db
}

func Test1(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "John", Email: "john@example.com"}
	address := &Address{Street: "123 Main St", City: "NYC"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if userID == 0 {
		t.Errorf("expected non-zero user ID")
	}
}

func Test2(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Jane", Email: "jane@example.com"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, nil)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if userID == 0 {
		t.Errorf("expected non-zero user ID")
	}
}

func Test3(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Bob", Email: "bob@example.com"}
	address := &Address{Street: "456 Elm St", City: "LA"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var count int
	db.QueryRow("SELECT COUNT(*) FROM addresses WHERE user_id = ?", userID).Scan(&count)
	if count != 1 {
		t.Errorf("expected 1 address, got %d", count)
	}
}

func Test4(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Alice", Email: "alice@example.com"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, nil)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var count int
	db.QueryRow("SELECT COUNT(*) FROM addresses WHERE user_id = ?", userID).Scan(&count)
	if count != 0 {
		t.Errorf("expected 0 addresses, got %d", count)
	}
}

func Test5(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Charlie", Email: "charlie@example.com"}
	address := &Address{Street: "789 Oak St", City: "Chicago"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var name string
	db.QueryRow("SELECT name FROM users WHERE id = ?", userID).Scan(&name)
	if name != "Charlie" {
		t.Errorf("expected Charlie, got %s", name)
	}
}

func Test6(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "David", Email: "david@example.com"}
	address := &Address{Street: "101 Pine St", City: "Seattle"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var city string
	db.QueryRow("SELECT city FROM addresses WHERE user_id = ?", userID).Scan(&city)
	if city != "Seattle" {
		t.Errorf("expected Seattle, got %s", city)
	}
}

func Test7(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Eve", Email: "eve@example.com"}
	address := &Address{Street: "", City: "Boston"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if userID == 0 {
		t.Errorf("expected non-zero user ID")
	}
}

func Test8(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user1 := User{Name: "Frank", Email: "frank@example.com"}
	user2 := User{Name: "Grace", Email: "grace@example.com"}

	userID1, _ := CreateUserWithAddress(context.Background(), db, user1, nil)
	userID2, _ := CreateUserWithAddress(context.Background(), db, user2, nil)

	if userID1 == userID2 {
		t.Errorf("expected different user IDs")
	}
}

func Test9(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	user := User{Name: "Henry", Email: "henry@example.com"}
	address := &Address{Street: "202 Maple Ave", City: "Denver"}

	userID, err := CreateUserWithAddress(context.Background(), db, user, address)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var email string
	db.QueryRow("SELECT email FROM users WHERE id = ?", userID).Scan(&email)
	if email != "henry@example.com" {
		t.Errorf("expected henry@example.com, got %s", email)
	}
}

func Test10(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	user := User{Name: "Isabel", Email: "isabel@example.com"}
	address := &Address{Street: "303 Cedar Rd", City: "Austin"}

	_, err := CreateUserWithAddress(ctx, db, user, address)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}`,
    hint1: `Execute "SAVEPOINT name" using tx.ExecContext() before the operation you want to be able to rollback partially.`,
    hint2: `If an error occurs, execute "ROLLBACK TO SAVEPOINT name" to undo only operations after the savepoint. If successful, execute "RELEASE SAVEPOINT name" to clean it up.`,
    whyItMatters: `Savepoints enable sophisticated transaction logic where you can rollback only part of a transaction instead of the entire thing. This is valuable for complex business workflows where some operations are optional or can gracefully fail without affecting the main transaction.

**Production Pattern:**
\`\`\`go
// Creating user with optional profile
tx.Exec("INSERT INTO users ...")
tx.Exec("SAVEPOINT profile_insert")

if err := tx.Exec("INSERT INTO profiles ..."); err != nil {
    tx.Exec("ROLLBACK TO SAVEPOINT profile_insert")
    // Continue without profile
} else {
    tx.Exec("RELEASE SAVEPOINT profile_insert")
}
tx.Commit() // User created in any case
\`\`\`

**Practical Benefits:**
- Partial rollback in complex processes
- Optional operations in transactions
- Flexibility in error handling`,
    order: 2,
    translations: {
        ru: {
            title: 'Симуляция Savepoint',
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

type Address struct {
    UserID  int64
    Street  string
    City    string
}

func CreateUserWithAddress(ctx context.Context, db *sql.DB, user User, address *Address) (int64, error) {
    // Начинаем транзакцию
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Вставляем пользователя
    result, err := tx.ExecContext(ctx,
        "INSERT INTO users (name, email) VALUES (?, ?)",
        user.Name, user.Email)
    if err != nil {
        return 0, err
    }

    userID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // Если адрес предоставлен, пытаемся вставить его с точкой сохранения
    if address != nil {
        // Создаем точку сохранения
        _, err := tx.ExecContext(ctx, "SAVEPOINT address_insert")
        if err != nil {
            return 0, err
        }

        // Пытаемся вставить адрес
        _, err = tx.ExecContext(ctx,
            "INSERT INTO addresses (user_id, street, city) VALUES (?, ?, ?)",
            userID, address.Street, address.City)

        if err != nil {
            // Откатываемся к точке сохранения при ошибке
            tx.ExecContext(ctx, "ROLLBACK TO SAVEPOINT address_insert")
            // Продолжаем без адреса
        } else {
            // Освобождаем точку сохранения при успехе
            tx.ExecContext(ctx, "RELEASE SAVEPOINT address_insert")
        }
    }

    // Фиксируем транзакцию
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return userID, nil
}`,
            description: `Реализуйте транзакцию с поддержкой точек сохранения для частичного отката. Хотя database/sql в Go не имеет встроенной поддержки точек сохранения, вы можете выполнять SAVEPOINT и ROLLBACK TO SAVEPOINT SQL команды напрямую для достижения поведения, похожего на вложенные транзакции.

**Требования:**
- Создайте точку сохранения используя сырой SQL
- Откатитесь к точке сохранения при специфичных ошибках
- Продолжите транзакцию после частичного отката
- Зафиксируйте основную транзакцию при успехе

**Команды точек сохранения:**
- Создать: \`SAVEPOINT name\`
- Откатить: \`ROLLBACK TO SAVEPOINT name\`
- Освободить: \`RELEASE SAVEPOINT name\``,
            hint1: `Выполните "SAVEPOINT name" используя tx.ExecContext() перед операцией, которую вы хотите иметь возможность откатить частично.`,
            hint2: `Если происходит ошибка, выполните "ROLLBACK TO SAVEPOINT name" чтобы отменить только операции после точки сохранения. Если успешно, выполните "RELEASE SAVEPOINT name" для очистки.`,
            whyItMatters: `Точки сохранения обеспечивают сложную логику транзакций, где вы можете откатить только часть транзакции вместо всей целиком. Это ценно для сложных бизнес-процессов, где некоторые операции опциональны или могут корректно провалиться без влияния на основную транзакцию.

**Продакшен паттерн:**
\`\`\`go
// Создание пользователя с опциональным профилем
tx.Exec("INSERT INTO users ...")
tx.Exec("SAVEPOINT profile_insert")

if err := tx.Exec("INSERT INTO profiles ..."); err != nil {
    tx.Exec("ROLLBACK TO SAVEPOINT profile_insert")
    // Продолжаем без профиля
} else {
    tx.Exec("RELEASE SAVEPOINT profile_insert")
}
tx.Commit() // Пользователь создан в любом случае
\`\`\`

**Практические преимущества:**
- Частичный откат в сложных процессах
- Опциональные операции в транзакциях
- Гибкость в обработке ошибок`
        },
        uz: {
            title: 'Savepoint simulyatsiyasi',
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

type Address struct {
    UserID  int64
    Street  string
    City    string
}

func CreateUserWithAddress(ctx context.Context, db *sql.DB, user User, address *Address) (int64, error) {
    // Tranzaksiyani boshlaymiz
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Foydalanuvchini qo'shamiz
    result, err := tx.ExecContext(ctx,
        "INSERT INTO users (name, email) VALUES (?, ?)",
        user.Name, user.Email)
    if err != nil {
        return 0, err
    }

    userID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // Agar manzil berilgan bo'lsa, uni saqlash nuqtasi bilan qo'shishga harakat qilamiz
    if address != nil {
        // Saqlash nuqtasini yaratamiz
        _, err := tx.ExecContext(ctx, "SAVEPOINT address_insert")
        if err != nil {
            return 0, err
        }

        // Manzilni qo'shishga harakat qilamiz
        _, err = tx.ExecContext(ctx,
            "INSERT INTO addresses (user_id, street, city) VALUES (?, ?, ?)",
            userID, address.Street, address.City)

        if err != nil {
            // Xatolik yuz berganda saqlash nuqtasiga qaytamiz
            tx.ExecContext(ctx, "ROLLBACK TO SAVEPOINT address_insert")
            // Manziisiz davom etamiz
        } else {
            // Muvaffaqiyat bo'lganda saqlash nuqtasini ozod qilamiz
            tx.ExecContext(ctx, "RELEASE SAVEPOINT address_insert")
        }
    }

    // Tranzaksiyani tasdiqlаymiz
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return userID, nil
}`,
            description: `Qisman bekor qilish uchun saqlash nuqtasi qo'llab-quvvatlaydigan tranzaksiyani amalga oshiring. Go ning database/sql paketi o'rnatilgan saqlash nuqtasi qo'llab-quvvatlashiga ega bo'lmasa-da, ichki tranzaksiyalarga o'xshash harakatga erishish uchun SAVEPOINT va ROLLBACK TO SAVEPOINT SQL buyruqlarini bevosita bajarishingiz mumkin.

**Talablar:**
- Xom SQL dan foydalanib saqlash nuqtasini yarating
- Maxsus xatoliklarda saqlash nuqtasiga qayting
- Qisman bekor qilishdan keyin tranzaksiyani davom ettiring
- Muvaffaqiyatli bo'lsa asosiy tranzaksiyani tasdiqlang

**Saqlash nuqtasi buyruqlari:**
- Yaratish: \`SAVEPOINT name\`
- Bekor qilish: \`ROLLBACK TO SAVEPOINT name\`
- Ozod qilish: \`RELEASE SAVEPOINT name\``,
            hint1: `Qisman bekor qilishni xohlagan operatsiyadan oldin tx.ExecContext() yordamida "SAVEPOINT name" ni bajaring.`,
            hint2: `Agar xatolik yuz bersa, saqlash nuqtasidan keyingi operatsiyalarni bekor qilish uchun "ROLLBACK TO SAVEPOINT name" ni bajaring. Agar muvaffaqiyatli bo'lsa, tozalash uchun "RELEASE SAVEPOINT name" ni bajaring.`,
            whyItMatters: `Saqlash nuqtalari butun tranzaksiyani emas, faqat tranzaksiyaning bir qismini bekor qilish imkonini beradigan murakkab tranzaksiya mantiqini ta'minlaydi. Bu ba'zi operatsiyalar ixtiyoriy bo'lgan yoki asosiy tranzaksiyaga ta'sir qilmasdan muvaffaqiyatsiz bo'lishi mumkin bo'lgan murakkab biznes ish oqimlari uchun qimmatlidir.

**Ishlab chiqarish patterni:**
\`\`\`go
// Ixtiyoriy profil bilan foydalanuvchi yaratish
tx.Exec("INSERT INTO users ...")
tx.Exec("SAVEPOINT profile_insert")

if err := tx.Exec("INSERT INTO profiles ..."); err != nil {
    tx.Exec("ROLLBACK TO SAVEPOINT profile_insert")
    // Profilsiz davom etamiz
} else {
    tx.Exec("RELEASE SAVEPOINT profile_insert")
}
tx.Commit() // Foydalanuvchi har qanday holatda yaratiladi
\`\`\`

**Amaliy foydalari:**
- Murakkab jarayonlarda qisman bekor qilish
- Tranzaksiyalarda ixtiyoriy operatsiyalar
- Xatolarni boshqarishda moslashuvchanlik`
        }
    }
};

export default task;
