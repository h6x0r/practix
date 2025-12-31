import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-rollback-on-error',
    title: 'Rollback with Defer',
    difficulty: 'medium',
    tags: ['go', 'database', 'transaction', 'defer'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a robust transaction function using the defer rollback pattern. This pattern ensures that the transaction is automatically rolled back if any error occurs, even if you forget to explicitly call Rollback() in error paths.

**Requirements:**
- Use defer tx.Rollback() immediately after BeginTx
- Track commit status to avoid rollback after successful commit
- Handle multiple operations within transaction
- Return appropriate errors

**Pattern:**
\`\`\`go
defer func() {
    if !committed {
        tx.Rollback()
    }
}()
\`\`\``,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
)

type Order struct {
    ID       int64
    UserID   int64
    Total    float64
}

type OrderItem struct {
    OrderID   int64
    ProductID int64
    Quantity  int
    Price     float64
}

// TODO: Create order with items using defer rollback pattern
func CreateOrder(ctx context.Context, db *sql.DB, order Order, items []OrderItem) (int64, error) {
    panic("TODO: implement with defer tx.Rollback() pattern")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type Order struct {
    ID       int64
    UserID   int64
    Total    float64
}

type OrderItem struct {
    OrderID   int64
    ProductID int64
    Quantity  int
    Price     float64
}

func CreateOrder(ctx context.Context, db *sql.DB, order Order, items []OrderItem) (int64, error) {
    // Begin transaction
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    // Setup automatic rollback on error
    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Insert order
    result, err := tx.ExecContext(ctx,
        "INSERT INTO orders (user_id, total) VALUES (?, ?)",
        order.UserID, order.Total)
    if err != nil {
        return 0, err
    }

    orderID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // Insert order items
    for _, item := range items {
        _, err := tx.ExecContext(ctx,
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
            orderID, item.ProductID, item.Quantity, item.Price)
        if err != nil {
            return 0, err
        }
    }

    // Commit transaction
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return orderID, nil
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

	_, err = db.Exec(\`CREATE TABLE orders (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id INTEGER,
		total REAL
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	_, err = db.Exec(\`CREATE TABLE order_items (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		order_id INTEGER,
		product_id INTEGER,
		quantity INTEGER,
		price REAL
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	return db
}

func Test1(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 1, Total: 100.0}
	items := []OrderItem{{ProductID: 1, Quantity: 2, Price: 50.0}}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if orderID == 0 {
		t.Errorf("expected non-zero order ID")
	}
}

func Test2(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 2, Total: 200.0}
	items := []OrderItem{
		{ProductID: 1, Quantity: 2, Price: 50.0},
		{ProductID: 2, Quantity: 1, Price: 100.0},
	}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var count int
	db.QueryRow("SELECT COUNT(*) FROM order_items WHERE order_id = ?", orderID).Scan(&count)
	if count != 2 {
		t.Errorf("expected 2 items, got %d", count)
	}
}

func Test3(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 3, Total: 0.0}
	items := []OrderItem{}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if orderID == 0 {
		t.Errorf("expected non-zero order ID")
	}
}

func Test4(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 4, Total: 150.0}
	items := []OrderItem{{ProductID: 1, Quantity: 3, Price: 50.0}}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var total float64
	db.QueryRow("SELECT total FROM orders WHERE id = ?", orderID).Scan(&total)
	if total != 150.0 {
		t.Errorf("expected 150.0, got %v", total)
	}
}

func Test5(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 5, Total: 500.0}
	items := []OrderItem{
		{ProductID: 1, Quantity: 5, Price: 100.0},
	}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM order_items WHERE order_id = ?", orderID).Scan(&quantity)
	if quantity != 5 {
		t.Errorf("expected 5, got %d", quantity)
	}
}

func Test6(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 6, Total: 99.99}
	items := []OrderItem{{ProductID: 1, Quantity: 1, Price: 99.99}}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var price float64
	db.QueryRow("SELECT price FROM order_items WHERE order_id = ?", orderID).Scan(&price)
	if price != 99.99 {
		t.Errorf("expected 99.99, got %v", price)
	}
}

func Test7(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order1 := Order{UserID: 7, Total: 100.0}
	items1 := []OrderItem{{ProductID: 1, Quantity: 1, Price: 100.0}}

	orderID1, _ := CreateOrder(context.Background(), db, order1, items1)

	order2 := Order{UserID: 8, Total: 200.0}
	items2 := []OrderItem{{ProductID: 2, Quantity: 1, Price: 200.0}}

	orderID2, _ := CreateOrder(context.Background(), db, order2, items2)

	if orderID1 == orderID2 {
		t.Errorf("expected different order IDs")
	}
}

func Test8(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 9, Total: 1000.0}
	items := []OrderItem{
		{ProductID: 1, Quantity: 10, Price: 100.0},
	}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var userID int64
	db.QueryRow("SELECT user_id FROM orders WHERE id = ?", orderID).Scan(&userID)
	if userID != 9 {
		t.Errorf("expected 9, got %d", userID)
	}
}

func Test9(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	order := Order{UserID: 10, Total: 250.0}
	items := []OrderItem{
		{ProductID: 1, Quantity: 1, Price: 100.0},
		{ProductID: 2, Quantity: 1, Price: 150.0},
	}

	orderID, err := CreateOrder(context.Background(), db, order, items)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var productID int64
	db.QueryRow("SELECT product_id FROM order_items WHERE order_id = ? LIMIT 1", orderID).Scan(&productID)
	if productID != 1 {
		t.Errorf("expected 1, got %d", productID)
	}
}

func Test10(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	order := Order{UserID: 11, Total: 100.0}
	items := []OrderItem{{ProductID: 1, Quantity: 1, Price: 100.0}}

	_, err := CreateOrder(ctx, db, order, items)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}`,
    hint1: `Use a boolean flag (committed) to track whether the transaction was successfully committed. Initialize it to false.`,
    hint2: `In the defer function, check if committed is false. If so, rollback the transaction. This ensures rollback happens on any error path.`,
    whyItMatters: `The defer rollback pattern is a defensive programming technique that prevents resource leaks and ensures transactions are always properly closed. It's especially important in complex functions with multiple error paths, eliminating the need to remember to rollback in each error case.

**Production Pattern:**
\`\`\`go
// Automatic rollback on any error
tx, _ := db.BeginTx(ctx, nil)
committed := false
defer func() {
    if !committed {
        tx.Rollback() // Rollback on panic or error
    }
}()

// Multiple operations
// Automatically rolled back on any error
tx.Commit()
committed = true
\`\`\`

**Practical Benefits:**
- Protection from transaction leaks
- Automatic rollback on panics
- Simplifies error handling`,
    order: 1,
    translations: {
        ru: {
            title: 'Откат при ошибке',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type Order struct {
    ID       int64
    UserID   int64
    Total    float64
}

type OrderItem struct {
    OrderID   int64
    ProductID int64
    Quantity  int
    Price     float64
}

func CreateOrder(ctx context.Context, db *sql.DB, order Order, items []OrderItem) (int64, error) {
    // Начинаем транзакцию
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    // Настраиваем автоматический откат при ошибке
    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Вставляем заказ
    result, err := tx.ExecContext(ctx,
        "INSERT INTO orders (user_id, total) VALUES (?, ?)",
        order.UserID, order.Total)
    if err != nil {
        return 0, err
    }

    orderID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // Вставляем элементы заказа
    for _, item := range items {
        _, err := tx.ExecContext(ctx,
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
            orderID, item.ProductID, item.Quantity, item.Price)
        if err != nil {
            return 0, err
        }
    }

    // Фиксируем транзакцию
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return orderID, nil
}`,
            description: `Реализуйте надежную функцию транзакции используя паттерн defer rollback. Этот паттерн гарантирует, что транзакция будет автоматически откачена при любой ошибке, даже если вы забудете явно вызвать Rollback() в путях ошибок.

**Требования:**
- Используйте defer tx.Rollback() сразу после BeginTx
- Отслеживайте статус коммита чтобы избежать отката после успешного коммита
- Обрабатывайте несколько операций внутри транзакции
- Возвращайте соответствующие ошибки

**Паттерн:**
\`\`\`go
defer func() {
    if !committed {
        tx.Rollback()
    }
}()
\`\`\``,
            hint1: `Используйте булев флаг (committed) для отслеживания успешного коммита транзакции. Инициализируйте его как false.`,
            hint2: `В функции defer проверьте, что committed равен false. Если да, откатите транзакцию. Это гарантирует откат на любом пути ошибки.`,
            whyItMatters: `Паттерн defer rollback - это техника защитного программирования, которая предотвращает утечки ресурсов и гарантирует правильное закрытие транзакций. Это особенно важно в сложных функциях с множественными путями ошибок, устраняя необходимость помнить об откате в каждом случае ошибки.

**Продакшен паттерн:**
\`\`\`go
// Автоматический откат при любой ошибке
tx, _ := db.BeginTx(ctx, nil)
committed := false
defer func() {
    if !committed {
        tx.Rollback() // Откат при панике или ошибке
    }
}()

// Несколько операций
// При любой ошибке автоматически откатится
tx.Commit()
committed = true
\`\`\`

**Практические преимущества:**
- Защита от утечек транзакций
- Автоматический откат при паниках
- Упрощение обработки ошибок`
        },
        uz: {
            title: 'Xatoda rollback',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type Order struct {
    ID       int64
    UserID   int64
    Total    float64
}

type OrderItem struct {
    OrderID   int64
    ProductID int64
    Quantity  int
    Price     float64
}

func CreateOrder(ctx context.Context, db *sql.DB, order Order, items []OrderItem) (int64, error) {
    // Tranzaksiyani boshlaymiz
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return 0, err
    }

    // Xatolik yuz berganda avtomatik bekor qilishni sozlaymiz
    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Buyurtmani qo'shamiz
    result, err := tx.ExecContext(ctx,
        "INSERT INTO orders (user_id, total) VALUES (?, ?)",
        order.UserID, order.Total)
    if err != nil {
        return 0, err
    }

    orderID, err := result.LastInsertId()
    if err != nil {
        return 0, err
    }

    // Buyurtma elementlarini qo'shamiz
    for _, item := range items {
        _, err := tx.ExecContext(ctx,
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
            orderID, item.ProductID, item.Quantity, item.Price)
        if err != nil {
            return 0, err
        }
    }

    // Tranzaksiyani tasdiqlаymiz
    if err := tx.Commit(); err != nil {
        return 0, err
    }

    committed = true
    return orderID, nil
}`,
            description: `defer rollback patternidan foydalanib ishonchli tranzaksiya funksiyasini amalga oshiring. Bu pattern har qanday xatolik yuz berganda tranzaksiyaning avtomatik ravishda bekor qilinishini ta'minlaydi, hatto xato yo'llarida Rollback() ni aniq chaqirishni unutsangiz ham.

**Talablar:**
- BeginTx dan keyin darhol defer tx.Rollback() dan foydalaning
- Muvaffaqiyatli commit dan keyin bekor qilishdan qochish uchun commit holatini kuzating
- Tranzaksiya ichida bir nechta operatsiyalarni boshqaring
- Tegishli xatolarni qaytaring

**Pattern:**
\`\`\`go
defer func() {
    if !committed {
        tx.Rollback()
    }
}()
\`\`\``,
            hint1: `Tranzaksiya muvaffaqiyatli commit qilinganligini kuzatish uchun boolean flag (committed) dan foydalaning. Uni false ga ishga tushiring.`,
            hint2: `defer funksiyasida committed ning false ekanligini tekshiring. Agar shunday bo'lsa, tranzaksiyani bekor qiling. Bu har qanday xato yo'lida bekor qilishni ta'minlaydi.`,
            whyItMatters: `defer rollback patterni himoya dasturlash texnikasi bo'lib, resurs oqishini oldini oladi va tranzaksiyalarning har doim to'g'ri yopilishini ta'minlaydi. Bu ayniqsa har bir xato holatida bekor qilishni eslab qolish zaruriyatini yo'qotib, ko'plab xato yo'llari bo'lgan murakkab funktsiyalarda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`go
// Har qanday xatolikda avtomatik bekor qilish
tx, _ := db.BeginTx(ctx, nil)
committed := false
defer func() {
    if !committed {
        tx.Rollback() // Panika yoki xatoda bekor qilish
    }
}()

// Bir nechta operatsiyalar
// Har qanday xatolikda avtomatik bekor qilinadi
tx.Commit()
committed = true
\`\`\`

**Amaliy foydalari:**
- Tranzaksiya oqishidan himoya
- Panikalarda avtomatik bekor qilish
- Xatolarni boshqarishni soddalash tirish`
        }
    }
};

export default task;
