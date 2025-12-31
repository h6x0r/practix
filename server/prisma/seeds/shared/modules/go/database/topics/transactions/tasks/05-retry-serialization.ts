import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-retry-serialization',
    title: 'Retry on Serialization Failure',
    difficulty: 'hard',
    tags: ['go', 'database', 'transaction', 'retry'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a transaction with automatic retry logic for serialization failures. When using high isolation levels like Serializable, concurrent transactions can fail with serialization errors and should be retried automatically.

**Requirements:**
- Detect serialization errors (error code 40001)
- Implement exponential backoff retry strategy
- Set maximum retry attempts
- Properly handle non-retryable errors

**Pattern:**
\`\`\`go
for attempt := 0; attempt < maxRetries; attempt++ {
    err := executeTransaction()
    if isSerializationError(err) {
        time.Sleep(backoff)
        continue
    }
    return err
}
\`\`\``,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
    "time"
)

// TODO: Execute transaction with retry on serialization failure
func UpdateInventoryWithRetry(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    panic("TODO: implement retry logic for serialization errors")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "strings"
    "time"
)

func UpdateInventoryWithRetry(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    const maxRetries = 3
    const initialBackoff = 10 * time.Millisecond

    var lastErr error

    for attempt := 0; attempt < maxRetries; attempt++ {
        // Execute transaction
        err := updateInventoryTransaction(ctx, db, productID, delta)

        if err == nil {
            return nil // Success
        }

        // Check if error is serialization failure
        if isSerializationError(err) {
            lastErr = err
            // Exponential backoff
            backoff := initialBackoff * time.Duration(1<<uint(attempt))
            time.Sleep(backoff)
            continue
        }

        // Non-retryable error
        return err
    }

    // Max retries exceeded
    return lastErr
}

func updateInventoryTransaction(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelSerializable,
    }

    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Read current inventory
    var current int
    err = tx.QueryRowContext(ctx,
        "SELECT quantity FROM inventory WHERE product_id = ?",
        productID).Scan(&current)
    if err != nil {
        return err
    }

    // Update inventory
    newQuantity := current + delta
    _, err = tx.ExecContext(ctx,
        "UPDATE inventory SET quantity = ? WHERE product_id = ?",
        newQuantity, productID)
    if err != nil {
        return err
    }

    // Commit
    if err := tx.Commit(); err != nil {
        return err
    }

    committed = true
    return nil
}

func isSerializationError(err error) bool {
    if err == nil {
        return false
    }
    // Check for serialization error code (40001)
    // Different databases may return different error formats
    errStr := strings.ToLower(err.Error())
    return strings.Contains(errStr, "serialization") ||
           strings.Contains(errStr, "40001") ||
           strings.Contains(errStr, "deadlock")
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

	_, err = db.Exec(\`CREATE TABLE inventory (
		product_id INTEGER PRIMARY KEY,
		quantity INTEGER
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	return db
}

func Test1(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (1, 100)")

	err := UpdateInventoryWithRetry(context.Background(), db, 1, 10)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 1").Scan(&quantity)
	if quantity != 110 {
		t.Errorf("expected 110, got %d", quantity)
	}
}

func Test2(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (2, 50)")

	err := UpdateInventoryWithRetry(context.Background(), db, 2, -10)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 2").Scan(&quantity)
	if quantity != 40 {
		t.Errorf("expected 40, got %d", quantity)
	}
}

func Test3(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (3, 100)")

	err := UpdateInventoryWithRetry(context.Background(), db, 3, 0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 3").Scan(&quantity)
	if quantity != 100 {
		t.Errorf("expected 100, got %d", quantity)
	}
}

func Test4(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (4, 200)")

	err := UpdateInventoryWithRetry(context.Background(), db, 4, -200)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 4").Scan(&quantity)
	if quantity != 0 {
		t.Errorf("expected 0, got %d", quantity)
	}
}

func Test5(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (5, 1000)")

	err := UpdateInventoryWithRetry(context.Background(), db, 5, 500)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 5").Scan(&quantity)
	if quantity != 1500 {
		t.Errorf("expected 1500, got %d", quantity)
	}
}

func Test6(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (6, 10)")

	err := UpdateInventoryWithRetry(context.Background(), db, 6, 1)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 6").Scan(&quantity)
	if quantity != 11 {
		t.Errorf("expected 11, got %d", quantity)
	}
}

func Test7(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	err := UpdateInventoryWithRetry(context.Background(), db, 999, 10)
	if err == nil {
		t.Errorf("expected error for non-existent product")
	}
}

func Test8(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (8, 50)")

	err := UpdateInventoryWithRetry(context.Background(), db, 8, -25)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var quantity int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 8").Scan(&quantity)
	if quantity != 25 {
		t.Errorf("expected 25, got %d", quantity)
	}
}

func Test9(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (9, 100)")
	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (10, 200)")

	err1 := UpdateInventoryWithRetry(context.Background(), db, 9, 50)
	err2 := UpdateInventoryWithRetry(context.Background(), db, 10, -50)

	if err1 != nil || err2 != nil {
		t.Errorf("expected both updates to succeed")
	}

	var q1, q2 int
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 9").Scan(&q1)
	db.QueryRow("SELECT quantity FROM inventory WHERE product_id = 10").Scan(&q2)

	if q1 != 150 || q2 != 150 {
		t.Errorf("expected 150 and 150, got %d and %d", q1, q2)
	}
}

func Test10(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	db.Exec("INSERT INTO inventory (product_id, quantity) VALUES (11, 100)")

	err := UpdateInventoryWithRetry(ctx, db, 11, 10)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}`,
    hint1: `Use a loop with attempt counter. After each serialization error, sleep for an increasing duration before retrying.`,
    hint2: `Check the error string for serialization-related keywords. Most databases include "serialization" or error code "40001" in the error message.`,
    whyItMatters: `Serialization failures are expected in high-concurrency environments using strict isolation levels. Automatic retry with exponential backoff is a standard pattern that improves reliability without manual intervention, allowing your system to gracefully handle transient conflicts.

**Production Pattern:**
\`\`\`go
// Automatic retry on conflicts
for attempt := 0; attempt < 3; attempt++ {
    err := updateInventory(ctx, db, productID, delta)

    if !isSerializationError(err) {
        return err // Success or non-retryable error
    }

    time.Sleep(10ms * (1 << attempt)) // Exponential backoff
}
\`\`\`

**Practical Benefits:**
- Automatic recovery from conflicts
- Improved throughput under load
- Transparent handling of transient failures`,
    order: 4,
    translations: {
        ru: {
            title: 'Повтор при ошибке сериализации',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "strings"
    "time"
)

func UpdateInventoryWithRetry(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    const maxRetries = 3
    const initialBackoff = 10 * time.Millisecond

    var lastErr error

    for attempt := 0; attempt < maxRetries; attempt++ {
        // Выполняем транзакцию
        err := updateInventoryTransaction(ctx, db, productID, delta)

        if err == nil {
            return nil // Успех
        }

        // Проверяем, является ли ошибка ошибкой сериализации
        if isSerializationError(err) {
            lastErr = err
            // Экспоненциальная задержка
            backoff := initialBackoff * time.Duration(1<<uint(attempt))
            time.Sleep(backoff)
            continue
        }

        // Неповторяемая ошибка
        return err
    }

    // Превышено максимальное количество повторов
    return lastErr
}

func updateInventoryTransaction(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelSerializable,
    }

    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Читаем текущие запасы
    var current int
    err = tx.QueryRowContext(ctx,
        "SELECT quantity FROM inventory WHERE product_id = ?",
        productID).Scan(&current)
    if err != nil {
        return err
    }

    // Обновляем запасы
    newQuantity := current + delta
    _, err = tx.ExecContext(ctx,
        "UPDATE inventory SET quantity = ? WHERE product_id = ?",
        newQuantity, productID)
    if err != nil {
        return err
    }

    // Фиксируем
    if err := tx.Commit(); err != nil {
        return err
    }

    committed = true
    return nil
}

func isSerializationError(err error) bool {
    if err == nil {
        return false
    }
    // Проверяем код ошибки сериализации (40001)
    // Разные базы данных могут возвращать разные форматы ошибок
    errStr := strings.ToLower(err.Error())
    return strings.Contains(errStr, "serialization") ||
           strings.Contains(errStr, "40001") ||
           strings.Contains(errStr, "deadlock")
}`,
            description: `Реализуйте транзакцию с автоматической логикой повтора для ошибок сериализации. При использовании высоких уровней изоляции, таких как Serializable, конкурентные транзакции могут провалиться с ошибками сериализации и должны автоматически повторяться.

**Требования:**
- Обнаруживайте ошибки сериализации (код ошибки 40001)
- Реализуйте стратегию повтора с экспоненциальной задержкой
- Установите максимальное количество попыток повтора
- Правильно обрабатывайте неповторяемые ошибки

**Паттерн:**
\`\`\`go
for attempt := 0; attempt < maxRetries; attempt++ {
    err := executeTransaction()
    if isSerializationError(err) {
        time.Sleep(backoff)
        continue
    }
    return err
}
\`\`\``,
            hint1: `Используйте цикл со счетчиком попыток. После каждой ошибки сериализации спите всё более длительное время перед повтором.`,
            hint2: `Проверьте строку ошибки на ключевые слова связанные с сериализацией. Большинство баз данных включают "serialization" или код ошибки "40001" в сообщение об ошибке.`,
            whyItMatters: `Ошибки сериализации ожидаемы в высоконагруженных средах с использованием строгих уровней изоляции. Автоматический повтор с экспоненциальной задержкой - стандартный паттерн, который улучшает надежность без ручного вмешательства, позволяя системе корректно обрабатывать временные конфликты.

**Продакшен паттерн:**
\`\`\`go
// Автоматический повтор при конфликтах
for attempt := 0; attempt < 3; attempt++ {
    err := updateInventory(ctx, db, productID, delta)

    if !isSerializationError(err) {
        return err // Успех или непереповторяемая ошибка
    }

    time.Sleep(10ms * (1 << attempt)) // Экспоненциальная задержка
}
\`\`\`

**Практические преимущества:**
- Автоматическое восстановление при конфликтах
- Улучшение пропускной способности под нагрузкой
- Прозрачная обработка временных сбоев`
        },
        uz: {
            title: 'Serializatsiya xatosida qayta urinish',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
    "strings"
    "time"
)

func UpdateInventoryWithRetry(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    const maxRetries = 3
    const initialBackoff = 10 * time.Millisecond

    var lastErr error

    for attempt := 0; attempt < maxRetries; attempt++ {
        // Tranzaksiyani bajaramiz
        err := updateInventoryTransaction(ctx, db, productID, delta)

        if err == nil {
            return nil // Muvaffaqiyat
        }

        // Xato serializatsiya xatosimi tekshiramiz
        if isSerializationError(err) {
            lastErr = err
            // Eksponensial kechikish
            backoff := initialBackoff * time.Duration(1<<uint(attempt))
            time.Sleep(backoff)
            continue
        }

        // Qayta urinib bo'lmaydigan xato
        return err
    }

    // Maksimal qayta urinishlar soni oshib ketdi
    return lastErr
}

func updateInventoryTransaction(ctx context.Context, db *sql.DB, productID int64, delta int) error {
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelSerializable,
    }

    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Joriy inventarni o'qiymiz
    var current int
    err = tx.QueryRowContext(ctx,
        "SELECT quantity FROM inventory WHERE product_id = ?",
        productID).Scan(&current)
    if err != nil {
        return err
    }

    // Inventarni yangilaymiz
    newQuantity := current + delta
    _, err = tx.ExecContext(ctx,
        "UPDATE inventory SET quantity = ? WHERE product_id = ?",
        newQuantity, productID)
    if err != nil {
        return err
    }

    // Tasdiqlаymiz
    if err := tx.Commit(); err != nil {
        return err
    }

    committed = true
    return nil
}

func isSerializationError(err error) bool {
    if err == nil {
        return false
    }
    // Serializatsiya xato kodini (40001) tekshiramiz
    // Turli ma'lumotlar bazalari turli xato formatlarini qaytarishi mumkin
    errStr := strings.ToLower(err.Error())
    return strings.Contains(errStr, "serialization") ||
           strings.Contains(errStr, "40001") ||
           strings.Contains(errStr, "deadlock")
}`,
            description: `Serializatsiya xatolari uchun avtomatik qayta urinish mantiqiga ega tranzaksiyani amalga oshiring. Serializable kabi yuqori izolyatsiya darajalaridan foydalanganda, bir vaqtdagi tranzaksiyalar serializatsiya xatolari bilan muvaffaqiyatsiz bo'lishi va avtomatik ravishda qayta urinilishi kerak.

**Talablar:**
- Serializatsiya xatolarini aniqlang (xato kodi 40001)
- Eksponensial kechikish bilan qayta urinish strategiyasini amalga oshiring
- Maksimal qayta urinishlar sonini o'rnating
- Qayta urinib bo'lmaydigan xatolarni to'g'ri boshqaring

**Pattern:**
\`\`\`go
for attempt := 0; attempt < maxRetries; attempt++ {
    err := executeTransaction()
    if isSerializationError(err) {
        time.Sleep(backoff)
        continue
    }
    return err
}
\`\`\``,
            hint1: `Urinish hisoblagichi bilan tsikldan foydalaning. Har bir serializatsiya xatosidan keyin qayta urinishdan oldin tobora uzoqroq vaqt uxlang.`,
            hint2: `Serializatsiya bilan bog'liq kalit so'zlar uchun xato qatorini tekshiring. Ko'pgina ma'lumotlar bazalari xato xabarida "serialization" yoki xato kodi "40001" ni o'z ichiga oladi.`,
            whyItMatters: `Serializatsiya xatolari qat'iy izolyatsiya darajalaridan foydalaniladigan yuqori konkurentsiyali muhitlarda kutiladi. Eksponensial kechikish bilan avtomatik qayta urinish ishonchlilikni qo'lda aralashuvsiz yaxshilaydigan standart pattern bo'lib, tizimingizga vaqtinchalik ziddiyatlarni to'g'ri hal qilish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Konfliktlarda avtomatik qayta urinish
for attempt := 0; attempt < 3; attempt++ {
    err := updateInventory(ctx, db, productID, delta)

    if !isSerializationError(err) {
        return err // Muvaffaqiyat yoki qayta urinib bo'lmaydigan xato
    }

    time.Sleep(10ms * (1 << attempt)) // Eksponensial kechikish
}
\`\`\`

**Amaliy foydalari:**
- Konfliktlarda avtomatik tiklanish
- Yuk ostida o'tkazuvchanlikni yaxshilash
- Vaqtinchalik nosozliklarni shaffof boshqarish`
        }
    }
};

export default task;
