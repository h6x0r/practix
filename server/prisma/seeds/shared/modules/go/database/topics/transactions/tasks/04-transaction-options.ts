import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-transaction-options',
    title: 'Transaction Isolation Levels',
    difficulty: 'medium',
    tags: ['go', 'database', 'transaction', 'isolation'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that uses specific transaction isolation levels. Different isolation levels provide different guarantees about visibility of concurrent transactions, allowing you to balance consistency and performance.

**Requirements:**
- Use sql.TxOptions to set isolation level
- Implement read-only transaction option
- Handle serialization errors appropriately
- Understand trade-offs of each level

**Isolation Levels:**
- ReadUncommitted: Lowest isolation, best performance
- ReadCommitted: Prevents dirty reads
- RepeatableRead: Prevents non-repeatable reads
- Serializable: Highest isolation, prevents phantom reads`,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
)

type AccountBalance struct {
    AccountID int64
    Balance   float64
}

// TODO: Get account balances with specific isolation level
func GetAccountBalances(ctx context.Context, db *sql.DB, accountIDs []int64, readOnly bool) ([]AccountBalance, error) {
    panic("TODO: implement with sql.TxOptions")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type AccountBalance struct {
    AccountID int64
    Balance   float64
}

func GetAccountBalances(ctx context.Context, db *sql.DB, accountIDs []int64, readOnly bool) ([]AccountBalance, error) {
    // Configure transaction options
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelRepeatableRead,
        ReadOnly:  readOnly,
    }

    // Begin transaction with options
    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return nil, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Query balances
    var balances []AccountBalance
    for _, id := range accountIDs {
        var balance AccountBalance
        err := tx.QueryRowContext(ctx,
            "SELECT account_id, balance FROM accounts WHERE account_id = ?",
            id).Scan(&balance.AccountID, &balance.Balance)

        if err != nil {
            if err == sql.ErrNoRows {
                continue
            }
            return nil, err
        }
        balances = append(balances, balance)
    }

    // Commit transaction
    if err := tx.Commit(); err != nil {
        return nil, err
    }

    committed = true
    return balances, nil
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

	_, err = db.Exec(\`CREATE TABLE accounts (
		account_id INTEGER PRIMARY KEY,
		balance REAL
	)\`)
	if err != nil {
		t.Fatal(err)
	}

	return db
}

func Test1(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 100.0)")
	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (2, 200.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1, 2}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if len(balances) != 2 {
		t.Errorf("expected 2 balances, got %d", len(balances))
	}
}

func Test2(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 100.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if len(balances) != 1 {
		t.Errorf("expected 1 balance, got %d", len(balances))
	}
	if balances[0].Balance != 100.0 {
		t.Errorf("expected 100.0, got %v", balances[0].Balance)
	}
}

func Test3(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	balances, err := GetAccountBalances(context.Background(), db, []int64{}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if len(balances) != 0 {
		t.Errorf("expected 0 balances, got %d", len(balances))
	}
}

func Test4(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 100.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1, 999}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if len(balances) != 1 {
		t.Errorf("expected 1 balance, got %d", len(balances))
	}
}

func Test5(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 100.0)")
	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (2, 200.0)")
	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (3, 300.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1, 2, 3}, false)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if len(balances) != 3 {
		t.Errorf("expected 3 balances, got %d", len(balances))
	}
}

func Test6(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 99.99)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if balances[0].Balance != 99.99 {
		t.Errorf("expected 99.99, got %v", balances[0].Balance)
	}
}

func Test7(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (5, 500.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{5}, false)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if balances[0].AccountID != 5 {
		t.Errorf("expected account_id 5, got %d", balances[0].AccountID)
	}
}

func Test8(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 0.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{1}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	if balances[0].Balance != 0.0 {
		t.Errorf("expected 0.0, got %v", balances[0].Balance)
	}
}

func Test9(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (10, 1000.0)")
	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (20, 2000.0)")

	balances, err := GetAccountBalances(context.Background(), db, []int64{10, 20}, true)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
	total := balances[0].Balance + balances[1].Balance
	if total != 3000.0 {
		t.Errorf("expected total 3000.0, got %v", total)
	}
}

func Test10(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	db.Exec("INSERT INTO accounts (account_id, balance) VALUES (1, 100.0)")

	_, err := GetAccountBalances(ctx, db, []int64{1}, true)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}`,
    hint1: `Create *sql.TxOptions struct with Isolation and ReadOnly fields. Pass it to db.BeginTx() as the second argument.`,
    hint2: `Use sql.LevelRepeatableRead for consistent reads within the transaction. For read-only queries that don't need strict consistency, use sql.LevelReadCommitted.`,
    whyItMatters: `Transaction isolation levels are critical for controlling concurrent access to data. Understanding them helps you choose the right balance between consistency and performance, preventing race conditions while avoiding unnecessary locking that can hurt throughput.

**Production Pattern:**
\`\`\`go
// Consistent reading of account balances
txOpts := &sql.TxOptions{
    Isolation: sql.LevelRepeatableRead, // Prevents phantom reads
    ReadOnly:  true, // Optimization for reads
}

tx, _ := db.BeginTx(ctx, txOpts)
// All reads see the same data snapshot
\`\`\`

**Practical Benefits:**
- Control visibility of concurrent changes
- Balance between performance and consistency
- Optimization for read-only operations`,
    order: 3,
    translations: {
        ru: {
            title: 'Уровни изоляции транзакций',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type AccountBalance struct {
    AccountID int64
    Balance   float64
}

func GetAccountBalances(ctx context.Context, db *sql.DB, accountIDs []int64, readOnly bool) ([]AccountBalance, error) {
    // Конфигурируем опции транзакции
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelRepeatableRead,
        ReadOnly:  readOnly,
    }

    // Начинаем транзакцию с опциями
    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return nil, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Запрашиваем балансы
    var balances []AccountBalance
    for _, id := range accountIDs {
        var balance AccountBalance
        err := tx.QueryRowContext(ctx,
            "SELECT account_id, balance FROM accounts WHERE account_id = ?",
            id).Scan(&balance.AccountID, &balance.Balance)

        if err != nil {
            if err == sql.ErrNoRows {
                continue
            }
            return nil, err
        }
        balances = append(balances, balance)
    }

    // Фиксируем транзакцию
    if err := tx.Commit(); err != nil {
        return nil, err
    }

    committed = true
    return balances, nil
}`,
            description: `Реализуйте функцию, которая использует специфичные уровни изоляции транзакций. Разные уровни изоляции предоставляют разные гарантии о видимости конкурентных транзакций, позволяя балансировать консистентность и производительность.

**Требования:**
- Используйте sql.TxOptions для установки уровня изоляции
- Реализуйте опцию транзакции только для чтения
- Правильно обрабатывайте ошибки сериализации
- Понимайте компромиссы каждого уровня

**Уровни изоляции:**
- ReadUncommitted: Низшая изоляция, лучшая производительность
- ReadCommitted: Предотвращает грязное чтение
- RepeatableRead: Предотвращает неповторяющееся чтение
- Serializable: Высшая изоляция, предотвращает фантомное чтение`,
            hint1: `Создайте структуру *sql.TxOptions с полями Isolation и ReadOnly. Передайте её в db.BeginTx() как второй аргумент.`,
            hint2: `Используйте sql.LevelRepeatableRead для консистентного чтения внутри транзакции. Для запросов только на чтение, которым не нужна строгая консистентность, используйте sql.LevelReadCommitted.`,
            whyItMatters: `Уровни изоляции транзакций критически важны для контроля конкурентного доступа к данным. Их понимание помогает выбрать правильный баланс между консистентностью и производительностью, предотвращая состояния гонки и избегая ненужной блокировки, которая может повредить пропускную способность.

**Продакшен паттерн:**
\`\`\`go
// Консистентное чтение балансов счетов
txOpts := &sql.TxOptions{
    Isolation: sql.LevelRepeatableRead, // Предотвращает фантомные чтения
    ReadOnly:  true, // Оптимизация для чтения
}

tx, _ := db.BeginTx(ctx, txOpts)
// Все чтения видят одинаковый снимок данных
\`\`\`

**Практические преимущества:**
- Контроль видимости конкурентных изменений
- Баланс между производительностью и консистентностью
- Оптимизация для read-only операций`
        },
        uz: {
            title: 'Tranzaksiya izolatsiya darajalari',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

type AccountBalance struct {
    AccountID int64
    Balance   float64
}

func GetAccountBalances(ctx context.Context, db *sql.DB, accountIDs []int64, readOnly bool) ([]AccountBalance, error) {
    // Tranzaksiya parametrlarini sozlaymiz
    txOpts := &sql.TxOptions{
        Isolation: sql.LevelRepeatableRead,
        ReadOnly:  readOnly,
    }

    // Parametrlar bilan tranzaksiyani boshlaymiz
    tx, err := db.BeginTx(ctx, txOpts)
    if err != nil {
        return nil, err
    }

    committed := false
    defer func() {
        if !committed {
            tx.Rollback()
        }
    }()

    // Balanslarni so'raymiz
    var balances []AccountBalance
    for _, id := range accountIDs {
        var balance AccountBalance
        err := tx.QueryRowContext(ctx,
            "SELECT account_id, balance FROM accounts WHERE account_id = ?",
            id).Scan(&balance.AccountID, &balance.Balance)

        if err != nil {
            if err == sql.ErrNoRows {
                continue
            }
            return nil, err
        }
        balances = append(balances, balance)
    }

    // Tranzaksiyani tasdiqlаymiz
    if err := tx.Commit(); err != nil {
        return nil, err
    }

    committed = true
    return balances, nil
}`,
            description: `Maxsus tranzaksiya izolyatsiya darajalaridan foydalanadigan funksiyani amalga oshiring. Turli izolyatsiya darajalari bir vaqtdagi tranzaksiyalarning ko'rinishi haqida turli kafolatlar beradi va bu izchillik va unumdorlik o'rtasida muvozanatni ta'minlaydi.

**Talablar:**
- Izolyatsiya darajasini o'rnatish uchun sql.TxOptions dan foydalaning
- Faqat o'qish uchun tranzaksiya parametrini amalga oshiring
- Serializatsiya xatolarini to'g'ri boshqaring
- Har bir darajaning kelishmovchiliklarini tushuning

**Izolyatsiya darajalari:**
- ReadUncommitted: Eng past izolyatsiya, eng yaxshi unumdorlik
- ReadCommitted: Iflos o'qishlarni oldini oladi
- RepeatableRead: Takrorlanmaydigan o'qishlarni oldini oladi
- Serializable: Eng yuqori izolyatsiya, fantom o'qishlarni oldini oladi`,
            hint1: `Isolation va ReadOnly maydonlari bilan *sql.TxOptions strukturasini yarating. Uni ikkinchi argument sifatida db.BeginTx() ga o'tkazing.`,
            hint2: `Tranzaksiya ichida izchil o'qish uchun sql.LevelRepeatableRead dan foydalaning. Qat'iy izchillik talab qilmaydigan faqat o'qish so'rovlari uchun sql.LevelReadCommitted dan foydalaning.`,
            whyItMatters: `Tranzaksiya izolyatsiya darajalari ma'lumotlarga bir vaqtdagi kirishni boshqarish uchun juda muhimdir. Ularni tushunish izchillik va unumdorlik o'rtasida to'g'ri muvozanatni tanlashga yordam beradi, poyga holatlarining oldini oladi va o'tkazuvchanlikka zarar etkazadigan keraksiz blokirovkadan qochadi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Hisob balanslarini izchil o'qish
txOpts := &sql.TxOptions{
    Isolation: sql.LevelRepeatableRead, // Fantom o'qishlarni oldini oladi
    ReadOnly:  true, // O'qish uchun optimallashtirish
}

tx, _ := db.BeginTx(ctx, txOpts)
// Barcha o'qishlar bir xil ma'lumotlar suratini ko'radi
\`\`\`

**Amaliy foydalari:**
- Bir vaqtdagi o'zgarishlar ko'rinishini boshqarish
- Unumdorlik va izchillik o'rtasida muvozanat
- Faqat o'qish operatsiyalari uchun optimallashtirish`
        }
    }
};

export default task;
