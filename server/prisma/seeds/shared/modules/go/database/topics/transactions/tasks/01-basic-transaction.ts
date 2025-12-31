import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-basic-transaction',
    title: 'Basic Transaction',
    difficulty: 'easy',
    tags: ['go', 'database', 'transaction', 'sql'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that transfers money between two accounts using a database transaction. The transaction ensures that both operations (debit and credit) either succeed together or fail together, maintaining data consistency.

**Requirements:**
- Use db.BeginTx to start transaction
- Execute both UPDATE statements within transaction
- Commit transaction if both operations succeed
- Return error if any operation fails

**Operation:**
1. Deduct amount from sender account
2. Add amount to receiver account
3. Commit transaction`,
    initialCode: `package dbx

import (
    "context"
    "database/sql"
)

// TODO: Transfer money between accounts in a transaction
func TransferMoney(ctx context.Context, db *sql.DB, fromID, toID int64, amount float64) error {
    panic("TODO: implement BeginTx, Exec, and Commit")
}`,
    solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

func TransferMoney(ctx context.Context, db *sql.DB, fromID, toID int64, amount float64) error {
    // Begin transaction
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    // Deduct from sender
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, fromID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Add to receiver
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, toID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Commit transaction
    return tx.Commit()
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
		id INTEGER PRIMARY KEY,
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

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 200.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 1").Scan(&balance)
	if balance != 800.0 {
		t.Errorf("expected 800.0, got %v", balance)
	}
}

func Test2(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 200.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 2").Scan(&balance)
	if balance != 700.0 {
		t.Errorf("expected 700.0, got %v", balance)
	}
}

func Test3(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 0.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 1").Scan(&balance)
	if balance != 1000.0 {
		t.Errorf("expected 1000.0, got %v", balance)
	}
}

func Test4(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 1000.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 1").Scan(&balance)
	if balance != 0.0 {
		t.Errorf("expected 0.0, got %v", balance)
	}
}

func Test5(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 1000.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 2").Scan(&balance)
	if balance != 1500.0 {
		t.Errorf("expected 1500.0, got %v", balance)
	}
}

func Test6(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 100.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 50.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 0.01)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 1").Scan(&balance)
	if balance != 99.99 {
		t.Errorf("expected 99.99, got %v", balance)
	}
}

func Test7(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")

	err := TransferMoney(context.Background(), db, 1, 999, 100.0)
	if err == nil {
		t.Errorf("expected error for non-existent account")
	}
}

func Test8(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 1, 2, 50.5)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 2").Scan(&balance)
	if balance != 550.5 {
		t.Errorf("expected 550.5, got %v", balance)
	}
}

func Test9(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	err := TransferMoney(context.Background(), db, 2, 1, 250.0)
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	var balance float64
	db.QueryRow("SELECT balance FROM accounts WHERE id = 1").Scan(&balance)
	if balance != 1250.0 {
		t.Errorf("expected 1250.0, got %v", balance)
	}
}

func Test10(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	db.Exec("INSERT INTO accounts (id, balance) VALUES (1, 1000.0)")
	db.Exec("INSERT INTO accounts (id, balance) VALUES (2, 500.0)")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := TransferMoney(ctx, db, 1, 2, 100.0)
	if err == nil {
		t.Errorf("expected error for cancelled context")
	}
}`,
    hint1: `Use db.BeginTx(ctx, nil) to start a transaction. It returns *sql.Tx which you use to execute statements.`,
    hint2: `If any operation fails, call tx.Rollback() to undo all changes. If all succeed, call tx.Commit() to make changes permanent.`,
    whyItMatters: `Transactions are fundamental for maintaining data consistency when multiple related operations must succeed or fail as a unit. They prevent partial updates that could leave your data in an inconsistent state, which is critical for financial systems, inventory management, and many other applications.

**Production Pattern:**
\`\`\`go
// Atomic money transfer
tx, _ := db.BeginTx(ctx, nil)
defer tx.Rollback() // Rollback on panic

tx.Exec("UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, from)
tx.Exec("UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, to)

tx.Commit() // Both operations or none
\`\`\`

**Practical Benefits:**
- Guarantees data consistency
- Protection from partial updates
- ACID properties for critical operations`,
    order: 0,
    translations: {
        ru: {
            title: 'Базовая транзакция',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

func TransferMoney(ctx context.Context, db *sql.DB, fromID, toID int64, amount float64) error {
    // Начинаем транзакцию
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    // Вычитаем у отправителя
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, fromID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Добавляем получателю
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, toID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Фиксируем транзакцию
    return tx.Commit()
}`,
            description: `Реализуйте функцию, которая переводит деньги между двумя счетами используя транзакцию базы данных. Транзакция гарантирует, что обе операции (дебет и кредит) либо выполнятся вместе, либо откатятся вместе, поддерживая целостность данных.

**Требования:**
- Используйте db.BeginTx для начала транзакции
- Выполните оба UPDATE запроса внутри транзакции
- Зафиксируйте транзакцию если обе операции успешны
- Верните ошибку если любая операция провалится

**Операция:**
1. Вычесть сумму со счета отправителя
2. Добавить сумму на счет получателя
3. Зафиксировать транзакцию`,
            hint1: `Используйте db.BeginTx(ctx, nil) для начала транзакции. Он возвращает *sql.Tx, который используется для выполнения запросов.`,
            hint2: `Если любая операция провалится, вызовите tx.Rollback() чтобы отменить все изменения. Если все успешно, вызовите tx.Commit() чтобы сделать изменения постоянными.`,
            whyItMatters: `Транзакции являются основой для поддержания целостности данных, когда несколько связанных операций должны выполниться или откатиться как единое целое. Они предотвращают частичные обновления, которые могут оставить данные в несогласованном состоянии, что критически важно для финансовых систем, управления запасами и многих других приложений.

**Продакшен паттерн:**
\`\`\`go
// Атомарный перевод денег
tx, _ := db.BeginTx(ctx, nil)
defer tx.Rollback() // Откат при панике

tx.Exec("UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, from)
tx.Exec("UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, to)

tx.Commit() // Обе операции или ни одна
\`\`\`

**Практические преимущества:**
- Гарантия консистентности данных
- Защита от частичных обновлений
- ACID свойства для критичных операций`
        },
        uz: {
            title: 'Asosiy tranzaksiya',
            solutionCode: `package dbx

import (
    "context"
    "database/sql"
)

func TransferMoney(ctx context.Context, db *sql.DB, fromID, toID int64, amount float64) error {
    // Tranzaksiyani boshlaymiz
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }

    // Jo'natuvchidan ayiramiz
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, fromID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Qabul qiluvchiga qo'shamiz
    _, err = tx.ExecContext(ctx, "UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, toID)
    if err != nil {
        tx.Rollback()
        return err
    }

    // Tranzaksiyani tasdiqlаymiz
    return tx.Commit()
}`,
            description: `Ma'lumotlar bazasi tranzaksiyasidan foydalanib ikki hisob o'rtasida pul o'tkaziladigan funksiyani amalga oshiring. Tranzaksiya ikkala operatsiya (debet va kredit) birgalikda muvaffaqiyatli bo'lishi yoki birgalikda bekor qilinishini ta'minlab, ma'lumotlar izchilligini saqlaydi.

**Talablar:**
- Tranzaksiyani boshlash uchun db.BeginTx dan foydalaning
- Ikkala UPDATE so'rovini tranzaksiya ichida bajaring
- Agar ikkala operatsiya muvaffaqiyatli bo'lsa, tranzaksiyani tasdiqlang
- Agar biron operatsiya muvaffaqiyatsiz bo'lsa, xato qaytaring

**Operatsiya:**
1. Jo'natuvchi hisobidan summani ayirish
2. Qabul qiluvchi hisobiga summani qo'shish
3. Tranzaksiyani tasdiqlash`,
            hint1: `Tranzaksiyani boshlash uchun db.BeginTx(ctx, nil) dan foydalaning. U so'rovlarni bajarish uchun ishlatiladigan *sql.Tx ni qaytaradi.`,
            hint2: `Agar biron operatsiya muvaffaqiyatsiz bo'lsa, barcha o'zgarishlarni bekor qilish uchun tx.Rollback() ni chaqiring. Agar hammasi muvaffaqiyatli bo'lsa, o'zgarishlarni doimiy qilish uchun tx.Commit() ni chaqiring.`,
            whyItMatters: `Tranzaksiyalar bir nechta bog'liq operatsiyalar bitta birlik sifatida muvaffaqiyatli bo'lishi yoki bekor qilinishi kerak bo'lganda ma'lumotlar izchilligini saqlash uchun asosdir. Ular ma'lumotlarni nomuvofiq holatda qoldiradigan qisman yangilanishlarning oldini oladi, bu moliyaviy tizimlar, inventar boshqaruvi va boshqa ko'plab ilovalar uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`go
// Atom pul o'tkazmasi
tx, _ := db.BeginTx(ctx, nil)
defer tx.Rollback() // Panikada bekor qilish

tx.Exec("UPDATE accounts SET balance = balance - ? WHERE id = ?", amount, from)
tx.Exec("UPDATE accounts SET balance = balance + ? WHERE id = ?", amount, to)

tx.Commit() // Ikkala operatsiya yoki hech biri
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar izchilligini kafolatlash
- Qisman yangilanishlardan himoya
- Muhim operatsiyalar uchun ACID xususiyatlari`
        }
    }
};

export default task;
