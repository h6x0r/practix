import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-pool-configuration',
    title: 'Configure Connection Pool',
    difficulty: 'easy',
    tags: ['go', 'database', 'connection-pool', 'configuration'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that configures a database connection pool with appropriate limits. Connection pooling is essential for performance, as creating new connections is expensive. Proper configuration prevents resource exhaustion and connection leaks.

**Requirements:**
- Set maximum open connections (SetMaxOpenConns)
- Set maximum idle connections (SetMaxIdleConns)
- Understand the relationship between these settings
- Return configured database instance

**Recommended Settings:**
- MaxOpenConns: 25-100 (depends on workload)
- MaxIdleConns: 5-25 (typically 25% of MaxOpen)
- Keep idle connections ready for quick reuse`,
    initialCode: `package dbx

import (
    "database/sql"
)

type PoolConfig struct {
    MaxOpenConns int
    MaxIdleConns int
}

// TODO: Configure connection pool with given settings
func ConfigurePool(db *sql.DB, config PoolConfig) {
    panic("TODO: implement SetMaxOpenConns and SetMaxIdleConns")
}`,
    solutionCode: `package dbx

import (
    "database/sql"
)

type PoolConfig struct {
    MaxOpenConns int
    MaxIdleConns int
}

func ConfigurePool(db *sql.DB, config PoolConfig) {
    // Set maximum number of open connections
    // This limits total connections (idle + in-use)
    db.SetMaxOpenConns(config.MaxOpenConns)

    // Set maximum number of idle connections
    // These are kept alive and ready for reuse
    // Should be less than or equal to MaxOpenConns
    db.SetMaxIdleConns(config.MaxIdleConns)
}`,
	testCode: `package dbx

import (
	"database/sql"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// ConfigurePool does not panic with valid config
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 25, MaxIdleConns: 5}
	ConfigurePool(db, config)
	// If we reach here, function didn't panic
}

func Test2(t *testing.T) {
	// ConfigurePool handles zero max conns
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 0, MaxIdleConns: 0}
	ConfigurePool(db, config)
}

func Test3(t *testing.T) {
	// ConfigurePool handles large values
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 1000, MaxIdleConns: 100}
	ConfigurePool(db, config)
}

func Test4(t *testing.T) {
	// ConfigurePool accepts idle greater than open
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 5, MaxIdleConns: 25}
	ConfigurePool(db, config)
}

func Test5(t *testing.T) {
	// ConfigurePool accepts equal values
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 10, MaxIdleConns: 10}
	ConfigurePool(db, config)
}

func Test6(t *testing.T) {
	// ConfigurePool with typical production values
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 50, MaxIdleConns: 10}
	ConfigurePool(db, config)
}

func Test7(t *testing.T) {
	// ConfigurePool with single connection
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 1, MaxIdleConns: 1}
	ConfigurePool(db, config)
}

func Test8(t *testing.T) {
	// ConfigurePool with minimal idle
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 100, MaxIdleConns: 1}
	ConfigurePool(db, config)
}

func Test9(t *testing.T) {
	// ConfigurePool stats reflect maxOpen
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := PoolConfig{MaxOpenConns: 42, MaxIdleConns: 5}
	ConfigurePool(db, config)

	stats := db.Stats()
	if stats.MaxOpenConnections != 42 {
		t.Errorf("expected MaxOpenConnections 42, got %d", stats.MaxOpenConnections)
	}
}

func Test10(t *testing.T) {
	// ConfigurePool can be called multiple times
	db, _, _ := sqlmock.New()
	defer db.Close()

	config1 := PoolConfig{MaxOpenConns: 10, MaxIdleConns: 2}
	ConfigurePool(db, config1)

	config2 := PoolConfig{MaxOpenConns: 50, MaxIdleConns: 10}
	ConfigurePool(db, config2)

	stats := db.Stats()
	if stats.MaxOpenConnections != 50 {
		t.Errorf("expected MaxOpenConnections 50, got %d", stats.MaxOpenConnections)
	}
}
`,
    hint1: `Use db.SetMaxOpenConns() to limit the total number of open connections (both in-use and idle).`,
    hint2: `Use db.SetMaxIdleConns() to limit the number of idle connections in the pool. This should be less than MaxOpenConns.`,
    whyItMatters: `Proper connection pool configuration is critical for application performance and stability. Too few connections cause contention and slow response times. Too many connections can overwhelm the database server and waste memory. Finding the right balance is essential for production systems.

**Production Pattern:**
\`\`\`go
// Optimal configuration for web server
db.SetMaxOpenConns(25)    // Max connections
db.SetMaxIdleConns(5)     // Ready connections
db.SetConnMaxLifetime(5 * time.Minute)

// Adapt to load
if highTraffic {
    db.SetMaxOpenConns(100)
    db.SetMaxIdleConns(25)
}
\`\`\`

**Practical Benefits:**
- Prevents connection exhaustion
- Fast reuse of ready connections
- Adapts to different loads`,
    order: 0,
    translations: {
        ru: {
            title: 'Конфигурация пула соединений',
            solutionCode: `package dbx

import (
    "database/sql"
)

type PoolConfig struct {
    MaxOpenConns int
    MaxIdleConns int
}

func ConfigurePool(db *sql.DB, config PoolConfig) {
    // Устанавливаем максимальное количество открытых соединений
    // Это ограничивает общее количество соединений (простаивающие + используемые)
    db.SetMaxOpenConns(config.MaxOpenConns)

    // Устанавливаем максимальное количество простаивающих соединений
    // Они поддерживаются живыми и готовыми к повторному использованию
    // Должно быть меньше или равно MaxOpenConns
    db.SetMaxIdleConns(config.MaxIdleConns)
}`,
            description: `Реализуйте функцию, которая настраивает пул соединений с базой данных с соответствующими ограничениями. Пул соединений необходим для производительности, так как создание новых соединений дорого. Правильная конфигурация предотвращает истощение ресурсов и утечки соединений.

**Требования:**
- Установите максимум открытых соединений (SetMaxOpenConns)
- Установите максимум простаивающих соединений (SetMaxIdleConns)
- Понимайте взаимосвязь между этими настройками
- Верните настроенный экземпляр базы данных

**Рекомендуемые настройки:**
- MaxOpenConns: 25-100 (зависит от нагрузки)
- MaxIdleConns: 5-25 (обычно 25% от MaxOpen)
- Держите простаивающие соединения готовыми для быстрого повторного использования`,
            hint1: `Используйте db.SetMaxOpenConns() для ограничения общего количества открытых соединений (используемых и простаивающих).`,
            hint2: `Используйте db.SetMaxIdleConns() для ограничения количества простаивающих соединений в пуле. Это должно быть меньше MaxOpenConns.`,
            whyItMatters: `Правильная конфигурация пула соединений критически важна для производительности и стабильности приложения. Слишком мало соединений вызывает конкуренцию и замедляет время отклика. Слишком много соединений может перегрузить сервер базы данных и потратить память. Нахождение правильного баланса необходимо для продакшн систем.

**Продакшен паттерн:**
\`\`\`go
// Оптимальная конфигурация для веб-сервера
db.SetMaxOpenConns(25)    // Макс соединений
db.SetMaxIdleConns(5)     // Готовые соединения
db.SetConnMaxLifetime(5 * time.Minute)

// Адаптация под нагрузку
if highTraffic {
    db.SetMaxOpenConns(100)
    db.SetMaxIdleConns(25)
}
\`\`\`

**Практические преимущества:**
- Предотвращение истощения соединений
- Быстрое переиспользование готовых соединений
- Адаптация к различным нагрузкам`
        },
        uz: {
            title: 'Ulanish pool konfiguratsiyasi',
            solutionCode: `package dbx

import (
    "database/sql"
)

type PoolConfig struct {
    MaxOpenConns int
    MaxIdleConns int
}

func ConfigurePool(db *sql.DB, config PoolConfig) {
    // Ochiq ulanishlarning maksimal sonini o'rnatamiz
    // Bu umumiy ulanishlar sonini cheklaydi (bo'sh + ishlatilayotgan)
    db.SetMaxOpenConns(config.MaxOpenConns)

    // Bo'sh ulanishlarning maksimal sonini o'rnatamiz
    // Ular tirik saqlanadi va qayta ishlatish uchun tayyor bo'ladi
    // MaxOpenConns dan kam yoki teng bo'lishi kerak
    db.SetMaxIdleConns(config.MaxIdleConns)
}`,
            description: `Tegishli cheklovlar bilan ma'lumotlar bazasi ulanish poolini sozlaydigan funksiyani amalga oshiring. Ulanish pooli unumdorlik uchun zarur, chunki yangi ulanishlar yaratish qimmat. To'g'ri konfiguratsiya resurslarning tugashini va ulanish oqishini oldini oladi.

**Talablar:**
- Maksimal ochiq ulanishlarni o'rnating (SetMaxOpenConns)
- Maksimal bo'sh ulanishlarni o'rnating (SetMaxIdleConns)
- Bu sozlamalar o'rtasidagi munosabatni tushuning
- Sozlangan ma'lumotlar bazasi instansiyasini qaytaring

**Tavsiya etilgan sozlamalar:**
- MaxOpenConns: 25-100 (ish yukiga bog'liq)
- MaxIdleConns: 5-25 (odatda MaxOpen ning 25%)
- Bo'sh ulanishlarni tez qayta ishlatish uchun tayyor saqlang`,
            hint1: `Ochiq ulanishlarning umumiy sonini cheklash uchun db.SetMaxOpenConns() dan foydalaning (ishlatilayotgan va bo'sh).`,
            hint2: `Pooldagi bo'sh ulanishlar sonini cheklash uchun db.SetMaxIdleConns() dan foydalaning. Bu MaxOpenConns dan kam bo'lishi kerak.`,
            whyItMatters: `To'g'ri ulanish pooli konfiguratsiyasi dastur unumdorligi va barqarorligi uchun juda muhimdir. Juda kam ulanishlar raqobatni keltirib chiqaradi va javob vaqtini sekinlashtiradi. Juda ko'p ulanishlar ma'lumotlar bazasi serverini ortiqcha yuklashi va xotirani isrof qilishi mumkin. To'g'ri muvozanatni topish ishlab chiqarish tizimlari uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`go
// Veb-server uchun optimal konfiguratsiya
db.SetMaxOpenConns(25)    // Maksimal ulanishlar
db.SetMaxIdleConns(5)     // Tayyor ulanishlar
db.SetConnMaxLifetime(5 * time.Minute)

// Yukka moslashish
if highTraffic {
    db.SetMaxOpenConns(100)
    db.SetMaxIdleConns(25)
}
\`\`\`

**Amaliy foydalari:**
- Ulanishlar tugashining oldini olish
- Tayyor ulanishlarni tez qayta ishlatish
- Turli yuklarga moslashish`
        }
    }
};

export default task;
