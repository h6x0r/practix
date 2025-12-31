import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-stats-monitoring',
    title: 'Monitor Pool Statistics',
    difficulty: 'medium',
    tags: ['go', 'database', 'connection-pool', 'monitoring'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that collects and returns connection pool statistics. Monitoring pool metrics helps identify performance bottlenecks, connection leaks, and configuration issues before they cause outages.

**Requirements:**
- Use db.Stats() to get pool statistics
- Return relevant metrics in a structured format
- Calculate utilization percentages
- Provide actionable insights

**Key Metrics:**
- OpenConnections: Currently open connections
- InUse: Connections currently in use
- Idle: Connections waiting in pool
- WaitCount: Total number of waits for a connection
- WaitDuration: Total time spent waiting`,
    initialCode: `package dbx

import (
    "database/sql"
    "time"
)

type PoolStats struct {
    MaxOpen          int
    OpenConnections  int
    InUse            int
    Idle             int
    WaitCount        int64
    WaitDuration     time.Duration
    UtilizationPct   float64
}

// TODO: Get connection pool statistics
func GetPoolStats(db *sql.DB) PoolStats {
    panic("TODO: implement with db.Stats()")
}`,
    solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type PoolStats struct {
    MaxOpen          int
    OpenConnections  int
    InUse            int
    Idle             int
    WaitCount        int64
    WaitDuration     time.Duration
    UtilizationPct   float64
}

func GetPoolStats(db *sql.DB) PoolStats {
    // Get statistics from database
    stats := db.Stats()

    // Calculate utilization percentage
    var utilizationPct float64
    if stats.MaxOpenConnections > 0 {
        utilizationPct = float64(stats.OpenConnections) / float64(stats.MaxOpenConnections) * 100
    }

    // Return structured metrics
    return PoolStats{
        MaxOpen:          stats.MaxOpenConnections,
        OpenConnections:  stats.OpenConnections,
        InUse:            stats.InUse,
        Idle:             stats.Idle,
        WaitCount:        stats.WaitCount,
        WaitDuration:     stats.WaitDuration,
        UtilizationPct:   utilizationPct,
    }
}`,
	testCode: `package dbx

import (
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// GetPoolStats returns PoolStats struct
	db, _, _ := sqlmock.New()
	defer db.Close()

	db.SetMaxOpenConns(25)

	stats := GetPoolStats(db)
	if stats.MaxOpen != 25 {
		t.Errorf("expected MaxOpen 25, got %d", stats.MaxOpen)
	}
}

func Test2(t *testing.T) {
	// GetPoolStats returns correct MaxOpen
	db, _, _ := sqlmock.New()
	defer db.Close()

	db.SetMaxOpenConns(100)

	stats := GetPoolStats(db)
	if stats.MaxOpen != 100 {
		t.Errorf("expected MaxOpen 100, got %d", stats.MaxOpen)
	}
}

func Test3(t *testing.T) {
	// GetPoolStats OpenConnections is non-negative
	db, _, _ := sqlmock.New()
	defer db.Close()

	stats := GetPoolStats(db)
	if stats.OpenConnections < 0 {
		t.Errorf("expected non-negative OpenConnections, got %d", stats.OpenConnections)
	}
}

func Test4(t *testing.T) {
	// GetPoolStats InUse is non-negative
	db, _, _ := sqlmock.New()
	defer db.Close()

	stats := GetPoolStats(db)
	if stats.InUse < 0 {
		t.Errorf("expected non-negative InUse, got %d", stats.InUse)
	}
}

func Test5(t *testing.T) {
	// GetPoolStats Idle is non-negative
	db, _, _ := sqlmock.New()
	defer db.Close()

	stats := GetPoolStats(db)
	if stats.Idle < 0 {
		t.Errorf("expected non-negative Idle, got %d", stats.Idle)
	}
}

func Test6(t *testing.T) {
	// GetPoolStats WaitCount is non-negative
	db, _, _ := sqlmock.New()
	defer db.Close()

	stats := GetPoolStats(db)
	if stats.WaitCount < 0 {
		t.Errorf("expected non-negative WaitCount, got %d", stats.WaitCount)
	}
}

func Test7(t *testing.T) {
	// GetPoolStats WaitDuration is non-negative
	db, _, _ := sqlmock.New()
	defer db.Close()

	stats := GetPoolStats(db)
	if stats.WaitDuration < 0 {
		t.Errorf("expected non-negative WaitDuration, got %v", stats.WaitDuration)
	}
}

func Test8(t *testing.T) {
	// GetPoolStats calculates utilization correctly
	db, _, _ := sqlmock.New()
	defer db.Close()

	db.SetMaxOpenConns(100)

	stats := GetPoolStats(db)
	// With no active connections, utilization should be 0 or very low
	if stats.UtilizationPct < 0 || stats.UtilizationPct > 100 {
		t.Errorf("expected utilization between 0 and 100, got %f", stats.UtilizationPct)
	}
}

func Test9(t *testing.T) {
	// GetPoolStats handles zero MaxOpen (unlimited)
	db, _, _ := sqlmock.New()
	defer db.Close()

	db.SetMaxOpenConns(0) // 0 means unlimited

	stats := GetPoolStats(db)
	// Should not panic and handle division by zero
	_ = stats
}

func Test10(t *testing.T) {
	// GetPoolStats can be called multiple times
	db, _, _ := sqlmock.New()
	defer db.Close()

	db.SetMaxOpenConns(50)

	stats1 := GetPoolStats(db)
	stats2 := GetPoolStats(db)

	if stats1.MaxOpen != 50 || stats2.MaxOpen != 50 {
		t.Errorf("expected consistent MaxOpen 50, got %d and %d", stats1.MaxOpen, stats2.MaxOpen)
	}
}
`,
    hint1: `Call db.Stats() to get sql.DBStats struct containing all pool metrics. Access its fields to extract the information you need.`,
    hint2: `Calculate utilization as (OpenConnections / MaxOpenConnections * 100). High utilization (>80%) suggests you may need more connections.`,
    whyItMatters: `Monitoring connection pool statistics is essential for maintaining healthy database operations. High wait counts indicate contention, high utilization suggests undersized pools, and connection leaks show up as growing open connections. These metrics enable proactive problem detection and capacity planning.

**Production Pattern:**
\`\`\`go
// Periodic metrics export to Prometheus
func exportMetrics() {
    stats := db.Stats()

    prometheus.Gauge("db_open_connections").Set(float64(stats.OpenConnections))
    prometheus.Gauge("db_in_use").Set(float64(stats.InUse))
    prometheus.Counter("db_wait_count").Add(float64(stats.WaitCount))

    // Alert on high utilization
    if utilization > 80 {
        alert("High DB pool utilization")
    }
}
\`\`\`

**Practical Benefits:**
- Early detection of connection leaks
- Pool size optimization
- Proactive capacity planning`,
    order: 3,
    translations: {
        ru: {
            title: 'Мониторинг статистики пула',
            solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type PoolStats struct {
    MaxOpen          int
    OpenConnections  int
    InUse            int
    Idle             int
    WaitCount        int64
    WaitDuration     time.Duration
    UtilizationPct   float64
}

func GetPoolStats(db *sql.DB) PoolStats {
    // Получаем статистику из базы данных
    stats := db.Stats()

    // Вычисляем процент использования
    var utilizationPct float64
    if stats.MaxOpenConnections > 0 {
        utilizationPct = float64(stats.OpenConnections) / float64(stats.MaxOpenConnections) * 100
    }

    // Возвращаем структурированные метрики
    return PoolStats{
        MaxOpen:          stats.MaxOpenConnections,
        OpenConnections:  stats.OpenConnections,
        InUse:            stats.InUse,
        Idle:             stats.Idle,
        WaitCount:        stats.WaitCount,
        WaitDuration:     stats.WaitDuration,
        UtilizationPct:   utilizationPct,
    }
}`,
            description: `Реализуйте функцию, которая собирает и возвращает статистику пула соединений. Мониторинг метрик пула помогает идентифицировать узкие места производительности, утечки соединений и проблемы конфигурации до того, как они вызовут сбои.

**Требования:**
- Используйте db.Stats() для получения статистики пула
- Верните релевантные метрики в структурированном формате
- Вычислите проценты использования
- Предоставьте практические выводы

**Ключевые метрики:**
- OpenConnections: Текущие открытые соединения
- InUse: Соединения в данный момент используемые
- Idle: Соединения ожидающие в пуле
- WaitCount: Общее количество ожиданий для соединения
- WaitDuration: Общее время проведенное в ожидании`,
            hint1: `Вызовите db.Stats() для получения структуры sql.DBStats содержащей все метрики пула. Обратитесь к её полям для извлечения нужной информации.`,
            hint2: `Вычислите использование как (OpenConnections / MaxOpenConnections * 100). Высокое использование (>80%) предполагает что вам может понадобиться больше соединений.`,
            whyItMatters: `Мониторинг статистики пула соединений необходим для поддержания здоровых операций базы данных. Высокие счетчики ожиданий указывают на конкуренцию, высокое использование предполагает недостаточный размер пула, а утечки соединений проявляются как растущие открытые соединения. Эти метрики обеспечивают проактивное обнаружение проблем и планирование мощности.

**Продакшен паттерн:**
\`\`\`go
// Периодический экспорт метрик в Prometheus
func exportMetrics() {
    stats := db.Stats()

    prometheus.Gauge("db_open_connections").Set(float64(stats.OpenConnections))
    prometheus.Gauge("db_in_use").Set(float64(stats.InUse))
    prometheus.Counter("db_wait_count").Add(float64(stats.WaitCount))

    // Тревога при высокой утилизации
    if utilization > 80 {
        alert("High DB pool utilization")
    }
}
\`\`\`

**Практические преимущества:**
- Раннее обнаружение утечек соединений
- Оптимизация размера пула
- Проактивное планирование мощности`
        },
        uz: {
            title: 'Pool statistikasini monitoring',
            solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type PoolStats struct {
    MaxOpen          int
    OpenConnections  int
    InUse            int
    Idle             int
    WaitCount        int64
    WaitDuration     time.Duration
    UtilizationPct   float64
}

func GetPoolStats(db *sql.DB) PoolStats {
    // Ma'lumotlar bazasidan statistikani olamiz
    stats := db.Stats()

    // Foydalanish foizini hisoblaymiz
    var utilizationPct float64
    if stats.MaxOpenConnections > 0 {
        utilizationPct = float64(stats.OpenConnections) / float64(stats.MaxOpenConnections) * 100
    }

    // Tuzilgan ko'rsatkichlarni qaytaramiz
    return PoolStats{
        MaxOpen:          stats.MaxOpenConnections,
        OpenConnections:  stats.OpenConnections,
        InUse:            stats.InUse,
        Idle:             stats.Idle,
        WaitCount:        stats.WaitCount,
        WaitDuration:     stats.WaitDuration,
        UtilizationPct:   utilizationPct,
    }
}`,
            description: `Ulanish pooli statistikasini yig'adigan va qaytaradigan funksiyani amalga oshiring. Pool ko'rsatkichlarini monitoring qilish unumdorlik to'siqlari, ulanish oqishlari va konfiguratsiya muammolarini ular uzilishlarga olib kelishidan oldin aniqlashga yordam beradi.

**Talablar:**
- Pool statistikasini olish uchun db.Stats() dan foydalaning
- Tegishli ko'rsatkichlarni tuzilgan formatda qaytaring
- Foydalanish foizlarini hisoblang
- Amaliy xulosalar taqdim eting

**Asosiy ko'rsatkichlar:**
- OpenConnections: Hozirgi ochiq ulanishlar
- InUse: Hozirda ishlatilayotgan ulanishlar
- Idle: Poolda kutayotgan ulanishlar
- WaitCount: Ulanish uchun kutishlarning umumiy soni
- WaitDuration: Kutishda o'tkazilgan umumiy vaqt`,
            hint1: `Barcha pool ko'rsatkichlarini o'z ichiga olgan sql.DBStats strukturasini olish uchun db.Stats() ni chaqiring. Kerakli ma'lumotni olish uchun uning maydonlariga murojaat qiling.`,
            hint2: `Foydalanishni (OpenConnections / MaxOpenConnections * 100) sifatida hisoblang. Yuqori foydalanish (>80%) ko'proq ulanishlar kerakligini ko'rsatadi.`,
            whyItMatters: `Ulanish pooli statistikasini monitoring qilish sog'lom ma'lumotlar bazasi operatsiyalarini saqlash uchun zarur. Yuqori kutish hisoblagichlari raqobatni ko'rsatadi, yuqori foydalanish kichik poolni ko'rsatadi va ulanish oqishlari o'sib borayotgan ochiq ulanishlar sifatida namoyon bo'ladi. Bu ko'rsatkichlar proaktiv muammo aniqlash va quvvat rejalashtirish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`go
// Prometheusga davriy ko'rsatkichlar eksporti
func exportMetrics() {
    stats := db.Stats()

    prometheus.Gauge("db_open_connections").Set(float64(stats.OpenConnections))
    prometheus.Gauge("db_in_use").Set(float64(stats.InUse))
    prometheus.Counter("db_wait_count").Add(float64(stats.WaitCount))

    // Yuqori foydalanishda ogohlantirish
    if utilization > 80 {
        alert("High DB pool utilization")
    }
}
\`\`\`

**Amaliy foydalari:**
- Ulanish oqishini erta aniqlash
- Pool hajmini optimallashtirish
- Proaktiv quvvat rejalashtirish`
        }
    }
};

export default task;
