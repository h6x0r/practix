import { Task } from '../../../../types';

export const task: Task = {
    slug: 'go-db-connection-lifetime',
    title: 'Connection Lifetime Settings',
    difficulty: 'medium',
    tags: ['go', 'database', 'connection-pool', 'lifetime'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a function that configures connection lifetime settings. Database connections shouldn't be kept alive forever - they should be recycled periodically to handle network issues, database restarts, and connection quality degradation.

**Requirements:**
- Set maximum connection lifetime (SetConnMaxLifetime)
- Set maximum idle time (SetConnMaxIdleTime)
- Understand why connections should be recycled
- Use appropriate durations for production

**Recommended Settings:**
- ConnMaxLifetime: 5-30 minutes
- ConnMaxIdleTime: 1-5 minutes
- Balance between connection reuse and freshness`,
    initialCode: `package dbx

import (
    "database/sql"
    "time"
)

type LifetimeConfig struct {
    MaxLifetime time.Duration
    MaxIdleTime time.Duration
}

// TODO: Configure connection lifetime settings
func ConfigureLifetime(db *sql.DB, config LifetimeConfig) {
    panic("TODO: implement SetConnMaxLifetime and SetConnMaxIdleTime")
}`,
    solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type LifetimeConfig struct {
    MaxLifetime time.Duration
    MaxIdleTime time.Duration
}

func ConfigureLifetime(db *sql.DB, config LifetimeConfig) {
    // Set maximum lifetime for connections
    // Connections will be closed after this duration
    // Helps handle server restarts, network issues, etc.
    db.SetConnMaxLifetime(config.MaxLifetime)

    // Set maximum idle time for connections
    // Idle connections will be closed after this duration
    // Prevents holding connections that won't be reused
    db.SetConnMaxIdleTime(config.MaxIdleTime)
}`,
	testCode: `package dbx

import (
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func Test1(t *testing.T) {
	// ConfigureLifetime does not panic with valid config
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 5 * time.Minute, MaxIdleTime: 2 * time.Minute}
	ConfigureLifetime(db, config)
}

func Test2(t *testing.T) {
	// ConfigureLifetime handles zero durations
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 0, MaxIdleTime: 0}
	ConfigureLifetime(db, config)
}

func Test3(t *testing.T) {
	// ConfigureLifetime handles large durations
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 24 * time.Hour, MaxIdleTime: 1 * time.Hour}
	ConfigureLifetime(db, config)
}

func Test4(t *testing.T) {
	// ConfigureLifetime accepts idle greater than lifetime
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 1 * time.Minute, MaxIdleTime: 5 * time.Minute}
	ConfigureLifetime(db, config)
}

func Test5(t *testing.T) {
	// ConfigureLifetime accepts equal values
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 10 * time.Minute, MaxIdleTime: 10 * time.Minute}
	ConfigureLifetime(db, config)
}

func Test6(t *testing.T) {
	// ConfigureLifetime with typical cloud values
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 30 * time.Minute, MaxIdleTime: 5 * time.Minute}
	ConfigureLifetime(db, config)
}

func Test7(t *testing.T) {
	// ConfigureLifetime with milliseconds
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 500 * time.Millisecond, MaxIdleTime: 100 * time.Millisecond}
	ConfigureLifetime(db, config)
}

func Test8(t *testing.T) {
	// ConfigureLifetime with seconds
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 30 * time.Second, MaxIdleTime: 10 * time.Second}
	ConfigureLifetime(db, config)
}

func Test9(t *testing.T) {
	// ConfigureLifetime can be called multiple times
	db, _, _ := sqlmock.New()
	defer db.Close()

	config1 := LifetimeConfig{MaxLifetime: 1 * time.Minute, MaxIdleTime: 30 * time.Second}
	ConfigureLifetime(db, config1)

	config2 := LifetimeConfig{MaxLifetime: 10 * time.Minute, MaxIdleTime: 2 * time.Minute}
	ConfigureLifetime(db, config2)
}

func Test10(t *testing.T) {
	// ConfigureLifetime handles nanoseconds
	db, _, _ := sqlmock.New()
	defer db.Close()

	config := LifetimeConfig{MaxLifetime: 1 * time.Nanosecond, MaxIdleTime: 1 * time.Nanosecond}
	ConfigureLifetime(db, config)
}
`,
    hint1: `Use db.SetConnMaxLifetime() to set how long a connection can be reused before being closed. This applies to all connections.`,
    hint2: `Use db.SetConnMaxIdleTime() to set how long a connection can be idle before being closed. This only applies to idle connections in the pool.`,
    whyItMatters: `Connection lifetime management prevents stale connections and helps recover from network issues automatically. Without proper lifetime limits, connections can become unusable due to firewall timeouts, database server restarts, or connection quality degradation. This is especially important in cloud environments with load balancers.

**Production Pattern:**
\`\`\`go
// Configuration for cloud environment
db.SetConnMaxLifetime(5 * time.Minute)  // Connection recycling
db.SetConnMaxIdleTime(2 * time.Minute)  // Close unused

// AWS RDS recommendations
db.SetConnMaxLifetime(30 * time.Minute)
db.SetConnMaxIdleTime(5 * time.Minute)
\`\`\`

**Practical Benefits:**
- Automatic recovery from network failures
- Works with load balancers
- Prevents firewall timeouts`,
    order: 1,
    translations: {
        ru: {
            title: 'Время жизни соединения',
            solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type LifetimeConfig struct {
    MaxLifetime time.Duration
    MaxIdleTime time.Duration
}

func ConfigureLifetime(db *sql.DB, config LifetimeConfig) {
    // Устанавливаем максимальное время жизни для соединений
    // Соединения будут закрыты после этой длительности
    // Помогает обрабатывать перезапуски сервера, проблемы сети и т.д.
    db.SetConnMaxLifetime(config.MaxLifetime)

    // Устанавливаем максимальное время простоя для соединений
    // Простаивающие соединения будут закрыты после этой длительности
    // Предотвращает удержание соединений, которые не будут переиспользованы
    db.SetConnMaxIdleTime(config.MaxIdleTime)
}`,
            description: `Реализуйте функцию, которая настраивает параметры времени жизни соединений. Соединения с базой данных не должны поддерживаться вечно - они должны периодически переиспользоваться для обработки сетевых проблем, перезапусков базы данных и деградации качества соединений.

**Требования:**
- Установите максимальное время жизни соединения (SetConnMaxLifetime)
- Установите максимальное время простоя (SetConnMaxIdleTime)
- Понимайте почему соединения должны переиспользоваться
- Используйте подходящие длительности для продакшена

**Рекомендуемые настройки:**
- ConnMaxLifetime: 5-30 минут
- ConnMaxIdleTime: 1-5 минут
- Баланс между переиспользованием соединений и их свежестью`,
            hint1: `Используйте db.SetConnMaxLifetime() для установки как долго соединение может переиспользоваться перед закрытием. Это применяется ко всем соединениям.`,
            hint2: `Используйте db.SetConnMaxIdleTime() для установки как долго соединение может простаивать перед закрытием. Это применяется только к простаивающим соединениям в пуле.`,
            whyItMatters: `Управление временем жизни соединений предотвращает устаревшие соединения и помогает автоматически восстанавливаться после сетевых проблем. Без правильных ограничений времени жизни соединения могут стать непригодными из-за тайм-аутов файрвола, перезапусков сервера базы данных или деградации качества соединения. Это особенно важно в облачных средах с балансировщиками нагрузки.

**Продакшен паттерн:**
\`\`\`go
// Конфигурация для облачной среды
db.SetConnMaxLifetime(5 * time.Minute)  // Переиспользование соединений
db.SetConnMaxIdleTime(2 * time.Minute)  // Закрытие неиспользуемых

// AWS RDS рекомендации
db.SetConnMaxLifetime(30 * time.Minute)
db.SetConnMaxIdleTime(5 * time.Minute)
\`\`\`

**Практические преимущества:**
- Автоматическое восстановление после сбоев сети
- Работа с балансировщиками нагрузки
- Предотвращение firewall тайм-аутов`
        },
        uz: {
            title: 'Ulanish lifetime',
            solutionCode: `package dbx

import (
    "database/sql"
    "time"
)

type LifetimeConfig struct {
    MaxLifetime time.Duration
    MaxIdleTime time.Duration
}

func ConfigureLifetime(db *sql.DB, config LifetimeConfig) {
    // Ulanishlar uchun maksimal umrni o'rnatamiz
    // Ulanishlar bu muddatdan keyin yopiladi
    // Server qayta ishga tushirish, tarmoq muammolari va boshqalarni hal qilishga yordam beradi
    db.SetConnMaxLifetime(config.MaxLifetime)

    // Ulanishlar uchun maksimal bo'sh vaqtni o'rnatamiz
    // Bo'sh ulanishlar bu muddatdan keyin yopiladi
    // Qayta ishlatilmaydigan ulanishlarni ushlab turishning oldini oladi
    db.SetConnMaxIdleTime(config.MaxIdleTime)
}`,
            description: `Ulanish umr sozlamalarini sozlaydigan funksiyani amalga oshiring. Ma'lumotlar bazasi ullanishlari abadiy tirik saqlanmasligi kerak - ular tarmoq muammolari, ma'lumotlar bazasi qayta ishga tushirish va ulanish sifati yomonlashuvini hal qilish uchun davriy ravishda qayta tiklanishi kerak.

**Talablar:**
- Maksimal ulanish umrini o'rnating (SetConnMaxLifetime)
- Maksimal bo'sh vaqtni o'rnating (SetConnMaxIdleTime)
- Ulanishlar nima uchun qayta tiklanishi kerakligini tushuning
- Ishlab chiqarish uchun tegishli muddatlardan foydalaning

**Tavsiya etilgan sozlamalar:**
- ConnMaxLifetime: 5-30 daqiqa
- ConnMaxIdleTime: 1-5 daqiqa
- Ulanishni qayta ishlatish va yangilik o'rtasida muvozanat`,
            hint1: `Ulanish yopilishidan oldin qancha vaqt qayta ishlatilishi mumkinligini o'rnatish uchun db.SetConnMaxLifetime() dan foydalaning. Bu barcha ullanishlarga taalluqlidir.`,
            hint2: `Ulanish yopilishidan oldin qancha vaqt bo'sh turishi mumkinligini o'rnatish uchun db.SetConnMaxIdleTime() dan foydalaning. Bu faqat pooldagi bo'sh ullanishlarga taalluqlidir.`,
            whyItMatters: `Ulanish umrini boshqarish eskirgan ulanishlarning oldini oladi va tarmoq muammolaridan avtomatik ravishda tiklanishga yordam beradi. To'g'ri umr cheklovlarisiz, ulanishlar firewall vaqt tugashi, ma'lumotlar bazasi serveri qayta ishga tushirish yoki ulanish sifati yomonlashuvi tufayli foydalanib bo'lmaydi. Bu ayniqsa load balancerlari bilan bulutli muhitlarda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`go
// Bulutli muhit uchun konfiguratsiya
db.SetConnMaxLifetime(5 * time.Minute)  // Ulanishlarni qayta ishlatish
db.SetConnMaxIdleTime(2 * time.Minute)  // Ishlatilmayotganlarni yopish

// AWS RDS tavsifalari
db.SetConnMaxLifetime(30 * time.Minute)
db.SetConnMaxIdleTime(5 * time.Minute)
\`\`\`

**Amaliy foydalari:**
- Tarmoq nosozliklaridan avtomatik tiklanish
- Load balancerlar bilan ishlash
- Firewall vaqt tugashining oldini olish`
        }
    }
};

export default task;
