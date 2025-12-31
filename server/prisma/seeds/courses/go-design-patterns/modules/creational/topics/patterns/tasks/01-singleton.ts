import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-singleton',
	title: 'Singleton Pattern',
	difficulty: 'easy',
	tags: ['go', 'design-patterns', 'creational', 'singleton'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Singleton pattern in Go - ensure a class has only one instance and provide a global point of access to it.

**You will implement:**

1. **Config struct** - Application configuration singleton
2. **GetConfig() *Config** - Thread-safe singleton accessor using sync.Once
3. **SetValue(key, value string)** - Set configuration value
4. **GetValue(key string) string** - Get configuration value

**Key Concepts:**
- **sync.Once**: Guarantees initialization happens exactly once
- **Thread Safety**: Safe for concurrent access
- **Lazy Initialization**: Instance created only when first requested

**Example Usage:**

\`\`\`go
// First call creates the instance
config1 := GetConfig()
config1.SetValue("database", "postgres://localhost:5432")

// Second call returns the same instance
config2 := GetConfig()
fmt.Println(config2.GetValue("database")) // postgres://localhost:5432

// Verify same instance
fmt.Println(config1 == config2) // true
\`\`\`

**When to use Singleton:**
- Database connection pools
- Configuration managers
- Logging services
- Thread pools
- Caches

**Constraints:**
- Must be thread-safe using sync.Once
- Must use lazy initialization
- GetConfig must always return the same instance`,
	initialCode: `package patterns

import (
	"sync"
)

type Config struct {
	mu     sync.RWMutex
	values map[string]string
}

)

func GetConfig() *Config {
}

func (c *Config) SetValue(key, value string) {
}

func (c *Config) GetValue(key string) string {
}`,
	solutionCode: `package patterns

import (
	"sync"
)

type Config struct {
	mu     sync.RWMutex	// protects concurrent access to values map
	values map[string]string	// stores configuration key-value pairs
}

var (
	instance *Config	// the single instance of Config
	once     sync.Once	// ensures initialization happens exactly once
)

func GetConfig() *Config {
	once.Do(func() {	// executes only on first call, all other goroutines wait
		instance = &Config{	// create the single instance
			values: make(map[string]string),	// initialize the map to avoid nil map panic
		}
	})
	return instance	// always returns the same instance
}

func (c *Config) SetValue(key, value string) {
	c.mu.Lock()	// acquire exclusive write lock, blocks all other readers and writers
	defer c.mu.Unlock()	// release lock when function returns, even on panic
	c.values[key] = value	// safe to write, we have exclusive access
}

func (c *Config) GetValue(key string) string {
	c.mu.RLock()	// acquire shared read lock, allows multiple concurrent readers
	defer c.mu.RUnlock()	// release read lock when function returns
	return c.values[key]	// returns empty string if key doesn't exist (Go map behavior)
}`,
	hint1: `Use sync.Once.Do() to initialize the instance. Inside the Do function, create a new Config with an initialized map using make(map[string]string).`,
	hint2: `For SetValue use c.mu.Lock()/Unlock(), for GetValue use c.mu.RLock()/RUnlock(). RLock allows multiple concurrent readers.`,
	whyItMatters: `The Singleton pattern is essential for managing shared resources in concurrent applications.

**Why Singleton Matters:**

**1. Resource Management**
Database connections, file handles, and network sockets are expensive to create. Singleton ensures we reuse a single instance:

\`\`\`go
// Without Singleton - creates new connection every time (MEMORY LEAK!)
func GetDB() *sql.DB {
    db, _ := sql.Open("postgres", connStr)	// new connection each call
    return db	// previous connections never closed!
}

// With Singleton - reuses single connection pool
var dbOnce sync.Once
var db *sql.DB

func GetDB() *sql.DB {
    dbOnce.Do(func() {	// runs only once
        db, _ = sql.Open("postgres", connStr)	// create pool once
        db.SetMaxOpenConns(25)	// configure pool size
    })
    return db	// always returns same pool
}
\`\`\`

**2. Configuration Consistency**
All parts of your application see the same configuration:

\`\`\`go
// Service A sets config
config := GetConfig()
config.SetValue("api_url", "https://api.example.com")

// Service B (different goroutine) reads same config
config := GetConfig()	// same instance!
url := config.GetValue("api_url")	// gets "https://api.example.com"
\`\`\`

**3. Thread Safety with sync.Once**
sync.Once is the idiomatic Go way to implement Singleton:

\`\`\`go
// sync.Once guarantees:
// 1. Initialization runs exactly once
// 2. All goroutines wait for initialization to complete
// 3. No race conditions - internally uses atomic operations

once.Do(func() {
    // This code runs only once, even with 1000 concurrent calls
    instance = &Config{values: make(map[string]string)}
})
\`\`\`

**Real-World Examples in Go Standard Library:**
- \`http.DefaultClient\` - Go's default HTTP client singleton
- \`http.DefaultServeMux\` - Default HTTP request multiplexer
- \`log.Logger\` - Standard logger instance
- \`database/sql.DB\` - Connection pool (should be singleton per database)

**Production Pattern:**
\`\`\`go
// Production-ready singleton with graceful shutdown
type AppConfig struct {
    once     sync.Once
    mu       sync.RWMutex
    settings map[string]interface{}
    shutdown chan struct{}
}

var appConfig *AppConfig

func GetAppConfig() *AppConfig {
    if appConfig == nil {
        appConfig = &AppConfig{
            shutdown: make(chan struct{}),
        }
    }
    appConfig.once.Do(func() {
        appConfig.settings = loadFromFile()	// load config once
        go appConfig.watchForChanges()	// hot reload in background
    })
    return appConfig
}
\`\`\`

**Common Mistakes to Avoid:**
- Using mutex instead of sync.Once (less efficient, more code)
- Forgetting thread safety for the instance's methods
- Creating multiple instances in tests (use build tags or interfaces)
- Not handling initialization errors (use sync.Once with error channel)`,
	order: 0,
	testCode: `package patterns

import (
	"sync"
	"testing"
)

// Test1: GetConfig returns same instance
func Test1(t *testing.T) {
	c1 := GetConfig()
	c2 := GetConfig()
	if c1 != c2 {
		t.Error("GetConfig should return same instance")
	}
}

// Test2: GetConfig returns non-nil
func Test2(t *testing.T) {
	c := GetConfig()
	if c == nil {
		t.Error("GetConfig should not return nil")
	}
}

// Test3: SetValue and GetValue work correctly
func Test3(t *testing.T) {
	c := GetConfig()
	c.SetValue("key1", "value1")
	if c.GetValue("key1") != "value1" {
		t.Error("GetValue should return set value")
	}
}

// Test4: GetValue returns empty string for missing key
func Test4(t *testing.T) {
	c := GetConfig()
	if c.GetValue("nonexistent") != "" {
		t.Error("GetValue should return empty string for missing key")
	}
}

// Test5: Multiple SetValue overwrites previous
func Test5(t *testing.T) {
	c := GetConfig()
	c.SetValue("key2", "first")
	c.SetValue("key2", "second")
	if c.GetValue("key2") != "second" {
		t.Error("SetValue should overwrite previous value")
	}
}

// Test6: Config struct has values map
func Test6(t *testing.T) {
	c := GetConfig()
	c.SetValue("test6", "val")
	if c.GetValue("test6") != "val" {
		t.Error("Config should store values in map")
	}
}

// Test7: Concurrent GetConfig returns same instance
func Test7(t *testing.T) {
	var wg sync.WaitGroup
	configs := make([]*Config, 10)
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			configs[idx] = GetConfig()
		}(i)
	}
	wg.Wait()
	for i := 1; i < 10; i++ {
		if configs[i] != configs[0] {
			t.Error("Concurrent GetConfig should return same instance")
		}
	}
}

// Test8: Value persists across GetConfig calls
func Test8(t *testing.T) {
	c1 := GetConfig()
	c1.SetValue("persist", "data")
	c2 := GetConfig()
	if c2.GetValue("persist") != "data" {
		t.Error("Value should persist across GetConfig calls")
	}
}

// Test9: Config has RWMutex
func Test9(t *testing.T) {
	c := &Config{values: make(map[string]string)}
	c.SetValue("mutex_test", "ok")
	if c.GetValue("mutex_test") != "ok" {
		t.Error("Config should use mutex for thread-safe access")
	}
}

// Test10: Empty string key works
func Test10(t *testing.T) {
	c := GetConfig()
	c.SetValue("", "empty_key_value")
	if c.GetValue("") != "empty_key_value" {
		t.Error("Empty string key should work")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Singleton (Одиночка)',
			description: `Реализуйте паттерн Singleton на Go — гарантируйте, что класс имеет только один экземпляр и предоставьте глобальную точку доступа к нему.

**Вы реализуете:**

1. **Config struct** — Singleton конфигурации приложения
2. **GetConfig() *Config** — Потокобезопасный доступ к singleton через sync.Once
3. **SetValue(key, value string)** — Установка значения конфигурации
4. **GetValue(key string) string** — Получение значения конфигурации

**Ключевые концепции:**
- **sync.Once**: Гарантирует однократную инициализацию
- **Потокобезопасность**: Безопасен для конкурентного доступа
- **Ленивая инициализация**: Экземпляр создаётся только при первом запросе

**Пример использования:**

\`\`\`go
// Первый вызов создаёт экземпляр
config1 := GetConfig()
config1.SetValue("database", "postgres://localhost:5432")

// Второй вызов возвращает тот же экземпляр
config2 := GetConfig()
fmt.Println(config2.GetValue("database")) // postgres://localhost:5432

// Проверка одного экземпляра
fmt.Println(config1 == config2) // true
\`\`\`

**Когда использовать Singleton:**
- Пулы подключений к БД
- Менеджеры конфигурации
- Сервисы логирования
- Пулы потоков
- Кэши

**Ограничения:**
- Должен быть потокобезопасным с использованием sync.Once
- Должен использовать ленивую инициализацию
- GetConfig всегда должен возвращать один и тот же экземпляр`,
			hint1: `Используйте sync.Once.Do() для инициализации экземпляра. Внутри функции Do создайте новый Config с инициализированной map через make(map[string]string).`,
			hint2: `Для SetValue используйте c.mu.Lock()/Unlock(), для GetValue используйте c.mu.RLock()/RUnlock(). RLock позволяет множественные конкурентные чтения.`,
			whyItMatters: `Паттерн Singleton необходим для управления общими ресурсами в конкурентных приложениях.

**Почему Singleton важен:**

**1. Управление ресурсами**
Подключения к БД, файловые дескрипторы и сетевые сокеты дороги в создании. Singleton обеспечивает повторное использование одного экземпляра:

\`\`\`go
// Без Singleton — создаёт новое подключение каждый раз (УТЕЧКА ПАМЯТИ!)
func GetDB() *sql.DB {
    db, _ := sql.Open("postgres", connStr)	// новое подключение при каждом вызове
    return db	// предыдущие подключения никогда не закрываются!
}

// С Singleton — переиспользует единый пул подключений
var dbOnce sync.Once
var db *sql.DB

func GetDB() *sql.DB {
    dbOnce.Do(func() {	// выполняется только один раз
        db, _ = sql.Open("postgres", connStr)	// создаём пул однократно
        db.SetMaxOpenConns(25)	// настраиваем размер пула
    })
    return db	// всегда возвращает тот же пул
}
\`\`\`

**2. Согласованность конфигурации**
Все части приложения видят одну и ту же конфигурацию:

\`\`\`go
// Сервис A устанавливает конфиг
config := GetConfig()
config.SetValue("api_url", "https://api.example.com")

// Сервис B (другая горутина) читает тот же конфиг
config := GetConfig()	// тот же экземпляр!
url := config.GetValue("api_url")	// получает "https://api.example.com"
\`\`\`

**3. Потокобезопасность с sync.Once**
sync.Once — идиоматичный способ реализации Singleton в Go:

\`\`\`go
// sync.Once гарантирует:
// 1. Инициализация выполняется ровно один раз
// 2. Все горутины ждут завершения инициализации
// 3. Нет гонок данных — внутри используются атомарные операции

once.Do(func() {
    // Этот код выполнится только один раз, даже при 1000 конкурентных вызовов
    instance = &Config{values: make(map[string]string)}
})
\`\`\`

**Реальные примеры в стандартной библиотеке Go:**
- \`http.DefaultClient\` — HTTP клиент по умолчанию в Go
- \`http.DefaultServeMux\` — Мультиплексор HTTP запросов по умолчанию
- \`log.Logger\` — Стандартный экземпляр логгера
- \`database/sql.DB\` — Пул соединений (должен быть singleton для каждой БД)

**Продакшен паттерн:**
\`\`\`go
// Production-ready singleton с graceful shutdown
type AppConfig struct {
    once     sync.Once
    mu       sync.RWMutex
    settings map[string]interface{}
    shutdown chan struct{}
}

var appConfig *AppConfig

func GetAppConfig() *AppConfig {
    if appConfig == nil {
        appConfig = &AppConfig{
            shutdown: make(chan struct{}),
        }
    }
    appConfig.once.Do(func() {
        appConfig.settings = loadFromFile()	// загружаем конфиг однократно
        go appConfig.watchForChanges()	// горячая перезагрузка в фоне
    })
    return appConfig
}
\`\`\`

**Распространённые ошибки:**
- Использование mutex вместо sync.Once (менее эффективно, больше кода)
- Забывают о потокобезопасности методов экземпляра
- Создание нескольких экземпляров в тестах (используйте build tags или интерфейсы)
- Не обрабатывают ошибки инициализации (используйте sync.Once с error channel)`,
			solutionCode: `package patterns

import (
	"sync"
)

type Config struct {
	mu     sync.RWMutex	// защищает конкурентный доступ к map values
	values map[string]string	// хранит пары ключ-значение конфигурации
}

var (
	instance *Config	// единственный экземпляр Config
	once     sync.Once	// гарантирует однократную инициализацию
)

func GetConfig() *Config {
	once.Do(func() {	// выполняется только при первом вызове, остальные горутины ждут
		instance = &Config{	// создаём единственный экземпляр
			values: make(map[string]string),	// инициализируем map чтобы избежать panic на nil map
		}
	})
	return instance	// всегда возвращаем один и тот же экземпляр
}

func (c *Config) SetValue(key, value string) {
	c.mu.Lock()	// захватываем эксклюзивную блокировку на запись, блокирует всех читателей и писателей
	defer c.mu.Unlock()	// освобождаем блокировку при выходе из функции, даже при panic
	c.values[key] = value	// безопасно записываем, у нас эксклюзивный доступ
}

func (c *Config) GetValue(key string) string {
	c.mu.RLock()	// захватываем разделяемую блокировку на чтение, позволяет множественным читателям
	defer c.mu.RUnlock()	// освобождаем блокировку чтения при выходе из функции
	return c.values[key]	// возвращает пустую строку если ключ не существует (поведение Go map)
}`
		},
		uz: {
			title: 'Singleton (Yagona) Pattern',
			description: `Go tilida Singleton patternini amalga oshiring — klassning faqat bitta nusxasi bo'lishini ta'minlang va unga global kirish nuqtasini taqdim eting.

**Siz amalga oshirasiz:**

1. **Config struct** — Ilova konfiguratsiyasi singletoni
2. **GetConfig() *Config** — sync.Once orqali thread-safe singleton kirish
3. **SetValue(key, value string)** — Konfiguratsiya qiymatini o'rnatish
4. **GetValue(key string) string** — Konfiguratsiya qiymatini olish

**Asosiy tushunchalar:**
- **sync.Once**: Initsializatsiya faqat bir marta bo'lishini kafolatlaydi
- **Thread xavfsizligi**: Parallel kirish uchun xavfsiz
- **Lazy initsializatsiya**: Nusxa faqat birinchi so'rov bo'lganda yaratiladi

**Foydalanish misoli:**

\`\`\`go
// Birinchi chaqiruv nusxani yaratadi
config1 := GetConfig()
config1.SetValue("database", "postgres://localhost:5432")

// Ikkinchi chaqiruv o'sha nusxani qaytaradi
config2 := GetConfig()
fmt.Println(config2.GetValue("database")) // postgres://localhost:5432

// Bir xil nusxa ekanligini tekshirish
fmt.Println(config1 == config2) // true
\`\`\`

**Qachon Singleton ishlatiladi:**
- Ma'lumotlar bazasi ulanish pullari
- Konfiguratsiya menejerlari
- Logging xizmatlari
- Thread pullari
- Keshlar

**Cheklovlar:**
- sync.Once ishlatib thread-safe bo'lishi shart
- Lazy initsializatsiya ishlatishi shart
- GetConfig har doim bir xil nusxani qaytarishi shart`,
			hint1: `sync.Once.Do() dan nusxani initsializatsiya qilish uchun foydalaning. Do funksiyasi ichida yangi Config yarating va map ni make(map[string]string) bilan initsializatsiya qiling.`,
			hint2: `SetValue uchun c.mu.Lock()/Unlock(), GetValue uchun c.mu.RLock()/RUnlock() ishlating. RLock bir vaqtda bir nechta o'qishga ruxsat beradi.`,
			whyItMatters: `Singleton patterni parallel ilovalarda umumiy resurslarni boshqarish uchun muhimdir.

**Singleton nima uchun muhim:**

**1. Resurslarni boshqarish**
Ma'lumotlar bazasi ulanishlari, fayl deskriptorlari va tarmoq soketlari yaratish qimmat. Singleton bitta nusxani qayta ishlatishni ta'minlaydi:

\`\`\`go
// Singleton siz — har safar yangi ulanish yaratadi (XOTIRA OQISHI!)
func GetDB() *sql.DB {
    db, _ := sql.Open("postgres", connStr)	// har chaqiruvda yangi ulanish
    return db	// oldingi ulanishlar hech qachon yopilmaydi!
}

// Singleton bilan — yagona ulanish pulini qayta ishlatadi
var dbOnce sync.Once
var db *sql.DB

func GetDB() *sql.DB {
    dbOnce.Do(func() {	// faqat bir marta bajariladi
        db, _ = sql.Open("postgres", connStr)	// pulni bir marta yaratamiz
        db.SetMaxOpenConns(25)	// pul hajmini sozlaymiz
    })
    return db	// har doim bir xil pulni qaytaradi
}
\`\`\`

**2. Konfiguratsiya muvofiqligi**
Ilovaning barcha qismlari bir xil konfiguratsiyani ko'radi:

\`\`\`go
// A servisi konfigni o'rnatadi
config := GetConfig()
config.SetValue("api_url", "https://api.example.com")

// B servisi (boshqa goroutine) bir xil konfigni o'qiydi
config := GetConfig()	// bir xil nusxa!
url := config.GetValue("api_url")	// "https://api.example.com" oladi
\`\`\`

**3. sync.Once bilan thread xavfsizligi**
sync.Once Go'da Singleton amalga oshirishning idiomatik usuli:

\`\`\`go
// sync.Once kafolatlaydi:
// 1. Initsializatsiya aynan bir marta bajariladi
// 2. Barcha goroutinelar initsializatsiya tugashini kutadi
// 3. Poyga sharoitlari yo'q — ichida atomik operatsiyalar ishlatiladi

once.Do(func() {
    // Bu kod faqat bir marta bajariladi, hatto 1000 parallel chaqiruvda ham
    instance = &Config{values: make(map[string]string)}
})
\`\`\`

**Go standart kutubxonasidagi real misollar:**
- \`http.DefaultClient\` — Go ning standart HTTP klienti singletoni
- \`http.DefaultServeMux\` — Standart HTTP so'rov multipleksori
- \`log.Logger\` — Standart logger nusxasi
- \`database/sql.DB\` — Ulanish puli (har bir baza uchun singleton bo'lishi kerak)

**Ishlab chiqarish patterni:**
\`\`\`go
// Graceful shutdown bilan production-ready singleton
type AppConfig struct {
    once     sync.Once
    mu       sync.RWMutex
    settings map[string]interface{}
    shutdown chan struct{}
}

var appConfig *AppConfig

func GetAppConfig() *AppConfig {
    if appConfig == nil {
        appConfig = &AppConfig{
            shutdown: make(chan struct{}),
        }
    }
    appConfig.once.Do(func() {
        appConfig.settings = loadFromFile()	// konfigni bir marta yuklaymiz
        go appConfig.watchForChanges()	// fonda issiq qayta yuklash
    })
    return appConfig
}
\`\`\`

**Oldini olish kerak bo'lgan xatolar:**
- sync.Once o'rniga mutex ishlatish (kamroq samarali, ko'proq kod)
- Nusxa metodlarining thread xavfsizligini unutish
- Testlarda bir nechta nusxa yaratish (build tags yoki interfeyslar ishlating)
- Initsializatsiya xatolarini qayta ishlamaslik (sync.Once bilan error channel ishlating)`,
			solutionCode: `package patterns

import (
	"sync"
)

type Config struct {
	mu     sync.RWMutex	// values map ga parallel kirishni himoya qiladi
	values map[string]string	// konfiguratsiya kalit-qiymat juftlarini saqlaydi
}

var (
	instance *Config	// Config ning yagona nusxasi
	once     sync.Once	// initsializatsiya faqat bir marta bo'lishini kafolatlaydi
)

func GetConfig() *Config {
	once.Do(func() {	// faqat birinchi chaqiruvda bajariladi, boshqa goroutinelar kutadi
		instance = &Config{	// yagona nusxani yaratamiz
			values: make(map[string]string),	// nil map panic dan qochish uchun map ni initsializatsiya qilamiz
		}
	})
	return instance	// har doim bir xil nusxani qaytaramiz
}

func (c *Config) SetValue(key, value string) {
	c.mu.Lock()	// eksklyuziv yozish qulfini olamiz, barcha o'quvchi va yozuvchilarni bloklaydi
	defer c.mu.Unlock()	// funksiyadan chiqishda qulfni bo'shatamiz, panic da ham
	c.values[key] = value	// xavfsiz yozamiz, bizda eksklyuziv kirish bor
}

func (c *Config) GetValue(key string) string {
	c.mu.RLock()	// umumiy o'qish qulfini olamiz, bir nechta parallel o'quvchilarga ruxsat beradi
	defer c.mu.RUnlock()	// funksiyadan chiqishda o'qish qulfini bo'shatamiz
	return c.values[key]	// kalit mavjud bo'lmasa bo'sh string qaytaradi (Go map xatti-harakati)
}`
		}
	}
};

export default task;
