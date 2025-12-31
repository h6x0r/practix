// GoF Design Patterns for Go
// Organized by pattern type: Creational, Structural, Behavioral

export const GO_PATTERNS_MODULES = [
	// ==================== CREATIONAL PATTERNS ====================
	{
		title: 'Creational Patterns',
		description: 'Patterns for object creation mechanisms, increasing flexibility and reuse.',
		section: 'creational',
		order: 1,
		topics: [
			{
				title: 'Singleton Pattern',
				description: 'Ensure a class has only one instance with global access point.',
				difficulty: 'easy',
				estimatedTime: '45m',
				order: 1,
				tasks: [
					{
						slug: 'go-pattern-singleton-config',
						title: 'Singleton: Global Config',
						difficulty: 'easy',
						tags: ['go', 'patterns', 'singleton', 'concurrency'],
						estimatedTime: '15m',
						isPremium: false,
						isImportant: true,
						youtubeUrl: '',
						description: `Implement a **thread-safe Singleton** for application configuration.

**Requirements:**
1. Create a \`GlobalConfig()\` function that returns the same instance every time
2. Use lazy initialization - create instance only when first accessed
3. Ensure thread-safety using \`sync.Once\`
4. Initialize internal map for storing key-value pairs

**Example:**
\`\`\`go
config1 := GlobalConfig()
config2 := GlobalConfig()
// config1 == config2 (same instance)

config1.Set("api_key", "secret123")
val, ok := config2.Get("api_key")
// val == "secret123", ok == true
\`\`\`

**Constraints:**
- Must be safe for concurrent access
- Only one instance should ever exist`,
						initialCode: `package creational

import "sync"

// ConfigSingleton provides thread-safe access to configuration.
type ConfigSingleton struct {
	mu     sync.RWMutex
	values map[string]string
}

var (
	configOnce     sync.Once
	configInstance *ConfigSingleton
)

// GlobalConfig returns the singleton configuration instance.
func GlobalConfig() *ConfigSingleton {
	// TODO: Implement lazy initialization using sync.Once
	// TODO: Implement
}`,
						solutionCode: `package creational

import "sync"

type ConfigSingleton struct {
	mu     sync.RWMutex      // Protects values map
	values map[string]string // Stores key-value pairs
}

var (
	configOnce     sync.Once        // Ensures single initialization
	configInstance *ConfigSingleton // Holds the singleton
)

func GlobalConfig() *ConfigSingleton {
	configOnce.Do(func() {                               // Guarantee one-time init
		configInstance = &ConfigSingleton{           // Create instance
			values: make(map[string]string),     // Initialize map
		}
	})
	return configInstance                                // Return ready singleton
}`,
						hint1: 'Use sync.Once to guarantee the initialization code runs exactly once, even with concurrent calls.',
						hint2: 'Initialize the values map inside the Once.Do() function to prevent nil map panics.',
						whyItMatters: `Singleton is one of the most practical GoF patterns for shared resources. In production, you'll use it for:

**Common Use Cases:**
- Application configuration (database URLs, API keys, feature flags)
- Connection pools (database, Redis, HTTP clients)
- Logger instances with shared file handles
- Cache managers that coordinate memory limits

**Why Thread-Safety Matters:**
Without sync.Once, multiple goroutines could create multiple instances during initialization, defeating the singleton purpose. Go's sync.Once uses atomic operations to ensure exactly one execution, even under heavy concurrent load.

**Real-World Examples:**
- Database connection pool: One pool shared across all HTTP handlers
- Logger: One instance writing to log file without corruption
- Metrics collector: Central instance aggregating stats from all services

**Important Note:** While Singleton is useful, avoid overuse. Global state can make testing harder and hide dependencies. Use dependency injection when possible, reserve Singleton for truly global resources.`,
						translations: {
							ru: {
								title: 'Singleton: Глобальная Конфигурация',
								description: `Реализуйте **потокобезопасный Singleton** для конфигурации приложения.

**Требования:**
1. Создайте функцию \`GlobalConfig()\`, возвращающую один и тот же экземпляр
2. Используйте ленивую инициализацию - создавайте экземпляр только при первом обращении
3. Обеспечьте потокобезопасность через \`sync.Once\`
4. Инициализируйте внутреннюю map для хранения пар ключ-значение

**Пример:**
\`\`\`go
config1 := GlobalConfig()
config2 := GlobalConfig()
// config1 == config2 (один экземпляр)

config1.Set("api_key", "secret123")
val, ok := config2.Get("api_key")
// val == "secret123", ok == true
\`\`\`

**Ограничения:**
- Должен быть безопасен для конкурентного доступа
- Должен существовать только один экземпляр`,
								hint1: 'Используйте sync.Once для гарантии однократного выполнения инициализации.',
								hint2: 'Инициализируйте map values внутри Once.Do() для предотвращения nil map panic.',
								whyItMatters: `Singleton - один из самых практичных GoF паттернов для общих ресурсов. В production вы будете использовать его для:

**Типичные случаи:**
- Конфигурация приложения (URL БД, API ключи, feature flags)
- Connection pools (БД, Redis, HTTP клиенты)
- Экземпляры логгера с общими file handles
- Cache менеджеры, координирующие лимиты памяти

**Почему важна потокобезопасность:**
Без sync.Once несколько горутин могут создать несколько экземпляров, нарушая суть паттерна. sync.Once использует атомарные операции для гарантии единственного выполнения.

**Real-world примеры:**
- Database connection pool: один пул для всех HTTP handlers
- Logger: один экземпляр, пишущий в файл без коррупции
- Metrics collector: центральный экземпляр, агрегирующий статистику

**Важно:** Избегайте злоупотребления Singleton. Глобальное состояние усложняет тестирование. Используйте dependency injection где возможно.`
							},
							uz: {
								title: 'Singleton: Global konfiguratsiya',
								description: `Ilova konfiguratsiyasi uchun **thread-safe Singleton** ni amalga oshiring.

**Talablar:**
1. \`GlobalConfig()\` funksiyasini yarating, har safar bir xil instance ni qaytaradi
2. Lazy initialization dan foydalaning - instance faqat birinchi marta murojaat qilinganda yaratiladi
3. \`sync.Once\` orqali thread-safety ni ta'minlang
4. Kalit-qiymat juftlarini saqlash uchun ichki map ni ishga tushiring

**Misol:**
\`\`\`go
config1 := GlobalConfig()
config2 := GlobalConfig()
// config1 == config2 (bir xil instance)

config1.Set("api_key", "secret123")
val, ok := config2.Get("api_key")
// val == "secret123", ok == true
\`\`\`

**Cheklovlar:**
- Concurrent kirish uchun xavfsiz bo'lishi kerak
- Faqat bitta instance mavjud bo'lishi kerak`,
								hint1: 'sync.Once dan foydalanib, initialization kodi faqat bir marta bajarilishini kafolatlang.',
								hint2: 'values map ni Once.Do() ichida ishga tushiring, nil map panic laridan qochish uchun.',
								whyItMatters: `Singleton - umumiy resurslar uchun eng amaliy GoF patternlaridan biri. Production da quyidagilar uchun ishlatiladi:

**Umumiy foydalanish holatlari:**
- Ilova konfiguratsiyasi (DB URL, API kalitlar, feature flag lar)
- Connection pool lar (database, Redis, HTTP clientlar)
- Logger instance lari shared file handle lar bilan
- Xotira limitlarini koordinatsiya qiluvchi cache manager lar

**Thread-safety nima uchun muhim:**
sync.Once siz bir nechta goroutine lar initialization vaqtida bir nechta instance yaratishi mumkin. sync.Once atomic operatsiyalardan foydalanib, faqat bir marta bajarilishini kafolatlaydi.

**Real-world misollar:**
- Database connection pool: barcha HTTP handler lar uchun bitta pool
- Logger: log faylga corruption siz yozuvchi bitta instance
- Metrics collector: barcha servislardan statistikani yig'uvchi markaziy instance

**Muhim eslatma:** Singleton dan ortiqcha foydalanmang. Global state testni qiyinlashtiradi. Iloji bo'lsa dependency injection dan foydalaning.`
							}
						}
					},
					{
						slug: 'go-pattern-singleton-set',
						title: 'Singleton: Set Method',
						difficulty: 'easy',
						tags: ['go', 'patterns', 'singleton'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement the **Set** method for the ConfigSingleton to store key-value pairs safely.

**Requirements:**
1. Accept a key and value as strings
2. Protect write operations with mutex
3. Handle nil receiver gracefully
4. Lazily initialize the values map if nil

**Example:**
\`\`\`go
config := GlobalConfig()
config.Set("db_host", "localhost")
config.Set("db_port", "5432")
config.Set("db_host", "prod.db.com") // Overwrites previous value
\`\`\`

**Constraints:**
- Must be thread-safe for concurrent writes
- Should not panic on nil receiver`,
						initialCode: `package creational

// Set stores a key-value pair in the singleton configuration.
func (c *ConfigSingleton) Set(key, value string) {
	// TODO: Implement thread-safe Set with lazy map initialization
	panic("TODO")
}`,
						solutionCode: `package creational

func (c *ConfigSingleton) Set(key, value string) {
	if c == nil {                                        // Ignore nil receiver
		return                                       // Nothing to write
	}
	c.mu.Lock()                                          // Protect write with mutex
	defer c.mu.Unlock()                                  // Release mutex on exit
	if c.values == nil {                                 // Lazily initialize map
		c.values = make(map[string]string)           // Create map for values
	}
	c.values[key] = value                                // Store key-value pair
}`,
						hint1: 'Use Lock() before writing and defer Unlock() to ensure the mutex is always released.',
						hint2: 'Check if values map is nil and initialize it before writing to prevent nil map panic.',
						whyItMatters: `Thread-safe write operations are critical for singletons accessed by multiple goroutines.

**Key Concepts:**
- **Mutex Protection:** Without mutex, concurrent writes can cause race conditions and data corruption
- **Lazy Initialization:** The map is created on first use, reducing memory if configuration is read-only
- **Nil Receiver Pattern:** Gracefully handling nil prevents panics in edge cases

**Production Implications:**
In a web application with 100 concurrent requests, multiple handlers might Set() configuration values. Without proper locking, you'd see:
- Lost writes (write A then B, but A's value remains)
- Corrupted map internals leading to crashes
- Race detector warnings in tests

**Pattern Usage:**
This defensive coding (nil check, lazy init) is common in Go libraries. Examples:
- database/sql: Connection pool configuration
- net/http: Default client settings
- Standard library sync types`,
						translations: {
							ru: {
								title: 'Singleton: Метод Set',
								description: `Реализуйте метод **Set** для ConfigSingleton для безопасного хранения пар ключ-значение.

**Требования:**
1. Принимайте ключ и значение как строки
2. Защищайте операции записи мьютексом
3. Корректно обрабатывайте nil receiver
4. Лениво инициализируйте map values, если она nil

**Пример:**
\`\`\`go
config := GlobalConfig()
config.Set("db_host", "localhost")
config.Set("db_port", "5432")
config.Set("db_host", "prod.db.com") // Перезаписывает предыдущее значение
\`\`\`

**Ограничения:**
- Должен быть потокобезопасен для конкурентной записи
- Не должен паниковать при nil receiver`,
								hint1: 'Используйте Lock() перед записью и defer Unlock() для гарантии освобождения мьютекса.',
								hint2: 'Проверьте, что values map не nil и инициализируйте её перед записью.',
								whyItMatters: `Потокобезопасные операции записи критичны для singleton'ов, к которым обращаются несколько горутин.

**Ключевые концепции:**
- **Защита мьютексом:** Без мьютекса конкурентная запись вызывает race conditions и коррупцию данных
- **Lazy Initialization:** Map создаётся при первом использовании, экономя память
- **Nil Receiver Pattern:** Корректная обработка nil предотвращает панику

**Production последствия:**
В веб-приложении со 100 конкурентными запросами несколько handlers могут вызывать Set(). Без блокировки:
- Потерянные записи
- Коррупция внутренностей map, приводящая к крашам
- Предупреждения race detector в тестах`
							},
							uz: {
								title: 'Singleton: Set metodi',
								description: `ConfigSingleton uchun kalit-qiymat juftlarini xavfsiz saqlash uchun **Set** metodini amalga oshiring.

**Talablar:**
1. Kalit va qiymatni string sifatida qabul qiling
2. Yozish operatsiyalarini mutex bilan himoyalang
3. nil receiver ni to'g'ri ishlang
4. values map ni nil bo'lsa lazy initialization qiling

**Misol:**
\`\`\`go
config := GlobalConfig()
config.Set("db_host", "localhost")
config.Set("db_port", "5432")
config.Set("db_host", "prod.db.com") // Oldingi qiymatni qayta yozadi
\`\`\`

**Cheklovlar:**
- Concurrent yozish uchun thread-safe bo'lishi kerak
- nil receiver da panic bo'lmasligi kerak`,
								hint1: 'Yozishdan oldin Lock() dan foydalaning va defer Unlock() bilan mutex ni chiqarishni kafolatlang.',
								hint2: 'values map nil emasligini tekshiring va yozishdan oldin ishga tushiring.',
								whyItMatters: `Thread-safe yozish operatsiyalari bir nechta goroutine lar tomonidan ishlatiladigan singleton lar uchun muhim.

**Asosiy tushunchalar:**
- **Mutex himoyasi:** Mutex siz concurrent yozish race condition va ma'lumot korruptsiyasiga olib keladi
- **Lazy Initialization:** Map birinchi foydalanishda yaratiladi, xotirani tejaydi
- **Nil Receiver Pattern:** nil ni to'g'ri ishlash panikdan qochadi

**Production ta'siri:**
100 concurrent request bilan web ilovada bir nechta handler Set() ni chaqirishi mumkin. Blokirovka bo'lmasa:
- Yo'qolgan yozuvlar
- Map ichki korruptsiyasi, crashga olib keladi
- Testlarda race detector ogohlantirishlari`
							}
						}
					},
					{
						slug: 'go-pattern-singleton-get',
						title: 'Singleton: Get Method',
						difficulty: 'easy',
						tags: ['go', 'patterns', 'singleton'],
						estimatedTime: '15m',
						isPremium: false,
						youtubeUrl: '',
						description: `Implement the **Get** method for the ConfigSingleton to retrieve values safely.

**Requirements:**
1. Accept a key as string
2. Return the value and a boolean indicating presence
3. Use read-lock (RLock) for concurrent reads
4. Handle nil receiver and nil map gracefully

**Example:**
\`\`\`go
config := GlobalConfig()
config.Set("timeout", "30s")

val, ok := config.Get("timeout")
// val == "30s", ok == true

val, ok = config.Get("missing_key")
// val == "", ok == false
\`\`\`

**Constraints:**
- Must allow concurrent reads without blocking other readers
- Should not panic on nil receiver or uninitialized map`,
						initialCode: `package creational

// Get retrieves a value by key from the singleton configuration.
func (c *ConfigSingleton) Get(key string) (string, bool) {
	// TODO: Implement thread-safe Get with RLock
	panic("TODO")
}`,
						solutionCode: `package creational

func (c *ConfigSingleton) Get(key string) (string, bool) {
	if c == nil {                                        // Nil receiver check
		return "", false                             // No data available
	}
	c.mu.RLock()                                         // Use read-lock for safe reading
	defer c.mu.RUnlock()                                 // Release read-lock after read
	if c.values == nil {                                 // Map not yet initialized
		return "", false                             // No values present
	}
	val, ok := c.values[key]                             // Read value from map
	return val, ok                                       // Return value and presence flag
}`,
						hint1: 'Use RLock() instead of Lock() to allow multiple concurrent readers without blocking each other.',
						hint2: 'Return empty string and false when the key is not found, following Go\'s comma-ok idiom.',
						whyItMatters: `Read-write lock optimization is crucial for high-performance concurrent systems.

**RLock vs Lock:**
- **Lock():** Exclusive access, blocks all readers and writers
- **RLock():** Shared access, allows multiple concurrent readers but blocks writers
- **Performance Impact:** In read-heavy workloads (90% reads, 10% writes), RLock can improve throughput 10-100x

**Real-World Scenario:**
Configuration is typically read thousands of times per second but written rarely (on reload/update). Using RLock means:
- All HTTP handlers can read config simultaneously without waiting
- Only Set() operations block briefly for write
- No contention in the hot path

**The Comma-Ok Idiom:**
\`val, ok := map[key]\` is Go's standard pattern for distinguishing:
- Key exists with empty value: \`val == "", ok == true\`
- Key doesn't exist: \`val == "", ok == false\`

This is used throughout the standard library (type assertions, channel receives, map lookups).

**Production Pattern:**
Most config singletons use sync.RWMutex exactly this way. Examples from popular libraries:
- viper: Configuration framework
- godotenv: Environment variable loader
- Various feature flag libraries`,
						translations: {
							ru: {
								title: 'Singleton: Метод Get',
								description: `Реализуйте метод **Get** для ConfigSingleton для безопасного получения значений.

**Требования:**
1. Принимайте ключ как строку
2. Возвращайте значение и boolean, указывающий наличие ключа
3. Используйте read-lock (RLock) для конкурентного чтения
4. Корректно обрабатывайте nil receiver и nil map

**Пример:**
\`\`\`go
config := GlobalConfig()
config.Set("timeout", "30s")

val, ok := config.Get("timeout")
// val == "30s", ok == true

val, ok = config.Get("missing_key")
// val == "", ok == false
\`\`\`

**Ограничения:**
- Должен разрешать конкурентное чтение без блокировки других читателей
- Не должен паниковать при nil receiver или неинициализированной map`,
								hint1: 'Используйте RLock() вместо Lock() для разрешения нескольких конкурентных читателей.',
								hint2: 'Возвращайте пустую строку и false, когда ключ не найден, следуя Go comma-ok идиоме.',
								whyItMatters: `Оптимизация read-write lock критична для высокопроизводительных конкурентных систем.

**RLock vs Lock:**
- **Lock():** Эксклюзивный доступ, блокирует всех читателей и писателей
- **RLock():** Общий доступ, разрешает множественных конкурентных читателей
- **Влияние на производительность:** При read-heavy нагрузке RLock улучшает throughput в 10-100 раз

**Real-world сценарий:**
Конфигурация читается тысячи раз в секунду, но пишется редко. RLock означает:
- Все HTTP handlers могут читать config одновременно
- Только Set() блокирует на запись
- Нет contention в hot path

**Comma-Ok идиома:**
\`val, ok := map[key]\` - стандартный Go паттерн для различения:
- Ключ существует с пустым значением: \`val == "", ok == true\`
- Ключ не существует: \`val == "", ok == false\`

Используется во всей стандартной библиотеке.`
							},
							uz: {
								title: 'Singleton: Get metodi',
								description: `ConfigSingleton uchun qiymatlarni xavfsiz olish uchun **Get** metodini amalga oshiring.

**Talablar:**
1. Kalitni string sifatida qabul qiling
2. Qiymat va mavjudligini ko'rsatuvchi boolean qaytaring
3. Concurrent o'qish uchun read-lock (RLock) dan foydalaning
4. nil receiver va nil map ni to'g'ri ishlang

**Misol:**
\`\`\`go
config := GlobalConfig()
config.Set("timeout", "30s")

val, ok := config.Get("timeout")
// val == "30s", ok == true

val, ok = config.Get("missing_key")
// val == "", ok == false
\`\`\`

**Cheklovlar:**
- Boshqa o'quvchilarni bloklamasdan concurrent o'qishga ruxsat berishi kerak
- nil receiver yoki ishga tushmagan map da panic bo'lmasligi kerak`,
								hint1: 'Lock() o\'rniga RLock() dan foydalaning, bir nechta concurrent o\'quvchilarga ruxsat berish uchun.',
								hint2: 'Kalit topilmaganda bo\'sh string va false qaytaring, Go comma-ok idiomaga amal qiling.',
								whyItMatters: `Read-write lock optimizatsiyasi yuqori unumdor concurrent sistemlar uchun muhim.

**RLock vs Lock:**
- **Lock():** Eksklyuziv kirish, barcha o'quvchi va yozuvchilarni bloklaydi
- **RLock():** Umumiy kirish, bir nechta concurrent o'quvchilarga ruxsat beradi
- **Performance ta'siri:** Read-heavy yuklamada RLock throughput ni 10-100 marta yaxshilaydi

**Real-world stsenariy:**
Konfiguratsiya sekundiga minglab marta o'qiladi, lekin kamdan-kam yoziladi. RLock:
- Barcha HTTP handler lar config ni bir vaqtda o'qishi mumkin
- Faqat Set() yozish uchun bloklanadi
- Hot path da contention yo'q

**Comma-Ok idioma:**
\`val, ok := map[key]\` - Go standart pattern farqlash uchun:
- Kalit bo'sh qiymat bilan mavjud: \`val == "", ok == true\`
- Kalit mavjud emas: \`val == "", ok == false\`

Butun standart kutubxonada ishlatiladi.`
							}
						}
					}
				]
			}
		]
	}
];
