import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-factory-method',
	title: 'Factory Method Pattern',
	difficulty: 'medium',
	tags: ['go', 'patterns', 'constructors', 'factory'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Factory Method Pattern** to create different types of database connections based on a driver string.

**Pattern Overview:**
The Factory Method Pattern provides an interface for creating objects, but allows subclasses or implementations to decide which concrete type to instantiate. In Go, this is typically implemented as a function that returns an interface type, hiding the concrete implementation details.

**Key Components:**

1. **Database Interface** - Defines common operations:
   \`\`\`go
   type Database interface {
       Connect() error
       Query(sql string) (string, error)
       Close() error
   }
   \`\`\`

2. **PostgresDB** - Concrete implementation for PostgreSQL:
   2.1. Private struct with DSN field
   2.2. Implements all Database interface methods
   2.3. Connect() validates DSN contains "postgres://" and sets connected flag
   2.4. Query() checks connection and returns formatted result
   2.5. Close() marks as disconnected

3. **MySQLDB** - Concrete implementation for MySQL:
   3.1. Private struct with DSN field
   3.2. Implements all Database interface methods
   3.3. Connect() validates DSN contains "mysql://" and sets connected flag
   3.4. Query() checks connection and returns formatted result
   3.5. Close() marks as disconnected

4. **NewDatabase Factory** - Creates appropriate database type:
   \`\`\`go
   func NewDatabase(driver, dsn string) (Database, error)
   \`\`\`
   4.1. Accepts "postgres" or "mysql" as driver
   4.2. Returns appropriate concrete type as Database interface
   4.3. Returns error for unsupported drivers
   4.4. Validates DSN is not empty

**Implementation Requirements:**

**PostgresDB struct:**
- Private fields: dsn (string), connected (bool)
- Connect(): Validate DSN contains "postgres://", return error if not, set connected=true
- Query(sql): Return error if not connected, otherwise return "PostgreSQL result: " + sql
- Close(): Set connected=false, return nil

**MySQLDB struct:**
- Private fields: dsn (string), connected (bool)
- Connect(): Validate DSN contains "mysql://", return error if not, set connected=true
- Query(sql): Return error if not connected, otherwise return "MySQL result: " + sql
- Close(): Set connected=false, return nil

**NewDatabase function:**
- Return error if dsn is empty
- Switch on driver string (case-insensitive comparison recommended)
- For "postgres": return &postgresDB{dsn: dsn}
- For "mysql": return &mysqlDB{dsn: dsn}
- For unknown driver: return error "unsupported driver: <driver>"

**Usage Example:**
\`\`\`go
// Create PostgreSQL database
db, err := NewDatabase("postgres", "postgres://localhost:5432/mydb")
if err != nil {
    log.Fatal(err)
}

err = db.Connect()
if err != nil {
    log.Fatal(err)
}

result, err := db.Query("SELECT * FROM users")
fmt.Println(result) // "PostgreSQL result: SELECT * FROM users"

db.Close()

// Create MySQL database
db, err = NewDatabase("mysql", "mysql://localhost:3306/mydb")
err = db.Connect()
result, err = db.Query("SELECT * FROM products")
fmt.Println(result) // "MySQL result: SELECT * FROM products"
\`\`\`

**Why This Pattern:**
- Decouples client code from concrete implementations
- Easy to add new database types without modifying existing code
- Client code works with interface, not concrete types
- Centralized object creation logic`,
	initialCode: `package structinit

import (
	"fmt"
	"strings"
)

// TODO: Define Database interface with Connect, Query, and Close methods
type Database interface {
	// TODO: Implement
}

// TODO: Define postgresDB struct (private) with dsn and connected fields
type postgresDB struct {
	// TODO: Add fields
}

// TODO: Implement Connect method for postgresDB
// Validate DSN contains "postgres://", set connected=true
func (p *postgresDB) Connect() error {
	return nil // TODO: Implement
}

// TODO: Implement Query method for postgresDB
// Return error if not connected, otherwise return formatted result
func (p *postgresDB) Query(sql string) (string, error) {
	// TODO: Implement
}

// TODO: Implement Close method for postgresDB
func (p *postgresDB) Close() error {
	return nil // TODO: Implement
}

// TODO: Define mysqlDB struct (private) with dsn and connected fields
type mysqlDB struct {
	// TODO: Add fields
}

// TODO: Implement Connect method for mysqlDB
// Validate DSN contains "mysql://", set connected=true
func (m *mysqlDB) Connect() error {
	return nil // TODO: Implement
}

// TODO: Implement Query method for mysqlDB
// Return error if not connected, otherwise return formatted result
func (m *mysqlDB) Query(sql string) (string, error) {
	// TODO: Implement
}

// TODO: Implement Close method for mysqlDB
func (m *mysqlDB) Close() error {
	return nil // TODO: Implement
}

// TODO: Implement NewDatabase factory function
// Return appropriate Database implementation based on driver
// Validate inputs and return errors for invalid cases
func NewDatabase(driver, dsn string) (Database, error) {
	var zero Database
	return zero, nil // TODO: Implement
}`,
	solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Database interface {
	Connect() error                              // Establish connection to database
	Query(sql string) (string, error)           // Execute query and return result
	Close() error                                // Close database connection
}

type postgresDB struct {
	dsn       string                             // Data source name for PostgreSQL
	connected bool                               // Connection state flag
}

func (p *postgresDB) Connect() error {
	if !strings.Contains(p.dsn, "postgres://") { // Validate PostgreSQL DSN format
		return fmt.Errorf("invalid postgres DSN") // Return error for invalid DSN
	}
	p.connected = true                           // Mark as connected
	return nil                                   // Return nil on success
}

func (p *postgresDB) Query(sql string) (string, error) {
	if !p.connected {                            // Check connection state
		return "", fmt.Errorf("not connected")   // Return error if not connected
	}
	return "PostgreSQL result: " + sql, nil      // Return formatted result
}

func (p *postgresDB) Close() error {
	p.connected = false                          // Mark as disconnected
	return nil                                   // Return nil (no error)
}

type mysqlDB struct {
	dsn       string                             // Data source name for MySQL
	connected bool                               // Connection state flag
}

func (m *mysqlDB) Connect() error {
	if !strings.Contains(m.dsn, "mysql://") {    // Validate MySQL DSN format
		return fmt.Errorf("invalid mysql DSN")   // Return error for invalid DSN
	}
	m.connected = true                           // Mark as connected
	return nil                                   // Return nil on success
}

func (m *mysqlDB) Query(sql string) (string, error) {
	if !m.connected {                            // Check connection state
		return "", fmt.Errorf("not connected")   // Return error if not connected
	}
	return "MySQL result: " + sql, nil           // Return formatted result
}

func (m *mysqlDB) Close() error {
	m.connected = false                          // Mark as disconnected
	return nil                                   // Return nil (no error)
}

func NewDatabase(driver, dsn string) (Database, error) {
	if strings.TrimSpace(dsn) == "" {            // Validate DSN is not empty
		return nil, fmt.Errorf("dsn cannot be empty") // Return error for empty DSN
	}

	switch strings.ToLower(driver) {             // Switch on lowercase driver name
	case "postgres":
		return &postgresDB{dsn: dsn}, nil        // Return PostgreSQL implementation
	case "mysql":
		return &mysqlDB{dsn: dsn}, nil           // Return MySQL implementation
	default:
		return nil, fmt.Errorf("unsupported driver: %s", driver) // Return error for unknown driver
	}
}`,
	testCode: `package structinit

import (
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Create postgres database
	db, err := NewDatabase("postgres", "postgres://localhost/test")
	if err != nil || db == nil {
		t.Errorf("expected postgres db, got err=%v", err)
	}
}

func Test2(t *testing.T) {
	// Create mysql database
	db, err := NewDatabase("mysql", "mysql://localhost/test")
	if err != nil || db == nil {
		t.Errorf("expected mysql db, got err=%v", err)
	}
}

func Test3(t *testing.T) {
	// Connect and query postgres
	db, _ := NewDatabase("postgres", "postgres://localhost/test")
	err := db.Connect()
	if err != nil {
		t.Errorf("postgres connect failed: %v", err)
	}
	result, err := db.Query("SELECT 1")
	if err != nil || !strings.Contains(result, "PostgreSQL") {
		t.Errorf("expected PostgreSQL result, got %s, err=%v", result, err)
	}
}

func Test4(t *testing.T) {
	// Connect and query mysql
	db, _ := NewDatabase("mysql", "mysql://localhost/test")
	err := db.Connect()
	if err != nil {
		t.Errorf("mysql connect failed: %v", err)
	}
	result, err := db.Query("SELECT 1")
	if err != nil || !strings.Contains(result, "MySQL") {
		t.Errorf("expected MySQL result, got %s, err=%v", result, err)
	}
}

func Test5(t *testing.T) {
	// Invalid postgres DSN
	db, _ := NewDatabase("postgres", "invalid")
	err := db.Connect()
	if err == nil {
		t.Error("expected error for invalid postgres DSN")
	}
}

func Test6(t *testing.T) {
	// Invalid mysql DSN
	db, _ := NewDatabase("mysql", "invalid")
	err := db.Connect()
	if err == nil {
		t.Error("expected error for invalid mysql DSN")
	}
}

func Test7(t *testing.T) {
	// Unsupported driver
	_, err := NewDatabase("sqlite", "test.db")
	if err == nil {
		t.Error("expected error for unsupported driver")
	}
}

func Test8(t *testing.T) {
	// Empty DSN
	_, err := NewDatabase("postgres", "")
	if err == nil {
		t.Error("expected error for empty DSN")
	}
}

func Test9(t *testing.T) {
	// Query without connect
	db, _ := NewDatabase("postgres", "postgres://localhost/test")
	_, err := db.Query("SELECT 1")
	if err == nil {
		t.Error("expected error for query without connect")
	}
}

func Test10(t *testing.T) {
	// Close connection
	db, _ := NewDatabase("postgres", "postgres://localhost/test")
	db.Connect()
	err := db.Close()
	if err != nil {
		t.Errorf("close failed: %v", err)
	}
	_, err = db.Query("SELECT 1")
	if err == nil {
		t.Error("expected error for query after close")
	}
}`,
	hint1: `Start by defining the Database interface with three methods. Then create two private structs (postgresDB and mysqlDB) that implement this interface. Each struct needs dsn and connected fields.`,
	hint2: `In NewDatabase, validate the dsn first, then use a switch statement on the driver. Return the appropriate concrete type (&postgresDB or &mysqlDB) as the Database interface. Remember to handle the default case for unsupported drivers.`,
	whyItMatters: `The Factory Method Pattern is essential for creating flexible, maintainable code that can easily accommodate new types without breaking existing functionality.

**Why Factory Method:**
- **Abstraction:** Client code depends on interfaces, not concrete types
- **Extensibility:** Add new database types without modifying client code
- **Encapsulation:** Hide concrete implementation details from clients
- **Testability:** Easy to mock Database interface for testing
- **Flexibility:** Switch implementations at runtime based on configuration

**Production Examples:**

\`\`\`go
// Example 1: Logger Factory
type Logger interface {
	Log(level, message string)
}

type fileLogger struct { file *os.File }
type consoleLogger struct {}
type cloudLogger struct { client *CloudClient }

func NewLogger(logType string, config map[string]string) (Logger, error) {
	switch logType {
	case "file":
		return &fileLogger{file: openFile(config["path"])}, nil
	case "console":
		return &consoleLogger{}, nil
	case "cloud":
		return &cloudLogger{client: newCloudClient(config)}, nil
	default:
		return nil, fmt.Errorf("unknown logger type: %s", logType)
	}
}
\`\`\`

\`\`\`go
// Example 2: Payment Processor Factory
type PaymentProcessor interface {
	ProcessPayment(amount float64) error
	Refund(transactionID string) error
}

type stripeProcessor struct { apiKey string }
type paypalProcessor struct { clientID, secret string }
type cryptoProcessor struct { walletAddress string }

func NewPaymentProcessor(provider string, creds Credentials) (PaymentProcessor, error) {
	switch provider {
	case "stripe":
		return &stripeProcessor{apiKey: creds.APIKey}, nil
	case "paypal":
		return &paypalProcessor{
			clientID: creds.ClientID,
			secret:   creds.Secret,
		}, nil
	case "crypto":
		return &cryptoProcessor{walletAddress: creds.Wallet}, nil
	default:
		return nil, fmt.Errorf("unsupported payment provider")
	}
}
\`\`\`

\`\`\`go
// Example 3: Cache Factory
type Cache interface {
	Get(key string) (interface{}, error)
	Set(key string, value interface{}) error
	Delete(key string) error
}

type memoryCache struct { data map[string]interface{} }
type redisCache struct { client *redis.Client }
type memcachedCache struct { client *memcache.Client }

func NewCache(cacheType, address string) (Cache, error) {
	switch cacheType {
	case "memory":
		return &memoryCache{data: make(map[string]interface{})}, nil
	case "redis":
		client := redis.NewClient(&redis.Options{Addr: address})
		return &redisCache{client: client}, nil
	case "memcached":
		client := memcache.New(address)
		return &memcachedCache{client: client}, nil
	default:
		return nil, fmt.Errorf("unknown cache type")
	}
}
\`\`\`

\`\`\`go
// Example 4: Message Queue Factory
type MessageQueue interface {
	Publish(topic string, msg []byte) error
	Subscribe(topic string, handler func([]byte)) error
}

type kafkaQueue struct { producer *kafka.Producer }
type rabbitmqQueue struct { conn *amqp.Connection }
type sqsQueue struct { client *sqs.SQS }

func NewMessageQueue(queueType string, config QueueConfig) (MessageQueue, error) {
	switch queueType {
	case "kafka":
		return &kafkaQueue{producer: newKafkaProducer(config)}, nil
	case "rabbitmq":
		return &rabbitmqQueue{conn: connectRabbitMQ(config)}, nil
	case "sqs":
		return &sqsQueue{client: newSQSClient(config)}, nil
	default:
		return nil, fmt.Errorf("unsupported queue type")
	}
}
\`\`\`

**Real-World Benefits:**
- **Kubernetes:** Uses factory pattern for creating different resource types
- **Docker:** Factory pattern for creating container runtimes (containerd, runc)
- **Database drivers:** database/sql package uses factory pattern for drivers
- **Cloud SDKs:** AWS, GCP, Azure all use factories for service clients

**Key Design Principles:**
- Return interface types, not concrete types
- Factory function name typically starts with "New"
- Validate inputs before creating objects
- Use clear error messages for unsupported types
- Consider using constants for type strings to avoid typos

Without factory methods, you'd need to expose all concrete types to clients, making the codebase harder to maintain and extend. Factory methods provide a clean abstraction layer that scales well.`,
	order: 1,
	translations: {
		ru: {
			title: 'Паттерн фабричный метод',
			description: `Реализуйте **Паттерн Фабричный Метод** для создания различных типов подключений к базе данных на основе строки драйвера.

**Обзор паттерна:**
Паттерн Фабричный Метод предоставляет интерфейс для создания объектов, но позволяет подклассам или реализациям решать, какой конкретный тип инстанцировать. В Go это обычно реализуется как функция, возвращающая интерфейсный тип, скрывая детали конкретной реализации.

**Ключевые компоненты:**

1. **Интерфейс Database** - определяет общие операции:
   \`\`\`go
   type Database interface {
       Connect() error
       Query(sql string) (string, error)
       Close() error
   }
   \`\`\`

2. **PostgresDB** - конкретная реализация для PostgreSQL:
   2.1. Приватная структура с полем DSN
   2.2. Реализует все методы интерфейса Database
   2.3. Connect() проверяет, что DSN содержит "postgres://" и устанавливает флаг подключения
   2.4. Query() проверяет подключение и возвращает отформатированный результат
   2.5. Close() помечает как отключенный

3. **MySQLDB** - конкретная реализация для MySQL:
   3.1. Приватная структура с полем DSN
   3.2. Реализует все методы интерфейса Database
   3.3. Connect() проверяет, что DSN содержит "mysql://" и устанавливает флаг подключения
   3.4. Query() проверяет подключение и возвращает отформатированный результат
   3.5. Close() помечает как отключенный

4. **Фабрика NewDatabase** - создает соответствующий тип базы данных:
   \`\`\`go
   func NewDatabase(driver, dsn string) (Database, error)
   \`\`\`
   4.1. Принимает "postgres" или "mysql" в качестве драйвера
   4.2. Возвращает соответствующий конкретный тип как интерфейс Database
   4.3. Возвращает ошибку для неподдерживаемых драйверов
   4.4. Проверяет, что DSN не пуст

**Пример использования:**
\`\`\`go
// Создание базы данных PostgreSQL
db, err := NewDatabase("postgres", "postgres://localhost:5432/mydb")
err = db.Connect()
result, err := db.Query("SELECT * FROM users")
fmt.Println(result) // "PostgreSQL result: SELECT * FROM users"

// Создание базы данных MySQL
db, err = NewDatabase("mysql", "mysql://localhost:3306/mydb")
err = db.Connect()
result, err = db.Query("SELECT * FROM products")
fmt.Println(result) // "MySQL result: SELECT * FROM products"
\`\`\`

**Почему этот паттерн:**
- Отделяет клиентский код от конкретных реализаций
- Легко добавлять новые типы баз данных без изменения существующего кода
- Клиентский код работает с интерфейсом, а не с конкретными типами
- Централизованная логика создания объектов`,
			hint1: `Начните с определения интерфейса Database с тремя методами. Затем создайте две приватные структуры (postgresDB и mysqlDB), которые реализуют этот интерфейс. Каждая структура нуждается в полях dsn и connected.`,
			hint2: `В NewDatabase сначала проверьте dsn, затем используйте оператор switch для драйвера. Верните соответствующий конкретный тип (&postgresDB или &mysqlDB) как интерфейс Database. Не забудьте обработать случай по умолчанию для неподдерживаемых драйверов.`,
			whyItMatters: `Паттерн Фабричный Метод необходим для создания гибкого, поддерживаемого кода, который может легко принимать новые типы без нарушения существующей функциональности.

**Почему Фабричный Метод:**
- **Абстракция:** Клиентский код зависит от интерфейсов, а не от конкретных типов
- **Расширяемость:** Добавление новых типов баз данных без изменения клиентского кода
- **Инкапсуляция:** Скрытие деталей конкретной реализации от клиентов
- **Тестируемость:** Легко мокировать интерфейс Database для тестирования
- **Гибкость:** Переключение реализаций во время выполнения на основе конфигурации

**Продакшен примеры:**

\`\`\`go
// Пример 1: Фабрика логгеров
type Logger interface {
	Log(level, message string)
}

type fileLogger struct { file *os.File }
type consoleLogger struct {}
type cloudLogger struct { client *CloudClient }

func NewLogger(logType string, config map[string]string) (Logger, error) {
	switch logType {
	case "file":
		return &fileLogger{file: openFile(config["path"])}, nil
	case "console":
		return &consoleLogger{}, nil
	case "cloud":
		return &cloudLogger{client: newCloudClient(config)}, nil
	default:
		return nil, fmt.Errorf("unknown logger type: %s", logType)
	}
}
\`\`\`

\`\`\`go
// Пример 2: Фабрика процессоров платежей
type PaymentProcessor interface {
	ProcessPayment(amount float64) error
	Refund(transactionID string) error
}

type stripeProcessor struct { apiKey string }
type paypalProcessor struct { clientID, secret string }
type cryptoProcessor struct { walletAddress string }

func NewPaymentProcessor(provider string, creds Credentials) (PaymentProcessor, error) {
	switch provider {
	case "stripe":
		return &stripeProcessor{apiKey: creds.APIKey}, nil
	case "paypal":
		return &paypalProcessor{
			clientID: creds.ClientID,
			secret:   creds.Secret,
		}, nil
	case "crypto":
		return &cryptoProcessor{walletAddress: creds.Wallet}, nil
	default:
		return nil, fmt.Errorf("unsupported payment provider")
	}
}
\`\`\`

\`\`\`go
// Пример 3: Фабрика кэша
type Cache interface {
	Get(key string) (interface{}, error)
	Set(key string, value interface{}) error
	Delete(key string) error
}

type memoryCache struct { data map[string]interface{} }
type redisCache struct { client *redis.Client }
type memcachedCache struct { client *memcache.Client }

func NewCache(cacheType, address string) (Cache, error) {
	switch cacheType {
	case "memory":
		return &memoryCache{data: make(map[string]interface{})}, nil
	case "redis":
		client := redis.NewClient(&redis.Options{Addr: address})
		return &redisCache{client: client}, nil
	case "memcached":
		client := memcache.New(address)
		return &memcachedCache{client: client}, nil
	default:
		return nil, fmt.Errorf("unknown cache type")
	}
}
\`\`\`

\`\`\`go
// Пример 4: Фабрика очереди сообщений
type MessageQueue interface {
	Publish(topic string, msg []byte) error
	Subscribe(topic string, handler func([]byte)) error
}

type kafkaQueue struct { producer *kafka.Producer }
type rabbitmqQueue struct { conn *amqp.Connection }
type sqsQueue struct { client *sqs.SQS }

func NewMessageQueue(queueType string, config QueueConfig) (MessageQueue, error) {
	switch queueType {
	case "kafka":
		return &kafkaQueue{producer: newKafkaProducer(config)}, nil
	case "rabbitmq":
		return &rabbitmqQueue{conn: connectRabbitMQ(config)}, nil
	case "sqs":
		return &sqsQueue{client: newSQSClient(config)}, nil
	default:
		return nil, fmt.Errorf("unsupported queue type")
	}
}
\`\`\`

**Практические преимущества:**
- **Kubernetes:** Использует фабричный паттерн для создания различных типов ресурсов
- **Docker:** Фабричный паттерн для создания container runtime (containerd, runc)
- **Драйверы БД:** пакет database/sql использует фабричный паттерн для драйверов
- **Cloud SDK:** AWS, GCP, Azure используют фабрики для клиентов сервисов

**Ключевые принципы проектирования:**
- Возвращать интерфейсные типы, а не конкретные типы
- Имя фабричной функции обычно начинается с "New"
- Валидировать входные данные перед созданием объектов
- Использовать чёткие сообщения об ошибках для неподдерживаемых типов
- Рассмотреть использование констант для строк типов во избежание опечаток

Без фабричных методов вам нужно было бы открывать все конкретные типы для клиентов, делая кодовую базу сложнее в поддержке и расширении. Фабричные методы обеспечивают чистый слой абстракции, который хорошо масштабируется.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Database interface {
	Connect() error                              // Установить соединение с базой данных
	Query(sql string) (string, error)           // Выполнить запрос и вернуть результат
	Close() error                                // Закрыть соединение с базой данных
}

type postgresDB struct {
	dsn       string                             // Строка подключения для PostgreSQL
	connected bool                               // Флаг состояния подключения
}

func (p *postgresDB) Connect() error {
	if !strings.Contains(p.dsn, "postgres://") { // Проверить формат DSN PostgreSQL
		return fmt.Errorf("invalid postgres DSN") // Вернуть ошибку для неверного DSN
	}
	p.connected = true                           // Пометить как подключенный
	return nil                                   // Вернуть nil при успехе
}

func (p *postgresDB) Query(sql string) (string, error) {
	if !p.connected {                            // Проверить состояние подключения
		return "", fmt.Errorf("not connected")   // Вернуть ошибку если не подключен
	}
	return "PostgreSQL result: " + sql, nil      // Вернуть отформатированный результат
}

func (p *postgresDB) Close() error {
	p.connected = false                          // Пометить как отключенный
	return nil                                   // Вернуть nil (без ошибки)
}

type mysqlDB struct {
	dsn       string                             // Строка подключения для MySQL
	connected bool                               // Флаг состояния подключения
}

func (m *mysqlDB) Connect() error {
	if !strings.Contains(m.dsn, "mysql://") {    // Проверить формат DSN MySQL
		return fmt.Errorf("invalid mysql DSN")   // Вернуть ошибку для неверного DSN
	}
	m.connected = true                           // Пометить как подключенный
	return nil                                   // Вернуть nil при успехе
}

func (m *mysqlDB) Query(sql string) (string, error) {
	if !m.connected {                            // Проверить состояние подключения
		return "", fmt.Errorf("not connected")   // Вернуть ошибку если не подключен
	}
	return "MySQL result: " + sql, nil           // Вернуть отформатированный результат
}

func (m *mysqlDB) Close() error {
	m.connected = false                          // Пометить как отключенный
	return nil                                   // Вернуть nil (без ошибки)
}

func NewDatabase(driver, dsn string) (Database, error) {
	if strings.TrimSpace(dsn) == "" {            // Проверить что DSN не пустой
		return nil, fmt.Errorf("dsn cannot be empty") // Вернуть ошибку для пустого DSN
	}

	switch strings.ToLower(driver) {             // Switch по строчному имени драйвера
	case "postgres":
		return &postgresDB{dsn: dsn}, nil        // Вернуть реализацию PostgreSQL
	case "mysql":
		return &mysqlDB{dsn: dsn}, nil           // Вернуть реализацию MySQL
	default:
		return nil, fmt.Errorf("unsupported driver: %s", driver) // Вернуть ошибку для неизвестного драйвера
	}
}`
		},
		uz: {
			title: 'Factory metod patterni',
			description: `Ma'lumotlar bazasi drayveri satriga asoslangan turli xil ma'lumotlar bazasi ulanishlarini yaratish uchun **Zavod Metodi Paradigmasini** amalga oshiring.

**Paradigma ta'rifi:**
Zavod Metodi Paradigmasi ob'ektlarni yaratish uchun interfeys taqdim etadi, ammo qaysi aniq turni yaratishni quyi sinflar yoki amalga oshirishlarga imkon beradi. Go da bu odatda interfeys turini qaytaruvchi funksiya sifatida amalga oshiriladi va aniq amalga oshirish tafsilotlarini yashiradi.

**Asosiy komponentlar:**

1. **Database Interfeysi** - umumiy operatsiyalarni belgilaydi:
   \`\`\`go
   type Database interface {
       Connect() error
       Query(sql string) (string, error)
       Close() error
   }
   \`\`\`

2. **PostgresDB** - PostgreSQL uchun aniq amalga oshirish:
   2.1. DSN maydonli xususiy struktura
   2.2. Database interfeysi barcha metodlarini amalga oshiradi
   2.3. Connect() DSN "postgres://" o'z ichiga olganligini tekshiradi va ulanish bayrog'ini o'rnatadi
   2.4. Query() ulanishni tekshiradi va formatlangan natijani qaytaradi
   2.5. Close() uzilgan deb belgilaydi

3. **MySQLDB** - MySQL uchun aniq amalga oshirish:
   3.1. DSN maydonli xususiy struktura
   3.2. Database interfeysi barcha metodlarini amalga oshiradi
   3.3. Connect() DSN "mysql://" o'z ichiga olganligini tekshiradi va ulanish bayrog'ini o'rnatadi
   3.4. Query() ulanishni tekshiradi va formatlangan natijani qaytaradi
   3.5. Close() uzilgan deb belgilaydi

4. **NewDatabase Zavodi** - mos ma'lumotlar bazasi turini yaratadi:
   \`\`\`go
   func NewDatabase(driver, dsn string) (Database, error)
   \`\`\`
   4.1. Drayver sifatida "postgres" yoki "mysql" qabul qiladi
   4.2. Mos aniq turni Database interfeysi sifatida qaytaradi
   4.3. Qo'llab-quvvatlanmaydigan drayverlar uchun xato qaytaradi
   4.4. DSN bo'sh emasligini tekshiradi

**Foydalanish misoli:**
\`\`\`go
// PostgreSQL ma'lumotlar bazasini yaratish
db, err := NewDatabase("postgres", "postgres://localhost:5432/mydb")
err = db.Connect()
result, err := db.Query("SELECT * FROM users")
fmt.Println(result) // "PostgreSQL result: SELECT * FROM users"

// MySQL ma'lumotlar bazasini yaratish
db, err = NewDatabase("mysql", "mysql://localhost:3306/mydb")
err = db.Connect()
result, err = db.Query("SELECT * FROM products")
fmt.Println(result) // "MySQL result: SELECT * FROM products"
\`\`\`

**Nima uchun bu paradigma:**
- Kliyent kodini aniq amalga oshirishlardan ajratadi
- Mavjud kodni o'zgartirmasdan yangi ma'lumotlar bazasi turlarini qo'shish oson
- Kliyent kodi aniq turlar bilan emas, balki interfeys bilan ishlaydi
- Ob'ektlarni yaratish mantigi markazlashtirilgan`,
			hint1: `Uchta metod bilan Database interfeysini aniqlashdan boshlang. Keyin bu interfeysni amalga oshiruvchi ikkita xususiy struktura (postgresDB va mysqlDB) yarating. Har bir strukturada dsn va connected maydonlari kerak.`,
			hint2: `NewDatabase da avval dsn ni tekshiring, keyin drayver uchun switch operatoridan foydalaning. Mos aniq turni (&postgresDB yoki &mysqlDB) Database interfeysi sifatida qaytaring. Qo'llab-quvvatlanmaydigan drayverlar uchun standart holatni qayta ishlashni unutmang.`,
			whyItMatters: `Zavod Metodi Paradigmasi mavjud funksionallikni buzmasdan yangi turlarni osongina qabul qilishi mumkin bo'lgan moslashuvchan, saqlash mumkin bo'lgan kod yaratish uchun zarur.

**Nima uchun Zavod Metodi:**
- **Abstraktsiya:** Kliyent kodi aniq turlarga emas, balki interfeyslarga bog'liq
- **Kengaytiriluvchilik:** Kliyent kodini o'zgartirmasdan yangi ma'lumotlar bazasi turlarini qo'shish
- **Inkapsulyatsiya:** Kliyentlardan aniq amalga oshirish tafsilotlarini yashirish
- **Testlanuvchilik:** Test uchun Database interfeysini osongina moklash
- **Moslashuvchanlik:** Konfiguratsiyaga asoslanib ish vaqtida amalga oshirishlarni almashtirish

**Ishlab chiqarish misollari:**

\`\`\`go
// Misol 1: Logger zavodi
type Logger interface {
	Log(level, message string)
}

type fileLogger struct { file *os.File }
type consoleLogger struct {}
type cloudLogger struct { client *CloudClient }

func NewLogger(logType string, config map[string]string) (Logger, error) {
	switch logType {
	case "file":
		return &fileLogger{file: openFile(config["path"])}, nil
	case "console":
		return &consoleLogger{}, nil
	case "cloud":
		return &cloudLogger{client: newCloudClient(config)}, nil
	default:
		return nil, fmt.Errorf("unknown logger type: %s", logType)
	}
}
\`\`\`

\`\`\`go
// Misol 2: To'lov protsessori zavodi
type PaymentProcessor interface {
	ProcessPayment(amount float64) error
	Refund(transactionID string) error
}

type stripeProcessor struct { apiKey string }
type paypalProcessor struct { clientID, secret string }
type cryptoProcessor struct { walletAddress string }

func NewPaymentProcessor(provider string, creds Credentials) (PaymentProcessor, error) {
	switch provider {
	case "stripe":
		return &stripeProcessor{apiKey: creds.APIKey}, nil
	case "paypal":
		return &paypalProcessor{
			clientID: creds.ClientID,
			secret:   creds.Secret,
		}, nil
	case "crypto":
		return &cryptoProcessor{walletAddress: creds.Wallet}, nil
	default:
		return nil, fmt.Errorf("unsupported payment provider")
	}
}
\`\`\`

\`\`\`go
// Misol 3: Kesh zavodi
type Cache interface {
	Get(key string) (interface{}, error)
	Set(key string, value interface{}) error
	Delete(key string) error
}

type memoryCache struct { data map[string]interface{} }
type redisCache struct { client *redis.Client }
type memcachedCache struct { client *memcache.Client }

func NewCache(cacheType, address string) (Cache, error) {
	switch cacheType {
	case "memory":
		return &memoryCache{data: make(map[string]interface{})}, nil
	case "redis":
		client := redis.NewClient(&redis.Options{Addr: address})
		return &redisCache{client: client}, nil
	case "memcached":
		client := memcache.New(address)
		return &memcachedCache{client: client}, nil
	default:
		return nil, fmt.Errorf("unknown cache type")
	}
}
\`\`\`

\`\`\`go
// Misol 4: Xabar navbati zavodi
type MessageQueue interface {
	Publish(topic string, msg []byte) error
	Subscribe(topic string, handler func([]byte)) error
}

type kafkaQueue struct { producer *kafka.Producer }
type rabbitmqQueue struct { conn *amqp.Connection }
type sqsQueue struct { client *sqs.SQS }

func NewMessageQueue(queueType string, config QueueConfig) (MessageQueue, error) {
	switch queueType {
	case "kafka":
		return &kafkaQueue{producer: newKafkaProducer(config)}, nil
	case "rabbitmq":
		return &rabbitmqQueue{conn: connectRabbitMQ(config)}, nil
	case "sqs":
		return &sqsQueue{client: newSQSClient(config)}, nil
	default:
		return nil, fmt.Errorf("unsupported queue type")
	}
}
\`\`\`

**Amaliy foydalari:**
- **Kubernetes:** Turli xil resurs turlarini yaratish uchun zavod paradigmasidan foydalanadi
- **Docker:** Konteyner runtimelarini yaratish uchun zavod paradigmasi (containerd, runc)
- **DB drayverlari:** database/sql paketi drayverlar uchun zavod paradigmasidan foydalanadi
- **Cloud SDK:** AWS, GCP, Azure xizmat kliyentlari uchun zavodlardan foydalanadi

**Asosiy dizayn tamoyillari:**
- Interfeys turlarini qaytarish, aniq turlarni emas
- Zavod funksiyasi nomi odatda "New" bilan boshlanadi
- Ob'ektlar yaratishdan oldin kirishlarni tekshirish
- Qo'llab-quvvatlanmaydigan turlar uchun aniq xato xabarlaridan foydalanish
- Xatolarni oldini olish uchun tur satrlar uchun konstantalardan foydalanishni ko'rib chiqing

Zavod metodlarisiz siz barcha aniq turlarni kliyentlarga ochishingiz kerak bo'ladi, bu esa kod bazasini saqlashni va kengaytirishni qiyinlashtiradi. Zavod metodlari yaxshi masshtablanadigan toza abstraktsiya qatlamini ta'minlaydi.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Database interface {
	Connect() error                              // Ma'lumotlar bazasiga ulanishni o'rnatish
	Query(sql string) (string, error)           // So'rovni bajarish va natijani qaytarish
	Close() error                                // Ma'lumotlar bazasi ulanishini yopish
}

type postgresDB struct {
	dsn       string                             // PostgreSQL uchun ulanish satri
	connected bool                               // Ulanish holati bayrog'i
}

func (p *postgresDB) Connect() error {
	if !strings.Contains(p.dsn, "postgres://") { // PostgreSQL DSN formatini tekshirish
		return fmt.Errorf("invalid postgres DSN") // Noto'g'ri DSN uchun xato qaytarish
	}
	p.connected = true                           // Ulangan deb belgilash
	return nil                                   // Muvaffaqiyatda nil qaytarish
}

func (p *postgresDB) Query(sql string) (string, error) {
	if !p.connected {                            // Ulanish holatini tekshirish
		return "", fmt.Errorf("not connected")   // Ulanmagan bo'lsa xato qaytarish
	}
	return "PostgreSQL result: " + sql, nil      // Formatlangan natijani qaytarish
}

func (p *postgresDB) Close() error {
	p.connected = false                          // Uzilgan deb belgilash
	return nil                                   // Nil qaytarish (xato yo'q)
}

type mysqlDB struct {
	dsn       string                             // MySQL uchun ulanish satri
	connected bool                               // Ulanish holati bayrog'i
}

func (m *mysqlDB) Connect() error {
	if !strings.Contains(m.dsn, "mysql://") {    // MySQL DSN formatini tekshirish
		return fmt.Errorf("invalid mysql DSN")   // Noto'g'ri DSN uchun xato qaytarish
	}
	m.connected = true                           // Ulangan deb belgilash
	return nil                                   // Muvaffaqiyatda nil qaytarish
}

func (m *mysqlDB) Query(sql string) (string, error) {
	if !m.connected {                            // Ulanish holatini tekshirish
		return "", fmt.Errorf("not connected")   // Ulanmagan bo'lsa xato qaytarish
	}
	return "MySQL result: " + sql, nil           // Formatlangan natijani qaytarish
}

func (m *mysqlDB) Close() error {
	m.connected = false                          // Uzilgan deb belgilash
	return nil                                   // Nil qaytarish (xato yo'q)
}

func NewDatabase(driver, dsn string) (Database, error) {
	if strings.TrimSpace(dsn) == "" {            // DSN bo'sh emasligini tekshirish
		return nil, fmt.Errorf("dsn cannot be empty") // Bo'sh DSN uchun xato qaytarish
	}

	switch strings.ToLower(driver) {             // Kichik harfli drayver nomi bo'yicha switch
	case "postgres":
		return &postgresDB{dsn: dsn}, nil        // PostgreSQL amalga oshirishini qaytarish
	case "mysql":
		return &mysqlDB{dsn: dsn}, nil           // MySQL amalga oshirishini qaytarish
	default:
		return nil, fmt.Errorf("unsupported driver: %s", driver) // Noma'lum drayver uchun xato qaytarish
	}
}`
		}
	}
};

export default task;
