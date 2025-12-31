import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-indirection',
	title: 'Indirection',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'indirection'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Indirection principle - assign responsibility to an intermediate object to mediate between components, reducing coupling.

**You will implement:**

1. **Database interface** - Abstraction for database operations
2. **PostgresDB struct** - Concrete PostgreSQL implementation
3. **DataAccessLayer struct** - Intermediary that uses Database interface
4. **UserService struct** - Business logic that uses DataAccessLayer (not DB directly!)

**Key Concepts:**
- **Indirection**: Add intermediate layer between components
- **Mediator**: Intermediate object reduces direct coupling
- **Adapter Pattern**: Interface adapts one component to another

**Example Usage:**

\`\`\`go
// Setup indirection layers
db := NewPostgresDB("connection-string")
dal := NewDataAccessLayer(db)  // DataAccessLayer mediates
service := NewUserService(dal)  // Service doesn't know about database

// Service talks to DAL, DAL talks to DB - no direct coupling!
user := service.CreateUser("john", "john@example.com")

// Easy to swap database - UserService unchanged!
db = NewMongoDBAdapter()  // different implementation
dal = NewDataAccessLayer(db)
service = NewUserService(dal)  // works the same!
\`\`\`

**Why Indirection?**
- **Reduced Coupling**: Components don't depend on each other directly
- **Flexibility**: Easy to change implementations
- **Testability**: Can inject mocks at any layer

**Anti-pattern (Don't do this):**
\`\`\`go
// HIGH COUPLING - BAD!
type UserService struct {
    postgresConn *sql.DB  // directly coupled to PostgreSQL!
}

func (s *UserService) CreateUser(name, email string) error {
    // Service knows SQL syntax - tightly coupled!
    _, err := s.postgresConn.Exec(
        "INSERT INTO users (name, email) VALUES ($1, $2)",
        name, email,
    )
    return err
}
// Can't switch databases without rewriting UserService!
// Can't test without real PostgreSQL!
\`\`\`

**Constraints:**
- UserService must NOT know about database implementation
- DataAccessLayer mediates between service and database
- All communication goes through interfaces`,
	initialCode: `package principles

import "fmt"

type Database interface {
}

type PostgresDB struct {
	connectionString string
	data             map[string]string // simulates database
}

func NewPostgresDB(connectionString string) *PostgresDB {
	}
}

func (p *PostgresDB) Execute(query string, args ...interface{}) error {
}

func (p *PostgresDB) Query(query string, args ...interface{}) (interface{}, error) {
}

type DataAccessLayer struct {
	db Database // depends on interface, not concrete type
}

func NewDataAccessLayer(db Database) *DataAccessLayer {
}

func (d *DataAccessLayer) SaveUser(name, email string) error {
}

func (d *DataAccessLayer) GetUser(name string) string {
}

type UserService struct {
	dal *DataAccessLayer // service depends on DAL, not database
}

func NewUserService(dal *DataAccessLayer) *UserService {
}

func (s *UserService) CreateUser(name, email string) error {
}`,
	solutionCode: `package principles

import "fmt"

// Database interface - abstraction layer
type Database interface {
	Execute(query string, args ...interface{}) error
	Query(query string, args ...interface{}) (interface{}, error)
}

// PostgresDB concrete implementation
type PostgresDB struct {
	connectionString string
	data             map[string]string
}

func NewPostgresDB(connectionString string) *PostgresDB {
	return &PostgresDB{
		connectionString: connectionString,
		data:             make(map[string]string),
	}
}

func (p *PostgresDB) Execute(query string, args ...interface{}) error {
	// Simulate INSERT - store name->email mapping
	if len(args) >= 2 {
		name := args[0].(string)
		email := args[1].(string)
		p.data[name] = email
	}
	return nil
}

func (p *PostgresDB) Query(query string, args ...interface{}) (interface{}, error) {
	// Simulate SELECT - retrieve by name
	if len(args) >= 1 {
		name := args[0].(string)
		return p.data[name], nil
	}
	return nil, nil
}

// DataAccessLayer - INDIRECTION between service and database
type DataAccessLayer struct {
	db Database	// interface provides indirection
}

func NewDataAccessLayer(db Database) *DataAccessLayer {
	return &DataAccessLayer{
		db: db,	// accepts any Database implementation
	}
}

func (d *DataAccessLayer) SaveUser(name, email string) error {
	// DAL handles database interaction
	// Hides SQL details from service layer
	query := "INSERT INTO users (name, email) VALUES (?, ?)"
	return d.db.Execute(query, name, email)
}

func (d *DataAccessLayer) GetUser(name string) string {
	// DAL handles database query
	query := "SELECT * FROM users WHERE name = ?"
	result, _ := d.db.Query(query, name)
	if result == nil {
		return ""
	}
	return result.(string)
}

// UserService - business logic layer
type UserService struct {
	dal *DataAccessLayer	// depends on DAL, not Database directly
}

func NewUserService(dal *DataAccessLayer) *UserService {
	return &UserService{
		dal: dal,	// indirection - service doesn't know about database
	}
}

func (s *UserService) CreateUser(name, email string) error {
	// Service uses DAL - no knowledge of database
	// DAL is the indirection that decouples service from database
	err := s.dal.SaveUser(name, email)
	if err != nil {
		return err
	}
	fmt.Printf("User created: %s\n", name)
	return nil
}`,
	hint1: `PostgresDB.Execute should store args[0] (name) and args[1] (email) in p.data map. Query should return p.data[args[0].(string)].`,
	hint2: `DataAccessLayer methods should call d.db.Execute or d.db.Query with appropriate parameters. UserService.CreateUser should call s.dal.SaveUser then print success message.`,
	whyItMatters: `Indirection reduces coupling by introducing intermediate objects that mediate between components.

**Why Indirection Matters:**

**1. Decouples Components**
Indirection layer prevents direct dependencies between components:

\`\`\`go
// WITHOUT INDIRECTION - HIGH COUPLING
type OrderService struct {
    paymentAPI *stripe.Client  // directly coupled to Stripe!
}

func (s *OrderService) ProcessOrder(order *Order) error {
    // Service knows Stripe API details - tightly coupled!
    charge, err := s.paymentAPI.Charges.New(&stripe.ChargeParams{
        Amount:   stripe.Int64(int64(order.Total * 100)),
        Currency: stripe.String("usd"),
        Source:   &stripe.SourceParams{Token: stripe.String(order.PaymentToken)},
    })
    // Can't switch to PayPal without rewriting OrderService!
}

// WITH INDIRECTION - LOW COUPLING
type PaymentGateway interface {
    Charge(amount float64, token string) (*Payment, error)
}

// StripeGateway - indirection for Stripe
type StripeGateway struct {
    client *stripe.Client
}

func (g *StripeGateway) Charge(amount float64, token string) (*Payment, error) {
    // Gateway handles Stripe-specific logic
    charge, err := g.client.Charges.New(&stripe.ChargeParams{
        Amount:   stripe.Int64(int64(amount * 100)),
        Currency: stripe.String("usd"),
        Source:   &stripe.SourceParams{Token: stripe.String(token)},
    })
    return &Payment{ID: charge.ID, Amount: amount}, err
}

type OrderService struct {
    gateway PaymentGateway  // service depends on interface
}

func (s *OrderService) ProcessOrder(order *Order) error {
    // Service doesn't know about Stripe - gateway is indirection!
    payment, err := s.gateway.Charge(order.Total, order.PaymentToken)
    // Easy to switch to PayPal, Square, etc. - just change gateway!
}
\`\`\`

**2. Real-World Example: Notification System**
\`\`\`go
// Multiple notification channels
type NotificationChannel interface {
    Send(recipient, message string) error
}

// Email channel
type EmailChannel struct {
    smtpServer string
}

func (e *EmailChannel) Send(recipient, message string) error {
    return smtp.SendMail(e.smtpServer, nil, "noreply@app.com", []string{recipient}, []byte(message))
}

// SMS channel
type SMSChannel struct {
    twilioClient *twilio.Client
}

func (s *SMSChannel) Send(recipient, message string) error {
    return s.twilioClient.SendSMS(recipient, message)
}

// Push notification channel
type PushChannel struct {
    fcmClient *fcm.Client
}

func (p *PushChannel) Send(recipient, message string) error {
    return p.fcmClient.SendPush(recipient, message)
}

// NotificationRouter - INDIRECTION that routes messages to appropriate channel
type NotificationRouter struct {
    channels map[string]NotificationChannel
}

func NewNotificationRouter() *NotificationRouter {
    return &NotificationRouter{
        channels: map[string]NotificationChannel{
            "email": &EmailChannel{smtpServer: "smtp.gmail.com"},
            "sms":   &SMSChannel{twilioClient: twilioClient},
            "push":  &PushChannel{fcmClient: fcmClient},
        },
    }
}

func (r *NotificationRouter) Route(channelType, recipient, message string) error {
    channel, ok := r.channels[channelType]
    if !ok {
        return errors.New("unknown channel")
    }
    return channel.Send(recipient, message)
}

// NotificationService uses router - doesn't know about channels!
type NotificationService struct {
    router *NotificationRouter
}

func (n *NotificationService) NotifyUser(userID int, message string) error {
    user := n.getUser(userID)

    // Service uses router indirection - no knowledge of channels
    if user.HasEmail {
        n.router.Route("email", user.Email, message)
    }
    if user.HasPhone {
        n.router.Route("sms", user.Phone, message)
    }
    if user.HasDevice {
        n.router.Route("push", user.DeviceToken, message)
    }

    return nil
}
\`\`\`

**3. Caching Layer as Indirection**
\`\`\`go
// Repository interface
type UserRepository interface {
    FindByID(id int) (*User, error)
    Save(user *User) error
}

// Database repository
type DBUserRepository struct {
    db *sql.DB
}

func (r *DBUserRepository) FindByID(id int) (*User, error) {
    var user User
    err := r.db.QueryRow("SELECT * FROM users WHERE id = ?", id).Scan(&user.ID, &user.Name)
    return &user, err
}

func (r *DBUserRepository) Save(user *User) error {
    _, err := r.db.Exec("INSERT INTO users (id, name) VALUES (?, ?)", user.ID, user.Name)
    return err
}

// CachingRepository - INDIRECTION that adds caching
type CachingRepository struct {
    cache      map[int]*User
    repository UserRepository  // wraps another repository
}

func NewCachingRepository(repo UserRepository) *CachingRepository {
    return &CachingRepository{
        cache:      make(map[int]*User),
        repository: repo,
    }
}

func (c *CachingRepository) FindByID(id int) (*User, error) {
    // Check cache first
    if user, ok := c.cache[id]; ok {
        return user, nil
    }

    // Cache miss - fetch from wrapped repository
    user, err := c.repository.FindByID(id)
    if err != nil {
        return nil, err
    }

    // Store in cache
    c.cache[id] = user
    return user, nil
}

func (c *CachingRepository) Save(user *User) error {
    // Save to wrapped repository
    err := c.repository.Save(user)
    if err != nil {
        return err
    }

    // Update cache
    c.cache[user.ID] = user
    return nil
}

// Usage - easy to add/remove caching
func main() {
    // Without caching
    dbRepo := &DBUserRepository{db: db}
    service := NewUserService(dbRepo)

    // With caching - just wrap with indirection!
    dbRepo := &DBUserRepository{db: db}
    cachedRepo := NewCachingRepository(dbRepo)
    service := NewUserService(cachedRepo)
}
\`\`\`

**4. API Version Indirection**
\`\`\`go
// APIClient interface
type APIClient interface {
    GetUser(id int) (*User, error)
    CreateOrder(order *Order) error
}

// V1Client
type V1Client struct {
    baseURL string
}

func (c *V1Client) GetUser(id int) (*User, error) {
    // V1 API format
    resp, _ := http.Get(fmt.Sprintf("%s/v1/users/%d", c.baseURL, id))
    var user User
    json.NewDecoder(resp.Body).Decode(&user)
    return &user, nil
}

// V2Client
type V2Client struct {
    baseURL string
}

func (c *V2Client) GetUser(id int) (*User, error) {
    // V2 API format (different response structure)
    resp, _ := http.Get(fmt.Sprintf("%s/v2/users/%d", c.baseURL, id))
    var response struct {
        Data User \`json:"data"\`
    }
    json.NewDecoder(resp.Body).Decode(&response)
    return &response.Data, nil
}

// APIClientFactory - INDIRECTION that creates appropriate client
type APIClientFactory struct{}

func (f *APIClientFactory) CreateClient(version string) APIClient {
    switch version {
    case "v1":
        return &V1Client{baseURL: "https://api.example.com"}
    case "v2":
        return &V2Client{baseURL: "https://api.example.com"}
    default:
        return &V1Client{baseURL: "https://api.example.com"}
    }
}

// Service uses factory - doesn't know about versions
type UserService struct {
    client APIClient
}

func main() {
    factory := &APIClientFactory{}
    client := factory.CreateClient("v2")
    service := &UserService{client: client}
}
\`\`\`

**5. Benefits of Indirection**
- **Flexibility**: Easy to swap implementations
- **Testability**: Mock intermediate layer
- **Evolution**: Add features without changing clients
- **Separation**: Clear boundaries between layers

**Common Mistakes:**
- Too many indirection layers (over-engineering)
- Indirection without clear benefit
- Leaking implementation details through indirection
- Not using interfaces for indirection

**Rule of Thumb:**
Add indirection when you need to decouple components that would otherwise be tightly coupled, especially for external dependencies.`,
	order: 7,
	testCode: `package principles

import (
	"testing"
)

// Test1: PostgresDB implements Database interface
func Test1(t *testing.T) {
	db := NewPostgresDB("connection-string")
	var _ Database = db // compile-time check
	if db == nil {
		t.Error("PostgresDB should not be nil")
	}
}

// Test2: PostgresDB Execute stores data
func Test2(t *testing.T) {
	db := NewPostgresDB("conn")
	err := db.Execute("INSERT", "john", "john@test.com")
	if err != nil {
		t.Errorf("Execute should succeed, got %v", err)
	}
}

// Test3: PostgresDB Query retrieves stored data
func Test3(t *testing.T) {
	db := NewPostgresDB("conn")
	db.Execute("INSERT", "john", "john@test.com")
	result, err := db.Query("SELECT", "john")
	if err != nil || result != "john@test.com" {
		t.Error("Query should return stored email")
	}
}

// Test4: DataAccessLayer SaveUser works
func Test4(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	err := dal.SaveUser("alice", "alice@test.com")
	if err != nil {
		t.Errorf("SaveUser should succeed, got %v", err)
	}
}

// Test5: DataAccessLayer GetUser returns stored data
func Test5(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	dal.SaveUser("bob", "bob@test.com")
	email := dal.GetUser("bob")
	if email != "bob@test.com" {
		t.Errorf("Expected bob@test.com, got %s", email)
	}
}

// Test6: UserService CreateUser works
func Test6(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	service := NewUserService(dal)
	err := service.CreateUser("charlie", "charlie@test.com")
	if err != nil {
		t.Errorf("CreateUser should succeed, got %v", err)
	}
}

// Test7: Full indirection chain works
func Test7(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	service := NewUserService(dal)
	service.CreateUser("dave", "dave@test.com")
	email := dal.GetUser("dave")
	if email != "dave@test.com" {
		t.Error("Data should be retrievable through full chain")
	}
}

// Test8: DataAccessLayer GetUser returns empty for non-existent
func Test8(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	email := dal.GetUser("nonexistent")
	if email != "" {
		t.Error("Should return empty string for non-existent user")
	}
}

// Test9: Multiple users can be stored
func Test9(t *testing.T) {
	db := NewPostgresDB("conn")
	dal := NewDataAccessLayer(db)
	dal.SaveUser("a", "a@test.com")
	dal.SaveUser("b", "b@test.com")
	if dal.GetUser("a") != "a@test.com" || dal.GetUser("b") != "b@test.com" {
		t.Error("Both users should be retrievable")
	}
}

// Test10: PostgresDB Query returns nil for non-existent key
func Test10(t *testing.T) {
	db := NewPostgresDB("conn")
	result, _ := db.Query("SELECT", "nobody")
	if result != "" {
		t.Error("Should return empty for non-existent key")
	}
}
`,
	translations: {
		ru: {
			title: 'Перенаправление',
			description: `Реализуйте принцип Перенаправления — назначьте ответственность промежуточному объекту для посредничества между компонентами, уменьшая связанность.

**Вы реализуете:**

1. **Database interface** — Абстракция для операций с БД
2. **PostgresDB struct** — Конкретная реализация PostgreSQL
3. **DataAccessLayer struct** — Посредник, использующий интерфейс Database
4. **UserService struct** — Бизнес-логика, использующая DataAccessLayer (а не БД напрямую!)

**Ключевые концепции:**
- **Перенаправление**: Добавление промежуточного слоя между компонентами
- **Посредник**: Промежуточный объект уменьшает прямую связанность
- **Паттерн адаптера**: Интерфейс адаптирует один компонент к другому

**Зачем нужно Перенаправление?**
- **Уменьшенная связанность**: Компоненты не зависят друг от друга напрямую
- **Гибкость**: Легко менять реализации
- **Тестируемость**: Можно внедрять моки на любом слое

**Ограничения:**
- UserService НЕ должен знать о реализации базы данных
- DataAccessLayer посредничает между сервисом и базой данных
- Вся коммуникация идёт через интерфейсы`,
			hint1: `PostgresDB.Execute должен сохранить args[0] (имя) и args[1] (email) в map p.data. Query должен вернуть p.data[args[0].(string)].`,
			hint2: `Методы DataAccessLayer должны вызывать d.db.Execute или d.db.Query с соответствующими параметрами. UserService.CreateUser должен вызвать s.dal.SaveUser, затем напечатать сообщение об успехе.`,
			whyItMatters: `Перенаправление уменьшает связанность путём введения промежуточных объектов, которые посредничают между компонентами.

**Почему Перенаправление важно:**

**1. Разделяет компоненты**
Слой перенаправления предотвращает прямые зависимости между компонентами.

**Распространённые ошибки:**
- Слишком много слоёв перенаправления (излишняя инженерия)
- Перенаправление без явной пользы
- Утечка деталей реализации через перенаправление
- Неиспользование интерфейсов для перенаправления`,
			solutionCode: `package principles

import "fmt"

// Database интерфейс - слой абстракции
type Database interface {
	Execute(query string, args ...interface{}) error
	Query(query string, args ...interface{}) (interface{}, error)
}

// PostgresDB конкретная реализация
type PostgresDB struct {
	connectionString string
	data             map[string]string
}

func NewPostgresDB(connectionString string) *PostgresDB {
	return &PostgresDB{
		connectionString: connectionString,
		data:             make(map[string]string),
	}
}

func (p *PostgresDB) Execute(query string, args ...interface{}) error {
	// Имитация INSERT - сохраняем связь имя->email
	if len(args) >= 2 {
		name := args[0].(string)
		email := args[1].(string)
		p.data[name] = email
	}
	return nil
}

func (p *PostgresDB) Query(query string, args ...interface{}) (interface{}, error) {
	// Имитация SELECT - получение по имени
	if len(args) >= 1 {
		name := args[0].(string)
		return p.data[name], nil
	}
	return nil, nil
}

// DataAccessLayer - ПЕРЕНАПРАВЛЕНИЕ между сервисом и базой данных
type DataAccessLayer struct {
	db Database	// интерфейс обеспечивает перенаправление
}

func NewDataAccessLayer(db Database) *DataAccessLayer {
	return &DataAccessLayer{
		db: db,	// принимает любую реализацию Database
	}
}

func (d *DataAccessLayer) SaveUser(name, email string) error {
	// DAL обрабатывает взаимодействие с БД
	// Скрывает детали SQL от слоя сервиса
	query := "INSERT INTO users (name, email) VALUES (?, ?)"
	return d.db.Execute(query, name, email)
}

func (d *DataAccessLayer) GetUser(name string) string {
	// DAL обрабатывает запрос к БД
	query := "SELECT * FROM users WHERE name = ?"
	result, _ := d.db.Query(query, name)
	if result == nil {
		return ""
	}
	return result.(string)
}

// UserService - слой бизнес-логики
type UserService struct {
	dal *DataAccessLayer	// зависит от DAL, не напрямую от Database
}

func NewUserService(dal *DataAccessLayer) *UserService {
	return &UserService{
		dal: dal,	// перенаправление - сервис не знает о базе данных
	}
}

func (s *UserService) CreateUser(name, email string) error {
	// Сервис использует DAL - нет знаний о базе данных
	// DAL - перенаправление, которое разделяет сервис и базу данных
	err := s.dal.SaveUser(name, email)
	if err != nil {
		return err
	}
	fmt.Printf("User created: %s\n", name)
	return nil
}`
		},
		uz: {
			title: 'Indirection (Bilvosita)',
			description: `Indirection prinsipini amalga oshiring — komponentlar orasida vositachilik qilish uchun oraliq ob'ektga mas'uliyatni belgilang, bog'lanishni kamaytiring.

**Siz amalga oshirasiz:**

1. **Database interface** — Ma'lumotlar bazasi operatsiyalari uchun abstraktsiya
2. **PostgresDB struct** — Konkret PostgreSQL implementatsiyasi
3. **DataAccessLayer struct** — Database interfeysidan foydalanadigan vositachi
4. **UserService struct** — DataAccessLayer dan foydalanadigan biznes mantiqi (to'g'ridan-to'g'ri DB dan emas!)

**Asosiy tushunchalar:**
- **Indirection**: Komponentlar orasida oraliq qatlam qo'shish
- **Vositachi**: Oraliq ob'ekt to'g'ridan-to'g'ri bog'lanishni kamaytiradi
- **Adapter pattern**: Interfeys bir komponentni boshqasiga moslashtiradi

**Nima uchun Indirection?**
- **Kamaytirilgan bog'lanish**: Komponentlar bir-biriga to'g'ridan-to'g'ri bog'liq emas
- **Moslashuvchanlik**: Implementatsiyalarni osongina o'zgartirish
- **Testlanish**: Har qanday qatlamda mocklar injeksiya qilish mumkin

**Cheklovlar:**
- UserService ma'lumotlar bazasi implementatsiyasini bilmasligi kerak
- DataAccessLayer servis va ma'lumotlar bazasi orasida vositachilik qiladi
- Barcha kommunikatsiya interfeyslar orqali bo'ladi`,
			hint1: `PostgresDB.Execute args[0] (nom) va args[1] (email) ni p.data map da saqlashi kerak. Query p.data[args[0].(string)] ni qaytarishi kerak.`,
			hint2: `DataAccessLayer metodlari mos parametrlar bilan d.db.Execute yoki d.db.Query ni chaqirishi kerak. UserService.CreateUser s.dal.SaveUser ni chaqirishi, keyin muvaffaqiyat xabarini chop etishi kerak.`,
			whyItMatters: `Indirection komponentlar orasida vositachilik qiluvchi oraliq ob'ektlarni kiritish orqali bog'lanishni kamaytiradi.

**Indirection nima uchun muhim:**

**1. Komponentlarni ajratadi**
Indirection qatlami komponentlar orasidagi to'g'ridan-to'g'ri bog'liqliklarni oldini oladi.

**Umumiy xatolar:**
- Juda ko'p indirection qatlamlari (ortiqcha muhandislik)
- Aniq foydasiz indirection
- Indirection orqali implementatsiya tafsilotlarini sizdirib yuborish
- Indirection uchun interfeyslardan foydalanmaslik`,
			solutionCode: `package principles

import "fmt"

// Database interfeysi - abstraktsiya qatlami
type Database interface {
	Execute(query string, args ...interface{}) error
	Query(query string, args ...interface{}) (interface{}, error)
}

// PostgresDB konkret implementatsiya
type PostgresDB struct {
	connectionString string
	data             map[string]string
}

func NewPostgresDB(connectionString string) *PostgresDB {
	return &PostgresDB{
		connectionString: connectionString,
		data:             make(map[string]string),
	}
}

func (p *PostgresDB) Execute(query string, args ...interface{}) error {
	// INSERT ni simulyatsiya qilish - nom->email mapping ni saqlash
	if len(args) >= 2 {
		name := args[0].(string)
		email := args[1].(string)
		p.data[name] = email
	}
	return nil
}

func (p *PostgresDB) Query(query string, args ...interface{}) (interface{}, error) {
	// SELECT ni simulyatsiya qilish - nomga ko'ra olish
	if len(args) >= 1 {
		name := args[0].(string)
		return p.data[name], nil
	}
	return nil, nil
}

// DataAccessLayer - servis va ma'lumotlar bazasi o'rtasida INDIRECTION
type DataAccessLayer struct {
	db Database	// interfeys indirection ni ta'minlaydi
}

func NewDataAccessLayer(db Database) *DataAccessLayer {
	return &DataAccessLayer{
		db: db,	// har qanday Database implementatsiyasini qabul qiladi
	}
}

func (d *DataAccessLayer) SaveUser(name, email string) error {
	// DAL ma'lumotlar bazasi bilan o'zaro ta'sirni boshqaradi
	// Servis qatlamidan SQL tafsilotlarini yashiradi
	query := "INSERT INTO users (name, email) VALUES (?, ?)"
	return d.db.Execute(query, name, email)
}

func (d *DataAccessLayer) GetUser(name string) string {
	// DAL ma'lumotlar bazasi so'rovini boshqaradi
	query := "SELECT * FROM users WHERE name = ?"
	result, _ := d.db.Query(query, name)
	if result == nil {
		return ""
	}
	return result.(string)
}

// UserService - biznes mantiqi qatlami
type UserService struct {
	dal *DataAccessLayer	// to'g'ridan-to'g'ri Database ga emas, DAL ga bog'liq
}

func NewUserService(dal *DataAccessLayer) *UserService {
	return &UserService{
		dal: dal,	// indirection - servis ma'lumotlar bazasi haqida bilmaydi
	}
}

func (s *UserService) CreateUser(name, email string) error {
	// Servis DAL dan foydalanadi - ma'lumotlar bazasi haqida bilim yo'q
	// DAL servis va ma'lumotlar bazasini ajratib turadigan indirection
	err := s.dal.SaveUser(name, email)
	if err != nil {
		return err
	}
	fmt.Printf("User created: %s\n", name)
	return nil
}`
		}
	}
};

export default task;
