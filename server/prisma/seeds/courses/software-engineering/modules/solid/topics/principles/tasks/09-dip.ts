import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-dip',
	title: 'Dependency Inversion Principle',
	difficulty: 'medium',
	tags: ['go', 'solid', 'dip', 'dependency-injection'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Dependency Inversion Principle (DIP) - high-level modules should not depend on low-level modules; both should depend on abstractions.

**Current Problem:**

A UserService directly depends on MySQL database implementation, making it impossible to switch databases or test easily.

**Your task:**

Refactor to depend on abstractions:

1. **UserRepository interface** - Abstract data access
2. **MySQLRepository** - Implements UserRepository for MySQL
3. **PostgresRepository** - Implements UserRepository for Postgres
4. **UserService** - Depends on UserRepository interface (not concrete implementation)

**Key Concepts:**
- **Depend on abstractions**: Use interfaces, not concrete types
- **Dependency injection**: Pass dependencies from outside
- **Inversion of control**: High-level code doesn't create low-level objects

**Example Usage:**

\`\`\`go
// Create repositories
mysql := &MySQLRepository{ConnectionString: "mysql://..."}
postgres := &PostgresRepository{ConnectionString: "postgres://..."}

// UserService works with any repository
service1 := NewUserService(mysql)      // uses MySQL
service2 := NewUserService(postgres)   // uses Postgres

// Same service code, different database!
user1 := service1.GetUser(1)
user2 := service2.GetUser(1)
\`\`\`

**Why DIP matters:**
- Easy to switch implementations
- Makes testing trivial (use mock repositories)
- Reduces coupling between layers
- Enables plugin architectures

**Constraints:**
- UserService must NOT import database packages
- UserService depends only on UserRepository interface
- Repository implementations injected via constructor`,
	initialCode: `package principles

import (
	"database/sql"
	"fmt"
)

type User struct {
	ID    int
	Name  string
	Email string
}

type UserService struct {
	db *sql.DB  // Direct dependency on database - VIOLATES DIP!
}

func NewUserService(connectionString string) *UserService {
}

func (us *UserService) GetUser(id int) (*User, error) {

	return user, err
}

func (us *UserService) SaveUser(user *User) error {
	return err
}

type UserRepository interface {
}

type MySQLRepository struct {
	ConnectionString string
}

func NewMySQLRepository(connectionString string) *MySQLRepository {
}

func (mr *MySQLRepository) FindByID(id int) (*User, error) {
}

func (mr *MySQLRepository) Save(user *User) error {
}

type PostgresRepository struct {
	ConnectionString string
}

func NewPostgresRepository(connectionString string) *PostgresRepository {
}

func (pr *PostgresRepository) FindByID(id int) (*User, error) {
}

func (pr *PostgresRepository) Save(user *User) error {
}

type UserServiceRefactored struct {
	repo UserRepository  // Depends on interface, not concrete type!
}

func NewUserServiceRefactored(repo UserRepository) *UserServiceRefactored {
}

func (us *UserServiceRefactored) GetUser(id int) (*User, error) {
}

func (us *UserServiceRefactored) SaveUser(user *User) error {
}`,
	solutionCode: `package principles

import (
	"database/sql"
	"fmt"
)

type User struct {
	ID    int
	Name  string
	Email string
}

// UserRepository - abstraction that both high and low-level modules depend on
// DIP compliant: defines contract without implementation details
type UserRepository interface {
	FindByID(id int) (*User, error)	// abstract data retrieval
	Save(user *User) error		// abstract data persistence
}

// MySQLRepository - low-level module implementing the abstraction
// DIP compliant: depends on same interface as high-level module
type MySQLRepository struct {
	ConnectionString string
	db               *sql.DB
}

// NewMySQLRepository creates MySQL repository with database connection
func NewMySQLRepository(connectionString string) *MySQLRepository {
	db, _ := sql.Open("mysql", connectionString)	// connect to MySQL
	return &MySQLRepository{
		ConnectionString: connectionString,
		db:               db,
	}
}

// FindByID retrieves user from MySQL database
func (mr *MySQLRepository) FindByID(id int) (*User, error) {
	// MySQL-specific query implementation
	row := mr.db.QueryRow("SELECT id, name, email FROM users WHERE id = ?", id)

	user := &User{}
	err := row.Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		return nil, err
	}

	fmt.Printf("MySQL: Retrieved user %d\\n", id)
	return user, nil
}

// Save persists user to MySQL database
func (mr *MySQLRepository) Save(user *User) error {
	// MySQL-specific insert implementation
	_, err := mr.db.Exec("INSERT INTO users (name, email) VALUES (?, ?)",
		user.Name, user.Email)

	if err != nil {
		return err
	}

	fmt.Printf("MySQL: Saved user %s\\n", user.Name)
	return nil
}

// PostgresRepository - alternative low-level implementation
// DIP compliant: implements same interface, different database
type PostgresRepository struct {
	ConnectionString string
	db               *sql.DB
}

// NewPostgresRepository creates Postgres repository with database connection
func NewPostgresRepository(connectionString string) *PostgresRepository {
	db, _ := sql.Open("postgres", connectionString)	// connect to Postgres
	return &PostgresRepository{
		ConnectionString: connectionString,
		db:               db,
	}
}

// FindByID retrieves user from Postgres database
func (pr *PostgresRepository) FindByID(id int) (*User, error) {
	// Postgres-specific query (uses $1 instead of ?)
	row := pr.db.QueryRow("SELECT id, name, email FROM users WHERE id = $1", id)

	user := &User{}
	err := row.Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		return nil, err
	}

	fmt.Printf("Postgres: Retrieved user %d\\n", id)
	return user, nil
}

// Save persists user to Postgres database
func (pr *PostgresRepository) Save(user *User) error {
	// Postgres-specific insert (uses $1, $2 instead of ?)
	_, err := pr.db.Exec("INSERT INTO users (name, email) VALUES ($1, $2)",
		user.Name, user.Email)

	if err != nil {
		return err
	}

	fmt.Printf("Postgres: Saved user %s\\n", user.Name)
	return nil
}

// UserServiceRefactored - high-level module depending on abstraction
// DIP compliant: depends on UserRepository interface, not concrete type
type UserServiceRefactored struct {
	repo UserRepository	// interface, not *MySQLRepository or *PostgresRepository
}

// NewUserServiceRefactored - dependency injection via constructor
// DIP compliant: receives dependency from outside, doesn't create it
func NewUserServiceRefactored(repo UserRepository) *UserServiceRefactored {
	return &UserServiceRefactored{
		repo: repo,	// any UserRepository implementation works
	}
}

// GetUser retrieves user using injected repository
// Works with MySQL, Postgres, or any UserRepository implementation
func (us *UserServiceRefactored) GetUser(id int) (*User, error) {
	return us.repo.FindByID(id)	// delegates to abstraction
}

// SaveUser persists user using injected repository
// Works with any database implementation
func (us *UserServiceRefactored) SaveUser(user *User) error {
	return us.repo.Save(user)	// delegates to abstraction
}

// Usage demonstrates DIP:
// mysql := NewMySQLRepository("mysql://...")
// service := NewUserServiceRefactored(mysql)  // inject MySQL
// user := service.GetUser(1)  // uses MySQL
//
// postgres := NewPostgresRepository("postgres://...")
// service2 := NewUserServiceRefactored(postgres)  // inject Postgres
// user2 := service2.GetUser(1)  // uses Postgres
//
// Same service code works with different databases!`,
	hint1: `For repository constructors, call sql.Open with appropriate driver name and return a new repository struct with db field set. For FindByID, use db.QueryRow with SQL SELECT and Scan into User. For Save, use db.Exec with INSERT.`,
	hint2: `For NewUserServiceRefactored, return &UserServiceRefactored{repo: repo}. For GetUser, return us.repo.FindByID(id). For SaveUser, return us.repo.Save(user).`,
	testCode: `package principles

import "testing"

// MockRepository for testing
type MockRepository struct {
	users map[int]*User
}

func (m *MockRepository) FindByID(id int) (*User, error) {
	if user, ok := m.users[id]; ok {
		return user, nil
	}
	return nil, nil
}

func (m *MockRepository) Save(user *User) error {
	if m.users == nil {
		m.users = make(map[int]*User)
	}
	m.users[user.ID] = user
	return nil
}

// Test1: NewUserServiceRefactored creates service
func Test1(t *testing.T) {
	mock := &MockRepository{}
	service := NewUserServiceRefactored(mock)
	if service == nil {
		t.Error("NewUserServiceRefactored returned nil")
	}
}

// Test2: GetUser delegates to repository
func Test2(t *testing.T) {
	mock := &MockRepository{
		users: map[int]*User{1: {ID: 1, Name: "John"}},
	}
	service := NewUserServiceRefactored(mock)
	user, err := service.GetUser(1)
	if err != nil {
		t.Errorf("GetUser error: %v", err)
	}
	if user == nil || user.Name != "John" {
		t.Error("GetUser returned wrong user")
	}
}

// Test3: SaveUser delegates to repository
func Test3(t *testing.T) {
	mock := &MockRepository{}
	service := NewUserServiceRefactored(mock)
	user := &User{ID: 1, Name: "Jane"}
	err := service.SaveUser(user)
	if err != nil {
		t.Errorf("SaveUser error: %v", err)
	}
}

// Test4: Service works with MySQL repository
func Test4(t *testing.T) {
	repo := NewMySQLRepository("test-connection")
	service := NewUserServiceRefactored(repo)
	_ = service
}

// Test5: Service works with Postgres repository
func Test5(t *testing.T) {
	repo := NewPostgresRepository("test-connection")
	service := NewUserServiceRefactored(repo)
	_ = service
}

// Test6: MySQLRepository implements UserRepository
func Test6(t *testing.T) {
	var repo UserRepository = NewMySQLRepository("conn")
	_ = repo
}

// Test7: PostgresRepository implements UserRepository
func Test7(t *testing.T) {
	var repo UserRepository = NewPostgresRepository("conn")
	_ = repo
}

// Test8: Multiple services with different repos
func Test8(t *testing.T) {
	mysql := NewMySQLRepository("mysql://...")
	postgres := NewPostgresRepository("postgres://...")

	service1 := NewUserServiceRefactored(mysql)
	service2 := NewUserServiceRefactored(postgres)

	_ = service1
	_ = service2
}

// Test9: Service can be reused
func Test9(t *testing.T) {
	mock := &MockRepository{users: make(map[int]*User)}
	service := NewUserServiceRefactored(mock)

	service.SaveUser(&User{ID: 1, Name: "User1"})
	service.SaveUser(&User{ID: 2, Name: "User2"})

	u1, _ := service.GetUser(1)
	u2, _ := service.GetUser(2)

	if u1.Name != "User1" || u2.Name != "User2" {
		t.Error("Users not saved correctly")
	}
}

// Test10: Repository interface is correct
func Test10(t *testing.T) {
	var _ UserRepository = &MockRepository{}
}
`,
	whyItMatters: `The Dependency Inversion Principle is the foundation of flexible, testable architecture.

**Why DIP Matters:**

**1. Testing Without Database**

\`\`\`go
// WITHOUT DIP - can't test without real database
type OrderService struct {
	db *sql.DB  // tightly coupled to database
}

func TestOrderService(t *testing.T) {
	// Must have real database running
	// Slow, fragile tests
	db, _ := sql.Open("mysql", "test_db")
	service := &OrderService{db: db}
	// test requires actual database...
}

// WITH DIP - easy to test with mock
type OrderRepository interface {
	FindByID(id int) (*Order, error)
}

type MockOrderRepository struct {
	orders map[int]*Order
}

func (m *MockOrderRepository) FindByID(id int) (*Order, error) {
	return m.orders[id], nil
}

func TestOrderServiceWithDIP(t *testing.T) {
	// No database needed!
	mock := &MockOrderRepository{
		orders: map[int]*Order{
			1: {ID: 1, Total: 100},
		},
	}

	service := NewOrderService(mock)  // inject mock
	order, _ := service.GetOrder(1)

	if order.Total != 100 {
		t.Error("wrong total")
	}
	// Fast, reliable, no database!
}
\`\`\`

**2. Switching Implementations**

\`\`\`go
// VIOLATES DIP
type EmailService struct {
	smtp SMTPClient  // tied to SMTP
}

func (es *EmailService) Send(to, message string) {
	es.smtp.Send(to, message)
	// Want to switch to SendGrid? Must rewrite EmailService!
}

// FOLLOWS DIP
type EmailSender interface {
	Send(to, message string) error
}

type SMTPSender struct{}

func (s *SMTPSender) Send(to, message string) error {
	// SMTP implementation
	return nil
}

type SendGridSender struct{}

func (s *SendGridSender) Send(to, message string) error {
	// SendGrid implementation
	return nil
}

type NotificationService struct {
	sender EmailSender  // depends on abstraction
}

// Switch email provider by injecting different implementation
// production := NewNotificationService(&SMTPSender{})
// testing := NewNotificationService(&SendGridSender{})
// No code changes to NotificationService!
\`\`\`

**3. Plugin Architecture**

\`\`\`go
// DIP enables plugin systems
type Logger interface {
	Log(message string) error
}

type Application struct {
	logger Logger  // depends on abstraction
}

// Plugins implement Logger interface
type FileLogger struct{}
type CloudLogger struct{}
type SyslogLogger struct{}

// Application works with any logger
app := &Application{logger: &FileLogger{}}    // file logging
app := &Application{logger: &CloudLogger{}}   // cloud logging
app := &Application{logger: &SyslogLogger{}}  // syslog
// Application code unchanged!
\`\`\``,
	order: 8,
	translations: {
		ru: {
			title: 'Принцип инверсии зависимостей',
			description: `Реализуйте принцип инверсии зависимостей (DIP) - модули высокого уровня не должны зависеть от модулей низкого уровня; оба должны зависеть от абстракций.`,
			hint1: `Для конструкторов репозиториев вызовите sql.Open с соответствующим именем драйвера и верните новую структуру репозитория с установленным полем db.`,
			hint2: `Для NewUserServiceRefactored верните &UserServiceRefactored{repo: repo}. Для GetUser верните us.repo.FindByID(id). Для SaveUser верните us.repo.Save(user).`,
			whyItMatters: `Принцип инверсии зависимостей — это основа гибкой, тестируемой архитектуры.`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
)

type User struct {
	ID    int
	Name  string
	Email string
}

type UserRepository interface {
	FindByID(id int) (*User, error)
	Save(user *User) error
}

type MySQLRepository struct {
	ConnectionString string
	db               *sql.DB
}

func NewMySQLRepository(connectionString string) *MySQLRepository {
	db, _ := sql.Open("mysql", connectionString)
	return &MySQLRepository{ConnectionString: connectionString, db: db}
}

func (mr *MySQLRepository) FindByID(id int) (*User, error) {
	row := mr.db.QueryRow("SELECT id, name, email FROM users WHERE id = ?", id)
	user := &User{}
	err := row.Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		return nil, err
	}
	fmt.Printf("MySQL: Получен пользователь %d\\n", id)
	return user, nil
}

func (mr *MySQLRepository) Save(user *User) error {
	_, err := mr.db.Exec("INSERT INTO users (name, email) VALUES (?, ?)", user.Name, user.Email)
	if err != nil {
		return err
	}
	fmt.Printf("MySQL: Сохранён пользователь %s\\n", user.Name)
	return nil
}

type UserServiceRefactored struct {
	repo UserRepository
}

func NewUserServiceRefactored(repo UserRepository) *UserServiceRefactored {
	return &UserServiceRefactored{repo: repo}
}

func (us *UserServiceRefactored) GetUser(id int) (*User, error) {
	return us.repo.FindByID(id)
}

func (us *UserServiceRefactored) SaveUser(user *User) error {
	return us.repo.Save(user)
}`
		},
		uz: {
			title: 'Bog\'liqlik inversiyasi printsipi',
			description: `Bog'liqlik inversiyasi prinsipini (DIP) amalga oshiring - yuqori darajadagi modullar past darajadagi modullarga bog'liq bo'lmasligi kerak; ikkalasi ham abstraktsiyalarga bog'liq bo'lishi kerak.`,
			hint1: `Repositoriy konstruktorlari uchun tegishli drayvyer nomi bilan sql.Open ni chaqiring va db maydonini o'rnatilgan yangi repositoriy strukturasini qaytaring.`,
			hint2: `NewUserServiceRefactored uchun &UserServiceRefactored{repo: repo} ni qaytaring. GetUser uchun us.repo.FindByID(id) ni qaytaring. SaveUser uchun us.repo.Save(user) ni qaytaring.`,
			whyItMatters: `Bog'liqlik inversiyasi printsipi moslashuvchan, test qilinadigan arxitekturaning asosi.`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
)

type User struct {
	ID    int
	Name  string
	Email string
}

type UserRepository interface {
	FindByID(id int) (*User, error)
	Save(user *User) error
}

type MySQLRepository struct {
	ConnectionString string
	db               *sql.DB
}

func NewMySQLRepository(connectionString string) *MySQLRepository {
	db, _ := sql.Open("mysql", connectionString)
	return &MySQLRepository{ConnectionString: connectionString, db: db}
}

func (mr *MySQLRepository) FindByID(id int) (*User, error) {
	row := mr.db.QueryRow("SELECT id, name, email FROM users WHERE id = ?", id)
	user := &User{}
	err := row.Scan(&user.ID, &user.Name, &user.Email)
	if err != nil {
		return nil, err
	}
	fmt.Printf("MySQL: Foydalanuvchi %d olindi\\n", id)
	return user, nil
}

func (mr *MySQLRepository) Save(user *User) error {
	_, err := mr.db.Exec("INSERT INTO users (name, email) VALUES (?, ?)", user.Name, user.Email)
	if err != nil {
		return err
	}
	fmt.Printf("MySQL: Foydalanuvchi %s saqlandi\\n", user.Name)
	return nil
}

type UserServiceRefactored struct {
	repo UserRepository
}

func NewUserServiceRefactored(repo UserRepository) *UserServiceRefactored {
	return &UserServiceRefactored{repo: repo}
}

func (us *UserServiceRefactored) GetUser(id int) (*User, error) {
	return us.repo.FindByID(id)
}

func (us *UserServiceRefactored) SaveUser(user *User) error {
	return us.repo.Save(user)
}`
		}
	}
};

export default task;
