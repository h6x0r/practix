import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-controller',
	title: 'Controller',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'controller'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Controller principle - assign the responsibility of handling system events to a controller class that represents the overall system or use case.

**You will implement:**

1. **UserController struct** - Handles user-related operations
2. **RegisterUser(username, email string) (*User, error)** - Handle user registration
3. **LoginUser(username, password string) (*User, error)** - Handle user login
4. **UserService struct** - Domain logic for user operations
5. **User struct** - User entity

**Key Concepts:**
- **Controller**: Coordinates system operations, doesn't do the work itself
- **Delegation**: Controller delegates to domain objects (UserService)
- **Facade**: Controller provides simple interface to complex subsystem

**Example Usage:**

\`\`\`go
service := NewUserService()
controller := NewUserController(service)

// Controller handles the system operation
user, err := controller.RegisterUser("john_doe", "john@example.com")
if err != nil {
    log.Fatal(err)
}

fmt.Println(user.Username) // john_doe
fmt.Println(user.ID)       // 1

// Controller handles login
loggedIn, err := controller.LoginUser("john_doe", "password123")
\`\`\`

**Why Controller?**
- **Single Entry Point**: One place to handle system operations
- **Coordination**: Orchestrates calls to domain objects
- **Reusability**: Same controller logic for different UIs (web, CLI, API)

**Anti-pattern (Don't do this):**
\`\`\`go
// BAD: UI code directly manipulating domain objects
func HandleHTTPRegister(w http.ResponseWriter, r *http.Request) {
    user := &User{} // UI knows about User creation!
    user.ID = generateID() // UI handles ID generation!
    user.Username = r.FormValue("username")
    // Complex validation logic in UI code!
}
\`\`\`

**Constraints:**
- Controller must delegate to UserService for business logic
- Controller handles coordination, not domain logic
- UserService must validate input and manage users`,
	initialCode: `package principles

import (
	"errors"
	"fmt"
)

type User struct {
	ID       int
	Username string
	Email    string
}

type UserService struct {
	users  map[string]*User
	nextID int
}

func NewUserService() *UserService {
	}
}

func (s *UserService) CreateUser(username, email string) (*User, error) {
}

func (s *UserService) FindUser(username string) *User {
}

type UserController struct {
	service *UserService
}

func NewUserController(service *UserService) *UserController {
	}
}

func (c *UserController) RegisterUser(username, email string) (*User, error) {
}

func (c *UserController) LoginUser(username, password string) (*User, error) {
}`,
	solutionCode: `package principles

import (
	"errors"
	"fmt"
)

type User struct {
	ID       int
	Username string
	Email    string
}

type UserService struct {
	users  map[string]*User	// stores users by username
	nextID int	// tracks next available user ID
}

func NewUserService() *UserService {
	return &UserService{
		users:  make(map[string]*User),
		nextID: 1,
	}
}

func (s *UserService) CreateUser(username, email string) (*User, error) {
	// Domain logic: check if user exists
	if _, exists := s.users[username]; exists {
		return nil, fmt.Errorf("username %s already exists", username)
	}

	// Domain logic: create user with proper ID
	user := &User{
		ID:       s.nextID,
		Username: username,
		Email:    email,
	}
	s.nextID++

	// Domain logic: store user
	s.users[username] = user
	return user, nil
}

func (s *UserService) FindUser(username string) *User {
	// Domain logic: lookup user
	return s.users[username]	// returns nil if not found
}

type UserController struct {
	service *UserService	// controller delegates to service
}

func NewUserController(service *UserService) *UserController {
	return &UserController{
		service: service,
	}
}

func (c *UserController) RegisterUser(username, email string) (*User, error) {
	// Controller responsibility: input validation
	if username == "" {
		return nil, errors.New("username is required")
	}
	if email == "" {
		return nil, errors.New("email is required")
	}

	// Controller responsibility: coordinate operation
	// Delegate actual work to domain service
	return c.service.CreateUser(username, email)
}

func (c *UserController) LoginUser(username, password string) (*User, error) {
	// Controller responsibility: input validation
	if username == "" {
		return nil, errors.New("username is required")
	}
	if password == "" {
		return nil, errors.New("password is required")
	}

	// Controller responsibility: coordinate operation
	// Delegate lookup to service
	user := c.service.FindUser(username)
	if user == nil {
		return nil, errors.New("user not found")
	}

	// In real app: verify password hash
	// Simplified for this exercise
	return user, nil
}`,
	hint1: `UserService.CreateUser should check if username exists in s.users map, create User with s.nextID, store in map, increment nextID.`,
	hint2: `Controller methods should validate input first, then call corresponding service methods. RegisterUser calls CreateUser, LoginUser calls FindUser.`,
	whyItMatters: `The Controller principle helps you organize system operations and keep UI logic separate from domain logic.

**Why Controller Matters:**

**1. Separation of Concerns**
Controller handles coordination, domain objects handle business logic:

\`\`\`go
// GOOD: Clear separation
type OrderController struct {
    orderService    *OrderService
    paymentService  *PaymentService
    inventoryService *InventoryService
}

func (c *OrderController) PlaceOrder(userID int, items []CartItem) (*Order, error) {
    // Controller coordinates the operation
    if len(items) == 0 {
        return nil, errors.New("cart is empty")
    }

    // Delegate to domain services
    order, err := c.orderService.CreateOrder(userID, items)
    if err != nil {
        return nil, err
    }

    // Coordinate multiple services
    if err := c.inventoryService.ReserveItems(items); err != nil {
        c.orderService.CancelOrder(order.ID)
        return nil, err
    }

    if err := c.paymentService.ProcessPayment(order); err != nil {
        c.inventoryService.ReleaseItems(items)
        c.orderService.CancelOrder(order.ID)
        return nil, err
    }

    return order, nil
}

// BAD: UI code doing everything
func HTTPPlaceOrder(w http.ResponseWriter, r *http.Request) {
    // UI code shouldn't contain business logic!
    items := parseItems(r)
    order := &Order{Items: items}
    // Direct database calls from UI!
    db.Save(order)
    // Payment logic in UI!
    stripe.Charge(order.Total)
}
\`\`\`

**2. Reusable Across Interfaces**
Same controller works with different UIs:

\`\`\`go
// Controller works with any interface
type ProductController struct {
    service *ProductService
}

func (c *ProductController) SearchProducts(query string, limit int) ([]*Product, error) {
    if query == "" {
        return nil, errors.New("search query required")
    }
    return c.service.Search(query, limit)
}

// HTTP handler uses controller
func HTTPSearchProducts(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    limit := 20
    products, err := controller.SearchProducts(query, limit)
    json.NewEncoder(w).Encode(products)
}

// CLI uses same controller
func CLISearchProducts(query string) {
    products, err := controller.SearchProducts(query, 10)
    for _, p := range products {
        fmt.Printf("%s - $%.2f\n", p.Name, p.Price)
    }
}

// gRPC uses same controller
func (s *Server) SearchProducts(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
    products, err := controller.SearchProducts(req.Query, int(req.Limit))
    return toProto(products), err
}
\`\`\`

**3. Testability**
Easy to test controller with mock services:

\`\`\`go
func TestUserController_RegisterUser(t *testing.T) {
    service := NewUserService()
    controller := NewUserController(service)

    // Test successful registration
    user, err := controller.RegisterUser("test_user", "test@example.com")
    if err != nil {
        t.Fatalf("expected no error, got %v", err)
    }
    if user.Username != "test_user" {
        t.Errorf("expected username test_user, got %s", user.Username)
    }

    // Test validation
    _, err = controller.RegisterUser("", "test@example.com")
    if err == nil {
        t.Error("expected error for empty username")
    }

    // Test duplicate username
    _, err = controller.RegisterUser("test_user", "other@example.com")
    if err == nil {
        t.Error("expected error for duplicate username")
    }
}
\`\`\`

**4. Real-World Example: E-commerce System**
\`\`\`go
type CheckoutController struct {
    cartService     *CartService
    orderService    *OrderService
    paymentService  *PaymentService
    shippingService *ShippingService
    emailService    *EmailService
}

func (c *CheckoutController) Checkout(userID int, paymentInfo PaymentInfo) (*Order, error) {
    // Controller orchestrates complex multi-step operation

    // Step 1: Get cart
    cart, err := c.cartService.GetCart(userID)
    if err != nil || len(cart.Items) == 0 {
        return nil, errors.New("cart is empty")
    }

    // Step 2: Calculate shipping
    shipping, err := c.shippingService.CalculateShipping(cart.Items, paymentInfo.Address)
    if err != nil {
        return nil, fmt.Errorf("shipping calculation failed: %w", err)
    }

    // Step 3: Create order
    order, err := c.orderService.CreateOrder(userID, cart.Items, shipping)
    if err != nil {
        return nil, fmt.Errorf("order creation failed: %w", err)
    }

    // Step 4: Process payment
    payment, err := c.paymentService.Charge(paymentInfo, order.Total)
    if err != nil {
        c.orderService.CancelOrder(order.ID)
        return nil, fmt.Errorf("payment failed: %w", err)
    }

    // Step 5: Confirm order
    order.PaymentID = payment.ID
    order.Status = "confirmed"
    c.orderService.UpdateOrder(order)

    // Step 6: Send confirmation
    c.emailService.SendOrderConfirmation(order)

    // Step 7: Clear cart
    c.cartService.ClearCart(userID)

    return order, nil
}
\`\`\`

**5. Controller vs Service Layer**
- **Controller**: Handles system operations, validates input, coordinates services
- **Service**: Contains business logic, works with domain entities

**Common Mistakes:**
- Putting business logic in controller (should be in service)
- Creating one controller per database table (use case-based controllers instead)
- Making controllers too thin (just pass-through) or too fat (doing everything)

**Rule of Thumb:**
Controller validates input and coordinates, but delegates actual work to domain services and entities.`,
	order: 2,
	testCode: `package principles

import (
	"testing"
)

// Test1: RegisterUser creates user with ID
func Test1(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	user, err := controller.RegisterUser("john", "john@test.com")
	if err != nil || user.ID != 1 {
		t.Error("First user should have ID 1")
	}
}

// Test2: RegisterUser validates empty username
func Test2(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	_, err := controller.RegisterUser("", "email@test.com")
	if err == nil {
		t.Error("Should error on empty username")
	}
}

// Test3: RegisterUser validates empty email
func Test3(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	_, err := controller.RegisterUser("user", "")
	if err == nil {
		t.Error("Should error on empty email")
	}
}

// Test4: LoginUser finds existing user
func Test4(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	controller.RegisterUser("testuser", "test@test.com")
	user, err := controller.LoginUser("testuser", "password")
	if err != nil || user.Username != "testuser" {
		t.Error("Should find registered user")
	}
}

// Test5: LoginUser returns error for non-existent user
func Test5(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	_, err := controller.LoginUser("nonexistent", "password")
	if err == nil {
		t.Error("Should error for non-existent user")
	}
}

// Test6: LoginUser validates empty username
func Test6(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	_, err := controller.LoginUser("", "password")
	if err == nil {
		t.Error("Should error on empty username")
	}
}

// Test7: LoginUser validates empty password
func Test7(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	_, err := controller.LoginUser("user", "")
	if err == nil {
		t.Error("Should error on empty password")
	}
}

// Test8: Duplicate username returns error
func Test8(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	controller.RegisterUser("duplicate", "first@test.com")
	_, err := controller.RegisterUser("duplicate", "second@test.com")
	if err == nil {
		t.Error("Should error on duplicate username")
	}
}

// Test9: User struct has correct fields
func Test9(t *testing.T) {
	user := User{ID: 1, Username: "test", Email: "test@test.com"}
	if user.ID != 1 || user.Username != "test" || user.Email != "test@test.com" {
		t.Error("User struct fields not set correctly")
	}
}

// Test10: Multiple users get incrementing IDs
func Test10(t *testing.T) {
	service := NewUserService()
	controller := NewUserController(service)
	user1, _ := controller.RegisterUser("user1", "u1@test.com")
	user2, _ := controller.RegisterUser("user2", "u2@test.com")
	if user1.ID != 1 || user2.ID != 2 {
		t.Error("User IDs should increment")
	}
}
`,
	translations: {
		ru: {
			title: 'Контроллер',
			description: `Реализуйте принцип Контроллера — назначьте ответственность за обработку системных событий классу-контроллеру, представляющему систему в целом или вариант использования.

**Вы реализуете:**

1. **UserController struct** — Обрабатывает операции с пользователями
2. **RegisterUser(username, email string) (*User, error)** — Обработка регистрации пользователя
3. **LoginUser(username, password string) (*User, error)** — Обработка входа пользователя
4. **UserService struct** — Доменная логика для операций с пользователями
5. **User struct** — Сущность пользователя

**Ключевые концепции:**
- **Контроллер**: Координирует системные операции, сам не выполняет работу
- **Делегирование**: Контроллер делегирует доменным объектам (UserService)
- **Фасад**: Контроллер предоставляет простой интерфейс к сложной подсистеме

**Пример использования:**

\`\`\`go
service := NewUserService()
controller := NewUserController(service)

// Контроллер обрабатывает системную операцию
user, err := controller.RegisterUser("john_doe", "john@example.com")
if err != nil {
    log.Fatal(err)
}

fmt.Println(user.Username) // john_doe
fmt.Println(user.ID)       // 1

// Контроллер обрабатывает вход
loggedIn, err := controller.LoginUser("john_doe", "password123")
\`\`\`

**Зачем нужен Контроллер?**
- **Единая точка входа**: Одно место для обработки системных операций
- **Координация**: Организует вызовы доменных объектов
- **Переиспользуемость**: Одна логика контроллера для разных UI (web, CLI, API)

**Ограничения:**
- Контроллер должен делегировать бизнес-логику UserService
- Контроллер обрабатывает координацию, а не доменную логику
- UserService должен валидировать ввод и управлять пользователями`,
			hint1: `UserService.CreateUser должен проверить существование username в map s.users, создать User с s.nextID, сохранить в map, увеличить nextID.`,
			hint2: `Методы контроллера должны сначала валидировать ввод, затем вызывать соответствующие методы сервиса. RegisterUser вызывает CreateUser, LoginUser вызывает FindUser.`,
			whyItMatters: `Принцип Контроллера помогает организовать системные операции и отделить логику UI от доменной логики.

**Почему Контроллер важен:**

**1. Разделение ответственности**
Контроллер обрабатывает координацию, доменные объекты обрабатывают бизнес-логику.

**Распространённые ошибки:**
- Размещение бизнес-логики в контроллере (должна быть в сервисе)
- Создание одного контроллера на таблицу БД (используйте контроллеры на основе вариантов использования)
- Слишком тонкие контроллеры (просто проксирование) или слишком толстые (делают всё)`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
)

type User struct {
	ID       int
	Username string
	Email    string
}

type UserService struct {
	users  map[string]*User	// хранит пользователей по username
	nextID int	// отслеживает следующий доступный ID пользователя
}

func NewUserService() *UserService {
	return &UserService{
		users:  make(map[string]*User),
		nextID: 1,
	}
}

func (s *UserService) CreateUser(username, email string) (*User, error) {
	// Доменная логика: проверка существования пользователя
	if _, exists := s.users[username]; exists {
		return nil, fmt.Errorf("username %s already exists", username)
	}

	// Доменная логика: создание пользователя с правильным ID
	user := &User{
		ID:       s.nextID,
		Username: username,
		Email:    email,
	}
	s.nextID++

	// Доменная логика: сохранение пользователя
	s.users[username] = user
	return user, nil
}

func (s *UserService) FindUser(username string) *User {
	// Доменная логика: поиск пользователя
	return s.users[username]	// возвращает nil если не найден
}

type UserController struct {
	service *UserService	// контроллер делегирует сервису
}

func NewUserController(service *UserService) *UserController {
	return &UserController{
		service: service,
	}
}

func (c *UserController) RegisterUser(username, email string) (*User, error) {
	// Ответственность контроллера: валидация ввода
	if username == "" {
		return nil, errors.New("username is required")
	}
	if email == "" {
		return nil, errors.New("email is required")
	}

	// Ответственность контроллера: координация операции
	// Делегируем реальную работу доменному сервису
	return c.service.CreateUser(username, email)
}

func (c *UserController) LoginUser(username, password string) (*User, error) {
	// Ответственность контроллера: валидация ввода
	if username == "" {
		return nil, errors.New("username is required")
	}
	if password == "" {
		return nil, errors.New("password is required")
	}

	// Ответственность контроллера: координация операции
	// Делегируем поиск сервису
	user := c.service.FindUser(username)
	if user == nil {
		return nil, errors.New("user not found")
	}

	return user, nil
}`
		},
		uz: {
			title: 'Controller (Boshqaruvchi)',
			description: `Controller prinsipini amalga oshiring — tizim hodisalarini boshqarish mas'uliyatini umumiy tizimni yoki foydalanish holatini ifodalovchi controller klassga belgilang.

**Siz amalga oshirasiz:**

1. **UserController struct** — Foydalanuvchi bilan bog'liq operatsiyalarni boshqaradi
2. **RegisterUser(username, email string) (*User, error)** — Foydalanuvchi ro'yxatdan o'tishni boshqarish
3. **LoginUser(username, password string) (*User, error)** — Foydalanuvchi kirishini boshqarish
4. **UserService struct** — Foydalanuvchi operatsiyalari uchun domen mantiqi
5. **User struct** — Foydalanuvchi ob'ekti

**Asosiy tushunchalar:**
- **Controller**: Tizim operatsiyalarini koordinatsiya qiladi, ishni o'zi bajarmaydi
- **Delegatsiya**: Controller domen ob'ektlariga (UserService) topshiradi
- **Fasad**: Controller murakkab quyi tizimga oddiy interfeys taqdim etadi

**Foydalanish misoli:**

\`\`\`go
service := NewUserService()
controller := NewUserController(service)

// Controller tizim operatsiyasini boshqaradi
user, err := controller.RegisterUser("john_doe", "john@example.com")
if err != nil {
    log.Fatal(err)
}

fmt.Println(user.Username) // john_doe
fmt.Println(user.ID)       // 1

// Controller kirishni boshqaradi
loggedIn, err := controller.LoginUser("john_doe", "password123")
\`\`\`

**Nima uchun Controller?**
- **Yagona kirish nuqtasi**: Tizim operatsiyalarini boshqarish uchun bitta joy
- **Koordinatsiya**: Domen ob'ektlariga chaqiruvlarni tashkil qiladi
- **Qayta ishlatish**: Turli UI lar uchun bir xil controller mantiqi (web, CLI, API)

**Cheklovlar:**
- Controller biznes mantiqini UserService ga topshirishi kerak
- Controller koordinatsiyani boshqaradi, domen mantiqini emas
- UserService kiritilgan ma'lumotni tasdiqlashi va foydalanuvchilarni boshqarishi kerak`,
			hint1: `UserService.CreateUser s.users map da username mavjudligini tekshirishi, s.nextID bilan User yaratishi, map ga saqlashi, nextID ni oshirishi kerak.`,
			hint2: `Controller metodlari avval kiritilgan ma'lumotni tasdiqlashi, keyin mos service metodlarini chaqirishi kerak. RegisterUser CreateUser ni chaqiradi, LoginUser FindUser ni chaqiradi.`,
			whyItMatters: `Controller printsipi tizim operatsiyalarini tashkil qilishga va UI mantiqini domen mantiqidan ajratishga yordam beradi.

**Controller nima uchun muhim:**

**1. Mas'uliyatlarni ajratish**
Controller koordinatsiyani boshqaradi, domen ob'ektlari biznes mantiqini boshqaradi.

**Umumiy xatolar:**
- Biznes mantiqini controller da joylashtirish (service da bo'lishi kerak)
- Har bir ma'lumotlar bazasi jadvali uchun bitta controller yaratish (foydalanish holatiga asoslangan controllerlar ishlating)
- Juda yupqa controllerlar (faqat proxylash) yoki juda qalin (hamma narsani qilish)`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
)

type User struct {
	ID       int
	Username string
	Email    string
}

type UserService struct {
	users  map[string]*User	// foydalanuvchilarni username bo'yicha saqlaydi
	nextID int	// keyingi mavjud foydalanuvchi ID ni kuzatadi
}

func NewUserService() *UserService {
	return &UserService{
		users:  make(map[string]*User),
		nextID: 1,
	}
}

func (s *UserService) CreateUser(username, email string) (*User, error) {
	// Domen mantiqi: foydalanuvchi mavjudligini tekshirish
	if _, exists := s.users[username]; exists {
		return nil, fmt.Errorf("username %s already exists", username)
	}

	// Domen mantiqi: to'g'ri ID bilan foydalanuvchi yaratish
	user := &User{
		ID:       s.nextID,
		Username: username,
		Email:    email,
	}
	s.nextID++

	// Domen mantiqi: foydalanuvchini saqlash
	s.users[username] = user
	return user, nil
}

func (s *UserService) FindUser(username string) *User {
	// Domen mantiqi: foydalanuvchini qidirish
	return s.users[username]	// topilmasa nil qaytaradi
}

type UserController struct {
	service *UserService	// controller service ga topshiradi
}

func NewUserController(service *UserService) *UserController {
	return &UserController{
		service: service,
	}
}

func (c *UserController) RegisterUser(username, email string) (*User, error) {
	// Controller mas'uliyati: kiritilgan ma'lumotni tasdiqlash
	if username == "" {
		return nil, errors.New("username is required")
	}
	if email == "" {
		return nil, errors.New("email is required")
	}

	// Controller mas'uliyati: operatsiyani koordinatsiya qilish
	// Haqiqiy ishni domen service ga topshiramiz
	return c.service.CreateUser(username, email)
}

func (c *UserController) LoginUser(username, password string) (*User, error) {
	// Controller mas'uliyati: kiritilgan ma'lumotni tasdiqlash
	if username == "" {
		return nil, errors.New("username is required")
	}
	if password == "" {
		return nil, errors.New("password is required")
	}

	// Controller mas'uliyati: operatsiyani koordinatsiya qilish
	// Qidirishni service ga topshiramiz
	user := c.service.FindUser(username)
	if user == nil {
		return nil, errors.New("user not found")
	}

	return user, nil
}`
		}
	}
};

export default task;
