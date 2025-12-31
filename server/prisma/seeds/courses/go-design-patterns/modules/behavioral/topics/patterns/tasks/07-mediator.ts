import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-mediator',
	title: 'Mediator Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'behavioral', 'mediator'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Mediator pattern in Go - define an object that encapsulates how objects interact.

**You will implement:**

1. **ChatMediator interface** - SendMessage, AddUser
2. **ChatRoom** - Concrete mediator
3. **User** - Colleague that uses mediator

**Example Usage:**

\`\`\`go
chatRoom := NewChatRoom()	// create mediator
user1 := NewUser("Alice", chatRoom)	// create user and register with mediator
user2 := NewUser("Bob", chatRoom)	// create another user
user3 := NewUser("Charlie", chatRoom)	// third user joins

user1.Send("Hello everyone!")	// Alice sends message through mediator
// Bob receives: "[Alice]: Hello everyone!"
// Charlie receives: "[Alice]: Hello everyone!"
// Alice does NOT receive her own message

user2.Send("Hi Alice!")	// Bob replies
// Alice receives: "[Bob]: Hi Alice!"
// Charlie receives: "[Bob]: Hi Alice!"

// Users don't communicate directly - all goes through ChatRoom
fmt.Println(user1.GetMessages())	// ["[Bob]: Hi Alice!"]
fmt.Println(user2.GetMessages())	// ["[Alice]: Hello everyone!"]
\`\`\``,
	initialCode: `package patterns

import "fmt"

type ChatMediator interface {
}

type User struct {
	name     string
	mediator ChatMediator
}

func NewUser(name string, mediator ChatMediator) *User {
	}
}

func (u *User) Send(message string) {
}

func (u *User) Receive(message string, sender string) {
}

func (u *User) GetMessages() []string {
}

func (u *User) GetName() string {
}

type ChatRoom struct {
	users []*User
}

func NewChatRoom() *ChatRoom {
}

func (c *ChatRoom) AddUser(user *User) {
}

func (c *ChatRoom) SendMessage(message string, sender *User) {
}`,
	solutionCode: `package patterns

import "fmt"

type ChatMediator interface {	// mediator contract
	SendMessage(message string, sender *User)	// broadcast message from sender
	AddUser(user *User)	// register user with mediator
}

type User struct {	// colleague - communicates through mediator
	name     string	// user identifier
	mediator ChatMediator	// reference to mediator
	messages []string	// received messages storage
}

func NewUser(name string, mediator ChatMediator) *User {	// factory constructor
	user := &User{
		name:     name,	// store user name
		mediator: mediator,	// store mediator reference
		messages: make([]string, 0),	// initialize empty message list
	}
	mediator.AddUser(user)	// register with mediator on creation
	return user
}

func (u *User) Send(message string) {	// send message to all other users
	u.mediator.SendMessage(message, u)	// delegate to mediator - no direct communication
}

func (u *User) Receive(message string, sender string) {	// called by mediator
	u.messages = append(u.messages, fmt.Sprintf("[%s]: %s", sender, message))	// format and store
}

func (u *User) GetMessages() []string {	// retrieve received messages
	return u.messages	// return message history
}

func (u *User) GetName() string {	// get user identifier
	return u.name	// return name
}

type ChatRoom struct {	// concrete mediator
	users []*User	// registered users
}

func NewChatRoom() *ChatRoom {	// factory constructor
	return &ChatRoom{users: make([]*User, 0)}	// initialize with empty user list
}

func (c *ChatRoom) AddUser(user *User) {	// register user
	c.users = append(c.users, user)	// add to user list
}

func (c *ChatRoom) SendMessage(message string, sender *User) {	// broadcast message
	for _, user := range c.users {	// iterate all registered users
		if user != sender {	// skip the sender
			user.Receive(message, sender.GetName())	// deliver message to receiver
		}
	}
}`,
	hint1: `**User Methods:**

User.Send delegates to the mediator - users never communicate directly:

\`\`\`go
// Send - delegate to mediator for delivery
func (u *User) Send(message string) {
	u.mediator.SendMessage(message, u)	// mediator handles distribution
}

// Receive - called BY the mediator when message arrives
func (u *User) Receive(message string, sender string) {
	formatted := fmt.Sprintf("[%s]: %s", sender, message)	// format with sender name
	u.messages = append(u.messages, formatted)	// store in message history
}
\`\`\`

The format is exactly: "[SenderName]: MessageContent"`,
	hint2: `**ChatRoom Methods:**

\`\`\`go
// AddUser - register user for message delivery
func (c *ChatRoom) AddUser(user *User) {
	c.users = append(c.users, user)	// add to user list
}

// SendMessage - broadcast to all EXCEPT sender
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {	// iterate all users
		if user != sender {	// IMPORTANT: skip the sender
			user.Receive(message, sender.GetName())	// deliver to others
		}
	}
}
\`\`\`

Key: Use pointer comparison (user != sender) to exclude the sender from receiving their own message.`,
	whyItMatters: `## Why Mediator Pattern Exists

**Problem:** Direct communication creates tight coupling and complex dependencies.

\`\`\`go
// Without Mediator - direct references
type User struct {
    friends []*User	// knows about all other users
}

func (u *User) SendToAll(msg string) {
    for _, friend := range u.friends {	// direct coupling
        friend.Receive(msg)	// must track who to message
    }
}
// Adding new user requires updating all existing users
// Removing user requires cleanup in all other users
// N users = N*(N-1) connections to manage!
\`\`\`

**Solution:** Mediator centralizes communication:

\`\`\`go
// With Mediator - central coordination
type User struct {
    mediator ChatMediator	// only knows mediator
}

func (u *User) Send(msg string) {
    u.mediator.SendMessage(msg, u)	// mediator handles delivery
}
// Adding user = register with mediator
// Removing user = unregister from mediator
// N users = N connections (to mediator)
\`\`\`

---

## Real-World Mediators in Go

**1. Message Brokers:**
- Kafka, RabbitMQ, Redis Pub/Sub
- Publishers and subscribers never communicate directly

**2. Air Traffic Control:**
- Planes don't coordinate with each other
- Tower (mediator) manages all coordination

**3. UI Event Systems:**
- Components emit events to mediator
- Mediator coordinates state updates across components

**4. Service Mesh:**
- Microservices communicate through mesh (Istio, Linkerd)
- Mesh handles routing, retries, load balancing

---

## Production Pattern: Event Bus Mediator

\`\`\`go
package mediator

import (
	"sync"
)

// Event represents a message in the system
type Event struct {
	Type    string	// event type for filtering
	Payload interface{}	// event data
	Source  string	// sender identifier
}

// Subscriber receives events
type Subscriber interface {
	OnEvent(event Event)	// handle received event
	GetID() string	// unique identifier
}

// EventBus is the mediator
type EventBus struct {
	mu          sync.RWMutex	// thread-safe access
	subscribers map[string][]Subscriber	// eventType -> subscribers
	allSubs     []Subscriber	// subscribers to all events
}

func NewEventBus() *EventBus {	// factory constructor
	return &EventBus{
		subscribers: make(map[string][]Subscriber),
		allSubs:     make([]Subscriber, 0),
	}
}

// Subscribe to specific event type
func (eb *EventBus) Subscribe(eventType string, sub Subscriber) {	// register for type
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.subscribers[eventType] = append(eb.subscribers[eventType], sub)
}

// SubscribeAll subscribes to all events
func (eb *EventBus) SubscribeAll(sub Subscriber) {	// register for everything
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.allSubs = append(eb.allSubs, sub)
}

// Unsubscribe removes subscriber from event type
func (eb *EventBus) Unsubscribe(eventType string, sub Subscriber) {	// unregister
	eb.mu.Lock()
	defer eb.mu.Unlock()
	subs := eb.subscribers[eventType]
	for i, s := range subs {
		if s.GetID() == sub.GetID() {
			eb.subscribers[eventType] = append(subs[:i], subs[i+1:]...)
			return
		}
	}
}

// Publish sends event to all matching subscribers
func (eb *EventBus) Publish(event Event) {	// broadcast event
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	// Notify type-specific subscribers
	for _, sub := range eb.subscribers[event.Type] {
		if sub.GetID() != event.Source {	// don't send to source
			go sub.OnEvent(event)	// async delivery
		}
	}

	// Notify all-event subscribers
	for _, sub := range eb.allSubs {
		if sub.GetID() != event.Source {
			go sub.OnEvent(event)
		}
	}
}

// Example subscriber implementation
type Logger struct {
	id   string
	logs []Event
	mu   sync.Mutex
}

func NewLogger(id string) *Logger {	// factory
	return &Logger{id: id, logs: make([]Event, 0)}
}

func (l *Logger) GetID() string {	// implement Subscriber
	return l.id
}

func (l *Logger) OnEvent(event Event) {	// implement Subscriber
	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, event)	// store event
}

func (l *Logger) GetLogs() []Event {	// retrieve logs
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.logs
}

// Usage:
// bus := NewEventBus()
// logger := NewLogger("audit-logger")
// bus.SubscribeAll(logger)
//
// orderService := NewOrderService("orders", bus)
// paymentService := NewPaymentService("payments", bus)
//
// bus.Publish(Event{Type: "order.created", Payload: order, Source: "orders"})
// // paymentService receives event, processes payment
// // logger records all events for audit
\`\`\`

---

## Common Mistakes to Avoid

**1. Direct communication bypassing mediator:**
\`\`\`go
// Wrong - defeats the purpose
func (u *User) SendDirect(other *User, msg string) {
	other.Receive(msg, u.name)	// direct coupling!
}

// Right - always go through mediator
func (u *User) Send(msg string) {
	u.mediator.SendMessage(msg, u)	// mediator decides who gets it
}
\`\`\`

**2. Mediator becoming a "god object":**
\`\`\`go
// Wrong - mediator does everything
type ChatRoom struct {
	users []*User
	// Also handles: authentication, persistence, validation...
}

// Right - mediator only coordinates communication
type ChatRoom struct {
	users []*User	// only coordinates message delivery
}
// Use separate services for auth, persistence, etc.
\`\`\`

**3. Forgetting to exclude sender:**
\`\`\`go
// Wrong - sender gets their own message
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		user.Receive(message, sender.GetName())	// everyone including sender!
	}
}

// Right - skip the sender
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		if user != sender {	// IMPORTANT: exclude sender
			user.Receive(message, sender.GetName())
		}
	}
}
\`\`\``,
	order: 6,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: NewChatRoom creates mediator
func Test1(t *testing.T) {
	room := NewChatRoom()
	if room == nil {
		t.Error("NewChatRoom should return non-nil")
	}
}

// Test2: NewUser registers with mediator
func Test2(t *testing.T) {
	room := NewChatRoom()
	user := NewUser("Alice", room)
	if user.GetName() != "Alice" {
		t.Error("User should have correct name")
	}
}

// Test3: Send broadcasts to others
func Test3(t *testing.T) {
	room := NewChatRoom()
	alice := NewUser("Alice", room)
	bob := NewUser("Bob", room)
	alice.Send("Hello")
	if len(bob.GetMessages()) != 1 {
		t.Error("Bob should receive message")
	}
}

// Test4: Sender does not receive own message
func Test4(t *testing.T) {
	room := NewChatRoom()
	alice := NewUser("Alice", room)
	NewUser("Bob", room)
	alice.Send("Hello")
	if len(alice.GetMessages()) != 0 {
		t.Error("Sender should not receive own message")
	}
}

// Test5: Message format is correct
func Test5(t *testing.T) {
	room := NewChatRoom()
	alice := NewUser("Alice", room)
	bob := NewUser("Bob", room)
	alice.Send("Hi there")
	msgs := bob.GetMessages()
	if !strings.Contains(msgs[0], "[Alice]") || !strings.Contains(msgs[0], "Hi there") {
		t.Error("Message format should be [sender]: message")
	}
}

// Test6: Multiple users receive messages
func Test6(t *testing.T) {
	room := NewChatRoom()
	alice := NewUser("Alice", room)
	bob := NewUser("Bob", room)
	charlie := NewUser("Charlie", room)
	alice.Send("Hello everyone")
	if len(bob.GetMessages()) != 1 || len(charlie.GetMessages()) != 1 {
		t.Error("All other users should receive message")
	}
}

// Test7: GetMessages returns all received messages
func Test7(t *testing.T) {
	room := NewChatRoom()
	alice := NewUser("Alice", room)
	bob := NewUser("Bob", room)
	alice.Send("First")
	alice.Send("Second")
	if len(bob.GetMessages()) != 2 {
		t.Error("Should have 2 messages")
	}
}

// Test8: GetName returns user name
func Test8(t *testing.T) {
	room := NewChatRoom()
	user := NewUser("TestUser", room)
	if user.GetName() != "TestUser" {
		t.Error("GetName should return correct name")
	}
}

// Test9: User Receive stores formatted message
func Test9(t *testing.T) {
	room := NewChatRoom()
	user := NewUser("Test", room)
	user.Receive("Hello", "Sender")
	msgs := user.GetMessages()
	if len(msgs) != 1 || !strings.Contains(msgs[0], "[Sender]") {
		t.Error("Receive should store formatted message")
	}
}

// Test10: Empty chat room user has no messages
func Test10(t *testing.T) {
	room := NewChatRoom()
	user := NewUser("Solo", room)
	if len(user.GetMessages()) != 0 {
		t.Error("New user should have no messages")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Mediator (Посредник)',
			description: `Реализуйте паттерн Mediator на Go — определите объект, инкапсулирующий взаимодействие объектов.

**Вы реализуете:**

1. **Интерфейс ChatMediator** - SendMessage, AddUser
2. **ChatRoom** - Конкретный посредник
3. **User** - Коллега, использующий посредника

**Пример использования:**

\`\`\`go
chatRoom := NewChatRoom()	// создать посредника
user1 := NewUser("Alice", chatRoom)	// создать пользователя и зарегистрировать
user2 := NewUser("Bob", chatRoom)	// создать другого пользователя
user3 := NewUser("Charlie", chatRoom)	// третий присоединяется

user1.Send("Hello everyone!")	// Alice отправляет через посредника
// Bob получает: "[Alice]: Hello everyone!"
// Charlie получает: "[Alice]: Hello everyone!"
// Alice НЕ получает своё сообщение

user2.Send("Hi Alice!")	// Bob отвечает
// Alice получает: "[Bob]: Hi Alice!"
// Charlie получает: "[Bob]: Hi Alice!"

// Пользователи не общаются напрямую - всё через ChatRoom
fmt.Println(user1.GetMessages())	// ["[Bob]: Hi Alice!"]
fmt.Println(user2.GetMessages())	// ["[Alice]: Hello everyone!"]
\`\`\``,
			hint1: `**Методы User:**

User.Send делегирует посреднику - пользователи никогда не общаются напрямую:

\`\`\`go
// Send - делегировать посреднику для доставки
func (u *User) Send(message string) {
	u.mediator.SendMessage(message, u)	// посредник занимается распределением
}

// Receive - вызывается ПОСРЕДНИКОМ когда приходит сообщение
func (u *User) Receive(message string, sender string) {
	formatted := fmt.Sprintf("[%s]: %s", sender, message)	// форматировать с именем отправителя
	u.messages = append(u.messages, formatted)	// сохранить в истории
}
\`\`\`

Формат точно такой: "[ИмяОтправителя]: СодержимоеСообщения"`,
			hint2: `**Методы ChatRoom:**

\`\`\`go
// AddUser - зарегистрировать пользователя для доставки сообщений
func (c *ChatRoom) AddUser(user *User) {
	c.users = append(c.users, user)	// добавить в список
}

// SendMessage - разослать всем КРОМЕ отправителя
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {	// перебрать всех пользователей
		if user != sender {	// ВАЖНО: пропустить отправителя
			user.Receive(message, sender.GetName())	// доставить остальным
		}
	}
}
\`\`\`

Ключ: Используйте сравнение указателей (user != sender) чтобы исключить отправителя из получателей.`,
			whyItMatters: `## Зачем нужен паттерн Mediator

**Проблема:** Прямое общение создаёт тесную связанность и сложные зависимости.

\`\`\`go
// Без Mediator - прямые ссылки
type User struct {
    friends []*User	// знает обо всех других пользователях
}

func (u *User) SendToAll(msg string) {
    for _, friend := range u.friends {	// прямая связанность
        friend.Receive(msg)	// нужно отслеживать кому отправлять
    }
}
// Добавление нового пользователя требует обновления всех существующих
// Удаление пользователя требует очистки у всех остальных
// N пользователей = N*(N-1) связей для управления!
\`\`\`

**Решение:** Mediator централизует коммуникацию:

\`\`\`go
// С Mediator - центральная координация
type User struct {
    mediator ChatMediator	// знает только посредника
}

func (u *User) Send(msg string) {
    u.mediator.SendMessage(msg, u)	// посредник занимается доставкой
}
// Добавление пользователя = регистрация у посредника
// Удаление пользователя = отмена регистрации у посредника
// N пользователей = N связей (с посредником)
\`\`\`

---

## Реальные посредники в Go

**1. Брокеры сообщений:**
- Kafka, RabbitMQ, Redis Pub/Sub
- Издатели и подписчики никогда не общаются напрямую

**2. Управление воздушным движением:**
- Самолёты не координируются друг с другом
- Башня (посредник) управляет всей координацией

**3. UI-системы событий:**
- Компоненты отправляют события посреднику
- Посредник координирует обновления состояния между компонентами

**4. Service Mesh:**
- Микросервисы общаются через mesh (Istio, Linkerd)
- Mesh обрабатывает маршрутизацию, повторы, балансировку

---

## Production-паттерн: Event Bus как посредник

\`\`\`go
package mediator

import (
	"sync"
)

// Event представляет сообщение в системе
type Event struct {
	Type    string	// тип события для фильтрации
	Payload interface{}	// данные события
	Source  string	// идентификатор отправителя
}

// Subscriber получает события
type Subscriber interface {
	OnEvent(event Event)	// обработать полученное событие
	GetID() string	// уникальный идентификатор
}

// EventBus - посредник
type EventBus struct {
	mu          sync.RWMutex	// потокобезопасный доступ
	subscribers map[string][]Subscriber	// eventType -> подписчики
	allSubs     []Subscriber	// подписчики на все события
}

func NewEventBus() *EventBus {	// конструктор-фабрика
	return &EventBus{
		subscribers: make(map[string][]Subscriber),
		allSubs:     make([]Subscriber, 0),
	}
}

// Subscribe на конкретный тип события
func (eb *EventBus) Subscribe(eventType string, sub Subscriber) {	// регистрация на тип
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.subscribers[eventType] = append(eb.subscribers[eventType], sub)
}

// SubscribeAll подписывается на все события
func (eb *EventBus) SubscribeAll(sub Subscriber) {	// регистрация на всё
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.allSubs = append(eb.allSubs, sub)
}

// Unsubscribe удаляет подписчика с типа события
func (eb *EventBus) Unsubscribe(eventType string, sub Subscriber) {	// отмена регистрации
	eb.mu.Lock()
	defer eb.mu.Unlock()
	subs := eb.subscribers[eventType]
	for i, s := range subs {
		if s.GetID() == sub.GetID() {
			eb.subscribers[eventType] = append(subs[:i], subs[i+1:]...)
			return
		}
	}
}

// Publish отправляет событие всем подходящим подписчикам
func (eb *EventBus) Publish(event Event) {	// рассылка события
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	// Уведомить подписчиков конкретного типа
	for _, sub := range eb.subscribers[event.Type] {
		if sub.GetID() != event.Source {	// не отправлять источнику
			go sub.OnEvent(event)	// асинхронная доставка
		}
	}

	// Уведомить подписчиков на все события
	for _, sub := range eb.allSubs {
		if sub.GetID() != event.Source {
			go sub.OnEvent(event)
		}
	}
}

// Пример реализации подписчика
type Logger struct {
	id   string
	logs []Event
	mu   sync.Mutex
}

func NewLogger(id string) *Logger {	// фабрика
	return &Logger{id: id, logs: make([]Event, 0)}
}

func (l *Logger) GetID() string {	// реализация Subscriber
	return l.id
}

func (l *Logger) OnEvent(event Event) {	// реализация Subscriber
	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, event)	// сохранить событие
}

func (l *Logger) GetLogs() []Event {	// получить логи
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.logs
}

// Использование:
// bus := NewEventBus()
// logger := NewLogger("audit-logger")
// bus.SubscribeAll(logger)
//
// orderService := NewOrderService("orders", bus)
// paymentService := NewPaymentService("payments", bus)
//
// bus.Publish(Event{Type: "order.created", Payload: order, Source: "orders"})
// // paymentService получает событие, обрабатывает платёж
// // logger записывает все события для аудита
\`\`\`

---

## Распространённые ошибки

**1. Прямое общение в обход посредника:**
\`\`\`go
// Неправильно - противоречит цели
func (u *User) SendDirect(other *User, msg string) {
	other.Receive(msg, u.name)	// прямая связанность!
}

// Правильно - всегда через посредника
func (u *User) Send(msg string) {
	u.mediator.SendMessage(msg, u)	// посредник решает кому доставить
}
\`\`\`

**2. Посредник становится "god object":**
\`\`\`go
// Неправильно - посредник делает всё
type ChatRoom struct {
	users []*User
	// Также занимается: аутентификация, персистентность, валидация...
}

// Правильно - посредник только координирует коммуникацию
type ChatRoom struct {
	users []*User	// только координирует доставку сообщений
}
// Используйте отдельные сервисы для auth, персистентности и т.д.
\`\`\`

**3. Забыть исключить отправителя:**
\`\`\`go
// Неправильно - отправитель получает своё сообщение
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		user.Receive(message, sender.GetName())	// все включая отправителя!
	}
}

// Правильно - пропустить отправителя
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		if user != sender {	// ВАЖНО: исключить отправителя
			user.Receive(message, sender.GetName())
		}
	}
}
\`\`\``
		},
		uz: {
			title: 'Mediator (Vositachi) Pattern',
			description: `Go tilida Mediator patternini amalga oshiring — ob'ektlar o'zaro ta'sirini inkapsulyatsiya qiluvchi ob'ektni aniqlang.

**Siz amalga oshirasiz:**

1. **ChatMediator interfeysi** - SendMessage, AddUser
2. **ChatRoom** - Aniq vositachi
3. **User** - Vositachidan foydalanadigan hamkasb

**Foydalanish namunasi:**

\`\`\`go
chatRoom := NewChatRoom()	// vositachi yaratish
user1 := NewUser("Alice", chatRoom)	// foydalanuvchi yaratish va ro'yxatdan o'tkazish
user2 := NewUser("Bob", chatRoom)	// boshqa foydalanuvchi yaratish
user3 := NewUser("Charlie", chatRoom)	// uchinchisi qo'shiladi

user1.Send("Hello everyone!")	// Alice vositachi orqali xabar yuboradi
// Bob qabul qiladi: "[Alice]: Hello everyone!"
// Charlie qabul qiladi: "[Alice]: Hello everyone!"
// Alice o'z xabarini qabul QILMAYDI

user2.Send("Hi Alice!")	// Bob javob beradi
// Alice qabul qiladi: "[Bob]: Hi Alice!"
// Charlie qabul qiladi: "[Bob]: Hi Alice!"

// Foydalanuvchilar to'g'ridan-to'g'ri muloqot qilmaydi - hammasi ChatRoom orqali
fmt.Println(user1.GetMessages())	// ["[Bob]: Hi Alice!"]
fmt.Println(user2.GetMessages())	// ["[Alice]: Hello everyone!"]
\`\`\``,
			hint1: `**User metodlari:**

User.Send vositachiga delegatsiya qiladi - foydalanuvchilar hech qachon to'g'ridan-to'g'ri muloqot qilmaydi:

\`\`\`go
// Send - yetkazish uchun vositachiga delegatsiya qilish
func (u *User) Send(message string) {
	u.mediator.SendMessage(message, u)	// vositachi tarqatishni boshqaradi
}

// Receive - xabar kelganda VOSITACHI tomonidan chaqiriladi
func (u *User) Receive(message string, sender string) {
	formatted := fmt.Sprintf("[%s]: %s", sender, message)	// jo'natuvchi nomi bilan formatlash
	u.messages = append(u.messages, formatted)	// tarixda saqlash
}
\`\`\`

Format aynan shunday: "[Jo'natuvchiNomi]: XabarMatni"`,
			hint2: `**ChatRoom metodlari:**

\`\`\`go
// AddUser - xabar yetkazish uchun foydalanuvchini ro'yxatdan o'tkazish
func (c *ChatRoom) AddUser(user *User) {
	c.users = append(c.users, user)	// ro'yxatga qo'shish
}

// SendMessage - jo'natuvchidan TASHQARI hammaga tarqatish
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {	// barcha foydalanuvchilarni takrorlash
		if user != sender {	// MUHIM: jo'natuvchini o'tkazib yuborish
			user.Receive(message, sender.GetName())	// boshqalarga yetkazish
		}
	}
}
\`\`\`

Kalit: Jo'natuvchini qabul qiluvchilardan chiqarib tashlash uchun pointer solishtirishdan (user != sender) foydalaning.`,
			whyItMatters: `## Mediator Pattern nima uchun kerak

**Muammo:** To'g'ridan-to'g'ri muloqot qattiq bog'lanish va murakkab bog'liqliklarni yaratadi.

\`\`\`go
// Mediator siz - to'g'ridan-to'g'ri havolalar
type User struct {
    friends []*User	// barcha boshqa foydalanuvchilarni biladi
}

func (u *User) SendToAll(msg string) {
    for _, friend := range u.friends {	// to'g'ridan-to'g'ri bog'lanish
        friend.Receive(msg)	// kimga xabar yuborishni kuzatish kerak
    }
}
// Yangi foydalanuvchi qo'shish barcha mavjudlarni yangilashni talab qiladi
// Foydalanuvchini o'chirish barcha boshqalarda tozalashni talab qiladi
// N foydalanuvchi = N*(N-1) ta boshqariladigan ulanish!
\`\`\`

**Yechim:** Mediator muloqotni markazlashtiradi:

\`\`\`go
// Mediator bilan - markaziy koordinatsiya
type User struct {
    mediator ChatMediator	// faqat vositachini biladi
}

func (u *User) Send(msg string) {
    u.mediator.SendMessage(msg, u)	// vositachi yetkazishni boshqaradi
}
// Foydalanuvchi qo'shish = vositachida ro'yxatdan o'tkazish
// Foydalanuvchi o'chirish = vositachidan ro'yxatdan chiqarish
// N foydalanuvchi = N ta ulanish (vositachiga)
\`\`\`

---

## Go da haqiqiy vositachilar

**1. Xabar brokerlari:**
- Kafka, RabbitMQ, Redis Pub/Sub
- Nashriyotchilar va obunachillar hech qachon to'g'ridan-to'g'ri muloqot qilmaydi

**2. Havo harakati boshqaruvi:**
- Samolyotlar bir-biri bilan koordinatsiya qilmaydi
- Minora (vositachi) barcha koordinatsiyani boshqaradi

**3. UI hodisalar tizimlari:**
- Komponentlar hodisalarni vositachiga yuboradi
- Vositachi komponentlar o'rtasida holat yangilanishlarini koordinatsiya qiladi

**4. Service Mesh:**
- Mikroservislar mesh orqali muloqot qiladi (Istio, Linkerd)
- Mesh marshrutizatsiya, qayta urinishlar, yuklarni taqsimlashni boshqaradi

---

## Production pattern: Event Bus vositachi sifatida

\`\`\`go
package mediator

import (
	"sync"
)

// Event tizimda xabarni ifodalaydi
type Event struct {
	Type    string	// filtrlash uchun hodisa turi
	Payload interface{}	// hodisa ma'lumotlari
	Source  string	// jo'natuvchi identifikatori
}

// Subscriber hodisalarni qabul qiladi
type Subscriber interface {
	OnEvent(event Event)	// qabul qilingan hodisani boshqarish
	GetID() string	// noyob identifikator
}

// EventBus - vositachi
type EventBus struct {
	mu          sync.RWMutex	// xavfsiz kirish
	subscribers map[string][]Subscriber	// eventType -> obunachillar
	allSubs     []Subscriber	// barcha hodisalarga obunachillar
}

func NewEventBus() *EventBus {	// fabrika konstruktori
	return &EventBus{
		subscribers: make(map[string][]Subscriber),
		allSubs:     make([]Subscriber, 0),
	}
}

// Subscribe aniq hodisa turiga
func (eb *EventBus) Subscribe(eventType string, sub Subscriber) {	// turga ro'yxatdan o'tish
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.subscribers[eventType] = append(eb.subscribers[eventType], sub)
}

// SubscribeAll barcha hodisalarga obuna bo'ladi
func (eb *EventBus) SubscribeAll(sub Subscriber) {	// hammaga ro'yxatdan o'tish
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.allSubs = append(eb.allSubs, sub)
}

// Unsubscribe obunachini hodisa turidan o'chiradi
func (eb *EventBus) Unsubscribe(eventType string, sub Subscriber) {	// ro'yxatdan chiqarish
	eb.mu.Lock()
	defer eb.mu.Unlock()
	subs := eb.subscribers[eventType]
	for i, s := range subs {
		if s.GetID() == sub.GetID() {
			eb.subscribers[eventType] = append(subs[:i], subs[i+1:]...)
			return
		}
	}
}

// Publish hodisani barcha mos obunachilarga yuboradi
func (eb *EventBus) Publish(event Event) {	// hodisani tarqatish
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	// Aniq turdagi obunachilarga xabar berish
	for _, sub := range eb.subscribers[event.Type] {
		if sub.GetID() != event.Source {	// manbaga yubormaslik
			go sub.OnEvent(event)	// asinxron yetkazish
		}
	}

	// Barcha hodisalar obunachilarga xabar berish
	for _, sub := range eb.allSubs {
		if sub.GetID() != event.Source {
			go sub.OnEvent(event)
		}
	}
}

// Obunachi amalga oshirish namunasi
type Logger struct {
	id   string
	logs []Event
	mu   sync.Mutex
}

func NewLogger(id string) *Logger {	// fabrika
	return &Logger{id: id, logs: make([]Event, 0)}
}

func (l *Logger) GetID() string {	// Subscriber ni amalga oshirish
	return l.id
}

func (l *Logger) OnEvent(event Event) {	// Subscriber ni amalga oshirish
	l.mu.Lock()
	defer l.mu.Unlock()
	l.logs = append(l.logs, event)	// hodisani saqlash
}

func (l *Logger) GetLogs() []Event {	// loglarni olish
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.logs
}

// Foydalanish:
// bus := NewEventBus()
// logger := NewLogger("audit-logger")
// bus.SubscribeAll(logger)
//
// orderService := NewOrderService("orders", bus)
// paymentService := NewPaymentService("payments", bus)
//
// bus.Publish(Event{Type: "order.created", Payload: order, Source: "orders"})
// // paymentService hodisani qabul qiladi, to'lovni qayta ishlaydi
// // logger audit uchun barcha hodisalarni yozib oladi
\`\`\`

---

## Keng tarqalgan xatolar

**1. Vositachini chetlab o'tib to'g'ridan-to'g'ri muloqot:**
\`\`\`go
// Noto'g'ri - maqsadga zid
func (u *User) SendDirect(other *User, msg string) {
	other.Receive(msg, u.name)	// to'g'ridan-to'g'ri bog'lanish!
}

// To'g'ri - har doim vositachi orqali
func (u *User) Send(msg string) {
	u.mediator.SendMessage(msg, u)	// vositachi kimga yetkazishni hal qiladi
}
\`\`\`

**2. Vositachi "god object" ga aylanishi:**
\`\`\`go
// Noto'g'ri - vositachi hamma narsani qiladi
type ChatRoom struct {
	users []*User
	// Shuningdek: autentifikatsiya, ma'lumotlarni saqlash, validatsiya...
}

// To'g'ri - vositachi faqat muloqotni koordinatsiya qiladi
type ChatRoom struct {
	users []*User	// faqat xabar yetkazishni koordinatsiya qiladi
}
// Auth, saqlash va boshqalar uchun alohida servislardan foydalaning.
\`\`\`

**3. Jo'natuvchini chiqarib tashlashni unutish:**
\`\`\`go
// Noto'g'ri - jo'natuvchi o'z xabarini oladi
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		user.Receive(message, sender.GetName())	// jo'natuvchi ham dahil!
	}
}

// To'g'ri - jo'natuvchini o'tkazib yuborish
func (c *ChatRoom) SendMessage(message string, sender *User) {
	for _, user := range c.users {
		if user != sender {	// MUHIM: jo'natuvchini chiqarib tashlash
			user.Receive(message, sender.GetName())
		}
	}
}
\`\`\``
		}
	}
};

export default task;
