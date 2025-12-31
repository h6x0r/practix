import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-observer',
	title: 'Observer Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'behavioral', 'observer'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Observer pattern in Go - define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified automatically.

The Observer pattern establishes a subscription mechanism where multiple objects (observers) can subscribe to events from another object (subject). When the subject's state changes, it broadcasts notifications to all registered observers without knowing their concrete types.

**You will implement:**

1. **Observer interface** - Common interface with Update(message) method
2. **Subject struct** - Maintains observer list, handles subscribe/unsubscribe/notify
3. **EmailSubscriber struct** - Concrete observer for email notifications
4. **SlackSubscriber struct** - Concrete observer for Slack notifications

**Example Usage:**

\`\`\`go
subject := NewSubject()	// create subject that will broadcast events

emailSub := &EmailSubscriber{email: "user@example.com"}	// email observer
slackSub := &SlackSubscriber{channel: "#alerts"}	// slack observer

subject.Subscribe(emailSub)	// register email observer
subject.Subscribe(slackSub)	// register slack observer

results := subject.Notify("Server is down!")	// broadcast to all
// results[0]: "Email sent to user@example.com: Server is down!"
// results[1]: "Slack message to #alerts: Server is down!"

subject.Unsubscribe(emailSub)	// remove email observer
results2 := subject.Notify("Server recovered")	// only slack receives
// results2[0]: "Slack message to #alerts: Server recovered"
\`\`\``,
	initialCode: `package patterns

type Observer interface {
}

type Subject struct {
	observers []Observer
}

func NewSubject() *Subject {
}

func (s *Subject) Subscribe(observer Observer) {
}

func (s *Subject) Unsubscribe(observer Observer) {
}

func (s *Subject) Notify(message string) []string {
}

type EmailSubscriber struct {
	email string
}

func (e *EmailSubscriber) Update(message string) string {
}

type SlackSubscriber struct {
	channel string
}

func (s *SlackSubscriber) Update(message string) string {
}`,
	solutionCode: `package patterns

import "fmt"	// import for string formatting

// Observer defines the interface for objects that should be notified
type Observer interface {	// all observers must implement this interface
	Update(message string) string	// method called when subject notifies
}

// Subject maintains a list of observers and notifies them
type Subject struct {	// the publisher/broadcaster
	observers []Observer	// slice of registered observers
}

// NewSubject creates a new Subject with empty observer list
func NewSubject() *Subject {	// factory function for subject
	return &Subject{observers: make([]Observer, 0)}	// initialize with empty slice
}

// Subscribe adds an observer to the notification list
func (s *Subject) Subscribe(observer Observer) {	// register new observer
	s.observers = append(s.observers, observer)	// add to slice
}

// Unsubscribe removes an observer from the list
func (s *Subject) Unsubscribe(observer Observer) {	// deregister observer
	for i, obs := range s.observers {	// iterate to find observer
		if obs == observer {	// found matching observer
			s.observers = append(s.observers[:i], s.observers[i+1:]...)	// remove from slice
			return	// exit after removal
		}
	}
}

// Notify sends message to all observers and collects responses
func (s *Subject) Notify(message string) []string {	// broadcast to all observers
	responses := make([]string, 0, len(s.observers))	// pre-allocate for efficiency
	for _, observer := range s.observers {	// iterate through all observers
		response := observer.Update(message)	// call Update on each observer
		responses = append(responses, response)	// collect response
	}
	return responses	// return all responses
}

// EmailSubscriber is a concrete observer for email notifications
type EmailSubscriber struct {	// concrete observer implementation
	email string	// email address to send to
}

// Update handles the notification for email subscriber
func (e *EmailSubscriber) Update(message string) string {	// implements Observer interface
	return fmt.Sprintf("Email sent to %s: %s", e.email, message)	// format email notification
}

// SlackSubscriber is a concrete observer for Slack notifications
type SlackSubscriber struct {	// another concrete observer
	channel string	// Slack channel to post to
}

// Update handles the notification for Slack subscriber
func (sl *SlackSubscriber) Update(message string) string {	// implements Observer interface
	return fmt.Sprintf("Slack message to %s: %s", sl.channel, message)	// format Slack notification
}`,
	hint1: `Subscribe uses append to add the observer to the slice. Unsubscribe needs to iterate through observers, find the matching one by reference comparison, and remove it using slice manipulation: append(observers[:i], observers[i+1:]...).`,
	hint2: `Notify creates a results slice, iterates through all observers calling Update(message) on each, collects the returned strings into the results slice, and returns it. EmailSubscriber and SlackSubscriber just format and return strings using fmt.Sprintf.`,
	whyItMatters: `**Why the Observer Pattern Exists**

Without Observer, you might notify components directly:

\`\`\`go
// Problem: Tight coupling, hard to add new notification types
type NotificationService struct {
    emailService *EmailService
    slackService *SlackService
    smsService   *SMSService  // adding new type requires changing this class
}

func (n *NotificationService) NotifyAll(msg string) {
    n.emailService.Send(msg)  // direct dependency
    n.slackService.Send(msg)  // direct dependency
    n.smsService.Send(msg)    // direct dependency
}
// To add PushNotification, we must modify NotificationService
\`\`\`

With Observer, adding new notification types is easy:

\`\`\`go
// Solution: Loose coupling through interface
type Subject struct {
    observers []Observer  // depends only on interface
}

func (s *Subject) Notify(msg string) {
    for _, obs := range s.observers {
        obs.Update(msg)  // polymorphic call
    }
}
// New notification types just implement Observer interface
// No changes needed to Subject
\`\`\`

**Real-World Observer Examples in Go**

1. **Event Bus for Microservices**:
\`\`\`go
type Event struct {
    Type    string
    Payload interface{}
}

type EventHandler interface {
    HandleEvent(event Event)
}

type EventBus struct {
    handlers map[string][]EventHandler  // event type -> handlers
    mu       sync.RWMutex
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mu.Lock()
    defer eb.mu.Unlock()
    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

func (eb *EventBus) Publish(event Event) {
    eb.mu.RLock()
    handlers := eb.handlers[event.Type]
    eb.mu.RUnlock()

    for _, handler := range handlers {
        go handler.HandleEvent(event)  // async notification
    }
}
\`\`\`

2. **Configuration Change Watcher**:
\`\`\`go
type ConfigObserver interface {
    OnConfigChange(key string, oldValue, newValue interface{})
}

type ConfigManager struct {
    config    map[string]interface{}
    observers []ConfigObserver
    mu        sync.RWMutex
}

func (cm *ConfigManager) Set(key string, value interface{}) {
    cm.mu.Lock()
    oldValue := cm.config[key]
    cm.config[key] = value
    observers := cm.observers
    cm.mu.Unlock()

    for _, obs := range observers {
        obs.OnConfigChange(key, oldValue, value)
    }
}
\`\`\`

**Production Pattern: Real-time Stock Price Notification System**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// PriceUpdate represents a stock price change
type PriceUpdate struct {
    Symbol    string
    OldPrice  float64
    NewPrice  float64
    Timestamp time.Time
}

// PriceObserver interface for receiving price updates
type PriceObserver interface {
    OnPriceChange(update PriceUpdate)
    GetID() string
}

// StockTicker is the subject that publishes price updates
type StockTicker struct {
    prices    map[string]float64
    observers map[string][]PriceObserver  // symbol -> observers
    mu        sync.RWMutex
}

func NewStockTicker() *StockTicker {
    return &StockTicker{
        prices:    make(map[string]float64),
        observers: make(map[string][]PriceObserver),
    }
}

func (st *StockTicker) Subscribe(symbol string, observer PriceObserver) {
    st.mu.Lock()
    defer st.mu.Unlock()
    st.observers[symbol] = append(st.observers[symbol], observer)
}

func (st *StockTicker) Unsubscribe(symbol string, observerID string) {
    st.mu.Lock()
    defer st.mu.Unlock()
    observers := st.observers[symbol]
    for i, obs := range observers {
        if obs.GetID() == observerID {
            st.observers[symbol] = append(observers[:i], observers[i+1:]...)
            return
        }
    }
}

func (st *StockTicker) UpdatePrice(symbol string, newPrice float64) {
    st.mu.Lock()
    oldPrice := st.prices[symbol]
    st.prices[symbol] = newPrice
    observers := st.observers[symbol]
    st.mu.Unlock()

    update := PriceUpdate{
        Symbol:    symbol,
        OldPrice:  oldPrice,
        NewPrice:  newPrice,
        Timestamp: time.Now(),
    }

    // Notify all observers asynchronously
    for _, observer := range observers {
        go observer.OnPriceChange(update)
    }
}

// AlertObserver triggers alerts on significant price changes
type AlertObserver struct {
    id        string
    threshold float64  // percentage change to trigger alert
}

func (a *AlertObserver) OnPriceChange(update PriceUpdate) {
    if update.OldPrice == 0 {
        return
    }
    change := (update.NewPrice - update.OldPrice) / update.OldPrice * 100
    if change > a.threshold || change < -a.threshold {
        fmt.Printf("ALERT [%s]: %s changed %.2f%% ($%.2f -> $%.2f)\n",
            a.id, update.Symbol, change, update.OldPrice, update.NewPrice)
    }
}

func (a *AlertObserver) GetID() string { return a.id }

// LoggingObserver logs all price changes
type LoggingObserver struct {
    id string
}

func (l *LoggingObserver) OnPriceChange(update PriceUpdate) {
    fmt.Printf("LOG [%s]: %s at $%.2f (was $%.2f)\n",
        update.Timestamp.Format("15:04:05"),
        update.Symbol, update.NewPrice, update.OldPrice)
}

func (l *LoggingObserver) GetID() string { return l.id }
\`\`\`

**Common Mistakes to Avoid**

1. **Memory leaks from forgotten unsubscribe** - Always unsubscribe observers when they're no longer needed
2. **Not handling concurrent access** - Use mutex when observers can be modified from multiple goroutines
3. **Synchronous notification blocking** - Consider async notification for long-running observer handlers
4. **Observer modifying subject during notification** - Can cause unexpected behavior or infinite loops
5. **Notification order dependency** - Don't rely on specific order of observer notifications`,
	order: 0,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: NewSubject creates subject with empty observers
func Test1(t *testing.T) {
	s := NewSubject()
	if s == nil {
		t.Error("NewSubject should return non-nil")
	}
}

// Test2: Subscribe adds observer to subject
func Test2(t *testing.T) {
	s := NewSubject()
	e := &EmailSubscriber{email: "test@test.com"}
	s.Subscribe(e)
	results := s.Notify("test")
	if len(results) != 1 {
		t.Error("Should have 1 observer after subscribe")
	}
}

// Test3: Unsubscribe removes observer
func Test3(t *testing.T) {
	s := NewSubject()
	e := &EmailSubscriber{email: "test@test.com"}
	s.Subscribe(e)
	s.Unsubscribe(e)
	results := s.Notify("test")
	if len(results) != 0 {
		t.Error("Should have 0 observers after unsubscribe")
	}
}

// Test4: Notify returns responses from all observers
func Test4(t *testing.T) {
	s := NewSubject()
	s.Subscribe(&EmailSubscriber{email: "a@a.com"})
	s.Subscribe(&SlackSubscriber{channel: "#test"})
	results := s.Notify("Hello")
	if len(results) != 2 {
		t.Errorf("Expected 2 responses, got %d", len(results))
	}
}

// Test5: EmailSubscriber.Update returns correct format
func Test5(t *testing.T) {
	e := &EmailSubscriber{email: "user@example.com"}
	result := e.Update("Server down")
	if !strings.Contains(result, "Email sent to user@example.com") {
		t.Error("Should contain email address")
	}
	if !strings.Contains(result, "Server down") {
		t.Error("Should contain message")
	}
}

// Test6: SlackSubscriber.Update returns correct format
func Test6(t *testing.T) {
	s := &SlackSubscriber{channel: "#alerts"}
	result := s.Update("Alert!")
	if !strings.Contains(result, "Slack message to #alerts") {
		t.Error("Should contain channel name")
	}
	if !strings.Contains(result, "Alert!") {
		t.Error("Should contain message")
	}
}

// Test7: Multiple subscribers receive same message
func Test7(t *testing.T) {
	subj := NewSubject()
	subj.Subscribe(&EmailSubscriber{email: "a@a.com"})
	subj.Subscribe(&EmailSubscriber{email: "b@b.com"})
	results := subj.Notify("Test message")
	for _, r := range results {
		if !strings.Contains(r, "Test message") {
			t.Error("All observers should receive the message")
		}
	}
}

// Test8: Unsubscribe non-existent observer does nothing
func Test8(t *testing.T) {
	s := NewSubject()
	e1 := &EmailSubscriber{email: "a@a.com"}
	e2 := &EmailSubscriber{email: "b@b.com"}
	s.Subscribe(e1)
	s.Unsubscribe(e2) // not subscribed
	results := s.Notify("test")
	if len(results) != 1 {
		t.Error("Should still have 1 observer")
	}
}

// Test9: Empty subject Notify returns empty slice
func Test9(t *testing.T) {
	s := NewSubject()
	results := s.Notify("test")
	if results == nil || len(results) != 0 {
		t.Error("Notify on empty subject should return empty slice")
	}
}

// Test10: Mixed observer types work correctly
func Test10(t *testing.T) {
	s := NewSubject()
	s.Subscribe(&EmailSubscriber{email: "test@test.com"})
	s.Subscribe(&SlackSubscriber{channel: "#dev"})
	results := s.Notify("Mixed test")
	hasEmail := strings.Contains(results[0], "Email")
	hasSlack := strings.Contains(results[1], "Slack")
	if !hasEmail || !hasSlack {
		t.Error("Should have both Email and Slack responses")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Observer (Наблюдатель)',
			description: `Реализуйте паттерн Observer на Go — определите зависимость один-ко-многим между объектами так, чтобы при изменении состояния одного объекта все зависимые уведомлялись автоматически.

Паттерн Observer устанавливает механизм подписки, где несколько объектов (наблюдатели) могут подписаться на события другого объекта (субъект). Когда состояние субъекта изменяется, он рассылает уведомления всем зарегистрированным наблюдателям, не зная их конкретных типов.

**Вы реализуете:**

1. **Интерфейс Observer** — Общий интерфейс с методом Update(message)
2. **Структура Subject** — Управляет списком наблюдателей, обрабатывает подписку/отписку/уведомление
3. **Структура EmailSubscriber** — Конкретный наблюдатель для email-уведомлений
4. **Структура SlackSubscriber** — Конкретный наблюдатель для Slack-уведомлений

**Пример использования:**

\`\`\`go
subject := NewSubject()	// создаём субъект, который будет рассылать события

emailSub := &EmailSubscriber{email: "user@example.com"}	// email наблюдатель
slackSub := &SlackSubscriber{channel: "#alerts"}	// slack наблюдатель

subject.Subscribe(emailSub)	// регистрируем email наблюдателя
subject.Subscribe(slackSub)	// регистрируем slack наблюдателя

results := subject.Notify("Server is down!")	// рассылка всем
// results[0]: "Email sent to user@example.com: Server is down!"
// results[1]: "Slack message to #alerts: Server is down!"

subject.Unsubscribe(emailSub)	// удаляем email наблюдателя
results2 := subject.Notify("Server recovered")	// только slack получает
// results2[0]: "Slack message to #alerts: Server recovered"
\`\`\``,
			hint1: `Subscribe использует append для добавления наблюдателя в срез. Unsubscribe должен перебрать наблюдателей, найти совпадающего по ссылке и удалить его манипуляцией среза: append(observers[:i], observers[i+1:]...).`,
			hint2: `Notify создаёт срез результатов, перебирает всех наблюдателей вызывая Update(message) на каждом, собирает возвращённые строки в срез результатов и возвращает его. EmailSubscriber и SlackSubscriber просто форматируют и возвращают строки используя fmt.Sprintf.`,
			whyItMatters: `**Зачем нужен паттерн Observer**

Без Observer вы можете уведомлять компоненты напрямую:

\`\`\`go
// Проблема: Жёсткая связанность, трудно добавить новые типы уведомлений
type NotificationService struct {
    emailService *EmailService
    slackService *SlackService
    smsService   *SMSService  // добавление нового типа требует изменения этого класса
}

func (n *NotificationService) NotifyAll(msg string) {
    n.emailService.Send(msg)  // прямая зависимость
    n.slackService.Send(msg)  // прямая зависимость
    n.smsService.Send(msg)    // прямая зависимость
}
// Чтобы добавить PushNotification, мы должны изменить NotificationService
\`\`\`

С Observer добавление новых типов уведомлений просто:

\`\`\`go
// Решение: Слабая связанность через интерфейс
type Subject struct {
    observers []Observer  // зависит только от интерфейса
}

func (s *Subject) Notify(msg string) {
    for _, obs := range s.observers {
        obs.Update(msg)  // полиморфный вызов
    }
}
// Новые типы уведомлений просто реализуют интерфейс Observer
// Изменения в Subject не нужны
\`\`\`

**Реальные примеры Observer в Go**

1. **Event Bus для микросервисов**:
\`\`\`go
type Event struct {
    Type    string
    Payload interface{}
}

type EventHandler interface {
    HandleEvent(event Event)
}

type EventBus struct {
    handlers map[string][]EventHandler  // тип события -> обработчики
    mu       sync.RWMutex
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mu.Lock()
    defer eb.mu.Unlock()
    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

func (eb *EventBus) Publish(event Event) {
    eb.mu.RLock()
    handlers := eb.handlers[event.Type]
    eb.mu.RUnlock()

    for _, handler := range handlers {
        go handler.HandleEvent(event)  // асинхронное уведомление
    }
}
\`\`\`

2. **Наблюдатель изменений конфигурации**:
\`\`\`go
type ConfigObserver interface {
    OnConfigChange(key string, oldValue, newValue interface{})
}

type ConfigManager struct {
    config    map[string]interface{}
    observers []ConfigObserver
    mu        sync.RWMutex
}

func (cm *ConfigManager) Set(key string, value interface{}) {
    cm.mu.Lock()
    oldValue := cm.config[key]
    cm.config[key] = value
    observers := cm.observers
    cm.mu.Unlock()

    for _, obs := range observers {
        obs.OnConfigChange(key, oldValue, value)
    }
}
\`\`\`

**Продакшен паттерн: Система уведомлений о ценах акций в реальном времени**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// PriceUpdate представляет изменение цены акции
type PriceUpdate struct {
    Symbol    string
    OldPrice  float64
    NewPrice  float64
    Timestamp time.Time
}

// PriceObserver интерфейс для получения обновлений цен
type PriceObserver interface {
    OnPriceChange(update PriceUpdate)
    GetID() string
}

// StockTicker — субъект, публикующий обновления цен
type StockTicker struct {
    prices    map[string]float64
    observers map[string][]PriceObserver  // символ -> наблюдатели
    mu        sync.RWMutex
}

func NewStockTicker() *StockTicker {
    return &StockTicker{
        prices:    make(map[string]float64),
        observers: make(map[string][]PriceObserver),
    }
}

func (st *StockTicker) Subscribe(symbol string, observer PriceObserver) {
    st.mu.Lock()
    defer st.mu.Unlock()
    st.observers[symbol] = append(st.observers[symbol], observer)
}

func (st *StockTicker) Unsubscribe(symbol string, observerID string) {
    st.mu.Lock()
    defer st.mu.Unlock()
    observers := st.observers[symbol]
    for i, obs := range observers {
        if obs.GetID() == observerID {
            st.observers[symbol] = append(observers[:i], observers[i+1:]...)
            return
        }
    }
}

func (st *StockTicker) UpdatePrice(symbol string, newPrice float64) {
    st.mu.Lock()
    oldPrice := st.prices[symbol]
    st.prices[symbol] = newPrice
    observers := st.observers[symbol]
    st.mu.Unlock()

    update := PriceUpdate{
        Symbol:    symbol,
        OldPrice:  oldPrice,
        NewPrice:  newPrice,
        Timestamp: time.Now(),
    }

    // Уведомляем всех наблюдателей асинхронно
    for _, observer := range observers {
        go observer.OnPriceChange(update)
    }
}

// AlertObserver запускает оповещения при значительных изменениях цены
type AlertObserver struct {
    id        string
    threshold float64  // процент изменения для срабатывания оповещения
}

func (a *AlertObserver) OnPriceChange(update PriceUpdate) {
    if update.OldPrice == 0 {
        return
    }
    change := (update.NewPrice - update.OldPrice) / update.OldPrice * 100
    if change > a.threshold || change < -a.threshold {
        fmt.Printf("ALERT [%s]: %s changed %.2f%% ($%.2f -> $%.2f)\n",
            a.id, update.Symbol, change, update.OldPrice, update.NewPrice)
    }
}

func (a *AlertObserver) GetID() string { return a.id }

// LoggingObserver логирует все изменения цен
type LoggingObserver struct {
    id string
}

func (l *LoggingObserver) OnPriceChange(update PriceUpdate) {
    fmt.Printf("LOG [%s]: %s at $%.2f (was $%.2f)\n",
        update.Timestamp.Format("15:04:05"),
        update.Symbol, update.NewPrice, update.OldPrice)
}

func (l *LoggingObserver) GetID() string { return l.id }
\`\`\`

**Распространённые ошибки**

1. **Утечки памяти из-за забытой отписки** — Всегда отписывайте наблюдателей, когда они больше не нужны
2. **Не обрабатывают конкурентный доступ** — Используйте mutex когда наблюдатели могут изменяться из нескольких горутин
3. **Синхронное уведомление блокирует** — Рассмотрите асинхронное уведомление для долгих обработчиков
4. **Наблюдатель изменяет субъект во время уведомления** — Может вызвать неожиданное поведение или бесконечные циклы
5. **Зависимость от порядка уведомлений** — Не полагайтесь на конкретный порядок уведомления наблюдателей`,
			solutionCode: `package patterns

import "fmt"	// импорт для форматирования строк

// Observer определяет интерфейс для объектов, которые должны быть уведомлены
type Observer interface {	// все наблюдатели должны реализовать этот интерфейс
	Update(message string) string	// метод вызывается при уведомлении субъектом
}

// Subject поддерживает список наблюдателей и уведомляет их
type Subject struct {	// издатель/рассылатель
	observers []Observer	// срез зарегистрированных наблюдателей
}

// NewSubject создаёт новый Subject с пустым списком наблюдателей
func NewSubject() *Subject {	// фабричная функция для субъекта
	return &Subject{observers: make([]Observer, 0)}	// инициализируем пустым срезом
}

// Subscribe добавляет наблюдателя в список уведомлений
func (s *Subject) Subscribe(observer Observer) {	// регистрируем нового наблюдателя
	s.observers = append(s.observers, observer)	// добавляем в срез
}

// Unsubscribe удаляет наблюдателя из списка
func (s *Subject) Unsubscribe(observer Observer) {	// отменяем регистрацию наблюдателя
	for i, obs := range s.observers {	// перебираем для поиска наблюдателя
		if obs == observer {	// нашли совпадающего наблюдателя
			s.observers = append(s.observers[:i], s.observers[i+1:]...)	// удаляем из среза
			return	// выходим после удаления
		}
	}
}

// Notify отправляет сообщение всем наблюдателям и собирает ответы
func (s *Subject) Notify(message string) []string {	// рассылка всем наблюдателям
	responses := make([]string, 0, len(s.observers))	// предварительное выделение для эффективности
	for _, observer := range s.observers {	// перебираем всех наблюдателей
		response := observer.Update(message)	// вызываем Update на каждом наблюдателе
		responses = append(responses, response)	// собираем ответ
	}
	return responses	// возвращаем все ответы
}

// EmailSubscriber — конкретный наблюдатель для email-уведомлений
type EmailSubscriber struct {	// конкретная реализация наблюдателя
	email string	// email-адрес для отправки
}

// Update обрабатывает уведомление для email-подписчика
func (e *EmailSubscriber) Update(message string) string {	// реализует интерфейс Observer
	return fmt.Sprintf("Email sent to %s: %s", e.email, message)	// форматируем email-уведомление
}

// SlackSubscriber — конкретный наблюдатель для Slack-уведомлений
type SlackSubscriber struct {	// другой конкретный наблюдатель
	channel string	// Slack-канал для публикации
}

// Update обрабатывает уведомление для Slack-подписчика
func (sl *SlackSubscriber) Update(message string) string {	// реализует интерфейс Observer
	return fmt.Sprintf("Slack message to %s: %s", sl.channel, message)	// форматируем Slack-уведомление
}`
		},
		uz: {
			title: 'Observer (Kuzatuvchi) Pattern',
			description: `Go tilida Observer patternini amalga oshiring — ob'ektlar orasida bir-ko'p bog'lanishni aniqlang, shunda bir ob'ekt holati o'zgarganda barcha bog'liqlar avtomatik xabardor qilinadi.

Observer patterni obuna mexanizmini o'rnatadi, bu yerda bir nechta ob'ektlar (kuzatuvchilar) boshqa ob'ektning (sub'ekt) hodisalariga obuna bo'lishi mumkin. Sub'ekt holati o'zgarganda, u barcha ro'yxatdan o'tgan kuzatuvchilarga ularning aniq turlarini bilmasdan xabarlar yuboradi.

**Siz amalga oshirasiz:**

1. **Observer interfeysi** — Update(message) metodi bilan umumiy interfeys
2. **Subject struct** — Kuzatuvchilar ro'yxatini boshqaradi, obuna/obunani bekor qilish/xabardor qilishni amalga oshiradi
3. **EmailSubscriber struct** — Email xabarnomalar uchun konkret kuzatuvchi
4. **SlackSubscriber struct** — Slack xabarnomalar uchun konkret kuzatuvchi

**Foydalanish namunasi:**

\`\`\`go
subject := NewSubject()	// hodisalarni tarqatadigan sub'ekt yaratamiz

emailSub := &EmailSubscriber{email: "user@example.com"}	// email kuzatuvchi
slackSub := &SlackSubscriber{channel: "#alerts"}	// slack kuzatuvchi

subject.Subscribe(emailSub)	// email kuzatuvchini ro'yxatdan o'tkazamiz
subject.Subscribe(slackSub)	// slack kuzatuvchini ro'yxatdan o'tkazamiz

results := subject.Notify("Server is down!")	// hammaga tarqatamiz
// results[0]: "Email sent to user@example.com: Server is down!"
// results[1]: "Slack message to #alerts: Server is down!"

subject.Unsubscribe(emailSub)	// email kuzatuvchini olib tashlaymiz
results2 := subject.Notify("Server recovered")	// faqat slack qabul qiladi
// results2[0]: "Slack message to #alerts: Server recovered"
\`\`\``,
			hint1: `Subscribe kuzatuvchini slice ga qo'shish uchun append ishlatadi. Unsubscribe kuzatuvchilarni takrorlashi, havola bo'yicha mos keluvchini topishi va slice manipulyatsiyasi bilan olib tashlashi kerak: append(observers[:i], observers[i+1:]...).`,
			hint2: `Notify natijalar slice ini yaratadi, barcha kuzatuvchilarni takrorlaydi har birida Update(message) ni chaqiradi, qaytarilgan satrlarni natijalar slice iga yig'adi va qaytaradi. EmailSubscriber va SlackSubscriber shunchaki fmt.Sprintf yordamida satrlarni formatlaydi va qaytaradi.`,
			whyItMatters: `**Observer Pattern nima uchun kerak**

Observer siz komponentlarni to'g'ridan-to'g'ri xabardor qilishingiz mumkin:

\`\`\`go
// Muammo: Qattiq bog'lanish, yangi xabarnoma turlarini qo'shish qiyin
type NotificationService struct {
    emailService *EmailService
    slackService *SlackService
    smsService   *SMSService  // yangi tur qo'shish bu sinfni o'zgartirishni talab qiladi
}

func (n *NotificationService) NotifyAll(msg string) {
    n.emailService.Send(msg)  // to'g'ridan-to'g'ri bog'lanish
    n.slackService.Send(msg)  // to'g'ridan-to'g'ri bog'lanish
    n.smsService.Send(msg)    // to'g'ridan-to'g'ri bog'lanish
}
// PushNotification qo'shish uchun NotificationService ni o'zgartirish kerak
\`\`\`

Observer bilan yangi xabarnoma turlarini qo'shish oson:

\`\`\`go
// Yechim: Interfeys orqali zaif bog'lanish
type Subject struct {
    observers []Observer  // faqat interfeysga bog'liq
}

func (s *Subject) Notify(msg string) {
    for _, obs := range s.observers {
        obs.Update(msg)  // polimorf chaqiruv
    }
}
// Yangi xabarnoma turlari shunchaki Observer interfeysini amalga oshiradi
// Subject da o'zgarishlar kerak emas
\`\`\`

**Go da Observer ning real dunyo misollari**

1. **Mikroservislar uchun Event Bus**:
\`\`\`go
type Event struct {
    Type    string
    Payload interface{}
}

type EventHandler interface {
    HandleEvent(event Event)
}

type EventBus struct {
    handlers map[string][]EventHandler  // hodisa turi -> ishlovchilar
    mu       sync.RWMutex
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mu.Lock()
    defer eb.mu.Unlock()
    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

func (eb *EventBus) Publish(event Event) {
    eb.mu.RLock()
    handlers := eb.handlers[event.Type]
    eb.mu.RUnlock()

    for _, handler := range handlers {
        go handler.HandleEvent(event)  // asinxron xabardor qilish
    }
}
\`\`\`

2. **Konfiguratsiya o'zgarishlarini kuzatuvchi**:
\`\`\`go
type ConfigObserver interface {
    OnConfigChange(key string, oldValue, newValue interface{})
}

type ConfigManager struct {
    config    map[string]interface{}
    observers []ConfigObserver
    mu        sync.RWMutex
}

func (cm *ConfigManager) Set(key string, value interface{}) {
    cm.mu.Lock()
    oldValue := cm.config[key]
    cm.config[key] = value
    observers := cm.observers
    cm.mu.Unlock()

    for _, obs := range observers {
        obs.OnConfigChange(key, oldValue, value)
    }
}
\`\`\`

**Prodakshen pattern: Real vaqtda aksiya narxlari xabarnoma tizimi**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// PriceUpdate aksiya narxi o'zgarishini ifodalaydi
type PriceUpdate struct {
    Symbol    string
    OldPrice  float64
    NewPrice  float64
    Timestamp time.Time
}

// PriceObserver narx yangilanishlarini olish uchun interfeys
type PriceObserver interface {
    OnPriceChange(update PriceUpdate)
    GetID() string
}

// StockTicker narx yangilanishlarini nashr etuvchi sub'ekt
type StockTicker struct {
    prices    map[string]float64
    observers map[string][]PriceObserver  // simvol -> kuzatuvchilar
    mu        sync.RWMutex
}

func NewStockTicker() *StockTicker {
    return &StockTicker{
        prices:    make(map[string]float64),
        observers: make(map[string][]PriceObserver),
    }
}

func (st *StockTicker) Subscribe(symbol string, observer PriceObserver) {
    st.mu.Lock()
    defer st.mu.Unlock()
    st.observers[symbol] = append(st.observers[symbol], observer)
}

func (st *StockTicker) Unsubscribe(symbol string, observerID string) {
    st.mu.Lock()
    defer st.mu.Unlock()
    observers := st.observers[symbol]
    for i, obs := range observers {
        if obs.GetID() == observerID {
            st.observers[symbol] = append(observers[:i], observers[i+1:]...)
            return
        }
    }
}

func (st *StockTicker) UpdatePrice(symbol string, newPrice float64) {
    st.mu.Lock()
    oldPrice := st.prices[symbol]
    st.prices[symbol] = newPrice
    observers := st.observers[symbol]
    st.mu.Unlock()

    update := PriceUpdate{
        Symbol:    symbol,
        OldPrice:  oldPrice,
        NewPrice:  newPrice,
        Timestamp: time.Now(),
    }

    // Barcha kuzatuvchilarni asinxron xabardor qilamiz
    for _, observer := range observers {
        go observer.OnPriceChange(update)
    }
}

// AlertObserver sezilarli narx o'zgarishlarida ogohlantirishlarni ishga tushiradi
type AlertObserver struct {
    id        string
    threshold float64  // ogohlantirish uchun foiz o'zgarishi
}

func (a *AlertObserver) OnPriceChange(update PriceUpdate) {
    if update.OldPrice == 0 {
        return
    }
    change := (update.NewPrice - update.OldPrice) / update.OldPrice * 100
    if change > a.threshold || change < -a.threshold {
        fmt.Printf("ALERT [%s]: %s changed %.2f%% ($%.2f -> $%.2f)\n",
            a.id, update.Symbol, change, update.OldPrice, update.NewPrice)
    }
}

func (a *AlertObserver) GetID() string { return a.id }

// LoggingObserver barcha narx o'zgarishlarini loglaydi
type LoggingObserver struct {
    id string
}

func (l *LoggingObserver) OnPriceChange(update PriceUpdate) {
    fmt.Printf("LOG [%s]: %s at $%.2f (was $%.2f)\n",
        update.Timestamp.Format("15:04:05"),
        update.Symbol, update.NewPrice, update.OldPrice)
}

func (l *LoggingObserver) GetID() string { return l.id }
\`\`\`

**Oldini olish kerak bo'lgan keng tarqalgan xatolar**

1. **Unutilgan obunani bekor qilishdan xotira oqishi** — Kuzatuvchilar kerak bo'lmaganda doimo obunani bekor qiling
2. **Parallel kirishni qayta ishlamaslik** — Kuzatuvchilar bir nechta gorutinlardan o'zgartirilishi mumkin bo'lganda mutex ishlating
3. **Sinxron xabardor qilish bloklaydi** — Uzoq ishlaydigan kuzatuvchi ishlovchilar uchun asinxron xabardor qilishni ko'rib chiqing
4. **Kuzatuvchi xabardor qilish vaqtida sub'ektni o'zgartiradi** — Kutilmagan xatti-harakatlar yoki cheksiz sikllarni keltirib chiqarishi mumkin
5. **Xabardor qilish tartibiga bog'liqlik** — Kuzatuvchilarni xabardor qilishning aniq tartibiga tayanmang`,
			solutionCode: `package patterns

import "fmt"	// satrlarni formatlash uchun import

// Observer xabardor qilinishi kerak bo'lgan ob'ektlar uchun interfeysni aniqlaydi
type Observer interface {	// barcha kuzatuvchilar bu interfeysni amalga oshirishi kerak
	Update(message string) string	// sub'ekt xabardor qilganda chaqiriladigan metod
}

// Subject kuzatuvchilar ro'yxatini boshqaradi va ularni xabardor qiladi
type Subject struct {	// nashriyotchi/tarqatuvchi
	observers []Observer	// ro'yxatdan o'tgan kuzatuvchilar slice i
}

// NewSubject bo'sh kuzatuvchilar ro'yxati bilan yangi Subject yaratadi
func NewSubject() *Subject {	// sub'ekt uchun fabrika funksiyasi
	return &Subject{observers: make([]Observer, 0)}	// bo'sh slice bilan initsializatsiya qilamiz
}

// Subscribe kuzatuvchini xabarnoma ro'yxatiga qo'shadi
func (s *Subject) Subscribe(observer Observer) {	// yangi kuzatuvchini ro'yxatdan o'tkazamiz
	s.observers = append(s.observers, observer)	// slice ga qo'shamiz
}

// Unsubscribe kuzatuvchini ro'yxatdan olib tashlaydi
func (s *Subject) Unsubscribe(observer Observer) {	// kuzatuvchi ro'yxatdan chiqaramiz
	for i, obs := range s.observers {	// kuzatuvchini topish uchun takrorlaymiz
		if obs == observer {	// mos keluvchi kuzatuvchi topildi
			s.observers = append(s.observers[:i], s.observers[i+1:]...)	// slice dan olib tashlaymiz
			return	// olib tashlagandan keyin chiqamiz
		}
	}
}

// Notify barcha kuzatuvchilarga xabar yuboradi va javoblarni yig'adi
func (s *Subject) Notify(message string) []string {	// barcha kuzatuvchilarga tarqatish
	responses := make([]string, 0, len(s.observers))	// samaradorlik uchun oldindan joy ajratamiz
	for _, observer := range s.observers {	// barcha kuzatuvchilarni takrorlaymiz
		response := observer.Update(message)	// har bir kuzatuvchida Update ni chaqiramiz
		responses = append(responses, response)	// javobni yig'amiz
	}
	return responses	// barcha javoblarni qaytaramiz
}

// EmailSubscriber email xabarnomalar uchun konkret kuzatuvchi
type EmailSubscriber struct {	// konkret kuzatuvchi amalga oshirilishi
	email string	// yuborish uchun email manzil
}

// Update email obunachi uchun xabarnomani qayta ishlaydi
func (e *EmailSubscriber) Update(message string) string {	// Observer interfeysini amalga oshiradi
	return fmt.Sprintf("Email sent to %s: %s", e.email, message)	// email xabarnomani formatlaymiz
}

// SlackSubscriber Slack xabarnomalar uchun konkret kuzatuvchi
type SlackSubscriber struct {	// boshqa konkret kuzatuvchi
	channel string	// joylash uchun Slack kanali
}

// Update Slack obunachi uchun xabarnomani qayta ishlaydi
func (sl *SlackSubscriber) Update(message string) string {	// Observer interfeysini amalga oshiradi
	return fmt.Sprintf("Slack message to %s: %s", sl.channel, message)	// Slack xabarnomani formatlaymiz
}`
		}
	}
};

export default task;
