import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-bridge',
	title: 'Bridge Pattern',
	difficulty: 'hard',
	tags: ['go', 'design-patterns', 'structural', 'bridge'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Bridge pattern in Go - decouple an abstraction from its implementation so that the two can vary independently.

**You will implement:**

1. **Device interface** - Implementation interface (TV, Radio)
2. **Remote struct** - Abstraction that uses Device
3. **TV struct** - Concrete implementation
4. **Radio struct** - Concrete implementation

**Example Usage:**

\`\`\`go
// Create devices (implementations)
tv := &TV{}
radio := &Radio{}

// Create remotes (abstractions) with different devices
tvRemote := NewRemote(tv)
radioRemote := NewRemote(radio)

// Same remote interface works with different devices
tvRemote.TogglePower()   // "TV is ON"
tvRemote.VolumeUp()      // "TV volume: 1"
tvRemote.ChannelUp()     // "TV channel: 1"

radioRemote.TogglePower() // "Radio is ON"
radioRemote.VolumeUp()    // "Radio volume: 1"

// Can swap devices at runtime
tvRemote.device = radio  // Now controls radio!
\`\`\``,
	initialCode: `package patterns

import "fmt"

type Device interface {
}

type TV struct {
	on      bool
	volume  int
	channel int
}

func (t *TV) IsEnabled() bool {
}

func (t *TV) Enable() {
}

func (t *TV) Disable() {
}

func (t *TV) GetVolume() int {
}

func (t *TV) SetVolume(volume int) {
}

func (t *TV) GetChannel() int {
}

func (t *TV) SetChannel(channel int) {
}

type Radio struct {
	on      bool
	volume  int
	channel int
}

func (r *Radio) IsEnabled() bool {
}

func (r *Radio) Enable() {
}

func (r *Radio) Disable() {
}

func (r *Radio) GetVolume() int {
}

func (r *Radio) SetVolume(volume int) {
}

func (r *Radio) GetChannel() int {
}

func (r *Radio) SetChannel(channel int) {
}

type Remote struct {
	device Device
}

func NewRemote(device Device) *Remote {
}

func (r *Remote) TogglePower() string {
}

func (r *Remote) VolumeUp() string {
}

func (r *Remote) VolumeDown() string {
}

func (r *Remote) ChannelUp() string {
}`,
	solutionCode: `package patterns

import "fmt"

type Device interface {	// Implementor - defines implementation interface
	IsEnabled() bool	// check if device is powered on
	Enable()	// turn device on
	Disable()	// turn device off
	GetVolume() int	// get current volume level
	SetVolume(volume int)	// set volume level
	GetChannel() int	// get current channel
	SetChannel(channel int)	// set channel
	GetName() string	// get device name for display
}

type TV struct {	// Concrete Implementor - TV device
	on      bool	// power state
	volume  int	// current volume
	channel int	// current channel
}

func (t *TV) IsEnabled() bool  { return t.on }	// returns power state
func (t *TV) Enable()          { t.on = true }	// turns TV on
func (t *TV) Disable()         { t.on = false }	// turns TV off
func (t *TV) GetVolume() int   { return t.volume }	// returns current volume
func (t *TV) SetVolume(v int)  { t.volume = v }	// sets volume level
func (t *TV) GetChannel() int  { return t.channel }	// returns current channel
func (t *TV) SetChannel(c int) { t.channel = c }	// sets channel number
func (t *TV) GetName() string  { return "TV" }	// returns device name

type Radio struct {	// Concrete Implementor - Radio device
	on      bool	// power state
	volume  int	// current volume
	channel int	// current frequency/station
}

func (r *Radio) IsEnabled() bool  { return r.on }	// returns power state
func (r *Radio) Enable()          { r.on = true }	// turns Radio on
func (r *Radio) Disable()         { r.on = false }	// turns Radio off
func (r *Radio) GetVolume() int   { return r.volume }	// returns current volume
func (r *Radio) SetVolume(v int)  { r.volume = v }	// sets volume level
func (r *Radio) GetChannel() int  { return r.channel }	// returns current station
func (r *Radio) SetChannel(c int) { r.channel = c }	// sets station number
func (r *Radio) GetName() string  { return "Radio" }	// returns device name

type Remote struct {	// Abstraction - remote control
	device Device	// reference to implementor (the bridge)
}

func NewRemote(device Device) *Remote {	// creates remote with device
	return &Remote{device: device}	// inject implementation
}

func (r *Remote) TogglePower() string {	// abstraction method - toggle power
	if r.device.IsEnabled() {	// check current state via bridge
		r.device.Disable()	// delegate to implementation
		return fmt.Sprintf("%s is OFF", r.device.GetName())	// return status
	}
	r.device.Enable()	// delegate to implementation
	return fmt.Sprintf("%s is ON", r.device.GetName())	// return status
}

func (r *Remote) VolumeUp() string {	// abstraction method - increase volume
	r.device.SetVolume(r.device.GetVolume() + 1)	// delegate through bridge
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// return result
}

func (r *Remote) VolumeDown() string {	// abstraction method - decrease volume
	r.device.SetVolume(r.device.GetVolume() - 1)	// delegate through bridge
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// return result
}

func (r *Remote) ChannelUp() string {	// abstraction method - next channel
	r.device.SetChannel(r.device.GetChannel() + 1)	// delegate through bridge
	return fmt.Sprintf("%s channel: %d", r.device.GetName(), r.device.GetChannel())	// return result
}`,
	hint1: `TV and Radio methods are simple getters/setters. IsEnabled returns the 'on' field, Enable sets it to true, Disable sets it to false. GetVolume/GetChannel return the fields, SetVolume/SetChannel assign values. GetName returns "TV" or "Radio".`,
	hint2: `Remote methods delegate to the device interface. TogglePower checks IsEnabled(), then calls Enable() or Disable(). VolumeUp gets current volume with GetVolume(), adds 1, and sets it with SetVolume(). Use fmt.Sprintf to format the return string with device name and value.`,
	whyItMatters: `**1. Why Bridge Exists**

Bridge pattern solves the problem of "class explosion" when you have multiple dimensions of variation. Without Bridge, M abstractions x N implementations = M*N classes. With Bridge, you only need M + N classes.

**The Problem It Solves:**

\`\`\`go
// WITHOUT Bridge - class explosion
// Each combination needs its own class!

type TVRemote struct{ /* TV-specific */ }
type TVAdvancedRemote struct{ /* TV-specific */ }
type RadioRemote struct{ /* Radio-specific */ }
type RadioAdvancedRemote struct{ /* Radio-specific */ }
type SpeakerRemote struct{ /* Speaker-specific */ }
type SpeakerAdvancedRemote struct{ /* Speaker-specific */ }

// 3 devices x 2 remote types = 6 classes
// Add 1 new device? +2 classes
// Add 1 new remote type? +3 classes
// n devices x m remotes = n*m classes!
\`\`\`

**WITH Bridge:**

\`\`\`go
// Abstraction (Remote) + Implementation (Device) separate hierarchies
// 3 devices + 2 remotes = 5 classes total

// Implementations
type TV struct{}
type Radio struct{}
type Speaker struct{}

// Abstractions (can extend independently)
type Remote struct{ device Device }
type AdvancedRemote struct{ Remote }

// Add new device? +1 class (just implement Device)
// Add new remote? +1 class (just embed Remote)
// n devices + m remotes = n+m classes!
\`\`\`

**2. Real-World Examples in Go**

**Database Driver Bridge:**

\`\`\`go
// Implementation interface - different databases
type DBDriver interface {
    Connect(dsn string) error
    Execute(query string) ([]Row, error)
    Close() error
}

// Concrete implementations
type PostgresDriver struct{}
type MySQLDriver struct{}
type SQLiteDriver struct{}

// Abstraction - query builder
type QueryBuilder struct {
    driver DBDriver  // bridge to implementation
}

func (qb *QueryBuilder) Select(table string, cols ...string) *QueryBuilder {
    // Build query using abstraction logic
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    return qb
}

func (qb *QueryBuilder) Execute() ([]Row, error) {
    query := qb.buildQuery()
    return qb.driver.Execute(query)  // delegate to implementation
}

// Extended abstraction - adds more features
type TransactionalQueryBuilder struct {
    QueryBuilder  // embed base abstraction
}

func (tqb *TransactionalQueryBuilder) BeginTransaction() error {
    return tqb.driver.Execute("BEGIN")
}
\`\`\`

**Rendering Engine Bridge:**

\`\`\`go
// Implementation - rendering APIs
type Renderer interface {
    RenderCircle(x, y, radius float64)
    RenderRectangle(x, y, width, height float64)
}

type OpenGLRenderer struct{}
type VulkanRenderer struct{}
type DirectXRenderer struct{}

// Abstraction - shapes
type Shape struct {
    renderer Renderer  // bridge
}

type Circle struct {
    Shape
    x, y, radius float64
}

func (c *Circle) Draw() {
    c.renderer.RenderCircle(c.x, c.y, c.radius)
}

type Rectangle struct {
    Shape
    x, y, width, height float64
}

func (r *Rectangle) Draw() {
    r.renderer.RenderRectangle(r.x, r.y, r.width, r.height)
}

// Can combine any shape with any renderer!
circle := &Circle{Shape: Shape{renderer: &OpenGLRenderer{}}}
rect := &Rectangle{Shape: Shape{renderer: &VulkanRenderer{}}}
\`\`\`

**3. Production Pattern - Message System**

\`\`\`go
package messaging

import "time"

// Implementation interface - transport protocols
type Transport interface {
    Send(to string, payload []byte) error
    Receive() ([]byte, error)
    Connect() error
    Disconnect() error
}

// Concrete implementations
type HTTPTransport struct {
    baseURL string
    client  *http.Client
}

func (h *HTTPTransport) Send(to string, payload []byte) error {
    _, err := h.client.Post(h.baseURL+"/"+to, "application/json", bytes.NewReader(payload))
    return err
}

type WebSocketTransport struct {
    conn *websocket.Conn
}

func (w *WebSocketTransport) Send(to string, payload []byte) error {
    return w.conn.WriteMessage(websocket.TextMessage, payload)
}

type GRPCTransport struct {
    client pb.MessagingClient
}

func (g *GRPCTransport) Send(to string, payload []byte) error {
    _, err := g.client.Send(context.Background(), &pb.Message{To: to, Data: payload})
    return err
}

// Abstraction - message sender
type MessageSender struct {
    transport Transport  // bridge to transport implementation
}

func NewMessageSender(transport Transport) *MessageSender {
    return &MessageSender{transport: transport}
}

func (m *MessageSender) SendText(to, text string) error {
    payload, _ := json.Marshal(map[string]string{"type": "text", "content": text})
    return m.transport.Send(to, payload)
}

func (m *MessageSender) SendFile(to string, file []byte) error {
    payload, _ := json.Marshal(map[string]interface{}{"type": "file", "data": file})
    return m.transport.Send(to, payload)
}

// Extended abstraction - adds reliability
type ReliableMessageSender struct {
    MessageSender
    maxRetries int
    retryDelay time.Duration
}

func (r *ReliableMessageSender) SendWithRetry(to, text string) error {
    var lastErr error
    for i := 0; i < r.maxRetries; i++ {
        if err := r.SendText(to, text); err == nil {
            return nil
        } else {
            lastErr = err
            time.Sleep(r.retryDelay)
        }
    }
    return lastErr
}

// Usage - any sender + any transport
httpSender := NewMessageSender(&HTTPTransport{baseURL: "https://api.example.com"})
wsSender := NewMessageSender(&WebSocketTransport{})
reliableSender := &ReliableMessageSender{
    MessageSender: *NewMessageSender(&GRPCTransport{}),
    maxRetries:    3,
    retryDelay:    time.Second,
}
\`\`\`

**4. Common Mistakes to Avoid**

\`\`\`go
// MISTAKE 1: Abstraction depends on concrete implementation
type Remote struct {
    tv *TV  // Wrong! Depends on concrete type
}

// CORRECT: Depend on interface
type Remote struct {
    device Device  // Correct! Depends on abstraction
}

// MISTAKE 2: Implementation knows about abstraction
type TV struct {
    remote *Remote  // Wrong! Creates circular dependency
}

// CORRECT: Implementation doesn't know about abstraction
type TV struct {
    on      bool
    volume  int
    // No reference to Remote
}

// MISTAKE 3: Bridge interface too wide
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
    GetChannel() int
    SetChannel(int)
    PlayDVD()        // Wrong! Not all devices have DVD
    TuneAntenna()    // Wrong! Not all devices have antenna
}

// CORRECT: Keep interface minimal and focused
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
}

// Add specific interfaces for specific capabilities
type ChannelDevice interface {
    Device
    GetChannel() int
    SetChannel(int)
}

// MISTAKE 4: Not using the bridge for delegation
func (r *Remote) VolumeUp() string {
    // Wrong! Duplicates implementation logic
    if r.device.GetVolume() < 100 {
        // Volume logic here instead of delegating
    }
}

// CORRECT: Delegate to implementation
func (r *Remote) VolumeUp() string {
    r.device.SetVolume(r.device.GetVolume() + 1)
    return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())
}

// MISTAKE 5: Creating bridge when simple composition works
// Don't use Bridge for single implementation!
type OnlyTVRemote struct {
    device Device  // Overkill if only TV exists
}

// Use Bridge when you have MULTIPLE implementations
// and MULTIPLE abstraction extensions
\`\`\``,
	order: 1,
	testCode: `package patterns

import (
	"testing"
)

// Test1: TV.Enable sets on to true
func Test1(t *testing.T) {
	tv := &TV{}
	tv.Enable()
	if !tv.IsEnabled() {
		t.Error("TV.Enable should set on to true")
	}
}

// Test2: TV.Disable sets on to false
func Test2(t *testing.T) {
	tv := &TV{}
	tv.Enable()
	tv.Disable()
	if tv.IsEnabled() {
		t.Error("TV.Disable should set on to false")
	}
}

// Test3: TV.SetVolume and GetVolume work
func Test3(t *testing.T) {
	tv := &TV{}
	tv.SetVolume(10)
	if tv.GetVolume() != 10 {
		t.Error("TV volume should be 10")
	}
}

// Test4: Remote.TogglePower returns correct message
func Test4(t *testing.T) {
	tv := &TV{}
	remote := NewRemote(tv)
	result := remote.TogglePower()
	if result != "TV is ON" {
		t.Errorf("Expected 'TV is ON', got '%s'", result)
	}
}

// Test5: Remote.TogglePower toggles off
func Test5(t *testing.T) {
	tv := &TV{}
	remote := NewRemote(tv)
	remote.TogglePower()
	result := remote.TogglePower()
	if result != "TV is OFF" {
		t.Errorf("Expected 'TV is OFF', got '%s'", result)
	}
}

// Test6: Remote.VolumeUp increases volume
func Test6(t *testing.T) {
	tv := &TV{}
	remote := NewRemote(tv)
	result := remote.VolumeUp()
	if result != "TV volume: 1" {
		t.Errorf("Expected 'TV volume: 1', got '%s'", result)
	}
}

// Test7: Radio implements Device interface
func Test7(t *testing.T) {
	var device Device = &Radio{}
	device.Enable()
	if !device.IsEnabled() {
		t.Error("Radio should implement Device interface")
	}
}

// Test8: Remote works with Radio
func Test8(t *testing.T) {
	radio := &Radio{}
	remote := NewRemote(radio)
	result := remote.TogglePower()
	if result != "Radio is ON" {
		t.Errorf("Expected 'Radio is ON', got '%s'", result)
	}
}

// Test9: Remote.ChannelUp increases channel
func Test9(t *testing.T) {
	tv := &TV{}
	remote := NewRemote(tv)
	result := remote.ChannelUp()
	if result != "TV channel: 1" {
		t.Errorf("Expected 'TV channel: 1', got '%s'", result)
	}
}

// Test10: NewRemote returns non-nil
func Test10(t *testing.T) {
	tv := &TV{}
	remote := NewRemote(tv)
	if remote == nil {
		t.Error("NewRemote should return non-nil Remote")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Bridge (Мост)',
			description: `Реализуйте паттерн Bridge на Go — отделите абстракцию от реализации так, чтобы обе могли изменяться независимо.

**Вы реализуете:**

1. **Интерфейс Device** - Интерфейс реализации (TV, Radio)
2. **Структура Remote** - Абстракция, использующая Device
3. **Структура TV** - Конкретная реализация
4. **Структура Radio** - Конкретная реализация

**Пример использования:**

\`\`\`go
// Создаём устройства (реализации)
tv := &TV{}
radio := &Radio{}

// Создаём пульты (абстракции) с разными устройствами
tvRemote := NewRemote(tv)
radioRemote := NewRemote(radio)

// Один интерфейс пульта работает с разными устройствами
tvRemote.TogglePower()   // "TV is ON"
tvRemote.VolumeUp()      // "TV volume: 1"
tvRemote.ChannelUp()     // "TV channel: 1"

radioRemote.TogglePower() // "Radio is ON"
radioRemote.VolumeUp()    // "Radio volume: 1"

// Можно менять устройства во время выполнения
tvRemote.device = radio  // Теперь управляет радио!
\`\`\``,
			hint1: `Методы TV и Radio — простые геттеры/сеттеры. IsEnabled возвращает поле 'on', Enable устанавливает его в true, Disable в false. GetVolume/GetChannel возвращают поля, SetVolume/SetChannel присваивают значения. GetName возвращает "TV" или "Radio".`,
			hint2: `Методы Remote делегируют интерфейсу device. TogglePower проверяет IsEnabled(), затем вызывает Enable() или Disable(). VolumeUp получает текущую громкость через GetVolume(), добавляет 1 и устанавливает через SetVolume(). Используйте fmt.Sprintf для форматирования строки с именем устройства и значением.`,
			whyItMatters: `**1. Зачем нужен Bridge**

Паттерн Bridge решает проблему "взрыва классов", когда есть несколько измерений вариативности. Без Bridge, M абстракций x N реализаций = M*N классов. С Bridge нужно только M + N классов.

**Проблема, которую он решает:**

\`\`\`go
// БЕЗ Bridge - взрыв классов
// Каждая комбинация требует отдельного класса!

type TVRemote struct{ /* TV-специфичное */ }
type TVAdvancedRemote struct{ /* TV-специфичное */ }
type RadioRemote struct{ /* Radio-специфичное */ }
type RadioAdvancedRemote struct{ /* Radio-специфичное */ }
type SpeakerRemote struct{ /* Speaker-специфичное */ }
type SpeakerAdvancedRemote struct{ /* Speaker-специфичное */ }

// 3 устройства x 2 типа пультов = 6 классов
// Добавить 1 устройство? +2 класса
// Добавить 1 тип пульта? +3 класса
// n устройств x m пультов = n*m классов!
\`\`\`

**С Bridge:**

\`\`\`go
// Абстракция (Remote) + Реализация (Device) отдельные иерархии
// 3 устройства + 2 пульта = 5 классов всего

// Реализации
type TV struct{}
type Radio struct{}
type Speaker struct{}

// Абстракции (могут расширяться независимо)
type Remote struct{ device Device }
type AdvancedRemote struct{ Remote }

// Добавить устройство? +1 класс (просто реализовать Device)
// Добавить пульт? +1 класс (просто встроить Remote)
// n устройств + m пультов = n+m классов!
\`\`\`

**2. Примеры из реального мира в Go**

**Мост драйвера базы данных:**

\`\`\`go
// Интерфейс реализации - разные базы данных
type DBDriver interface {
    Connect(dsn string) error
    Execute(query string) ([]Row, error)
    Close() error
}

// Конкретные реализации
type PostgresDriver struct{}
type MySQLDriver struct{}
type SQLiteDriver struct{}

// Абстракция - построитель запросов
type QueryBuilder struct {
    driver DBDriver  // мост к реализации
}

func (qb *QueryBuilder) Select(table string, cols ...string) *QueryBuilder {
    // Строим запрос используя логику абстракции
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    return qb
}

func (qb *QueryBuilder) Execute() ([]Row, error) {
    query := qb.buildQuery()
    return qb.driver.Execute(query)  // делегируем реализации
}

// Расширенная абстракция - добавляет больше функций
type TransactionalQueryBuilder struct {
    QueryBuilder  // встраиваем базовую абстракцию
}

func (tqb *TransactionalQueryBuilder) BeginTransaction() error {
    return tqb.driver.Execute("BEGIN")
}
\`\`\`

**Мост движка рендеринга:**

\`\`\`go
// Реализация - API рендеринга
type Renderer interface {
    RenderCircle(x, y, radius float64)
    RenderRectangle(x, y, width, height float64)
}

type OpenGLRenderer struct{}
type VulkanRenderer struct{}
type DirectXRenderer struct{}

// Абстракция - фигуры
type Shape struct {
    renderer Renderer  // мост
}

type Circle struct {
    Shape
    x, y, radius float64
}

func (c *Circle) Draw() {
    c.renderer.RenderCircle(c.x, c.y, c.radius)
}

type Rectangle struct {
    Shape
    x, y, width, height float64
}

func (r *Rectangle) Draw() {
    r.renderer.RenderRectangle(r.x, r.y, r.width, r.height)
}

// Можно комбинировать любую фигуру с любым рендерером!
circle := &Circle{Shape: Shape{renderer: &OpenGLRenderer{}}}
rect := &Rectangle{Shape: Shape{renderer: &VulkanRenderer{}}}
\`\`\`

**3. Продакшн паттерн - Система сообщений**

\`\`\`go
package messaging

import "time"

// Интерфейс реализации - транспортные протоколы
type Transport interface {
    Send(to string, payload []byte) error
    Receive() ([]byte, error)
    Connect() error
    Disconnect() error
}

// Конкретные реализации
type HTTPTransport struct {
    baseURL string
    client  *http.Client
}

func (h *HTTPTransport) Send(to string, payload []byte) error {
    _, err := h.client.Post(h.baseURL+"/"+to, "application/json", bytes.NewReader(payload))
    return err
}

type WebSocketTransport struct {
    conn *websocket.Conn
}

func (w *WebSocketTransport) Send(to string, payload []byte) error {
    return w.conn.WriteMessage(websocket.TextMessage, payload)
}

type GRPCTransport struct {
    client pb.MessagingClient
}

func (g *GRPCTransport) Send(to string, payload []byte) error {
    _, err := g.client.Send(context.Background(), &pb.Message{To: to, Data: payload})
    return err
}

// Абстракция - отправитель сообщений
type MessageSender struct {
    transport Transport  // мост к реализации транспорта
}

func NewMessageSender(transport Transport) *MessageSender {
    return &MessageSender{transport: transport}
}

func (m *MessageSender) SendText(to, text string) error {
    payload, _ := json.Marshal(map[string]string{"type": "text", "content": text})
    return m.transport.Send(to, payload)
}

func (m *MessageSender) SendFile(to string, file []byte) error {
    payload, _ := json.Marshal(map[string]interface{}{"type": "file", "data": file})
    return m.transport.Send(to, payload)
}

// Расширенная абстракция - добавляет надёжность
type ReliableMessageSender struct {
    MessageSender
    maxRetries int
    retryDelay time.Duration
}

func (r *ReliableMessageSender) SendWithRetry(to, text string) error {
    var lastErr error
    for i := 0; i < r.maxRetries; i++ {
        if err := r.SendText(to, text); err == nil {
            return nil
        } else {
            lastErr = err
            time.Sleep(r.retryDelay)
        }
    }
    return lastErr
}

// Использование - любой отправитель + любой транспорт
httpSender := NewMessageSender(&HTTPTransport{baseURL: "https://api.example.com"})
wsSender := NewMessageSender(&WebSocketTransport{})
reliableSender := &ReliableMessageSender{
    MessageSender: *NewMessageSender(&GRPCTransport{}),
    maxRetries:    3,
    retryDelay:    time.Second,
}
\`\`\`

**4. Типичные ошибки**

\`\`\`go
// ОШИБКА 1: Абстракция зависит от конкретной реализации
type Remote struct {
    tv *TV  // Неправильно! Зависит от конкретного типа
}

// ПРАВИЛЬНО: Зависеть от интерфейса
type Remote struct {
    device Device  // Правильно! Зависит от абстракции
}

// ОШИБКА 2: Реализация знает об абстракции
type TV struct {
    remote *Remote  // Неправильно! Создаёт циклическую зависимость
}

// ПРАВИЛЬНО: Реализация не знает об абстракции
type TV struct {
    on      bool
    volume  int
    // Нет ссылки на Remote
}

// ОШИБКА 3: Слишком широкий интерфейс моста
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
    GetChannel() int
    SetChannel(int)
    PlayDVD()        // Неправильно! Не у всех устройств есть DVD
    TuneAntenna()    // Неправильно! Не у всех устройств есть антенна
}

// ПРАВИЛЬНО: Минимальный и сфокусированный интерфейс
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
}

// Добавляйте специфичные интерфейсы для специфичных возможностей
type ChannelDevice interface {
    Device
    GetChannel() int
    SetChannel(int)
}

// ОШИБКА 4: Не использовать мост для делегирования
func (r *Remote) VolumeUp() string {
    // Неправильно! Дублирует логику реализации
    if r.device.GetVolume() < 100 {
        // Логика громкости здесь вместо делегирования
    }
}

// ПРАВИЛЬНО: Делегировать реализации
func (r *Remote) VolumeUp() string {
    r.device.SetVolume(r.device.GetVolume() + 1)
    return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())
}

// ОШИБКА 5: Создание моста когда достаточно простой композиции
// Не используйте Bridge для единственной реализации!
type OnlyTVRemote struct {
    device Device  // Избыточно если существует только TV
}

// Используйте Bridge когда есть НЕСКОЛЬКО реализаций
// и НЕСКОЛЬКО расширений абстракции
\`\`\``,
			solutionCode: `package patterns

import "fmt"

type Device interface {	// Implementor - определяет интерфейс реализации
	IsEnabled() bool	// проверка включено ли устройство
	Enable()	// включить устройство
	Disable()	// выключить устройство
	GetVolume() int	// получить текущий уровень громкости
	SetVolume(volume int)	// установить уровень громкости
	GetChannel() int	// получить текущий канал
	SetChannel(channel int)	// установить канал
	GetName() string	// получить имя устройства для отображения
}

type TV struct {	// Concrete Implementor - устройство ТВ
	on      bool	// состояние питания
	volume  int	// текущая громкость
	channel int	// текущий канал
}

func (t *TV) IsEnabled() bool  { return t.on }	// возвращает состояние питания
func (t *TV) Enable()          { t.on = true }	// включает ТВ
func (t *TV) Disable()         { t.on = false }	// выключает ТВ
func (t *TV) GetVolume() int   { return t.volume }	// возвращает текущую громкость
func (t *TV) SetVolume(v int)  { t.volume = v }	// устанавливает уровень громкости
func (t *TV) GetChannel() int  { return t.channel }	// возвращает текущий канал
func (t *TV) SetChannel(c int) { t.channel = c }	// устанавливает номер канала
func (t *TV) GetName() string  { return "TV" }	// возвращает имя устройства

type Radio struct {	// Concrete Implementor - устройство Радио
	on      bool	// состояние питания
	volume  int	// текущая громкость
	channel int	// текущая частота/станция
}

func (r *Radio) IsEnabled() bool  { return r.on }	// возвращает состояние питания
func (r *Radio) Enable()          { r.on = true }	// включает Радио
func (r *Radio) Disable()         { r.on = false }	// выключает Радио
func (r *Radio) GetVolume() int   { return r.volume }	// возвращает текущую громкость
func (r *Radio) SetVolume(v int)  { r.volume = v }	// устанавливает уровень громкости
func (r *Radio) GetChannel() int  { return r.channel }	// возвращает текущую станцию
func (r *Radio) SetChannel(c int) { r.channel = c }	// устанавливает номер станции
func (r *Radio) GetName() string  { return "Radio" }	// возвращает имя устройства

type Remote struct {	// Abstraction - пульт управления
	device Device	// ссылка на реализацию (мост)
}

func NewRemote(device Device) *Remote {	// создаёт пульт с устройством
	return &Remote{device: device}	// внедряем реализацию
}

func (r *Remote) TogglePower() string {	// метод абстракции - переключение питания
	if r.device.IsEnabled() {	// проверяем текущее состояние через мост
		r.device.Disable()	// делегируем реализации
		return fmt.Sprintf("%s is OFF", r.device.GetName())	// возвращаем статус
	}
	r.device.Enable()	// делегируем реализации
	return fmt.Sprintf("%s is ON", r.device.GetName())	// возвращаем статус
}

func (r *Remote) VolumeUp() string {	// метод абстракции - увеличение громкости
	r.device.SetVolume(r.device.GetVolume() + 1)	// делегируем через мост
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// возвращаем результат
}

func (r *Remote) VolumeDown() string {	// метод абстракции - уменьшение громкости
	r.device.SetVolume(r.device.GetVolume() - 1)	// делегируем через мост
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// возвращаем результат
}

func (r *Remote) ChannelUp() string {	// метод абстракции - следующий канал
	r.device.SetChannel(r.device.GetChannel() + 1)	// делегируем через мост
	return fmt.Sprintf("%s channel: %d", r.device.GetName(), r.device.GetChannel())	// возвращаем результат
}`
		},
		uz: {
			title: 'Bridge (Ko\'prik) Pattern',
			description: `Go tilida Bridge patternini amalga oshiring — abstraktsiyani realizatsiyadan ajrating, shunda ikkalasi mustaqil o'zgarishi mumkin.

**Siz amalga oshirasiz:**

1. **Device interfeysi** - Realizatsiya interfeysi (TV, Radio)
2. **Remote strukturasi** - Device ni ishlatadigan abstraktsiya
3. **TV strukturasi** - Konkret realizatsiya
4. **Radio strukturasi** - Konkret realizatsiya

**Foydalanish namunasi:**

\`\`\`go
// Qurilmalarni yaratish (realizatsiyalar)
tv := &TV{}
radio := &Radio{}

// Turli qurilmalar bilan pultlarni yaratish (abstraktsiyalar)
tvRemote := NewRemote(tv)
radioRemote := NewRemote(radio)

// Bitta pult interfeysi turli qurilmalar bilan ishlaydi
tvRemote.TogglePower()   // "TV is ON"
tvRemote.VolumeUp()      // "TV volume: 1"
tvRemote.ChannelUp()     // "TV channel: 1"

radioRemote.TogglePower() // "Radio is ON"
radioRemote.VolumeUp()    // "Radio volume: 1"

// Runtime da qurilmalarni almashtirish mumkin
tvRemote.device = radio  // Endi radio ni boshqaradi!
\`\`\``,
			hint1: `TV va Radio metodlari oddiy getter/setterlar. IsEnabled 'on' maydonini qaytaradi, Enable uni true ga o'rnatadi, Disable false ga. GetVolume/GetChannel maydonlarni qaytaradi, SetVolume/SetChannel qiymatlarni tayinlaydi. GetName "TV" yoki "Radio" qaytaradi.`,
			hint2: `Remote metodlari device interfeysiga delegatsiya qiladi. TogglePower IsEnabled() ni tekshiradi, keyin Enable() yoki Disable() ni chaqiradi. VolumeUp GetVolume() orqali joriy ovozni oladi, 1 qo'shadi va SetVolume() bilan o'rnatadi. Qurilma nomi va qiymat bilan stringni formatlash uchun fmt.Sprintf dan foydalaning.`,
			whyItMatters: `**1. Bridge nima uchun kerak**

Bridge pattern "klasslar portlashi" muammosini hal qiladi, bir nechta o'zgaruvchanlik o'lchamlari bo'lganda. Bridge siz, M abstraktsiya x N realizatsiya = M*N klass. Bridge bilan, faqat M + N klass kerak.

**U hal qiladigan muammo:**

\`\`\`go
// Bridge SIZ - klasslar portlashi
// Har bir kombinatsiya o'z klassini talab qiladi!

type TVRemote struct{ /* TV-maxsus */ }
type TVAdvancedRemote struct{ /* TV-maxsus */ }
type RadioRemote struct{ /* Radio-maxsus */ }
type RadioAdvancedRemote struct{ /* Radio-maxsus */ }
type SpeakerRemote struct{ /* Speaker-maxsus */ }
type SpeakerAdvancedRemote struct{ /* Speaker-maxsus */ }

// 3 qurilma x 2 pult turi = 6 klass
// 1 yangi qurilma qo'shish? +2 klass
// 1 yangi pult turi qo'shish? +3 klass
// n qurilma x m pult = n*m klass!
\`\`\`

**Bridge BILAN:**

\`\`\`go
// Abstraktsiya (Remote) + Realizatsiya (Device) alohida ierarxiyalar
// 3 qurilma + 2 pult = jami 5 klass

// Realizatsiyalar
type TV struct{}
type Radio struct{}
type Speaker struct{}

// Abstraktsiyalar (mustaqil kengaytirish mumkin)
type Remote struct{ device Device }
type AdvancedRemote struct{ Remote }

// Yangi qurilma qo'shish? +1 klass (faqat Device ni amalga oshirish)
// Yangi pult qo'shish? +1 klass (faqat Remote ni embed qilish)
// n qurilma + m pult = n+m klass!
\`\`\`

**2. Go'da real hayotiy misollar**

**Ma'lumotlar bazasi drayveri ko'prigi:**

\`\`\`go
// Realizatsiya interfeysi - turli bazalar
type DBDriver interface {
    Connect(dsn string) error
    Execute(query string) ([]Row, error)
    Close() error
}

// Konkret realizatsiyalar
type PostgresDriver struct{}
type MySQLDriver struct{}
type SQLiteDriver struct{}

// Abstraktsiya - so'rov quruvchisi
type QueryBuilder struct {
    driver DBDriver  // realizatsiyaga ko'prik
}

func (qb *QueryBuilder) Select(table string, cols ...string) *QueryBuilder {
    // Abstraktsiya logikasi yordamida so'rov quramiz
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    return qb
}

func (qb *QueryBuilder) Execute() ([]Row, error) {
    query := qb.buildQuery()
    return qb.driver.Execute(query)  // realizatsiyaga delegatsiya
}

// Kengaytirilgan abstraktsiya - ko'proq funksiyalar qo'shadi
type TransactionalQueryBuilder struct {
    QueryBuilder  // asosiy abstraktsiyani embed qilish
}

func (tqb *TransactionalQueryBuilder) BeginTransaction() error {
    return tqb.driver.Execute("BEGIN")
}
\`\`\`

**Rendering dvigateli ko'prigi:**

\`\`\`go
// Realizatsiya - rendering APIlari
type Renderer interface {
    RenderCircle(x, y, radius float64)
    RenderRectangle(x, y, width, height float64)
}

type OpenGLRenderer struct{}
type VulkanRenderer struct{}
type DirectXRenderer struct{}

// Abstraktsiya - shakllar
type Shape struct {
    renderer Renderer  // ko'prik
}

type Circle struct {
    Shape
    x, y, radius float64
}

func (c *Circle) Draw() {
    c.renderer.RenderCircle(c.x, c.y, c.radius)
}

type Rectangle struct {
    Shape
    x, y, width, height float64
}

func (r *Rectangle) Draw() {
    r.renderer.RenderRectangle(r.x, r.y, r.width, r.height)
}

// Har qanday shaklni har qanday renderer bilan birlashtirib bo'ladi!
circle := &Circle{Shape: Shape{renderer: &OpenGLRenderer{}}}
rect := &Rectangle{Shape: Shape{renderer: &VulkanRenderer{}}}
\`\`\`

**3. Production pattern - Xabar tizimi**

\`\`\`go
package messaging

import "time"

// Realizatsiya interfeysi - transport protokollari
type Transport interface {
    Send(to string, payload []byte) error
    Receive() ([]byte, error)
    Connect() error
    Disconnect() error
}

// Konkret realizatsiyalar
type HTTPTransport struct {
    baseURL string
    client  *http.Client
}

func (h *HTTPTransport) Send(to string, payload []byte) error {
    _, err := h.client.Post(h.baseURL+"/"+to, "application/json", bytes.NewReader(payload))
    return err
}

type WebSocketTransport struct {
    conn *websocket.Conn
}

func (w *WebSocketTransport) Send(to string, payload []byte) error {
    return w.conn.WriteMessage(websocket.TextMessage, payload)
}

type GRPCTransport struct {
    client pb.MessagingClient
}

func (g *GRPCTransport) Send(to string, payload []byte) error {
    _, err := g.client.Send(context.Background(), &pb.Message{To: to, Data: payload})
    return err
}

// Abstraktsiya - xabar yuboruvchi
type MessageSender struct {
    transport Transport  // transport realizatsiyasiga ko'prik
}

func NewMessageSender(transport Transport) *MessageSender {
    return &MessageSender{transport: transport}
}

func (m *MessageSender) SendText(to, text string) error {
    payload, _ := json.Marshal(map[string]string{"type": "text", "content": text})
    return m.transport.Send(to, payload)
}

func (m *MessageSender) SendFile(to string, file []byte) error {
    payload, _ := json.Marshal(map[string]interface{}{"type": "file", "data": file})
    return m.transport.Send(to, payload)
}

// Kengaytirilgan abstraktsiya - ishonchlilik qo'shadi
type ReliableMessageSender struct {
    MessageSender
    maxRetries int
    retryDelay time.Duration
}

func (r *ReliableMessageSender) SendWithRetry(to, text string) error {
    var lastErr error
    for i := 0; i < r.maxRetries; i++ {
        if err := r.SendText(to, text); err == nil {
            return nil
        } else {
            lastErr = err
            time.Sleep(r.retryDelay)
        }
    }
    return lastErr
}

// Foydalanish - har qanday yuboruvchi + har qanday transport
httpSender := NewMessageSender(&HTTPTransport{baseURL: "https://api.example.com"})
wsSender := NewMessageSender(&WebSocketTransport{})
reliableSender := &ReliableMessageSender{
    MessageSender: *NewMessageSender(&GRPCTransport{}),
    maxRetries:    3,
    retryDelay:    time.Second,
}
\`\`\`

**4. Keng tarqalgan xatolar**

\`\`\`go
// XATO 1: Abstraktsiya konkret realizatsiyaga bog'liq
type Remote struct {
    tv *TV  // Noto'g'ri! Konkret turga bog'liq
}

// TO'G'RI: Interfeysga bog'liq bo'lish
type Remote struct {
    device Device  // To'g'ri! Abstraktsiyaga bog'liq
}

// XATO 2: Realizatsiya abstraktsiyani biladi
type TV struct {
    remote *Remote  // Noto'g'ri! Aylanma bog'liqlik yaratadi
}

// TO'G'RI: Realizatsiya abstraktsiyani bilmaydi
type TV struct {
    on      bool
    volume  int
    // Remote ga havola yo'q
}

// XATO 3: Ko'prik interfeysi juda keng
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
    GetChannel() int
    SetChannel(int)
    PlayDVD()        // Noto'g'ri! Hamma qurilmalarda DVD yo'q
    TuneAntenna()    // Noto'g'ri! Hamma qurilmalarda antenna yo'q
}

// TO'G'RI: Minimal va fokuslanganinterfeys saqlang
type Device interface {
    IsEnabled() bool
    Enable()
    Disable()
    GetVolume() int
    SetVolume(int)
}

// Maxsus qobiliyatlar uchun maxsus interfeyslar qo'shing
type ChannelDevice interface {
    Device
    GetChannel() int
    SetChannel(int)
}

// XATO 4: Delegatsiya uchun ko'prikni ishlatmaslik
func (r *Remote) VolumeUp() string {
    // Noto'g'ri! Realizatsiya logikasini takrorlaydi
    if r.device.GetVolume() < 100 {
        // Delegatsiya o'rniga bu yerda ovoz logikasi
    }
}

// TO'G'RI: Realizatsiyaga delegatsiya qilish
func (r *Remote) VolumeUp() string {
    r.device.SetVolume(r.device.GetVolume() + 1)
    return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())
}

// XATO 5: Oddiy kompozitsiya yetarli bo'lganda ko'prik yaratish
// Bitta realizatsiya uchun Bridge ishlatmang!
type OnlyTVRemote struct {
    device Device  // Faqat TV mavjud bo'lsa ortiqcha
}

// Bridge ni NECHTA realizatsiya mavjud bo'lganda ishlating
// va NECHTA abstraktsiya kengaytmalari bo'lganda
\`\`\``,
			solutionCode: `package patterns

import "fmt"

type Device interface {	// Implementor - realizatsiya interfeysini belgilaydi
	IsEnabled() bool	// qurilma yoqilganligini tekshirish
	Enable()	// qurilmani yoqish
	Disable()	// qurilmani o'chirish
	GetVolume() int	// joriy ovoz darajasini olish
	SetVolume(volume int)	// ovoz darajasini o'rnatish
	GetChannel() int	// joriy kanalni olish
	SetChannel(channel int)	// kanalni o'rnatish
	GetName() string	// ko'rsatish uchun qurilma nomini olish
}

type TV struct {	// Concrete Implementor - TV qurilmasi
	on      bool	// quvvat holati
	volume  int	// joriy ovoz
	channel int	// joriy kanal
}

func (t *TV) IsEnabled() bool  { return t.on }	// quvvat holatini qaytaradi
func (t *TV) Enable()          { t.on = true }	// TV ni yoqadi
func (t *TV) Disable()         { t.on = false }	// TV ni o'chiradi
func (t *TV) GetVolume() int   { return t.volume }	// joriy ovozni qaytaradi
func (t *TV) SetVolume(v int)  { t.volume = v }	// ovoz darajasini o'rnatadi
func (t *TV) GetChannel() int  { return t.channel }	// joriy kanalni qaytaradi
func (t *TV) SetChannel(c int) { t.channel = c }	// kanal raqamini o'rnatadi
func (t *TV) GetName() string  { return "TV" }	// qurilma nomini qaytaradi

type Radio struct {	// Concrete Implementor - Radio qurilmasi
	on      bool	// quvvat holati
	volume  int	// joriy ovoz
	channel int	// joriy chastota/stansiya
}

func (r *Radio) IsEnabled() bool  { return r.on }	// quvvat holatini qaytaradi
func (r *Radio) Enable()          { r.on = true }	// Radio ni yoqadi
func (r *Radio) Disable()         { r.on = false }	// Radio ni o'chiradi
func (r *Radio) GetVolume() int   { return r.volume }	// joriy ovozni qaytaradi
func (r *Radio) SetVolume(v int)  { r.volume = v }	// ovoz darajasini o'rnatadi
func (r *Radio) GetChannel() int  { return r.channel }	// joriy stansiyani qaytaradi
func (r *Radio) SetChannel(c int) { r.channel = c }	// stansiya raqamini o'rnatadi
func (r *Radio) GetName() string  { return "Radio" }	// qurilma nomini qaytaradi

type Remote struct {	// Abstraction - masofadan boshqarish pulti
	device Device	// realizatsiyaga havola (ko'prik)
}

func NewRemote(device Device) *Remote {	// qurilma bilan pult yaratadi
	return &Remote{device: device}	// realizatsiyani kiritish
}

func (r *Remote) TogglePower() string {	// abstraktsiya metodi - quvvatni almashtirish
	if r.device.IsEnabled() {	// ko'prik orqali joriy holatni tekshirish
		r.device.Disable()	// realizatsiyaga delegatsiya
		return fmt.Sprintf("%s is OFF", r.device.GetName())	// holatni qaytarish
	}
	r.device.Enable()	// realizatsiyaga delegatsiya
	return fmt.Sprintf("%s is ON", r.device.GetName())	// holatni qaytarish
}

func (r *Remote) VolumeUp() string {	// abstraktsiya metodi - ovozni oshirish
	r.device.SetVolume(r.device.GetVolume() + 1)	// ko'prik orqali delegatsiya
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// natijani qaytarish
}

func (r *Remote) VolumeDown() string {	// abstraktsiya metodi - ovozni kamaytirish
	r.device.SetVolume(r.device.GetVolume() - 1)	// ko'prik orqali delegatsiya
	return fmt.Sprintf("%s volume: %d", r.device.GetName(), r.device.GetVolume())	// natijani qaytarish
}

func (r *Remote) ChannelUp() string {	// abstraktsiya metodi - keyingi kanal
	r.device.SetChannel(r.device.GetChannel() + 1)	// ko'prik orqali delegatsiya
	return fmt.Sprintf("%s channel: %d", r.device.GetName(), r.device.GetChannel())	// natijani qaytarish
}`
		}
	}
};

export default task;
