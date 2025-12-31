import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-prototype',
	title: 'Prototype Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'creational', 'prototype'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Prototype pattern in Go - create new objects by copying an existing object (prototype) rather than creating from scratch.

**You will implement:**

1. **Cloneable interface** - Clone() method
2. **Document struct** - Has title, content, author, tags
3. **Clone method** - Creates deep copy of Document
4. **DocumentRegistry** - Stores and retrieves prototypes

**Example Usage:**

\`\`\`go
// Create original document as prototype
original := &Document{
    Title:   "Template",
    Content: "Default content",
    Author:  "System",
    Tags:    []string{"template", "default"},
}

// Register prototype in registry
registry := NewDocumentRegistry()
registry.Register("template", original)

// Get a clone from registry
clone := registry.GetClone("template").(*Document)
clone.Title = "My Document"
clone.Tags = append(clone.Tags, "custom")

// Original is unchanged (deep copy)
fmt.Println(original.Title) // "Template"
fmt.Println(len(original.Tags)) // 2 (not 3)

// Create multiple independent clones
report := registry.GetClone("template").(*Document)
report.Title = "Monthly Report"
\`\`\``,
	initialCode: `package patterns

type Cloneable interface {
}

type Document struct {
	Title   string
	Content string
	Author  string
}

func (d *Document) Clone() Cloneable {
}

type DocumentRegistry struct {
	prototypes map[string]Cloneable
}

func NewDocumentRegistry() *DocumentRegistry {
	}
}

func (r *DocumentRegistry) Register(name string, prototype Cloneable) {
}

func (r *DocumentRegistry) GetClone(name string) Cloneable {
}`,
	solutionCode: `package patterns

type Cloneable interface {	// Prototype interface - defines cloning contract
	Clone() Cloneable	// returns a copy of the object
}

type Document struct {	// Concrete Prototype - object that can be cloned
	Title   string	// document title
	Content string	// document content
	Author  string	// document author
	Tags    []string	// document tags (requires deep copy!)
}

func (d *Document) Clone() Cloneable {	// implements Cloneable interface
	tagsCopy := make([]string, len(d.Tags))	// create new slice with same capacity
	copy(tagsCopy, d.Tags)	// copy elements to avoid sharing underlying array

	return &Document{	// return new Document with copied values
		Title:   d.Title,	// strings are immutable in Go, safe to copy directly
		Content: d.Content,	// same for content
		Author:  d.Author,	// same for author
		Tags:    tagsCopy,	// use the deep-copied slice
	}
}

type DocumentRegistry struct {	// Prototype Registry - stores and manages prototypes
	prototypes map[string]Cloneable	// map of prototype name to prototype object
}

func NewDocumentRegistry() *DocumentRegistry {	// creates new registry instance
	return &DocumentRegistry{	// initialize with empty map
		prototypes: make(map[string]Cloneable),	// avoid nil map panic
	}
}

func (r *DocumentRegistry) Register(name string, prototype Cloneable) {	// stores prototype
	r.prototypes[name] = prototype	// add to map with given key
}

func (r *DocumentRegistry) GetClone(name string) Cloneable {	// retrieves clone of prototype
	if prototype, exists := r.prototypes[name]; exists {	// check if prototype exists
		return prototype.Clone()	// return cloned copy, not original
	}
	return nil	// return nil if prototype not found
}`,
	hint1: `Clone must create a DEEP copy. For slices like Tags, create a new slice using make() and use copy() function to copy elements. If you just assign d.Tags to the new document, both will share the same underlying array!`,
	hint2: `Register simply stores the prototype in the map using the name as key. GetClone checks if the prototype exists in the map, and if so, calls Clone() on it to return a new copy. Return nil if the prototype is not found.`,
	whyItMatters: `**1. Why Prototype Exists**

Prototype pattern solves the problem of creating new objects when:
- Object creation is expensive (DB queries, network calls, complex calculations)
- You want to avoid subclass explosion when objects differ only in configuration
- Objects have complex internal state that's hard to configure via constructors

**The Problem It Solves:**

\`\`\`go
// WITHOUT Prototype - expensive object creation
func CreateGameEnemy(enemyType string) *Enemy {
    enemy := &Enemy{}

    // Expensive operations for EVERY new enemy:
    enemy.Model = LoadModelFromDisk(enemyType + ".obj")     // disk I/O
    enemy.Texture = LoadTextureFromDisk(enemyType + ".png") // disk I/O
    enemy.Sounds = LoadSoundsFromDisk(enemyType)            // disk I/O
    enemy.AIBehavior = CompileAIScript(enemyType + ".lua")  // CPU intensive
    enemy.Stats = FetchStatsFromDB(enemyType)               // network I/O

    return enemy
}

// Creating 100 zombies = 100x disk reads, 100x DB queries!
for i := 0; i < 100; i++ {
    zombie := CreateGameEnemy("zombie")  // SLOW!
}
\`\`\`

**WITH Prototype:**

\`\`\`go
// Create prototype ONCE with expensive initialization
zombiePrototype := CreateGameEnemy("zombie")  // expensive, but only once
registry.Register("zombie", zombiePrototype)

// Create 100 zombies by cloning - fast!
for i := 0; i < 100; i++ {
    zombie := registry.GetClone("zombie").(*Enemy)
    zombie.Position = randomPosition()  // customize the clone
    // Model, Texture, Sounds, AI are all copied instantly
}
\`\`\`

**2. Real-World Examples in Go**

**Deep vs Shallow Copy (Critical Concept):**

\`\`\`go
// SHALLOW COPY - WRONG! Shares underlying data
func (d *Document) ShallowClone() *Document {
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    d.Tags,  // DANGER! Same slice reference
    }
}

original := &Document{Tags: []string{"a", "b"}}
clone := original.ShallowClone()
clone.Tags[0] = "MODIFIED"
fmt.Println(original.Tags[0])  // "MODIFIED" - original corrupted!

// DEEP COPY - CORRECT! Independent data
func (d *Document) DeepClone() *Document {
    tagsCopy := make([]string, len(d.Tags))
    copy(tagsCopy, d.Tags)
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    tagsCopy,  // New slice, safe to modify
    }
}
\`\`\`

**Nested Objects (Multiple Levels of Deep Copy):**

\`\`\`go
type Order struct {
    ID       string
    Customer *Customer        // pointer - needs deep copy
    Items    []*OrderItem     // slice of pointers - needs deep copy
    Metadata map[string]string // map - needs deep copy
}

func (o *Order) Clone() *Order {
    // Clone customer
    customerCopy := &Customer{
        Name:  o.Customer.Name,
        Email: o.Customer.Email,
    }

    // Clone items slice with each item
    itemsCopy := make([]*OrderItem, len(o.Items))
    for i, item := range o.Items {
        itemsCopy[i] = &OrderItem{
            ProductID: item.ProductID,
            Quantity:  item.Quantity,
            Price:     item.Price,
        }
    }

    // Clone metadata map
    metaCopy := make(map[string]string)
    for k, v := range o.Metadata {
        metaCopy[k] = v
    }

    return &Order{
        ID:       o.ID,
        Customer: customerCopy,
        Items:    itemsCopy,
        Metadata: metaCopy,
    }
}
\`\`\`

**3. Production Pattern with Concurrent-Safe Registry**

\`\`\`go
package prototype

import (
    "sync"
    "encoding/json"
)

// Cloneable interface
type Cloneable interface {
    Clone() Cloneable
}

// Thread-safe prototype registry
type Registry struct {
    mu         sync.RWMutex
    prototypes map[string]Cloneable
}

func NewRegistry() *Registry {
    return &Registry{
        prototypes: make(map[string]Cloneable),
    }
}

func (r *Registry) Register(name string, prototype Cloneable) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.prototypes[name] = prototype
}

func (r *Registry) Unregister(name string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.prototypes, name)
}

func (r *Registry) GetClone(name string) (Cloneable, bool) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    prototype, exists := r.prototypes[name]
    if !exists {
        return nil, false
    }
    return prototype.Clone(), true
}

func (r *Registry) List() []string {
    r.mu.RLock()
    defer r.mu.RUnlock()

    names := make([]string, 0, len(r.prototypes))
    for name := range r.prototypes {
        names = append(names, name)
    }
    return names
}

// Generic deep clone using JSON (for complex objects)
func DeepCloneViaJSON[T any](src T) (T, error) {
    var dst T
    data, err := json.Marshal(src)
    if err != nil {
        return dst, err
    }
    err = json.Unmarshal(data, &dst)
    return dst, err
}
\`\`\`

**4. Common Mistakes to Avoid**

\`\`\`go
// MISTAKE 1: Shallow copy of slices
func (d *Document) Clone() Cloneable {
    return &Document{
        Title: d.Title,
        Tags:  d.Tags,  // Wrong! Shares slice
    }
}

// MISTAKE 2: Shallow copy of maps
type Config struct {
    Settings map[string]string
}

func (c *Config) Clone() Cloneable {
    return &Config{
        Settings: c.Settings,  // Wrong! Shares map
    }
}

// CORRECT: Deep copy map
func (c *Config) Clone() Cloneable {
    settingsCopy := make(map[string]string)
    for k, v := range c.Settings {
        settingsCopy[k] = v
    }
    return &Config{Settings: settingsCopy}
}

// MISTAKE 3: Shallow copy of pointers
type Node struct {
    Value int
    Next  *Node
}

func (n *Node) Clone() Cloneable {
    return &Node{
        Value: n.Value,
        Next:  n.Next,  // Wrong! Shares pointer
    }
}

// CORRECT: Deep copy pointer
func (n *Node) Clone() Cloneable {
    clone := &Node{Value: n.Value}
    if n.Next != nil {
        clone.Next = n.Next.Clone().(*Node)  // Recursive deep copy
    }
    return clone
}

// MISTAKE 4: Returning original instead of clone from registry
func (r *Registry) GetClone(name string) Cloneable {
    return r.prototypes[name]  // Wrong! Returns original
}

// CORRECT: Always call Clone()
func (r *Registry) GetClone(name string) Cloneable {
    if proto, ok := r.prototypes[name]; ok {
        return proto.Clone()  // Returns copy
    }
    return nil
}

// MISTAKE 5: Forgetting interface methods in clone
type Document struct {
    // ...fields
    onSave func()  // function field - cannot be deep copied!
}
// Be careful with function fields, channels, mutexes - they need special handling
\`\`\``,
	order: 4,
	testCode: `package patterns

import (
	"testing"
)

// Test1: Document.Clone creates copy
func Test1(t *testing.T) {
	original := &Document{Title: "Test", Content: "Content", Author: "Author", Tags: []string{"a", "b"}}
	clone := original.Clone().(*Document)
	if clone.Title != "Test" {
		t.Error("Clone should copy Title")
	}
}

// Test2: Clone creates deep copy of Tags
func Test2(t *testing.T) {
	original := &Document{Tags: []string{"a", "b"}}
	clone := original.Clone().(*Document)
	clone.Tags[0] = "modified"
	if original.Tags[0] == "modified" {
		t.Error("Clone should create deep copy of Tags")
	}
}

// Test3: Clone returns different instance
func Test3(t *testing.T) {
	original := &Document{Title: "Test"}
	clone := original.Clone().(*Document)
	if original == clone {
		t.Error("Clone should return different instance")
	}
}

// Test4: DocumentRegistry Register works
func Test4(t *testing.T) {
	registry := NewDocumentRegistry()
	doc := &Document{Title: "Template"}
	registry.Register("template", doc)
	clone := registry.GetClone("template")
	if clone == nil {
		t.Error("Register should store prototype")
	}
}

// Test5: GetClone returns nil for unknown name
func Test5(t *testing.T) {
	registry := NewDocumentRegistry()
	clone := registry.GetClone("unknown")
	if clone != nil {
		t.Error("GetClone should return nil for unknown name")
	}
}

// Test6: GetClone returns clone, not original
func Test6(t *testing.T) {
	registry := NewDocumentRegistry()
	original := &Document{Title: "Original"}
	registry.Register("doc", original)
	clone := registry.GetClone("doc").(*Document)
	if clone == original {
		t.Error("GetClone should return clone, not original")
	}
}

// Test7: Multiple clones are independent
func Test7(t *testing.T) {
	registry := NewDocumentRegistry()
	registry.Register("doc", &Document{Tags: []string{"tag"}})
	clone1 := registry.GetClone("doc").(*Document)
	clone2 := registry.GetClone("doc").(*Document)
	clone1.Title = "Modified"
	if clone2.Title == "Modified" {
		t.Error("Clones should be independent")
	}
}

// Test8: Clone copies all string fields
func Test8(t *testing.T) {
	original := &Document{Title: "T", Content: "C", Author: "A"}
	clone := original.Clone().(*Document)
	if clone.Content != "C" || clone.Author != "A" {
		t.Error("Clone should copy all string fields")
	}
}

// Test9: Document implements Cloneable
func Test9(t *testing.T) {
	var c Cloneable = &Document{}
	if c == nil {
		t.Error("Document should implement Cloneable")
	}
}

// Test10: NewDocumentRegistry creates empty registry
func Test10(t *testing.T) {
	registry := NewDocumentRegistry()
	if registry == nil {
		t.Error("NewDocumentRegistry should return non-nil registry")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Prototype (Прототип)',
			description: `Реализуйте паттерн Prototype на Go — создавайте новые объекты путём копирования существующего объекта (прототипа), а не создания с нуля.

**Вы реализуете:**

1. **Интерфейс Cloneable** - Метод Clone()
2. **Структура Document** - Содержит title, content, author, tags
3. **Метод Clone** - Создаёт глубокую копию Document
4. **DocumentRegistry** - Хранит и выдаёт прототипы

**Пример использования:**

\`\`\`go
// Создаём оригинальный документ как прототип
original := &Document{
    Title:   "Template",
    Content: "Default content",
    Author:  "System",
    Tags:    []string{"template", "default"},
}

// Регистрируем прототип в реестре
registry := NewDocumentRegistry()
registry.Register("template", original)

// Получаем клон из реестра
clone := registry.GetClone("template").(*Document)
clone.Title = "My Document"
clone.Tags = append(clone.Tags, "custom")

// Оригинал не изменён (глубокая копия)
fmt.Println(original.Title) // "Template"
fmt.Println(len(original.Tags)) // 2 (не 3)

// Создаём несколько независимых клонов
report := registry.GetClone("template").(*Document)
report.Title = "Monthly Report"
\`\`\``,
			hint1: `Clone должен создавать ГЛУБОКУЮ копию. Для срезов как Tags создайте новый срез с помощью make() и используйте copy() для копирования элементов. Если просто присвоить d.Tags новому документу, оба будут разделять один массив!`,
			hint2: `Register просто сохраняет прототип в map используя имя как ключ. GetClone проверяет существование прототипа в map, и если есть, вызывает Clone() чтобы вернуть новую копию. Верните nil если прототип не найден.`,
			whyItMatters: `**1. Зачем нужен Prototype**

Паттерн Prototype решает проблему создания новых объектов когда:
- Создание объекта дорогое (запросы к БД, сетевые вызовы, сложные вычисления)
- Вы хотите избежать взрыва подклассов когда объекты отличаются только конфигурацией
- Объекты имеют сложное внутреннее состояние которое трудно настроить через конструкторы

**Проблема, которую он решает:**

\`\`\`go
// БЕЗ Prototype - дорогое создание объектов
func CreateGameEnemy(enemyType string) *Enemy {
    enemy := &Enemy{}

    // Дорогие операции для КАЖДОГО нового врага:
    enemy.Model = LoadModelFromDisk(enemyType + ".obj")     // дисковый I/O
    enemy.Texture = LoadTextureFromDisk(enemyType + ".png") // дисковый I/O
    enemy.Sounds = LoadSoundsFromDisk(enemyType)            // дисковый I/O
    enemy.AIBehavior = CompileAIScript(enemyType + ".lua")  // интенсивный CPU
    enemy.Stats = FetchStatsFromDB(enemyType)               // сетевой I/O

    return enemy
}

// Создание 100 зомби = 100x чтений с диска, 100x запросов к БД!
for i := 0; i < 100; i++ {
    zombie := CreateGameEnemy("zombie")  // МЕДЛЕННО!
}
\`\`\`

**С Prototype:**

\`\`\`go
// Создаём прототип ОДИН раз с дорогой инициализацией
zombiePrototype := CreateGameEnemy("zombie")  // дорого, но только один раз
registry.Register("zombie", zombiePrototype)

// Создаём 100 зомби клонированием - быстро!
for i := 0; i < 100; i++ {
    zombie := registry.GetClone("zombie").(*Enemy)
    zombie.Position = randomPosition()  // настраиваем клон
    // Model, Texture, Sounds, AI все копируются мгновенно
}
\`\`\`

**2. Примеры из реального мира в Go**

**Глубокое vs Поверхностное копирование (Критичная концепция):**

\`\`\`go
// ПОВЕРХНОСТНОЕ КОПИРОВАНИЕ - НЕПРАВИЛЬНО! Разделяет данные
func (d *Document) ShallowClone() *Document {
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    d.Tags,  // ОПАСНОСТЬ! Та же ссылка на срез
    }
}

original := &Document{Tags: []string{"a", "b"}}
clone := original.ShallowClone()
clone.Tags[0] = "MODIFIED"
fmt.Println(original.Tags[0])  // "MODIFIED" - оригинал повреждён!

// ГЛУБОКОЕ КОПИРОВАНИЕ - ПРАВИЛЬНО! Независимые данные
func (d *Document) DeepClone() *Document {
    tagsCopy := make([]string, len(d.Tags))
    copy(tagsCopy, d.Tags)
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    tagsCopy,  // Новый срез, безопасно изменять
    }
}
\`\`\`

**Вложенные объекты (Многоуровневое глубокое копирование):**

\`\`\`go
type Order struct {
    ID       string
    Customer *Customer        // указатель - нужно глубокое копирование
    Items    []*OrderItem     // срез указателей - нужно глубокое копирование
    Metadata map[string]string // map - нужно глубокое копирование
}

func (o *Order) Clone() *Order {
    // Клонируем customer
    customerCopy := &Customer{
        Name:  o.Customer.Name,
        Email: o.Customer.Email,
    }

    // Клонируем срез items с каждым элементом
    itemsCopy := make([]*OrderItem, len(o.Items))
    for i, item := range o.Items {
        itemsCopy[i] = &OrderItem{
            ProductID: item.ProductID,
            Quantity:  item.Quantity,
            Price:     item.Price,
        }
    }

    // Клонируем map metadata
    metaCopy := make(map[string]string)
    for k, v := range o.Metadata {
        metaCopy[k] = v
    }

    return &Order{
        ID:       o.ID,
        Customer: customerCopy,
        Items:    itemsCopy,
        Metadata: metaCopy,
    }
}
\`\`\`

**3. Продакшн паттерн с потокобезопасным реестром**

\`\`\`go
package prototype

import (
    "sync"
    "encoding/json"
)

// Cloneable интерфейс
type Cloneable interface {
    Clone() Cloneable
}

// Потокобезопасный реестр прототипов
type Registry struct {
    mu         sync.RWMutex
    prototypes map[string]Cloneable
}

func NewRegistry() *Registry {
    return &Registry{
        prototypes: make(map[string]Cloneable),
    }
}

func (r *Registry) Register(name string, prototype Cloneable) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.prototypes[name] = prototype
}

func (r *Registry) Unregister(name string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.prototypes, name)
}

func (r *Registry) GetClone(name string) (Cloneable, bool) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    prototype, exists := r.prototypes[name]
    if !exists {
        return nil, false
    }
    return prototype.Clone(), true
}

func (r *Registry) List() []string {
    r.mu.RLock()
    defer r.mu.RUnlock()

    names := make([]string, 0, len(r.prototypes))
    for name := range r.prototypes {
        names = append(names, name)
    }
    return names
}

// Универсальное глубокое клонирование через JSON (для сложных объектов)
func DeepCloneViaJSON[T any](src T) (T, error) {
    var dst T
    data, err := json.Marshal(src)
    if err != nil {
        return dst, err
    }
    err = json.Unmarshal(data, &dst)
    return dst, err
}
\`\`\`

**4. Типичные ошибки**

\`\`\`go
// ОШИБКА 1: Поверхностное копирование срезов
func (d *Document) Clone() Cloneable {
    return &Document{
        Title: d.Title,
        Tags:  d.Tags,  // Неправильно! Разделяет срез
    }
}

// ОШИБКА 2: Поверхностное копирование map
type Config struct {
    Settings map[string]string
}

func (c *Config) Clone() Cloneable {
    return &Config{
        Settings: c.Settings,  // Неправильно! Разделяет map
    }
}

// ПРАВИЛЬНО: Глубокое копирование map
func (c *Config) Clone() Cloneable {
    settingsCopy := make(map[string]string)
    for k, v := range c.Settings {
        settingsCopy[k] = v
    }
    return &Config{Settings: settingsCopy}
}

// ОШИБКА 3: Поверхностное копирование указателей
type Node struct {
    Value int
    Next  *Node
}

func (n *Node) Clone() Cloneable {
    return &Node{
        Value: n.Value,
        Next:  n.Next,  // Неправильно! Разделяет указатель
    }
}

// ПРАВИЛЬНО: Глубокое копирование указателя
func (n *Node) Clone() Cloneable {
    clone := &Node{Value: n.Value}
    if n.Next != nil {
        clone.Next = n.Next.Clone().(*Node)  // Рекурсивное глубокое копирование
    }
    return clone
}

// ОШИБКА 4: Возврат оригинала вместо клона из реестра
func (r *Registry) GetClone(name string) Cloneable {
    return r.prototypes[name]  // Неправильно! Возвращает оригинал
}

// ПРАВИЛЬНО: Всегда вызывайте Clone()
func (r *Registry) GetClone(name string) Cloneable {
    if proto, ok := r.prototypes[name]; ok {
        return proto.Clone()  // Возвращает копию
    }
    return nil
}

// ОШИБКА 5: Забыли про методы интерфейса в клоне
type Document struct {
    // ...поля
    onSave func()  // поле-функция - нельзя глубоко скопировать!
}
// Осторожно с полями-функциями, каналами, мьютексами - требуют особой обработки
\`\`\``,
			solutionCode: `package patterns

type Cloneable interface {	// интерфейс Прототипа - определяет контракт клонирования
	Clone() Cloneable	// возвращает копию объекта
}

type Document struct {	// Конкретный Прототип - объект который можно клонировать
	Title   string	// заголовок документа
	Content string	// содержимое документа
	Author  string	// автор документа
	Tags    []string	// теги документа (требует глубокого копирования!)
}

func (d *Document) Clone() Cloneable {	// реализует интерфейс Cloneable
	tagsCopy := make([]string, len(d.Tags))	// создаём новый срез той же ёмкости
	copy(tagsCopy, d.Tags)	// копируем элементы чтобы не разделять массив

	return &Document{	// возвращаем новый Document со скопированными значениями
		Title:   d.Title,	// строки в Go неизменяемы, безопасно копировать напрямую
		Content: d.Content,	// то же для содержимого
		Author:  d.Author,	// то же для автора
		Tags:    tagsCopy,	// используем глубоко скопированный срез
	}
}

type DocumentRegistry struct {	// Реестр Прототипов - хранит и управляет прототипами
	prototypes map[string]Cloneable	// map от имени прототипа к объекту прототипа
}

func NewDocumentRegistry() *DocumentRegistry {	// создаёт новый экземпляр реестра
	return &DocumentRegistry{	// инициализируем пустым map
		prototypes: make(map[string]Cloneable),	// избегаем паники от nil map
	}
}

func (r *DocumentRegistry) Register(name string, prototype Cloneable) {	// сохраняет прототип
	r.prototypes[name] = prototype	// добавляем в map с заданным ключом
}

func (r *DocumentRegistry) GetClone(name string) Cloneable {	// получает клон прототипа
	if prototype, exists := r.prototypes[name]; exists {	// проверяем существование прототипа
		return prototype.Clone()	// возвращаем клонированную копию, не оригинал
	}
	return nil	// возвращаем nil если прототип не найден
}`
		},
		uz: {
			title: 'Prototype (Prototip) Pattern',
			description: `Go tilida Prototype patternini amalga oshiring — yangi ob'ektlarni noldan yaratish o'rniga mavjud ob'ektni (prototipni) nusxalash orqali yarating.

**Siz amalga oshirasiz:**

1. **Cloneable interfeysi** - Clone() metodi
2. **Document strukturasi** - title, content, author, tags ni o'z ichiga oladi
3. **Clone metodi** - Document ning chuqur nusxasini yaratadi
4. **DocumentRegistry** - Prototiplarni saqlaydi va beradi

**Foydalanish namunasi:**

\`\`\`go
// Asl hujjatni prototip sifatida yaratish
original := &Document{
    Title:   "Template",
    Content: "Default content",
    Author:  "System",
    Tags:    []string{"template", "default"},
}

// Prototipni reestrga ro'yxatdan o'tkazish
registry := NewDocumentRegistry()
registry.Register("template", original)

// Reestrdan klon olish
clone := registry.GetClone("template").(*Document)
clone.Title = "My Document"
clone.Tags = append(clone.Tags, "custom")

// Asl nusxa o'zgarmagan (chuqur nusxa)
fmt.Println(original.Title) // "Template"
fmt.Println(len(original.Tags)) // 2 (3 emas)

// Bir nechta mustaqil klonlar yaratish
report := registry.GetClone("template").(*Document)
report.Title = "Monthly Report"
\`\`\``,
			hint1: `Clone CHUQUR nusxa yaratishi kerak. Tags kabi slicelar uchun make() yordamida yangi slice yarating va elementlarni nusxalash uchun copy() funksiyasidan foydalaning. Agar shunchaki d.Tags ni yangi hujjatga tayinlasangiz, ikkalasi ham bitta massivni bo'lishadi!`,
			hint2: `Register shunchaki prototipni map ga nomni kalit sifatida ishlatib saqlaydi. GetClone map da prototip mavjudligini tekshiradi va agar bor bo'lsa, yangi nusxa qaytarish uchun Clone() ni chaqiradi. Prototip topilmasa nil qaytaring.`,
			whyItMatters: `**1. Prototype nima uchun kerak**

Prototype pattern yangi ob'ektlar yaratish muammosini hal qiladi qachonki:
- Ob'ekt yaratish qimmat (DB so'rovlari, tarmoq chaqiruvlari, murakkab hisoblashlar)
- Ob'ektlar faqat konfiguratsiyada farq qilganda subklass portlashidan qochmoqchisiz
- Ob'ektlar konstruktorlar orqali sozlash qiyin bo'lgan murakkab ichki holatga ega

**U hal qiladigan muammo:**

\`\`\`go
// Prototype SIZ - qimmat ob'ekt yaratish
func CreateGameEnemy(enemyType string) *Enemy {
    enemy := &Enemy{}

    // HAR BIR yangi dushman uchun qimmat operatsiyalar:
    enemy.Model = LoadModelFromDisk(enemyType + ".obj")     // disk I/O
    enemy.Texture = LoadTextureFromDisk(enemyType + ".png") // disk I/O
    enemy.Sounds = LoadSoundsFromDisk(enemyType)            // disk I/O
    enemy.AIBehavior = CompileAIScript(enemyType + ".lua")  // CPU intensiv
    enemy.Stats = FetchStatsFromDB(enemyType)               // tarmoq I/O

    return enemy
}

// 100 ta zombi yaratish = 100x diskdan o'qish, 100x DB so'rovlari!
for i := 0; i < 100; i++ {
    zombie := CreateGameEnemy("zombie")  // SEKIN!
}
\`\`\`

**Prototype BILAN:**

\`\`\`go
// Prototipni qimmat initsializatsiya bilan BIR MARTA yaratish
zombiePrototype := CreateGameEnemy("zombie")  // qimmat, lekin faqat bir marta
registry.Register("zombie", zombiePrototype)

// 100 ta zombini klonlash orqali yaratish - tez!
for i := 0; i < 100; i++ {
    zombie := registry.GetClone("zombie").(*Enemy)
    zombie.Position = randomPosition()  // klonni sozlash
    // Model, Texture, Sounds, AI hammasi bir zumda nusxalanadi
}
\`\`\`

**2. Go'da real hayotiy misollar**

**Chuqur vs Sayoz nusxalash (Muhim tushuncha):**

\`\`\`go
// SAYOZ NUSXALASH - NOTO'G'RI! Ma'lumotlarni bo'lishadi
func (d *Document) ShallowClone() *Document {
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    d.Tags,  // XAVF! Xuddi shu slice havolasi
    }
}

original := &Document{Tags: []string{"a", "b"}}
clone := original.ShallowClone()
clone.Tags[0] = "MODIFIED"
fmt.Println(original.Tags[0])  // "MODIFIED" - asl nusxa buzildi!

// CHUQUR NUSXALASH - TO'G'RI! Mustaqil ma'lumotlar
func (d *Document) DeepClone() *Document {
    tagsCopy := make([]string, len(d.Tags))
    copy(tagsCopy, d.Tags)
    return &Document{
        Title:   d.Title,
        Content: d.Content,
        Author:  d.Author,
        Tags:    tagsCopy,  // Yangi slice, o'zgartirish xavfsiz
    }
}
\`\`\`

**Ichma-ich ob'ektlar (Ko'p darajali chuqur nusxalash):**

\`\`\`go
type Order struct {
    ID       string
    Customer *Customer        // pointer - chuqur nusxalash kerak
    Items    []*OrderItem     // pointerlar slicei - chuqur nusxalash kerak
    Metadata map[string]string // map - chuqur nusxalash kerak
}

func (o *Order) Clone() *Order {
    // Customerni klonlash
    customerCopy := &Customer{
        Name:  o.Customer.Name,
        Email: o.Customer.Email,
    }

    // Har bir element bilan items sliceini klonlash
    itemsCopy := make([]*OrderItem, len(o.Items))
    for i, item := range o.Items {
        itemsCopy[i] = &OrderItem{
            ProductID: item.ProductID,
            Quantity:  item.Quantity,
            Price:     item.Price,
        }
    }

    // metadata mapini klonlash
    metaCopy := make(map[string]string)
    for k, v := range o.Metadata {
        metaCopy[k] = v
    }

    return &Order{
        ID:       o.ID,
        Customer: customerCopy,
        Items:    itemsCopy,
        Metadata: metaCopy,
    }
}
\`\`\`

**3. Thread-safe reestr bilan production pattern**

\`\`\`go
package prototype

import (
    "sync"
    "encoding/json"
)

// Cloneable interfeys
type Cloneable interface {
    Clone() Cloneable
}

// Thread-safe prototip reestri
type Registry struct {
    mu         sync.RWMutex
    prototypes map[string]Cloneable
}

func NewRegistry() *Registry {
    return &Registry{
        prototypes: make(map[string]Cloneable),
    }
}

func (r *Registry) Register(name string, prototype Cloneable) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.prototypes[name] = prototype
}

func (r *Registry) Unregister(name string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.prototypes, name)
}

func (r *Registry) GetClone(name string) (Cloneable, bool) {
    r.mu.RLock()
    defer r.mu.RUnlock()

    prototype, exists := r.prototypes[name]
    if !exists {
        return nil, false
    }
    return prototype.Clone(), true
}

func (r *Registry) List() []string {
    r.mu.RLock()
    defer r.mu.RUnlock()

    names := make([]string, 0, len(r.prototypes))
    for name := range r.prototypes {
        names = append(names, name)
    }
    return names
}

// JSON orqali universal chuqur klonlash (murakkab ob'ektlar uchun)
func DeepCloneViaJSON[T any](src T) (T, error) {
    var dst T
    data, err := json.Marshal(src)
    if err != nil {
        return dst, err
    }
    err = json.Unmarshal(data, &dst)
    return dst, err
}
\`\`\`

**4. Keng tarqalgan xatolar**

\`\`\`go
// XATO 1: Slicelarning sayoz nusxasi
func (d *Document) Clone() Cloneable {
    return &Document{
        Title: d.Title,
        Tags:  d.Tags,  // Noto'g'ri! Sliceni bo'lishadi
    }
}

// XATO 2: Maplarning sayoz nusxasi
type Config struct {
    Settings map[string]string
}

func (c *Config) Clone() Cloneable {
    return &Config{
        Settings: c.Settings,  // Noto'g'ri! Mapni bo'lishadi
    }
}

// TO'G'RI: Mapning chuqur nusxasi
func (c *Config) Clone() Cloneable {
    settingsCopy := make(map[string]string)
    for k, v := range c.Settings {
        settingsCopy[k] = v
    }
    return &Config{Settings: settingsCopy}
}

// XATO 3: Pointerlarning sayoz nusxasi
type Node struct {
    Value int
    Next  *Node
}

func (n *Node) Clone() Cloneable {
    return &Node{
        Value: n.Value,
        Next:  n.Next,  // Noto'g'ri! Pointerni bo'lishadi
    }
}

// TO'G'RI: Pointerning chuqur nusxasi
func (n *Node) Clone() Cloneable {
    clone := &Node{Value: n.Value}
    if n.Next != nil {
        clone.Next = n.Next.Clone().(*Node)  // Rekursiv chuqur nusxalash
    }
    return clone
}

// XATO 4: Reestrdan klon o'rniga asl nusxani qaytarish
func (r *Registry) GetClone(name string) Cloneable {
    return r.prototypes[name]  // Noto'g'ri! Asl nusxani qaytaradi
}

// TO'G'RI: Doimo Clone() ni chaqiring
func (r *Registry) GetClone(name string) Cloneable {
    if proto, ok := r.prototypes[name]; ok {
        return proto.Clone()  // Nusxani qaytaradi
    }
    return nil
}

// XATO 5: Klonda interfeys metodlarini unutish
type Document struct {
    // ...maydonlar
    onSave func()  // funksiya maydoni - chuqur nusxalab bo'lmaydi!
}
// Funksiya maydonlari, kanallar, mutekslar bilan ehtiyot bo'ling - maxsus ishlov kerak
\`\`\``,
			solutionCode: `package patterns

type Cloneable interface {	// Prototip interfeysi - klonlash kontraktini belgilaydi
	Clone() Cloneable	// ob'ektning nusxasini qaytaradi
}

type Document struct {	// Konkret Prototip - klonlanishi mumkin bo'lgan ob'ekt
	Title   string	// hujjat sarlavhasi
	Content string	// hujjat mazmuni
	Author  string	// hujjat muallifi
	Tags    []string	// hujjat teglari (chuqur nusxalash kerak!)
}

func (d *Document) Clone() Cloneable {	// Cloneable interfeysini amalga oshiradi
	tagsCopy := make([]string, len(d.Tags))	// xuddi shu sig'imli yangi slice yaratish
	copy(tagsCopy, d.Tags)	// massivni bo'lishmaslik uchun elementlarni nusxalash

	return &Document{	// nusxalangan qiymatlar bilan yangi Document qaytarish
		Title:   d.Title,	// Go'da stringlar o'zgarmas, to'g'ridan-to'g'ri nusxalash xavfsiz
		Content: d.Content,	// mazmun uchun ham xuddi shunday
		Author:  d.Author,	// muallif uchun ham xuddi shunday
		Tags:    tagsCopy,	// chuqur nusxalangan slicedan foydalanish
	}
}

type DocumentRegistry struct {	// Prototiplar Reestri - prototiplarni saqlaydi va boshqaradi
	prototypes map[string]Cloneable	// prototip nomidan prototip ob'ektiga map
}

func NewDocumentRegistry() *DocumentRegistry {	// yangi reestr instansiyasini yaratadi
	return &DocumentRegistry{	// bo'sh map bilan initsializatsiya
		prototypes: make(map[string]Cloneable),	// nil map panikasidan qochish
	}
}

func (r *DocumentRegistry) Register(name string, prototype Cloneable) {	// prototipni saqlaydi
	r.prototypes[name] = prototype	// berilgan kalit bilan mapga qo'shish
}

func (r *DocumentRegistry) GetClone(name string) Cloneable {	// prototip klonini oladi
	if prototype, exists := r.prototypes[name]; exists {	// prototip mavjudligini tekshirish
		return prototype.Clone()	// asl nusxa emas, klonlangan nusxani qaytarish
	}
	return nil	// prototip topilmasa nil qaytarish
}`
		}
	}
};

export default task;
