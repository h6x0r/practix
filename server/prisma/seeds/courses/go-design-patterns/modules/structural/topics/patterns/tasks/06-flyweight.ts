import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-flyweight',
	title: 'Flyweight Pattern',
	difficulty: 'hard',
	tags: ['go', 'design-patterns', 'structural', 'flyweight'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Flyweight pattern in Go - use sharing to support large numbers of fine-grained objects efficiently.

The Flyweight pattern reduces memory usage by sharing as much data as possible between similar objects. It separates object state into intrinsic (shared, immutable) and extrinsic (unique per instance) parts.

**You will implement:**

1. **TreeType struct** - Intrinsic state (shared data: name, color, texture)
2. **Tree struct** - Extrinsic state (unique per instance: x, y coordinates)
3. **TreeFactory** - Creates and reuses TreeType instances via caching

**Example Usage:**

\`\`\`go
// Create factory that manages flyweight objects
factory := NewTreeFactory()	// manages TreeType cache

// Create trees - both share the same TreeType flyweight
tree1 := factory.GetTree(10, 20, "Oak", "green", "rough")	// creates new TreeType
tree2 := factory.GetTree(30, 40, "Oak", "green", "rough")	// reuses existing TreeType
tree3 := factory.GetTree(50, 60, "Pine", "dark_green", "smooth")	// creates new TreeType

// Memory savings: only 2 TreeType objects for 3 trees
factory.GetTreeTypeCount()	// returns 2 (Oak and Pine)

// Each tree can be rendered at its unique position
tree1.Draw()	// "Drawing Oak tree at (10,20)"
tree2.Draw()	// "Drawing Oak tree at (30,40)"
tree3.Draw()	// "Drawing Pine tree at (50,60)"
\`\`\``,
	initialCode: `package patterns

import "fmt"

type TreeType struct {
	Name    string
	Color   string
	Texture string
}

func (t *TreeType) Draw(x, y int) string {
}

type Tree struct {
	X        int
	Y        int
}

func (t *Tree) Draw() string {
}

type TreeFactory struct {
	treeTypes map[string]*TreeType
}

func NewTreeFactory() *TreeFactory {
	}
}

func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
}

func (f *TreeFactory) GetTree(x, y int, name, color, texture string) *Tree {
}

func (f *TreeFactory) GetTreeTypeCount() int {
}`,
	solutionCode: `package patterns

import "fmt"

// TreeType contains intrinsic (shared) state
type TreeType struct {	// flyweight - shared between many Tree instances
	Name    string	// tree species name (shared)
	Color   string	// leaf/needle color (shared)
	Texture string	// bark texture (shared)
}

// Draw renders tree at position
func (t *TreeType) Draw(x, y int) string {	// extrinsic state (x, y) passed as parameters
	return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)	// combine shared and unique data
}

// Tree contains extrinsic (unique) state
type Tree struct {	// context object - holds unique data and reference to flyweight
	X        int	// unique x-coordinate
	Y        int	// unique y-coordinate
	TreeType *TreeType	// pointer to shared flyweight
}

// Draw delegates to TreeType
func (t *Tree) Draw() string {	// combines context (x, y) with flyweight
	return t.TreeType.Draw(t.X, t.Y)	// pass extrinsic state to flyweight method
}

// TreeFactory manages TreeType instances
type TreeFactory struct {	// flyweight factory - ensures flyweight sharing
	treeTypes map[string]*TreeType	// cache of flyweight objects
}

// NewTreeFactory creates a new factory
func NewTreeFactory() *TreeFactory {	// factory constructor
	return &TreeFactory{	// initialize with empty cache
		treeTypes: make(map[string]*TreeType),	// map for O(1) lookup
	}
}

// getTreeType returns existing or creates new TreeType
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {	// flyweight lookup/creation
	key := fmt.Sprintf("%s_%s_%s", name, color, texture)	// create unique key from intrinsic state
	if treeType, exists := f.treeTypes[key]; exists {	// check if flyweight already exists
		return treeType	// return cached flyweight (memory savings)
	}
	treeType := &TreeType{	// create new flyweight if not found
		Name:    name,	// set intrinsic state
		Color:   color,	// set intrinsic state
		Texture: texture,	// set intrinsic state
	}
	f.treeTypes[key] = treeType	// cache the new flyweight
	return treeType	// return newly created flyweight
}

// GetTree creates Tree with shared TreeType
func (f *TreeFactory) GetTree(x, y int, name, color, texture string) *Tree {	// create context with flyweight
	treeType := f.getTreeType(name, color, texture)	// get or create flyweight
	return &Tree{	// create new context object
		X:        x,	// unique x position
		Y:        y,	// unique y position
		TreeType: treeType,	// reference to shared flyweight
	}
}

// GetTreeTypeCount returns number of unique TreeTypes
func (f *TreeFactory) GetTreeTypeCount() int {	// for monitoring memory savings
	return len(f.treeTypes)	// number of cached flyweights
}`,
	hint1: `TreeType.Draw receives extrinsic state (coordinates) as parameters and combines it with intrinsic state (Name):
\`\`\`go
func (t *TreeType) Draw(x, y int) string {
    return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)
}
\`\`\`

Tree.Draw simply delegates to TreeType.Draw, passing its unique coordinates:
\`\`\`go
func (t *Tree) Draw() string {
    return t.TreeType.Draw(t.X, t.Y)
}
\`\`\``,
	hint2: `getTreeType is the key to Flyweight - it caches and reuses flyweight objects:
\`\`\`go
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := fmt.Sprintf("%s_%s_%s", name, color, texture)  // unique key from intrinsic state
    if treeType, exists := f.treeTypes[key]; exists {
        return treeType  // return cached flyweight
    }
    treeType := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = treeType  // cache new flyweight
    return treeType
}
\`\`\`

GetTree creates the context (Tree) with the flyweight reference.`,
	whyItMatters: `## Why Flyweight Exists

The Flyweight pattern solves the problem of high memory consumption when creating millions of similar objects. Without it, each object stores all its data independently, leading to massive memory waste.

**Problem - Without Flyweight:**
\`\`\`go
// Each tree stores all data independently
type Tree struct {
    X, Y    int
    Name    string   // "Oak" duplicated millions of times
    Color   string   // "green" duplicated millions of times
    Texture string   // "rough" duplicated millions of times
    // 3 strings × 1 million trees = massive memory waste
}

// 1 million trees in a forest
trees := make([]*Tree, 1_000_000)
for i := range trees {
    trees[i] = &Tree{X: i, Y: i*2, Name: "Oak", Color: "green", Texture: "rough"}
}
// Memory: 1M × (16 bytes + 3 strings) = huge!
\`\`\`

**Solution - With Flyweight:**
\`\`\`go
// Shared flyweight (1 instance for all Oak trees)
type TreeType struct {
    Name, Color, Texture string  // stored once
}

// Context with unique data only
type Tree struct {
    X, Y     int        // unique per tree
    TreeType *TreeType  // pointer to shared data (8 bytes)
}

// Memory: 1M × 24 bytes + 1 TreeType = minimal!
\`\`\`

## Real-World Go Examples

**1. Text Editor Character Formatting:**
\`\`\`go
type CharacterStyle struct {  // flyweight
    Font     string
    Size     int
    Bold     bool
    Color    string
}

type Character struct {  // context
    Char  rune
    Style *CharacterStyle
}

type StyleFactory struct {
    styles map[string]*CharacterStyle
}

func (f *StyleFactory) GetStyle(font string, size int, bold bool, color string) *CharacterStyle {
    key := fmt.Sprintf("%s_%d_%t_%s", font, size, bold, color)
    if style, ok := f.styles[key]; ok {
        return style
    }
    style := &CharacterStyle{Font: font, Size: size, Bold: bold, Color: color}
    f.styles[key] = style
    return style
}

// Document with 100,000 characters uses maybe 10 unique styles
\`\`\`

**2. Game Particle System:**
\`\`\`go
type ParticleType struct {  // flyweight - heavy data
    Sprite  *Texture  // large texture data
    Color   Color
    Effects []Effect
}

type Particle struct {  // context - lightweight
    X, Y      float32
    VelocityX float32
    VelocityY float32
    Type      *ParticleType  // pointer to shared heavy data
}

type ParticleFactory struct {
    types map[string]*ParticleType
}

// 10,000 fire particles share 1 ParticleType
func SpawnFireParticles(count int) []*Particle {
    fireType := factory.GetParticleType("fire")
    particles := make([]*Particle, count)
    for i := range particles {
        particles[i] = &Particle{
            X: rand.Float32() * 100,
            Y: rand.Float32() * 100,
            Type: fireType,  // all share same type
        }
    }
    return particles
}
\`\`\`

## Production Pattern: Connection Pool Metadata

\`\`\`go
// ConnectionConfig is the flyweight - rarely changes
type ConnectionConfig struct {
    Host           string
    Port           int
    Database       string
    SSLMode        string
    MaxIdleConns   int
    ConnectTimeout time.Duration
}

// Connection is the context - unique per connection
type Connection struct {
    ID        string
    CreatedAt time.Time
    LastUsed  time.Time
    InUse     bool
    Config    *ConnectionConfig  // shared config
    conn      *sql.Conn          // actual connection
}

type ConnectionPool struct {
    configs     map[string]*ConnectionConfig  // flyweight cache
    connections []*Connection
    mu          sync.RWMutex
}

func (p *ConnectionPool) GetConfig(host string, port int, db, ssl string) *ConnectionConfig {
    p.mu.RLock()
    key := fmt.Sprintf("%s:%d/%s?ssl=%s", host, port, db, ssl)
    if config, ok := p.configs[key]; ok {
        p.mu.RUnlock()
        return config
    }
    p.mu.RUnlock()

    p.mu.Lock()
    defer p.mu.Unlock()
    // Double-check after acquiring write lock
    if config, ok := p.configs[key]; ok {
        return config
    }
    config := &ConnectionConfig{
        Host: host, Port: port, Database: db, SSLMode: ssl,
        MaxIdleConns: 10, ConnectTimeout: 5 * time.Second,
    }
    p.configs[key] = config
    return config
}

func (p *ConnectionPool) NewConnection(host string, port int, db, ssl string) *Connection {
    config := p.GetConfig(host, port, db, ssl)
    return &Connection{
        ID:        uuid.New().String(),
        CreatedAt: time.Now(),
        Config:    config,  // shared config
    }
}
\`\`\`

## Common Mistakes

**1. Making Flyweight Mutable:**
\`\`\`go
// Bad - mutable flyweight causes bugs
type TreeType struct {
    Name  string
    Color string  // if one tree changes color, all trees change!
}

func (t *TreeType) SetColor(color string) {
    t.Color = color  // affects ALL trees sharing this flyweight!
}

// Good - flyweight is immutable
type TreeType struct {
    name  string  // unexported = immutable from outside
    color string
}

// If you need different color, get/create a different flyweight
tree := factory.GetTree(x, y, "Oak", "yellow", "rough")  // new flyweight
\`\`\`

**2. Storing Extrinsic State in Flyweight:**
\`\`\`go
// Bad - position stored in flyweight
type TreeType struct {
    Name    string
    Color   string
    X, Y    int  // wrong! this is extrinsic state
}

// Good - only intrinsic state in flyweight
type TreeType struct {
    Name  string  // shared
    Color string  // shared
}

type Tree struct {
    X, Y     int        // extrinsic (unique)
    TreeType *TreeType  // reference to flyweight
}
\`\`\`

**3. Not Using Thread-Safe Factory:**
\`\`\`go
// Bad - race condition in concurrent access
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)
    if t, ok := f.treeTypes[key]; ok {  // read
        return t
    }
    t := &TreeType{...}
    f.treeTypes[key] = t  // write - race condition!
    return t
}

// Good - thread-safe with sync.RWMutex
type TreeFactory struct {
    treeTypes map[string]*TreeType
    mu        sync.RWMutex
}

func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)

    f.mu.RLock()
    if t, ok := f.treeTypes[key]; ok {
        f.mu.RUnlock()
        return t
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()
    // Double-check after acquiring write lock
    if t, ok := f.treeTypes[key]; ok {
        return t
    }
    t := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = t
    return t
}
\`\`\`

**Key Principles:**
- Intrinsic state (flyweight) must be immutable and shareable
- Extrinsic state (context) is unique per instance and passed to flyweight methods
- Factory ensures flyweight sharing through caching
- Use when you have many similar objects with common repeating data
- Thread-safety is critical for concurrent factory access`,
	order: 5,
	testCode: `package patterns

import (
	"testing"
)

// Test1: TreeType.Draw returns formatted string
func Test1(t *testing.T) {
	tt := &TreeType{Name: "Oak", Color: "green", Texture: "rough"}
	result := tt.Draw(10, 20)
	if result != "Drawing Oak tree at (10,20)" {
		t.Errorf("Unexpected result: %s", result)
	}
}

// Test2: Tree.Draw delegates to TreeType
func Test2(t *testing.T) {
	tt := &TreeType{Name: "Pine"}
	tree := &Tree{X: 5, Y: 10, TreeType: tt}
	result := tree.Draw()
	if result != "Drawing Pine tree at (5,10)" {
		t.Errorf("Unexpected result: %s", result)
	}
}

// Test3: NewTreeFactory returns non-nil
func Test3(t *testing.T) {
	f := NewTreeFactory()
	if f == nil {
		t.Error("NewTreeFactory should return non-nil")
	}
}

// Test4: GetTree creates tree with correct position
func Test4(t *testing.T) {
	f := NewTreeFactory()
	tree := f.GetTree(100, 200, "Oak", "green", "rough")
	if tree.X != 100 || tree.Y != 200 {
		t.Error("Tree should have correct position")
	}
}

// Test5: GetTreeTypeCount starts at 0
func Test5(t *testing.T) {
	f := NewTreeFactory()
	if f.GetTreeTypeCount() != 0 {
		t.Error("New factory should have 0 tree types")
	}
}

// Test6: Same tree type is reused
func Test6(t *testing.T) {
	f := NewTreeFactory()
	f.GetTree(0, 0, "Oak", "green", "rough")
	f.GetTree(1, 1, "Oak", "green", "rough")
	if f.GetTreeTypeCount() != 1 {
		t.Error("Same tree type should be reused")
	}
}

// Test7: Different tree types are created
func Test7(t *testing.T) {
	f := NewTreeFactory()
	f.GetTree(0, 0, "Oak", "green", "rough")
	f.GetTree(1, 1, "Pine", "dark_green", "smooth")
	if f.GetTreeTypeCount() != 2 {
		t.Error("Different tree types should be created")
	}
}

// Test8: Trees share same TreeType instance
func Test8(t *testing.T) {
	f := NewTreeFactory()
	t1 := f.GetTree(0, 0, "Oak", "green", "rough")
	t2 := f.GetTree(1, 1, "Oak", "green", "rough")
	if t1.TreeType != t2.TreeType {
		t.Error("Trees should share same TreeType instance")
	}
}

// Test9: TreeType struct has Name field
func Test9(t *testing.T) {
	tt := TreeType{Name: "Test", Color: "red", Texture: "smooth"}
	if tt.Name != "Test" {
		t.Error("TreeType should have Name field")
	}
}

// Test10: Tree struct has TreeType reference
func Test10(t *testing.T) {
	tt := &TreeType{Name: "Oak"}
	tree := Tree{X: 0, Y: 0, TreeType: tt}
	if tree.TreeType == nil {
		t.Error("Tree should have TreeType reference")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Flyweight (Приспособленец)',
			description: `Реализуйте паттерн Flyweight на Go — используйте разделение для эффективной поддержки большого количества мелких объектов.

Паттерн Flyweight сокращает использование памяти, разделяя максимум данных между похожими объектами. Он разделяет состояние объекта на внутреннее (разделяемое, неизменяемое) и внешнее (уникальное для экземпляра).

**Вы реализуете:**

1. **Структура TreeType** - Внутреннее состояние (разделяемые данные: имя, цвет, текстура)
2. **Структура Tree** - Внешнее состояние (уникальное для экземпляра: координаты x, y)
3. **TreeFactory** - Создаёт и переиспользует экземпляры TreeType через кэширование

**Пример использования:**

\`\`\`go
// Создаём фабрику, управляющую flyweight-объектами
factory := NewTreeFactory()	// управляет кэшем TreeType

// Создаём деревья — оба разделяют один и тот же flyweight TreeType
tree1 := factory.GetTree(10, 20, "Oak", "green", "rough")	// создаёт новый TreeType
tree2 := factory.GetTree(30, 40, "Oak", "green", "rough")	// переиспользует существующий TreeType
tree3 := factory.GetTree(50, 60, "Pine", "dark_green", "smooth")	// создаёт новый TreeType

// Экономия памяти: только 2 объекта TreeType для 3 деревьев
factory.GetTreeTypeCount()	// возвращает 2 (Oak и Pine)

// Каждое дерево может отрисовываться в своей уникальной позиции
tree1.Draw()	// "Drawing Oak tree at (10,20)"
tree2.Draw()	// "Drawing Oak tree at (30,40)"
tree3.Draw()	// "Drawing Pine tree at (50,60)"
\`\`\``,
			hint1: `TreeType.Draw получает внешнее состояние (координаты) как параметры и комбинирует его с внутренним состоянием (Name):
\`\`\`go
func (t *TreeType) Draw(x, y int) string {
    return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)
}
\`\`\`

Tree.Draw просто делегирует TreeType.Draw, передавая свои уникальные координаты:
\`\`\`go
func (t *Tree) Draw() string {
    return t.TreeType.Draw(t.X, t.Y)
}
\`\`\``,
			hint2: `getTreeType — ключ к Flyweight — он кэширует и переиспользует flyweight-объекты:
\`\`\`go
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := fmt.Sprintf("%s_%s_%s", name, color, texture)  // уникальный ключ из внутреннего состояния
    if treeType, exists := f.treeTypes[key]; exists {
        return treeType  // возвращаем кэшированный flyweight
    }
    treeType := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = treeType  // кэшируем новый flyweight
    return treeType
}
\`\`\`

GetTree создаёт контекст (Tree) со ссылкой на flyweight.`,
			whyItMatters: `## Зачем нужен Flyweight

Паттерн Flyweight решает проблему высокого потребления памяти при создании миллионов похожих объектов. Без него каждый объект хранит все свои данные независимо, что приводит к огромной трате памяти.

**Проблема — без Flyweight:**
\`\`\`go
// Каждое дерево хранит все данные независимо
type Tree struct {
    X, Y    int
    Name    string   // "Oak" дублируется миллионы раз
    Color   string   // "green" дублируется миллионы раз
    Texture string   // "rough" дублируется миллионы раз
    // 3 строки × 1 миллион деревьев = огромная трата памяти
}

// 1 миллион деревьев в лесу
trees := make([]*Tree, 1_000_000)
for i := range trees {
    trees[i] = &Tree{X: i, Y: i*2, Name: "Oak", Color: "green", Texture: "rough"}
}
// Память: 1M × (16 байт + 3 строки) = огромно!
\`\`\`

**Решение — с Flyweight:**
\`\`\`go
// Разделяемый flyweight (1 экземпляр для всех дубов)
type TreeType struct {
    Name, Color, Texture string  // хранится один раз
}

// Контекст только с уникальными данными
type Tree struct {
    X, Y     int        // уникально для каждого дерева
    TreeType *TreeType  // указатель на разделяемые данные (8 байт)
}

// Память: 1M × 24 байта + 1 TreeType = минимально!
\`\`\`

## Реальные примеры на Go

**1. Форматирование символов в текстовом редакторе:**
\`\`\`go
type CharacterStyle struct {  // flyweight
    Font     string
    Size     int
    Bold     bool
    Color    string
}

type Character struct {  // контекст
    Char  rune
    Style *CharacterStyle
}

type StyleFactory struct {
    styles map[string]*CharacterStyle
}

func (f *StyleFactory) GetStyle(font string, size int, bold bool, color string) *CharacterStyle {
    key := fmt.Sprintf("%s_%d_%t_%s", font, size, bold, color)
    if style, ok := f.styles[key]; ok {
        return style
    }
    style := &CharacterStyle{Font: font, Size: size, Bold: bold, Color: color}
    f.styles[key] = style
    return style
}

// Документ с 100 000 символов использует может быть 10 уникальных стилей
\`\`\`

**2. Система частиц в игре:**
\`\`\`go
type ParticleType struct {  // flyweight — тяжёлые данные
    Sprite  *Texture  // большие данные текстуры
    Color   Color
    Effects []Effect
}

type Particle struct {  // контекст — лёгкий
    X, Y      float32
    VelocityX float32
    VelocityY float32
    Type      *ParticleType  // указатель на разделяемые тяжёлые данные
}

type ParticleFactory struct {
    types map[string]*ParticleType
}

// 10 000 частиц огня разделяют 1 ParticleType
func SpawnFireParticles(count int) []*Particle {
    fireType := factory.GetParticleType("fire")
    particles := make([]*Particle, count)
    for i := range particles {
        particles[i] = &Particle{
            X: rand.Float32() * 100,
            Y: rand.Float32() * 100,
            Type: fireType,  // все разделяют один тип
        }
    }
    return particles
}
\`\`\`

## Продакшен паттерн: Метаданные пула соединений

\`\`\`go
// ConnectionConfig — flyweight — редко меняется
type ConnectionConfig struct {
    Host           string
    Port           int
    Database       string
    SSLMode        string
    MaxIdleConns   int
    ConnectTimeout time.Duration
}

// Connection — контекст — уникален для каждого соединения
type Connection struct {
    ID        string
    CreatedAt time.Time
    LastUsed  time.Time
    InUse     bool
    Config    *ConnectionConfig  // разделяемая конфигурация
    conn      *sql.Conn          // реальное соединение
}

type ConnectionPool struct {
    configs     map[string]*ConnectionConfig  // кэш flyweight
    connections []*Connection
    mu          sync.RWMutex
}

func (p *ConnectionPool) GetConfig(host string, port int, db, ssl string) *ConnectionConfig {
    p.mu.RLock()
    key := fmt.Sprintf("%s:%d/%s?ssl=%s", host, port, db, ssl)
    if config, ok := p.configs[key]; ok {
        p.mu.RUnlock()
        return config
    }
    p.mu.RUnlock()

    p.mu.Lock()
    defer p.mu.Unlock()
    // Двойная проверка после получения блокировки на запись
    if config, ok := p.configs[key]; ok {
        return config
    }
    config := &ConnectionConfig{
        Host: host, Port: port, Database: db, SSLMode: ssl,
        MaxIdleConns: 10, ConnectTimeout: 5 * time.Second,
    }
    p.configs[key] = config
    return config
}

func (p *ConnectionPool) NewConnection(host string, port int, db, ssl string) *Connection {
    config := p.GetConfig(host, port, db, ssl)
    return &Connection{
        ID:        uuid.New().String(),
        CreatedAt: time.Now(),
        Config:    config,  // разделяемая конфигурация
    }
}
\`\`\`

## Частые ошибки

**1. Делать Flyweight изменяемым:**
\`\`\`go
// Плохо — изменяемый flyweight вызывает баги
type TreeType struct {
    Name  string
    Color string  // если одно дерево изменит цвет, изменятся все деревья!
}

func (t *TreeType) SetColor(color string) {
    t.Color = color  // влияет на ВСЕ деревья, разделяющие этот flyweight!
}

// Хорошо — flyweight неизменяем
type TreeType struct {
    name  string  // неэкспортированное = неизменяемое снаружи
    color string
}

// Если нужен другой цвет, получите/создайте другой flyweight
tree := factory.GetTree(x, y, "Oak", "yellow", "rough")  // новый flyweight
\`\`\`

**2. Хранение внешнего состояния в Flyweight:**
\`\`\`go
// Плохо — позиция хранится в flyweight
type TreeType struct {
    Name    string
    Color   string
    X, Y    int  // неправильно! это внешнее состояние
}

// Хорошо — только внутреннее состояние в flyweight
type TreeType struct {
    Name  string  // разделяемое
    Color string  // разделяемое
}

type Tree struct {
    X, Y     int        // внешнее (уникальное)
    TreeType *TreeType  // ссылка на flyweight
}
\`\`\`

**3. Не использовать потокобезопасную фабрику:**
\`\`\`go
// Плохо — гонка данных при параллельном доступе
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)
    if t, ok := f.treeTypes[key]; ok {  // чтение
        return t
    }
    t := &TreeType{...}
    f.treeTypes[key] = t  // запись — гонка данных!
    return t
}

// Хорошо — потокобезопасно с sync.RWMutex
type TreeFactory struct {
    treeTypes map[string]*TreeType
    mu        sync.RWMutex
}

func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)

    f.mu.RLock()
    if t, ok := f.treeTypes[key]; ok {
        f.mu.RUnlock()
        return t
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()
    // Двойная проверка после получения блокировки на запись
    if t, ok := f.treeTypes[key]; ok {
        return t
    }
    t := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = t
    return t
}
\`\`\`

**Ключевые принципы:**
- Внутреннее состояние (flyweight) должно быть неизменяемым и разделяемым
- Внешнее состояние (контекст) уникально для экземпляра и передаётся в методы flyweight
- Фабрика обеспечивает разделение flyweight через кэширование
- Используйте, когда у вас много похожих объектов с общими повторяющимися данными
- Потокобезопасность критична для параллельного доступа к фабрике`,
			solutionCode: `package patterns

import "fmt"

// TreeType содержит внутреннее (разделяемое) состояние
type TreeType struct {	// flyweight — разделяется между многими экземплярами Tree
	Name    string	// название вида дерева (разделяемое)
	Color   string	// цвет листьев/хвои (разделяемый)
	Texture string	// текстура коры (разделяемая)
}

// Draw отрисовывает дерево в позиции
func (t *TreeType) Draw(x, y int) string {	// внешнее состояние (x, y) передаётся как параметры
	return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)	// комбинируем разделяемые и уникальные данные
}

// Tree содержит внешнее (уникальное) состояние
type Tree struct {	// объект контекста — хранит уникальные данные и ссылку на flyweight
	X        int	// уникальная x-координата
	Y        int	// уникальная y-координата
	TreeType *TreeType	// указатель на разделяемый flyweight
}

// Draw делегирует TreeType
func (t *Tree) Draw() string {	// комбинирует контекст (x, y) с flyweight
	return t.TreeType.Draw(t.X, t.Y)	// передаёт внешнее состояние в метод flyweight
}

// TreeFactory управляет экземплярами TreeType
type TreeFactory struct {	// фабрика flyweight — обеспечивает разделение flyweight
	treeTypes map[string]*TreeType	// кэш flyweight-объектов
}

// NewTreeFactory создаёт новую фабрику
func NewTreeFactory() *TreeFactory {	// конструктор фабрики
	return &TreeFactory{	// инициализация с пустым кэшем
		treeTypes: make(map[string]*TreeType),	// map для O(1) поиска
	}
}

// getTreeType возвращает существующий или создаёт новый TreeType
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {	// поиск/создание flyweight
	key := fmt.Sprintf("%s_%s_%s", name, color, texture)	// создаём уникальный ключ из внутреннего состояния
	if treeType, exists := f.treeTypes[key]; exists {	// проверяем существует ли flyweight
		return treeType	// возвращаем кэшированный flyweight (экономия памяти)
	}
	treeType := &TreeType{	// создаём новый flyweight если не найден
		Name:    name,	// устанавливаем внутреннее состояние
		Color:   color,	// устанавливаем внутреннее состояние
		Texture: texture,	// устанавливаем внутреннее состояние
	}
	f.treeTypes[key] = treeType	// кэшируем новый flyweight
	return treeType	// возвращаем только что созданный flyweight
}

// GetTree создаёт Tree с разделяемым TreeType
func (f *TreeFactory) GetTree(x, y int, name, color, texture string) *Tree {	// создаём контекст с flyweight
	treeType := f.getTreeType(name, color, texture)	// получаем или создаём flyweight
	return &Tree{	// создаём новый объект контекста
		X:        x,	// уникальная позиция x
		Y:        y,	// уникальная позиция y
		TreeType: treeType,	// ссылка на разделяемый flyweight
	}
}

// GetTreeTypeCount возвращает количество уникальных TreeType
func (f *TreeFactory) GetTreeTypeCount() int {	// для мониторинга экономии памяти
	return len(f.treeTypes)	// количество кэшированных flyweight
}`
		},
		uz: {
			title: 'Flyweight (Yengil vazn) Pattern',
			description: `Go tilida Flyweight patternini amalga oshiring — ko'p sonli mayda ob'ektlarni samarali qo'llab-quvvatlash uchun ulashishdan foydalaning.

Flyweight patterni o'xshash ob'ektlar orasida imkon qadar ko'p ma'lumotlarni ulashish orqali xotira sarfini kamaytiradi. U ob'ekt holatini ichki (ulashiladigan, o'zgarmas) va tashqi (har bir nusxa uchun noyob) qismlarga ajratadi.

**Siz amalga oshirasiz:**

1. **TreeType strukturasi** - Ichki holat (ulashiladigan ma'lumotlar: nom, rang, tekstura)
2. **Tree strukturasi** - Tashqi holat (har bir nusxa uchun noyob: x, y koordinatalari)
3. **TreeFactory** - Keshlash orqali TreeType nusxalarini yaratadi va qayta ishlatadi

**Foydalanish namunasi:**

\`\`\`go
// Flyweight ob'ektlarini boshqaradigan fabrika yaratish
factory := NewTreeFactory()	// TreeType keshini boshqaradi

// Daraxtlar yaratish — ikkisi ham bir xil TreeType flyweight ni ulashadi
tree1 := factory.GetTree(10, 20, "Oak", "green", "rough")	// yangi TreeType yaratadi
tree2 := factory.GetTree(30, 40, "Oak", "green", "rough")	// mavjud TreeType ni qayta ishlatadi
tree3 := factory.GetTree(50, 60, "Pine", "dark_green", "smooth")	// yangi TreeType yaratadi

// Xotira tejalishi: 3 ta daraxt uchun faqat 2 ta TreeType ob'ekti
factory.GetTreeTypeCount()	// 2 qaytaradi (Oak va Pine)

// Har bir daraxt o'zining noyob pozitsiyasida chizilishi mumkin
tree1.Draw()	// "Drawing Oak tree at (10,20)"
tree2.Draw()	// "Drawing Oak tree at (30,40)"
tree3.Draw()	// "Drawing Pine tree at (50,60)"
\`\`\``,
			hint1: `TreeType.Draw tashqi holatni (koordinatalar) parametr sifatida qabul qiladi va uni ichki holat (Name) bilan birlashtiradi:
\`\`\`go
func (t *TreeType) Draw(x, y int) string {
    return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)
}
\`\`\`

Tree.Draw shunchaki TreeType.Draw ga delegatsiya qiladi, o'zining noyob koordinatalarini uzatadi:
\`\`\`go
func (t *Tree) Draw() string {
    return t.TreeType.Draw(t.X, t.Y)
}
\`\`\``,
			hint2: `getTreeType Flyweight ning kalitidir — u flyweight ob'ektlarini keshlaydi va qayta ishlatadi:
\`\`\`go
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := fmt.Sprintf("%s_%s_%s", name, color, texture)  // ichki holatdan noyob kalit
    if treeType, exists := f.treeTypes[key]; exists {
        return treeType  // keshlangan flyweight ni qaytarish
    }
    treeType := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = treeType  // yangi flyweight ni keshlash
    return treeType
}
\`\`\`

GetTree flyweight havolasi bilan kontekst (Tree) yaratadi.`,
			whyItMatters: `## Nega Flyweight kerak

Flyweight patterni millionlab o'xshash ob'ektlarni yaratishda yuqori xotira sarfi muammosini hal qiladi. Busiz har bir ob'ekt barcha ma'lumotlarini mustaqil saqlaydi, bu katta xotira isrofiga olib keladi.

**Muammo — Flyweight siz:**
\`\`\`go
// Har bir daraxt barcha ma'lumotlarni mustaqil saqlaydi
type Tree struct {
    X, Y    int
    Name    string   // "Oak" millionlab marta takrorlanadi
    Color   string   // "green" millionlab marta takrorlanadi
    Texture string   // "rough" millionlab marta takrorlanadi
    // 3 ta satr × 1 million daraxt = katta xotira isrofi
}

// O'rmonda 1 million daraxt
trees := make([]*Tree, 1_000_000)
for i := range trees {
    trees[i] = &Tree{X: i, Y: i*2, Name: "Oak", Color: "green", Texture: "rough"}
}
// Xotira: 1M × (16 bayt + 3 satr) = juda katta!
\`\`\`

**Yechim — Flyweight bilan:**
\`\`\`go
// Ulashiladigan flyweight (barcha eman daraxtlari uchun 1 ta nusxa)
type TreeType struct {
    Name, Color, Texture string  // bir marta saqlanadi
}

// Faqat noyob ma'lumotlari bilan kontekst
type Tree struct {
    X, Y     int        // har bir daraxt uchun noyob
    TreeType *TreeType  // ulashiladigan ma'lumotlarga ko'rsatkich (8 bayt)
}

// Xotira: 1M × 24 bayt + 1 TreeType = minimal!
\`\`\`

## Go dagi real dunyo misollar

**1. Matn muharririda belgilar formatlash:**
\`\`\`go
type CharacterStyle struct {  // flyweight
    Font     string
    Size     int
    Bold     bool
    Color    string
}

type Character struct {  // kontekst
    Char  rune
    Style *CharacterStyle
}

type StyleFactory struct {
    styles map[string]*CharacterStyle
}

func (f *StyleFactory) GetStyle(font string, size int, bold bool, color string) *CharacterStyle {
    key := fmt.Sprintf("%s_%d_%t_%s", font, size, bold, color)
    if style, ok := f.styles[key]; ok {
        return style
    }
    style := &CharacterStyle{Font: font, Size: size, Bold: bold, Color: color}
    f.styles[key] = style
    return style
}

// 100 000 belgili hujjat ehtimol 10 ta noyob uslubdan foydalanadi
\`\`\`

**2. O'yindagi zarracha tizimi:**
\`\`\`go
type ParticleType struct {  // flyweight — og'ir ma'lumotlar
    Sprite  *Texture  // katta tekstura ma'lumotlari
    Color   Color
    Effects []Effect
}

type Particle struct {  // kontekst — yengil
    X, Y      float32
    VelocityX float32
    VelocityY float32
    Type      *ParticleType  // ulashiladigan og'ir ma'lumotlarga ko'rsatkich
}

type ParticleFactory struct {
    types map[string]*ParticleType
}

// 10 000 olov zarrachasi 1 ta ParticleType ni ulashadi
func SpawnFireParticles(count int) []*Particle {
    fireType := factory.GetParticleType("fire")
    particles := make([]*Particle, count)
    for i := range particles {
        particles[i] = &Particle{
            X: rand.Float32() * 100,
            Y: rand.Float32() * 100,
            Type: fireType,  // hammasi bir xil turni ulashadi
        }
    }
    return particles
}
\`\`\`

## Prodakshen pattern: Ulanish havzasi metadata

\`\`\`go
// ConnectionConfig — flyweight — kamdan-kam o'zgaradi
type ConnectionConfig struct {
    Host           string
    Port           int
    Database       string
    SSLMode        string
    MaxIdleConns   int
    ConnectTimeout time.Duration
}

// Connection — kontekst — har bir ulanish uchun noyob
type Connection struct {
    ID        string
    CreatedAt time.Time
    LastUsed  time.Time
    InUse     bool
    Config    *ConnectionConfig  // ulashiladigan konfiguratsiya
    conn      *sql.Conn          // haqiqiy ulanish
}

type ConnectionPool struct {
    configs     map[string]*ConnectionConfig  // flyweight keshi
    connections []*Connection
    mu          sync.RWMutex
}

func (p *ConnectionPool) GetConfig(host string, port int, db, ssl string) *ConnectionConfig {
    p.mu.RLock()
    key := fmt.Sprintf("%s:%d/%s?ssl=%s", host, port, db, ssl)
    if config, ok := p.configs[key]; ok {
        p.mu.RUnlock()
        return config
    }
    p.mu.RUnlock()

    p.mu.Lock()
    defer p.mu.Unlock()
    // Yozish qulfini olgandan keyin ikki marta tekshirish
    if config, ok := p.configs[key]; ok {
        return config
    }
    config := &ConnectionConfig{
        Host: host, Port: port, Database: db, SSLMode: ssl,
        MaxIdleConns: 10, ConnectTimeout: 5 * time.Second,
    }
    p.configs[key] = config
    return config
}

func (p *ConnectionPool) NewConnection(host string, port int, db, ssl string) *Connection {
    config := p.GetConfig(host, port, db, ssl)
    return &Connection{
        ID:        uuid.New().String(),
        CreatedAt: time.Now(),
        Config:    config,  // ulashiladigan konfiguratsiya
    }
}
\`\`\`

## Keng tarqalgan xatolar

**1. Flyweight ni o'zgaruvchan qilish:**
\`\`\`go
// Yomon — o'zgaruvchan flyweight xatolarga olib keladi
type TreeType struct {
    Name  string
    Color string  // agar bitta daraxt rangini o'zgartirsa, barcha daraxtlar o'zgaradi!
}

func (t *TreeType) SetColor(color string) {
    t.Color = color  // bu flyweight ni ulashayotgan BARCHA daraxtlarga ta'sir qiladi!
}

// Yaxshi — flyweight o'zgarmas
type TreeType struct {
    name  string  // eksport qilinmagan = tashqaridan o'zgarmas
    color string
}

// Agar boshqa rang kerak bo'lsa, boshqa flyweight oling/yarating
tree := factory.GetTree(x, y, "Oak", "yellow", "rough")  // yangi flyweight
\`\`\`

**2. Tashqi holatni Flyweight da saqlash:**
\`\`\`go
// Yomon — pozitsiya flyweight da saqlanadi
type TreeType struct {
    Name    string
    Color   string
    X, Y    int  // noto'g'ri! bu tashqi holat
}

// Yaxshi — flyweight da faqat ichki holat
type TreeType struct {
    Name  string  // ulashiladigan
    Color string  // ulashiladigan
}

type Tree struct {
    X, Y     int        // tashqi (noyob)
    TreeType *TreeType  // flyweight ga havola
}
\`\`\`

**3. Thread-safe bo'lmagan fabrika ishlatish:**
\`\`\`go
// Yomon — parallel kirishda poyga holati
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)
    if t, ok := f.treeTypes[key]; ok {  // o'qish
        return t
    }
    t := &TreeType{...}
    f.treeTypes[key] = t  // yozish — poyga holati!
    return t
}

// Yaxshi — sync.RWMutex bilan thread-safe
type TreeFactory struct {
    treeTypes map[string]*TreeType
    mu        sync.RWMutex
}

func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {
    key := makeKey(name, color, texture)

    f.mu.RLock()
    if t, ok := f.treeTypes[key]; ok {
        f.mu.RUnlock()
        return t
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()
    // Yozish qulfini olgandan keyin ikki marta tekshirish
    if t, ok := f.treeTypes[key]; ok {
        return t
    }
    t := &TreeType{Name: name, Color: color, Texture: texture}
    f.treeTypes[key] = t
    return t
}
\`\`\`

**Asosiy tamoyillar:**
- Ichki holat (flyweight) o'zgarmas va ulashiladigan bo'lishi kerak
- Tashqi holat (kontekst) har bir nusxa uchun noyob va flyweight metodlariga uzatiladi
- Fabrika keshlash orqali flyweight ulashishni ta'minlaydi
- Umumiy takrorlanuvchi ma'lumotlarga ega ko'p o'xshash ob'ektlar mavjud bo'lganda foydalaning
- Fabrikaga parallel kirish uchun thread-xavfsizlik muhim`,
			solutionCode: `package patterns

import "fmt"

// TreeType ichki (ulashiladigan) holatni o'z ichiga oladi
type TreeType struct {	// flyweight — ko'p Tree nusxalari orasida ulashiladi
	Name    string	// daraxt turi nomi (ulashiladigan)
	Color   string	// barg/igna rangi (ulashiladigan)
	Texture string	// po'stloq teksturasi (ulashiladigan)
}

// Draw daraxtni pozitsiyada chizadi
func (t *TreeType) Draw(x, y int) string {	// tashqi holat (x, y) parametr sifatida uzatiladi
	return fmt.Sprintf("Drawing %s tree at (%d,%d)", t.Name, x, y)	// ulashiladigan va noyob ma'lumotlarni birlashtirish
}

// Tree tashqi (noyob) holatni o'z ichiga oladi
type Tree struct {	// kontekst ob'ekti — noyob ma'lumotlar va flyweight ga havolani saqlaydi
	X        int	// noyob x-koordinata
	Y        int	// noyob y-koordinata
	TreeType *TreeType	// ulashiladigan flyweight ga ko'rsatkich
}

// Draw TreeType ga delegatsiya qiladi
func (t *Tree) Draw() string {	// kontekst (x, y) ni flyweight bilan birlashtiradi
	return t.TreeType.Draw(t.X, t.Y)	// tashqi holatni flyweight metodiga uzatish
}

// TreeFactory TreeType nusxalarini boshqaradi
type TreeFactory struct {	// flyweight fabrikasi — flyweight ulashishni ta'minlaydi
	treeTypes map[string]*TreeType	// flyweight ob'ektlari keshi
}

// NewTreeFactory yangi fabrika yaratadi
func NewTreeFactory() *TreeFactory {	// fabrika konstruktori
	return &TreeFactory{	// bo'sh kesh bilan ishga tushirish
		treeTypes: make(map[string]*TreeType),	// O(1) qidirish uchun map
	}
}

// getTreeType mavjudini qaytaradi yoki yangi TreeType yaratadi
func (f *TreeFactory) getTreeType(name, color, texture string) *TreeType {	// flyweight qidirish/yaratish
	key := fmt.Sprintf("%s_%s_%s", name, color, texture)	// ichki holatdan noyob kalit yaratish
	if treeType, exists := f.treeTypes[key]; exists {	// flyweight mavjudligini tekshirish
		return treeType	// keshlangan flyweight ni qaytarish (xotira tejalishi)
	}
	treeType := &TreeType{	// topilmasa yangi flyweight yaratish
		Name:    name,	// ichki holatni o'rnatish
		Color:   color,	// ichki holatni o'rnatish
		Texture: texture,	// ichki holatni o'rnatish
	}
	f.treeTypes[key] = treeType	// yangi flyweight ni keshlash
	return treeType	// yangi yaratilgan flyweight ni qaytarish
}

// GetTree ulashiladigan TreeType bilan Tree yaratadi
func (f *TreeFactory) GetTree(x, y int, name, color, texture string) *Tree {	// flyweight bilan kontekst yaratish
	treeType := f.getTreeType(name, color, texture)	// flyweight olish yoki yaratish
	return &Tree{	// yangi kontekst ob'ekti yaratish
		X:        x,	// noyob x pozitsiyasi
		Y:        y,	// noyob y pozitsiyasi
		TreeType: treeType,	// ulashiladigan flyweight ga havola
	}
}

// GetTreeTypeCount noyob TreeType lar sonini qaytaradi
func (f *TreeFactory) GetTreeTypeCount() int {	// xotira tejalishini kuzatish uchun
	return len(f.treeTypes)	// keshlangan flyweight lar soni
}`
		}
	}
};

export default task;
