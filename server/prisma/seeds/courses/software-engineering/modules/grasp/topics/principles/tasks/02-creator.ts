import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-creator',
	title: 'Creator',
	difficulty: 'easy',
	tags: ['go', 'software-engineering', 'grasp', 'creator'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Creator principle - assign class B the responsibility to create class A if B contains, aggregates, or closely uses A.

**You will implement:**

1. **Library struct** - Contains books collection
2. **AddBook(title, author string) *Book** - Create and add a book (Library creates Books!)
3. **GetBooks() []*Book** - Return all books
4. **Book struct** - Represents a book with ID, title, and author

**Key Concepts:**
- **Creator**: The class that contains or uses objects should create them
- **Natural Responsibility**: Library manages books, so it creates them
- **ID Generation**: Library assigns unique IDs to books it creates

**Example Usage:**

\`\`\`go
library := NewLibrary()

// Library creates books (it contains/manages them)
book1 := library.AddBook("Clean Code", "Robert Martin")
book2 := library.AddBook("Design Patterns", "Gang of Four")

fmt.Println(book1.ID)     // 1
fmt.Println(book1.Title)  // Clean Code

books := library.GetBooks()
fmt.Println(len(books))   // 2
\`\`\`

**Why Creator?**
- **Low Coupling**: Other code doesn't need to know how to create Books
- **Consistency**: Library ensures all Books have unique IDs
- **Natural Design**: Container creates contents

**Anti-pattern (Don't do this):**
\`\`\`go
// BAD: External code creates books and manages IDs
book := &Book{
    ID:     generateID(), // ID generation logic scattered!
    Title:  "Some Book",
    Author: "Some Author",
}
library.books = append(library.books, book) // exposes internal structure!
\`\`\`

**Constraints:**
- Library must create Book instances
- Library must assign unique IDs (incrementing from 1)
- Books array should not be publicly accessible`,
	initialCode: `package principles

type Book struct {
	ID     int
	Title  string
	Author string
}

type Library struct {
	books  []*Book
	nextID int
}

func NewLibrary() *Library {
	}
}

func (l *Library) AddBook(title, author string) *Book {
}

func (l *Library) GetBooks() []*Book {
}`,
	solutionCode: `package principles

type Book struct {
	ID     int
	Title  string
	Author string
}

type Library struct {
	books  []*Book	// private: Library manages book creation and storage
	nextID int	// tracks next available ID for book creation
}

func NewLibrary() *Library {
	return &Library{
		books:  make([]*Book, 0),
		nextID: 1,	// start IDs at 1
	}
}

func (l *Library) AddBook(title, author string) *Book {
	// Library creates the Book because:
	// 1. Library contains/aggregates Books
	// 2. Library has the information needed (nextID)
	// 3. Library uses Books
	book := &Book{
		ID:     l.nextID,	// assign unique ID
		Title:  title,
		Author: author,
	}
	l.nextID++	// increment for next book
	l.books = append(l.books, book)	// add to collection
	return book	// return created instance
}

func (l *Library) GetBooks() []*Book {
	// Return all books
	// In production, you might return a copy to prevent external modification
	return l.books
}`,
	hint1: `Create a new Book with ID set to l.nextID, then increment l.nextID. Append the book to l.books and return it.`,
	hint2: `GetBooks should return l.books. The Creator pattern is about who creates objects, not who returns them.`,
	whyItMatters: `The Creator principle guides you on where to place object creation responsibility for low coupling.

**Why Creator Matters:**

**1. Natural Object Creation**
The object that contains or uses other objects should create them:

\`\`\`go
// GOOD: Order creates OrderItems (it contains them)
type Order struct {
    items []*OrderItem
    nextID int
}

func (o *Order) AddItem(productID string, quantity int, price float64) *OrderItem {
    item := &OrderItem{
        ID:        o.nextID,
        ProductID: productID,
        Quantity:  quantity,
        Price:     price,
    }
    o.nextID++
    o.items = append(o.items, item)
    return item
}

// BAD: External code creates items
item := &OrderItem{
    ID:        someID, // where does this come from?
    ProductID: productID,
    Quantity:  quantity,
    Price:     price,
}
order.Items = append(order.Items, item) // requires public Items field!
\`\`\`

**2. Centralized Creation Logic**
Creator pattern keeps initialization logic in one place:

\`\`\`go
// GOOD: Playlist creates Songs with consistent initialization
type Playlist struct {
    songs []*Song
}

func (p *Playlist) AddSong(title, artist string, duration int) *Song {
    song := &Song{
        Title:     title,
        Artist:    artist,
        Duration:  duration,
        AddedAt:   time.Now(), // Playlist sets timestamp
        PlayCount: 0,          // Playlist initializes play count
    }
    p.songs = append(p.songs, song)
    return song
}

// BAD: Each caller must remember all initialization steps
song := &Song{
    Title:     title,
    Artist:    artist,
    Duration:  duration,
    AddedAt:   time.Now(), // easy to forget!
    PlayCount: 0,          // easy to forget!
}
playlist.Songs = append(playlist.Songs, song)
\`\`\`

**3. ID Generation and Uniqueness**
Creator pattern is perfect for managing unique identifiers:

\`\`\`go
type Database struct {
    users  []*User
    nextID int
}

// Database creates Users and ensures unique IDs
func (db *Database) CreateUser(username, email string) *User {
    user := &User{
        ID:       db.nextID, // guaranteed unique
        Username: username,
        Email:    email,
        Created:  time.Now(),
    }
    db.nextID++
    db.users = append(db.users, user)
    return user
}
// No way for duplicate IDs - Database controls creation
\`\`\`

**4. Real-World Example: Shopping System**
\`\`\`go
// ShoppingCart creates CartItems
type ShoppingCart struct {
    items  []*CartItem
    nextID int
}

func (c *ShoppingCart) AddProduct(product *Product, quantity int) *CartItem {
    // Check if item already exists
    for _, item := range c.items {
        if item.ProductID == product.ID {
            item.Quantity += quantity // update existing
            return item
        }
    }

    // Create new item
    item := &CartItem{
        ID:         c.nextID,
        ProductID:  product.ID,
        Name:       product.Name,
        Price:      product.Price,
        Quantity:   quantity,
        AddedAt:    time.Now(),
    }
    c.nextID++
    c.items = append(c.items, item)
    return item
}
\`\`\`

**5. When NOT to use Creator**
Sometimes creation is too complex and needs a Factory:

\`\`\`go
// Complex creation logic - use Factory Pattern instead
type ReportFactory struct {
    templates map[string]ReportTemplate
    db        *Database
}

func (rf *ReportFactory) CreateReport(reportType string, data interface{}) (*Report, error) {
    template, ok := rf.templates[reportType]
    if !ok {
        return nil, errors.New("unknown report type")
    }

    // Complex logic: fetch data, apply template, validate
    report := &Report{
        Type:      reportType,
        Data:      rf.fetchReportData(data),
        Template:  template,
        Generated: time.Now(),
    }

    return report, nil
}
\`\`\`

**Creator vs Factory Pattern:**
- **Creator**: Simple object creation by container/aggregator
- **Factory**: Complex creation logic, multiple types, conditional creation

**Common Mistakes:**
- Creating objects outside their natural container
- Scattering ID generation logic
- Making collections public just to add items
- Not using Creator when aggregation relationship exists

**Rule of Thumb:**
If class A contains, aggregates, or closely uses class B instances, class A should create B instances.`,
	order: 1,
	testCode: `package principles

import (
	"testing"
)

// Test1: NewLibrary creates empty library
func Test1(t *testing.T) {
	library := NewLibrary()
	books := library.GetBooks()
	if len(books) != 0 {
		t.Error("New library should have 0 books")
	}
}

// Test2: AddBook returns book with ID
func Test2(t *testing.T) {
	library := NewLibrary()
	book := library.AddBook("Clean Code", "Robert Martin")
	if book.ID != 1 {
		t.Errorf("First book ID should be 1, got %d", book.ID)
	}
}

// Test3: AddBook sets title and author
func Test3(t *testing.T) {
	library := NewLibrary()
	book := library.AddBook("Design Patterns", "GoF")
	if book.Title != "Design Patterns" || book.Author != "GoF" {
		t.Error("Book title or author not set correctly")
	}
}

// Test4: Multiple books get incrementing IDs
func Test4(t *testing.T) {
	library := NewLibrary()
	book1 := library.AddBook("Book1", "Author1")
	book2 := library.AddBook("Book2", "Author2")
	book3 := library.AddBook("Book3", "Author3")
	if book1.ID != 1 || book2.ID != 2 || book3.ID != 3 {
		t.Error("Book IDs should increment: 1, 2, 3")
	}
}

// Test5: GetBooks returns all books
func Test5(t *testing.T) {
	library := NewLibrary()
	library.AddBook("A", "Author")
	library.AddBook("B", "Author")
	books := library.GetBooks()
	if len(books) != 2 {
		t.Errorf("expected 2 books, got %d", len(books))
	}
}

// Test6: Book struct fields
func Test6(t *testing.T) {
	book := Book{ID: 1, Title: "Test", Author: "Tester"}
	if book.ID != 1 || book.Title != "Test" || book.Author != "Tester" {
		t.Error("Book struct fields not set correctly")
	}
}

// Test7: Library struct initialization
func Test7(t *testing.T) {
	library := NewLibrary()
	if library == nil {
		t.Error("NewLibrary should not return nil")
	}
}

// Test8: Added book is in GetBooks
func Test8(t *testing.T) {
	library := NewLibrary()
	added := library.AddBook("Find Me", "Author")
	books := library.GetBooks()
	found := false
	for _, b := range books {
		if b.ID == added.ID {
			found = true
		}
	}
	if !found {
		t.Error("Added book should be in GetBooks result")
	}
}

// Test9: Empty strings for title and author
func Test9(t *testing.T) {
	library := NewLibrary()
	book := library.AddBook("", "")
	if book.ID != 1 {
		t.Error("Should create book even with empty strings")
	}
}

// Test10: Multiple libraries are independent
func Test10(t *testing.T) {
	lib1 := NewLibrary()
	lib2 := NewLibrary()
	lib1.AddBook("Lib1 Book", "Author")
	lib2.AddBook("Lib2 Book", "Author")
	lib2.AddBook("Lib2 Book2", "Author")
	if len(lib1.GetBooks()) != 1 || len(lib2.GetBooks()) != 2 {
		t.Error("Libraries should be independent")
	}
}
`,
	translations: {
		ru: {
			title: 'Создатель',
			description: `Реализуйте принцип Создателя — назначьте классу B ответственность за создание класса A, если B содержит, агрегирует или тесно использует A.

**Вы реализуете:**

1. **Library struct** — Содержит коллекцию книг
2. **AddBook(title, author string) *Book** — Создать и добавить книгу (Library создаёт книги!)
3. **GetBooks() []*Book** — Вернуть все книги
4. **Book struct** — Представляет книгу с ID, названием и автором

**Ключевые концепции:**
- **Создатель**: Класс, содержащий или использующий объекты, должен их создавать
- **Естественная ответственность**: Library управляет книгами, поэтому создаёт их
- **Генерация ID**: Library назначает уникальные ID создаваемым книгам

**Пример использования:**

\`\`\`go
library := NewLibrary()

// Library создаёт книги (она содержит/управляет ими)
book1 := library.AddBook("Clean Code", "Robert Martin")
book2 := library.AddBook("Design Patterns", "Gang of Four")

fmt.Println(book1.ID)     // 1
fmt.Println(book1.Title)  // Clean Code

books := library.GetBooks()
fmt.Println(len(books))   // 2
\`\`\`

**Зачем нужен Создатель?**
- **Низкая связанность**: Другому коду не нужно знать, как создавать книги
- **Согласованность**: Library гарантирует уникальность ID всех книг
- **Естественный дизайн**: Контейнер создаёт содержимое

**Анти-паттерн (Не делайте так):**
\`\`\`go
// ПЛОХО: Внешний код создаёт книги и управляет ID
book := &Book{
    ID:     generateID(), // логика генерации ID разбросана!
    Title:  "Some Book",
    Author: "Some Author",
}
library.books = append(library.books, book) // раскрывает внутреннюю структуру!
\`\`\`

**Ограничения:**
- Library должна создавать экземпляры Book
- Library должна назначать уникальные ID (с инкрементом от 1)
- Массив books не должен быть публично доступен`,
			hint1: `Создайте новую Book с ID равным l.nextID, затем увеличьте l.nextID. Добавьте книгу в l.books и верните её.`,
			hint2: `GetBooks должен вернуть l.books. Принцип Creator касается того, кто создаёт объекты, а не кто их возвращает.`,
			whyItMatters: `Принцип Создателя направляет вас, где разместить ответственность за создание объектов для низкой связанности.

**Почему Создатель важен:**

**1. Естественное создание объектов**
Объект, содержащий или использующий другие объекты, должен их создавать:

\`\`\`go
// ХОРОШО: Order создаёт OrderItems (он их содержит)
type Order struct {
    items []*OrderItem
    nextID int
}

func (o *Order) AddItem(productID string, quantity int, price float64) *OrderItem {
    item := &OrderItem{
        ID:        o.nextID,
        ProductID: productID,
        Quantity:  quantity,
        Price:     price,
    }
    o.nextID++
    o.items = append(o.items, item)
    return item
}
\`\`\`

**Распространённые ошибки:**
- Создание объектов вне их естественного контейнера
- Разбросанная логика генерации ID
- Публичные коллекции только для добавления элементов`,
			solutionCode: `package principles

type Book struct {
	ID     int
	Title  string
	Author string
}

type Library struct {
	books  []*Book	// приватное: Library управляет созданием и хранением книг
	nextID int	// отслеживает следующий доступный ID для создания книг
}

func NewLibrary() *Library {
	return &Library{
		books:  make([]*Book, 0),
		nextID: 1,	// начинаем ID с 1
	}
}

func (l *Library) AddBook(title, author string) *Book {
	// Library создаёт Book потому что:
	// 1. Library содержит/агрегирует книги
	// 2. Library имеет необходимую информацию (nextID)
	// 3. Library использует книги
	book := &Book{
		ID:     l.nextID,	// назначаем уникальный ID
		Title:  title,
		Author: author,
	}
	l.nextID++	// увеличиваем для следующей книги
	l.books = append(l.books, book)	// добавляем в коллекцию
	return book	// возвращаем созданный экземпляр
}

func (l *Library) GetBooks() []*Book {
	// Возвращаем все книги
	return l.books
}`
		},
		uz: {
			title: 'Creator (Yaratuvchi)',
			description: `Creator prinsipini amalga oshiring — agar B klass A ni o'z ichiga olsa, agregatsiya qilsa yoki yaqindan ishlatsas, A ni yaratish mas'uliyatini B klassga belgilang.

**Siz amalga oshirasiz:**

1. **Library struct** — Kitoblar to'plamini o'z ichiga oladi
2. **AddBook(title, author string) *Book** — Kitob yaratish va qo'shish (Library kitoblarni yaratadi!)
3. **GetBooks() []*Book** — Barcha kitoblarni qaytarish
4. **Book struct** — ID, sarlavha va muallif bilan kitobni ifodalaydi

**Asosiy tushunchalar:**
- **Creator**: Ob'ektlarni o'z ichiga olgan yoki ishlatadigan klass ularni yaratishi kerak
- **Tabiiy mas'uliyat**: Library kitoblarni boshqaradi, shuning uchun ularni yaratadi
- **ID generatsiyasi**: Library yaratgan kitoblarga noyob ID lar beradi

**Foydalanish misoli:**

\`\`\`go
library := NewLibrary()

// Library kitoblarni yaratadi (u ularni o'z ichiga oladi/boshqaradi)
book1 := library.AddBook("Clean Code", "Robert Martin")
book2 := library.AddBook("Design Patterns", "Gang of Four")

fmt.Println(book1.ID)     // 1
fmt.Println(book1.Title)  // Clean Code

books := library.GetBooks()
fmt.Println(len(books))   // 2
\`\`\`

**Nima uchun Creator?**
- **Past bog'lanish**: Boshqa kod kitoblarni qanday yaratishni bilishi shart emas
- **Izchillik**: Library barcha kitoblar noyob ID ga ega bo'lishini ta'minlaydi
- **Tabiiy dizayn**: Konteyner tarkibni yaratadi

**Anti-pattern (Buni qilmang):**
\`\`\`go
// YOMON: Tashqi kod kitoblarni yaratadi va ID larni boshqaradi
book := &Book{
    ID:     generateID(), // ID generatsiya mantiqi tarqalgan!
    Title:  "Some Book",
    Author: "Some Author",
}
library.books = append(library.books, book) // ichki tuzilishni ochib beradi!
\`\`\`

**Cheklovlar:**
- Library Book nusxalarini yaratishi kerak
- Library noyob ID lar berishi kerak (1 dan boshlab oshirish)
- Books massivi ommaviy kirish uchun ochiq bo'lmasligi kerak`,
			hint1: `Yangi Book yarating, ID ni l.nextID ga o'rnating, keyin l.nextID ni oshiring. Kitobni l.books ga qo'shing va qaytaring.`,
			hint2: `GetBooks l.books ni qaytarishi kerak. Creator patterni ob'ektlarni kim yaratishi haqida, ularni kim qaytarishi haqida emas.`,
			whyItMatters: `Creator printsipi past bog'lanish uchun ob'ekt yaratish mas'uliyatini qaerga joylashtirish kerakligini ko'rsatadi.

**Creator nima uchun muhim:**

**1. Tabiiy ob'ekt yaratish**
Boshqa ob'ektlarni o'z ichiga olgan yoki ishlatadigan ob'ekt ularni yaratishi kerak:

\`\`\`go
// YAXSHI: Order OrderItems yaratadi (u ularni o'z ichiga oladi)
type Order struct {
    items []*OrderItem
    nextID int
}

func (o *Order) AddItem(productID string, quantity int, price float64) *OrderItem {
    item := &OrderItem{
        ID:        o.nextID,
        ProductID: productID,
        Quantity:  quantity,
        Price:     price,
    }
    o.nextID++
    o.items = append(o.items, item)
    return item
}
\`\`\`

**Umumiy xatolar:**
- Ob'ektlarni ularning tabiiy konteyneridan tashqarida yaratish
- Tarqalgan ID generatsiya mantiqi
- Elementlarni qo'shish uchun faqat ommaviy to'plamlar`,
			solutionCode: `package principles

type Book struct {
	ID     int
	Title  string
	Author string
}

type Library struct {
	books  []*Book	// privat: Library kitob yaratish va saqlashni boshqaradi
	nextID int	// kitob yaratish uchun keyingi mavjud ID ni kuzatadi
}

func NewLibrary() *Library {
	return &Library{
		books:  make([]*Book, 0),
		nextID: 1,	// ID larni 1 dan boshlang
	}
}

func (l *Library) AddBook(title, author string) *Book {
	// Library Book yaratadi chunki:
	// 1. Library kitoblarni o'z ichiga oladi/agregatsiya qiladi
	// 2. Library kerakli ma'lumotga ega (nextID)
	// 3. Library kitoblarni ishlatadi
	book := &Book{
		ID:     l.nextID,	// noyob ID ni tayinlaymiz
		Title:  title,
		Author: author,
	}
	l.nextID++	// keyingi kitob uchun oshiramiz
	l.books = append(l.books, book)	// to'plamga qo'shamiz
	return book	// yaratilgan nusxani qaytaramiz
}

func (l *Library) GetBooks() []*Book {
	// Barcha kitoblarni qaytaramiz
	return l.books
}`
		}
	}
};

export default task;
