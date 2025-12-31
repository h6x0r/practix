import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-iterator',
	title: 'Iterator Pattern',
	difficulty: 'easy',
	tags: ['go', 'design-patterns', 'behavioral', 'iterator'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Iterator pattern in Go - provide a way to access elements sequentially without exposing underlying representation.

**You will implement:**

1. **Iterator interface** - HasNext, Next, Current
2. **BookCollection** - Aggregate with CreateIterator
3. **BookIterator** - Concrete iterator

**Example Usage:**

\`\`\`go
collection := NewBookCollection()	// create empty collection
collection.AddBook("Go Programming")	// add first book
collection.AddBook("Design Patterns")	// add second book
collection.AddBook("Clean Code")	// add third book

iterator := collection.CreateIterator()	// get iterator for traversal
for iterator.HasNext() {	// check if more elements exist
    book := iterator.Next()	// get current book and advance
    fmt.Println(book.Title)	// print book title
}
// Output: Go Programming, Design Patterns, Clean Code

// Multiple iterators are independent
iter1 := collection.CreateIterator()	// first iterator
iter2 := collection.CreateIterator()	// second iterator - independent position
iter1.Next()	// advances iter1 only
iter2.Current()	// still at first book
\`\`\``,
	initialCode: `package patterns

type Book struct {
	Title string
}

type Iterator interface {
}

type BookCollection struct {
	books []*Book
}

func NewBookCollection() *BookCollection {
}

func (c *BookCollection) AddBook(title string) {
}

func (c *BookCollection) CreateIterator() Iterator {
}

type BookIterator struct {
	collection *BookCollection
	index      int
}

func (i *BookIterator) HasNext() bool {
}

func (i *BookIterator) Next() *Book {
}

func (i *BookIterator) Current() *Book {
}`,
	solutionCode: `package patterns

type Book struct {	// data element being iterated
	Title string	// book title
}

type Iterator interface {	// iterator contract
	HasNext() bool	// check if more elements exist
	Next() *Book	// get current element and advance
	Current() *Book	// get current element without advancing
}

type BookCollection struct {	// aggregate/container
	books []*Book	// internal storage - hidden from clients
}

func NewBookCollection() *BookCollection {	// factory constructor
	return &BookCollection{books: make([]*Book, 0)}	// initialize with empty slice
}

func (c *BookCollection) AddBook(title string) {	// add book to collection
	c.books = append(c.books, &Book{Title: title})	// create and append new book
}

func (c *BookCollection) CreateIterator() Iterator {	// factory method for iterator
	return &BookIterator{collection: c, index: 0}	// return new iterator at start position
}

type BookIterator struct {	// concrete iterator implementation
	collection *BookCollection	// reference to collection being iterated
	index      int	// current position in iteration
}

func (i *BookIterator) HasNext() bool {	// check for more elements
	return i.index < len(i.collection.books)	// true if index within bounds
}

func (i *BookIterator) Next() *Book {	// get and advance
	if i.HasNext() {	// ensure element exists
		book := i.collection.books[i.index]	// get current book
		i.index++	// advance to next position
		return book	// return retrieved book
	}
	return nil	// no more elements
}

func (i *BookIterator) Current() *Book {	// peek without advancing
	if i.index < len(i.collection.books) {	// check bounds
		return i.collection.books[i.index]	// return current without incrementing
	}
	return nil	// out of bounds
}`,
	hint1: `**Iterator Methods:**

- **HasNext()**: Compare current index with collection length
- **Next()**: Return current element and increment index
- **Current()**: Return current element without incrementing

\`\`\`go
// HasNext - check if more elements
func (i *BookIterator) HasNext() bool {
	return i.index < len(i.collection.books)	// true if index is valid
}

// Next - get current and advance
func (i *BookIterator) Next() *Book {
	if i.HasNext() {	// safety check
		book := i.collection.books[i.index]	// get current
		i.index++	// advance position
		return book
	}
	return nil	// nothing left
}
\`\`\``,
	hint2: `**Collection Methods:**

\`\`\`go
// AddBook - append new book to internal slice
func (c *BookCollection) AddBook(title string) {
	c.books = append(c.books, &Book{Title: title})
}

// CreateIterator - factory method returns fresh iterator
func (c *BookCollection) CreateIterator() Iterator {
	return &BookIterator{
		collection: c,	// reference to this collection
		index:      0,	// start at beginning
	}
}

// Current - peek at current without advancing
func (i *BookIterator) Current() *Book {
	if i.index < len(i.collection.books) {
		return i.collection.books[i.index]	// no index++ here!
	}
	return nil
}
\`\`\``,
	whyItMatters: `## Why Iterator Pattern Exists

**Problem:** Direct access to collection internals couples client code to implementation.

\`\`\`go
// Without Iterator - client knows about slice
collection := &BookCollection{books: []*Book{...}}
for i := 0; i < len(collection.books); i++ {	// knows it's a slice
    book := collection.books[i]	// direct access to internal field
    fmt.Println(book.Title)
}
// What if collection changes to map? Tree? Database cursor?
// All client code breaks!
\`\`\`

**Solution:** Iterator provides uniform traversal interface:

\`\`\`go
// With Iterator - client only knows interface
iterator := collection.CreateIterator()	// get abstract iterator
for iterator.HasNext() {	// standard traversal pattern
    book := iterator.Next()	// works regardless of internal structure
    fmt.Println(book.Title)
}
// Collection can be slice, map, tree, database - code doesn't change
\`\`\`

---

## Real-World Iterators in Go

**1. Database Cursors:**
- sql.Rows.Next() for iterating query results
- Each Next() fetches next row from database

**2. File System Traversal:**
- filepath.WalkDir iterates directories recursively
- bufio.Scanner iterates lines/tokens in a file

**3. Channel Iteration:**
- range over channels is Go's iterator pattern
- Producer/consumer without exposing internal queue

**4. Pagination APIs:**
- ListUsers(pageToken) -> (users, nextPageToken)
- Iterator pattern for API results

---

## Production Pattern: Paginated Database Iterator

\`\`\`go
package iterator

import (
	"context"
	"database/sql"
)

// User represents a database user
type User struct {
	ID    int	// user identifier
	Name  string	// user name
	Email string	// user email
}

// UserIterator provides paginated iteration over users
type UserIterator struct {
	db         *sql.DB	// database connection
	pageSize   int	// users per page
	offset     int	// current offset
	buffer     []*User	// prefetched users
	bufferIdx  int	// position in buffer
	done       bool	// iteration complete
	totalCount int	// total users available
}

// NewUserIterator creates iterator with specified page size
func NewUserIterator(db *sql.DB, pageSize int) *UserIterator {	// factory
	return &UserIterator{
		db:       db,
		pageSize: pageSize,
		offset:   0,
		buffer:   nil,	// lazy load on first HasNext
		done:     false,
	}
}

// fetchPage loads next page of users from database
func (it *UserIterator) fetchPage(ctx context.Context) error {	// internal helper
	query := "SELECT id, name, email FROM users ORDER BY id LIMIT ? OFFSET ?"
	rows, err := it.db.QueryContext(ctx, query, it.pageSize, it.offset)
	if err != nil {
		return err	// propagate database error
	}
	defer rows.Close()

	it.buffer = make([]*User, 0, it.pageSize)	// reset buffer
	for rows.Next() {	// iterate database rows
		user := &User{}
		if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
			return err	// propagate scan error
		}
		it.buffer = append(it.buffer, user)	// add to buffer
	}

	it.bufferIdx = 0	// reset buffer position
	it.offset += len(it.buffer)	// advance offset for next page

	if len(it.buffer) < it.pageSize {	// partial page = last page
		it.done = true	// mark iteration complete
	}

	return rows.Err()	// check for iteration errors
}

// HasNext checks if more users are available
func (it *UserIterator) HasNext(ctx context.Context) bool {	// check for more
	if it.buffer != nil && it.bufferIdx < len(it.buffer) {
		return true	// have buffered users
	}
	if it.done {
		return false	// already fetched all pages
	}
	// Try to fetch next page
	if err := it.fetchPage(ctx); err != nil {
		it.done = true	// error = stop iteration
		return false
	}
	return len(it.buffer) > 0	// have users after fetch?
}

// Next returns current user and advances
func (it *UserIterator) Next() *User {	// get and advance
	if it.bufferIdx < len(it.buffer) {
		user := it.buffer[it.bufferIdx]	// get current
		it.bufferIdx++	// advance position
		return user
	}
	return nil	// no more users
}

// Reset restarts iteration from beginning
func (it *UserIterator) Reset() {	// restart iterator
	it.offset = 0
	it.buffer = nil
	it.bufferIdx = 0
	it.done = false
}

// Usage:
// db, _ := sql.Open("postgres", connectionString)
// iterator := NewUserIterator(db, 100) // 100 users per page
// ctx := context.Background()
// for iterator.HasNext(ctx) {
//     user := iterator.Next()
//     fmt.Printf("User: %s (%s)\n", user.Name, user.Email)
// }
\`\`\`

---

## Common Mistakes to Avoid

**1. Modifying collection during iteration:**
\`\`\`go
// Wrong - undefined behavior
iterator := collection.CreateIterator()
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        collection.RemoveBook(book)	// modifies during iteration!
    }
}

// Right - collect items to remove, then remove
var toRemove []*Book
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        toRemove = append(toRemove, book)
    }
}
for _, book := range toRemove {	// remove after iteration
    collection.RemoveBook(book)
}
\`\`\`

**2. Not checking HasNext before Next:**
\`\`\`go
// Wrong - may return nil
book := iterator.Next()	// no HasNext check
fmt.Println(book.Title)	// panic if nil!

// Right - always check first
if iterator.HasNext() {
    book := iterator.Next()
    fmt.Println(book.Title)
}
// Or use the for loop pattern
for iterator.HasNext() {
    book := iterator.Next()	// safe - HasNext already checked
    fmt.Println(book.Title)
}
\`\`\`

**3. Sharing iterator state:**
\`\`\`go
// Wrong - single iterator shared
iterator := collection.CreateIterator()
go process1(iterator)	// concurrent access
go process2(iterator)	// race condition!

// Right - each goroutine gets own iterator
go func() {
    iter := collection.CreateIterator()	// independent iterator
    process1(iter)
}()
go func() {
    iter := collection.CreateIterator()	// another independent iterator
    process2(iter)
}()
\`\`\``,
	order: 5,
	testCode: `package patterns

import (
	"testing"
)

// Test1: NewBookCollection creates empty collection
func Test1(t *testing.T) {
	c := NewBookCollection()
	if c == nil {
		t.Error("NewBookCollection should return non-nil")
	}
}

// Test2: AddBook adds books to collection
func Test2(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Test Book")
	iter := c.CreateIterator()
	if !iter.HasNext() {
		t.Error("Collection should have one book")
	}
}

// Test3: CreateIterator returns iterator
func Test3(t *testing.T) {
	c := NewBookCollection()
	iter := c.CreateIterator()
	if iter == nil {
		t.Error("CreateIterator should return non-nil")
	}
}

// Test4: HasNext returns false for empty collection
func Test4(t *testing.T) {
	c := NewBookCollection()
	iter := c.CreateIterator()
	if iter.HasNext() {
		t.Error("Empty collection should have no next")
	}
}

// Test5: Next returns books in order
func Test5(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Book 1")
	c.AddBook("Book 2")
	iter := c.CreateIterator()
	first := iter.Next()
	second := iter.Next()
	if first.Title != "Book 1" || second.Title != "Book 2" {
		t.Error("Next should return books in order")
	}
}

// Test6: Next returns nil when exhausted
func Test6(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Only Book")
	iter := c.CreateIterator()
	iter.Next()
	book := iter.Next()
	if book != nil {
		t.Error("Next should return nil when exhausted")
	}
}

// Test7: Current returns current without advancing
func Test7(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Test")
	iter := c.CreateIterator()
	first := iter.Current()
	second := iter.Current()
	if first != second {
		t.Error("Current should not advance iterator")
	}
}

// Test8: Multiple iterators are independent
func Test8(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Book 1")
	c.AddBook("Book 2")
	iter1 := c.CreateIterator()
	iter2 := c.CreateIterator()
	iter1.Next()
	if iter1.Current() == iter2.Current() {
		t.Error("Iterators should be independent")
	}
}

// Test9: HasNext after iteration returns false
func Test9(t *testing.T) {
	c := NewBookCollection()
	c.AddBook("Single")
	iter := c.CreateIterator()
	iter.Next()
	if iter.HasNext() {
		t.Error("HasNext should be false after iterating all")
	}
}

// Test10: Current on empty iterator returns nil
func Test10(t *testing.T) {
	c := NewBookCollection()
	iter := c.CreateIterator()
	if iter.Current() != nil {
		t.Error("Current should return nil for empty collection")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Iterator (Итератор)',
			description: `Реализуйте паттерн Iterator на Go — обеспечьте последовательный доступ к элементам без раскрытия внутренней структуры.

**Вы реализуете:**

1. **Интерфейс Iterator** - HasNext, Next, Current
2. **BookCollection** - Агрегат с методом CreateIterator
3. **BookIterator** - Конкретный итератор

**Пример использования:**

\`\`\`go
collection := NewBookCollection()	// создать пустую коллекцию
collection.AddBook("Go Programming")	// добавить первую книгу
collection.AddBook("Design Patterns")	// добавить вторую книгу
collection.AddBook("Clean Code")	// добавить третью книгу

iterator := collection.CreateIterator()	// получить итератор для обхода
for iterator.HasNext() {	// проверить есть ли ещё элементы
    book := iterator.Next()	// получить текущую книгу и продвинуться
    fmt.Println(book.Title)	// вывести название книги
}
// Вывод: Go Programming, Design Patterns, Clean Code

// Несколько итераторов независимы
iter1 := collection.CreateIterator()	// первый итератор
iter2 := collection.CreateIterator()	// второй итератор - независимая позиция
iter1.Next()	// продвигает только iter1
iter2.Current()	// всё ещё на первой книге
\`\`\``,
			hint1: `**Методы итератора:**

- **HasNext()**: Сравнить текущий индекс с длиной коллекции
- **Next()**: Вернуть текущий элемент и увеличить индекс
- **Current()**: Вернуть текущий элемент без увеличения индекса

\`\`\`go
// HasNext - проверить есть ли ещё элементы
func (i *BookIterator) HasNext() bool {
	return i.index < len(i.collection.books)	// true если индекс валидный
}

// Next - получить текущий и продвинуться
func (i *BookIterator) Next() *Book {
	if i.HasNext() {	// проверка безопасности
		book := i.collection.books[i.index]	// получить текущий
		i.index++	// продвинуть позицию
		return book
	}
	return nil	// ничего не осталось
}
\`\`\``,
			hint2: `**Методы коллекции:**

\`\`\`go
// AddBook - добавить новую книгу во внутренний slice
func (c *BookCollection) AddBook(title string) {
	c.books = append(c.books, &Book{Title: title})
}

// CreateIterator - фабричный метод возвращает новый итератор
func (c *BookCollection) CreateIterator() Iterator {
	return &BookIterator{
		collection: c,	// ссылка на эту коллекцию
		index:      0,	// начать с начала
	}
}

// Current - посмотреть текущий без продвижения
func (i *BookIterator) Current() *Book {
	if i.index < len(i.collection.books) {
		return i.collection.books[i.index]	// нет index++ здесь!
	}
	return nil
}
\`\`\``,
			whyItMatters: `## Зачем нужен паттерн Iterator

**Проблема:** Прямой доступ к внутренностям коллекции связывает клиентский код с реализацией.

\`\`\`go
// Без Iterator - клиент знает про slice
collection := &BookCollection{books: []*Book{...}}
for i := 0; i < len(collection.books); i++ {	// знает что это slice
    book := collection.books[i]	// прямой доступ к внутреннему полю
    fmt.Println(book.Title)
}
// Что если коллекция станет map? Деревом? Курсором БД?
// Весь клиентский код ломается!
\`\`\`

**Решение:** Iterator предоставляет единый интерфейс обхода:

\`\`\`go
// С Iterator - клиент знает только интерфейс
iterator := collection.CreateIterator()	// получить абстрактный итератор
for iterator.HasNext() {	// стандартный паттерн обхода
    book := iterator.Next()	// работает независимо от внутренней структуры
    fmt.Println(book.Title)
}
// Коллекция может быть slice, map, деревом, БД - код не меняется
\`\`\`

---

## Реальные итераторы в Go

**1. Курсоры баз данных:**
- sql.Rows.Next() для итерации результатов запроса
- Каждый Next() получает следующую строку из БД

**2. Обход файловой системы:**
- filepath.WalkDir итерирует директории рекурсивно
- bufio.Scanner итерирует строки/токены в файле

**3. Итерация по каналам:**
- range по каналам — это паттерн итератора в Go
- Производитель/потребитель без раскрытия внутренней очереди

**4. API с пагинацией:**
- ListUsers(pageToken) -> (users, nextPageToken)
- Паттерн итератора для результатов API

---

## Production-паттерн: Итератор с пагинацией БД

\`\`\`go
package iterator

import (
	"context"
	"database/sql"
)

// User представляет пользователя из БД
type User struct {
	ID    int	// идентификатор пользователя
	Name  string	// имя пользователя
	Email string	// email пользователя
}

// UserIterator обеспечивает постраничную итерацию по пользователям
type UserIterator struct {
	db         *sql.DB	// соединение с БД
	pageSize   int	// пользователей на страницу
	offset     int	// текущее смещение
	buffer     []*User	// предзагруженные пользователи
	bufferIdx  int	// позиция в буфере
	done       bool	// итерация завершена
	totalCount int	// всего доступно пользователей
}

// NewUserIterator создаёт итератор с указанным размером страницы
func NewUserIterator(db *sql.DB, pageSize int) *UserIterator {	// фабрика
	return &UserIterator{
		db:       db,
		pageSize: pageSize,
		offset:   0,
		buffer:   nil,	// ленивая загрузка при первом HasNext
		done:     false,
	}
}

// fetchPage загружает следующую страницу пользователей из БД
func (it *UserIterator) fetchPage(ctx context.Context) error {	// внутренний помощник
	query := "SELECT id, name, email FROM users ORDER BY id LIMIT ? OFFSET ?"
	rows, err := it.db.QueryContext(ctx, query, it.pageSize, it.offset)
	if err != nil {
		return err	// передать ошибку БД
	}
	defer rows.Close()

	it.buffer = make([]*User, 0, it.pageSize)	// сбросить буфер
	for rows.Next() {	// итерировать строки БД
		user := &User{}
		if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
			return err	// передать ошибку сканирования
		}
		it.buffer = append(it.buffer, user)	// добавить в буфер
	}

	it.bufferIdx = 0	// сбросить позицию буфера
	it.offset += len(it.buffer)	// сдвинуть offset для следующей страницы

	if len(it.buffer) < it.pageSize {	// неполная страница = последняя
		it.done = true	// отметить итерацию завершённой
	}

	return rows.Err()	// проверить ошибки итерации
}

// HasNext проверяет доступны ли ещё пользователи
func (it *UserIterator) HasNext(ctx context.Context) bool {	// проверить наличие
	if it.buffer != nil && it.bufferIdx < len(it.buffer) {
		return true	// есть буферизованные пользователи
	}
	if it.done {
		return false	// уже загрузили все страницы
	}
	// Попробовать загрузить следующую страницу
	if err := it.fetchPage(ctx); err != nil {
		it.done = true	// ошибка = остановить итерацию
		return false
	}
	return len(it.buffer) > 0	// есть пользователи после загрузки?
}

// Next возвращает текущего пользователя и продвигается
func (it *UserIterator) Next() *User {	// получить и продвинуться
	if it.bufferIdx < len(it.buffer) {
		user := it.buffer[it.bufferIdx]	// получить текущего
		it.bufferIdx++	// продвинуть позицию
		return user
	}
	return nil	// больше нет пользователей
}

// Reset перезапускает итерацию с начала
func (it *UserIterator) Reset() {	// перезапустить итератор
	it.offset = 0
	it.buffer = nil
	it.bufferIdx = 0
	it.done = false
}

// Использование:
// db, _ := sql.Open("postgres", connectionString)
// iterator := NewUserIterator(db, 100) // 100 пользователей на страницу
// ctx := context.Background()
// for iterator.HasNext(ctx) {
//     user := iterator.Next()
//     fmt.Printf("User: %s (%s)\n", user.Name, user.Email)
// }
\`\`\`

---

## Распространённые ошибки

**1. Изменение коллекции во время итерации:**
\`\`\`go
// Неправильно - неопределённое поведение
iterator := collection.CreateIterator()
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        collection.RemoveBook(book)	// изменяет во время итерации!
    }
}

// Правильно - собрать элементы для удаления, потом удалить
var toRemove []*Book
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        toRemove = append(toRemove, book)
    }
}
for _, book := range toRemove {	// удалить после итерации
    collection.RemoveBook(book)
}
\`\`\`

**2. Не проверять HasNext перед Next:**
\`\`\`go
// Неправильно - может вернуть nil
book := iterator.Next()	// нет проверки HasNext
fmt.Println(book.Title)	// паника если nil!

// Правильно - всегда сначала проверять
if iterator.HasNext() {
    book := iterator.Next()
    fmt.Println(book.Title)
}
// Или использовать паттерн for-цикла
for iterator.HasNext() {
    book := iterator.Next()	// безопасно - HasNext уже проверен
    fmt.Println(book.Title)
}
\`\`\`

**3. Разделение состояния итератора:**
\`\`\`go
// Неправильно - один итератор на всех
iterator := collection.CreateIterator()
go process1(iterator)	// конкурентный доступ
go process2(iterator)	// состояние гонки!

// Правильно - каждая горутина получает свой итератор
go func() {
    iter := collection.CreateIterator()	// независимый итератор
    process1(iter)
}()
go func() {
    iter := collection.CreateIterator()	// другой независимый итератор
    process2(iter)
}()
\`\`\``
		},
		uz: {
			title: 'Iterator (Iterator) Pattern',
			description: `Go tilida Iterator patternini amalga oshiring — ichki tuzilmani ochmasdan elementlarga ketma-ket kirish imkonini bering.

**Siz amalga oshirasiz:**

1. **Iterator interfeysi** - HasNext, Next, Current
2. **BookCollection** - CreateIterator bilan agregat
3. **BookIterator** - Aniq iterator

**Foydalanish namunasi:**

\`\`\`go
collection := NewBookCollection()	// bo'sh to'plam yaratish
collection.AddBook("Go Programming")	// birinchi kitobni qo'shish
collection.AddBook("Design Patterns")	// ikkinchi kitobni qo'shish
collection.AddBook("Clean Code")	// uchinchi kitobni qo'shish

iterator := collection.CreateIterator()	// o'tish uchun iterator olish
for iterator.HasNext() {	// yana elementlar borligini tekshirish
    book := iterator.Next()	// joriy kitobni olish va oldinga siljish
    fmt.Println(book.Title)	// kitob nomini chiqarish
}
// Chiqish: Go Programming, Design Patterns, Clean Code

// Bir nechta iterator mustaqil
iter1 := collection.CreateIterator()	// birinchi iterator
iter2 := collection.CreateIterator()	// ikkinchi iterator - mustaqil pozitsiya
iter1.Next()	// faqat iter1 ni siljitadi
iter2.Current()	// hali birinchi kitobda
\`\`\``,
			hint1: `**Iterator metodlari:**

- **HasNext()**: Joriy indexni to'plam uzunligi bilan solishtirish
- **Next()**: Joriy elementni qaytarish va indexni oshirish
- **Current()**: Joriy elementni index oshirmasdan qaytarish

\`\`\`go
// HasNext - yana elementlar borligini tekshirish
func (i *BookIterator) HasNext() bool {
	return i.index < len(i.collection.books)	// index to'g'ri bo'lsa true
}

// Next - joriyni olish va oldinga siljish
func (i *BookIterator) Next() *Book {
	if i.HasNext() {	// xavfsizlik tekshiruvi
		book := i.collection.books[i.index]	// joriyni olish
		i.index++	// pozitsiyani siljitish
		return book
	}
	return nil	// hech narsa qolmadi
}
\`\`\``,
			hint2: `**To'plam metodlari:**

\`\`\`go
// AddBook - yangi kitobni ichki slice ga qo'shish
func (c *BookCollection) AddBook(title string) {
	c.books = append(c.books, &Book{Title: title})
}

// CreateIterator - fabrika metodi yangi iterator qaytaradi
func (c *BookCollection) CreateIterator() Iterator {
	return &BookIterator{
		collection: c,	// bu to'plamga havola
		index:      0,	// boshidan boshlash
	}
}

// Current - oldinga siljmasdan joriyga qarash
func (i *BookIterator) Current() *Book {
	if i.index < len(i.collection.books) {
		return i.collection.books[i.index]	// bu yerda index++ yo'q!
	}
	return nil
}
\`\`\``,
			whyItMatters: `## Iterator Pattern nima uchun kerak

**Muammo:** To'plam ichki qismlariga to'g'ridan-to'g'ri kirish mijoz kodini amalga oshirishga bog'laydi.

\`\`\`go
// Iterator siz - mijoz slice haqida biladi
collection := &BookCollection{books: []*Book{...}}
for i := 0; i < len(collection.books); i++ {	// bu slice ekanini biladi
    book := collection.books[i]	// ichki maydoniga to'g'ridan-to'g'ri kirish
    fmt.Println(book.Title)
}
// Agar to'plam map ga o'zgarsa? Daraxtga? DB kursoriga?
// Barcha mijoz kodi buziladi!
\`\`\`

**Yechim:** Iterator yagona o'tish interfeysini taqdim etadi:

\`\`\`go
// Iterator bilan - mijoz faqat interfeysni biladi
iterator := collection.CreateIterator()	// abstrakt iterator olish
for iterator.HasNext() {	// standart o'tish patterni
    book := iterator.Next()	// ichki tuzilishdan qat'i nazar ishlaydi
    fmt.Println(book.Title)
}
// To'plam slice, map, daraxt, DB bo'lishi mumkin - kod o'zgarmaydi
\`\`\`

---

## Go da haqiqiy iteratorlar

**1. Ma'lumotlar bazasi kursorlari:**
- sql.Rows.Next() so'rov natijalarini iteratsiya qilish uchun
- Har bir Next() ma'lumotlar bazasidan keyingi qatorni oladi

**2. Fayl tizimi o'tishi:**
- filepath.WalkDir kataloglarni rekursiv iteratsiya qiladi
- bufio.Scanner fayldagi qatorlar/tokenlarni iteratsiya qiladi

**3. Kanal iteratsiyasi:**
- kanallar bo'yicha range - bu Go da iterator patterni
- Ichki navbatni ochmasdan ishlab chiqaruvchi/iste'molchi

**4. Sahifalashli API lar:**
- ListUsers(pageToken) -> (users, nextPageToken)
- API natijalari uchun iterator patterni

---

## Production pattern: Sahifalashli DB Iterator

\`\`\`go
package iterator

import (
	"context"
	"database/sql"
)

// User ma'lumotlar bazasidagi foydalanuvchini ifodalaydi
type User struct {
	ID    int	// foydalanuvchi identifikatori
	Name  string	// foydalanuvchi nomi
	Email string	// foydalanuvchi emaili
}

// UserIterator foydalanuvchilar bo'yicha sahifalashli iteratsiyani ta'minlaydi
type UserIterator struct {
	db         *sql.DB	// ma'lumotlar bazasi ulanishi
	pageSize   int	// sahifadagi foydalanuvchilar soni
	offset     int	// joriy siljish
	buffer     []*User	// oldindan yuklangan foydalanuvchilar
	bufferIdx  int	// buferdagi pozitsiya
	done       bool	// iteratsiya tugadi
	totalCount int	// mavjud foydalanuvchilar soni
}

// NewUserIterator belgilangan sahifa o'lchami bilan iterator yaratadi
func NewUserIterator(db *sql.DB, pageSize int) *UserIterator {	// fabrika
	return &UserIterator{
		db:       db,
		pageSize: pageSize,
		offset:   0,
		buffer:   nil,	// birinchi HasNext da dangasa yuklash
		done:     false,
	}
}

// fetchPage ma'lumotlar bazasidan keyingi sahifani yuklaydi
func (it *UserIterator) fetchPage(ctx context.Context) error {	// ichki yordamchi
	query := "SELECT id, name, email FROM users ORDER BY id LIMIT ? OFFSET ?"
	rows, err := it.db.QueryContext(ctx, query, it.pageSize, it.offset)
	if err != nil {
		return err	// ma'lumotlar bazasi xatosini uzatish
	}
	defer rows.Close()

	it.buffer = make([]*User, 0, it.pageSize)	// buferni tozalash
	for rows.Next() {	// ma'lumotlar bazasi qatorlarini iteratsiya qilish
		user := &User{}
		if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
			return err	// skanerlash xatosini uzatish
		}
		it.buffer = append(it.buffer, user)	// buferga qo'shish
	}

	it.bufferIdx = 0	// bufer pozitsiyasini qayta o'rnatish
	it.offset += len(it.buffer)	// keyingi sahifa uchun offsetni siljitish

	if len(it.buffer) < it.pageSize {	// to'liq bo'lmagan sahifa = oxirgi
		it.done = true	// iteratsiyani tugagan deb belgilash
	}

	return rows.Err()	// iteratsiya xatolarini tekshirish
}

// HasNext yana foydalanuvchilar borligini tekshiradi
func (it *UserIterator) HasNext(ctx context.Context) bool {	// mavjudligini tekshirish
	if it.buffer != nil && it.bufferIdx < len(it.buffer) {
		return true	// buferlangan foydalanuvchilar bor
	}
	if it.done {
		return false	// barcha sahifalar allaqachon yuklangan
	}
	// Keyingi sahifani yuklashga urinish
	if err := it.fetchPage(ctx); err != nil {
		it.done = true	// xato = iteratsiyani to'xtatish
		return false
	}
	return len(it.buffer) > 0	// yuklashdan keyin foydalanuvchilar bormi?
}

// Next joriy foydalanuvchini qaytaradi va oldinga siljiydi
func (it *UserIterator) Next() *User {	// olish va siljish
	if it.bufferIdx < len(it.buffer) {
		user := it.buffer[it.bufferIdx]	// joriyni olish
		it.bufferIdx++	// pozitsiyani siljitish
		return user
	}
	return nil	// boshqa foydalanuvchilar yo'q
}

// Reset iteratsiyani boshidan qayta boshlaydi
func (it *UserIterator) Reset() {	// iteratorni qayta boshlash
	it.offset = 0
	it.buffer = nil
	it.bufferIdx = 0
	it.done = false
}

// Foydalanish:
// db, _ := sql.Open("postgres", connectionString)
// iterator := NewUserIterator(db, 100) // sahifada 100 foydalanuvchi
// ctx := context.Background()
// for iterator.HasNext(ctx) {
//     user := iterator.Next()
//     fmt.Printf("User: %s (%s)\n", user.Name, user.Email)
// }
\`\`\`

---

## Keng tarqalgan xatolar

**1. Iteratsiya paytida to'plamni o'zgartirish:**
\`\`\`go
// Noto'g'ri - aniqlanmagan xatti-harakat
iterator := collection.CreateIterator()
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        collection.RemoveBook(book)	// iteratsiya paytida o'zgartiradi!
    }
}

// To'g'ri - o'chirish uchun elementlarni to'plash, keyin o'chirish
var toRemove []*Book
for iterator.HasNext() {
    book := iterator.Next()
    if book.Title == "Bad Book" {
        toRemove = append(toRemove, book)
    }
}
for _, book := range toRemove {	// iteratsiyadan keyin o'chirish
    collection.RemoveBook(book)
}
\`\`\`

**2. Next dan oldin HasNext ni tekshirmaslik:**
\`\`\`go
// Noto'g'ri - nil qaytarishi mumkin
book := iterator.Next()	// HasNext tekshiruvi yo'q
fmt.Println(book.Title)	// nil bo'lsa panika!

// To'g'ri - har doim avval tekshirish
if iterator.HasNext() {
    book := iterator.Next()
    fmt.Println(book.Title)
}
// Yoki for sikl patternidan foydalanish
for iterator.HasNext() {
    book := iterator.Next()	// xavfsiz - HasNext allaqachon tekshirilgan
    fmt.Println(book.Title)
}
\`\`\`

**3. Iterator holatini ulashish:**
\`\`\`go
// Noto'g'ri - bitta iterator hammaga
iterator := collection.CreateIterator()
go process1(iterator)	// bir vaqtda kirish
go process2(iterator)	// poyga holati!

// To'g'ri - har bir goroutina o'z iteratorini oladi
go func() {
    iter := collection.CreateIterator()	// mustaqil iterator
    process1(iter)
}()
go func() {
    iter := collection.CreateIterator()	// boshqa mustaqil iterator
    process2(iter)
}()
\`\`\``
		}
	}
};

export default task;
