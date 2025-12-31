import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-iterator',
	title: 'Iterator Pattern',
	difficulty: 'easy',
	tags: ['java', 'design-patterns', 'behavioral', 'iterator'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Iterator Pattern

The **Iterator** pattern provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **Iterator** | Interface with \`hasNext()\` and \`next()\` methods |
| **ConcreteIterator** | Implements Iterator; tracks current position in collection |
| **Aggregate** | Interface with \`createIterator()\` method |
| **ConcreteAggregate** | Implements Aggregate; creates appropriate iterator |

---

### Your Task

Implement a book collection with iterator:

1. **Iterator interface** - \`hasNext()\` and \`next()\` methods
2. **BookIterator** - tracks position in array, returns books sequentially
3. **BookCollection (Aggregate)** - \`createIterator()\` returns BookIterator

---

### Example Usage

\`\`\`java
BookCollection collection = new BookCollection(5);	// create collection with capacity 5
collection.addBook("Clean Code");	// add first book
collection.addBook("Design Patterns");	// add second book
collection.addBook("Refactoring");	// add third book

Iterator<Book> iterator = collection.createIterator();	// get iterator from collection

while (iterator.hasNext()) {	// iterate while more elements exist
    Book book = iterator.next();	// get next book
    System.out.println(book.getTitle());	// print book title
}
// Output:
// Clean Code
// Design Patterns
// Refactoring
\`\`\`

---

### Key Insight

Iterator **separates traversal logic** from the collection itself. The client doesn't need to know if the collection is an array, linked list, tree, or any other data structure - it just uses hasNext() and next().`,
	initialCode: `interface Iterator<T> {
    boolean hasNext();
}

interface Aggregate<T> {
}

class Book {
    private String title;
    public Book(String title) { this.title = title; }
    public String getTitle() { return title; }
}

class BookIterator implements Iterator<Book> {
    private Book[] books;
    private int index = 0;

    public BookIterator(Book[] books) {
    }

    @Override
    public boolean hasNext() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public Book next() {
        throw new UnsupportedOperationException("TODO");
    }
}

class BookCollection implements Aggregate<Book> {
    private Book[] books;
    private int count = 0;

    public BookCollection(int size) {
    }

    public void addBook(String title) {
        }
    }

    @Override
    public Iterator<Book> createIterator() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface Iterator<T> {	// Iterator interface - defines traversal contract
    boolean hasNext();	// check if more elements exist
    T next();	// return next element
}

interface Aggregate<T> {	// Aggregate interface - collection that can be iterated
    Iterator<T> createIterator();	// factory method for creating iterator
}

class Book {	// Element class - items stored in collection
    private String title;	// book title

    public Book(String title) { this.title = title; }	// constructor
    public String getTitle() { return title; }	// getter
}

class BookIterator implements Iterator<Book> {	// ConcreteIterator - iterates over Book array
    private Book[] books;	// reference to the collection
    private int index = 0;	// current position in iteration

    public BookIterator(Book[] books) {	// constructor receives collection
        this.books = books;	// store reference to books array
    }

    @Override
    public boolean hasNext() {	// check if more books to iterate
        return index < books.length && books[index] != null;	// within bounds and not null
    }

    @Override
    public Book next() {	// return next book
        if (hasNext()) {	// if more elements exist
            return books[index++];	// return current and advance position
        }
        return null;	// no more elements
    }
}

class BookCollection implements Aggregate<Book> {	// ConcreteAggregate - collection of books
    private Book[] books;	// internal storage
    private int count = 0;	// number of books added

    public BookCollection(int size) {	// constructor with capacity
        books = new Book[size];	// initialize array
    }

    public void addBook(String title) {	// add book to collection
        if (count < books.length) {	// if space available
            books[count++] = new Book(title);	// create and add book
        }
    }

    @Override
    public Iterator<Book> createIterator() {	// factory method for iterator
        return new BookIterator(books);	// return new iterator over books
    }
}`,
	hint1: `### Understanding Iterator Structure

The Iterator pattern has two main interfaces:

\`\`\`java
// 1. Iterator interface - for traversal
interface Iterator<T> {
    boolean hasNext();  // Returns true if more elements exist
    T next();           // Returns current element and advances
}

// 2. Aggregate interface - for collections
interface Aggregate<T> {
    Iterator<T> createIterator();  // Factory method for iterators
}

// BookIterator implementation:
class BookIterator implements Iterator<Book> {
    private Book[] books;
    private int index = 0;  // Current position

    public boolean hasNext() {
        // Check: within array bounds AND element exists (not null)
        return index < books.length && books[index] != null;
    }

    public Book next() {
        if (hasNext()) {
            return books[index++];  // Return and increment
        }
        return null;
    }
}
\`\`\``,
	hint2: `### Implementing hasNext() and next()

**hasNext()** must check two conditions:
1. \`index < books.length\` - within array bounds
2. \`books[index] != null\` - element exists at current position

\`\`\`java
public boolean hasNext() {
    return index < books.length && books[index] != null;
}
\`\`\`

**next()** returns current element and advances position:
\`\`\`java
public Book next() {
    if (hasNext()) {           // Safety check
        return books[index++]; // Return current, then increment
    }
    return null;               // No more elements
}
\`\`\`

**createIterator()** simply returns a new iterator:
\`\`\`java
public Iterator<Book> createIterator() {
    return new BookIterator(books);  // Pass internal array
}
\`\`\``,
	whyItMatters: `## Why Iterator Pattern Matters

### The Problem and Solution

**Without Iterator:**
\`\`\`java
// Client knows internal structure - tight coupling
class BookCollection {
    public Book[] books;  // Exposed internal array!
}

// Client code depends on implementation details
BookCollection collection = new BookCollection();
for (int i = 0; i < collection.books.length; i++) {	// knows it's an array
    if (collection.books[i] != null) {	// knows about null elements
        System.out.println(collection.books[i].getTitle());
    }
}
// What if we change to ArrayList? Client code breaks!
\`\`\`

**With Iterator:**
\`\`\`java
// Client uses uniform interface - loose coupling
BookCollection collection = new BookCollection();
Iterator<Book> it = collection.createIterator();	// get iterator

while (it.hasNext()) {	// standard traversal
    Book book = it.next();	// get next element
    System.out.println(book.getTitle());
}
// Internal structure can change without breaking client code!
\`\`\`

---

### Real-World Applications

| Application | Iterator | Collection |
|-------------|----------|------------|
| **java.util.Iterator** | Iterator<E> | List, Set, Queue |
| **Enhanced for loop** | Implicit iterator | Any Iterable |
| **Stream API** | Spliterator | Collections, Arrays |
| **Database cursors** | ResultSet | Query results |
| **File readers** | BufferedReader | File lines |

---

### Production Pattern: Custom Collection with Multiple Iterators

\`\`\`java
// Aggregate interface with multiple iterator types
interface Playlist extends Iterable<Song> {	// iterable collection
    Iterator<Song> iterator();	// default order
    Iterator<Song> reverseIterator();	// reverse order
    Iterator<Song> shuffleIterator();	// random order
}

class Song {	// element in collection
    private final String title;	// song title
    private final String artist;	// song artist
    private final int durationSeconds;	// song duration

    public Song(String title, String artist, int duration) {	// constructor
        this.title = title;
        this.artist = artist;
        this.durationSeconds = duration;
    }
    // getters...
}

class MusicPlaylist implements Playlist {	// concrete collection
    private final List<Song> songs = new ArrayList<>();	// internal storage

    public void addSong(Song song) {	// add to playlist
        songs.add(song);	// add to internal list
    }

    @Override
    public Iterator<Song> iterator() {	// standard iterator (default order)
        return songs.iterator();	// use ArrayList's iterator
    }

    @Override
    public Iterator<Song> reverseIterator() {	// reverse order iterator
        return new Iterator<Song>() {	// anonymous iterator class
            private int index = songs.size() - 1;	// start from end

            @Override
            public boolean hasNext() { return index >= 0; }	// check bounds

            @Override
            public Song next() {
                if (!hasNext()) throw new NoSuchElementException();	// guard
                return songs.get(index--);	// return and move backward
            }
        };
    }

    @Override
    public Iterator<Song> shuffleIterator() {	// random order iterator
        List<Song> shuffled = new ArrayList<>(songs);	// copy list
        Collections.shuffle(shuffled);	// shuffle the copy
        return shuffled.iterator();	// return iterator over shuffled list
    }
}

// Filtered iterator - only songs by specific artist
class ArtistFilterIterator implements Iterator<Song> {	// filtering iterator
    private final Iterator<Song> innerIterator;	// wrapped iterator
    private final String artist;	// filter criterion
    private Song nextSong;	// buffered next element

    public ArtistFilterIterator(Iterator<Song> iterator, String artist) {
        this.innerIterator = iterator;	// store inner iterator
        this.artist = artist;	// store filter criterion
        findNext();	// find first matching element
    }

    private void findNext() {	// find next matching element
        nextSong = null;	// reset buffer
        while (innerIterator.hasNext()) {	// search through remaining elements
            Song song = innerIterator.next();	// get next
            if (song.getArtist().equals(artist)) {	// check filter
                nextSong = song;	// found match
                break;	// stop searching
            }
        }
    }

    @Override
    public boolean hasNext() { return nextSong != null; }	// check buffer

    @Override
    public Song next() {
        if (!hasNext()) throw new NoSuchElementException();	// guard
        Song result = nextSong;	// get buffered song
        findNext();	// find next match
        return result;	// return buffered song
    }
}

// Usage:
MusicPlaylist playlist = new MusicPlaylist();	// create playlist
playlist.addSong(new Song("Song 1", "Artist A", 180));
playlist.addSong(new Song("Song 2", "Artist B", 200));
playlist.addSong(new Song("Song 3", "Artist A", 220));

// Standard iteration
for (Song song : playlist) {	// uses default iterator via for-each
    System.out.println(song.getTitle());
}

// Reverse iteration
Iterator<Song> reverse = playlist.reverseIterator();
while (reverse.hasNext()) {
    System.out.println("Reverse: " + reverse.next().getTitle());
}

// Filtered iteration
Iterator<Song> filtered = new ArtistFilterIterator(playlist.iterator(), "Artist A");
while (filtered.hasNext()) {
    System.out.println("By Artist A: " + filtered.next().getTitle());
}
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Modifying collection while iterating** | ConcurrentModificationException | Use iterator's remove() or create copy |
| **Not checking hasNext()** | NoSuchElementException | Always check hasNext() before next() |
| **Reusing iterator** | Wrong position | Create new iterator for each traversal |
| **Exposing internal structure** | Tight coupling | Only expose Iterator interface |
| **Thread-unsafe iterator** | Race conditions | Use synchronized iterator or copy |`,
	order: 5,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: Empty collection hasNext is false
class Test1 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        Iterator<Book> it = collection.createIterator();
        assertFalse(it.hasNext());
    }
}

// Test2: Collection with book hasNext is true
class Test2 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("Book1");
        Iterator<Book> it = collection.createIterator();
        assertTrue(it.hasNext());
    }
}

// Test3: next returns correct book
class Test3 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("Clean Code");
        Iterator<Book> it = collection.createIterator();
        Book book = it.next();
        assertEquals("Clean Code", book.getTitle());
    }
}

// Test4: Iterator traverses all books
class Test4 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("A");
        collection.addBook("B");
        collection.addBook("C");
        Iterator<Book> it = collection.createIterator();
        int count = 0;
        while (it.hasNext()) {
            it.next();
            count++;
        }
        assertEquals(3, count);
    }
}

// Test5: hasNext false after all consumed
class Test5 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("Single");
        Iterator<Book> it = collection.createIterator();
        it.next();
        assertFalse(it.hasNext());
    }
}

// Test6: createIterator returns new iterator
class Test6 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("Book");
        Iterator<Book> it1 = collection.createIterator();
        Iterator<Book> it2 = collection.createIterator();
        assertNotSame(it1, it2);
    }
}

// Test7: Book stores title correctly
class Test7 {
    @Test
    public void test() {
        Book book = new Book("Design Patterns");
        assertEquals("Design Patterns", book.getTitle());
    }
}

// Test8: Iterator order is preserved
class Test8 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        collection.addBook("First");
        collection.addBook("Second");
        Iterator<Book> it = collection.createIterator();
        assertEquals("First", it.next().getTitle());
        assertEquals("Second", it.next().getTitle());
    }
}

// Test9: next returns null when exhausted
class Test9 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(5);
        Iterator<Book> it = collection.createIterator();
        Book book = it.next();
        assertNull(book);
    }
}

// Test10: Full collection iteration
class Test10 {
    @Test
    public void test() {
        BookCollection collection = new BookCollection(3);
        collection.addBook("One");
        collection.addBook("Two");
        collection.addBook("Three");
        Iterator<Book> it = collection.createIterator();
        StringBuilder sb = new StringBuilder();
        while (it.hasNext()) {
            sb.append(it.next().getTitle());
        }
        assertEquals("OneTwoThree", sb.toString());
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Iterator (Итератор)',
			description: `## Паттерн Iterator (Итератор)

Паттерн **Iterator** предоставляет способ последовательного доступа к элементам агрегатного объекта без раскрытия его внутреннего представления.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Iterator** | Интерфейс с методами \`hasNext()\` и \`next()\` |
| **ConcreteIterator** | Реализует Iterator; отслеживает текущую позицию |
| **Aggregate** | Интерфейс с методом \`createIterator()\` |
| **ConcreteAggregate** | Реализует Aggregate; создаёт соответствующий итератор |

---

### Ваша задача

Реализуйте коллекцию книг с итератором:

1. **Интерфейс Iterator** - методы \`hasNext()\` и \`next()\`
2. **BookIterator** - отслеживает позицию в массиве, возвращает книги последовательно
3. **BookCollection (Aggregate)** - \`createIterator()\` возвращает BookIterator

---

### Пример использования

\`\`\`java
BookCollection collection = new BookCollection(5);	// создаём коллекцию с ёмкостью 5
collection.addBook("Clean Code");	// добавляем первую книгу
collection.addBook("Design Patterns");	// добавляем вторую книгу
collection.addBook("Refactoring");	// добавляем третью книгу

Iterator<Book> iterator = collection.createIterator();	// получаем итератор из коллекции

while (iterator.hasNext()) {	// итерируем пока есть элементы
    Book book = iterator.next();	// получаем следующую книгу
    System.out.println(book.getTitle());	// выводим название книги
}
// Вывод:
// Clean Code
// Design Patterns
// Refactoring
\`\`\`

---

### Ключевая идея

Iterator **отделяет логику обхода** от самой коллекции. Клиенту не нужно знать, является ли коллекция массивом, связным списком, деревом или любой другой структурой данных — он просто использует hasNext() и next().`,
			hint1: `### Понимание структуры Iterator

Паттерн Iterator имеет два основных интерфейса:

\`\`\`java
// 1. Интерфейс Iterator - для обхода
interface Iterator<T> {
    boolean hasNext();  // Возвращает true если есть ещё элементы
    T next();           // Возвращает текущий элемент и продвигается
}

// 2. Интерфейс Aggregate - для коллекций
interface Aggregate<T> {
    Iterator<T> createIterator();  // Фабричный метод для итераторов
}

// Реализация BookIterator:
class BookIterator implements Iterator<Book> {
    private Book[] books;
    private int index = 0;  // Текущая позиция

    public boolean hasNext() {
        // Проверка: в пределах массива И элемент существует (не null)
        return index < books.length && books[index] != null;
    }

    public Book next() {
        if (hasNext()) {
            return books[index++];  // Вернуть и инкрементировать
        }
        return null;
    }
}
\`\`\``,
			hint2: `### Реализация hasNext() и next()

**hasNext()** должен проверять два условия:
1. \`index < books.length\` - в пределах границ массива
2. \`books[index] != null\` - элемент существует на текущей позиции

\`\`\`java
public boolean hasNext() {
    return index < books.length && books[index] != null;
}
\`\`\`

**next()** возвращает текущий элемент и продвигает позицию:
\`\`\`java
public Book next() {
    if (hasNext()) {           // Проверка безопасности
        return books[index++]; // Вернуть текущий, затем инкрементировать
    }
    return null;               // Больше нет элементов
}
\`\`\`

**createIterator()** просто возвращает новый итератор:
\`\`\`java
public Iterator<Book> createIterator() {
    return new BookIterator(books);  // Передать внутренний массив
}
\`\`\``,
			whyItMatters: `## Почему паттерн Iterator важен

### Проблема и решение

**Без Iterator:**
\`\`\`java
// Клиент знает внутреннюю структуру - тесная связанность
class BookCollection {
    public Book[] books;  // Открытый внутренний массив!
}

// Клиентский код зависит от деталей реализации
BookCollection collection = new BookCollection();
for (int i = 0; i < collection.books.length; i++) {	// знает что это массив
    if (collection.books[i] != null) {	// знает о null элементах
        System.out.println(collection.books[i].getTitle());
    }
}
// Что если мы поменяем на ArrayList? Клиентский код сломается!
\`\`\`

**С Iterator:**
\`\`\`java
// Клиент использует единый интерфейс - слабая связанность
BookCollection collection = new BookCollection();
Iterator<Book> it = collection.createIterator();	// получить итератор

while (it.hasNext()) {	// стандартный обход
    Book book = it.next();	// получить следующий элемент
    System.out.println(book.getTitle());
}
// Внутренняя структура может меняться без поломки клиентского кода!
\`\`\`

---

### Применение в реальном мире

| Применение | Iterator | Коллекция |
|------------|----------|-----------|
| **java.util.Iterator** | Iterator<E> | List, Set, Queue |
| **Enhanced for loop** | Неявный итератор | Любой Iterable |
| **Stream API** | Spliterator | Collections, Arrays |
| **Курсоры БД** | ResultSet | Результаты запроса |
| **Чтение файлов** | BufferedReader | Строки файла |

---

### Продакшен паттерн: Коллекция с множественными итераторами

\`\`\`java
// Интерфейс Aggregate с несколькими типами итераторов
interface Playlist extends Iterable<Song> {	// итерируемая коллекция
    Iterator<Song> iterator();	// порядок по умолчанию
    Iterator<Song> reverseIterator();	// обратный порядок
    Iterator<Song> shuffleIterator();	// случайный порядок
}

class Song {	// элемент коллекции
    private final String title;	// название песни
    private final String artist;	// исполнитель
    private final int durationSeconds;	// длительность

    public Song(String title, String artist, int duration) {	// конструктор
        this.title = title;
        this.artist = artist;
        this.durationSeconds = duration;
    }
    // геттеры...
}

class MusicPlaylist implements Playlist {	// конкретная коллекция
    private final List<Song> songs = new ArrayList<>();	// внутреннее хранилище

    public void addSong(Song song) {	// добавить в плейлист
        songs.add(song);	// добавить во внутренний список
    }

    @Override
    public Iterator<Song> iterator() {	// стандартный итератор (порядок по умолчанию)
        return songs.iterator();	// использовать итератор ArrayList
    }

    @Override
    public Iterator<Song> reverseIterator() {	// итератор обратного порядка
        return new Iterator<Song>() {	// анонимный класс итератора
            private int index = songs.size() - 1;	// начать с конца

            @Override
            public boolean hasNext() { return index >= 0; }	// проверить границы

            @Override
            public Song next() {
                if (!hasNext()) throw new NoSuchElementException();	// защита
                return songs.get(index--);	// вернуть и двигаться назад
            }
        };
    }

    @Override
    public Iterator<Song> shuffleIterator() {	// итератор случайного порядка
        List<Song> shuffled = new ArrayList<>(songs);	// копировать список
        Collections.shuffle(shuffled);	// перемешать копию
        return shuffled.iterator();	// вернуть итератор по перемешанному
    }
}

// Фильтрующий итератор - только песни конкретного исполнителя
class ArtistFilterIterator implements Iterator<Song> {	// фильтрующий итератор
    private final Iterator<Song> innerIterator;	// обёрнутый итератор
    private final String artist;	// критерий фильтра
    private Song nextSong;	// буферизованный следующий элемент

    public ArtistFilterIterator(Iterator<Song> iterator, String artist) {
        this.innerIterator = iterator;	// сохранить внутренний итератор
        this.artist = artist;	// сохранить критерий фильтра
        findNext();	// найти первый подходящий элемент
    }

    private void findNext() {	// найти следующий подходящий элемент
        nextSong = null;	// сбросить буфер
        while (innerIterator.hasNext()) {	// искать в оставшихся элементах
            Song song = innerIterator.next();	// получить следующий
            if (song.getArtist().equals(artist)) {	// проверить фильтр
                nextSong = song;	// найдено совпадение
                break;	// остановить поиск
            }
        }
    }

    @Override
    public boolean hasNext() { return nextSong != null; }	// проверить буфер

    @Override
    public Song next() {
        if (!hasNext()) throw new NoSuchElementException();	// защита
        Song result = nextSong;	// получить буферизованную песню
        findNext();	// найти следующее совпадение
        return result;	// вернуть буферизованную песню
    }
}

// Использование:
MusicPlaylist playlist = new MusicPlaylist();	// создать плейлист
playlist.addSong(new Song("Song 1", "Artist A", 180));
playlist.addSong(new Song("Song 2", "Artist B", 200));
playlist.addSong(new Song("Song 3", "Artist A", 220));

// Стандартная итерация
for (Song song : playlist) {	// использует итератор по умолчанию через for-each
    System.out.println(song.getTitle());
}

// Обратная итерация
Iterator<Song> reverse = playlist.reverseIterator();
while (reverse.hasNext()) {
    System.out.println("Reverse: " + reverse.next().getTitle());
}

// Фильтрованная итерация
Iterator<Song> filtered = new ArtistFilterIterator(playlist.iterator(), "Artist A");
while (filtered.hasNext()) {
    System.out.println("By Artist A: " + filtered.next().getTitle());
}
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Модификация коллекции во время итерации** | ConcurrentModificationException | Используйте remove() итератора или создайте копию |
| **Не проверять hasNext()** | NoSuchElementException | Всегда проверяйте hasNext() перед next() |
| **Повторное использование итератора** | Неправильная позиция | Создавайте новый итератор для каждого обхода |
| **Раскрытие внутренней структуры** | Тесная связанность | Раскрывайте только интерфейс Iterator |
| **Потоконебезопасный итератор** | Race conditions | Используйте синхронизированный итератор или копию |`
		},
		uz: {
			title: 'Iterator Pattern',
			description: `## Iterator Pattern

**Iterator** pattern agregat obyektining elementlariga uning ichki ko'rinishini ochmasdan ketma-ket kirishni ta'minlaydi.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **Iterator** | \`hasNext()\` va \`next()\` metodlari bilan interfeys |
| **ConcreteIterator** | Iterator ni amalga oshiradi; kolleksiyada joriy pozitsiyani kuzatadi |
| **Aggregate** | \`createIterator()\` metodi bilan interfeys |
| **ConcreteAggregate** | Aggregate ni amalga oshiradi; tegishli iteratorni yaratadi |

---

### Vazifangiz

Iterator bilan kitoblar kolleksiyasini amalga oshiring:

1. **Iterator interfeysi** - \`hasNext()\` va \`next()\` metodlari
2. **BookIterator** - massivda pozitsiyani kuzatadi, kitoblarni ketma-ket qaytaradi
3. **BookCollection (Aggregate)** - \`createIterator()\` BookIterator qaytaradi

---

### Foydalanish Namunasi

\`\`\`java
BookCollection collection = new BookCollection(5);	// 5 sig'imli kolleksiya yaratamiz
collection.addBook("Clean Code");	// birinchi kitobni qo'shamiz
collection.addBook("Design Patterns");	// ikkinchi kitobni qo'shamiz
collection.addBook("Refactoring");	// uchinchi kitobni qo'shamiz

Iterator<Book> iterator = collection.createIterator();	// kolleksiyadan iterator olamiz

while (iterator.hasNext()) {	// ko'proq elementlar mavjud ekan iteratsiya
    Book book = iterator.next();	// keyingi kitobni olamiz
    System.out.println(book.getTitle());	// kitob nomini chop etamiz
}
// Chiqish:
// Clean Code
// Design Patterns
// Refactoring
\`\`\`

---

### Asosiy Fikr

Iterator **o'tish mantiqini** kolleksiyaning o'zidan ajratadi. Mijoz kolleksiya massiv, bog'langan ro'yxat, daraxt yoki boshqa ma'lumotlar strukturasi ekanligini bilishi shart emas — u shunchaki hasNext() va next() ishlatadi.`,
			hint1: `### Iterator Strukturasini Tushunish

Iterator pattern ikkita asosiy interfeysga ega:

\`\`\`java
// 1. Iterator interfeysi - o'tish uchun
interface Iterator<T> {
    boolean hasNext();  // Ko'proq elementlar mavjud bo'lsa true qaytaradi
    T next();           // Joriy elementni qaytaradi va oldinga siljiydi
}

// 2. Aggregate interfeysi - kolleksiyalar uchun
interface Aggregate<T> {
    Iterator<T> createIterator();  // Iteratorlar uchun fabrika metodi
}

// BookIterator amalga oshirish:
class BookIterator implements Iterator<Book> {
    private Book[] books;
    private int index = 0;  // Joriy pozitsiya

    public boolean hasNext() {
        // Tekshirish: massiv chegarasida VA element mavjud (null emas)
        return index < books.length && books[index] != null;
    }

    public Book next() {
        if (hasNext()) {
            return books[index++];  // Qaytarish va oshirish
        }
        return null;
    }
}
\`\`\``,
			hint2: `### hasNext() va next() ni Amalga Oshirish

**hasNext()** ikki shartni tekshirishi kerak:
1. \`index < books.length\` - massiv chegarasida
2. \`books[index] != null\` - joriy pozitsiyada element mavjud

\`\`\`java
public boolean hasNext() {
    return index < books.length && books[index] != null;
}
\`\`\`

**next()** joriy elementni qaytaradi va pozitsiyani siljitadi:
\`\`\`java
public Book next() {
    if (hasNext()) {           // Xavfsizlik tekshiruvi
        return books[index++]; // Joriyni qaytarish, keyin oshirish
    }
    return null;               // Ko'proq elementlar yo'q
}
\`\`\`

**createIterator()** shunchaki yangi iterator qaytaradi:
\`\`\`java
public Iterator<Book> createIterator() {
    return new BookIterator(books);  // Ichki massivni uzatish
}
\`\`\``,
			whyItMatters: `## Nima Uchun Iterator Pattern Muhim

### Muammo va Yechim

**Iterator siz:**
\`\`\`java
// Mijoz ichki strukturani biladi - qattiq bog'lanish
class BookCollection {
    public Book[] books;  // Ochiq ichki massiv!
}

// Mijoz kodi amalga oshirish tafsilotlariga bog'liq
BookCollection collection = new BookCollection();
for (int i = 0; i < collection.books.length; i++) {	// massiv ekanini biladi
    if (collection.books[i] != null) {	// null elementlar haqida biladi
        System.out.println(collection.books[i].getTitle());
    }
}
// ArrayList ga o'zgartirsak nima bo'ladi? Mijoz kodi buziladi!
\`\`\`

**Iterator bilan:**
\`\`\`java
// Mijoz yagona interfeys ishlatadi - zaif bog'lanish
BookCollection collection = new BookCollection();
Iterator<Book> it = collection.createIterator();	// iterator olish

while (it.hasNext()) {	// standart o'tish
    Book book = it.next();	// keyingi elementni olish
    System.out.println(book.getTitle());
}
// Ichki struktura mijoz kodini buzmasdan o'zgarishi mumkin!
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Iterator | Kolleksiya |
|------------|----------|------------|
| **java.util.Iterator** | Iterator<E> | List, Set, Queue |
| **Enhanced for loop** | Yashirin iterator | Har qanday Iterable |
| **Stream API** | Spliterator | Collections, Arrays |
| **DB kursorlari** | ResultSet | So'rov natijalari |
| **Fayl o'quvchilar** | BufferedReader | Fayl satrlari |

---

### Prodakshen Pattern: Bir Nechta Iteratorli Maxsus Kolleksiya

\`\`\`java
// Bir nechta iterator turlari bilan Aggregate interfeysi
interface Playlist extends Iterable<Song> {	// iteratsiya qilinadigan kolleksiya
    Iterator<Song> iterator();	// standart tartib
    Iterator<Song> reverseIterator();	// teskari tartib
    Iterator<Song> shuffleIterator();	// tasodifiy tartib
}

class Song {	// kolleksiyada element
    private final String title;	// qo'shiq nomi
    private final String artist;	// ijrochi
    private final int durationSeconds;	// davomiylik

    public Song(String title, String artist, int duration) {	// konstruktor
        this.title = title;
        this.artist = artist;
        this.durationSeconds = duration;
    }
    // getterlar...
}

class MusicPlaylist implements Playlist {	// aniq kolleksiya
    private final List<Song> songs = new ArrayList<>();	// ichki saqlash

    public void addSong(Song song) {	// pleylistga qo'shish
        songs.add(song);	// ichki ro'yxatga qo'shish
    }

    @Override
    public Iterator<Song> iterator() {	// standart iterator (standart tartib)
        return songs.iterator();	// ArrayList ning iteratorini ishlatish
    }

    @Override
    public Iterator<Song> reverseIterator() {	// teskari tartib iteratori
        return new Iterator<Song>() {	// anonim iterator klassi
            private int index = songs.size() - 1;	// oxiridan boshlash

            @Override
            public boolean hasNext() { return index >= 0; }	// chegaralarni tekshirish

            @Override
            public Song next() {
                if (!hasNext()) throw new NoSuchElementException();	// himoya
                return songs.get(index--);	// qaytarish va orqaga siljish
            }
        };
    }

    @Override
    public Iterator<Song> shuffleIterator() {	// tasodifiy tartib iteratori
        List<Song> shuffled = new ArrayList<>(songs);	// ro'yxatni nusxalash
        Collections.shuffle(shuffled);	// nusxani aralashtirish
        return shuffled.iterator();	// aralashtirilgan ustida iterator qaytarish
    }
}

// Filtrlash iteratori - faqat ma'lum ijrochining qo'shiqlari
class ArtistFilterIterator implements Iterator<Song> {	// filtrlash iteratori
    private final Iterator<Song> innerIterator;	// o'ralgan iterator
    private final String artist;	// filtr mezoni
    private Song nextSong;	// buferlanhan keyingi element

    public ArtistFilterIterator(Iterator<Song> iterator, String artist) {
        this.innerIterator = iterator;	// ichki iteratorni saqlash
        this.artist = artist;	// filtr mezonini saqlash
        findNext();	// birinchi mos elementni topish
    }

    private void findNext() {	// keyingi mos elementni topish
        nextSong = null;	// buferni tozalash
        while (innerIterator.hasNext()) {	// qolgan elementlarda qidirish
            Song song = innerIterator.next();	// keyingisini olish
            if (song.getArtist().equals(artist)) {	// filtrni tekshirish
                nextSong = song;	// moslik topildi
                break;	// qidirishni to'xtatish
            }
        }
    }

    @Override
    public boolean hasNext() { return nextSong != null; }	// buferni tekshirish

    @Override
    public Song next() {
        if (!hasNext()) throw new NoSuchElementException();	// himoya
        Song result = nextSong;	// buferlangan qo'shiqni olish
        findNext();	// keyingi moslikni topish
        return result;	// buferlangan qo'shiqni qaytarish
    }
}

// Foydalanish:
MusicPlaylist playlist = new MusicPlaylist();	// pleylist yaratish
playlist.addSong(new Song("Song 1", "Artist A", 180));
playlist.addSong(new Song("Song 2", "Artist B", 200));
playlist.addSong(new Song("Song 3", "Artist A", 220));

// Standart iteratsiya
for (Song song : playlist) {	// for-each orqali standart iteratorni ishlatadi
    System.out.println(song.getTitle());
}

// Teskari iteratsiya
Iterator<Song> reverse = playlist.reverseIterator();
while (reverse.hasNext()) {
    System.out.println("Reverse: " + reverse.next().getTitle());
}

// Filtrlangan iteratsiya
Iterator<Song> filtered = new ArtistFilterIterator(playlist.iterator(), "Artist A");
while (filtered.hasNext()) {
    System.out.println("By Artist A: " + filtered.next().getTitle());
}
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Iteratsiya paytida kolleksiyani o'zgartirish** | ConcurrentModificationException | Iteratorning remove() ni ishlating yoki nusxa yarating |
| **hasNext() ni tekshirmaslik** | NoSuchElementException | next() dan oldin har doim hasNext() ni tekshiring |
| **Iteratorni qayta ishlatish** | Noto'g'ri pozitsiya | Har bir o'tish uchun yangi iterator yarating |
| **Ichki strukturani ochish** | Qattiq bog'lanish | Faqat Iterator interfeysini oching |
| **Thread-xavfsiz bo'lmagan iterator** | Poyga sharoitlari | Sinxronlangan iterator yoki nusxa ishlating |`
		}
	}
};

export default task;
