import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-proxy',
	title: 'Proxy Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'structural', 'proxy'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Proxy Pattern** in Java — provide a surrogate or placeholder to control access to another object.

## Overview

The Proxy pattern provides a substitute for another object to control access, reduce cost of creation, or add functionality. The proxy has the same interface as the real object.

## Proxy Types

| Type | Purpose | Example |
|------|---------|---------|
| **Virtual Proxy** | Lazy loading | Defer expensive object creation |
| **Protection Proxy** | Access control | Check permissions before delegating |
| **Remote Proxy** | Network communication | RMI stubs |
| **Caching Proxy** | Cache results | Avoid repeated expensive operations |

## Your Task

Implement a virtual proxy for image lazy loading:

1. **Image** interface - Subject with display() method
2. **RealImage** - Heavy object that loads from disk on creation
3. **ProxyImage** - Lightweight proxy that creates RealImage only when needed

## Example Usage

\`\`\`java
Image image1 = new ProxyImage("photo1.jpg");	// no loading yet!
Image image2 = new ProxyImage("photo2.jpg");	// no loading yet!

// Only loads when actually displayed
image1.display();	// "Loading photo1.jpg" then "Displaying photo1.jpg"
image1.display();	// "Displaying photo1.jpg" (no reload)
\`\`\`

## Key Insight

ProxyImage is lightweight — RealImage is only created when first needed, saving resources!`,
	initialCode: `interface Image {
    String display();
}

class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
    }

    private void loadFromDisk() {
    }

    @Override
    public String display() {
        throw new UnsupportedOperationException("TODO");
    }
}

class ProxyImage implements Image {
    private String filename;
    private RealImage realImage;

    public ProxyImage(String filename) {
    }

    @Override
    public String display() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface Image {	// Subject - common interface for real and proxy
    String display();	// method to display image
}

class RealImage implements Image {	// RealSubject - expensive object
    private String filename;	// image filename

    public RealImage(String filename) {	// constructor - loads immediately
        this.filename = filename;	// store filename
        loadFromDisk();	// expensive operation on creation
    }

    private void loadFromDisk() {	// simulate expensive loading
        System.out.println("Loading " + filename);	// log loading operation
    }

    @Override
    public String display() {	// actual display implementation
        return "Displaying " + filename;	// return display message
    }
}

class ProxyImage implements Image {	// Proxy - controls access to RealImage
    private String filename;	// store filename for lazy creation
    private RealImage realImage;	// reference to real object (initially null)

    public ProxyImage(String filename) {	// lightweight constructor
        this.filename = filename;	// just store filename - no loading!
    }

    @Override
    public String display() {	// lazy loading implementation
        if (realImage == null) {	// check if real object exists
            realImage = new RealImage(filename);	// create only when needed
        }
        return realImage.display();	// delegate to real object
    }
}`,
	hint1: `**RealImage Implementation**

RealImage is the heavy object that loads on construction:

\`\`\`java
class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();  // Called immediately!
    }

    private void loadFromDisk() {
        System.out.println("Loading " + filename);
    }

    @Override
    public String display() {
        return "Displaying " + filename;
    }
}
\`\`\`

The loading happens in constructor — this is what proxy helps defer.`,
	hint2: `**ProxyImage Implementation**

ProxyImage defers creation until actually needed:

\`\`\`java
class ProxyImage implements Image {
    private String filename;
    private RealImage realImage;  // Initially null

    public ProxyImage(String filename) {
        this.filename = filename;  // Just store, don't load
    }

    @Override
    public String display() {
        if (realImage == null) {        // First access?
            realImage = new RealImage(filename);  // Create now
        }
        return realImage.display();     // Delegate
    }
}
\`\`\`

Key: Only create RealImage on first display() call!`,
	whyItMatters: `## Problem & Solution

**Without Proxy:**
\`\`\`java
// All images load immediately - slow startup!
Image img1 = new RealImage("huge1.jpg");	// loads immediately - 2 seconds
Image img2 = new RealImage("huge2.jpg");	// loads immediately - 2 seconds
Image img3 = new RealImage("huge3.jpg");	// loads immediately - 2 seconds
// User waits 6 seconds before seeing anything!	// poor UX
\`\`\`

**With Proxy:**
\`\`\`java
// Images load only when displayed - instant startup!
Image img1 = new ProxyImage("huge1.jpg");	// instant - no loading
Image img2 = new ProxyImage("huge2.jpg");	// instant - no loading
Image img3 = new ProxyImage("huge3.jpg");	// instant - no loading
// User sees UI immediately!	// great UX

img1.display();	// loads only when needed
\`\`\`

---

## Real-World Examples

| Domain | Proxy Type | Real Subject |
|--------|------------|--------------|
| **Hibernate** | Virtual | Entity objects (lazy loading) |
| **Spring AOP** | Enhancement | Service beans (@Transactional) |
| **Java RMI** | Remote | Remote objects |
| **CDI** | Protection | Security-controlled beans |
| **JPA EntityManager** | Caching | Database entities |
| **Browser images** | Virtual | Large image files |

---

## Production Pattern: Access Control Proxy

\`\`\`java
// Subject interface
interface Document {	// common interface
    String read();	// read document content
    void write(String content);	// write to document
    void delete();	// delete document
}

// Real Subject - actual document
class RealDocument implements Document {	// real implementation
    private String filename;	// document path
    private String content;	// document content

    public RealDocument(String filename) {	// constructor
        this.filename = filename;	// store filename
        this.content = loadContent();	// load from storage
    }

    private String loadContent() {	// simulate loading
        System.out.println("Loading document: " + filename);	// log loading
        return "Content of " + filename;	// return content
    }

    @Override
    public String read() {	// read implementation
        return content;	// return content
    }

    @Override
    public void write(String content) {	// write implementation
        this.content = content;	// update content
        System.out.println("Saving document: " + filename);	// log save
    }

    @Override
    public void delete() {	// delete implementation
        System.out.println("Deleting document: " + filename);	// log delete
    }
}

// Protection Proxy - controls access based on user role
class SecureDocumentProxy implements Document {	// access control proxy
    private String filename;	// document identifier
    private RealDocument realDocument;	// lazy-loaded real document
    private User currentUser;	// user making requests

    public SecureDocumentProxy(String filename, User currentUser) {	// constructor
        this.filename = filename;	// store filename
        this.currentUser = currentUser;	// store user
    }

    private RealDocument getDocument() {	// lazy loading
        if (realDocument == null) {	// check if loaded
            realDocument = new RealDocument(filename);	// load on demand
        }
        return realDocument;	// return document
    }

    @Override
    public String read() {	// read with access check
        if (!currentUser.hasPermission("READ")) {	// check permission
            throw new SecurityException("No read permission");	// deny access
        }
        logAccess("READ");	// audit logging
        return getDocument().read();	// delegate to real
    }

    @Override
    public void write(String content) {	// write with access check
        if (!currentUser.hasPermission("WRITE")) {	// check permission
            throw new SecurityException("No write permission");	// deny access
        }
        logAccess("WRITE");	// audit logging
        getDocument().write(content);	// delegate to real
    }

    @Override
    public void delete() {	// delete with access check
        if (!currentUser.hasPermission("DELETE")) {	// check permission
            throw new SecurityException("No delete permission");	// deny access
        }
        logAccess("DELETE");	// audit logging
        getDocument().delete();	// delegate to real
    }

    private void logAccess(String operation) {	// audit logging
        System.out.println("User " + currentUser.getName() +	// log user
            " performed " + operation + " on " + filename);	// log operation
    }
}

// User class for permissions
class User {	// user with permissions
    private String name;	// user name
    private Set<String> permissions;	// user permissions

    public User(String name, String... perms) {	// constructor
        this.name = name;	// store name
        this.permissions = new HashSet<>(Arrays.asList(perms));	// store permissions
    }

    public String getName() { return name; }	// get name
    public boolean hasPermission(String perm) {	// check permission
        return permissions.contains(perm);	// return result
    }
}

// Usage
User admin = new User("admin", "READ", "WRITE", "DELETE");	// admin with all permissions
User reader = new User("guest", "READ");	// guest with read-only

Document adminDoc = new SecureDocumentProxy("secret.txt", admin);	// admin's proxy
Document readerDoc = new SecureDocumentProxy("secret.txt", reader);	// guest's proxy

adminDoc.read();	// OK - admin can read
adminDoc.write("New content");	// OK - admin can write

readerDoc.read();	// OK - guest can read
readerDoc.write("Hack!");	// SecurityException - no write permission!
\`\`\`

---

## Caching Proxy Example

\`\`\`java
interface DataService {	// service interface
    String fetchData(String key);	// fetch data by key
}

class RealDataService implements DataService {	// real service - slow
    @Override
    public String fetchData(String key) {	// expensive operation
        System.out.println("Fetching from database: " + key);	// log fetch
        sleep(1000);	// simulate slow database
        return "Data for " + key;	// return data
    }
}

class CachingProxy implements DataService {	// caching proxy
    private DataService realService;	// real service
    private Map<String, String> cache = new HashMap<>();	// cache storage

    public CachingProxy(DataService realService) {	// constructor
        this.realService = realService;	// store real service
    }

    @Override
    public String fetchData(String key) {	// cached fetch
        if (cache.containsKey(key)) {	// check cache
            System.out.println("Cache hit: " + key);	// log cache hit
            return cache.get(key);	// return cached
        }
        String data = realService.fetchData(key);	// fetch from real
        cache.put(key, data);	// store in cache
        return data;	// return data
    }
}

// Usage
DataService service = new CachingProxy(new RealDataService());	// wrap with cache

service.fetchData("user:123");	// slow - fetches from DB
service.fetchData("user:123");	// instant - from cache!
service.fetchData("user:123");	// instant - from cache!
\`\`\`

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Different interface** | Proxy not interchangeable with real | Implement same interface as real subject |
| **Eager loading in proxy** | Creating real object in proxy constructor | Only create in methods when needed |
| **Missing null check** | NullPointerException on cached reference | Always check before creating |
| **Thread safety** | Race condition in lazy initialization | Use synchronized or double-checked locking |
| **Memory leaks** | Proxy holds reference forever | Consider weak references or clear method |`,
	order: 6,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

class Test1 {
    @Test
    void realImageDisplaysCorrectly() {
        RealImage image = new RealImage("photo.jpg");
        String result = image.display();
        assertEquals("Displaying photo.jpg", result, "RealImage should display correctly");
    }
}

class Test2 {
    @Test
    void realImageLoadsOnConstruction() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(out));
        new RealImage("test.jpg");
        System.setOut(originalOut);
        assertTrue(out.toString().contains("Loading"), "RealImage should load on construction");
    }
}

class Test3 {
    @Test
    void proxyImageDoesNotLoadOnConstruction() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(out));
        new ProxyImage("test.jpg");
        System.setOut(originalOut);
        assertFalse(out.toString().contains("Loading"), "ProxyImage should not load on construction");
    }
}

class Test4 {
    @Test
    void proxyImageLoadsOnFirstDisplay() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(out));
        Image proxy = new ProxyImage("test.jpg");
        proxy.display();
        System.setOut(originalOut);
        assertTrue(out.toString().contains("Loading"), "ProxyImage should load on first display");
    }
}

class Test5 {
    @Test
    void proxyImageDisplaysCorrectly() {
        Image proxy = new ProxyImage("photo.jpg");
        String result = proxy.display();
        assertEquals("Displaying photo.jpg", result, "ProxyImage should display correctly");
    }
}

class Test6 {
    @Test
    void proxyImageDoesNotReloadOnSecondDisplay() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        Image proxy = new ProxyImage("test.jpg");
        proxy.display();
        System.setOut(new PrintStream(out));
        proxy.display();
        System.setOut(originalOut);
        assertFalse(out.toString().contains("Loading"), "ProxyImage should not reload on second display");
    }
}

class Test7 {
    @Test
    void proxyImageImplementsSameInterface() {
        Image proxy = new ProxyImage("test.jpg");
        assertTrue(proxy instanceof Image, "ProxyImage should implement Image interface");
    }
}

class Test8 {
    @Test
    void realImageImplementsSameInterface() {
        Image real = new RealImage("test.jpg");
        assertTrue(real instanceof Image, "RealImage should implement Image interface");
    }
}

class Test9 {
    @Test
    void multipleProxiesLoadIndependently() {
        Image proxy1 = new ProxyImage("file1.jpg");
        Image proxy2 = new ProxyImage("file2.jpg");
        String result1 = proxy1.display();
        String result2 = proxy2.display();
        assertTrue(result1.contains("file1"), "First proxy should display file1");
        assertTrue(result2.contains("file2"), "Second proxy should display file2");
    }
}

class Test10 {
    @Test
    void proxyDisplayReturnsRealImageResult() {
        Image proxy = new ProxyImage("test.jpg");
        Image real = new RealImage("test.jpg");
        assertEquals(real.display(), proxy.display(), "Proxy display should match real display");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Proxy (Заместитель)',
			description: `Реализуйте **паттерн Proxy** на Java — предоставьте суррогат или заместитель для контроля доступа к другому объекту.

## Обзор

Паттерн Proxy предоставляет заменитель другого объекта для контроля доступа, снижения стоимости создания или добавления функциональности. Прокси имеет тот же интерфейс, что и реальный объект.

## Типы прокси

| Тип | Назначение | Пример |
|-----|------------|--------|
| **Virtual Proxy** | Ленивая загрузка | Отложить создание дорогого объекта |
| **Protection Proxy** | Контроль доступа | Проверка прав перед делегированием |
| **Remote Proxy** | Сетевое взаимодействие | RMI stubs |
| **Caching Proxy** | Кэширование результатов | Избежать повторных дорогих операций |

## Ваша задача

Реализуйте виртуальный прокси для ленивой загрузки изображений:

1. **Image** интерфейс - Subject с методом display()
2. **RealImage** - Тяжёлый объект, загружающийся с диска при создании
3. **ProxyImage** - Легковесный прокси, создающий RealImage только при необходимости

## Пример использования

\`\`\`java
Image image1 = new ProxyImage("photo1.jpg");	// пока не загружается!
Image image2 = new ProxyImage("photo2.jpg");	// пока не загружается!

// Загружается только при фактическом отображении
image1.display();	// "Loading photo1.jpg" затем "Displaying photo1.jpg"
image1.display();	// "Displaying photo1.jpg" (без перезагрузки)
\`\`\`

## Ключевая идея

ProxyImage легковесный — RealImage создаётся только при первой необходимости, экономя ресурсы!`,
			hint1: `**Реализация RealImage**

RealImage — тяжёлый объект, загружающийся при создании:

\`\`\`java
class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();  // Вызывается сразу!
    }

    private void loadFromDisk() {
        System.out.println("Loading " + filename);
    }

    @Override
    public String display() {
        return "Displaying " + filename;
    }
}
\`\`\`

Загрузка происходит в конструкторе — это то, что прокси помогает отложить.`,
			hint2: `**Реализация ProxyImage**

ProxyImage откладывает создание до фактической необходимости:

\`\`\`java
class ProxyImage implements Image {
    private String filename;
    private RealImage realImage;  // Изначально null

    public ProxyImage(String filename) {
        this.filename = filename;  // Просто сохраняем, не загружаем
    }

    @Override
    public String display() {
        if (realImage == null) {        // Первый доступ?
            realImage = new RealImage(filename);  // Создаём сейчас
        }
        return realImage.display();     // Делегируем
    }
}
\`\`\`

Ключ: Создавать RealImage только при первом вызове display()!`,
			whyItMatters: `## Проблема и решение

**Без Proxy:**
\`\`\`java
// Все изображения загружаются сразу - медленный запуск!
Image img1 = new RealImage("huge1.jpg");	// загружается сразу - 2 секунды
Image img2 = new RealImage("huge2.jpg");	// загружается сразу - 2 секунды
Image img3 = new RealImage("huge3.jpg");	// загружается сразу - 2 секунды
// Пользователь ждёт 6 секунд до отображения!	// плохой UX
\`\`\`

**С Proxy:**
\`\`\`java
// Изображения загружаются только при отображении - мгновенный запуск!
Image img1 = new ProxyImage("huge1.jpg");	// мгновенно - без загрузки
Image img2 = new ProxyImage("huge2.jpg");	// мгновенно - без загрузки
Image img3 = new ProxyImage("huge3.jpg");	// мгновенно - без загрузки
// Пользователь видит UI сразу!	// отличный UX

img1.display();	// загружается только при необходимости
\`\`\`

---

## Примеры из реального мира

| Домен | Тип прокси | Real Subject |
|-------|------------|--------------|
| **Hibernate** | Virtual | Entity объекты (ленивая загрузка) |
| **Spring AOP** | Enhancement | Service бины (@Transactional) |
| **Java RMI** | Remote | Удалённые объекты |
| **CDI** | Protection | Бины с контролем безопасности |
| **JPA EntityManager** | Caching | Сущности базы данных |
| **Браузер** | Virtual | Большие файлы изображений |

---

## Production паттерн: Прокси контроля доступа

\`\`\`java
// Интерфейс Subject
interface Document {	// общий интерфейс
    String read();	// читать содержимое документа
    void write(String content);	// записать в документ
    void delete();	// удалить документ
}

// Real Subject - реальный документ
class RealDocument implements Document {	// реальная реализация
    private String filename;	// путь к документу
    private String content;	// содержимое документа

    public RealDocument(String filename) {	// конструктор
        this.filename = filename;	// сохраняем имя файла
        this.content = loadContent();	// загружаем из хранилища
    }

    private String loadContent() {	// симулируем загрузку
        System.out.println("Loading document: " + filename);	// логируем загрузку
        return "Content of " + filename;	// возвращаем содержимое
    }

    @Override
    public String read() {	// реализация чтения
        return content;	// возвращаем содержимое
    }

    @Override
    public void write(String content) {	// реализация записи
        this.content = content;	// обновляем содержимое
        System.out.println("Saving document: " + filename);	// логируем сохранение
    }

    @Override
    public void delete() {	// реализация удаления
        System.out.println("Deleting document: " + filename);	// логируем удаление
    }
}

// Protection Proxy - контролирует доступ на основе роли пользователя
class SecureDocumentProxy implements Document {	// прокси контроля доступа
    private String filename;	// идентификатор документа
    private RealDocument realDocument;	// ленивый реальный документ
    private User currentUser;	// пользователь, делающий запросы

    public SecureDocumentProxy(String filename, User currentUser) {	// конструктор
        this.filename = filename;	// сохраняем имя файла
        this.currentUser = currentUser;	// сохраняем пользователя
    }

    private RealDocument getDocument() {	// ленивая загрузка
        if (realDocument == null) {	// проверяем загружен ли
            realDocument = new RealDocument(filename);	// загружаем по требованию
        }
        return realDocument;	// возвращаем документ
    }

    @Override
    public String read() {	// чтение с проверкой доступа
        if (!currentUser.hasPermission("READ")) {	// проверяем разрешение
            throw new SecurityException("No read permission");	// отказ в доступе
        }
        logAccess("READ");	// логирование аудита
        return getDocument().read();	// делегируем реальному
    }

    @Override
    public void write(String content) {	// запись с проверкой доступа
        if (!currentUser.hasPermission("WRITE")) {	// проверяем разрешение
            throw new SecurityException("No write permission");	// отказ в доступе
        }
        logAccess("WRITE");	// логирование аудита
        getDocument().write(content);	// делегируем реальному
    }

    @Override
    public void delete() {	// удаление с проверкой доступа
        if (!currentUser.hasPermission("DELETE")) {	// проверяем разрешение
            throw new SecurityException("No delete permission");	// отказ в доступе
        }
        logAccess("DELETE");	// логирование аудита
        getDocument().delete();	// делегируем реальному
    }

    private void logAccess(String operation) {	// логирование аудита
        System.out.println("User " + currentUser.getName() +	// логируем пользователя
            " performed " + operation + " on " + filename);	// логируем операцию
    }
}

// Класс User для разрешений
class User {	// пользователь с разрешениями
    private String name;	// имя пользователя
    private Set<String> permissions;	// разрешения пользователя

    public User(String name, String... perms) {	// конструктор
        this.name = name;	// сохраняем имя
        this.permissions = new HashSet<>(Arrays.asList(perms));	// сохраняем разрешения
    }

    public String getName() { return name; }	// получить имя
    public boolean hasPermission(String perm) {	// проверить разрешение
        return permissions.contains(perm);	// возвращаем результат
    }
}

// Использование
User admin = new User("admin", "READ", "WRITE", "DELETE");	// админ со всеми разрешениями
User reader = new User("guest", "READ");	// гость только для чтения

Document adminDoc = new SecureDocumentProxy("secret.txt", admin);	// прокси админа
Document readerDoc = new SecureDocumentProxy("secret.txt", reader);	// прокси гостя

adminDoc.read();	// OK - админ может читать
adminDoc.write("New content");	// OK - админ может писать

readerDoc.read();	// OK - гость может читать
readerDoc.write("Hack!");	// SecurityException - нет разрешения на запись!
\`\`\`

---

## Пример кэширующего прокси

\`\`\`java
interface DataService {	// интерфейс сервиса
    String fetchData(String key);	// получить данные по ключу
}

class RealDataService implements DataService {	// реальный сервис - медленный
    @Override
    public String fetchData(String key) {	// дорогая операция
        System.out.println("Fetching from database: " + key);	// логируем запрос
        sleep(1000);	// симулируем медленную БД
        return "Data for " + key;	// возвращаем данные
    }
}

class CachingProxy implements DataService {	// кэширующий прокси
    private DataService realService;	// реальный сервис
    private Map<String, String> cache = new HashMap<>();	// хранилище кэша

    public CachingProxy(DataService realService) {	// конструктор
        this.realService = realService;	// сохраняем реальный сервис
    }

    @Override
    public String fetchData(String key) {	// запрос с кэшированием
        if (cache.containsKey(key)) {	// проверяем кэш
            System.out.println("Cache hit: " + key);	// логируем попадание в кэш
            return cache.get(key);	// возвращаем кэшированное
        }
        String data = realService.fetchData(key);	// запрос к реальному
        cache.put(key, data);	// сохраняем в кэш
        return data;	// возвращаем данные
    }
}

// Использование
DataService service = new CachingProxy(new RealDataService());	// оборачиваем кэшем

service.fetchData("user:123");	// медленно - запрос к БД
service.fetchData("user:123");	// мгновенно - из кэша!
service.fetchData("user:123");	// мгновенно - из кэша!
\`\`\`

---

## Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Разный интерфейс** | Прокси не взаимозаменяем с реальным | Реализуйте тот же интерфейс, что и real subject |
| **Жадная загрузка в прокси** | Создание реального объекта в конструкторе прокси | Создавайте только в методах при необходимости |
| **Отсутствие проверки на null** | NullPointerException на кэшированной ссылке | Всегда проверяйте перед созданием |
| **Потокобезопасность** | Race condition при ленивой инициализации | Используйте synchronized или double-checked locking |
| **Утечки памяти** | Прокси хранит ссылку вечно | Рассмотрите слабые ссылки или метод clear |`
		},
		uz: {
			title: 'Proxy (Proksi) Pattern',
			description: `Java da **Proxy patternini** amalga oshiring — boshqa ob'ektga kirishni nazorat qilish uchun o'rinbosar yoki joy ushlab turuvchi taqdim eting.

## Umumiy ko'rinish

Proxy patterni kirishni nazorat qilish, yaratish xarajatini kamaytirish yoki funksionallik qo'shish uchun boshqa ob'ektning o'rnini bosuvchini taqdim etadi. Proksi haqiqiy ob'ekt bilan bir xil interfeysga ega.

## Proksi turlari

| Tur | Maqsad | Namuna |
|-----|--------|--------|
| **Virtual Proxy** | Dangasa yuklash | Qimmat ob'ekt yaratishni kechiktirish |
| **Protection Proxy** | Kirish nazorati | Delegatsiyadan oldin ruxsatlarni tekshirish |
| **Remote Proxy** | Tarmoq aloqasi | RMI stubs |
| **Caching Proxy** | Natijalarni keshlash | Takroriy qimmat operatsiyalardan qochish |

## Vazifangiz

Rasmlar uchun dangasa yuklash virtual proksi amalga oshiring:

1. **Image** interfeysi - display() metodi bilan Subject
2. **RealImage** - Yaratilganda diskdan yuklanadigan og'ir ob'ekt
3. **ProxyImage** - RealImage ni faqat kerak bo'lganda yaratadigan yengil proksi

## Foydalanish namunasi

\`\`\`java
Image image1 = new ProxyImage("photo1.jpg");	// hali yuklanmaydi!
Image image2 = new ProxyImage("photo2.jpg");	// hali yuklanmaydi!

// Faqat haqiqatan ko'rsatilganda yuklanadi
image1.display();	// "Loading photo1.jpg" keyin "Displaying photo1.jpg"
image1.display();	// "Displaying photo1.jpg" (qayta yuklanmaydi)
\`\`\`

## Asosiy tushuncha

ProxyImage yengil — RealImage faqat birinchi kerak bo'lganda yaratiladi, resurslarni tejaydi!`,
			hint1: `**RealImage amalga oshirish**

RealImage yaratilganda yuklanadigan og'ir ob'ekt:

\`\`\`java
class RealImage implements Image {
    private String filename;

    public RealImage(String filename) {
        this.filename = filename;
        loadFromDisk();  // Darhol chaqiriladi!
    }

    private void loadFromDisk() {
        System.out.println("Loading " + filename);
    }

    @Override
    public String display() {
        return "Displaying " + filename;
    }
}
\`\`\`

Yuklash konstruktorda sodir bo'ladi — bu proksi kechiktirishga yordam beradigan narsa.`,
			hint2: `**ProxyImage amalga oshirish**

ProxyImage yaratishni haqiqiy kerak bo'lgunicha kechiktiradi:

\`\`\`java
class ProxyImage implements Image {
    private String filename;
    private RealImage realImage;  // Dastlab null

    public ProxyImage(String filename) {
        this.filename = filename;  // Shunchaki saqlash, yuklamaslik
    }

    @Override
    public String display() {
        if (realImage == null) {        // Birinchi kirish?
            realImage = new RealImage(filename);  // Hozir yaratish
        }
        return realImage.display();     // Delegatsiya
    }
}
\`\`\`

Kalit: RealImage ni faqat birinchi display() chaqiruvida yaratish!`,
			whyItMatters: `## Muammo va yechim

**Proxy siz:**
\`\`\`java
// Barcha rasmlar darhol yuklanadi - sekin ishga tushish!
Image img1 = new RealImage("huge1.jpg");	// darhol yuklanadi - 2 soniya
Image img2 = new RealImage("huge2.jpg");	// darhol yuklanadi - 2 soniya
Image img3 = new RealImage("huge3.jpg");	// darhol yuklanadi - 2 soniya
// Foydalanuvchi ko'rishdan oldin 6 soniya kutadi!	// yomon UX
\`\`\`

**Proxy bilan:**
\`\`\`java
// Rasmlar faqat ko'rsatilganda yuklanadi - darhol ishga tushish!
Image img1 = new ProxyImage("huge1.jpg");	// bir lahzada - yuklanmaydi
Image img2 = new ProxyImage("huge2.jpg");	// bir lahzada - yuklanmaydi
Image img3 = new ProxyImage("huge3.jpg");	// bir lahzada - yuklanmaydi
// Foydalanuvchi UI ni darhol ko'radi!	// ajoyib UX

img1.display();	// faqat kerak bo'lganda yuklanadi
\`\`\`

---

## Haqiqiy dunyo namunalari

| Domen | Proksi turi | Real Subject |
|-------|-------------|--------------|
| **Hibernate** | Virtual | Entity ob'ektlari (dangasa yuklash) |
| **Spring AOP** | Enhancement | Service beans (@Transactional) |
| **Java RMI** | Remote | Masofaviy ob'ektlar |
| **CDI** | Protection | Xavfsizlik nazorati bilan beanlar |
| **JPA EntityManager** | Caching | Ma'lumotlar bazasi entitylari |
| **Brauzer rasmlari** | Virtual | Katta rasm fayllari |

---

## Production pattern: Kirish nazorati proksisi

\`\`\`java
// Subject interfeysi
interface Document {	// umumiy interfeys
    String read();	// hujjat mazmunini o'qish
    void write(String content);	// hujjatga yozish
    void delete();	// hujjatni o'chirish
}

// Real Subject - haqiqiy hujjat
class RealDocument implements Document {	// haqiqiy amalga oshirish
    private String filename;	// hujjat yo'li
    private String content;	// hujjat mazmuni

    public RealDocument(String filename) {	// konstruktor
        this.filename = filename;	// fayl nomini saqlash
        this.content = loadContent();	// saqlash joyidan yuklash
    }

    private String loadContent() {	// yuklashni simulyatsiya qilish
        System.out.println("Loading document: " + filename);	// yuklashni log qilish
        return "Content of " + filename;	// mazmunni qaytarish
    }

    @Override
    public String read() {	// o'qish amalga oshirish
        return content;	// mazmunni qaytarish
    }

    @Override
    public void write(String content) {	// yozish amalga oshirish
        this.content = content;	// mazmunni yangilash
        System.out.println("Saving document: " + filename);	// saqlashni log qilish
    }

    @Override
    public void delete() {	// o'chirish amalga oshirish
        System.out.println("Deleting document: " + filename);	// o'chirishni log qilish
    }
}

// Protection Proxy - foydalanuvchi roliga asoslangan kirishni nazorat qiladi
class SecureDocumentProxy implements Document {	// kirish nazorati proksisi
    private String filename;	// hujjat identifikatori
    private RealDocument realDocument;	// dangasa yuklangan haqiqiy hujjat
    private User currentUser;	// so'rov qilayotgan foydalanuvchi

    public SecureDocumentProxy(String filename, User currentUser) {	// konstruktor
        this.filename = filename;	// fayl nomini saqlash
        this.currentUser = currentUser;	// foydalanuvchini saqlash
    }

    private RealDocument getDocument() {	// dangasa yuklash
        if (realDocument == null) {	// yuklangan mi tekshirish
            realDocument = new RealDocument(filename);	// talab bo'yicha yuklash
        }
        return realDocument;	// hujjatni qaytarish
    }

    @Override
    public String read() {	// kirish tekshiruvi bilan o'qish
        if (!currentUser.hasPermission("READ")) {	// ruxsatni tekshirish
            throw new SecurityException("No read permission");	// kirishni rad etish
        }
        logAccess("READ");	// audit loglash
        return getDocument().read();	// haqiqiyga delegatsiya
    }

    @Override
    public void write(String content) {	// kirish tekshiruvi bilan yozish
        if (!currentUser.hasPermission("WRITE")) {	// ruxsatni tekshirish
            throw new SecurityException("No write permission");	// kirishni rad etish
        }
        logAccess("WRITE");	// audit loglash
        getDocument().write(content);	// haqiqiyga delegatsiya
    }

    @Override
    public void delete() {	// kirish tekshiruvi bilan o'chirish
        if (!currentUser.hasPermission("DELETE")) {	// ruxsatni tekshirish
            throw new SecurityException("No delete permission");	// kirishni rad etish
        }
        logAccess("DELETE");	// audit loglash
        getDocument().delete();	// haqiqiyga delegatsiya
    }

    private void logAccess(String operation) {	// audit loglash
        System.out.println("User " + currentUser.getName() +	// foydalanuvchini log qilish
            " performed " + operation + " on " + filename);	// operatsiyani log qilish
    }
}

// Ruxsatlar uchun User klassi
class User {	// ruxsatlari bor foydalanuvchi
    private String name;	// foydalanuvchi nomi
    private Set<String> permissions;	// foydalanuvchi ruxsatlari

    public User(String name, String... perms) {	// konstruktor
        this.name = name;	// nomni saqlash
        this.permissions = new HashSet<>(Arrays.asList(perms));	// ruxsatlarni saqlash
    }

    public String getName() { return name; }	// nomni olish
    public boolean hasPermission(String perm) {	// ruxsatni tekshirish
        return permissions.contains(perm);	// natijani qaytarish
    }
}

// Foydalanish
User admin = new User("admin", "READ", "WRITE", "DELETE");	// barcha ruxsatlari bor admin
User reader = new User("guest", "READ");	// faqat o'qish ruxsati bor mehmon

Document adminDoc = new SecureDocumentProxy("secret.txt", admin);	// adminning proksisi
Document readerDoc = new SecureDocumentProxy("secret.txt", reader);	// mehmonning proksisi

adminDoc.read();	// OK - admin o'qiy oladi
adminDoc.write("New content");	// OK - admin yoza oladi

readerDoc.read();	// OK - mehmon o'qiy oladi
readerDoc.write("Hack!");	// SecurityException - yozish ruxsati yo'q!
\`\`\`

---

## Keshlash proksisi namunasi

\`\`\`java
interface DataService {	// xizmat interfeysi
    String fetchData(String key);	// kalit bo'yicha ma'lumot olish
}

class RealDataService implements DataService {	// haqiqiy xizmat - sekin
    @Override
    public String fetchData(String key) {	// qimmat operatsiya
        System.out.println("Fetching from database: " + key);	// so'rovni log qilish
        sleep(1000);	// sekin ma'lumotlar bazasini simulyatsiya qilish
        return "Data for " + key;	// ma'lumotni qaytarish
    }
}

class CachingProxy implements DataService {	// keshlash proksisi
    private DataService realService;	// haqiqiy xizmat
    private Map<String, String> cache = new HashMap<>();	// kesh saqlash joyi

    public CachingProxy(DataService realService) {	// konstruktor
        this.realService = realService;	// haqiqiy xizmatni saqlash
    }

    @Override
    public String fetchData(String key) {	// keshlangan so'rov
        if (cache.containsKey(key)) {	// keshni tekshirish
            System.out.println("Cache hit: " + key);	// keshga tushganini log qilish
            return cache.get(key);	// keshlanganni qaytarish
        }
        String data = realService.fetchData(key);	// haqiqiydan so'rash
        cache.put(key, data);	// keshda saqlash
        return data;	// ma'lumotni qaytarish
    }
}

// Foydalanish
DataService service = new CachingProxy(new RealDataService());	// kesh bilan o'rash

service.fetchData("user:123");	// sekin - BDdan so'rash
service.fetchData("user:123");	// bir lahzada - keshdan!
service.fetchData("user:123");	// bir lahzada - keshdan!
\`\`\`

---

## Keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Boshqa interfeys** | Proksi haqiqiy bilan almashtirib bo'lmaydi | Real subject bilan bir xil interfeysni amalga oshiring |
| **Proksida tezkor yuklash** | Proksi konstruktorida haqiqiy ob'ektni yaratish | Faqat metodlarda kerak bo'lganda yarating |
| **Null tekshiruvi yo'q** | Keshlangan havolada NullPointerException | Yaratishdan oldin doimo tekshiring |
| **Thread xavfsizligi** | Dangasa ishga tushirishda Race condition | synchronized yoki double-checked locking dan foydalaning |
| **Xotira oqishlari** | Proksi havolani abadiy ushlab turadi | Weak references yoki clear metodini ko'rib chiqing |`
		}
	}
};

export default task;
