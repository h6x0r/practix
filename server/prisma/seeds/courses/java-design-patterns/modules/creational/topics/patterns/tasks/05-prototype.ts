import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-prototype',
	title: 'Prototype Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'creational', 'prototype'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Prototype pattern in Java - create new objects by copying existing ones using Cloneable interface.

**You will implement:**

1. **Shape abstract class** implementing Cloneable
2. **Circle, Rectangle** - Concrete prototypes
3. **ShapeCache** - Registry for prototypes

**Example Usage:**

\`\`\`java
ShapeCache.loadCache();	// initialize prototype registry
Shape circle = ShapeCache.getShape("circle");	// get cloned circle
Shape anotherCircle = ShapeCache.getShape("circle");	// get another clone
boolean different = circle != anotherCircle;	// true - different objects
boolean sameType = circle.getType().equals(anotherCircle.getType());	// true - same type

Shape rect = ShapeCache.getShape("rectangle");	// get cloned rectangle
String drawing = rect.draw();	// "Drawing Rectangle 20x30"
\`\`\``,
	initialCode: `abstract class Shape implements Cloneable {
    private String id;
    protected String type;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getType() { return type; }

    @Override
    public Shape clone() {
        throw new UnsupportedOperationException("TODO");
    }

    public abstract String draw();
}

class Circle extends Shape {
    private int radius;

    public Circle() {
    }

    public Circle(int radius) {
    }

    public int getRadius() { return radius; }

    @Override
    public String draw() {
        throw new UnsupportedOperationException("TODO");
    }
}

class Rectangle extends Shape {
    private int width;
    private int height;

    public Rectangle() {
    }

    public Rectangle(int width, int height) {
    }

    @Override
    public String draw() {
        throw new UnsupportedOperationException("TODO");
    }
}

class ShapeCache {
    private static java.util.Map<String, Shape> cache = new java.util.HashMap<>();

    public static Shape getShape(String id) {
        throw new UnsupportedOperationException("TODO");
    }

    public static void loadCache() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `abstract class Shape implements Cloneable {	// prototype interface with Cloneable
    private String id;	// unique identifier
    protected String type;	// shape type

    public String getId() { return id; }	// getter for id
    public void setId(String id) { this.id = id; }	// setter for id
    public String getType() { return type; }	// getter for type

    @Override
    public Shape clone() {	// clone method - creates copy
        try {
            return (Shape) super.clone();	// delegate to Object.clone()
        } catch (CloneNotSupportedException e) {	// required by Cloneable contract
            return null;	// should never happen if Cloneable is implemented
        }
    }

    public abstract String draw();	// abstract method for subclasses
}

class Circle extends Shape {	// concrete prototype - Circle
    private int radius;	// circle-specific field

    public Circle() {	// default constructor
        type = "Circle";	// set type
    }

    public Circle(int radius) {	// constructor with radius
        this();	// call default constructor
        this.radius = radius;	// set radius
    }

    public int getRadius() { return radius; }	// getter for radius

    @Override
    public String draw() {	// implement abstract method
        return "Drawing Circle with radius " + radius;	// describe the shape
    }
}

class Rectangle extends Shape {	// concrete prototype - Rectangle
    private int width;	// rectangle width
    private int height;	// rectangle height

    public Rectangle() {	// default constructor
        type = "Rectangle";	// set type
    }

    public Rectangle(int width, int height) {	// constructor with dimensions
        this();	// call default constructor
        this.width = width;	// set width
        this.height = height;	// set height
    }

    @Override
    public String draw() {	// implement abstract method
        return "Drawing Rectangle " + width + "x" + height;	// describe the shape
    }
}

class ShapeCache {	// prototype registry
    private static java.util.Map<String, Shape> cache = new java.util.HashMap<>();	// store prototypes

    public static Shape getShape(String id) {	// get clone from registry
        Shape cachedShape = cache.get(id);	// look up prototype
        return cachedShape != null ? cachedShape.clone() : null;	// return clone, not original
    }

    public static void loadCache() {	// initialize registry with prototypes
        Circle circle = new Circle(10);	// create circle prototype
        circle.setId("circle");	// set id
        cache.put(circle.getId(), circle);	// store in registry

        Rectangle rectangle = new Rectangle(20, 30);	// create rectangle prototype
        rectangle.setId("rectangle");	// set id
        cache.put(rectangle.getId(), rectangle);	// store in registry
    }
}`,
	hint1: `**Clone Method Implementation:**

The Shape class must implement the clone() method using Object's clone():

\`\`\`java
@Override
public Shape clone() {
    try {
        return (Shape) super.clone();	// Object.clone() creates shallow copy
    } catch (CloneNotSupportedException e) {	// required by Java's Cloneable
        return null;	// won't happen since Shape implements Cloneable
    }
}
\`\`\`

Circle and Rectangle inherit this clone() and don't need to override it for shallow copying. Their draw() methods return descriptive strings:

\`\`\`java
@Override
public String draw() {
    return "Drawing Circle with radius " + radius;	// Circle
}

@Override
public String draw() {
    return "Drawing Rectangle " + width + "x" + height;	// Rectangle
}
\`\`\``,
	hint2: `**ShapeCache Implementation:**

The cache stores prototype objects and returns clones:

\`\`\`java
public static Shape getShape(String id) {
    Shape cachedShape = cache.get(id);	// get prototype from map
    return cachedShape != null ? cachedShape.clone() : null;	// return CLONE, not original
}
\`\`\`

loadCache initializes the registry with prototype instances:

\`\`\`java
public static void loadCache() {
    Circle circle = new Circle(10);	// create prototype
    circle.setId("circle");	// set identifier
    cache.put(circle.getId(), circle);	// store prototype

    Rectangle rectangle = new Rectangle(20, 30);	// create prototype
    rectangle.setId("rectangle");	// set identifier
    cache.put(rectangle.getId(), rectangle);	// store prototype
}
\`\`\``,
	whyItMatters: `## Why Prototype Exists

Prototype creates new objects by cloning existing ones, avoiding expensive construction. It's useful when object creation is costly or when you need many similar objects.

**Problem - Expensive Object Creation:**

\`\`\`java
// ❌ Bad: Creating complex objects from scratch every time
class GameWorld {
    public Monster createMonster(String type) {
        Monster monster = new Monster();	// expensive construction
        monster.loadTextures();	// load from disk
        monster.loadAnimations();	// more I/O
        monster.loadSounds();	// even more I/O
        monster.configureAI();	// complex setup
        return monster;	// took 500ms
    }
}
// Creating 100 monsters = 50 seconds!
\`\`\`

**Solution - Clone Pre-configured Prototypes:**

\`\`\`java
// ✅ Good: Clone pre-loaded prototypes
class GameWorld {
    private Map<String, Monster> prototypes = new HashMap<>();	// prototype registry

    public void preload() {	// load once at startup
        Monster goblin = new Monster();	// expensive setup once
        goblin.loadTextures();
        goblin.loadAnimations();
        prototypes.put("goblin", goblin);
    }

    public Monster createMonster(String type) {
        return prototypes.get(type).clone();	// instant clone!
    }
}
// Creating 100 monsters = milliseconds!
\`\`\`

---

## Real-World Examples

1. **Object.clone()** - Java's built-in cloning mechanism
2. **Spreadsheet copy/paste** - Copying cells with formulas and formatting
3. **Game object spawning** - Clone enemy templates with pre-loaded assets
4. **Document templates** - Clone template documents with default styling
5. **Database connection pools** - Clone pre-configured connection objects

---

## Production Pattern: Document Template System

\`\`\`java
// Prototype interface
interface DocumentPrototype extends Cloneable {	// all documents can be cloned
    DocumentPrototype clone();	// clone method
    void customize(String title, String author);	// customize after cloning
    String render();	// render document
}

// Concrete Prototypes
class ReportDocument implements DocumentPrototype {	// report template
    private String title;	// document title
    private String author;	// document author
    private String header;	// header template
    private String footer;	// footer template
    private List<String> styles;	// CSS styles
    private Map<String, Object> metadata;	// document metadata

    public ReportDocument() {	// default constructor with complex setup
        this.header = "<header>Company Logo | Report</header>";	// default header
        this.footer = "<footer>Confidential | Page {page}</footer>";	// default footer
        this.styles = new ArrayList<>(Arrays.asList(	// load default styles
            "body { font-family: Arial; }",
            ".title { font-size: 24px; }",
            ".section { margin: 20px; }"
        ));
        this.metadata = new HashMap<>();	// initialize metadata
        this.metadata.put("version", "1.0");	// default version
        this.metadata.put("department", "Engineering");	// default department
    }

    @Override
    public DocumentPrototype clone() {	// deep clone implementation
        try {
            ReportDocument clone = (ReportDocument) super.clone();	// shallow clone first
            clone.styles = new ArrayList<>(this.styles);	// deep copy styles
            clone.metadata = new HashMap<>(this.metadata);	// deep copy metadata
            return clone;	// return deep clone
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// wrap exception
        }
    }

    @Override
    public void customize(String title, String author) {	// customize clone
        this.title = title;	// set custom title
        this.author = author;	// set custom author
        this.metadata.put("created", LocalDateTime.now());	// add creation time
    }

    @Override
    public String render() {	// render document
        return String.format("%s\n<h1>%s</h1>\nBy: %s\n%s",
            header, title, author, footer);	// combine parts
    }
}

class InvoiceDocument implements DocumentPrototype {	// invoice template
    private String title;	// document title
    private String author;	// document author
    private String companyInfo;	// company information
    private String paymentTerms;	// payment terms
    private List<String> lineItems;	// invoice line items
    private BigDecimal taxRate;	// tax rate

    public InvoiceDocument() {	// default constructor
        this.companyInfo = "Acme Corp\n123 Business St\nTax ID: 12-3456789";	// company info
        this.paymentTerms = "Net 30 - Payment due within 30 days";	// default terms
        this.lineItems = new ArrayList<>();	// empty line items
        this.taxRate = new BigDecimal("0.08");	// 8% tax rate
    }

    @Override
    public DocumentPrototype clone() {	// deep clone implementation
        try {
            InvoiceDocument clone = (InvoiceDocument) super.clone();	// shallow clone
            clone.lineItems = new ArrayList<>(this.lineItems);	// deep copy line items
            return clone;	// return deep clone
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// wrap exception
        }
    }

    @Override
    public void customize(String title, String author) {	// customize clone
        this.title = "Invoice: " + title;	// prefix title
        this.author = author;	// set author/contact
    }

    public void addLineItem(String item) {	// add invoice item
        this.lineItems.add(item);	// add to list
    }

    @Override
    public String render() {	// render invoice
        return String.format("INVOICE\n%s\n\n%s\nPrepared by: %s\n\nItems:\n%s\n\n%s",
            companyInfo, title, author,
            String.join("\n", lineItems), paymentTerms);	// combine parts
    }
}

// Prototype Registry
class DocumentRegistry {	// manages document prototypes
    private static final Map<String, DocumentPrototype> prototypes = new HashMap<>();	// registry

    static {	// static initializer - load prototypes once
        prototypes.put("report", new ReportDocument());	// register report template
        prototypes.put("invoice", new InvoiceDocument());	// register invoice template
    }

    public static DocumentPrototype create(String type, String title, String author) {
        DocumentPrototype prototype = prototypes.get(type);	// get prototype
        if (prototype == null) {
            throw new IllegalArgumentException("Unknown document type: " + type);
        }
        DocumentPrototype document = prototype.clone();	// clone the prototype
        document.customize(title, author);	// customize the clone
        return document;	// return customized document
    }

    public static void registerPrototype(String type, DocumentPrototype prototype) {
        prototypes.put(type, prototype);	// add new prototype to registry
    }
}

// Usage
class DocumentService {	// uses the prototype system
    public DocumentPrototype createQuarterlyReport(String quarter, String analyst) {
        return DocumentRegistry.create("report", 	// create from report template
            "Q" + quarter + " Financial Report", analyst);	// customize title and author
    }

    public InvoiceDocument createInvoice(String customer, String[] items) {
        InvoiceDocument invoice = (InvoiceDocument) DocumentRegistry.create(
            "invoice", customer, "Sales Team");	// create from invoice template
        for (String item : items) {	// add line items
            invoice.addLineItem(item);	// add each item
        }
        return invoice;	// return customized invoice
    }
}
\`\`\`

---

## Shallow vs Deep Copy

\`\`\`java
// Shallow Copy - Object.clone() default
class ShallowExample implements Cloneable {
    private int[] data = {1, 2, 3};	// array field

    public Object clone() throws CloneNotSupportedException {
        return super.clone();	// shallow copy - data array shared!
    }
}

// Deep Copy - manually copy nested objects
class DeepExample implements Cloneable {
    private int[] data = {1, 2, 3};	// array field

    public Object clone() throws CloneNotSupportedException {
        DeepExample clone = (DeepExample) super.clone();	// shallow first
        clone.data = this.data.clone();	// deep copy the array
        return clone;	// now data is independent
    }
}
\`\`\`

---

## Common Mistakes to Avoid

1. **Forgetting deep copy** - Shallow clone shares mutable nested objects
2. **Not implementing Cloneable** - Will throw CloneNotSupportedException
3. **Returning original instead of clone** - Registry must return clones
4. **Not handling circular references** - Deep copy can cause infinite loops
5. **Ignoring final fields** - Final fields can't be modified after clone`,
	order: 4,
	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void shapeCloneCreatesCopy() {
        Circle original = new Circle(10);
        Shape clone = original.clone();
        assertNotSame(original, clone, "Clone should be different object");
    }
}

class Test2 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void circleDrawReturnsDescription() {
        Circle circle = new Circle(10);
        String result = circle.draw();
        assertTrue(result.contains("Circle"), "Should mention Circle");
        assertTrue(result.contains("10"), "Should include radius");
    }
}

class Test3 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void rectangleDrawReturnsDescription() {
        Rectangle rect = new Rectangle(20, 30);
        String result = rect.draw();
        assertTrue(result.contains("Rectangle"), "Should mention Rectangle");
    }
}

class Test4 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void shapeCacheReturnsCircle() {
        Shape shape = ShapeCache.getShape("circle");
        assertNotNull(shape, "Should return circle");
        assertEquals("Circle", shape.getType(), "Type should be Circle");
    }
}

class Test5 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void shapeCacheReturnsRectangle() {
        Shape shape = ShapeCache.getShape("rectangle");
        assertNotNull(shape, "Should return rectangle");
        assertEquals("Rectangle", shape.getType(), "Type should be Rectangle");
    }
}

class Test6 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void shapeCacheReturnsDifferentClones() {
        Shape s1 = ShapeCache.getShape("circle");
        Shape s2 = ShapeCache.getShape("circle");
        assertNotSame(s1, s2, "Should return different clone objects");
    }
}

class Test7 {
    @BeforeEach
    void setup() {
        ShapeCache.loadCache();
    }

    @Test
    void clonedCircleHasSameType() {
        Shape s1 = ShapeCache.getShape("circle");
        Shape s2 = ShapeCache.getShape("circle");
        assertEquals(s1.getType(), s2.getType(), "Clones should have same type");
    }
}

class Test8 {
    @Test
    void circleHasRadius() {
        Circle circle = new Circle(15);
        assertEquals(15, circle.getRadius(), "Radius should be 15");
    }
}

class Test9 {
    @Test
    void circleTypeIsCircle() {
        Circle circle = new Circle();
        assertEquals("Circle", circle.getType(), "Type should be Circle");
    }
}

class Test10 {
    @Test
    void rectangleTypeIsRectangle() {
        Rectangle rect = new Rectangle();
        assertEquals("Rectangle", rect.getType(), "Type should be Rectangle");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Prototype (Прототип)',
			description: `Реализуйте паттерн Prototype на Java — создавайте новые объекты копированием существующих через Cloneable.

**Вы реализуете:**

1. **Абстрактный класс Shape** реализующий Cloneable
2. **Circle, Rectangle** - Конкретные прототипы
3. **ShapeCache** - Реестр прототипов

**Пример использования:**

\`\`\`java
ShapeCache.loadCache();	// инициализируем реестр прототипов
Shape circle = ShapeCache.getShape("circle");	// получаем клон круга
Shape anotherCircle = ShapeCache.getShape("circle");	// получаем ещё один клон
boolean different = circle != anotherCircle;	// true - разные объекты
boolean sameType = circle.getType().equals(anotherCircle.getType());	// true - одинаковый тип

Shape rect = ShapeCache.getShape("rectangle");	// получаем клон прямоугольника
String drawing = rect.draw();	// "Drawing Rectangle 20x30"
\`\`\``,
			hint1: `**Реализация метода Clone:**

Класс Shape должен реализовать метод clone() используя clone() из Object:

\`\`\`java
@Override
public Shape clone() {
    try {
        return (Shape) super.clone();	// Object.clone() создаёт поверхностную копию
    } catch (CloneNotSupportedException e) {	// требуется Java Cloneable
        return null;	// не случится, т.к. Shape реализует Cloneable
    }
}
\`\`\`

Circle и Rectangle наследуют этот clone() и не нуждаются в переопределении для поверхностного копирования. Их методы draw() возвращают описательные строки:

\`\`\`java
@Override
public String draw() {
    return "Drawing Circle with radius " + radius;	// Circle
}

@Override
public String draw() {
    return "Drawing Rectangle " + width + "x" + height;	// Rectangle
}
\`\`\``,
			hint2: `**Реализация ShapeCache:**

Кэш хранит объекты-прототипы и возвращает их клоны:

\`\`\`java
public static Shape getShape(String id) {
    Shape cachedShape = cache.get(id);	// получаем прототип из map
    return cachedShape != null ? cachedShape.clone() : null;	// возвращаем КЛОН, не оригинал
}
\`\`\`

loadCache инициализирует реестр экземплярами прототипов:

\`\`\`java
public static void loadCache() {
    Circle circle = new Circle(10);	// создаём прототип
    circle.setId("circle");	// устанавливаем идентификатор
    cache.put(circle.getId(), circle);	// сохраняем прототип

    Rectangle rectangle = new Rectangle(20, 30);	// создаём прототип
    rectangle.setId("rectangle");	// устанавливаем идентификатор
    cache.put(rectangle.getId(), rectangle);	// сохраняем прототип
}
\`\`\``,
			whyItMatters: `## Зачем нужен Prototype

Prototype создаёт новые объекты клонированием существующих, избегая дорогостоящего конструирования. Он полезен когда создание объекта затратно или когда нужно много похожих объектов.

**Проблема - Дорогое создание объектов:**

\`\`\`java
// ❌ Плохо: Создание сложных объектов с нуля каждый раз
class GameWorld {
    public Monster createMonster(String type) {
        Monster monster = new Monster();	// дорогое конструирование
        monster.loadTextures();	// загрузка с диска
        monster.loadAnimations();	// ещё I/O
        monster.loadSounds();	// и ещё I/O
        monster.configureAI();	// сложная настройка
        return monster;	// заняло 500мс
    }
}
// Создание 100 монстров = 50 секунд!
\`\`\`

**Решение - Клонирование предварительно настроенных прототипов:**

\`\`\`java
// ✅ Хорошо: Клонирование предзагруженных прототипов
class GameWorld {
    private Map<String, Monster> prototypes = new HashMap<>();	// реестр прототипов

    public void preload() {	// загружаем один раз при старте
        Monster goblin = new Monster();	// дорогая настройка один раз
        goblin.loadTextures();
        goblin.loadAnimations();
        prototypes.put("goblin", goblin);
    }

    public Monster createMonster(String type) {
        return prototypes.get(type).clone();	// мгновенный клон!
    }
}
// Создание 100 монстров = миллисекунды!
\`\`\`

---

## Примеры из реального мира

1. **Object.clone()** - Встроенный механизм клонирования Java
2. **Копирование ячеек в таблицах** - Копирование ячеек с формулами и форматированием
3. **Спавн игровых объектов** - Клонирование шаблонов врагов с предзагруженными ресурсами
4. **Шаблоны документов** - Клонирование шаблонов документов со стилями по умолчанию
5. **Пулы соединений с БД** - Клонирование предварительно настроенных объектов соединений

---

## Production паттерн: Система шаблонов документов

\`\`\`java
// Интерфейс прототипа
interface DocumentPrototype extends Cloneable {	// все документы можно клонировать
    DocumentPrototype clone();	// метод клонирования
    void customize(String title, String author);	// настройка после клонирования
    String render();	// рендеринг документа
}

// Конкретные прототипы
class ReportDocument implements DocumentPrototype {	// шаблон отчёта
    private String title;	// заголовок документа
    private String author;	// автор документа
    private String header;	// шаблон заголовка
    private String footer;	// шаблон подвала
    private List<String> styles;	// CSS стили
    private Map<String, Object> metadata;	// метаданные документа

    public ReportDocument() {	// конструктор по умолчанию со сложной настройкой
        this.header = "<header>Company Logo | Report</header>";	// заголовок по умолчанию
        this.footer = "<footer>Confidential | Page {page}</footer>";	// подвал по умолчанию
        this.styles = new ArrayList<>(Arrays.asList(	// загрузка стилей по умолчанию
            "body { font-family: Arial; }",
            ".title { font-size: 24px; }",
            ".section { margin: 20px; }"
        ));
        this.metadata = new HashMap<>();	// инициализация метаданных
        this.metadata.put("version", "1.0");	// версия по умолчанию
        this.metadata.put("department", "Engineering");	// отдел по умолчанию
    }

    @Override
    public DocumentPrototype clone() {	// реализация глубокого клонирования
        try {
            ReportDocument clone = (ReportDocument) super.clone();	// сначала поверхностная копия
            clone.styles = new ArrayList<>(this.styles);	// глубокое копирование стилей
            clone.metadata = new HashMap<>(this.metadata);	// глубокое копирование метаданных
            return clone;	// возвращаем глубокую копию
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// оборачиваем исключение
        }
    }

    @Override
    public void customize(String title, String author) {	// настройка клона
        this.title = title;	// устанавливаем заголовок
        this.author = author;	// устанавливаем автора
        this.metadata.put("created", LocalDateTime.now());	// добавляем время создания
    }

    @Override
    public String render() {	// рендеринг документа
        return String.format("%s\n<h1>%s</h1>\nBy: %s\n%s",
            header, title, author, footer);	// объединяем части
    }
}

class InvoiceDocument implements DocumentPrototype {	// шаблон счёта
    private String title;	// заголовок документа
    private String author;	// автор документа
    private String companyInfo;	// информация о компании
    private String paymentTerms;	// условия оплаты
    private List<String> lineItems;	// позиции счёта
    private BigDecimal taxRate;	// ставка налога

    public InvoiceDocument() {	// конструктор по умолчанию
        this.companyInfo = "Acme Corp\n123 Business St\nTax ID: 12-3456789";	// инфо о компании
        this.paymentTerms = "Net 30 - Payment due within 30 days";	// условия по умолчанию
        this.lineItems = new ArrayList<>();	// пустые позиции
        this.taxRate = new BigDecimal("0.08");	// 8% налог
    }

    @Override
    public DocumentPrototype clone() {	// реализация глубокого клонирования
        try {
            InvoiceDocument clone = (InvoiceDocument) super.clone();	// поверхностная копия
            clone.lineItems = new ArrayList<>(this.lineItems);	// глубокое копирование позиций
            return clone;	// возвращаем глубокую копию
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// оборачиваем исключение
        }
    }

    @Override
    public void customize(String title, String author) {	// настройка клона
        this.title = "Invoice: " + title;	// добавляем префикс
        this.author = author;	// устанавливаем автора/контакт
    }

    public void addLineItem(String item) {	// добавить позицию счёта
        this.lineItems.add(item);	// добавляем в список
    }

    @Override
    public String render() {	// рендеринг счёта
        return String.format("INVOICE\n%s\n\n%s\nPrepared by: %s\n\nItems:\n%s\n\n%s",
            companyInfo, title, author,
            String.join("\n", lineItems), paymentTerms);	// объединяем части
    }
}

// Реестр прототипов
class DocumentRegistry {	// управляет прототипами документов
    private static final Map<String, DocumentPrototype> prototypes = new HashMap<>();	// реестр

    static {	// статический инициализатор - загружаем прототипы один раз
        prototypes.put("report", new ReportDocument());	// регистрируем шаблон отчёта
        prototypes.put("invoice", new InvoiceDocument());	// регистрируем шаблон счёта
    }

    public static DocumentPrototype create(String type, String title, String author) {
        DocumentPrototype prototype = prototypes.get(type);	// получаем прототип
        if (prototype == null) {
            throw new IllegalArgumentException("Unknown document type: " + type);
        }
        DocumentPrototype document = prototype.clone();	// клонируем прототип
        document.customize(title, author);	// настраиваем клон
        return document;	// возвращаем настроенный документ
    }

    public static void registerPrototype(String type, DocumentPrototype prototype) {
        prototypes.put(type, prototype);	// добавляем новый прототип в реестр
    }
}

// Использование
class DocumentService {	// использует систему прототипов
    public DocumentPrototype createQuarterlyReport(String quarter, String analyst) {
        return DocumentRegistry.create("report", 	// создаём из шаблона отчёта
            "Q" + quarter + " Financial Report", analyst);	// настраиваем заголовок и автора
    }

    public InvoiceDocument createInvoice(String customer, String[] items) {
        InvoiceDocument invoice = (InvoiceDocument) DocumentRegistry.create(
            "invoice", customer, "Sales Team");	// создаём из шаблона счёта
        for (String item : items) {	// добавляем позиции
            invoice.addLineItem(item);	// добавляем каждую позицию
        }
        return invoice;	// возвращаем настроенный счёт
    }
}
\`\`\`

---

## Поверхностное vs Глубокое копирование

\`\`\`java
// Поверхностное копирование - Object.clone() по умолчанию
class ShallowExample implements Cloneable {
    private int[] data = {1, 2, 3};	// поле-массив

    public Object clone() throws CloneNotSupportedException {
        return super.clone();	// поверхностная копия - массив data общий!
    }
}

// Глубокое копирование - вручную копируем вложенные объекты
class DeepExample implements Cloneable {
    private int[] data = {1, 2, 3};	// поле-массив

    public Object clone() throws CloneNotSupportedException {
        DeepExample clone = (DeepExample) super.clone();	// сначала поверхностная
        clone.data = this.data.clone();	// глубокое копирование массива
        return clone;	// теперь data независим
    }
}
\`\`\`

---

## Частые ошибки, которых следует избегать

1. **Забывать глубокое копирование** - Поверхностный клон разделяет изменяемые вложенные объекты
2. **Не реализовывать Cloneable** - Бросит CloneNotSupportedException
3. **Возвращать оригинал вместо клона** - Реестр должен возвращать клоны
4. **Не обрабатывать циклические ссылки** - Глубокое копирование может вызвать бесконечный цикл
5. **Игнорировать final поля** - Final поля нельзя изменить после клонирования`
		},
		uz: {
			title: 'Prototype (Prototip) Pattern',
			description: `Java da Prototype patternini amalga oshiring — Cloneable orqali mavjud ob'ektlarni nusxalash orqali yangilarini yarating.

**Siz amalga oshirasiz:**

1. **Shape abstrakt klassi** Cloneable ni implement qilgan
2. **Circle, Rectangle** - Konkret prototiplar
3. **ShapeCache** - Prototiplar reyestri

**Foydalanish misoli:**

\`\`\`java
ShapeCache.loadCache();	// prototiplar reyestrini initsializatsiya qilamiz
Shape circle = ShapeCache.getShape("circle");	// aylana klonini olamiz
Shape anotherCircle = ShapeCache.getShape("circle");	// yana bir klon olamiz
boolean different = circle != anotherCircle;	// true - turli ob'ektlar
boolean sameType = circle.getType().equals(anotherCircle.getType());	// true - bir xil tur

Shape rect = ShapeCache.getShape("rectangle");	// to'rtburchak klonini olamiz
String drawing = rect.draw();	// "Drawing Rectangle 20x30"
\`\`\``,
			hint1: `**Clone metodi amalga oshirish:**

Shape klassi clone() metodini Object'ning clone() dan foydalanib amalga oshirishi kerak:

\`\`\`java
@Override
public Shape clone() {
    try {
        return (Shape) super.clone();	// Object.clone() sayoz nusxa yaratadi
    } catch (CloneNotSupportedException e) {	// Java Cloneable talab qiladi
        return null;	// bo'lmaydi, chunki Shape Cloneable ni implement qilgan
    }
}
\`\`\`

Circle va Rectangle bu clone() ni meros oladi va sayoz nusxalash uchun qayta aniqlamasligi mumkin. Ularning draw() metodlari tavsiflovchi satrlar qaytaradi:

\`\`\`java
@Override
public String draw() {
    return "Drawing Circle with radius " + radius;	// Circle
}

@Override
public String draw() {
    return "Drawing Rectangle " + width + "x" + height;	// Rectangle
}
\`\`\``,
			hint2: `**ShapeCache amalga oshirish:**

Kesh prototip ob'ektlarni saqlaydi va ularning klonlarini qaytaradi:

\`\`\`java
public static Shape getShape(String id) {
    Shape cachedShape = cache.get(id);	// map dan prototipni olamiz
    return cachedShape != null ? cachedShape.clone() : null;	// KLON qaytaramiz, asl emas
}
\`\`\`

loadCache reyestrni prototip instansiyalari bilan initsializatsiya qiladi:

\`\`\`java
public static void loadCache() {
    Circle circle = new Circle(10);	// prototip yaratamiz
    circle.setId("circle");	// identifikatorni o'rnatamiz
    cache.put(circle.getId(), circle);	// prototipni saqlaymiz

    Rectangle rectangle = new Rectangle(20, 30);	// prototip yaratamiz
    rectangle.setId("rectangle");	// identifikatorni o'rnatamiz
    cache.put(rectangle.getId(), rectangle);	// prototipni saqlaymiz
}
\`\`\``,
			whyItMatters: `## Prototype nima uchun kerak

Prototype yangi ob'ektlarni mavjudlarini klonlash orqali yaratadi, qimmat konstruksiyadan qochadi. U ob'ekt yaratish qimmat bo'lganda yoki ko'p o'xshash ob'ektlar kerak bo'lganda foydali.

**Muammo - Qimmat ob'ekt yaratish:**

\`\`\`java
// ❌ Yomon: Har safar noldan murakkab ob'ektlar yaratish
class GameWorld {
    public Monster createMonster(String type) {
        Monster monster = new Monster();	// qimmat konstruksiya
        monster.loadTextures();	// diskdan yuklash
        monster.loadAnimations();	// yana I/O
        monster.loadSounds();	// va yana I/O
        monster.configureAI();	// murakkab sozlash
        return monster;	// 500ms vaqt oldi
    }
}
// 100 ta monster yaratish = 50 sekund!
\`\`\`

**Yechim - Oldindan sozlangan prototiplarni klonlash:**

\`\`\`java
// ✅ Yaxshi: Oldindan yuklangan prototiplarni klonlash
class GameWorld {
    private Map<String, Monster> prototypes = new HashMap<>();	// prototiplar reyestri

    public void preload() {	// ishga tushirishda bir marta yuklaymiz
        Monster goblin = new Monster();	// qimmat sozlash bir marta
        goblin.loadTextures();
        goblin.loadAnimations();
        prototypes.put("goblin", goblin);
    }

    public Monster createMonster(String type) {
        return prototypes.get(type).clone();	// tezkor klon!
    }
}
// 100 ta monster yaratish = millisekundlar!
\`\`\`

---

## Haqiqiy dunyo misollari

1. **Object.clone()** - Java ning o'rnatilgan klonlash mexanizmi
2. **Jadval hujayralarini nusxalash** - Formulalar va formatlash bilan hujayralarni nusxalash
3. **O'yin ob'ektlarini spawn qilish** - Oldindan yuklangan resurslar bilan dushman shablonlarini klonlash
4. **Hujjat shablonlari** - Standart stillar bilan shablon hujjatlarni klonlash
5. **Ma'lumotlar bazasi ulanish pullari** - Oldindan sozlangan ulanish ob'ektlarini klonlash

---

## Production pattern: Hujjat shablonlari tizimi

\`\`\`java
// Prototip interfeysi
interface DocumentPrototype extends Cloneable {	// barcha hujjatlar klonlanishi mumkin
    DocumentPrototype clone();	// klonlash metodi
    void customize(String title, String author);	// klonlashdan keyin sozlash
    String render();	// hujjatni render qilish
}

// Konkret prototiplar
class ReportDocument implements DocumentPrototype {	// hisobot shabloni
    private String title;	// hujjat sarlavhasi
    private String author;	// hujjat muallifi
    private String header;	// sarlavha shabloni
    private String footer;	// pastki qism shabloni
    private List<String> styles;	// CSS stillar
    private Map<String, Object> metadata;	// hujjat metadata

    public ReportDocument() {	// murakkab sozlash bilan standart konstruktor
        this.header = "<header>Company Logo | Report</header>";	// standart sarlavha
        this.footer = "<footer>Confidential | Page {page}</footer>";	// standart pastki qism
        this.styles = new ArrayList<>(Arrays.asList(	// standart stillarni yuklash
            "body { font-family: Arial; }",
            ".title { font-size: 24px; }",
            ".section { margin: 20px; }"
        ));
        this.metadata = new HashMap<>();	// metadata initsializatsiya
        this.metadata.put("version", "1.0");	// standart versiya
        this.metadata.put("department", "Engineering");	// standart bo'lim
    }

    @Override
    public DocumentPrototype clone() {	// chuqur klonlash amalga oshirish
        try {
            ReportDocument clone = (ReportDocument) super.clone();	// avval sayoz nusxa
            clone.styles = new ArrayList<>(this.styles);	// stillarni chuqur nusxalash
            clone.metadata = new HashMap<>(this.metadata);	// metadata ni chuqur nusxalash
            return clone;	// chuqur nusxani qaytaramiz
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// istisno o'rash
        }
    }

    @Override
    public void customize(String title, String author) {	// klonni sozlash
        this.title = title;	// sarlavhani o'rnatamiz
        this.author = author;	// muallifni o'rnatamiz
        this.metadata.put("created", LocalDateTime.now());	// yaratilish vaqtini qo'shamiz
    }

    @Override
    public String render() {	// hujjatni render qilish
        return String.format("%s\n<h1>%s</h1>\nBy: %s\n%s",
            header, title, author, footer);	// qismlarni birlashtiramiz
    }
}

class InvoiceDocument implements DocumentPrototype {	// hisob-faktura shabloni
    private String title;	// hujjat sarlavhasi
    private String author;	// hujjat muallifi
    private String companyInfo;	// kompaniya ma'lumotlari
    private String paymentTerms;	// to'lov shartlari
    private List<String> lineItems;	// hisob-faktura qatorlari
    private BigDecimal taxRate;	// soliq stavkasi

    public InvoiceDocument() {	// standart konstruktor
        this.companyInfo = "Acme Corp\n123 Business St\nTax ID: 12-3456789";	// kompaniya info
        this.paymentTerms = "Net 30 - Payment due within 30 days";	// standart shartlar
        this.lineItems = new ArrayList<>();	// bo'sh qatorlar
        this.taxRate = new BigDecimal("0.08");	// 8% soliq
    }

    @Override
    public DocumentPrototype clone() {	// chuqur klonlash amalga oshirish
        try {
            InvoiceDocument clone = (InvoiceDocument) super.clone();	// sayoz nusxa
            clone.lineItems = new ArrayList<>(this.lineItems);	// qatorlarni chuqur nusxalash
            return clone;	// chuqur nusxani qaytaramiz
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);	// istisno o'rash
        }
    }

    @Override
    public void customize(String title, String author) {	// klonni sozlash
        this.title = "Invoice: " + title;	// prefiks qo'shamiz
        this.author = author;	// muallif/kontaktni o'rnatamiz
    }

    public void addLineItem(String item) {	// hisob-faktura qatori qo'shish
        this.lineItems.add(item);	// ro'yxatga qo'shamiz
    }

    @Override
    public String render() {	// hisob-fakturani render qilish
        return String.format("INVOICE\n%s\n\n%s\nPrepared by: %s\n\nItems:\n%s\n\n%s",
            companyInfo, title, author,
            String.join("\n", lineItems), paymentTerms);	// qismlarni birlashtiramiz
    }
}

// Prototiplar reyestri
class DocumentRegistry {	// hujjat prototiplarini boshqaradi
    private static final Map<String, DocumentPrototype> prototypes = new HashMap<>();	// reyestr

    static {	// statik initsializator - prototiplarni bir marta yuklaymiz
        prototypes.put("report", new ReportDocument());	// hisobot shablonini ro'yxatga olamiz
        prototypes.put("invoice", new InvoiceDocument());	// hisob-faktura shablonini ro'yxatga olamiz
    }

    public static DocumentPrototype create(String type, String title, String author) {
        DocumentPrototype prototype = prototypes.get(type);	// prototipni olamiz
        if (prototype == null) {
            throw new IllegalArgumentException("Unknown document type: " + type);
        }
        DocumentPrototype document = prototype.clone();	// prototipni klonlaymiz
        document.customize(title, author);	// klonni sozlaymiz
        return document;	// sozlangan hujjatni qaytaramiz
    }

    public static void registerPrototype(String type, DocumentPrototype prototype) {
        prototypes.put(type, prototype);	// reyestrga yangi prototip qo'shamiz
    }
}

// Foydalanish
class DocumentService {	// prototip tizimidan foydalanadi
    public DocumentPrototype createQuarterlyReport(String quarter, String analyst) {
        return DocumentRegistry.create("report", 	// hisobot shablonidan yaratamiz
            "Q" + quarter + " Financial Report", analyst);	// sarlavha va muallifni sozlaymiz
    }

    public InvoiceDocument createInvoice(String customer, String[] items) {
        InvoiceDocument invoice = (InvoiceDocument) DocumentRegistry.create(
            "invoice", customer, "Sales Team");	// hisob-faktura shablonidan yaratamiz
        for (String item : items) {	// qatorlarni qo'shamiz
            invoice.addLineItem(item);	// har bir qatorni qo'shamiz
        }
        return invoice;	// sozlangan hisob-fakturani qaytaramiz
    }
}
\`\`\`

---

## Sayoz vs Chuqur nusxalash

\`\`\`java
// Sayoz nusxalash - Object.clone() standart
class ShallowExample implements Cloneable {
    private int[] data = {1, 2, 3};	// massiv maydon

    public Object clone() throws CloneNotSupportedException {
        return super.clone();	// sayoz nusxa - data massivi umumiy!
    }
}

// Chuqur nusxalash - ichki ob'ektlarni qo'lda nusxalash
class DeepExample implements Cloneable {
    private int[] data = {1, 2, 3};	// massiv maydon

    public Object clone() throws CloneNotSupportedException {
        DeepExample clone = (DeepExample) super.clone();	// avval sayoz
        clone.data = this.data.clone();	// massivni chuqur nusxalash
        return clone;	// endi data mustaqil
    }
}
\`\`\`

---

## Qochish kerak bo'lgan keng tarqalgan xatolar

1. **Chuqur nusxalashni unutish** - Sayoz klon o'zgaruvchan ichki ob'ektlarni ulashadi
2. **Cloneable ni implement qilmaslik** - CloneNotSupportedException tashlaydi
3. **Klon o'rniga aslni qaytarish** - Reyestr klonlarni qaytarishi kerak
4. **Siklik havolalarni boshqarmaslik** - Chuqur nusxalash cheksiz tsiklga olib kelishi mumkin
5. **Final maydonlarni e'tiborsiz qoldirish** - Final maydonlarni klonlashdan keyin o'zgartirib bo'lmaydi`
		}
	}
};

export default task;
