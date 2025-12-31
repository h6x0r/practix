import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-builder',
	title: 'Builder Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'creational', 'builder'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Builder pattern in Java - separate construction of complex objects using fluent interface.

**You will implement:**

1. **House class** with nested Builder
2. **Fluent interface** - each setter returns Builder
3. **build() method** - returns constructed House

**Example Usage:**

\`\`\`java
House house = new House.Builder()	// create builder instance
    .foundation("concrete")	// set foundation type
    .walls("brick")	// set wall material
    .roof("tile")	// set roof type
    .garage(true)	// include garage
    .swimmingPool(false)	// no swimming pool
    .build();	// construct the House object

// Access built object
String walls = house.getWalls();	// "brick"
boolean hasGarage = house.hasGarage();	// true
\`\`\``,
	initialCode: `class House {
    private final String foundation;
    private final String walls;
    private final String roof;
    private final boolean hasGarage;
    private final boolean hasSwimmingPool;

    private House(Builder builder) {
    }

    public String getFoundation() { return foundation; }
    public String getWalls() { return walls; }
    public String getRoof() { return roof; }
    public boolean hasGarage() { return hasGarage; }
    public boolean hasSwimmingPool() { return hasSwimmingPool; }

    public static class Builder {
        private String foundation = "";
        private String walls = "";
        private String roof = "";
        private boolean hasGarage = false;
        private boolean hasSwimmingPool = false;

        public Builder foundation(String foundation) {
            throw new UnsupportedOperationException("TODO");
        }

        public Builder walls(String walls) {
            throw new UnsupportedOperationException("TODO");
        }

        public Builder roof(String roof) {
            throw new UnsupportedOperationException("TODO");
        }

        public Builder garage(boolean hasGarage) {
            throw new UnsupportedOperationException("TODO");
        }

        public Builder swimmingPool(boolean hasPool) {
            throw new UnsupportedOperationException("TODO");
        }

        public House build() {
            throw new UnsupportedOperationException("TODO");
        }
    }
}`,
	solutionCode: `class House {	// immutable product class
    private final String foundation;	// final fields for immutability
    private final String walls;	// wall material
    private final String roof;	// roof type
    private final boolean hasGarage;	// optional feature
    private final boolean hasSwimmingPool;	// optional feature

    private House(Builder builder) {	// private constructor - only builder can create
        this.foundation = builder.foundation;	// copy from builder
        this.walls = builder.walls;	// copy from builder
        this.roof = builder.roof;	// copy from builder
        this.hasGarage = builder.hasGarage;	// copy from builder
        this.hasSwimmingPool = builder.hasSwimmingPool;	// copy from builder
    }

    public String getFoundation() { return foundation; }	// getter for foundation
    public String getWalls() { return walls; }	// getter for walls
    public String getRoof() { return roof; }	// getter for roof
    public boolean hasGarage() { return hasGarage; }	// getter for garage flag
    public boolean hasSwimmingPool() { return hasSwimmingPool; }	// getter for pool flag

    public static class Builder {	// static nested builder class
        private String foundation = "";	// default value
        private String walls = "";	// default value
        private String roof = "";	// default value
        private boolean hasGarage = false;	// default: no garage
        private boolean hasSwimmingPool = false;	// default: no pool

        public Builder foundation(String foundation) {	// fluent setter
            this.foundation = foundation;	// set the value
            return this;	// return this for chaining
        }

        public Builder walls(String walls) {	// fluent setter
            this.walls = walls;	// set the value
            return this;	// return this for chaining
        }

        public Builder roof(String roof) {	// fluent setter
            this.roof = roof;	// set the value
            return this;	// return this for chaining
        }

        public Builder garage(boolean hasGarage) {	// fluent setter
            this.hasGarage = hasGarage;	// set the value
            return this;	// return this for chaining
        }

        public Builder swimmingPool(boolean hasPool) {	// fluent setter
            this.hasSwimmingPool = hasPool;	// set the value
            return this;	// return this for chaining
        }

        public House build() {	// terminal operation
            return new House(this);	// create immutable House from builder state
        }
    }
}`,
	hint1: `**Fluent Setter Pattern:**

Each setter method in the Builder follows the same pattern:

\`\`\`java
public Builder foundation(String foundation) {
    this.foundation = foundation;	// assign value to field
    return this;	// return this for method chaining
}

public Builder walls(String walls) {
    this.walls = walls;	// assign value to field
    return this;	// return this for method chaining
}
\`\`\`

The key is returning \`this\` to enable chaining:
\`\`\`java
new Builder().foundation("concrete").walls("brick")...	// chained calls
\`\`\``,
	hint2: `**Build Method:**

The build() method creates the final immutable object:

\`\`\`java
public House build() {
    return new House(this);	// pass builder to House constructor
}
\`\`\`

The House constructor is private and only accepts a Builder:

\`\`\`java
private House(Builder builder) {	// private - only builder can call
    this.foundation = builder.foundation;	// copy all values
    this.walls = builder.walls;	// from builder to final fields
    this.roof = builder.roof;
    this.hasGarage = builder.hasGarage;
    this.hasSwimmingPool = builder.hasSwimmingPool;
}
\`\`\``,
	whyItMatters: `## Why Builder Exists

Builder solves the "telescoping constructor" problem - when a class has many optional parameters. It creates immutable objects step-by-step with clear, readable code.

**Problem - Telescoping Constructors:**

\`\`\`java
// ❌ Bad: Multiple constructor overloads
class House {
    public House(String foundation) {...}
    public House(String foundation, String walls) {...}
    public House(String foundation, String walls, String roof) {...}
    public House(String foundation, String walls, String roof, boolean garage) {...}
    // Gets worse with more parameters!
}

// Confusing to use - what does 'true' mean?
House h = new House("concrete", "brick", "tile", true, false, true);
\`\`\`

**Solution - Builder with Named Parameters:**

\`\`\`java
// ✅ Good: Builder with fluent interface
House house = new House.Builder()	// clear, readable construction
    .foundation("concrete")	// named parameter
    .walls("brick")	// named parameter
    .roof("tile")	// named parameter
    .garage(true)	// obvious what this means
    .build();	// immutable result
\`\`\`

---

## Real-World Examples

1. **StringBuilder** - Builds strings efficiently
2. **Stream.Builder** - Creates streams incrementally
3. **ProcessBuilder** - Configures external processes
4. **Lombok @Builder** - Generates builders automatically
5. **Protocol Buffers** - Message builders in serialization
6. **HttpRequest.Builder** - Java 11 HTTP client

---

## Production Pattern: HTTP Request Builder

\`\`\`java
class HttpRequest {	// immutable HTTP request
    private final String method;	// GET, POST, etc.
    private final String url;	// request URL
    private final Map<String, String> headers;	// HTTP headers
    private final String body;	// request body
    private final int timeout;	// timeout in milliseconds
    private final boolean followRedirects;	// follow 3xx redirects

    private HttpRequest(Builder builder) {	// private constructor
        this.method = builder.method;	// copy from builder
        this.url = builder.url;	// copy from builder
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));	// defensive copy
        this.body = builder.body;	// copy from builder
        this.timeout = builder.timeout;	// copy from builder
        this.followRedirects = builder.followRedirects;	// copy from builder
    }

    // Getters
    public String getMethod() { return method; }	// HTTP method
    public String getUrl() { return url; }	// request URL
    public Map<String, String> getHeaders() { return headers; }	// immutable headers
    public String getBody() { return body; }	// request body
    public int getTimeout() { return timeout; }	// timeout
    public boolean isFollowRedirects() { return followRedirects; }	// redirect policy

    public static class Builder {	// builder class
        private String method = "GET";	// default method
        private String url;	// required field
        private final Map<String, String> headers = new HashMap<>();	// mutable during build
        private String body = "";	// default empty body
        private int timeout = 30000;	// default 30 seconds
        private boolean followRedirects = true;	// default follow redirects

        public Builder(String url) {	// required parameter in constructor
            this.url = Objects.requireNonNull(url, "URL is required");	// validate required field
        }

        public Builder method(String method) {	// set HTTP method
            this.method = method;	// store value
            return this;	// fluent return
        }

        public Builder get() {	// convenience method for GET
            return method("GET");	// delegate to method()
        }

        public Builder post() {	// convenience method for POST
            return method("POST");	// delegate to method()
        }

        public Builder put() {	// convenience method for PUT
            return method("PUT");	// delegate to method()
        }

        public Builder delete() {	// convenience method for DELETE
            return method("DELETE");	// delegate to method()
        }

        public Builder header(String name, String value) {	// add single header
            this.headers.put(name, value);	// add to map
            return this;	// fluent return
        }

        public Builder headers(Map<String, String> headers) {	// add multiple headers
            this.headers.putAll(headers);	// merge all
            return this;	// fluent return
        }

        public Builder contentType(String type) {	// convenience for Content-Type
            return header("Content-Type", type);	// delegate to header()
        }

        public Builder authorization(String token) {	// convenience for Auth header
            return header("Authorization", "Bearer " + token);	// add Bearer token
        }

        public Builder body(String body) {	// set request body
            this.body = body;	// store value
            return this;	// fluent return
        }

        public Builder jsonBody(Object object) {	// JSON body helper
            this.body = toJson(object);	// convert to JSON
            return contentType("application/json");	// set Content-Type
        }

        public Builder timeout(int millis) {	// set timeout
            this.timeout = millis;	// store value
            return this;	// fluent return
        }

        public Builder followRedirects(boolean follow) {	// set redirect policy
            this.followRedirects = follow;	// store value
            return this;	// fluent return
        }

        public HttpRequest build() {	// terminal operation
            validate();	// validate before building
            return new HttpRequest(this);	// create immutable request
        }

        private void validate() {	// validation logic
            if (url == null || url.isEmpty()) {	// check required field
                throw new IllegalStateException("URL is required");
            }
            if (("POST".equals(method) || "PUT".equals(method)) && body.isEmpty()) {
                // Warning: POST/PUT without body
            }
        }

        private String toJson(Object obj) {	// helper method
            return "{}";	// simplified - would use Jackson/Gson
        }
    }
}

// Usage examples
class ApiClient {	// using the builder
    public HttpRequest createGetRequest(String url) {	// simple GET
        return new HttpRequest.Builder(url)	// URL is required
            .get()	// GET method
            .header("Accept", "application/json")	// accept JSON
            .timeout(5000)	// 5 second timeout
            .build();	// build immutable request
    }

    public HttpRequest createPostRequest(String url, Object data, String token) {	// POST with auth
        return new HttpRequest.Builder(url)	// URL is required
            .post()	// POST method
            .authorization(token)	// add auth header
            .jsonBody(data)	// JSON body with Content-Type
            .timeout(10000)	// 10 second timeout
            .build();	// build immutable request
    }

    public HttpRequest createCustomRequest() {	// complex request
        return new HttpRequest.Builder("https://api.example.com/data")
            .method("PATCH")	// custom method
            .header("X-Request-Id", UUID.randomUUID().toString())	// custom header
            .header("X-Client-Version", "2.0")	// another custom header
            .contentType("application/merge-patch+json")	// specific content type
            .body("{\"status\": \"active\"}")	// raw body
            .timeout(15000)	// 15 second timeout
            .followRedirects(false)	// don't follow redirects
            .build();	// build immutable request
    }
}
\`\`\`

---

## Common Mistakes to Avoid

1. **Not making product immutable** - Use final fields and private constructor
2. **Allowing direct construction** - Make constructor private, force builder usage
3. **Returning new Builder in setters** - Return \`this\`, not \`new Builder()\`
4. **No validation in build()** - Validate required fields before creating object
5. **Not using defensive copies** - Copy mutable collections in constructor`,
	order: 3,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void builderCreatesHouse() {
        House house = new House.Builder().build();
        assertNotNull(house, "Builder should create non-null House");
    }
}

class Test2 {
    @Test
    void builderSetsFoundation() {
        House house = new House.Builder().foundation("concrete").build();
        assertEquals("concrete", house.getFoundation(), "Foundation should be set");
    }
}

class Test3 {
    @Test
    void builderSetsWalls() {
        House house = new House.Builder().walls("brick").build();
        assertEquals("brick", house.getWalls(), "Walls should be set");
    }
}

class Test4 {
    @Test
    void builderSetsRoof() {
        House house = new House.Builder().roof("tile").build();
        assertEquals("tile", house.getRoof(), "Roof should be set");
    }
}

class Test5 {
    @Test
    void builderSetsGarage() {
        House house = new House.Builder().garage(true).build();
        assertTrue(house.hasGarage(), "Garage should be true");
    }
}

class Test6 {
    @Test
    void builderSetsSwimmingPool() {
        House house = new House.Builder().swimmingPool(true).build();
        assertTrue(house.hasSwimmingPool(), "Swimming pool should be true");
    }
}

class Test7 {
    @Test
    void builderFluentChaining() {
        House house = new House.Builder()
            .foundation("concrete")
            .walls("brick")
            .roof("tile")
            .build();
        assertEquals("concrete", house.getFoundation());
        assertEquals("brick", house.getWalls());
        assertEquals("tile", house.getRoof());
    }
}

class Test8 {
    @Test
    void builderDefaultValues() {
        House house = new House.Builder().build();
        assertFalse(house.hasGarage(), "Default garage should be false");
        assertFalse(house.hasSwimmingPool(), "Default pool should be false");
    }
}

class Test9 {
    @Test
    void builderReturnsBuilder() {
        House.Builder builder = new House.Builder();
        assertSame(builder, builder.foundation("test"), "Method should return this");
        assertSame(builder, builder.walls("test"), "Method should return this");
    }
}

class Test10 {
    @Test
    void builderCreatesImmutableHouse() {
        House house = new House.Builder()
            .foundation("concrete")
            .walls("brick")
            .roof("tile")
            .garage(true)
            .swimmingPool(false)
            .build();
        assertNotNull(house);
        assertEquals("concrete", house.getFoundation());
        assertTrue(house.hasGarage());
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Builder (Строитель)',
			description: `Реализуйте паттерн Builder на Java — отделите конструирование сложных объектов с fluent-интерфейсом.

**Вы реализуете:**

1. **Класс House** с вложенным Builder
2. **Fluent-интерфейс** - каждый сеттер возвращает Builder
3. **Метод build()** - возвращает сконструированный House

**Пример использования:**

\`\`\`java
House house = new House.Builder()	// создаём экземпляр builder
    .foundation("concrete")	// устанавливаем тип фундамента
    .walls("brick")	// устанавливаем материал стен
    .roof("tile")	// устанавливаем тип крыши
    .garage(true)	// включаем гараж
    .swimmingPool(false)	// без бассейна
    .build();	// конструируем объект House

// Доступ к созданному объекту
String walls = house.getWalls();	// "brick"
boolean hasGarage = house.hasGarage();	// true
\`\`\``,
			hint1: `**Паттерн Fluent-сеттер:**

Каждый метод-сеттер в Builder следует одной схеме:

\`\`\`java
public Builder foundation(String foundation) {
    this.foundation = foundation;	// присваиваем значение полю
    return this;	// возвращаем this для цепочки вызовов
}

public Builder walls(String walls) {
    this.walls = walls;	// присваиваем значение полю
    return this;	// возвращаем this для цепочки вызовов
}
\`\`\`

Ключ в возврате \`this\` для цепочки:
\`\`\`java
new Builder().foundation("concrete").walls("brick")...	// цепочка вызовов
\`\`\``,
			hint2: `**Метод Build:**

Метод build() создаёт финальный неизменяемый объект:

\`\`\`java
public House build() {
    return new House(this);	// передаём builder в конструктор House
}
\`\`\`

Конструктор House приватный и принимает только Builder:

\`\`\`java
private House(Builder builder) {	// private - только builder может вызвать
    this.foundation = builder.foundation;	// копируем все значения
    this.walls = builder.walls;	// из builder в final поля
    this.roof = builder.roof;
    this.hasGarage = builder.hasGarage;
    this.hasSwimmingPool = builder.hasSwimmingPool;
}
\`\`\``,
			whyItMatters: `## Зачем нужен Builder

Builder решает проблему "телескопических конструкторов" — когда класс имеет много опциональных параметров. Он создаёт неизменяемые объекты пошагово с чистым, читаемым кодом.

**Проблема - Телескопические конструкторы:**

\`\`\`java
// ❌ Плохо: Множество перегрузок конструктора
class House {
    public House(String foundation) {...}
    public House(String foundation, String walls) {...}
    public House(String foundation, String walls, String roof) {...}
    public House(String foundation, String walls, String roof, boolean garage) {...}
    // Становится хуже с большим количеством параметров!
}

// Непонятно при использовании - что означает 'true'?
House h = new House("concrete", "brick", "tile", true, false, true);
\`\`\`

**Решение - Builder с именованными параметрами:**

\`\`\`java
// ✅ Хорошо: Builder с fluent-интерфейсом
House house = new House.Builder()	// чистое, читаемое конструирование
    .foundation("concrete")	// именованный параметр
    .walls("brick")	// именованный параметр
    .roof("tile")	// именованный параметр
    .garage(true)	// очевидно что это значит
    .build();	// неизменяемый результат
\`\`\`

---

## Примеры из реального мира

1. **StringBuilder** - Эффективно строит строки
2. **Stream.Builder** - Инкрементально создаёт потоки
3. **ProcessBuilder** - Конфигурирует внешние процессы
4. **Lombok @Builder** - Автоматически генерирует билдеры
5. **Protocol Buffers** - Билдеры сообщений для сериализации
6. **HttpRequest.Builder** - HTTP клиент Java 11

---

## Production паттерн: HTTP Request Builder

\`\`\`java
class HttpRequest {	// неизменяемый HTTP запрос
    private final String method;	// GET, POST, и т.д.
    private final String url;	// URL запроса
    private final Map<String, String> headers;	// HTTP заголовки
    private final String body;	// тело запроса
    private final int timeout;	// таймаут в миллисекундах
    private final boolean followRedirects;	// следовать 3xx редиректам

    private HttpRequest(Builder builder) {	// приватный конструктор
        this.method = builder.method;	// копируем из builder
        this.url = builder.url;	// копируем из builder
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));	// защитная копия
        this.body = builder.body;	// копируем из builder
        this.timeout = builder.timeout;	// копируем из builder
        this.followRedirects = builder.followRedirects;	// копируем из builder
    }

    // Геттеры
    public String getMethod() { return method; }	// HTTP метод
    public String getUrl() { return url; }	// URL запроса
    public Map<String, String> getHeaders() { return headers; }	// неизменяемые заголовки
    public String getBody() { return body; }	// тело запроса
    public int getTimeout() { return timeout; }	// таймаут
    public boolean isFollowRedirects() { return followRedirects; }	// политика редиректов

    public static class Builder {	// класс билдера
        private String method = "GET";	// метод по умолчанию
        private String url;	// обязательное поле
        private final Map<String, String> headers = new HashMap<>();	// изменяемая во время сборки
        private String body = "";	// по умолчанию пустое тело
        private int timeout = 30000;	// по умолчанию 30 секунд
        private boolean followRedirects = true;	// по умолчанию следовать редиректам

        public Builder(String url) {	// обязательный параметр в конструкторе
            this.url = Objects.requireNonNull(url, "URL is required");	// валидация обязательного поля
        }

        public Builder method(String method) {	// установить HTTP метод
            this.method = method;	// сохраняем значение
            return this;	// fluent возврат
        }

        public Builder get() {	// удобный метод для GET
            return method("GET");	// делегируем в method()
        }

        public Builder post() {	// удобный метод для POST
            return method("POST");	// делегируем в method()
        }

        public Builder put() {	// удобный метод для PUT
            return method("PUT");	// делегируем в method()
        }

        public Builder delete() {	// удобный метод для DELETE
            return method("DELETE");	// делегируем в method()
        }

        public Builder header(String name, String value) {	// добавить один заголовок
            this.headers.put(name, value);	// добавляем в map
            return this;	// fluent возврат
        }

        public Builder headers(Map<String, String> headers) {	// добавить несколько заголовков
            this.headers.putAll(headers);	// объединяем все
            return this;	// fluent возврат
        }

        public Builder contentType(String type) {	// удобный метод для Content-Type
            return header("Content-Type", type);	// делегируем в header()
        }

        public Builder authorization(String token) {	// удобный метод для Auth заголовка
            return header("Authorization", "Bearer " + token);	// добавляем Bearer токен
        }

        public Builder body(String body) {	// установить тело запроса
            this.body = body;	// сохраняем значение
            return this;	// fluent возврат
        }

        public Builder jsonBody(Object object) {	// помощник для JSON тела
            this.body = toJson(object);	// конвертируем в JSON
            return contentType("application/json");	// устанавливаем Content-Type
        }

        public Builder timeout(int millis) {	// установить таймаут
            this.timeout = millis;	// сохраняем значение
            return this;	// fluent возврат
        }

        public Builder followRedirects(boolean follow) {	// установить политику редиректов
            this.followRedirects = follow;	// сохраняем значение
            return this;	// fluent возврат
        }

        public HttpRequest build() {	// терминальная операция
            validate();	// валидируем перед созданием
            return new HttpRequest(this);	// создаём неизменяемый запрос
        }

        private void validate() {	// логика валидации
            if (url == null || url.isEmpty()) {	// проверяем обязательное поле
                throw new IllegalStateException("URL is required");
            }
            if (("POST".equals(method) || "PUT".equals(method)) && body.isEmpty()) {
                // Предупреждение: POST/PUT без тела
            }
        }

        private String toJson(Object obj) {	// вспомогательный метод
            return "{}";	// упрощённо - использовал бы Jackson/Gson
        }
    }
}

// Примеры использования
class ApiClient {	// используем builder
    public HttpRequest createGetRequest(String url) {	// простой GET
        return new HttpRequest.Builder(url)	// URL обязателен
            .get()	// GET метод
            .header("Accept", "application/json")	// принимаем JSON
            .timeout(5000)	// таймаут 5 секунд
            .build();	// строим неизменяемый запрос
    }

    public HttpRequest createPostRequest(String url, Object data, String token) {	// POST с авторизацией
        return new HttpRequest.Builder(url)	// URL обязателен
            .post()	// POST метод
            .authorization(token)	// добавляем auth заголовок
            .jsonBody(data)	// JSON тело с Content-Type
            .timeout(10000)	// таймаут 10 секунд
            .build();	// строим неизменяемый запрос
    }

    public HttpRequest createCustomRequest() {	// сложный запрос
        return new HttpRequest.Builder("https://api.example.com/data")
            .method("PATCH")	// кастомный метод
            .header("X-Request-Id", UUID.randomUUID().toString())	// кастомный заголовок
            .header("X-Client-Version", "2.0")	// ещё один кастомный заголовок
            .contentType("application/merge-patch+json")	// специфичный content type
            .body("{\"status\": \"active\"}")	// сырое тело
            .timeout(15000)	// таймаут 15 секунд
            .followRedirects(false)	// не следовать редиректам
            .build();	// строим неизменяемый запрос
    }
}
\`\`\`

---

## Частые ошибки, которых следует избегать

1. **Не делать продукт неизменяемым** - Используйте final поля и private конструктор
2. **Разрешать прямое конструирование** - Делайте конструктор private, форсируйте использование builder
3. **Возвращать new Builder в сеттерах** - Возвращайте \`this\`, не \`new Builder()\`
4. **Нет валидации в build()** - Валидируйте обязательные поля перед созданием объекта
5. **Не использовать защитные копии** - Копируйте изменяемые коллекции в конструкторе`
		},
		uz: {
			title: 'Builder (Quruvchi) Pattern',
			description: `Java da Builder patternini amalga oshiring — murakkab ob'ektlar konstruksiyasini fluent-interfeys bilan ajrating.

**Siz amalga oshirasiz:**

1. **House klassi** ichki Builder bilan
2. **Fluent-interfeys** - har bir setter Builder qaytaradi
3. **build() metodi** - qurilgan House qaytaradi

**Foydalanish misoli:**

\`\`\`java
House house = new House.Builder()	// builder instansiyasini yaratamiz
    .foundation("concrete")	// poydevor turini o'rnatamiz
    .walls("brick")	// devor materialini o'rnatamiz
    .roof("tile")	// tom turini o'rnatamiz
    .garage(true)	// garaj qo'shamiz
    .swimmingPool(false)	// basseyn yo'q
    .build();	// House ob'ektini quramiz

// Qurilgan ob'ektga kirish
String walls = house.getWalls();	// "brick"
boolean hasGarage = house.hasGarage();	// true
\`\`\``,
			hint1: `**Fluent-setter patterni:**

Builder dagi har bir setter metodi bir xil sxemaga amal qiladi:

\`\`\`java
public Builder foundation(String foundation) {
    this.foundation = foundation;	// qiymatni maydonga tayinlaymiz
    return this;	// zanjir uchun this qaytaramiz
}

public Builder walls(String walls) {
    this.walls = walls;	// qiymatni maydonga tayinlaymiz
    return this;	// zanjir uchun this qaytaramiz
}
\`\`\`

Kalit \`this\` qaytarishda - zanjir qilish uchun:
\`\`\`java
new Builder().foundation("concrete").walls("brick")...	// zanjirlangan chaqiruvlar
\`\`\``,
			hint2: `**Build metodi:**

build() metodi yakuniy o'zgarmas ob'ektni yaratadi:

\`\`\`java
public House build() {
    return new House(this);	// builder ni House konstruktoriga uzatamiz
}
\`\`\`

House konstruktori private va faqat Builder qabul qiladi:

\`\`\`java
private House(Builder builder) {	// private - faqat builder chaqira oladi
    this.foundation = builder.foundation;	// barcha qiymatlarni ko'chiramiz
    this.walls = builder.walls;	// builder dan final maydonlarga
    this.roof = builder.roof;
    this.hasGarage = builder.hasGarage;
    this.hasSwimmingPool = builder.hasSwimmingPool;
}
\`\`\``,
			whyItMatters: `## Builder nima uchun kerak

Builder "teleskopik konstruktorlar" muammosini hal qiladi — klass ko'p ixtiyoriy parametrlarga ega bo'lganda. U o'zgarmas ob'ektlarni bosqichma-bosqich toza, o'qiladigan kod bilan yaratadi.

**Muammo - Teleskopik konstruktorlar:**

\`\`\`java
// ❌ Yomon: Ko'p konstruktor overloadlari
class House {
    public House(String foundation) {...}
    public House(String foundation, String walls) {...}
    public House(String foundation, String walls, String roof) {...}
    public House(String foundation, String walls, String roof, boolean garage) {...}
    // Ko'proq parametrlar bilan yomonlashadi!
}

// Foydalanishda chalkash - 'true' nimani anglatadi?
House h = new House("concrete", "brick", "tile", true, false, true);
\`\`\`

**Yechim - Nomlangan parametrli Builder:**

\`\`\`java
// ✅ Yaxshi: Fluent-interfeys bilan Builder
House house = new House.Builder()	// toza, o'qiladigan konstruksiya
    .foundation("concrete")	// nomlangan parametr
    .walls("brick")	// nomlangan parametr
    .roof("tile")	// nomlangan parametr
    .garage(true)	// bu nimani anglatishi ravshan
    .build();	// o'zgarmas natija
\`\`\`

---

## Haqiqiy dunyo misollari

1. **StringBuilder** - Satrlarni samarali quradi
2. **Stream.Builder** - Streamlarni bosqichma-bosqich yaratadi
3. **ProcessBuilder** - Tashqi jarayonlarni sozlaydi
4. **Lombok @Builder** - Builderlarni avtomatik generatsiya qiladi
5. **Protocol Buffers** - Serializatsiya uchun xabar builderlari
6. **HttpRequest.Builder** - Java 11 HTTP klienti

---

## Production pattern: HTTP Request Builder

\`\`\`java
class HttpRequest {	// o'zgarmas HTTP so'rov
    private final String method;	// GET, POST, va h.k.
    private final String url;	// so'rov URL
    private final Map<String, String> headers;	// HTTP headerlar
    private final String body;	// so'rov tanasi
    private final int timeout;	// millisekundlarda timeout
    private final boolean followRedirects;	// 3xx redirectlarga ergashish

    private HttpRequest(Builder builder) {	// private konstruktor
        this.method = builder.method;	// builder dan ko'chiramiz
        this.url = builder.url;	// builder dan ko'chiramiz
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));	// himoya nusxasi
        this.body = builder.body;	// builder dan ko'chiramiz
        this.timeout = builder.timeout;	// builder dan ko'chiramiz
        this.followRedirects = builder.followRedirects;	// builder dan ko'chiramiz
    }

    // Getterlar
    public String getMethod() { return method; }	// HTTP metod
    public String getUrl() { return url; }	// so'rov URL
    public Map<String, String> getHeaders() { return headers; }	// o'zgarmas headerlar
    public String getBody() { return body; }	// so'rov tanasi
    public int getTimeout() { return timeout; }	// timeout
    public boolean isFollowRedirects() { return followRedirects; }	// redirect siyosati

    public static class Builder {	// builder klassi
        private String method = "GET";	// standart metod
        private String url;	// majburiy maydon
        private final Map<String, String> headers = new HashMap<>();	// qurish vaqtida o'zgaruvchan
        private String body = "";	// standart bo'sh tana
        private int timeout = 30000;	// standart 30 sekund
        private boolean followRedirects = true;	// standart redirectlarga ergashish

        public Builder(String url) {	// konstruktorda majburiy parametr
            this.url = Objects.requireNonNull(url, "URL is required");	// majburiy maydonni tekshirish
        }

        public Builder method(String method) {	// HTTP metodini o'rnatish
            this.method = method;	// qiymatni saqlaymiz
            return this;	// fluent qaytarish
        }

        public Builder get() {	// GET uchun qulay metod
            return method("GET");	// method() ga delegatsiya
        }

        public Builder post() {	// POST uchun qulay metod
            return method("POST");	// method() ga delegatsiya
        }

        public Builder put() {	// PUT uchun qulay metod
            return method("PUT");	// method() ga delegatsiya
        }

        public Builder delete() {	// DELETE uchun qulay metod
            return method("DELETE");	// method() ga delegatsiya
        }

        public Builder header(String name, String value) {	// bitta header qo'shish
            this.headers.put(name, value);	// map ga qo'shamiz
            return this;	// fluent qaytarish
        }

        public Builder headers(Map<String, String> headers) {	// bir nechta header qo'shish
            this.headers.putAll(headers);	// barchasini birlashtiramiz
            return this;	// fluent qaytarish
        }

        public Builder contentType(String type) {	// Content-Type uchun qulay metod
            return header("Content-Type", type);	// header() ga delegatsiya
        }

        public Builder authorization(String token) {	// Auth header uchun qulay metod
            return header("Authorization", "Bearer " + token);	// Bearer token qo'shamiz
        }

        public Builder body(String body) {	// so'rov tanasini o'rnatish
            this.body = body;	// qiymatni saqlaymiz
            return this;	// fluent qaytarish
        }

        public Builder jsonBody(Object object) {	// JSON tana yordamchisi
            this.body = toJson(object);	// JSON ga aylantiramiz
            return contentType("application/json");	// Content-Type o'rnatamiz
        }

        public Builder timeout(int millis) {	// timeout o'rnatish
            this.timeout = millis;	// qiymatni saqlaymiz
            return this;	// fluent qaytarish
        }

        public Builder followRedirects(boolean follow) {	// redirect siyosatini o'rnatish
            this.followRedirects = follow;	// qiymatni saqlaymiz
            return this;	// fluent qaytarish
        }

        public HttpRequest build() {	// terminal operatsiya
            validate();	// yaratishdan oldin tekshiramiz
            return new HttpRequest(this);	// o'zgarmas so'rov yaratamiz
        }

        private void validate() {	// tekshirish mantiqi
            if (url == null || url.isEmpty()) {	// majburiy maydonni tekshiramiz
                throw new IllegalStateException("URL is required");
            }
            if (("POST".equals(method) || "PUT".equals(method)) && body.isEmpty()) {
                // Ogohlantirish: POST/PUT tanasiz
            }
        }

        private String toJson(Object obj) {	// yordamchi metod
            return "{}";	// soddalashtirilgan - Jackson/Gson ishlatiladi
        }
    }
}

// Foydalanish misollari
class ApiClient {	// builder dan foydalanamiz
    public HttpRequest createGetRequest(String url) {	// oddiy GET
        return new HttpRequest.Builder(url)	// URL majburiy
            .get()	// GET metodi
            .header("Accept", "application/json")	// JSON qabul qilamiz
            .timeout(5000)	// 5 sekund timeout
            .build();	// o'zgarmas so'rov quramiz
    }

    public HttpRequest createPostRequest(String url, Object data, String token) {	// avtorizatsiya bilan POST
        return new HttpRequest.Builder(url)	// URL majburiy
            .post()	// POST metodi
            .authorization(token)	// auth header qo'shamiz
            .jsonBody(data)	// Content-Type bilan JSON tana
            .timeout(10000)	// 10 sekund timeout
            .build();	// o'zgarmas so'rov quramiz
    }

    public HttpRequest createCustomRequest() {	// murakkab so'rov
        return new HttpRequest.Builder("https://api.example.com/data")
            .method("PATCH")	// maxsus metod
            .header("X-Request-Id", UUID.randomUUID().toString())	// maxsus header
            .header("X-Client-Version", "2.0")	// yana bir maxsus header
            .contentType("application/merge-patch+json")	// maxsus content type
            .body("{\"status\": \"active\"}")	// xom tana
            .timeout(15000)	// 15 sekund timeout
            .followRedirects(false)	// redirectlarga ergashmaslik
            .build();	// o'zgarmas so'rov quramiz
    }
}
\`\`\`

---

## Qochish kerak bo'lgan keng tarqalgan xatolar

1. **Mahsulotni o'zgarmas qilmaslik** - Final maydonlar va private konstruktor ishlating
2. **To'g'ridan-to'g'ri konstruksiyaga ruxsat berish** - Konstruktorni private qiling, builder ishlatishni majburlang
3. **Setterlarda new Builder qaytarish** - \`new Builder()\` emas, \`this\` qaytaring
4. **build() da validatsiya yo'q** - Ob'ekt yaratishdan oldin majburiy maydonlarni tekshiring
5. **Himoya nusxalari ishlatmaslik** - Konstruktorda o'zgaruvchan to'plamlarni ko'chiring`
		}
	}
};

export default task;
