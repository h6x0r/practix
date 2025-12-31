import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-chain-of-responsibility',
	title: 'Chain of Responsibility',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'chain-of-responsibility'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Chain of Responsibility Pattern

The **Chain of Responsibility** pattern avoids coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. It chains the receiving objects and passes the request along the chain until an object handles it.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **Handler** | Abstract class/interface with \`setNext()\` and \`handle()\` methods |
| **ConcreteHandler** | Handles requests it's responsible for; passes others to next handler |
| **Client** | Initiates the request to a handler in the chain |

---

### Your Task

Implement a request processing pipeline:

1. **Handler abstract class** - \`setNext()\` and \`handle()\` methods
2. **AuthHandler** - checks authentication, fails if not authenticated
3. **ValidationHandler** - checks data validity, fails if data is empty
4. **LoggingHandler** - logs request and passes to next handler

---

### Example Usage

\`\`\`java
Handler auth = new AuthHandler();	// create authentication handler
Handler validation = new ValidationHandler();	// create validation handler
Handler logging = new LoggingHandler();	// create logging handler

auth.setNext(validation).setNext(logging);	// build the chain: auth → validation → logging

// Test 1: Unauthenticated request
Request req1 = new Request("john", false, "data");	// user not authenticated
String r1 = auth.handle(req1);	// "Auth failed for john" - chain stops at AuthHandler

// Test 2: Empty data
Request req2 = new Request("john", true, "");	// authenticated but empty data
String r2 = auth.handle(req2);	// "Validation failed: empty data" - chain stops at ValidationHandler

// Test 3: Valid request
Request req3 = new Request("john", true, "payload");	// valid request
String r3 = auth.handle(req3);	// "Logged: john - Request processed successfully"
\`\`\`

---

### Key Insight

Chain of Responsibility enables **loose coupling** between senders and receivers. The sender doesn't need to know which handler will process the request - it just sends to the first handler in the chain.`,
	initialCode: `abstract class Handler {
    protected Handler next;

    public Handler setNext(Handler next) {
    }

    public abstract String handle(Request request);

    protected String handleNext(Request request) {
        }
    }
}

class Request {
    public String user;
    public boolean authenticated;
    public String data;

    public Request(String user, boolean authenticated, String data) {
    }
}

class AuthHandler extends Handler {
    @Override
    public String handle(Request request) {
        throw new UnsupportedOperationException("TODO");
    }
}

class ValidationHandler extends Handler {
    @Override
    public String handle(Request request) {
        throw new UnsupportedOperationException("TODO");
    }
}

class LoggingHandler extends Handler {
    @Override
    public String handle(Request request) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `abstract class Handler {	// Handler - base class for all handlers in chain
    protected Handler next;	// reference to next handler in chain

    public Handler setNext(Handler next) {	// link to next handler
        this.next = next;	// set the next handler
        return next;	// return next for fluent chaining
    }

    public abstract String handle(Request request);	// handle request - implemented by subclasses

    protected String handleNext(Request request) {	// pass to next handler in chain
        if (next != null) {	// if there's a next handler
            return next.handle(request);	// delegate to it
        }
        return "Request processed successfully";	// end of chain - success
    }
}

class Request {	// Request object passed through chain
    public String user;	// username
    public boolean authenticated;	// authentication status
    public String data;	// request data

    public Request(String user, boolean authenticated, String data) {	// constructor
        this.user = user;	// set user
        this.authenticated = authenticated;	// set auth status
        this.data = data;	// set data
    }
}

class AuthHandler extends Handler {	// ConcreteHandler - authentication check
    @Override
    public String handle(Request request) {	// handle authentication
        if (!request.authenticated) {	// check if not authenticated
            return "Auth failed for " + request.user;	// reject request
        }
        return handleNext(request);	// pass to next handler
    }
}

class ValidationHandler extends Handler {	// ConcreteHandler - data validation
    @Override
    public String handle(Request request) {	// handle validation
        if (request.data == null || request.data.isEmpty()) {	// check for empty data
            return "Validation failed: empty data";	// reject request
        }
        return handleNext(request);	// pass to next handler
    }
}

class LoggingHandler extends Handler {	// ConcreteHandler - logging
    @Override
    public String handle(Request request) {	// handle logging
        return "Logged: " + request.user + " - " + handleNext(request);	// log and continue
    }
}`,
	hint1: `### Understanding Handler Chain Structure

The pattern has a base Handler class and concrete handlers:

\`\`\`java
abstract class Handler {
    protected Handler next;  // Reference to next handler

    public Handler setNext(Handler next) {
        this.next = next;
        return next;  // Enables chaining: a.setNext(b).setNext(c)
    }

    protected String handleNext(Request request) {
        if (next != null) {
            return next.handle(request);  // Pass to next handler
        }
        return "Request processed successfully";  // End of chain
    }
}

// Each concrete handler follows this pattern:
class AuthHandler extends Handler {
    @Override
    public String handle(Request request) {
        if (!request.authenticated) {  // Check condition
            return "Auth failed for " + request.user;  // Stop chain
        }
        return handleNext(request);  // Continue chain
    }
}
\`\`\``,
	hint2: `### Implementing Each Handler

**AuthHandler** - checks authentication:
\`\`\`java
public String handle(Request request) {
    if (!request.authenticated) {
        return "Auth failed for " + request.user;
    }
    return handleNext(request);
}
\`\`\`

**ValidationHandler** - checks data:
\`\`\`java
public String handle(Request request) {
    if (request.data == null || request.data.isEmpty()) {
        return "Validation failed: empty data";
    }
    return handleNext(request);
}
\`\`\`

**LoggingHandler** - logs and continues:
\`\`\`java
public String handle(Request request) {
    // Always continues to next, wraps result with log prefix
    return "Logged: " + request.user + " - " + handleNext(request);
}
\`\`\``,
	whyItMatters: `## Why Chain of Responsibility Matters

### The Problem and Solution

**Without Chain of Responsibility:**
\`\`\`java
// Tight coupling - processor knows all handlers
class RequestProcessor {
    public String process(Request request) {
        // All checks in one place - hard to modify
        if (!request.authenticated) {	// auth check
            return "Auth failed";
        }
        if (request.data == null) {	// validation check
            return "Validation failed";
        }
        log(request);	// logging
        // Adding new check requires modifying this class!
        return "Success";
    }
}
\`\`\`

**With Chain of Responsibility:**
\`\`\`java
// Loose coupling - each handler is independent
Handler auth = new AuthHandler();	// create handlers
Handler validation = new ValidationHandler();
Handler logging = new LoggingHandler();

auth.setNext(validation).setNext(logging);	// build chain

String result = auth.handle(request);	// start processing
// Adding new handler = add new class, insert into chain!
\`\`\`

---

### Real-World Applications

| Application | Chain | Handlers |
|-------------|-------|----------|
| **Servlet Filters** | FilterChain | Auth, CORS, Compression filters |
| **Logging** | Logger chain | Console, File, Network handlers |
| **Event Handling** | DOM events | Capture, Target, Bubble phases |
| **Middleware** | Express/Koa | Auth, Parsing, Error middleware |
| **Support System** | Ticket escalation | L1, L2, L3 support |

---

### Production Pattern: HTTP Middleware Pipeline

\`\`\`java
// Middleware interface for HTTP processing
interface Middleware {	// middleware contract
    Response handle(HttpRequest request, MiddlewareChain chain);	// handle request
}

class MiddlewareChain {	// manages middleware execution
    private final List<Middleware> middlewares;	// list of middleware
    private int index = 0;	// current position in chain

    public MiddlewareChain(List<Middleware> middlewares) {	// constructor
        this.middlewares = middlewares;	// store middleware list
    }

    public Response proceed(HttpRequest request) {	// proceed to next middleware
        if (index < middlewares.size()) {	// if more middleware
            Middleware current = middlewares.get(index++);	// get and advance
            return current.handle(request, this);	// execute middleware
        }
        return new Response(200, "OK");	// end of chain - success
    }
}

class AuthMiddleware implements Middleware {	// authentication middleware
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String token = request.getHeader("Authorization");	// get auth token
        if (token == null || !validateToken(token)) {	// validate token
            return new Response(401, "Unauthorized");	// reject if invalid
        }
        request.setAttribute("user", extractUser(token));	// add user to request
        return chain.proceed(request);	// continue chain
    }
}

class RateLimitMiddleware implements Middleware {	// rate limiting middleware
    private final RateLimiter limiter;	// rate limiter instance

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String clientId = request.getClientId();	// get client identifier
        if (!limiter.allowRequest(clientId)) {	// check rate limit
            return new Response(429, "Too Many Requests");	// reject if exceeded
        }
        return chain.proceed(request);	// continue chain
    }
}

class LoggingMiddleware implements Middleware {	// logging middleware
    private final Logger logger;	// logger instance

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        long start = System.currentTimeMillis();	// record start time
        logger.info("Request: " + request.getPath());	// log request

        Response response = chain.proceed(request);	// continue chain

        long duration = System.currentTimeMillis() - start;	// calculate duration
        logger.info("Response: " + response.getStatus() + " in " + duration + "ms");	// log response
        return response;	// return response
    }
}

class ErrorHandlingMiddleware implements Middleware {	// error handling middleware
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        try {
            return chain.proceed(request);	// try to proceed
        } catch (Exception e) {	// catch any exception
            logger.error("Error processing request", e);	// log error
            return new Response(500, "Internal Server Error");	// return error response
        }
    }
}

// Usage - build and use the pipeline:
List<Middleware> middlewares = Arrays.asList(	// create middleware list
    new ErrorHandlingMiddleware(),	// outermost - catches all errors
    new LoggingMiddleware(),	// logs all requests
    new RateLimitMiddleware(),	// rate limiting
    new AuthMiddleware()	// authentication
);

MiddlewareChain chain = new MiddlewareChain(middlewares);	// create chain
Response response = chain.proceed(request);	// process request
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Forgetting to call next** | Request stops unexpectedly | Always call handleNext unless intentionally stopping |
| **Infinite loops** | Handler calls itself | Ensure handlers don't create circular references |
| **Order dependency** | Wrong processing order | Document required order; use builder pattern |
| **No handler catches** | Request falls through | Add fallback handler at end of chain |
| **Tight coupling** | Handlers know about each other | Handlers should only know about abstract Handler |`,
	order: 4,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: AuthHandler rejects unauthenticated
class Test1 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Request req = new Request("john", false, "data");
        String result = auth.handle(req);
        assertEquals("Auth failed for john", result);
    }
}

// Test2: AuthHandler passes authenticated
class Test2 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Request req = new Request("jane", true, "data");
        String result = auth.handle(req);
        assertEquals("Request processed successfully", result);
    }
}

// Test3: ValidationHandler rejects empty data
class Test3 {
    @Test
    public void test() {
        Handler validation = new ValidationHandler();
        Request req = new Request("john", true, "");
        String result = validation.handle(req);
        assertEquals("Validation failed: empty data", result);
    }
}

// Test4: ValidationHandler rejects null data
class Test4 {
    @Test
    public void test() {
        Handler validation = new ValidationHandler();
        Request req = new Request("john", true, null);
        String result = validation.handle(req);
        assertEquals("Validation failed: empty data", result);
    }
}

// Test5: LoggingHandler logs and continues
class Test5 {
    @Test
    public void test() {
        Handler logging = new LoggingHandler();
        Request req = new Request("alice", true, "test");
        String result = logging.handle(req);
        assertTrue(result.startsWith("Logged: alice"));
    }
}

// Test6: Chain auth -> validation
class Test6 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Handler validation = new ValidationHandler();
        auth.setNext(validation);
        Request req = new Request("bob", true, "");
        String result = auth.handle(req);
        assertEquals("Validation failed: empty data", result);
    }
}

// Test7: Full chain success
class Test7 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Handler validation = new ValidationHandler();
        Handler logging = new LoggingHandler();
        auth.setNext(validation).setNext(logging);
        Request req = new Request("user", true, "payload");
        String result = auth.handle(req);
        assertTrue(result.contains("Logged: user"));
    }
}

// Test8: setNext returns next handler
class Test8 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Handler validation = new ValidationHandler();
        Handler returned = auth.setNext(validation);
        assertSame(validation, returned);
    }
}

// Test9: Chain stops at first failure
class Test9 {
    @Test
    public void test() {
        Handler auth = new AuthHandler();
        Handler logging = new LoggingHandler();
        auth.setNext(logging);
        Request req = new Request("test", false, "data");
        String result = auth.handle(req);
        assertEquals("Auth failed for test", result);
    }
}

// Test10: Logging wraps success message
class Test10 {
    @Test
    public void test() {
        Handler logging = new LoggingHandler();
        Request req = new Request("admin", true, "data");
        String result = logging.handle(req);
        assertEquals("Logged: admin - Request processed successfully", result);
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Chain of Responsibility',
			description: `## Паттерн Chain of Responsibility

Паттерн **Chain of Responsibility** избегает связывания отправителя запроса с его получателем, давая более чем одному объекту возможность обработать запрос. Он связывает получающие объекты в цепочку и передаёт запрос по цепочке, пока какой-либо объект не обработает его.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Handler** | Абстрактный класс/интерфейс с методами \`setNext()\` и \`handle()\` |
| **ConcreteHandler** | Обрабатывает запросы, за которые отвечает; передаёт остальные следующему |
| **Client** | Инициирует запрос к обработчику в цепочке |

---

### Ваша задача

Реализуйте конвейер обработки запросов:

1. **Абстрактный класс Handler** - методы \`setNext()\` и \`handle()\`
2. **AuthHandler** - проверяет аутентификацию, отказывает если не аутентифицирован
3. **ValidationHandler** - проверяет валидность данных, отказывает если данные пусты
4. **LoggingHandler** - логирует запрос и передаёт следующему обработчику

---

### Пример использования

\`\`\`java
Handler auth = new AuthHandler();	// создаём обработчик аутентификации
Handler validation = new ValidationHandler();	// создаём обработчик валидации
Handler logging = new LoggingHandler();	// создаём обработчик логирования

auth.setNext(validation).setNext(logging);	// строим цепочку: auth → validation → logging

// Тест 1: Неаутентифицированный запрос
Request req1 = new Request("john", false, "data");	// пользователь не аутентифицирован
String r1 = auth.handle(req1);	// "Auth failed for john" - цепочка останавливается на AuthHandler

// Тест 2: Пустые данные
Request req2 = new Request("john", true, "");	// аутентифицирован, но данные пусты
String r2 = auth.handle(req2);	// "Validation failed: empty data" - останавливается на ValidationHandler

// Тест 3: Валидный запрос
Request req3 = new Request("john", true, "payload");	// валидный запрос
String r3 = auth.handle(req3);	// "Logged: john - Request processed successfully"
\`\`\`

---

### Ключевая идея

Chain of Responsibility обеспечивает **слабую связанность** между отправителями и получателями. Отправителю не нужно знать, какой обработчик обработает запрос — он просто отправляет первому обработчику в цепочке.`,
			hint1: `### Понимание структуры цепочки обработчиков

Паттерн имеет базовый класс Handler и конкретные обработчики:

\`\`\`java
abstract class Handler {
    protected Handler next;  // Ссылка на следующий обработчик

    public Handler setNext(Handler next) {
        this.next = next;
        return next;  // Позволяет цепочку: a.setNext(b).setNext(c)
    }

    protected String handleNext(Request request) {
        if (next != null) {
            return next.handle(request);  // Передать следующему
        }
        return "Request processed successfully";  // Конец цепочки
    }
}

// Каждый конкретный обработчик следует этому шаблону:
class AuthHandler extends Handler {
    @Override
    public String handle(Request request) {
        if (!request.authenticated) {  // Проверить условие
            return "Auth failed for " + request.user;  // Остановить цепочку
        }
        return handleNext(request);  // Продолжить цепочку
    }
}
\`\`\``,
			hint2: `### Реализация каждого обработчика

**AuthHandler** - проверяет аутентификацию:
\`\`\`java
public String handle(Request request) {
    if (!request.authenticated) {
        return "Auth failed for " + request.user;
    }
    return handleNext(request);
}
\`\`\`

**ValidationHandler** - проверяет данные:
\`\`\`java
public String handle(Request request) {
    if (request.data == null || request.data.isEmpty()) {
        return "Validation failed: empty data";
    }
    return handleNext(request);
}
\`\`\`

**LoggingHandler** - логирует и продолжает:
\`\`\`java
public String handle(Request request) {
    // Всегда продолжает к следующему, оборачивает результат префиксом лога
    return "Logged: " + request.user + " - " + handleNext(request);
}
\`\`\``,
			whyItMatters: `## Почему паттерн Chain of Responsibility важен

### Проблема и решение

**Без Chain of Responsibility:**
\`\`\`java
// Тесная связанность - процессор знает все обработчики
class RequestProcessor {
    public String process(Request request) {
        // Все проверки в одном месте - сложно модифицировать
        if (!request.authenticated) {	// проверка авторизации
            return "Auth failed";
        }
        if (request.data == null) {	// проверка валидации
            return "Validation failed";
        }
        log(request);	// логирование
        // Добавление новой проверки требует изменения этого класса!
        return "Success";
    }
}
\`\`\`

**С Chain of Responsibility:**
\`\`\`java
// Слабая связанность - каждый обработчик независим
Handler auth = new AuthHandler();	// создаём обработчики
Handler validation = new ValidationHandler();
Handler logging = new LoggingHandler();

auth.setNext(validation).setNext(logging);	// строим цепочку

String result = auth.handle(request);	// начинаем обработку
// Добавление нового обработчика = добавить новый класс, вставить в цепочку!
\`\`\`

---

### Применение в реальном мире

| Применение | Цепочка | Обработчики |
|------------|---------|-------------|
| **Servlet Filters** | FilterChain | Auth, CORS, Compression фильтры |
| **Логирование** | Цепочка логгеров | Console, File, Network обработчики |
| **Обработка событий** | DOM события | Capture, Target, Bubble фазы |
| **Middleware** | Express/Koa | Auth, Parsing, Error middleware |
| **Система поддержки** | Эскалация тикетов | L1, L2, L3 поддержка |

---

### Продакшен паттерн: HTTP Middleware Pipeline

\`\`\`java
// Интерфейс middleware для HTTP обработки
interface Middleware {	// контракт middleware
    Response handle(HttpRequest request, MiddlewareChain chain);	// обработать запрос
}

class MiddlewareChain {	// управляет выполнением middleware
    private final List<Middleware> middlewares;	// список middleware
    private int index = 0;	// текущая позиция в цепочке

    public MiddlewareChain(List<Middleware> middlewares) {	// конструктор
        this.middlewares = middlewares;	// сохраняем список
    }

    public Response proceed(HttpRequest request) {	// перейти к следующему middleware
        if (index < middlewares.size()) {	// если есть ещё middleware
            Middleware current = middlewares.get(index++);	// получить и продвинуться
            return current.handle(request, this);	// выполнить middleware
        }
        return new Response(200, "OK");	// конец цепочки - успех
    }
}

class AuthMiddleware implements Middleware {	// middleware аутентификации
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String token = request.getHeader("Authorization");	// получить токен
        if (token == null || !validateToken(token)) {	// проверить токен
            return new Response(401, "Unauthorized");	// отклонить если невалидный
        }
        request.setAttribute("user", extractUser(token));	// добавить пользователя
        return chain.proceed(request);	// продолжить цепочку
    }
}

class RateLimitMiddleware implements Middleware {	// middleware ограничения запросов
    private final RateLimiter limiter;	// экземпляр лимитера

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String clientId = request.getClientId();	// получить ID клиента
        if (!limiter.allowRequest(clientId)) {	// проверить лимит
            return new Response(429, "Too Many Requests");	// отклонить если превышен
        }
        return chain.proceed(request);	// продолжить цепочку
    }
}

class LoggingMiddleware implements Middleware {	// middleware логирования
    private final Logger logger;	// экземпляр логгера

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        long start = System.currentTimeMillis();	// записать время начала
        logger.info("Request: " + request.getPath());	// залогировать запрос

        Response response = chain.proceed(request);	// продолжить цепочку

        long duration = System.currentTimeMillis() - start;	// вычислить длительность
        logger.info("Response: " + response.getStatus() + " in " + duration + "ms");	// залогировать ответ
        return response;	// вернуть ответ
    }
}

class ErrorHandlingMiddleware implements Middleware {	// middleware обработки ошибок
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        try {
            return chain.proceed(request);	// попытаться продолжить
        } catch (Exception e) {	// поймать любое исключение
            logger.error("Error processing request", e);	// залогировать ошибку
            return new Response(500, "Internal Server Error");	// вернуть ответ об ошибке
        }
    }
}

// Использование - создание и использование конвейера:
List<Middleware> middlewares = Arrays.asList(	// создать список middleware
    new ErrorHandlingMiddleware(),	// внешний - ловит все ошибки
    new LoggingMiddleware(),	// логирует все запросы
    new RateLimitMiddleware(),	// ограничение запросов
    new AuthMiddleware()	// аутентификация
);

MiddlewareChain chain = new MiddlewareChain(middlewares);	// создать цепочку
Response response = chain.proceed(request);	// обработать запрос
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Забыли вызвать next** | Запрос неожиданно останавливается | Всегда вызывайте handleNext если не останавливаете намеренно |
| **Бесконечные циклы** | Обработчик вызывает сам себя | Убедитесь что обработчики не создают циклические ссылки |
| **Зависимость от порядка** | Неправильный порядок обработки | Документируйте требуемый порядок; используйте builder паттерн |
| **Нет перехвата обработчиком** | Запрос проходит насквозь | Добавьте fallback обработчик в конце цепочки |
| **Тесная связанность** | Обработчики знают друг о друге | Обработчики должны знать только абстрактный Handler |`
		},
		uz: {
			title: 'Chain of Responsibility Pattern',
			description: `## Chain of Responsibility Pattern

**Chain of Responsibility** pattern so'rovni jo'natuvchini qabul qiluvchi bilan bog'lashdan qochadi, bir nechta obyektga so'rovni qayta ishlash imkoniyatini beradi. U qabul qiluvchi obyektlarni zanjirga bog'laydi va so'rovni biror obyekt qayta ishlamaguncha zanjir bo'ylab uzatadi.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **Handler** | \`setNext()\` va \`handle()\` metodlari bilan abstrakt klass/interfeys |
| **ConcreteHandler** | Javobgar so'rovlarni qayta ishlaydi; boshqalarni keyingisiga uzatadi |
| **Client** | Zanjirdagi ishlov beruvchiga so'rovni boshlaydi |

---

### Vazifangiz

So'rovlarni qayta ishlash quvurini amalga oshiring:

1. **Handler abstrakt klassi** - \`setNext()\` va \`handle()\` metodlari
2. **AuthHandler** - autentifikatsiyani tekshiradi, autentifikatsiya qilinmagan bo'lsa rad etadi
3. **ValidationHandler** - ma'lumotlar haqiqiyligini tekshiradi, bo'sh bo'lsa rad etadi
4. **LoggingHandler** - so'rovni loglaydi va keyingi ishlov beruvchiga uzatadi

---

### Foydalanish Namunasi

\`\`\`java
Handler auth = new AuthHandler();	// autentifikatsiya ishlov beruvchisini yaratamiz
Handler validation = new ValidationHandler();	// validatsiya ishlov beruvchisini yaratamiz
Handler logging = new LoggingHandler();	// loglash ishlov beruvchisini yaratamiz

auth.setNext(validation).setNext(logging);	// zanjirni quramiz: auth → validation → logging

// Test 1: Autentifikatsiya qilinmagan so'rov
Request req1 = new Request("john", false, "data");	// foydalanuvchi autentifikatsiya qilinmagan
String r1 = auth.handle(req1);	// "Auth failed for john" - zanjir AuthHandler da to'xtaydi

// Test 2: Bo'sh ma'lumotlar
Request req2 = new Request("john", true, "");	// autentifikatsiya qilingan lekin ma'lumotlar bo'sh
String r2 = auth.handle(req2);	// "Validation failed: empty data" - ValidationHandler da to'xtaydi

// Test 3: Yaroqli so'rov
Request req3 = new Request("john", true, "payload");	// yaroqli so'rov
String r3 = auth.handle(req3);	// "Logged: john - Request processed successfully"
\`\`\`

---

### Asosiy Fikr

Chain of Responsibility jo'natuvchilar va qabul qiluvchilar o'rtasida **zaif bog'lanish**ni ta'minlaydi. Jo'natuvchi qaysi ishlov beruvchi so'rovni qayta ishlashini bilishi shart emas — u shunchaki zanjirdagi birinchi ishlov beruvchiga yuboradi.`,
			hint1: `### Ishlov Beruvchilar Zanjiri Strukturasini Tushunish

Pattern bazaviy Handler klassi va aniq ishlov beruvchilarga ega:

\`\`\`java
abstract class Handler {
    protected Handler next;  // Keyingi ishlov beruvchiga havola

    public Handler setNext(Handler next) {
        this.next = next;
        return next;  // Zanjirni yoqadi: a.setNext(b).setNext(c)
    }

    protected String handleNext(Request request) {
        if (next != null) {
            return next.handle(request);  // Keyingisiga uzatish
        }
        return "Request processed successfully";  // Zanjir oxiri
    }
}

// Har bir aniq ishlov beruvchi ushbu shablonga amal qiladi:
class AuthHandler extends Handler {
    @Override
    public String handle(Request request) {
        if (!request.authenticated) {  // Shartni tekshirish
            return "Auth failed for " + request.user;  // Zanjirni to'xtatish
        }
        return handleNext(request);  // Zanjirni davom ettirish
    }
}
\`\`\``,
			hint2: `### Har Bir Ishlov Beruvchini Amalga Oshirish

**AuthHandler** - autentifikatsiyani tekshiradi:
\`\`\`java
public String handle(Request request) {
    if (!request.authenticated) {
        return "Auth failed for " + request.user;
    }
    return handleNext(request);
}
\`\`\`

**ValidationHandler** - ma'lumotlarni tekshiradi:
\`\`\`java
public String handle(Request request) {
    if (request.data == null || request.data.isEmpty()) {
        return "Validation failed: empty data";
    }
    return handleNext(request);
}
\`\`\`

**LoggingHandler** - loglaydi va davom ettiradi:
\`\`\`java
public String handle(Request request) {
    // Har doim keyingisiga davom etadi, natijani log prefiksi bilan o'raydi
    return "Logged: " + request.user + " - " + handleNext(request);
}
\`\`\``,
			whyItMatters: `## Nima Uchun Chain of Responsibility Muhim

### Muammo va Yechim

**Chain of Responsibility siz:**
\`\`\`java
// Qattiq bog'lanish - protsessor barcha ishlov beruvchilarni biladi
class RequestProcessor {
    public String process(Request request) {
        // Barcha tekshiruvlar bir joyda - o'zgartirish qiyin
        if (!request.authenticated) {	// auth tekshirish
            return "Auth failed";
        }
        if (request.data == null) {	// validatsiya tekshirish
            return "Validation failed";
        }
        log(request);	// loglash
        // Yangi tekshiruv qo'shish ushbu klassni o'zgartirishni talab qiladi!
        return "Success";
    }
}
\`\`\`

**Chain of Responsibility bilan:**
\`\`\`java
// Zaif bog'lanish - har bir ishlov beruvchi mustaqil
Handler auth = new AuthHandler();	// ishlov beruvchilarni yaratish
Handler validation = new ValidationHandler();
Handler logging = new LoggingHandler();

auth.setNext(validation).setNext(logging);	// zanjirni qurish

String result = auth.handle(request);	// qayta ishlashni boshlash
// Yangi ishlov beruvchi qo'shish = yangi klass qo'shish, zanjirga kiritish!
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Zanjir | Ishlov beruvchilar |
|------------|--------|-------------------|
| **Servlet Filters** | FilterChain | Auth, CORS, Compression filtrlari |
| **Loglash** | Logger zanjiri | Console, File, Network ishlov beruvchilari |
| **Hodisalarni Qayta Ishlash** | DOM hodisalari | Capture, Target, Bubble fazalari |
| **Middleware** | Express/Koa | Auth, Parsing, Error middleware |
| **Qo'llab-quvvatlash Tizimi** | Tiket eskalatsiyasi | L1, L2, L3 qo'llab-quvvatlash |

---

### Prodakshen Pattern: HTTP Middleware Pipeline

\`\`\`java
// HTTP qayta ishlash uchun middleware interfeysi
interface Middleware {	// middleware shartnomasi
    Response handle(HttpRequest request, MiddlewareChain chain);	// so'rovni qayta ishlash
}

class MiddlewareChain {	// middleware bajarilishini boshqaradi
    private final List<Middleware> middlewares;	// middleware ro'yxati
    private int index = 0;	// zanjirdagi joriy pozitsiya

    public MiddlewareChain(List<Middleware> middlewares) {	// konstruktor
        this.middlewares = middlewares;	// ro'yxatni saqlash
    }

    public Response proceed(HttpRequest request) {	// keyingi middleware ga o'tish
        if (index < middlewares.size()) {	// agar ko'proq middleware bo'lsa
            Middleware current = middlewares.get(index++);	// olish va oldinga siljish
            return current.handle(request, this);	// middleware ni bajarish
        }
        return new Response(200, "OK");	// zanjir oxiri - muvaffaqiyat
    }
}

class AuthMiddleware implements Middleware {	// autentifikatsiya middleware
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String token = request.getHeader("Authorization");	// auth tokenni olish
        if (token == null || !validateToken(token)) {	// tokenni tekshirish
            return new Response(401, "Unauthorized");	// yaroqsiz bo'lsa rad etish
        }
        request.setAttribute("user", extractUser(token));	// foydalanuvchini qo'shish
        return chain.proceed(request);	// zanjirni davom ettirish
    }
}

class RateLimitMiddleware implements Middleware {	// tezlikni cheklash middleware
    private final RateLimiter limiter;	// tezlik cheklagich nusxasi

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        String clientId = request.getClientId();	// mijoz identifikatorini olish
        if (!limiter.allowRequest(clientId)) {	// tezlik chegarasini tekshirish
            return new Response(429, "Too Many Requests");	// oshgan bo'lsa rad etish
        }
        return chain.proceed(request);	// zanjirni davom ettirish
    }
}

class LoggingMiddleware implements Middleware {	// loglash middleware
    private final Logger logger;	// logger nusxasi

    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        long start = System.currentTimeMillis();	// boshlanish vaqtini yozish
        logger.info("Request: " + request.getPath());	// so'rovni loglash

        Response response = chain.proceed(request);	// zanjirni davom ettirish

        long duration = System.currentTimeMillis() - start;	// davomiylikni hisoblash
        logger.info("Response: " + response.getStatus() + " in " + duration + "ms");	// javobni loglash
        return response;	// javobni qaytarish
    }
}

class ErrorHandlingMiddleware implements Middleware {	// xatolarni qayta ishlash middleware
    @Override
    public Response handle(HttpRequest request, MiddlewareChain chain) {
        try {
            return chain.proceed(request);	// davom etishga urinish
        } catch (Exception e) {	// har qanday istisnoni ushlash
            logger.error("Error processing request", e);	// xatoni loglash
            return new Response(500, "Internal Server Error");	// xato javobini qaytarish
        }
    }
}

// Foydalanish - quvurni qurish va ishlatish:
List<Middleware> middlewares = Arrays.asList(	// middleware ro'yxatini yaratish
    new ErrorHandlingMiddleware(),	// eng tashqi - barcha xatolarni ushlaydi
    new LoggingMiddleware(),	// barcha so'rovlarni loglaydi
    new RateLimitMiddleware(),	// tezlikni cheklash
    new AuthMiddleware()	// autentifikatsiya
);

MiddlewareChain chain = new MiddlewareChain(middlewares);	// zanjirni yaratish
Response response = chain.proceed(request);	// so'rovni qayta ishlash
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **next ni chaqirishni unutish** | So'rov kutilmaganda to'xtaydi | Ataylab to'xtatmaguningizcha har doim handleNext ni chaqiring |
| **Cheksiz sikllar** | Ishlov beruvchi o'zini chaqiradi | Ishlov beruvchilar siklik havolalar yaratmasligiga ishonch hosil qiling |
| **Tartibga bog'liqlik** | Noto'g'ri qayta ishlash tartibi | Kerakli tartibni hujjatlashtiring; builder pattern ishlating |
| **Ishlov beruvchi ushlamaydi** | So'rov o'tib ketadi | Zanjir oxirida fallback ishlov beruvchi qo'shing |
| **Qattiq bog'lanish** | Ishlov beruvchilar bir-biri haqida biladi | Ishlov beruvchilar faqat abstrakt Handler ni bilishi kerak |`
		}
	}
};

export default task;
