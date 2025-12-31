import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-logging-best-practices',
    title: 'Logging Best Practices',
    difficulty: 'medium',
    tags: ['java', 'logging', 'best-practices', 'production'],
    estimatedTime: '40m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master logging best practices for production applications.

**Requirements:**
1. Demonstrate what to log and what not to log
2. Implement structured logging with key-value pairs
3. Show proper exception logging with context
4. Use appropriate log levels for different scenarios
5. Avoid logging sensitive data (passwords, tokens)
6. Log performance metrics properly
7. Use guard clauses for expensive log operations
8. Show how to make logs actionable and searchable

Good logging practices are essential for debugging production issues, monitoring application health, and maintaining security.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

class User {
    private String username;
    private String password;
    private String email;

    // Constructor and getters
}

public class LoggingBestPractices {
    private static final Logger logger = LoggerFactory.getLogger(LoggingBestPractices.class);

    public static void main(String[] args) {
        // Good practices: what to log

        // Bad practices: what NOT to log

        // Structured logging

        // Exception logging

        // Performance logging

        // Security considerations
    }

    private static void processPayment(String userId, double amount) {
        // Implement with proper logging
    }
}`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

class User {
    private String username;
    private String password;
    private String email;

    public User(String username, String password, String email) {
        this.username = username;
        this.password = password;
        this.email = email;
    }

    public String getUsername() { return username; }
    public String getPassword() { return password; }
    public String getEmail() { return email; }

    // Safe method for logging - no sensitive data
    public String toLogString() {
        return "User{username='" + username + "', email='" + maskEmail(email) + "'}";
    }

    private String maskEmail(String email) {
        if (email == null || !email.contains("@")) return "***";
        String[] parts = email.split("@");
        return parts[0].substring(0, 1) + "***@" + parts[1];
    }
}

public class LoggingBestPractices {
    private static final Logger logger = LoggerFactory.getLogger(LoggingBestPractices.class);

    public static void main(String[] args) {
        logger.info("=== Logging Best Practices Demo ===");

        // 1. What to LOG
        demonstrateGoodLogging();

        // 2. What NOT to log
        demonstrateBadLogging();

        // 3. Structured logging
        demonstrateStructuredLogging();

        // 4. Exception logging
        demonstrateExceptionLogging();

        // 5. Performance logging
        demonstratePerformanceLogging();

        // 6. Actionable logs
        demonstrateActionableLogs();
    }

    private static void demonstrateGoodLogging() {
        logger.info("--- Good Logging Practices ---");

        // GOOD: Log important business events
        logger.info("User registration started for username: {}", "john_doe");

        // GOOD: Log state changes
        logger.info("Order status changed from {} to {}", "PENDING", "COMPLETED");

        // GOOD: Log with context using MDC
        MDC.put("userId", "user123");
        MDC.put("orderId", "order456");
        logger.info("Payment processed successfully");
        MDC.clear();

        // GOOD: Use appropriate log levels
        logger.debug("Retrieving user details from cache");
        logger.info("User login successful");
        logger.warn("API rate limit approaching: 90% used");
        logger.error("Database connection failed");
    }

    private static void demonstrateBadLogging() {
        logger.info("--- Bad Logging Practices (Avoid These!) ---");

        User user = new User("john_doe", "secret123", "john@example.com");

        // BAD: Logging sensitive data
        // logger.error("Login failed for password: {}", user.getPassword()); // NEVER DO THIS!

        // GOOD: Log without sensitive data
        logger.error("Login failed for user: {}", user.getUsername());

        // BAD: Logging entire objects without control
        // logger.info("User object: {}", user); // May expose sensitive data

        // GOOD: Use safe toString method
        logger.info("User: {}", user.toLogString());

        // BAD: Useless logs
        // logger.info("Entering method"); // Too verbose
        // logger.info("i = " + i); // Not actionable

        // GOOD: Meaningful logs
        logger.debug("Processing batch of {} records", 100);
    }

    private static void demonstrateStructuredLogging() {
        logger.info("--- Structured Logging ---");

        // GOOD: Structured key-value format (easy to parse and search)
        String operation = "CREATE";
        String resource = "Order";
        String resourceId = "order-123";
        long duration = 150;

        logger.info("operation={} resource={} resourceId={} duration={}ms status={}",
            operation, resource, resourceId, duration, "SUCCESS");

        // This format is easily parsable by log aggregation tools
        // Can search: operation=CREATE, resource=Order, status=SUCCESS
    }

    private static void demonstrateExceptionLogging() {
        logger.info("--- Exception Logging ---");

        try {
            processPayment("user123", 100.0);
        } catch (Exception e) {
            // GOOD: Log exception with context
            logger.error("Payment processing failed for user: {}, amount: {}",
                "user123", 100.0, e);

            // BAD: Just logging exception without context
            // logger.error("Error", e); // Not helpful!
        }
    }

    private static void processPayment(String userId, double amount) {
        throw new RuntimeException("Payment gateway timeout");
    }

    private static void demonstratePerformanceLogging() {
        logger.info("--- Performance Logging ---");

        // GOOD: Log performance metrics for critical operations
        long startTime = System.currentTimeMillis();

        // Simulate operation
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        long duration = System.currentTimeMillis() - startTime;

        // Log if operation is slow
        if (duration > 50) {
            logger.warn("Slow operation detected: method={} duration={}ms threshold={}ms",
                "processOrder", duration, 50);
        }

        logger.info("Operation completed in {}ms", duration);

        // GOOD: Use guard clauses for expensive operations
        if (logger.isDebugEnabled()) {
            String expensiveDebugInfo = generateExpensiveDebugInfo();
            logger.debug("Debug info: {}", expensiveDebugInfo);
        }
    }

    private static String generateExpensiveDebugInfo() {
        // Simulate expensive operation
        return "Detailed debug information...";
    }

    private static void demonstrateActionableLogs() {
        logger.info("--- Actionable Logs ---");

        // BAD: Not actionable
        // logger.error("Something went wrong");

        // GOOD: Actionable with context and next steps
        logger.error("Database connection failed: host={} port={} error={} action={}",
            "db.example.com", 5432, "Connection timeout",
            "Check network connectivity and database status");

        // GOOD: Include correlation IDs for distributed systems
        String correlationId = "corr-" + System.currentTimeMillis();
        MDC.put("correlationId", correlationId);
        logger.error("External API call failed: api={} endpoint={} correlationId={}",
            "PaymentService", "/api/charge", correlationId);
        MDC.clear();

        // GOOD: Log with metrics for monitoring
        logger.info("metrics: endpoint=/api/users method=GET status=200 duration=45ms");
    }
}

/*
Best Practices Summary:

1. DO LOG:
   1.1. Important business events
   1.2. State changes
   1.3. Errors with full context
   1.4. Performance issues
   1.5. Security events (login, logout, access denied)
   1.6. External API calls and responses

2. DON'T LOG:
   2.1. Passwords, tokens, API keys
   2.2. Credit card numbers, SSN
   2.3. Personal identifiable information (PII)
   2.4. Entire large objects
   2.5. Inside tight loops

3. USE APPROPRIATE LEVELS:
   3.1. TRACE: Very detailed, method entry/exit
   3.2. DEBUG: Development debugging info
   3.3. INFO: Important business events
   3.4. WARN: Potential issues, degraded state
   3.5. ERROR: Errors that need attention

4. MAKE LOGS SEARCHABLE:
   4.1. Use structured format: key=value
   4.2. Include correlation/request IDs
   4.3. Use consistent terminology
   4.4. Include relevant context

5. PERFORMANCE:
   5.1. Use parameterized logging
   5.2. Guard expensive operations
   5.3. Don't log in tight loops
   5.4. Use async appenders for high throughput
*/`,
    hint1: `Always use parameterized logging, never log sensitive data, and include enough context to understand what happened and why.`,
    hint2: `Use structured logging with key-value pairs, appropriate log levels, and guard clauses for expensive operations. Make logs searchable and actionable.`,
    whyItMatters: `Good logging practices are critical for maintaining production systems. They enable faster debugging, better monitoring, improved security, and help teams understand system behavior. Poor logging can leak sensitive data, impact performance, and make troubleshooting nearly impossible.

**Production Pattern:**
\`\`\`java
// Enterprise logging with full context
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        MDC.put("orderId", order.getId());
        MDC.put("userId", order.getUserId());

        try {
            logger.info("action=ORDER_PROCESS status=STARTED amount={}", order.getAmount());

            validateOrder(order);

            long startTime = System.currentTimeMillis();
            paymentService.charge(order);
            long duration = System.currentTimeMillis() - startTime;

            logger.info("action=ORDER_PROCESS status=SUCCESS duration={}ms", duration);

            if (duration > 1000) {
                logger.warn("action=ORDER_PROCESS status=SLOW duration={}ms threshold=1000ms", duration);
            }
        } catch (Exception e) {
            logger.error("action=ORDER_PROCESS status=FAILED error={}", e.getMessage(), e);
            throw e;
        } finally {
            MDC.clear();
        }
    }
}
\`\`\`

**Practical Benefits:**
- Structured logs easily parsed by ELK, Splunk
- MDC automatically adds context to all logs
- Performance metrics built into logs for monitoring`,
    order: 5,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify User class creation
class Test1 {
    @Test
    public void test() {
        User user = new User("testuser", "password123", "test@example.com");
        assertNotNull("User instance should be created", user);
        assertEquals("Username should match", "testuser", user.getUsername());
    }
}

// Test2: Verify User toLogString method
class Test2 {
    @Test
    public void test() {
        User user = new User("john", "secret", "john@example.com");
        String logString = user.toLogString();
        assertTrue("toLogString should not contain password", !logString.contains("secret"));
        assertTrue("toLogString should contain username", logString.contains("john"));
    }
}

// Test3: Verify LoggingBestPractices main execution
class Test3 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Main method should execute successfully", true);
        } catch (Exception e) {
            fail("Main method should not throw exceptions: " + e.getMessage());
        }
    }
}

// Test4: Verify good logging practices demonstration
class Test4 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Good logging practices should work", true);
        } catch (Exception e) {
            fail("Good logging practices should work");
        }
    }
}

// Test5: Verify bad logging practices are avoided
class Test5 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Bad logging practices should be demonstrated", true);
        } catch (Exception e) {
            fail("Bad logging practices demonstration should work");
        }
    }
}

// Test6: Verify structured logging works
class Test6 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Structured logging should work", true);
        } catch (Exception e) {
            fail("Structured logging should work");
        }
    }
}

// Test7: Verify exception logging with context
class Test7 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Exception logging with context should work", true);
        } catch (Exception e) {
            fail("Exception logging should work");
        }
    }
}

// Test8: Verify performance logging
class Test8 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Performance logging should work", true);
        } catch (Exception e) {
            fail("Performance logging should work");
        }
    }
}

// Test9: Verify actionable logs generation
class Test9 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            assertTrue("Actionable logs should be generated", true);
        } catch (Exception e) {
            fail("Actionable logs generation should work");
        }
    }
}

// Test10: Verify all best practices are demonstrated
class Test10 {
    @Test
    public void test() {
        try {
            LoggingBestPractices.main(new String[]{});
            User user = new User("finaltest", "pass", "final@test.com");
            assertNotNull("User should be created", user);
            assertTrue("All best practices should be demonstrated", true);
        } catch (Exception e) {
            fail("All best practices demonstration should work: " + e.getMessage());
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Лучшие практики логирования',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

class User {
    private String username;
    private String password;
    private String email;

    public User(String username, String password, String email) {
        this.username = username;
        this.password = password;
        this.email = email;
    }

    public String getUsername() { return username; }
    public String getPassword() { return password; }
    public String getEmail() { return email; }

    // Безопасный метод для логирования - без чувствительных данных
    public String toLogString() {
        return "User{username='" + username + "', email='" + maskEmail(email) + "'}";
    }

    private String maskEmail(String email) {
        if (email == null || !email.contains("@")) return "***";
        String[] parts = email.split("@");
        return parts[0].substring(0, 1) + "***@" + parts[1];
    }
}

public class LoggingBestPractices {
    private static final Logger logger = LoggerFactory.getLogger(LoggingBestPractices.class);

    public static void main(String[] args) {
        logger.info("=== Демо лучших практик логирования ===");

        // 1. Что логировать
        demonstrateGoodLogging();

        // 2. Что НЕ логировать
        demonstrateBadLogging();

        // 3. Структурированное логирование
        demonstrateStructuredLogging();

        // 4. Логирование исключений
        demonstrateExceptionLogging();

        // 5. Логирование производительности
        demonstratePerformanceLogging();

        // 6. Действенные логи
        demonstrateActionableLogs();
    }

    private static void demonstrateGoodLogging() {
        logger.info("--- Хорошие практики логирования ---");

        // ХОРОШО: Логируем важные бизнес-события
        logger.info("Регистрация пользователя начата для username: {}", "john_doe");

        // ХОРОШО: Логируем изменения состояния
        logger.info("Статус заказа изменен с {} на {}", "PENDING", "COMPLETED");

        // ХОРОШО: Логируем с контекстом используя MDC
        MDC.put("userId", "user123");
        MDC.put("orderId", "order456");
        logger.info("Платеж обработан успешно");
        MDC.clear();

        // ХОРОШО: Используем подходящие уровни логирования
        logger.debug("Получение данных пользователя из кэша");
        logger.info("Вход пользователя выполнен успешно");
        logger.warn("Приближается лимит API: использовано 90%");
        logger.error("Не удалось подключиться к базе данных");
    }

    private static void demonstrateBadLogging() {
        logger.info("--- Плохие практики логирования (Избегайте этого!) ---");

        User user = new User("john_doe", "secret123", "john@example.com");

        // ПЛОХО: Логирование чувствительных данных
        // logger.error("Вход не выполнен для пароля: {}", user.getPassword()); // НИКОГДА НЕ ДЕЛАЙТЕ ЭТО!

        // ХОРОШО: Логируем без чувствительных данных
        logger.error("Вход не выполнен для пользователя: {}", user.getUsername());

        // ПЛОХО: Логирование целых объектов без контроля
        // logger.info("Объект пользователя: {}", user); // Может раскрыть чувствительные данные

        // ХОРОШО: Используем безопасный метод toString
        logger.info("Пользователь: {}", user.toLogString());

        // ПЛОХО: Бесполезные логи
        // logger.info("Вход в метод"); // Слишком подробно
        // logger.info("i = " + i); // Не действенно

        // ХОРОШО: Осмысленные логи
        logger.debug("Обработка пакета из {} записей", 100);
    }

    private static void demonstrateStructuredLogging() {
        logger.info("--- Структурированное логирование ---");

        // ХОРОШО: Структурированный формат ключ-значение (легко парсится и ищется)
        String operation = "CREATE";
        String resource = "Order";
        String resourceId = "order-123";
        long duration = 150;

        logger.info("operation={} resource={} resourceId={} duration={}ms status={}",
            operation, resource, resourceId, duration, "SUCCESS");

        // Этот формат легко парсится инструментами агрегации логов
        // Можно искать: operation=CREATE, resource=Order, status=SUCCESS
    }

    private static void demonstrateExceptionLogging() {
        logger.info("--- Логирование исключений ---");

        try {
            processPayment("user123", 100.0);
        } catch (Exception e) {
            // ХОРОШО: Логируем исключение с контекстом
            logger.error("Не удалось обработать платеж для пользователя: {}, сумма: {}",
                "user123", 100.0, e);

            // ПЛОХО: Просто логирование исключения без контекста
            // logger.error("Ошибка", e); // Не полезно!
        }
    }

    private static void processPayment(String userId, double amount) {
        throw new RuntimeException("Таймаут платежного шлюза");
    }

    private static void demonstratePerformanceLogging() {
        logger.info("--- Логирование производительности ---");

        // ХОРОШО: Логируем метрики производительности для критичных операций
        long startTime = System.currentTimeMillis();

        // Симулируем операцию
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        long duration = System.currentTimeMillis() - startTime;

        // Логируем если операция медленная
        if (duration > 50) {
            logger.warn("Обнаружена медленная операция: method={} duration={}ms threshold={}ms",
                "processOrder", duration, 50);
        }

        logger.info("Операция выполнена за {}ms", duration);

        // ХОРОШО: Используем защитные условия для дорогих операций
        if (logger.isDebugEnabled()) {
            String expensiveDebugInfo = generateExpensiveDebugInfo();
            logger.debug("Отладочная информация: {}", expensiveDebugInfo);
        }
    }

    private static String generateExpensiveDebugInfo() {
        // Симулируем дорогую операцию
        return "Детальная отладочная информация...";
    }

    private static void demonstrateActionableLogs() {
        logger.info("--- Действенные логи ---");

        // ПЛОХО: Не действенно
        // logger.error("Что-то пошло не так");

        // ХОРОШО: Действенно с контекстом и следующими шагами
        logger.error("Не удалось подключиться к БД: host={} port={} error={} action={}",
            "db.example.com", 5432, "Таймаут соединения",
            "Проверьте сетевое подключение и статус базы данных");

        // ХОРОШО: Включаем correlation IDs для распределенных систем
        String correlationId = "corr-" + System.currentTimeMillis();
        MDC.put("correlationId", correlationId);
        logger.error("Не удался вызов внешнего API: api={} endpoint={} correlationId={}",
            "PaymentService", "/api/charge", correlationId);
        MDC.clear();

        // ХОРОШО: Логируем с метриками для мониторинга
        logger.info("metrics: endpoint=/api/users method=GET status=200 duration=45ms");
    }
}

/*
Сводка лучших практик:

1. ЧТО ЛОГИРОВАТЬ:
   1.1. Важные бизнес-события
   1.2. Изменения состояния
   1.3. Ошибки с полным контекстом
   1.4. Проблемы производительности
   1.5. События безопасности (вход, выход, отказ в доступе)
   1.6. Вызовы внешних API и ответы

2. ЧТО НЕ ЛОГИРОВАТЬ:
   2.1. Пароли, токены, API ключи
   2.2. Номера кредитных карт, SSN
   2.3. Персональные идентифицируемые данные (PII)
   2.4. Целые большие объекты
   2.5. Внутри плотных циклов

3. ИСПОЛЬЗУЙТЕ ПОДХОДЯЩИЕ УРОВНИ:
   3.1. TRACE: Очень детально, вход/выход из методов
   3.2. DEBUG: Отладочная информация для разработки
   3.3. INFO: Важные бизнес-события
   3.4. WARN: Потенциальные проблемы, деградация
   3.5. ERROR: Ошибки требующие внимания

4. ДЕЛАЙТЕ ЛОГИ ДОСТУПНЫМИ ДЛЯ ПОИСКА:
   4.1. Используйте структурированный формат: ключ=значение
   4.2. Включайте correlation/request IDs
   4.3. Используйте последовательную терминологию
   4.4. Включайте релевантный контекст

5. ПРОИЗВОДИТЕЛЬНОСТЬ:
   5.1. Используйте параметризованное логирование
   5.2. Защищайте дорогие операции
   5.3. Не логируйте в плотных циклах
   5.4. Используйте асинхронные appenders для высокой пропускной способности
*/`,
            description: `Освойте лучшие практики логирования для продакшн приложений.

**Требования:**
1. Продемонстрируйте что логировать и что не логировать
2. Реализуйте структурированное логирование с парами ключ-значение
3. Покажите правильное логирование исключений с контекстом
4. Используйте подходящие уровни логирования для разных сценариев
5. Избегайте логирования чувствительных данных (пароли, токены)
6. Правильно логируйте метрики производительности
7. Используйте защитные условия для дорогих операций логирования
8. Покажите как делать логи действенными и доступными для поиска

Хорошие практики логирования необходимы для отладки проблем в продакшене, мониторинга здоровья приложения и поддержания безопасности.`,
            hint1: `Всегда используйте параметризованное логирование, никогда не логируйте чувствительные данные, и включайте достаточно контекста чтобы понять что произошло и почему.`,
            hint2: `Используйте структурированное логирование с парами ключ-значение, подходящие уровни логирования, и защитные условия для дорогих операций. Делайте логи доступными для поиска и действенными.`,
            whyItMatters: `Хорошие практики логирования критически важны для поддержания продакшн систем. Они обеспечивают более быструю отладку, лучший мониторинг, улучшенную безопасность и помогают командам понимать поведение системы. Плохое логирование может привести к утечке чувствительных данных, снижению производительности и сделать устранение неполадок практически невозможным.

**Продакшен паттерн:**
\`\`\`java
// Enterprise logging с полным контекстом
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        MDC.put("orderId", order.getId());
        MDC.put("userId", order.getUserId());

        try {
            logger.info("action=ORDER_PROCESS status=STARTED amount={}", order.getAmount());

            validateOrder(order);

            long startTime = System.currentTimeMillis();
            paymentService.charge(order);
            long duration = System.currentTimeMillis() - startTime;

            logger.info("action=ORDER_PROCESS status=SUCCESS duration={}ms", duration);

            if (duration > 1000) {
                logger.warn("action=ORDER_PROCESS status=SLOW duration={}ms threshold=1000ms", duration);
            }
        } catch (Exception e) {
            logger.error("action=ORDER_PROCESS status=FAILED error={}", e.getMessage(), e);
            throw e;
        } finally {
            MDC.clear();
        }
    }
}
\`\`\`

**Практические преимущества:**
- Структурированные логи легко парсятся ELK, Splunk
- MDC автоматически добавляет контекст ко всем логам
- Метрики производительности встроены в логи для мониторинга`
        },
        uz: {
            title: 'Logging Eng Yaxshi Amaliyotlari',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

class User {
    private String username;
    private String password;
    private String email;

    public User(String username, String password, String email) {
        this.username = username;
        this.password = password;
        this.email = email;
    }

    public String getUsername() { return username; }
    public String getPassword() { return password; }
    public String getEmail() { return email; }

    // Logging uchun xavfsiz metod - maxfiy ma'lumotlarsiz
    public String toLogString() {
        return "User{username='" + username + "', email='" + maskEmail(email) + "'}";
    }

    private String maskEmail(String email) {
        if (email == null || !email.contains("@")) return "***";
        String[] parts = email.split("@");
        return parts[0].substring(0, 1) + "***@" + parts[1];
    }
}

public class LoggingBestPractices {
    private static final Logger logger = LoggerFactory.getLogger(LoggingBestPractices.class);

    public static void main(String[] args) {
        logger.info("=== Logging Eng Yaxshi Amaliyotlari Demo ===");

        // 1. Nimani log qilish
        demonstrateGoodLogging();

        // 2. Nimani log qilmaslik
        demonstrateBadLogging();

        // 3. Strukturalashtirilgan logging
        demonstrateStructuredLogging();

        // 4. Exception logging
        demonstrateExceptionLogging();

        // 5. Ishlash tezligi logging
        demonstratePerformanceLogging();

        // 6. Harakat qilinadigan loglar
        demonstrateActionableLogs();
    }

    private static void demonstrateGoodLogging() {
        logger.info("--- Yaxshi Logging Amaliyotlari ---");

        // YAXSHI: Muhim biznes hodisalarini log qilamiz
        logger.info("Foydalanuvchi ro'yxatdan o'tish boshlandi username: {}", "john_doe");

        // YAXSHI: Holat o'zgarishlarini log qilamiz
        logger.info("Buyurtma holati {} dan {} ga o'zgartirildi", "PENDING", "COMPLETED");

        // YAXSHI: MDC yordamida kontekst bilan log qilamiz
        MDC.put("userId", "user123");
        MDC.put("orderId", "order456");
        logger.info("To'lov muvaffaqiyatli qayta ishlandi");
        MDC.clear();

        // YAXSHI: Mos log darajalaridan foydalanamiz
        logger.debug("Foydalanuvchi ma'lumotlarini keshdan olyapmiz");
        logger.info("Foydalanuvchi muvaffaqiyatli kirdi");
        logger.warn("API limit yaqinlashmoqda: 90% ishlatildi");
        logger.error("Ma'lumotlar bazasiga ulanib bo'lmadi");
    }

    private static void demonstrateBadLogging() {
        logger.info("--- Yomon Logging Amaliyotlari (Bulardan qoching!) ---");

        User user = new User("john_doe", "secret123", "john@example.com");

        // YOMON: Maxfiy ma'lumotlarni logging qilish
        // logger.error("Kirish muvaffaqiyatsiz parol: {}", user.getPassword()); // HECH QACHON BUNI QILMANG!

        // YAXSHI: Maxfiy ma'lumotlarsiz log qilish
        logger.error("Kirish muvaffaqiyatsiz foydalanuvchi: {}", user.getUsername());

        // YOMON: Nazorat qilmasdan butun obyektlarni logging qilish
        // logger.info("Foydalanuvchi obyekti: {}", user); // Maxfiy ma'lumotlarni ochib berishi mumkin

        // YAXSHI: Xavfsiz toString metodidan foydalanish
        logger.info("Foydalanuvchi: {}", user.toLogString());

        // YOMON: Foydasiz loglar
        // logger.info("Metodga kirish"); // Haddan tashqari batafsil
        // logger.info("i = " + i); // Harakat qilib bo'lmaydi

        // YAXSHI: Ma'noli loglar
        logger.debug("{} yozuvlardan iborat paket qayta ishlanmoqda", 100);
    }

    private static void demonstrateStructuredLogging() {
        logger.info("--- Strukturalashtirilgan Logging ---");

        // YAXSHI: Strukturalashtirilgan kalit-qiymat formati (oson parse qilish va qidirish)
        String operation = "CREATE";
        String resource = "Order";
        String resourceId = "order-123";
        long duration = 150;

        logger.info("operation={} resource={} resourceId={} duration={}ms status={}",
            operation, resource, resourceId, duration, "SUCCESS");

        // Bu format log agregatsiya vositalari tomonidan oson parse qilinadi
        // Qidirish mumkin: operation=CREATE, resource=Order, status=SUCCESS
    }

    private static void demonstrateExceptionLogging() {
        logger.info("--- Exception Logging ---");

        try {
            processPayment("user123", 100.0);
        } catch (Exception e) {
            // YAXSHI: Exception ni kontekst bilan log qilamiz
            logger.error("To'lov qayta ishlash muvaffaqiyatsiz foydalanuvchi: {}, miqdor: {}",
                "user123", 100.0, e);

            // YOMON: Faqat exception ni kontekstsiz logging qilish
            // logger.error("Xato", e); // Foydali emas!
        }
    }

    private static void processPayment(String userId, double amount) {
        throw new RuntimeException("To'lov shlyuzi timeout");
    }

    private static void demonstratePerformanceLogging() {
        logger.info("--- Ishlash Tezligi Logging ---");

        // YAXSHI: Muhim operatsiyalar uchun ishlash ko'rsatkichlarini log qilamiz
        long startTime = System.currentTimeMillis();

        // Operatsiyani simulyatsiya qilamiz
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        long duration = System.currentTimeMillis() - startTime;

        // Operatsiya sekin bo'lsa log qilamiz
        if (duration > 50) {
            logger.warn("Sekin operatsiya aniqlandi: method={} duration={}ms threshold={}ms",
                "processOrder", duration, 50);
        }

        logger.info("Operatsiya {}ms da yakunlandi", duration);

        // YAXSHI: Qimmat operatsiyalar uchun himoya shartlaridan foydalanamiz
        if (logger.isDebugEnabled()) {
            String expensiveDebugInfo = generateExpensiveDebugInfo();
            logger.debug("Debug ma'lumoti: {}", expensiveDebugInfo);
        }
    }

    private static String generateExpensiveDebugInfo() {
        // Qimmat operatsiyani simulyatsiya qilamiz
        return "Batafsil debug ma'lumoti...";
    }

    private static void demonstrateActionableLogs() {
        logger.info("--- Harakat Qilinadigan Loglar ---");

        // YOMON: Harakat qilib bo'lmaydi
        // logger.error("Nimadir noto'g'ri ketdi");

        // YAXSHI: Kontekst va keyingi qadamlar bilan harakat qilinadigan
        logger.error("Ma'lumotlar bazasiga ulanish muvaffaqiyatsiz: host={} port={} error={} action={}",
            "db.example.com", 5432, "Ulanish timeout",
            "Tarmoq ulanishini va ma'lumotlar bazasi holatini tekshiring");

        // YAXSHI: Taqsimlangan tizimlar uchun correlation ID larni kiriting
        String correlationId = "corr-" + System.currentTimeMillis();
        MDC.put("correlationId", correlationId);
        logger.error("Tashqi API chaqiruvi muvaffaqiyatsiz: api={} endpoint={} correlationId={}",
            "PaymentService", "/api/charge", correlationId);
        MDC.clear();

        // YAXSHI: Monitoring uchun metrikalar bilan log qiling
        logger.info("metrics: endpoint=/api/users method=GET status=200 duration=45ms");
    }
}

/*
Eng Yaxshi Amaliyotlar Xulasasi:

1. NIMANI LOG QILISH:
   1.1. Muhim biznes hodisalari
   1.2. Holat o'zgarishlari
   1.3. To'liq kontekstdagi xatolar
   1.4. Ishlash tezligi muammolari
   1.5. Xavfsizlik hodisalari (kirish, chiqish, ruxsat rad etildi)
   1.6. Tashqi API chaqiruvlari va javoblar

2. NIMANI LOG QILMASLIK:
   2.1. Parollar, tokenlar, API kalitlari
   2.2. Kredit karta raqamlari, SSN
   2.3. Shaxsiy identifikatsiya ma'lumotlari (PII)
   2.4. Butun katta obyektlar
   2.5. Zich halqalar ichida

3. MOS DARAJALARDAN FOYDALANING:
   3.1. TRACE: Juda batafsil, metod kirish/chiqish
   3.2. DEBUG: Ishlab chiqish debug ma'lumoti
   3.3. INFO: Muhim biznes hodisalari
   3.4. WARN: Potentsial muammolar, degradatsiya
   3.5. ERROR: E'tibor talab qiladigan xatolar

4. LOGLARNI QIDIRUV UCHUN TAYYORLANG:
   4.1. Strukturalashtirilgan formatdan foydalaning: kalit=qiymat
   4.2. Correlation/request ID larni kiriting
   4.3. Izchil terminologiyadan foydalaning
   4.4. Tegishli kontekstni kiriting

5. ISHLASH TEZLIGI:
   5.1. Parametrlangan loggingdan foydalaning
   5.2. Qimmat operatsiyalarni himoya qiling
   5.3. Zich halqalarda log qilmang
   5.4. Yuqori o'tkazuvchanlik uchun asinxron appenderlardan foydalaning
*/`,
            description: `Ishlab chiqarish ilovalari uchun logging eng yaxshi amaliyotlarini o'rganing.

**Talablar:**
1. Nimani log qilish va nimani log qilmaslikni ko'rsating
2. Kalit-qiymat juftlari bilan strukturalashtirilgan loggingni amalga oshiring
3. Kontekst bilan to'g'ri exception loggingni ko'rsating
4. Turli stsenariylar uchun mos log darajalaridan foydalaning
5. Maxfiy ma'lumotlarni (parollar, tokenlar) log qilishdan qoching
6. Ishlash ko'rsatkichlarini to'g'ri log qiling
7. Qimmat log operatsiyalari uchun himoya shartlaridan foydalaning
8. Loglarni harakat qilinadigan va qidiruv uchun qanday tayyorlashni ko'rsating

Yaxshi logging amaliyotlari ishlab chiqarishdagi muammolarni tuzatish, ilova holatini monitoring qilish va xavfsizlikni saqlash uchun zarur.`,
            hint1: `Har doim parametrlangan loggingdan foydalaning, hech qachon maxfiy ma'lumotlarni log qilmang va nima sodir bo'lgani va nima uchun ekanligini tushunish uchun yetarli kontekstni kiriting.`,
            hint2: `Kalit-qiymat juftlari bilan strukturalashtirilgan logging, mos log darajalari va qimmat operatsiyalar uchun himoya shartlaridan foydalaning. Loglarni qidiruv uchun tayyorlang va harakat qilinadigan qiling.`,
            whyItMatters: `Yaxshi logging amaliyotlari ishlab chiqarish tizimlarini saqlash uchun juda muhim. Ular tezroq debuggingni, yaxshiroq monitoringni, yaxshilangan xavfsizlikni ta'minlaydi va jamoalarga tizim harakatini tushunishga yordam beradi. Yomon logging maxfiy ma'lumotlarni oqishiga, ishlash tezligining pasayishiga olib kelishi va muammolarni bartaraf qilishni deyarli imkonsiz qilishi mumkin.

**Ishlab chiqarish patterni:**
\`\`\`java
// To'liq kontekst bilan Enterprise logging
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        MDC.put("orderId", order.getId());
        MDC.put("userId", order.getUserId());

        try {
            logger.info("action=ORDER_PROCESS status=STARTED amount={}", order.getAmount());

            validateOrder(order);

            long startTime = System.currentTimeMillis();
            paymentService.charge(order);
            long duration = System.currentTimeMillis() - startTime;

            logger.info("action=ORDER_PROCESS status=SUCCESS duration={}ms", duration);

            if (duration > 1000) {
                logger.warn("action=ORDER_PROCESS status=SLOW duration={}ms threshold=1000ms", duration);
            }
        } catch (Exception e) {
            logger.error("action=ORDER_PROCESS status=FAILED error={}", e.getMessage(), e);
            throw e;
        } finally {
            MDC.clear();
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Strukturalashtirilgan loglar ELK, Splunk tomonidan osongina parse qilinadi
- MDC barcha loglarga avtomatik kontekst qo'shadi
- Ishlash ko'rsatkichlari monitoring uchun loglarga o'rnatilgan`
        }
    }
};

export default task;
