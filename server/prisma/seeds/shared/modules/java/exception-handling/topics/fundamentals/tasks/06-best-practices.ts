import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-exception-best-practices',
    title: 'Exception Handling Best Practices',
    difficulty: 'medium',
    tags: ['java', 'exceptions', 'best-practices', 'patterns', 'clean-code'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Refactor a poorly-written service class to follow exception handling best practices.

Requirements:
1. Replace generic Exception catches with specific exceptions
2. Avoid catching Throwable or Error
3. Don't use exceptions for control flow
4. Always clean up resources (use try-with-resources)
5. Add proper logging without swallowing exceptions
6. Create meaningful custom exceptions where appropriate
7. Document checked exceptions with @throws

The initial code has multiple anti-patterns. Fix them all!`,
    initialCode: `import java.io.*;
import java.util.*;

// Bad example - fix all the issues!
public class OrderService {

    public void processOrder(String orderId) {
        try {
            // Issue 1: Generic Exception catch
            // Issue 2: No resource management
            // Issue 3: Swallowing exceptions
            FileReader reader = new FileReader("orders.txt");
            BufferedReader br = new BufferedReader(reader);
            String line = br.readLine();
            System.out.println(line);
        } catch (Exception e) {
            // Silently fails!
        }
    }

    public double calculateDiscount(int itemCount) {
        // Issue 4: Using exceptions for control flow
        try {
            if (itemCount == 0) {
                throw new Exception("No items");
            }
            return itemCount > 10 ? 0.2 : 0.1;
        } catch (Exception e) {
            return 0;
        }
    }

    public void validateOrder(String data) throws Exception {
        // Issue 5: Throwing generic Exception
        // Issue 6: Poor error messages
        if (data == null) {
            throw new Exception("Bad");
        }
    }

    public static void main(String[] args) {
        OrderService service = new OrderService();
        service.processOrder("ORD-001");
        double discount = service.calculateDiscount(5);
        System.out.println("Discount: " + discount);
    }
}`,
    solutionCode: `import java.io.*;
import java.util.*;
import java.util.logging.*;

// Custom exception for business logic errors
class OrderValidationException extends Exception {
    private String orderId;
    private String validationError;

    public OrderValidationException(String orderId, String validationError) {
        super(String.format("Order validation failed for %s: %s", orderId, validationError));
        this.orderId = orderId;
        this.validationError = validationError;
    }

    public String getOrderId() { return orderId; }
    public String getValidationError() { return validationError; }
}

class OrderNotFoundException extends Exception {
    public OrderNotFoundException(String orderId) {
        super("Order not found: " + orderId);
    }
}

public class OrderService {
    private static final Logger logger = Logger.getLogger(OrderService.class.getName());

    /**
     * Process an order by reading from file system
     * @param orderId the order ID to process
     * @throws OrderNotFoundException if order file doesn't exist
     * @throws IOException if there's an error reading the file
     */
    public void processOrder(String orderId) throws OrderNotFoundException, IOException {
        // Best Practice 1: Use try-with-resources for automatic cleanup
        // Best Practice 2: Catch specific exceptions, not generic Exception
        // Best Practice 3: Document checked exceptions

        String filename = "orders/" + orderId + ".txt";

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            // Best Practice 4: Add proper logging
            logger.info("Processing order: " + orderId);

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("Order data: " + line);
            }

            logger.info("Order processed successfully: " + orderId);

        } catch (FileNotFoundException e) {
            // Best Practice 5: Catch specific exceptions and add context
            logger.warning("Order file not found: " + filename);
            throw new OrderNotFoundException(orderId);

        } catch (IOException e) {
            // Best Practice 6: Log before rethrowing
            logger.severe("Error reading order file: " + e.getMessage());
            throw e;  // Rethrow for caller to handle
        }
        // Resources automatically closed by try-with-resources
    }

    /**
     * Calculate discount based on item count
     * @param itemCount number of items in order
     * @return discount percentage (0.0 to 1.0)
     */
    public double calculateDiscount(int itemCount) {
        // Best Practice 7: Don't use exceptions for control flow
        // Use normal conditionals instead

        if (itemCount < 0) {
            // Best Practice 8: Use IllegalArgumentException for invalid arguments
            throw new IllegalArgumentException("Item count cannot be negative: " + itemCount);
        }

        if (itemCount == 0) {
            return 0.0;  // No discount for empty orders
        }

        // Business logic without exceptions
        if (itemCount > 10) {
            return 0.2;  // 20% discount
        } else if (itemCount > 5) {
            return 0.1;  // 10% discount
        } else {
            return 0.05; // 5% discount
        }
    }

    /**
     * Validate order data
     * @param orderId the order ID
     * @param data the order data
     * @throws OrderValidationException if validation fails
     */
    public void validateOrder(String orderId, String data) throws OrderValidationException {
        // Best Practice 9: Throw specific custom exceptions
        // Best Practice 10: Provide detailed error messages

        if (orderId == null || orderId.trim().isEmpty()) {
            throw new OrderValidationException(
                orderId,
                "Order ID cannot be null or empty"
            );
        }

        if (data == null) {
            throw new OrderValidationException(
                orderId,
                "Order data cannot be null"
            );
        }

        if (data.length() < 10) {
            throw new OrderValidationException(
                orderId,
                "Order data too short (minimum 10 characters required)"
            );
        }

        logger.info("Order validation passed: " + orderId);
    }

    /**
     * Demonstrates proper exception handling pattern
     */
    public void demonstrateBestPractices() {
        System.out.println("=== Exception Best Practices Demo ===\n");

        // Example 1: Proper resource management and specific exceptions
        System.out.println("1. Processing order with proper exception handling:");
        try {
            processOrder("ORD-001");
        } catch (OrderNotFoundException e) {
            // Handle specific business exception
            System.out.println("Business error: " + e.getMessage());
        } catch (IOException e) {
            // Handle technical exception
            System.out.println("Technical error: " + e.getMessage());
        }

        // Example 2: No exceptions for control flow
        System.out.println("");
        System.out.println("2. Calculating discount without exceptions:");
        try {
            double discount1 = calculateDiscount(15);
            System.out.println("Discount for 15 items: " + (discount1 * 100) + "%");

            double discount2 = calculateDiscount(0);
            System.out.println("Discount for 0 items: " + (discount2 * 100) + "%");

            // This will throw IllegalArgumentException
            double discount3 = calculateDiscount(-5);

        } catch (IllegalArgumentException e) {
            // Proper exception for invalid input
            System.out.println("Invalid input: " + e.getMessage());
        }

        // Example 3: Specific custom exceptions with context
        System.out.println("");
        System.out.println("3. Validating orders with custom exceptions:");
        try {
            validateOrder("ORD-123", "Valid order data here");
            System.out.println("Validation successful");
        } catch (OrderValidationException e) {
            System.out.println("Validation failed: " + e.getValidationError());
        }

        try {
            validateOrder("ORD-456", "Short");
        } catch (OrderValidationException e) {
            System.out.println("Validation failed: " + e.getValidationError());
        }

        System.out.println("");
        System.out.println("=== Demo Complete ===");
    }

    public static void main(String[] args) {
        OrderService service = new OrderService();
        service.demonstrateBestPractices();
    }
}`,
    hint1: `Replace catch (Exception e) with specific exception types like IOException, FileNotFoundException. Never catch Throwable or Error unless you have a very good reason.`,
    hint2: `Don't use exceptions for normal program flow. Use if-statements for validation. Only throw exceptions for truly exceptional conditions.`,
    hint3: `Always close resources using try-with-resources. Add logging before rethrowing or wrapping exceptions. Create custom exceptions for business logic errors with meaningful names and messages.`,
    whyItMatters: `Following exception handling best practices makes your code more maintainable, debuggable, and reliable. Poor exception handling is a common source of bugs, security vulnerabilities, and production issues. Learning these patterns is essential for writing professional Java code.

**Production Pattern:**
\`\`\`java
@Service
public class UserService {
    private static final Logger log = LoggerFactory.getLogger(UserService.class);

    @Transactional
    public User createUser(UserDTO dto) throws UserCreationException {
        try (Connection conn = dataSource.getConnection()) {
            validateUser(dto);
            User user = userRepository.save(dto);
            emailService.sendWelcomeEmail(user);
            auditLog.log("USER_CREATED", user.getId());
            return user;
        } catch (ValidationException e) {
            log.warn("Invalid user data: {}", e.getMessage());
            throw new UserCreationException("VALIDATION_FAILED", e);
        } catch (DuplicateEmailException e) {
            log.info("Duplicate email attempt: {}", dto.getEmail());
            throw new UserCreationException("EMAIL_EXISTS", e);
        } catch (Exception e) {
            log.error("Unexpected error creating user", e);
            alertService.sendAlert("User creation failed", e);
            throw new UserCreationException("SYSTEM_ERROR", e);
        }
    }
}
\`\`\`

**Practical Benefits:**
- Comprehensive handling of all error types
- Integration with logging and monitoring
- Proper transaction and resource management`,
    order: 6,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify OrderProcessor instantiation
class Test1 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        assertNotNull("OrderProcessor should be created", processor);
    }
}

// Test2: Verify successful order validation
class Test2 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        boolean validationPassed = false;
        try {
            processor.validateOrder("ORD-001", "Valid order data with enough length");
            validationPassed = true;
        } catch (OrderValidationException e) {
            fail("Valid order should not throw exception: " + e.getMessage());
        }
        assertTrue("Valid order should pass validation", validationPassed);
    }
}

// Test3: Verify null order ID throws exception
class Test3 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        try {
            processor.validateOrder(null, "Some data");
            fail("Should throw OrderValidationException for null ID");
        } catch (OrderValidationException e) {
            assertNotNull("Exception should have message", e.getMessage());
            assertNotNull("Should contain validation error", e.getValidationError());
            assertTrue("Validation error should describe the issue",
                e.getValidationError().length() > 0);
        }
    }
}

// Test4: Verify short data throws exception
class Test4 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        try {
            processor.validateOrder("ORD-002", "Short");
            fail("Should throw OrderValidationException for short data");
        } catch (OrderValidationException e) {
            assertTrue("Exception message should contain error details",
                e.getMessage().contains("too short"));
        }
    }
}

// Test5: demonstrateBestPractices shows demo output
class Test5 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream oldOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        OrderProcessor processor = new OrderProcessor();
        processor.demonstrateBestPractices();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show best practices demo",
            output.contains("Best Practices Demo") || output.contains("Demo Complete") ||
            output.contains("Processing order") || output.contains("Calculating discount") ||
            output.contains("Лучшие практики") || output.contains("Eng yaxshi amaliyotlar"));
    }
}

// Test6: main produces expected output
class Test6 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream oldOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        OrderProcessor.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Main should produce output with demo sections",
            output.contains("1.") || output.contains("2.") || output.contains("3.") ||
            output.contains("===") || output.contains("discount") || output.contains("Validation"));
    }
}

// Test7: Verify custom exception contains order ID
class Test7 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        try {
            processor.validateOrder("ORD-007", "Bad");
            fail("Should throw exception");
        } catch (OrderValidationException e) {
            assertEquals("Order ID should match", "ORD-007", e.getOrderId());
        }
    }
}

// Test8: Verify validation error message
class Test8 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        try {
            processor.validateOrder("ORD-008", "");
            fail("Should throw exception for empty data");
        } catch (OrderValidationException e) {
            assertNotNull("Validation error should exist", e.getValidationError());
            assertTrue("Error message should be meaningful",
                e.getValidationError().length() > 0);
        }
    }
}

// Test9: Verify multiple validations
class Test9 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        int validationCount = 0;
        try {
            processor.validateOrder("ORD-009", "Valid data here");
            validationCount++;
            processor.validateOrder("ORD-010", "Another valid order data");
            validationCount++;
        } catch (OrderValidationException e) {
            fail("Valid orders should not fail: " + e.getMessage());
        }
        assertEquals("Both validations should complete", 2, validationCount);
    }
}

// Test10: Verify exception best practices
class Test10 {
    @Test
    public void test() {
        OrderProcessor processor = new OrderProcessor();
        try {
            processor.validateOrder(null, null);
            fail("Should throw exception");
        } catch (OrderValidationException e) {
            assertTrue("Exception should be specific type", e instanceof OrderValidationException);
            assertNotNull("Exception message should exist", e.getMessage());
            assertNotNull("Validation error should exist", e.getValidationError());
            assertNotNull("Order ID should be stored", e.getOrderId());
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Лучшие Практики Обработки Исключений',
            solutionCode: `import java.io.*;
import java.util.*;
import java.util.logging.*;

// Пользовательское исключение для ошибок бизнес-логики
class OrderValidationException extends Exception {
    private String orderId;
    private String validationError;

    public OrderValidationException(String orderId, String validationError) {
        super(String.format("Ошибка валидации заказа %s: %s", orderId, validationError));
        this.orderId = orderId;
        this.validationError = validationError;
    }

    public String getOrderId() { return orderId; }
    public String getValidationError() { return validationError; }
}

class OrderNotFoundException extends Exception {
    public OrderNotFoundException(String orderId) {
        super("Заказ не найден: " + orderId);
    }
}

public class OrderService {
    private static final Logger logger = Logger.getLogger(OrderService.class.getName());

    /**
     * Обработка заказа путем чтения из файловой системы
     * @param orderId идентификатор заказа для обработки
     * @throws OrderNotFoundException если файл заказа не существует
     * @throws IOException если возникла ошибка чтения файла
     */
    public void processOrder(String orderId) throws OrderNotFoundException, IOException {
        // Лучшая практика 1: Использование try-with-resources для автоматической очистки
        // Лучшая практика 2: Ловить конкретные исключения, а не общий Exception
        // Лучшая практика 3: Документировать проверяемые исключения

        String filename = "orders/" + orderId + ".txt";

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            // Лучшая практика 4: Добавить правильное логирование
            logger.info("Обработка заказа: " + orderId);

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("Данные заказа: " + line);
            }

            logger.info("Заказ успешно обработан: " + orderId);

        } catch (FileNotFoundException e) {
            // Лучшая практика 5: Ловить конкретные исключения и добавлять контекст
            logger.warning("Файл заказа не найден: " + filename);
            throw new OrderNotFoundException(orderId);

        } catch (IOException e) {
            // Лучшая практика 6: Логировать перед повторным выбросом
            logger.severe("Ошибка чтения файла заказа: " + e.getMessage());
            throw e;  // Повторный выброс для обработки вызывающим кодом
        }
        // Ресурсы автоматически закрываются try-with-resources
    }

    /**
     * Расчет скидки на основе количества товаров
     * @param itemCount количество товаров в заказе
     * @return процент скидки (от 0.0 до 1.0)
     */
    public double calculateDiscount(int itemCount) {
        // Лучшая практика 7: Не использовать исключения для управления потоком
        // Вместо этого использовать обычные условия

        if (itemCount < 0) {
            // Лучшая практика 8: Использовать IllegalArgumentException для неверных аргументов
            throw new IllegalArgumentException("Количество товаров не может быть отрицательным: " + itemCount);
        }

        if (itemCount == 0) {
            return 0.0;  // Нет скидки для пустых заказов
        }

        // Бизнес-логика без исключений
        if (itemCount > 10) {
            return 0.2;  // Скидка 20%
        } else if (itemCount > 5) {
            return 0.1;  // Скидка 10%
        } else {
            return 0.05; // Скидка 5%
        }
    }

    /**
     * Валидация данных заказа
     * @param orderId идентификатор заказа
     * @param data данные заказа
     * @throws OrderValidationException если валидация не прошла
     */
    public void validateOrder(String orderId, String data) throws OrderValidationException {
        // Лучшая практика 9: Выбрасывать конкретные пользовательские исключения
        // Лучшая практика 10: Предоставлять детальные сообщения об ошибках

        if (orderId == null || orderId.trim().isEmpty()) {
            throw new OrderValidationException(
                orderId,
                "Идентификатор заказа не может быть null или пустым"
            );
        }

        if (data == null) {
            throw new OrderValidationException(
                orderId,
                "Данные заказа не могут быть null"
            );
        }

        if (data.length() < 10) {
            throw new OrderValidationException(
                orderId,
                "Данные заказа слишком короткие (требуется минимум 10 символов)"
            );
        }

        logger.info("Валидация заказа пройдена: " + orderId);
    }

    /**
     * Демонстрация паттерна правильной обработки исключений
     */
    public void demonstrateBestPractices() {
        System.out.println("=== Демонстрация Лучших Практик Исключений ===\n");

        // Пример 1: Правильное управление ресурсами и конкретные исключения
        System.out.println("1. Обработка заказа с правильной обработкой исключений:");
        try {
            processOrder("ORD-001");
        } catch (OrderNotFoundException e) {
            // Обработка конкретного бизнес-исключения
            System.out.println("Бизнес-ошибка: " + e.getMessage());
        } catch (IOException e) {
            // Обработка технического исключения
            System.out.println("Техническая ошибка: " + e.getMessage());
        }

        // Пример 2: Никаких исключений для управления потоком
        System.out.println("");
        System.out.println("2. Расчет скидки без исключений:");
        try {
            double discount1 = calculateDiscount(15);
            System.out.println("Скидка на 15 товаров: " + (discount1 * 100) + "%");

            double discount2 = calculateDiscount(0);
            System.out.println("Скидка на 0 товаров: " + (discount2 * 100) + "%");

            // Это выбросит IllegalArgumentException
            double discount3 = calculateDiscount(-5);

        } catch (IllegalArgumentException e) {
            // Правильное исключение для неверного ввода
            System.out.println("Неверный ввод: " + e.getMessage());
        }

        // Пример 3: Конкретные пользовательские исключения с контекстом
        System.out.println("");
        System.out.println("3. Валидация заказов с пользовательскими исключениями:");
        try {
            validateOrder("ORD-123", "Валидные данные заказа здесь");
            System.out.println("Валидация успешна");
        } catch (OrderValidationException e) {
            System.out.println("Валидация не прошла: " + e.getValidationError());
        }

        try {
            validateOrder("ORD-456", "Короткий");
        } catch (OrderValidationException e) {
            System.out.println("Валидация не прошла: " + e.getValidationError());
        }

        System.out.println("");
        System.out.println("=== Демонстрация Завершена ===");
    }

    public static void main(String[] args) {
        OrderService service = new OrderService();
        service.demonstrateBestPractices();
    }
}`,
            description: `Выполните рефакторинг плохо написанного класса сервиса для соответствия лучшим практикам обработки исключений.

Требования:
1. Замените общие перехваты Exception на конкретные исключения
2. Избегайте перехвата Throwable или Error
3. Не используйте исключения для управления потоком
4. Всегда очищайте ресурсы (используйте try-with-resources)
5. Добавьте правильное логирование без подавления исключений
6. Создайте осмысленные пользовательские исключения где необходимо
7. Документируйте проверяемые исключения с помощью @throws

Исходный код содержит множество анти-паттернов. Исправьте их все!`,
            hint1: `Замените catch (Exception e) на конкретные типы исключений, такие как IOException, FileNotFoundException. Никогда не ловите Throwable или Error, если у вас нет очень веской причины.`,
            hint2: `Не используйте исключения для нормального потока программы. Используйте if-операторы для валидации. Выбрасывайте исключения только для действительно исключительных условий.`,
            hint3: `Всегда закрывайте ресурсы, используя try-with-resources. Добавляйте логирование перед повторным выбросом или оборачиванием исключений. Создавайте пользовательские исключения для ошибок бизнес-логики со значимыми именами и сообщениями.`,
            whyItMatters: `Следование лучшим практикам обработки исключений делает ваш код более поддерживаемым, отлаживаемым и надежным. Плохая обработка исключений - это распространенный источник ошибок, уязвимостей безопасности и проблем в продакшене. Изучение этих паттернов необходимо для написания профессионального Java-кода.

**Продакшен паттерн:**
\`\`\`java
@Service
public class UserService {
    private static final Logger log = LoggerFactory.getLogger(UserService.class);

    @Transactional
    public User createUser(UserDTO dto) throws UserCreationException {
        try (Connection conn = dataSource.getConnection()) {
            validateUser(dto);
            User user = userRepository.save(dto);
            emailService.sendWelcomeEmail(user);
            auditLog.log("USER_CREATED", user.getId());
            return user;
        } catch (ValidationException e) {
            log.warn("Invalid user data: {}", e.getMessage());
            throw new UserCreationException("VALIDATION_FAILED", e);
        } catch (DuplicateEmailException e) {
            log.info("Duplicate email attempt: {}", dto.getEmail());
            throw new UserCreationException("EMAIL_EXISTS", e);
        } catch (Exception e) {
            log.error("Unexpected error creating user", e);
            alertService.sendAlert("User creation failed", e);
            throw new UserCreationException("SYSTEM_ERROR", e);
        }
    }
}
\`\`\`

**Практические преимущества:**
- Комплексная обработка всех типов ошибок
- Интеграция с логированием и мониторингом
- Правильное управление транзакциями и ресурсами`
        },
        uz: {
            title: `Istisnolarni Qayta Ishlashning Eng Yaxshi Amaliyotlari`,
            solutionCode: `import java.io.*;
import java.util.*;
import java.util.logging.*;

// Biznes mantiq xatolari uchun maxsus istisno
class OrderValidationException extends Exception {
    private String orderId;
    private String validationError;

    public OrderValidationException(String orderId, String validationError) {
        super(String.format("%s buyurtmasini tekshirish xatosi: %s", orderId, validationError));
        this.orderId = orderId;
        this.validationError = validationError;
    }

    public String getOrderId() { return orderId; }
    public String getValidationError() { return validationError; }
}

class OrderNotFoundException extends Exception {
    public OrderNotFoundException(String orderId) {
        super("Buyurtma topilmadi: " + orderId);
    }
}

public class OrderService {
    private static final Logger logger = Logger.getLogger(OrderService.class.getName());

    /**
     * Fayl tizimidan o'qish orqali buyurtmani qayta ishlash
     * @param orderId qayta ishlash uchun buyurtma identifikatori
     * @throws OrderNotFoundException agar buyurtma fayli mavjud bo'lmasa
     * @throws IOException agar faylni o'qishda xato yuz bersa
     */
    public void processOrder(String orderId) throws OrderNotFoundException, IOException {
        // Eng yaxshi amaliyot 1: Avtomatik tozalash uchun try-with-resources dan foydalanish
        // Eng yaxshi amaliyot 2: Umumiy Exception emas, aniq istisnolarni ushlash
        // Eng yaxshi amaliyot 3: Tekshiriladigan istisnolarni hujjatlashtirish

        String filename = "orders/" + orderId + ".txt";

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            // Eng yaxshi amaliyot 4: To'g'ri jurnallashtirish qo'shish
            logger.info("Buyurtmani qayta ishlash: " + orderId);

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("Buyurtma ma'lumotlari: " + line);
            }

            logger.info("Buyurtma muvaffaqiyatli qayta ishlandi: " + orderId);

        } catch (FileNotFoundException e) {
            // Eng yaxshi amaliyot 5: Aniq istisnolarni ushlash va kontekst qo'shish
            logger.warning("Buyurtma fayli topilmadi: " + filename);
            throw new OrderNotFoundException(orderId);

        } catch (IOException e) {
            // Eng yaxshi amaliyot 6: Qayta tashlashdan oldin jurnallashtirish
            logger.severe("Buyurtma faylini o'qishda xato: " + e.getMessage());
            throw e;  // Chaqiruvchi tomonidan qayta ishlash uchun qayta tashlash
        }
        // Resurslar try-with-resources tomonidan avtomatik yopiladi
    }

    /**
     * Mahsulotlar soni asosida chegirma hisoblash
     * @param itemCount buyurtmadagi mahsulotlar soni
     * @return chegirma foizi (0.0 dan 1.0 gacha)
     */
    public double calculateDiscount(int itemCount) {
        // Eng yaxshi amaliyot 7: Boshqaruv oqimi uchun istisnolardan foydalanmaslik
        // O'rniga oddiy shartlardan foydalanish

        if (itemCount < 0) {
            // Eng yaxshi amaliyot 8: Noto'g'ri argumentlar uchun IllegalArgumentException dan foydalanish
            throw new IllegalArgumentException("Mahsulotlar soni manfiy bo'lishi mumkin emas: " + itemCount);
        }

        if (itemCount == 0) {
            return 0.0;  // Bo'sh buyurtmalar uchun chegirma yo'q
        }

        // Istisnolarsiz biznes mantiq
        if (itemCount > 10) {
            return 0.2;  // 20% chegirma
        } else if (itemCount > 5) {
            return 0.1;  // 10% chegirma
        } else {
            return 0.05; // 5% chegirma
        }
    }

    /**
     * Buyurtma ma'lumotlarini tekshirish
     * @param orderId buyurtma identifikatori
     * @param data buyurtma ma'lumotlari
     * @throws OrderValidationException agar tekshirish muvaffaqiyatsiz bo'lsa
     */
    public void validateOrder(String orderId, String data) throws OrderValidationException {
        // Eng yaxshi amaliyot 9: Aniq maxsus istisnolarni tashlash
        // Eng yaxshi amaliyot 10: Batafsil xato xabarlarini taqdim etish

        if (orderId == null || orderId.trim().isEmpty()) {
            throw new OrderValidationException(
                orderId,
                "Buyurtma identifikatori null yoki bo'sh bo'lishi mumkin emas"
            );
        }

        if (data == null) {
            throw new OrderValidationException(
                orderId,
                "Buyurtma ma'lumotlari null bo'lishi mumkin emas"
            );
        }

        if (data.length() < 10) {
            throw new OrderValidationException(
                orderId,
                "Buyurtma ma'lumotlari juda qisqa (kamida 10 ta belgi talab qilinadi)"
            );
        }

        logger.info("Buyurtma tekshiruvi o'tdi: " + orderId);
    }

    /**
     * To'g'ri istisno qayta ishlash naqshini ko'rsatish
     */
    public void demonstrateBestPractices() {
        System.out.println("=== Istisnolar Eng Yaxshi Amaliyotlari Namoyishi ===\n");

        // Misol 1: To'g'ri resurs boshqaruvi va aniq istisnolar
        System.out.println("1. To'g'ri istisno qayta ishlash bilan buyurtmani qayta ishlash:");
        try {
            processOrder("ORD-001");
        } catch (OrderNotFoundException e) {
            // Aniq biznes istisnoni qayta ishlash
            System.out.println("Biznes xatosi: " + e.getMessage());
        } catch (IOException e) {
            // Texnik istisnoni qayta ishlash
            System.out.println("Texnik xato: " + e.getMessage());
        }

        // Misol 2: Boshqaruv oqimi uchun istisnolar yo'q
        System.out.println("");
        System.out.println("2. Istisnolarsiz chegirmani hisoblash:");
        try {
            double discount1 = calculateDiscount(15);
            System.out.println("15 mahsulot uchun chegirma: " + (discount1 * 100) + "%");

            double discount2 = calculateDiscount(0);
            System.out.println("0 mahsulot uchun chegirma: " + (discount2 * 100) + "%");

            // Bu IllegalArgumentException tashlaydi
            double discount3 = calculateDiscount(-5);

        } catch (IllegalArgumentException e) {
            // Noto'g'ri kirish uchun to'g'ri istisno
            System.out.println("Noto'g'ri kirish: " + e.getMessage());
        }

        // Misol 3: Kontekst bilan aniq maxsus istisnolar
        System.out.println("");
        System.out.println("3. Maxsus istisnolar bilan buyurtmalarni tekshirish:");
        try {
            validateOrder("ORD-123", "Bu yerda to'g'ri buyurtma ma'lumotlari");
            System.out.println("Tekshirish muvaffaqiyatli");
        } catch (OrderValidationException e) {
            System.out.println("Tekshirish muvaffaqiyatsiz: " + e.getValidationError());
        }

        try {
            validateOrder("ORD-456", "Qisqa");
        } catch (OrderValidationException e) {
            System.out.println("Tekshirish muvaffaqiyatsiz: " + e.getValidationError());
        }

        System.out.println("");
        System.out.println("=== Namoyish Tugadi ===");
    }

    public static void main(String[] args) {
        OrderService service = new OrderService();
        service.demonstrateBestPractices();
    }
}`,
            description: `Istisnolarni qayta ishlashning eng yaxshi amaliyotlariga rioya qilish uchun yomon yozilgan xizmat klassini qayta ishlang.

Talablar:
1. Umumiy Exception ushlashlarini aniq istisnolar bilan almashtiring
2. Throwable yoki Error ni ushlashdan saqlaning
3. Boshqaruv oqimi uchun istisnolardan foydalanmang
4. Har doim resurslarni tozalang (try-with-resources dan foydalaning)
5. Istisnolarni bostirmasdan to'g'ri jurnallashtirish qo'shing
6. Kerak bo'lganda mazmunli maxsus istisnolar yarating
7. @throws bilan tekshiriladigan istisnolarni hujjatlashtiring

Boshlang'ich kod ko'plab anti-naqshlarni o'z ichiga oladi. Barchasini tuzating!`,
            hint1: `catch (Exception e) ni IOException, FileNotFoundException kabi aniq istisno turlari bilan almashtiring. Juda yaxshi sabab bo'lmasa, hech qachon Throwable yoki Error ni ushlamang.`,
            hint2: `Oddiy dastur oqimi uchun istisnolardan foydalanmang. Tekshirish uchun if-operatorlaridan foydalaning. Faqat haqiqatan ham istisno vaziyatlar uchun istisnolarni tashlang.`,
            hint3: `Har doim try-with-resources dan foydalanib resurslarni yoping. Istisnolarni qayta tashlash yoki o'rashdan oldin jurnallashtirish qo'shing. Mazmunli nomlar va xabarlar bilan biznes mantiq xatolari uchun maxsus istisnolar yarating.`,
            whyItMatters: `Istisnolarni qayta ishlashning eng yaxshi amaliyotlariga rioya qilish kodingizni yanada saqlash, disk raskadrovka qilish va ishonchli qiladi. Yomon istisno qayta ishlash xatolar, xavfsizlik zaiflik va ishlab chiqarish muammolarining umumiy manbai hisoblanadi. Ushbu naqshlarni o'rganish professional Java kodi yozish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class UserService {
    private static final Logger log = LoggerFactory.getLogger(UserService.class);

    @Transactional
    public User createUser(UserDTO dto) throws UserCreationException {
        try (Connection conn = dataSource.getConnection()) {
            validateUser(dto);
            User user = userRepository.save(dto);
            emailService.sendWelcomeEmail(user);
            auditLog.log("USER_CREATED", user.getId());
            return user;
        } catch (ValidationException e) {
            log.warn("Noto'g'ri foydalanuvchi ma'lumotlari: {}", e.getMessage());
            throw new UserCreationException("VALIDATION_FAILED", e);
        } catch (DuplicateEmailException e) {
            log.info("Dublikat email urinishi: {}", dto.getEmail());
            throw new UserCreationException("EMAIL_EXISTS", e);
        } catch (Exception e) {
            log.error("Foydalanuvchi yaratishda kutilmagan xato", e);
            alertService.sendAlert("Foydalanuvchi yaratish muvaffaqiyatsiz", e);
            throw new UserCreationException("SYSTEM_ERROR", e);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Barcha xato turlarini kompleks qayta ishlash
- Jurnallashtirish va monitoring bilan integratsiya
- Tranzaksiyalar va resurslarni to'g'ri boshqarish`
        }
    }
};

export default task;
