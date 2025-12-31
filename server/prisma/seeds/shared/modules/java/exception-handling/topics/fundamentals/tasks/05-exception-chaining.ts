import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-exception-chaining',
    title: 'Exception Chaining and Wrapping',
    difficulty: 'medium',
    tags: ['java', 'exceptions', 'chaining', 'stack-trace', 'wrapping'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement a data access layer that demonstrates exception chaining to preserve stack traces and add context.

Requirements:
1. Create DataAccessException as a custom exception
2. Implement UserRepository with methods that catch low-level exceptions and wrap them
3. Use exception chaining to preserve the original cause
4. Add contextual information when rethrowing exceptions
5. Demonstrate getCause() and printStackTrace() to show the full chain

Example:
\`\`\`java
UserRepository repo = new UserRepository();
repo.findUserById(999);  // Wraps SQLException with context
// Full stack trace shows both exceptions
\`\`\``,
    initialCode: `import java.sql.*;

// TODO: Create DataAccessException class with support for cause

class User {
    private int id;
    private String name;

    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "User{id=" + id + ", name='" + name + "'}";
    }
}

public class UserRepository {

    public User findUserById(int userId) throws DataAccessException {
        // TODO: Simulate database access and wrap exceptions
        return null;
    }

    public void saveUser(User user) throws DataAccessException {
        // TODO: Simulate database save and wrap exceptions
    }

    public static void main(String[] args) {
        UserRepository repo = new UserRepository();

        try {
            User user = repo.findUserById(999);
            System.out.println("Found: " + user);
        } catch (DataAccessException e) {
            System.out.println("Error: " + e.getMessage());
            System.out.println("Caused by: " + e.getCause());
        }
    }
}`,
    solutionCode: `import java.sql.*;

// Custom exception that supports chaining
class DataAccessException extends Exception {
    private String operation;
    private String entity;

    // Constructor with message and cause
    public DataAccessException(String message, Throwable cause) {
        super(message, cause);
    }

    // Constructor with detailed context
    public DataAccessException(String operation, String entity, Throwable cause) {
        super(String.format("Data access failed: %s on %s", operation, entity), cause);
        this.operation = operation;
        this.entity = entity;
    }

    public String getOperation() { return operation; }
    public String getEntity() { return entity; }
}

class User {
    private int id;
    private String name;

    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "User{id=" + id + ", name='" + name + "'}";
    }
}

public class UserRepository {

    public User findUserById(int userId) throws DataAccessException {
        try {
            // Simulate database connection
            Connection conn = simulateConnection();

            // Simulate query execution (this will throw SQLException)
            String query = "SELECT * FROM users WHERE id = " + userId;
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(query);

            if (rs.next()) {
                return new User(rs.getInt("id"), rs.getString("name"));
            }
            return null;

        } catch (SQLException e) {
            // Wrap the SQLException with context
            // Original exception is preserved as the cause
            throw new DataAccessException(
                "findUserById",
                "User[id=" + userId + "]",
                e  // Chain the original exception
            );
        }
    }

    public void saveUser(User user) throws DataAccessException {
        try {
            // Simulate database save
            Connection conn = simulateConnection();
            String query = "INSERT INTO users VALUES (?, ?)";
            PreparedStatement stmt = conn.prepareStatement(query);
            // This will throw SQLException
            stmt.execute();

        } catch (SQLException e) {
            // Wrap and add context
            throw new DataAccessException(
                "saveUser",
                "User[" + user + "]",
                e
            );
        }
    }

    public void demonstrateExceptionChaining() {
        System.out.println("=== Exception Chaining Demo ===\n");

        try {
            // This will fail and throw chained exception
            User user = findUserById(999);

        } catch (DataAccessException e) {
            System.out.println("Top-level exception:");
            System.out.println("  Message: " + e.getMessage());
            System.out.println("  Operation: " + e.getOperation());
            System.out.println("  Entity: " + e.getEntity());

            // Access the cause (original exception)
            Throwable cause = e.getCause();
            if (cause != null) {
                System.out.println("\nOriginal cause:");
                System.out.println("  Type: " + cause.getClass().getSimpleName());
                System.out.println("  Message: " + cause.getMessage());
            }

            // Print full stack trace showing chain
            System.out.println("\nFull stack trace:");
            e.printStackTrace();
        }
    }

    // Simulate database connection that throws SQLException
    private Connection simulateConnection() throws SQLException {
        throw new SQLException("Unable to connect to database at localhost:5432");
    }

    public static void main(String[] args) {
        UserRepository repo = new UserRepository();

        System.out.println("=== Example 1: Simple Error Handling ===");
        try {
            User user = repo.findUserById(123);
            System.out.println("Found: " + user);
        } catch (DataAccessException e) {
            System.out.println("Error: " + e.getMessage());
            System.out.println("Root cause: " + e.getCause().getMessage());
        }

        System.out.println("\n=== Example 2: Detailed Exception Chain ===");
        repo.demonstrateExceptionChaining();

        System.out.println("\n=== Example 3: Multiple Operations ===");
        try {
            // Both operations will fail
            User user1 = repo.findUserById(1);
            User user2 = new User(100, "John");
            repo.saveUser(user2);
        } catch (DataAccessException e) {
            System.out.println("Operation failed: " + e.getOperation());
            System.out.println("On entity: " + e.getEntity());
            System.out.println("Due to: " + e.getCause().getClass().getSimpleName());
        }
    }
}`,
    hint1: `Use the constructor super(message, cause) to chain exceptions. The cause parameter preserves the original exception. You can retrieve it later with getCause().`,
    hint2: `When wrapping exceptions, add contextual information in your custom exception class. Include details like the operation name, affected entity, and any relevant parameters.`,
    whyItMatters: `Exception chaining is essential for debugging complex applications. It preserves the full error context from low-level operations while adding high-level business context. This helps developers quickly identify root causes without losing important diagnostic information.

**Production Pattern:**
\`\`\`java
public class OrderService {
    public Order createOrder(OrderRequest req) throws OrderException {
        try {
            validateInventory(req.getItems());
            Payment payment = processPayment(req);
            return saveOrder(req, payment);
        } catch (InventoryException e) {
            logger.error("Inventory check failed for order", e);
            throw new OrderException("OUT_OF_STOCK",
                "Order creation failed: items unavailable", e);
        } catch (PaymentException e) {
            logger.error("Payment processing failed", e);
            metrics.incrementCounter("order.payment_failed");
            throw new OrderException("PAYMENT_FAILED",
                "Order creation failed: payment declined", e);
        }
    }
}
\`\`\`

**Practical Benefits:**
- Preserving full stack trace
- Adding business context to technical errors
- Simplified problem diagnosis in production`,
    order: 5,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify UserRepository instantiation
class Test1 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        assertNotNull("UserRepository should be created", repo);
    }
}

// Test2: Verify DataAccessException is thrown
class Test2 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.findUserById(1);
            fail("Should throw DataAccessException");
        } catch (DataAccessException e) {
            assertNotNull("Exception should have a message", e.getMessage());
            assertNotNull("Exception should have a cause", e.getCause());
        } catch (Exception e) {
            fail("Wrong exception type: " + e.getMessage());
        }
    }
}

// Test3: Verify exception chaining
class Test3 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.findUserById(1);
            fail("Should throw DataAccessException");
        } catch (DataAccessException e) {
            assertTrue("Should have SQLException as cause",
                e.getCause() instanceof java.sql.SQLException);
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}

// Test4: Verify exception context information
class Test4 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.findUserById(123);
            fail("Should throw DataAccessException");
        } catch (DataAccessException e) {
            assertTrue("Exception message should contain context",
                e.getMessage().contains("findUserById"));
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}

// Test5: Verify saveUser throws exception
class Test5 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        User user = new User(1, "TestUser");
        try {
            repo.saveUser(user);
            fail("Should throw DataAccessException");
        } catch (DataAccessException e) {
            assertNotNull("Exception should be thrown", e);
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}

// Test6: Verify exception chaining demonstration
class Test6 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.demonstrateExceptionChaining();
            assertTrue("Demonstration should complete", true);
        } catch (Exception e) {
            fail("Demonstration should not throw: " + e.getMessage());
        }
    }
}

// Test7: Verify main method execution
class Test7 {
    @Test
    public void test() {
        try {
            UserRepository.main(new String[]{});
            assertTrue("Main method should execute", true);
        } catch (Exception e) {
            fail("Main should not throw: " + e.getMessage());
        }
    }
}

// Test8: Verify User class creation
class Test8 {
    @Test
    public void test() {
        User user = new User(1, "TestUser");
        assertNotNull("User should be created", user);
        assertEquals("User ID should match", 1, user.getId());
        assertEquals("User name should match", "TestUser", user.getName());
    }
}

// Test9: Verify getCause returns original exception
class Test9 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.findUserById(999);
            fail("Should throw DataAccessException");
        } catch (DataAccessException e) {
            Throwable cause = e.getCause();
            assertNotNull("Cause should not be null", cause);
            assertTrue("Cause should be SQLException",
                cause.getClass().getSimpleName().contains("SQLException"));
        } catch (Exception e) {
            fail("Wrong exception: " + e.getMessage());
        }
    }
}

// Test10: Verify full exception chain
class Test10 {
    @Test
    public void test() {
        UserRepository repo = new UserRepository();
        try {
            repo.findUserById(1);
            repo.saveUser(new User(1, "Test"));
            fail("Should throw exceptions");
        } catch (DataAccessException e) {
            assertNotNull("Exception message should exist", e.getMessage());
            assertNotNull("Exception cause should exist", e.getCause());
            assertTrue("Should preserve exception chain", e.getCause() != null);
        } catch (Exception e) {
            fail("Unexpected exception: " + e.getMessage());
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Цепочка и Обертывание Исключений',
            solutionCode: `import java.sql.*;

// Пользовательское исключение с поддержкой цепочки
class DataAccessException extends Exception {
    private String operation;
    private String entity;

    // Конструктор с сообщением и причиной
    public DataAccessException(String message, Throwable cause) {
        super(message, cause);
    }

    // Конструктор с детальным контекстом
    public DataAccessException(String operation, String entity, Throwable cause) {
        super(String.format("Ошибка доступа к данным: %s для %s", operation, entity), cause);
        this.operation = operation;
        this.entity = entity;
    }

    public String getOperation() { return operation; }
    public String getEntity() { return entity; }
}

class User {
    private int id;
    private String name;

    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "User{id=" + id + ", name='" + name + "'}";
    }
}

public class UserRepository {

    public User findUserById(int userId) throws DataAccessException {
        try {
            // Имитация подключения к базе данных
            Connection conn = simulateConnection();

            // Имитация выполнения запроса (это выбросит SQLException)
            String query = "SELECT * FROM users WHERE id = " + userId;
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(query);

            if (rs.next()) {
                return new User(rs.getInt("id"), rs.getString("name"));
            }
            return null;

        } catch (SQLException e) {
            // Оборачиваем SQLException с контекстом
            // Оригинальное исключение сохраняется как причина
            throw new DataAccessException(
                "findUserById",
                "User[id=" + userId + "]",
                e  // Цепочка оригинального исключения
            );
        }
    }

    public void saveUser(User user) throws DataAccessException {
        try {
            // Имитация сохранения в базу данных
            Connection conn = simulateConnection();
            String query = "INSERT INTO users VALUES (?, ?)";
            PreparedStatement stmt = conn.prepareStatement(query);
            // Это выбросит SQLException
            stmt.execute();

        } catch (SQLException e) {
            // Оборачиваем и добавляем контекст
            throw new DataAccessException(
                "saveUser",
                "User[" + user + "]",
                e
            );
        }
    }

    public void demonstrateExceptionChaining() {
        System.out.println("=== Демонстрация Цепочки Исключений ===\n");

        try {
            // Это не удастся и выбросит цепочное исключение
            User user = findUserById(999);

        } catch (DataAccessException e) {
            System.out.println("Исключение верхнего уровня:");
            System.out.println("  Сообщение: " + e.getMessage());
            System.out.println("  Операция: " + e.getOperation());
            System.out.println("  Сущность: " + e.getEntity());

            // Доступ к причине (оригинальное исключение)
            Throwable cause = e.getCause();
            if (cause != null) {
                System.out.println("\nОригинальная причина:");
                System.out.println("  Тип: " + cause.getClass().getSimpleName());
                System.out.println("  Сообщение: " + cause.getMessage());
            }

            // Вывод полного стека вызовов, показывающего цепочку
            System.out.println("\nПолный стек вызовов:");
            e.printStackTrace();
        }
    }

    // Имитация подключения к базе данных, которое выбрасывает SQLException
    private Connection simulateConnection() throws SQLException {
        throw new SQLException("Невозможно подключиться к базе данных на localhost:5432");
    }

    public static void main(String[] args) {
        UserRepository repo = new UserRepository();

        System.out.println("=== Пример 1: Простая Обработка Ошибок ===");
        try {
            User user = repo.findUserById(123);
            System.out.println("Найдено: " + user);
        } catch (DataAccessException e) {
            System.out.println("Ошибка: " + e.getMessage());
            System.out.println("Корневая причина: " + e.getCause().getMessage());
        }

        System.out.println("\n=== Пример 2: Детальная Цепочка Исключений ===");
        repo.demonstrateExceptionChaining();

        System.out.println("\n=== Пример 3: Множественные Операции ===");
        try {
            // Обе операции не удастся
            User user1 = repo.findUserById(1);
            User user2 = new User(100, "Иван");
            repo.saveUser(user2);
        } catch (DataAccessException e) {
            System.out.println("Операция не удалась: " + e.getOperation());
            System.out.println("Для сущности: " + e.getEntity());
            System.out.println("Из-за: " + e.getCause().getClass().getSimpleName());
        }
    }
}`,
            description: `Реализуйте слой доступа к данным, демонстрирующий цепочку исключений для сохранения стека вызовов и добавления контекста.

Требования:
1. Создайте DataAccessException как пользовательское исключение
2. Реализуйте UserRepository с методами, которые перехватывают низкоуровневые исключения и оборачивают их
3. Используйте цепочку исключений для сохранения оригинальной причины
4. Добавьте контекстную информацию при повторном выбросе исключений
5. Продемонстрируйте getCause() и printStackTrace() для показа полной цепочки

Пример:
\`\`\`java
UserRepository repo = new UserRepository();
repo.findUserById(999);  // Оборачивает SQLException с контекстом
// Полный стек вызовов показывает оба исключения
\`\`\``,
            hint1: `Используйте конструктор super(message, cause) для создания цепочки исключений. Параметр cause сохраняет оригинальное исключение. Вы можете получить его позже с помощью getCause().`,
            hint2: `При оборачивании исключений добавляйте контекстную информацию в ваш класс пользовательского исключения. Включайте детали, такие как имя операции, затронутая сущность и любые релевантные параметры.`,
            whyItMatters: `Цепочка исключений необходима для отладки сложных приложений. Она сохраняет полный контекст ошибки из низкоуровневых операций, добавляя при этом высокоуровневый бизнес-контекст. Это помогает разработчикам быстро определить корневые причины, не теряя важную диагностическую информацию.

**Продакшен паттерн:**
\`\`\`java
public class OrderService {
    public Order createOrder(OrderRequest req) throws OrderException {
        try {
            validateInventory(req.getItems());
            Payment payment = processPayment(req);
            return saveOrder(req, payment);
        } catch (InventoryException e) {
            logger.error("Inventory check failed for order", e);
            throw new OrderException("OUT_OF_STOCK",
                "Order creation failed: items unavailable", e);
        } catch (PaymentException e) {
            logger.error("Payment processing failed", e);
            metrics.incrementCounter("order.payment_failed");
            throw new OrderException("PAYMENT_FAILED",
                "Order creation failed: payment declined", e);
        }
    }
}
\`\`\`

**Практические преимущества:**
- Сохранение полного стека вызовов
- Добавление бизнес-контекста к техническим ошибкам
- Упрощенная диагностика проблем в продакшене`
        },
        uz: {
            title: `Istisnolarni Zanjir va O'rash`,
            solutionCode: `import java.sql.*;

// Zanjir qo'llab-quvvatlaydigan maxsus istisno
class DataAccessException extends Exception {
    private String operation;
    private String entity;

    // Xabar va sabab bilan konstruktor
    public DataAccessException(String message, Throwable cause) {
        super(message, cause);
    }

    // Batafsil kontekst bilan konstruktor
    public DataAccessException(String operation, String entity, Throwable cause) {
        super(String.format("Ma'lumotlarga kirish xatosi: %s %s uchun", operation, entity), cause);
        this.operation = operation;
        this.entity = entity;
    }

    public String getOperation() { return operation; }
    public String getEntity() { return entity; }
}

class User {
    private int id;
    private String name;

    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "User{id=" + id + ", name='" + name + "'}";
    }
}

public class UserRepository {

    public User findUserById(int userId) throws DataAccessException {
        try {
            // Ma'lumotlar bazasiga ulanishni taqlid qilish
            Connection conn = simulateConnection();

            // So'rov bajarilishini taqlid qilish (bu SQLException tashlaydi)
            String query = "SELECT * FROM users WHERE id = " + userId;
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(query);

            if (rs.next()) {
                return new User(rs.getInt("id"), rs.getString("name"));
            }
            return null;

        } catch (SQLException e) {
            // SQLException ni kontekst bilan o'rash
            // Asl istisno sabab sifatida saqlanadi
            throw new DataAccessException(
                "findUserById",
                "User[id=" + userId + "]",
                e  // Asl istisnoni zanjir qilish
            );
        }
    }

    public void saveUser(User user) throws DataAccessException {
        try {
            // Ma'lumotlar bazasiga saqlashni taqlid qilish
            Connection conn = simulateConnection();
            String query = "INSERT INTO users VALUES (?, ?)";
            PreparedStatement stmt = conn.prepareStatement(query);
            // Bu SQLException tashlaydi
            stmt.execute();

        } catch (SQLException e) {
            // O'rash va kontekst qo'shish
            throw new DataAccessException(
                "saveUser",
                "User[" + user + "]",
                e
            );
        }
    }

    public void demonstrateExceptionChaining() {
        System.out.println("=== Istisnolar Zanjiri Namoyishi ===\n");

        try {
            // Bu muvaffaqiyatsiz bo'ladi va zanjirli istisno tashlaydi
            User user = findUserById(999);

        } catch (DataAccessException e) {
            System.out.println("Yuqori daraja istisnosi:");
            System.out.println("  Xabar: " + e.getMessage());
            System.out.println("  Operatsiya: " + e.getOperation());
            System.out.println("  Ob'ekt: " + e.getEntity());

            // Sababga kirish (asl istisno)
            Throwable cause = e.getCause();
            if (cause != null) {
                System.out.println("\nAsl sabab:");
                System.out.println("  Turi: " + cause.getClass().getSimpleName());
                System.out.println("  Xabar: " + cause.getMessage());
            }

            // Zanjirni ko'rsatuvchi to'liq stek izini chop etish
            System.out.println("\nTo'liq stek izi:");
            e.printStackTrace();
        }
    }

    // SQLException tashlaydigan ma'lumotlar bazasi ulanishini taqlid qilish
    private Connection simulateConnection() throws SQLException {
        throw new SQLException("localhost:5432 da ma'lumotlar bazasiga ulanib bo'lmadi");
    }

    public static void main(String[] args) {
        UserRepository repo = new UserRepository();

        System.out.println("=== Misol 1: Oddiy Xatolarni Qayta Ishlash ===");
        try {
            User user = repo.findUserById(123);
            System.out.println("Topildi: " + user);
        } catch (DataAccessException e) {
            System.out.println("Xato: " + e.getMessage());
            System.out.println("Ildiz sababi: " + e.getCause().getMessage());
        }

        System.out.println("\n=== Misol 2: Batafsil Istisnolar Zanjiri ===");
        repo.demonstrateExceptionChaining();

        System.out.println("\n=== Misol 3: Ko'p Operatsiyalar ===");
        try {
            // Ikkala operatsiya ham muvaffaqiyatsiz bo'ladi
            User user1 = repo.findUserById(1);
            User user2 = new User(100, "Ali");
            repo.saveUser(user2);
        } catch (DataAccessException e) {
            System.out.println("Operatsiya muvaffaqiyatsiz: " + e.getOperation());
            System.out.println("Ob'ektda: " + e.getEntity());
            System.out.println("Sababi: " + e.getCause().getClass().getSimpleName());
        }
    }
}`,
            description: `Stek izlarini saqlash va kontekst qo'shish uchun istisnolar zanjirini ko'rsatadigan ma'lumotlarga kirish qatlamini amalga oshiring.

Talablar:
1. DataAccessException ni maxsus istisno sifatida yarating
2. Past darajadagi istisnolarni ushlaydigаn va o'raydigan metodlar bilan UserRepository ni yarating
3. Asl sababni saqlash uchun istisnolar zanjiridan foydalaning
4. Istisnolarni qayta tashlashda kontekstual ma'lumot qo'shing
5. To'liq zanjirni ko'rsatish uchun getCause() va printStackTrace() ni ko'rsating

Misol:
\`\`\`java
UserRepository repo = new UserRepository();
repo.findUserById(999);  // SQLException ni kontekst bilan o'raydi
// To'liq stek izi ikkala istisnoni ham ko'rsatadi
\`\`\``,
            hint1: `Istisnolarni zanjir qilish uchun super(message, cause) konstruktordan foydalaning. cause parametri asl istisnoni saqlaydi. Uni keyinchalik getCause() bilan olishingiz mumkin.`,
            hint2: `Istisnolarni o'rashda maxsus istisno klassingizga kontekstual ma'lumot qo'shing. Operatsiya nomi, ta'sirlangan ob'ekt va tegishli parametrlar kabi tafsilotlarni kiriting.`,
            whyItMatters: `Istisnolar zanjiri murakkab ilovalarni disk raskadrovka qilish uchun zarur. U past darajadagi operatsiyalardan to'liq xato kontekstini saqlaydi va yuqori darajadagi biznes kontekstini qo'shadi. Bu dasturchilarga muhim diagnostika ma'lumotlarini yo'qotmasdan ildiz sabablarini tezda aniqlashga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
public class OrderService {
    public Order createOrder(OrderRequest req) throws OrderException {
        try {
            validateInventory(req.getItems());
            Payment payment = processPayment(req);
            return saveOrder(req, payment);
        } catch (InventoryException e) {
            logger.error("Buyurtma uchun inventarizatsiya tekshiruvi muvaffaqiyatsiz", e);
            throw new OrderException("OUT_OF_STOCK",
                "Buyurtma yaratish muvaffaqiyatsiz: mahsulotlar mavjud emas", e);
        } catch (PaymentException e) {
            logger.error("To'lovni qayta ishlash muvaffaqiyatsiz", e);
            metrics.incrementCounter("order.payment_failed");
            throw new OrderException("PAYMENT_FAILED",
                "Buyurtma yaratish muvaffaqiyatsiz: to'lov rad etildi", e);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- To'liq stek izini saqlash
- Texnik xatolarga biznes kontekstini qo'shish
- Ishlab chiqarishda muammolarni tashxislashni soddalashtirish`
        }
    }
};

export default task;
