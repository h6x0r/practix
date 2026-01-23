import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-parameterized-logging',
    title: 'Parameterized Logging',
    difficulty: 'medium',
    tags: ['java', 'logging', 'slf4j', 'parameters'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master parameterized logging in SLF4J to avoid string concatenation.

**Requirements:**
1. Create a User class with name and id
2. Log user information using parameterized logging
3. Demonstrate the problem with string concatenation
4. Use placeholder syntax {} for single and multiple parameters
5. Log with 3+ parameters using varargs
6. Show performance benefits of parameterized logging
7. Log complex objects with toString()

Parameterized logging is more efficient because strings are only constructed when the log level is enabled.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class User {
    // Add fields and constructor
}

public class ParameterizedLogging {
    private static final Logger logger = LoggerFactory.getLogger(ParameterizedLogging.class);

    public static void main(String[] args) {
        User user = new User("Alice", 1001);

        // Bad: String concatenation (always creates string)

        // Good: Parameterized logging (only creates string if needed)

        // Multiple parameters

        // Three or more parameters

        // Complex objects
    }
}`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class User {
    private String name;
    private int id;

    public User(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() { return name; }
    public int getId() { return id; }

    @Override
    public String toString() {
        return "User{name='" + name + "', id=" + id + "}";
    }
}

public class ParameterizedLogging {
    private static final Logger logger = LoggerFactory.getLogger(ParameterizedLogging.class);

    public static void main(String[] args) {
        User user = new User("Alice", 1001);

        // Bad: String concatenation (always creates string, even if debug is disabled)
        logger.debug("User logged in: " + user.getName() + " with ID: " + user.getId());

        // Good: Parameterized logging (only creates string if debug is enabled)
        logger.info("User {} logged in with ID: {}", user.getName(), user.getId());

        // Multiple parameters - more efficient
        String action = "update";
        String resource = "profile";
        long duration = 150;
        logger.info("User {} performed {} on {} in {}ms",
            user.getName(), action, resource, duration);

        // Three or more parameters using varargs
        logger.info("User: {}, Action: {}, Resource: {}, Duration: {}ms, Status: {}",
            user.getName(), action, resource, duration, "success");

        // Complex objects - toString() is called only if log level is enabled
        logger.debug("User object: {}", user);

        // Demonstrate performance difference
        demonstratePerformance();
    }

    private static void demonstratePerformance() {
        User user = new User("Bob", 2002);

        System.out.println("\\nPerformance comparison:");

        // String concatenation - always creates string
        long start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: " + user.getName() + " " + user.getId());
        }
        long concatTime = System.nanoTime() - start;

        // Parameterized - only creates string if level enabled
        start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: {} {}", user.getName(), user.getId());
        }
        long paramTime = System.nanoTime() - start;

        System.out.println("String concatenation: " + concatTime + "ns");
        System.out.println("Parameterized logging: " + paramTime + "ns");
        System.out.println("Parameterized is " + (concatTime / (double) paramTime) + "x faster");
    }
}`,
    hint1: `Use {} as placeholders in the log message. SLF4J will replace them with the parameters in order.`,
    hint2: `Parameterized logging avoids creating strings when the log level is disabled, improving performance significantly.`,
    whyItMatters: `Parameterized logging is crucial for performance in production applications. String concatenation always creates objects, even when logging is disabled, wasting CPU and memory. Parameterized logging only constructs the string when needed.

**Production Pattern:**
\`\`\`java
// Efficient logging in high-load systems
public class PaymentProcessor {
    private static final Logger logger = LoggerFactory.getLogger(PaymentProcessor.class);

    public void processPayment(Payment payment) {
        // BAD: concatenation always creates string
        // logger.debug("Processing: " + payment.getId() + " " + payment.getAmount());

        // GOOD: string created only if DEBUG enabled
        logger.debug("Processing payment: {} amount: {}", payment.getId(), payment.getAmount());

        // For expensive operations use guard
        if (logger.isDebugEnabled()) {
            logger.debug("Payment details: {}", payment.toDetailedString());
        }
    }
}
\`\`\`

**Practical Benefits:**
- Save up to 90% resources on logging in production
- Avoid creating millions of temporary strings
- Improve GC performance`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify User class creation
class Test1 {
    @Test
    public void test() {
        User user = new User("Alice", 1001);
        assertNotNull("User should be created", user);
        assertEquals("Name should be Alice", "Alice", user.getName());
        assertEquals("ID should be 1001", 1001, user.getId());
    }
}

// Test2: Verify User toString method
class Test2 {
    @Test
    public void test() {
        User user = new User("Bob", 2002);
        String userString = user.toString();
        assertTrue("toString should contain name", userString.contains("Bob"));
        assertTrue("toString should contain id", userString.contains("2002"));
    }
}

// Test3: main outputs performance comparison
class Test3 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream oldOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        ParameterizedLogging.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should print performance comparison",
            output.contains("comparison") || output.contains("Performance") ||
            output.contains("сравнение") || output.contains("Сравнение") ||
            output.contains("solishtirish") || output.contains("ns"));
    }
}

// Test4: Verify User getter methods
class Test4 {
    @Test
    public void test() {
        User user = new User("Charlie", 3003);
        assertEquals("getName should return Charlie", "Charlie", user.getName());
        assertEquals("getId should return 3003", 3003, user.getId());
    }
}

// Test5: Verify multiple User instances
class Test5 {
    @Test
    public void test() {
        User user1 = new User("Alice", 1001);
        User user2 = new User("Bob", 2002);
        assertNotEquals("Users should be different", user1.getName(), user2.getName());
        assertNotEquals("IDs should be different", user1.getId(), user2.getId());
    }
}

// Test6: Verify User with different names
class Test6 {
    @Test
    public void test() {
        User user1 = new User("John", 100);
        User user2 = new User("Jane", 200);
        User user3 = new User("Jack", 300);
        assertNotNull("User1 should be created", user1);
        assertNotNull("User2 should be created", user2);
        assertNotNull("User3 should be created", user3);
    }
}

// Test7: static logger field exists
class Test7 {
    @Test
    public void test() {
        try {
            java.lang.reflect.Field loggerField = ParameterizedLogging.class.getDeclaredField("logger");
            assertTrue("Logger should be static",
                java.lang.reflect.Modifier.isStatic(loggerField.getModifiers()));
            assertTrue("Logger should be final",
                java.lang.reflect.Modifier.isFinal(loggerField.getModifiers()));
        } catch (NoSuchFieldException e) {
            fail("Should have a 'logger' field");
        }
    }
}

// Test8: Verify no null values in User
class Test8 {
    @Test
    public void test() {
        User user = new User("ValidName", 123);
        assertNotNull("User name should not be null", user.getName());
        assertTrue("User ID should be valid", user.getId() > 0);
    }
}

// Test9: Verify ParameterizedLogging instantiation
class Test9 {
    @Test
    public void test() {
        ParameterizedLogging obj = new ParameterizedLogging();
        assertNotNull("ParameterizedLogging should be instantiated", obj);
    }
}

// Test10: output shows timing results
class Test10 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream oldOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        ParameterizedLogging.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show timing results",
            output.contains("ns") || output.contains("нс") ||
            output.contains("faster") || output.contains("быстрее") || output.contains("tezroq"));
    }
}
`,
    translations: {
        ru: {
            title: 'Параметризованное логирование',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class User {
    private String name;
    private int id;

    public User(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() { return name; }
    public int getId() { return id; }

    @Override
    public String toString() {
        return "User{name='" + name + "', id=" + id + "}";
    }
}

public class ParameterizedLogging {
    private static final Logger logger = LoggerFactory.getLogger(ParameterizedLogging.class);

    public static void main(String[] args) {
        User user = new User("Alice", 1001);

        // Плохо: конкатенация строк (всегда создает строку, даже если debug отключен)
        logger.debug("Пользователь вошел: " + user.getName() + " с ID: " + user.getId());

        // Хорошо: параметризованное логирование (создает строку только если debug включен)
        logger.info("Пользователь {} вошел с ID: {}", user.getName(), user.getId());

        // Несколько параметров - более эффективно
        String action = "update";
        String resource = "profile";
        long duration = 150;
        logger.info("Пользователь {} выполнил {} на {} за {}мс",
            user.getName(), action, resource, duration);

        // Три или более параметров используя varargs
        logger.info("Пользователь: {}, Действие: {}, Ресурс: {}, Длительность: {}мс, Статус: {}",
            user.getName(), action, resource, duration, "success");

        // Сложные объекты - toString() вызывается только если уровень логирования включен
        logger.debug("Объект пользователя: {}", user);

        // Демонстрируем разницу в производительности
        demonstratePerformance();
    }

    private static void demonstratePerformance() {
        User user = new User("Bob", 2002);

        System.out.println("\\nСравнение производительности:");

        // Конкатенация строк - всегда создает строку
        long start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: " + user.getName() + " " + user.getId());
        }
        long concatTime = System.nanoTime() - start;

        // Параметризованное - создает строку только если уровень включен
        start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: {} {}", user.getName(), user.getId());
        }
        long paramTime = System.nanoTime() - start;

        System.out.println("Конкатенация строк: " + concatTime + "нс");
        System.out.println("Параметризованное логирование: " + paramTime + "нс");
        System.out.println("Параметризованное быстрее в " + (concatTime / (double) paramTime) + " раз");
    }
}`,
            description: `Освойте параметризованное логирование в SLF4J для избежания конкатенации строк.

**Требования:**
1. Создайте класс User с именем и id
2. Запишите информацию о пользователе используя параметризованное логирование
3. Продемонстрируйте проблему с конкатенацией строк
4. Используйте синтаксис заполнителей {} для одного и нескольких параметров
5. Запишите с 3+ параметрами используя varargs
6. Покажите преимущества производительности параметризованного логирования
7. Запишите сложные объекты с toString()

Параметризованное логирование более эффективно, потому что строки создаются только когда уровень логирования включен.`,
            hint1: `Используйте {} как заполнители в сообщении лога. SLF4J заменит их параметрами по порядку.`,
            hint2: `Параметризованное логирование избегает создания строк когда уровень логирования отключен, значительно улучшая производительность.`,
            whyItMatters: `Параметризованное логирование критически важно для производительности в продакшн приложениях. Конкатенация строк всегда создает объекты, даже когда логирование отключено, тратя CPU и память. Параметризованное логирование строит строку только когда это необходимо.

**Продакшен паттерн:**
\`\`\`java
// Эффективное логирование в высоконагруженных системах
public class PaymentProcessor {
    private static final Logger logger = LoggerFactory.getLogger(PaymentProcessor.class);

    public void processPayment(Payment payment) {
        // ПЛОХО: конкатенация всегда создает строку
        // logger.debug("Processing: " + payment.getId() + " " + payment.getAmount());

        // ХОРОШО: строка создается только если DEBUG включен
        logger.debug("Processing payment: {} amount: {}", payment.getId(), payment.getAmount());

        // Для дорогих операций используйте guard
        if (logger.isDebugEnabled()) {
            logger.debug("Payment details: {}", payment.toDetailedString());
        }
    }
}
\`\`\`

**Практические преимущества:**
- Экономия до 90% ресурсов на логирование в продакшене
- Избегание создания миллионов временных строк
- Улучшение производительности GC`
        },
        uz: {
            title: 'Parametrlangan Logging',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class User {
    private String name;
    private int id;

    public User(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() { return name; }
    public int getId() { return id; }

    @Override
    public String toString() {
        return "User{name='" + name + "', id=" + id + "}";
    }
}

public class ParameterizedLogging {
    private static final Logger logger = LoggerFactory.getLogger(ParameterizedLogging.class);

    public static void main(String[] args) {
        User user = new User("Alice", 1001);

        // Yomon: satr birlashtirish (har doim satr yaratadi, debug o'chirilgan bo'lsa ham)
        logger.debug("Foydalanuvchi kirdi: " + user.getName() + " ID bilan: " + user.getId());

        // Yaxshi: parametrlangan logging (faqat debug yoqilgan bo'lsa satr yaratadi)
        logger.info("Foydalanuvchi {} ID {} bilan kirdi", user.getName(), user.getId());

        // Ko'p parametrlar - samaraliroq
        String action = "update";
        String resource = "profile";
        long duration = 150;
        logger.info("Foydalanuvchi {} {} ni {} da {}ms ichida bajardi",
            user.getName(), action, resource, duration);

        // Uch yoki undan ko'p parametrlar varargs yordamida
        logger.info("Foydalanuvchi: {}, Harakat: {}, Resurs: {}, Davomiyligi: {}ms, Holati: {}",
            user.getName(), action, resource, duration, "success");

        // Murakkab obyektlar - toString() faqat log darajasi yoqilgan bo'lsa chaqiriladi
        logger.debug("Foydalanuvchi obyekti: {}", user);

        // Ishlash tezligidagi farqni ko'rsatamiz
        demonstratePerformance();
    }

    private static void demonstratePerformance() {
        User user = new User("Bob", 2002);

        System.out.println("\\nIshlash tezligini solishtirish:");

        // Satr birlashtirish - har doim satr yaratadi
        long start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: " + user.getName() + " " + user.getId());
        }
        long concatTime = System.nanoTime() - start;

        // Parametrlangan - faqat daraja yoqilgan bo'lsa satr yaratadi
        start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            logger.debug("Debug: {} {}", user.getName(), user.getId());
        }
        long paramTime = System.nanoTime() - start;

        System.out.println("Satr birlashtirish: " + concatTime + "ns");
        System.out.println("Parametrlangan logging: " + paramTime + "ns");
        System.out.println("Parametrlangan " + (concatTime / (double) paramTime) + "x tezroq");
    }
}`,
            description: `Satr birlashtirishdan qochish uchun SLF4J da parametrlangan loggingni o'rganing.

**Talablar:**
1. Ism va id ga ega User klassini yarating
2. Parametrlangan logging yordamida foydalanuvchi ma'lumotlarini yozing
3. Satr birlashtirish muammosini ko'rsating
4. Bitta va ko'p parametrlar uchun {} to'ldiruvchi sintaksisidan foydalaning
5. Varargs yordamida 3+ parametrlar bilan yozing
6. Parametrlangan logging ning ishlash tezligi afzalliklarini ko'rsating
7. toString() bilan murakkab obyektlarni yozing

Parametrlangan logging samaraliroq, chunki satrlar faqat log darajasi yoqilgan bo'lsagina yaratiladi.`,
            hint1: `Log xabarida to'ldiruvchi sifatida {} dan foydalaning. SLF4J ularni tartib bo'yicha parametrlar bilan almashtiradi.`,
            hint2: `Parametrlangan logging log darajasi o'chirilgan bo'lsa satr yaratishdan qochadi va ishlash tezligini sezilarli darajada yaxshilaydi.`,
            whyItMatters: `Parametrlangan logging ishlab chiqarish ilovalarida ishlash tezligi uchun juda muhim. Satr birlashtirish har doim obyektlar yaratadi, hatto logging o'chirilgan bo'lsa ham, CPU va xotirani behuda sarflaydi. Parametrlangan logging satrni faqat kerak bo'lganda yaratadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Yuqori yukli tizimlarda samarali logging
public class PaymentProcessor {
    private static final Logger logger = LoggerFactory.getLogger(PaymentProcessor.class);

    public void processPayment(Payment payment) {
        // YOMON: birlashtirish har doim satr yaratadi
        // logger.debug("Processing: " + payment.getId() + " " + payment.getAmount());

        // YAXSHI: satr faqat DEBUG yoqilgan bo'lsa yaratiladi
        logger.debug("Processing payment: {} amount: {}", payment.getId(), payment.getAmount());

        // Qimmat operatsiyalar uchun guard dan foydalaning
        if (logger.isDebugEnabled()) {
            logger.debug("Payment details: {}", payment.toDetailedString());
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Ishlab chiqarishda loggingda 90% gacha resurslarni tejash
- Millionlab vaqtinchalik satrlar yaratishdan qochish
- GC ishlashini yaxshilash`
        }
    }
};

export default task;
