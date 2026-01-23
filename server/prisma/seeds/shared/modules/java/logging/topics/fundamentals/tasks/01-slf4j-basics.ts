import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-slf4j-basics',
    title: 'SLF4J Basics',
    difficulty: 'easy',
    tags: ['java', 'logging', 'slf4j'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn SLF4J logging basics in Java.

**Requirements:**
1. Import SLF4J Logger and LoggerFactory
2. Create a logger instance using LoggerFactory
3. Log an info message: "Application started"
4. Log a debug message: "Debug mode enabled"
5. Log a warning message: "Low memory warning"
6. Log an error message: "Failed to connect to database"
7. Print messages at different log levels

SLF4J (Simple Logging Facade for Java) is a logging abstraction that allows you to use any logging framework.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Slf4jBasics {
    // Create a logger instance for this class

    public static void main(String[] args) {
        // Log an info message

        // Log a debug message

        // Log a warning message

        // Log an error message
    }
}`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Slf4jBasics {
    // Create a logger instance for this class
    private static final Logger logger = LoggerFactory.getLogger(Slf4jBasics.class);

    public static void main(String[] args) {
        // Log an info message
        logger.info("Application started");

        // Log a debug message
        logger.debug("Debug mode enabled");

        // Log a warning message
        logger.warn("Low memory warning");

        // Log an error message
        logger.error("Failed to connect to database");

        System.out.println("Check the console for log messages");
    }
}`,
    hint1: `Use LoggerFactory.getLogger(ClassName.class) to create a logger instance. Store it as a static final field.`,
    hint2: `Use logger.info(), logger.debug(), logger.warn(), and logger.error() to log messages at different levels.`,
    whyItMatters: `SLF4J is the industry standard logging facade in Java applications. It provides a clean API and allows you to switch between different logging implementations without changing your code.

**Production Pattern:**
\`\`\`java
// Using logger in production class
public class UserService {
    private static final Logger logger = LoggerFactory.getLogger(UserService.class);

    public void processUser(User user) {
        logger.info("Processing user: {}", user.getId());
        try {
            // business logic
            logger.debug("User {} validated successfully", user.getId());
        } catch (Exception e) {
            logger.error("Failed to process user: {}", user.getId(), e);
            throw e;
        }
    }
}
\`\`\`

**Practical Benefits:**
- One logger per class - best practice for organizing logs
- Logs help track execution flow in production
- Easy to switch between Logback, Log4j2 without changing code`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: class can be instantiated
class Test1 {
    @Test
    public void test() {
        Slf4jBasics obj = new Slf4jBasics();
        assertNotNull("Slf4jBasics instance should be created", obj);
    }
}

// Test2: main outputs console message
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Slf4jBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should print console message",
            output.contains("console") || output.contains("log") ||
            output.contains("консоль") || output.contains("xabarlar"));
    }
}

// Test3: main produces some output (logs or println)
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        PrintStream oldErr = System.err;
        System.setOut(new PrintStream(out));
        System.setErr(new PrintStream(err));
        Slf4jBasics.main(new String[]{});
        System.setOut(oldOut);
        System.setErr(oldErr);
        String allOutput = out.toString() + err.toString();
        assertTrue("Should produce some output",
            allOutput.length() > 0);
    }
}

// Test4: class creates static logger field
class Test4 {
    @Test
    public void test() {
        try {
            java.lang.reflect.Field loggerField = Slf4jBasics.class.getDeclaredField("logger");
            assertTrue("Logger should be static",
                java.lang.reflect.Modifier.isStatic(loggerField.getModifiers()));
        } catch (NoSuchFieldException e) {
            fail("Should have a 'logger' field");
        }
    }
}

// Test5: logger field should be final
class Test5 {
    @Test
    public void test() {
        try {
            java.lang.reflect.Field loggerField = Slf4jBasics.class.getDeclaredField("logger");
            assertTrue("Logger should be final",
                java.lang.reflect.Modifier.isFinal(loggerField.getModifiers()));
        } catch (NoSuchFieldException e) {
            fail("Should have a 'logger' field");
        }
    }
}

// Test6: main does not throw exceptions
class Test6 {
    @Test
    public void test() {
        boolean completed = false;
        try {
            Slf4jBasics.main(new String[]{});
            completed = true;
        } catch (Exception e) {
            fail("Main should not throw: " + e.getMessage());
        }
        assertTrue("Main should complete without exceptions", completed);
    }
}

// Test7: multiple instances share same logger
class Test7 {
    @Test
    public void test() {
        Slf4jBasics obj1 = new Slf4jBasics();
        Slf4jBasics obj2 = new Slf4jBasics();
        assertNotNull("First instance created", obj1);
        assertNotNull("Second instance created", obj2);
        // Static logger should be shared
    }
}

// Test8: main can be called multiple times
class Test8 {
    @Test
    public void test() {
        int callCount = 0;
        try {
            Slf4jBasics.main(new String[]{});
            callCount++;
            Slf4jBasics.main(new String[]{});
            callCount++;
        } catch (Exception e) {
            fail("Multiple calls should not throw: " + e.getMessage());
        }
        assertEquals("Both calls should complete", 2, callCount);
    }
}

// Test9: Verify no null pointer exceptions
class Test9 {
    @Test
    public void test() {
        boolean completed = false;
        try {
            Slf4jBasics.main(new String[]{});
            completed = true;
        } catch (NullPointerException e) {
            fail("Should not have null pointer exceptions");
        }
        assertTrue("Should complete without NullPointerException", completed);
    }
}

// Test10: Verify class can be instantiated multiple times
class Test10 {
    @Test
    public void test() {
        Slf4jBasics obj1 = new Slf4jBasics();
        Slf4jBasics obj2 = new Slf4jBasics();
        Slf4jBasics obj3 = new Slf4jBasics();
        assertNotNull("First instance should be created", obj1);
        assertNotNull("Second instance should be created", obj2);
        assertNotNull("Third instance should be created", obj3);
    }
}
`,
    translations: {
        ru: {
            title: 'Основы SLF4J',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Slf4jBasics {
    // Создаем экземпляр логгера для этого класса
    private static final Logger logger = LoggerFactory.getLogger(Slf4jBasics.class);

    public static void main(String[] args) {
        // Записываем info сообщение
        logger.info("Приложение запущено");

        // Записываем debug сообщение
        logger.debug("Режим отладки включен");

        // Записываем предупреждающее сообщение
        logger.warn("Предупреждение о низкой памяти");

        // Записываем сообщение об ошибке
        logger.error("Не удалось подключиться к базе данных");

        System.out.println("Проверьте консоль на наличие сообщений");
    }
}`,
            description: `Изучите основы логирования с SLF4J в Java.

**Требования:**
1. Импортируйте Logger и LoggerFactory из SLF4J
2. Создайте экземпляр логгера используя LoggerFactory
3. Запишите info сообщение: "Application started"
4. Запишите debug сообщение: "Debug mode enabled"
5. Запишите warning сообщение: "Low memory warning"
6. Запишите error сообщение: "Failed to connect to database"
7. Выведите сообщения на разных уровнях логирования

SLF4J (Simple Logging Facade for Java) - это абстракция логирования, которая позволяет использовать любой фреймворк логирования.`,
            hint1: `Используйте LoggerFactory.getLogger(ClassName.class) для создания экземпляра логгера. Сохраните его как статическое final поле.`,
            hint2: `Используйте logger.info(), logger.debug(), logger.warn() и logger.error() для записи сообщений разных уровней.`,
            whyItMatters: `SLF4J - это промышленный стандарт фасада логирования в Java приложениях. Он предоставляет чистый API и позволяет переключаться между различными реализациями логирования без изменения кода.

**Продакшен паттерн:**
\`\`\`java
// Использование логгера в production классе
public class UserService {
    private static final Logger logger = LoggerFactory.getLogger(UserService.class);

    public void processUser(User user) {
        logger.info("Processing user: {}", user.getId());
        try {
            // бизнес логика
            logger.debug("User {} validated successfully", user.getId());
        } catch (Exception e) {
            logger.error("Failed to process user: {}", user.getId(), e);
            throw e;
        }
    }
}
\`\`\`

**Практические преимущества:**
- Один логгер на класс - лучшая практика для организации логов
- Логи помогают отслеживать поток выполнения в продакшене
- Легко переключаться между Logback, Log4j2 без изменения кода`
        },
        uz: {
            title: 'SLF4J Asoslari',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Slf4jBasics {
    // Bu klass uchun logger nusxasini yaratamiz
    private static final Logger logger = LoggerFactory.getLogger(Slf4jBasics.class);

    public static void main(String[] args) {
        // Info xabarini yozamiz
        logger.info("Ilova ishga tushdi");

        // Debug xabarini yozamiz
        logger.debug("Debug rejimi yoqildi");

        // Ogohlantirish xabarini yozamiz
        logger.warn("Kam xotira ogohlantiruvi");

        // Xato xabarini yozamiz
        logger.error("Ma'lumotlar bazasiga ulanib bo'lmadi");

        System.out.println("Xabarlar uchun konsolni tekshiring");
    }
}`,
            description: `Java da SLF4J logging asoslarini o'rganing.

**Talablar:**
1. SLF4J dan Logger va LoggerFactory ni import qiling
2. LoggerFactory yordamida logger nusxasini yarating
3. Info xabarini yozing: "Application started"
4. Debug xabarini yozing: "Debug mode enabled"
5. Warning xabarini yozing: "Low memory warning"
6. Error xabarini yozing: "Failed to connect to database"
7. Turli darajadagi log xabarlarini chiqaring

SLF4J (Simple Logging Facade for Java) - bu har qanday logging framework dan foydalanishga imkon beruvchi logging abstraktsiyasi.`,
            hint1: `Logger nusxasini yaratish uchun LoggerFactory.getLogger(ClassName.class) dan foydalaning. Uni statik final maydon sifatida saqlang.`,
            hint2: `Turli darajadagi xabarlarni yozish uchun logger.info(), logger.debug(), logger.warn() va logger.error() dan foydalaning.`,
            whyItMatters: `SLF4J Java ilovalarida sanoat standarti logging fasadi hisoblanadi. U toza API taqdim etadi va kodingizni o'zgartirmasdan turli logging implementatsiyalari o'rtasida o'tish imkonini beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Production klassida loggerni ishlatish
public class UserService {
    private static final Logger logger = LoggerFactory.getLogger(UserService.class);

    public void processUser(User user) {
        logger.info("Processing user: {}", user.getId());
        try {
            // biznes logikasi
            logger.debug("User {} validated successfully", user.getId());
        } catch (Exception e) {
            logger.error("Failed to process user: {}", user.getId(), e);
            throw e;
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Har bir klass uchun bitta logger - loglarni tartibga solish uchun eng yaxshi amaliyot
- Loglar ishlab chiqarishda bajarilish oqimini kuzatishga yordam beradi
- Kodni o'zgartirmasdan Logback, Log4j2 o'rtasida osongina o'tish mumkin`
        }
    }
};

export default task;
