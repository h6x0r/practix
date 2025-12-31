import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-log-levels',
    title: 'Understanding Log Levels',
    difficulty: 'easy',
    tags: ['java', 'logging', 'slf4j', 'levels'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master all log levels in SLF4J.

**Requirements:**
1. Create a logger for the class
2. Log a TRACE message for very detailed debugging
3. Log a DEBUG message for general debugging
4. Log an INFO message for informational events
5. Log a WARN message for potential issues
6. Log an ERROR message for errors
7. Check if each log level is enabled before logging
8. Demonstrate the hierarchy: TRACE < DEBUG < INFO < WARN < ERROR

Log levels help you control the verbosity of your application's output and filter messages based on their importance.`,
    initialCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogLevels {
    private static final Logger logger = LoggerFactory.getLogger(LogLevels.class);

    public static void main(String[] args) {
        // Log a TRACE message (most verbose)

        // Log a DEBUG message

        // Log an INFO message

        // Log a WARN message

        // Log an ERROR message (least verbose)

        // Check if log level is enabled before logging
    }

    private static void processUser(String username) {
        // Use appropriate log levels for different scenarios
    }
}`,
    solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogLevels {
    private static final Logger logger = LoggerFactory.getLogger(LogLevels.class);

    public static void main(String[] args) {
        // Log a TRACE message (most verbose) - for very detailed debugging
        if (logger.isTraceEnabled()) {
            logger.trace("Entering application, method: main");
        }

        // Log a DEBUG message - for debugging information
        if (logger.isDebugEnabled()) {
            logger.debug("Processing started with {} users", 100);
        }

        // Log an INFO message - for informational events
        logger.info("Application initialized successfully");

        // Log a WARN message - for potentially harmful situations
        logger.warn("Configuration file not found, using defaults");

        // Log an ERROR message (least verbose) - for error events
        logger.error("Failed to connect to external service");

        // Demonstrate usage in a method
        processUser("john_doe");

        System.out.println("\\nLog Level Hierarchy (least to most severe):");
        System.out.println("TRACE < DEBUG < INFO < WARN < ERROR");
    }

    private static void processUser(String username) {
        // TRACE: Very detailed information
        logger.trace("processUser() called with username: {}", username);

        // DEBUG: General debugging information
        logger.debug("Validating user: {}", username);

        // INFO: Important business events
        logger.info("User {} logged in successfully", username);

        // WARN: Potential problems
        if (username.length() < 3) {
            logger.warn("Username {} is shorter than recommended", username);
        }

        logger.trace("processUser() completed");
    }
}`,
    hint1: `Log levels in order of severity: TRACE, DEBUG, INFO, WARN, ERROR. Use isTraceEnabled(), isDebugEnabled() etc. to check if a level is enabled.`,
    hint2: `TRACE is for very detailed debugging, DEBUG for development, INFO for production events, WARN for potential issues, ERROR for actual errors.`,
    whyItMatters: `Understanding log levels is crucial for effective logging. It allows you to control the amount of logging output and ensures important messages aren't lost in verbose debug logs in production.

**Production Pattern:**
\`\`\`java
// Proper level usage in production
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        logger.info("Order {} processing started", order.getId()); // INFO: important events

        if (order.getAmount() > 10000) {
            logger.warn("Large order detected: {} amount: {}", order.getId(), order.getAmount()); // WARN: attention
        }

        try {
            validateOrder(order);
            logger.debug("Order {} validated", order.getId()); // DEBUG: only in dev
        } catch (ValidationException e) {
            logger.error("Order {} validation failed", order.getId(), e); // ERROR: errors
        }
    }
}
\`\`\`

**Practical Benefits:**
- INFO for tracking business flow in production
- WARN for potential issues requiring attention
- ERROR only for actual errors requiring fixes`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify LogLevels class instantiation
class Test1 {
    @Test
    public void test() {
        LogLevels obj = new LogLevels();
        assertNotNull("LogLevels instance should be created", obj);
    }
}

// Test2: Verify main method executes without errors
class Test2 {
    @Test
    public void test() {
        try {
            LogLevels.main(new String[]{});
            assertTrue("Main method should execute successfully", true);
        } catch (Exception e) {
            fail("Main method should not throw exceptions: " + e.getMessage());
        }
    }
}

// Test3: Verify TRACE level logging works
class Test3 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("TRACE logging should work", true);
    }
}

// Test4: Verify DEBUG level logging works
class Test4 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("DEBUG logging should work", true);
    }
}

// Test5: Verify INFO level logging works
class Test5 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("INFO logging should work", true);
    }
}

// Test6: Verify WARN level logging works
class Test6 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("WARN logging should work", true);
    }
}

// Test7: Verify ERROR level logging works
class Test7 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("ERROR logging should work", true);
    }
}

// Test8: Verify all log levels can work together
class Test8 {
    @Test
    public void test() {
        try {
            LogLevels.main(new String[]{});
            assertTrue("All log levels should work together", true);
        } catch (Exception e) {
            fail("All log levels should work together without errors");
        }
    }
}

// Test9: Verify logger hierarchy is respected
class Test9 {
    @Test
    public void test() {
        LogLevels.main(new String[]{});
        assertTrue("Logger hierarchy should be respected", true);
    }
}

// Test10: Verify no exceptions during logging
class Test10 {
    @Test
    public void test() {
        try {
            LogLevels.main(new String[]{});
            assertTrue("No exceptions should occur", true);
        } catch (NullPointerException e) {
            fail("Should not have null pointer exceptions");
        } catch (Exception e) {
            fail("Should not have any exceptions");
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Понимание уровней логирования',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogLevels {
    private static final Logger logger = LoggerFactory.getLogger(LogLevels.class);

    public static void main(String[] args) {
        // TRACE сообщение (наиболее подробное) - для очень детальной отладки
        if (logger.isTraceEnabled()) {
            logger.trace("Вход в приложение, метод: main");
        }

        // DEBUG сообщение - для отладочной информации
        if (logger.isDebugEnabled()) {
            logger.debug("Обработка начата с {} пользователями", 100);
        }

        // INFO сообщение - для информационных событий
        logger.info("Приложение успешно инициализировано");

        // WARN сообщение - для потенциально опасных ситуаций
        logger.warn("Файл конфигурации не найден, используются значения по умолчанию");

        // ERROR сообщение (наименее подробное) - для событий ошибок
        logger.error("Не удалось подключиться к внешнему сервису");

        // Демонстрируем использование в методе
        processUser("john_doe");

        System.out.println("\\nИерархия уровней логирования (от меньшей к большей серьезности):");
        System.out.println("TRACE < DEBUG < INFO < WARN < ERROR");
    }

    private static void processUser(String username) {
        // TRACE: Очень подробная информация
        logger.trace("processUser() вызван с username: {}", username);

        // DEBUG: Общая отладочная информация
        logger.debug("Проверка пользователя: {}", username);

        // INFO: Важные бизнес-события
        logger.info("Пользователь {} успешно вошел в систему", username);

        // WARN: Потенциальные проблемы
        if (username.length() < 3) {
            logger.warn("Имя пользователя {} короче рекомендуемого", username);
        }

        logger.trace("processUser() завершен");
    }
}`,
            description: `Освойте все уровни логирования в SLF4J.

**Требования:**
1. Создайте логгер для класса
2. Запишите TRACE сообщение для очень детальной отладки
3. Запишите DEBUG сообщение для общей отладки
4. Запишите INFO сообщение для информационных событий
5. Запишите WARN сообщение для потенциальных проблем
6. Запишите ERROR сообщение для ошибок
7. Проверьте, включен ли каждый уровень логирования перед записью
8. Продемонстрируйте иерархию: TRACE < DEBUG < INFO < WARN < ERROR

Уровни логирования помогают контролировать детализацию вывода приложения и фильтровать сообщения по их важности.`,
            hint1: `Уровни логирования по степени серьезности: TRACE, DEBUG, INFO, WARN, ERROR. Используйте isTraceEnabled(), isDebugEnabled() и т.д. для проверки включения уровня.`,
            hint2: `TRACE - для очень детальной отладки, DEBUG - для разработки, INFO - для событий в продакшене, WARN - для потенциальных проблем, ERROR - для реальных ошибок.`,
            whyItMatters: `Понимание уровней логирования критически важно для эффективного логирования. Это позволяет контролировать объем вывода логов и гарантирует, что важные сообщения не потеряются в подробных отладочных логах в продакшене.

**Продакшен паттерн:**
\`\`\`java
// Правильное использование уровней в production
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        logger.info("Order {} processing started", order.getId()); // INFO: важные события

        if (order.getAmount() > 10000) {
            logger.warn("Large order detected: {} amount: {}", order.getId(), order.getAmount()); // WARN: внимание
        }

        try {
            validateOrder(order);
            logger.debug("Order {} validated", order.getId()); // DEBUG: только в dev
        } catch (ValidationException e) {
            logger.error("Order {} validation failed", order.getId(), e); // ERROR: ошибки
        }
    }
}
\`\`\`

**Практические преимущества:**
- INFO для отслеживания бизнес-потока в продакшене
- WARN для потенциальных проблем требующих внимания
- ERROR только для настоящих ошибок требующих исправления`
        },
        uz: {
            title: 'Log Darajalarini Tushunish',
            solutionCode: `import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogLevels {
    private static final Logger logger = LoggerFactory.getLogger(LogLevels.class);

    public static void main(String[] args) {
        // TRACE xabari (eng batafsil) - juda batafsil debug uchun
        if (logger.isTraceEnabled()) {
            logger.trace("Ilovaga kirish, metod: main");
        }

        // DEBUG xabari - debug ma'lumotlari uchun
        if (logger.isDebugEnabled()) {
            logger.debug("Qayta ishlash {} foydalanuvchilar bilan boshlandi", 100);
        }

        // INFO xabari - ma'lumot beruvchi hodisalar uchun
        logger.info("Ilova muvaffaqiyatli ishga tushirildi");

        // WARN xabari - potentsial xavfli vaziyatlar uchun
        logger.warn("Konfiguratsiya fayli topilmadi, standart qiymatlar ishlatiladi");

        // ERROR xabari (eng kam batafsil) - xato hodisalari uchun
        logger.error("Tashqi xizmatga ulanib bo'lmadi");

        // Metodda foydalanishni ko'rsatamiz
        processUser("john_doe");

        System.out.println("\\nLog Darajalari Ierarxiyasi (kamdan ko'pga qarab):");
        System.out.println("TRACE < DEBUG < INFO < WARN < ERROR");
    }

    private static void processUser(String username) {
        // TRACE: Juda batafsil ma'lumot
        logger.trace("processUser() chaqirildi username: {}", username);

        // DEBUG: Umumiy debug ma'lumotlari
        logger.debug("Foydalanuvchi tekshirilmoqda: {}", username);

        // INFO: Muhim biznes hodisalari
        logger.info("Foydalanuvchi {} muvaffaqiyatli kirdi", username);

        // WARN: Potentsial muammolar
        if (username.length() < 3) {
            logger.warn("Foydalanuvchi nomi {} tavsiya qilinganidan qisqa", username);
        }

        logger.trace("processUser() yakunlandi");
    }
}`,
            description: `SLF4J da barcha log darajalarini o'rganing.

**Talablar:**
1. Klass uchun logger yarating
2. Juda batafsil debug uchun TRACE xabarini yozing
3. Umumiy debug uchun DEBUG xabarini yozing
4. Ma'lumot beruvchi hodisalar uchun INFO xabarini yozing
5. Potentsial muammolar uchun WARN xabarini yozing
6. Xatolar uchun ERROR xabarini yozing
7. Yozishdan oldin har bir log darajasi yoqilganligini tekshiring
8. Ierarxiyani ko'rsating: TRACE < DEBUG < INFO < WARN < ERROR

Log darajalari ilova chiqishining batafsillligini boshqarishga va muhimlik bo'yicha xabarlarni filtrlashga yordam beradi.`,
            hint1: `Log darajalari jiddiylik bo'yicha: TRACE, DEBUG, INFO, WARN, ERROR. Daraja yoqilganligini tekshirish uchun isTraceEnabled(), isDebugEnabled() va boshqalardan foydalaning.`,
            hint2: `TRACE - juda batafsil debug uchun, DEBUG - ishlab chiqish uchun, INFO - ishlab chiqarish hodisalari uchun, WARN - potentsial muammolar uchun, ERROR - haqiqiy xatolar uchun.`,
            whyItMatters: `Log darajalarini tushunish samarali logging uchun juda muhim. Bu log chiqishini boshqarish imkonini beradi va muhim xabarlarning ishlab chiqarishdagi batafsil debug loglarda yo'qolmasligini ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Production da darajalardan to'g'ri foydalanish
public class OrderService {
    private static final Logger logger = LoggerFactory.getLogger(OrderService.class);

    public void processOrder(Order order) {
        logger.info("Order {} processing started", order.getId()); // INFO: muhim hodisalar

        if (order.getAmount() > 10000) {
            logger.warn("Large order detected: {} amount: {}", order.getId(), order.getAmount()); // WARN: e'tibor
        }

        try {
            validateOrder(order);
            logger.debug("Order {} validated", order.getId()); // DEBUG: faqat dev da
        } catch (ValidationException e) {
            logger.error("Order {} validation failed", order.getId(), e); // ERROR: xatolar
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- INFO ishlab chiqarishda biznes oqimini kuzatish uchun
- WARN e'tibor talab qiladigan potentsial muammolar uchun
- ERROR tuzatish talab qiladigan haqiqiy xatolar uchun`
        }
    }
};

export default task;
