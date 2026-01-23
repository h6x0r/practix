import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-default-methods',
    title: 'Default Methods in Interfaces',
    difficulty: 'medium',
    tags: ['java', 'interfaces', 'default-methods', 'java8'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Default Methods in Interfaces

Java 8 introduced default methods in interfaces - methods with a default implementation. This allows adding new functionality to interfaces without breaking existing implementations.

## Requirements:
1. Create a \`Logger\` interface with:
   1.1. Abstract method: \`void log(String message)\`
   1.2. Default method: \`void logError(String message)\` that prefixes with "ERROR: "
   1.3. Default method: \`void logWarning(String message)\` that prefixes with "WARNING: "

2. Create a \`ConsoleLogger\` class that:
   2.1. Implements \`Logger\`
   2.2. Overrides only \`log()\` to print to console
   2.3. Uses inherited default methods

3. Create a \`FileLogger\` class that:
   3.1. Implements \`Logger\`
   3.2. Overrides \`log()\` to simulate file writing
   3.3. Overrides \`logError()\` to add timestamp and write to "error log"

4. Demonstrate both loggers with all methods

## Example Output:
\`\`\`
[Console] Hello World
ERROR: [Console] Database connection failed
WARNING: [Console] Low memory

[File] Application started
ERROR: [2024-01-15 10:30:45] [File] Critical failure
WARNING: [File] Deprecated API used
\`\`\``,
    initialCode: `// TODO: Create Logger interface with abstract and default methods

// TODO: Create ConsoleLogger class

// TODO: Create FileLogger class that overrides default method

public class DefaultMethods {
    public static void main(String[] args) {
        // TODO: Test ConsoleLogger

        // TODO: Test FileLogger
    }
}`,
    solutionCode: `// Logger interface with abstract and default methods
interface Logger {
    // Abstract method - must be implemented
    void log(String message);

    // Default method - provides default implementation
    default void logError(String message) {
        log("ERROR: " + message);
    }

    // Another default method
    default void logWarning(String message) {
        log("WARNING: " + message);
    }
}

// ConsoleLogger uses default implementations
class ConsoleLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[Console] " + message);
    }
    // logError and logWarning are inherited from interface
}

// FileLogger overrides a default method
class FileLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[File] " + message);
    }

    // Override default method to add custom behavior
    @Override
    public void logError(String message) {
        // Simulate timestamp
        String timestamp = java.time.LocalDateTime.now()
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        log("ERROR: [" + timestamp + "] " + message);
    }
    // logWarning still uses default implementation
}

public class DefaultMethods {
    public static void main(String[] args) {
        System.out.println("=== Console Logger ===");
        Logger console = new ConsoleLogger();
        console.log("Hello World");
        console.logError("Database connection failed");
        console.logWarning("Low memory");

        System.out.println("\\n=== File Logger ===");
        Logger file = new FileLogger();
        file.log("Application started");
        file.logError("Critical failure");
        file.logWarning("Deprecated API used");
    }
}`,
    hint1: `Use the 'default' keyword before the method return type to create a default method: default void methodName() { }`,
    hint2: `Default methods can call abstract methods of the same interface. When overriding, you can still call the log() method to utilize the custom implementation.`,
    whyItMatters: `Default methods solved a major problem in Java: how to evolve interfaces without breaking existing code. Before Java 8, adding a method to an interface broke all implementing classes. Default methods enable backward compatibility while adding new functionality. This feature was crucial for adding lambda support to existing Java Collections interfaces.`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;

// Test 1: ConsoleLogger implements Logger
class Test1 {
    @Test
    void testConsoleLoggerImplementsLogger() {
        Logger logger = new ConsoleLogger();
        assertNotNull(logger);
        assertTrue(logger instanceof Logger);
    }
}

// Test 2: FileLogger implements Logger
class Test2 {
    @Test
    void testFileLoggerImplementsLogger() {
        Logger logger = new FileLogger();
        assertNotNull(logger);
        assertTrue(logger instanceof Logger);
    }
}

// Test 3: ConsoleLogger log method works
class Test3 {
    @Test
    void testConsoleLoggerLog() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new ConsoleLogger();
        logger.log("Test message");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("Console") || output.contains("Test message"));
    }
}

// Test 4: ConsoleLogger uses default logError
class Test4 {
    @Test
    void testConsoleLoggerDefaultLogError() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new ConsoleLogger();
        logger.logError("Error occurred");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("ERROR"));
    }
}

// Test 5: ConsoleLogger uses default logWarning
class Test5 {
    @Test
    void testConsoleLoggerDefaultLogWarning() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new ConsoleLogger();
        logger.logWarning("Warning message");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("WARNING"));
    }
}

// Test 6: FileLogger log method works
class Test6 {
    @Test
    void testFileLoggerLog() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new FileLogger();
        logger.log("File message");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("File") || output.contains("message"));
    }
}

// Test 7: FileLogger overrides logError with timestamp
class Test7 {
    @Test
    void testFileLoggerOverridesLogError() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new FileLogger();
        logger.logError("Critical error");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("ERROR"));
    }
}

// Test 8: FileLogger uses default logWarning
class Test8 {
    @Test
    void testFileLoggerDefaultLogWarning() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        Logger logger = new FileLogger();
        logger.logWarning("Warning");

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("WARNING"));
    }
}

// Test 9: Both loggers can be used polymorphically
class Test9 {
    @Test
    void testPolymorphicUsage() {
        Logger[] loggers = { new ConsoleLogger(), new FileLogger() };

        for (Logger logger : loggers) {
            assertDoesNotThrow(() -> logger.log("Test"));
            assertDoesNotThrow(() -> logger.logError("Error"));
            assertDoesNotThrow(() -> logger.logWarning("Warning"));
        }
    }
}

// Test 10: Default methods are inherited
class Test10 {
    @Test
    void testDefaultMethodsInherited() {
        ConsoleLogger console = new ConsoleLogger();
        assertDoesNotThrow(() -> console.logError("test"));
        assertDoesNotThrow(() -> console.logWarning("test"));
    }
}`,
    translations: {
        ru: {
            title: 'Методы по умолчанию в интерфейсах',
            solutionCode: `// Интерфейс Logger с абстрактными и методами по умолчанию
interface Logger {
    // Абстрактный метод - должен быть реализован
    void log(String message);

    // Метод по умолчанию - предоставляет реализацию по умолчанию
    default void logError(String message) {
        log("ERROR: " + message);
    }

    // Еще один метод по умолчанию
    default void logWarning(String message) {
        log("WARNING: " + message);
    }
}

// ConsoleLogger использует реализации по умолчанию
class ConsoleLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[Console] " + message);
    }
    // logError и logWarning наследуются из интерфейса
}

// FileLogger переопределяет метод по умолчанию
class FileLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[File] " + message);
    }

    // Переопределяем метод по умолчанию для добавления пользовательского поведения
    @Override
    public void logError(String message) {
        // Имитируем временную метку
        String timestamp = java.time.LocalDateTime.now()
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        log("ERROR: [" + timestamp + "] " + message);
    }
    // logWarning все еще использует реализацию по умолчанию
}

public class DefaultMethods {
    public static void main(String[] args) {
        System.out.println("=== Console Logger ===");
        Logger console = new ConsoleLogger();
        console.log("Hello World");
        console.logError("Database connection failed");
        console.logWarning("Low memory");

        System.out.println("\\n=== File Logger ===");
        Logger file = new FileLogger();
        file.log("Application started");
        file.logError("Critical failure");
        file.logWarning("Deprecated API used");
    }
}`,
            description: `# Методы по умолчанию в интерфейсах

Java 8 представила методы по умолчанию в интерфейсах - методы с реализацией по умолчанию. Это позволяет добавлять новую функциональность к интерфейсам без нарушения существующих реализаций.

## Требования:
1. Создайте интерфейс \`Logger\` с:
   1.1. Абстрактным методом: \`void log(String message)\`
   1.2. Методом по умолчанию: \`void logError(String message)\`, который добавляет префикс "ERROR: "
   1.3. Методом по умолчанию: \`void logWarning(String message)\`, который добавляет префикс "WARNING: "

2. Создайте класс \`ConsoleLogger\`, который:
   2.1. Реализует \`Logger\`
   2.2. Переопределяет только \`log()\` для вывода в консоль
   2.3. Использует унаследованные методы по умолчанию

3. Создайте класс \`FileLogger\`, который:
   3.1. Реализует \`Logger\`
   3.2. Переопределяет \`log()\` для имитации записи в файл
   3.3. Переопределяет \`logError()\` для добавления временной метки и записи в "лог ошибок"

4. Продемонстрируйте оба логгера со всеми методами

## Пример вывода:
\`\`\`
[Console] Hello World
ERROR: [Console] Database connection failed
WARNING: [Console] Low memory

[File] Application started
ERROR: [2024-01-15 10:30:45] [File] Critical failure
WARNING: [File] Deprecated API used
\`\`\``,
            hint1: `Используйте ключевое слово 'default' перед типом возврата метода для создания метода по умолчанию: default void methodName() { }`,
            hint2: `Методы по умолчанию могут вызывать абстрактные методы того же интерфейса. При переопределении вы все еще можете вызвать метод log() для использования пользовательской реализации.`,
            whyItMatters: `Методы по умолчанию решили важную проблему в Java: как развивать интерфейсы без нарушения существующего кода. До Java 8 добавление метода в интерфейс ломало все реализующие классы. Методы по умолчанию обеспечивают обратную совместимость при добавлении новой функциональности. Эта возможность была критически важна для добавления поддержки лямбда-выражений в существующие интерфейсы Java Collections.`
        },
        uz: {
            title: `Interfeyslarда standart metodlar`,
            solutionCode: `// Logger interfeysi abstrakt va standart metodlar bilan
interface Logger {
    // Abstrakt metod - amalga oshirilishi shart
    void log(String message);

    // Standart metod - standart implementatsiyani taqdim etadi
    default void logError(String message) {
        log("ERROR: " + message);
    }

    // Yana bir standart metod
    default void logWarning(String message) {
        log("WARNING: " + message);
    }
}

// ConsoleLogger standart implementatsiyalardan foydalanadi
class ConsoleLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[Console] " + message);
    }
    // logError va logWarning interfeysdan meros olinadi
}

// FileLogger standart metodini qayta yozadi
class FileLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println("[File] " + message);
    }

    // Maxsus xatti-harakatni qo'shish uchun standart metodini qayta yozamiz
    @Override
    public void logError(String message) {
        // Vaqt tamg'asini taqlid qilamiz
        String timestamp = java.time.LocalDateTime.now()
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        log("ERROR: [" + timestamp + "] " + message);
    }
    // logWarning hali ham standart implementatsiyadan foydalanadi
}

public class DefaultMethods {
    public static void main(String[] args) {
        System.out.println("=== Console Logger ===");
        Logger console = new ConsoleLogger();
        console.log("Hello World");
        console.logError("Database connection failed");
        console.logWarning("Low memory");

        System.out.println("\\n=== File Logger ===");
        Logger file = new FileLogger();
        file.log("Application started");
        file.logError("Critical failure");
        file.logWarning("Deprecated API used");
    }
}`,
            description: `# Interfeyslarда standart metodlar

Java 8 interfeyslarга standart metodlarni - standart implementatsiyaga ega metodlarni kiritdi. Bu mavjud implementatsiyalarni buzmasdan interfeyslarга yangi funksionallik qo'shishga imkon beradi.

## Talablar:
1. \`Logger\` interfeysini yarating:
   1.1. Abstrakt metod: \`void log(String message)\`
   1.2. Standart metod: \`void logError(String message)\` "ERROR: " prefiksini qo'shadi
   1.3. Standart metod: \`void logWarning(String message)\` "WARNING: " prefiksini qo'shadi

2. \`ConsoleLogger\` klassini yarating:
   2.1. \`Logger\` ni amalga oshiradi
   2.2. Faqat \`log()\` ni konsolga chiqarish uchun qayta yozadi
   2.3. Meros olingan standart metodlardan foydalanadi

3. \`FileLogger\` klassini yarating:
   3.1. \`Logger\` ni amalga oshiradi
   3.2. \`log()\` ni faylga yozishni taqlid qilish uchun qayta yozadi
   3.3. \`logError()\` ni vaqt tamg'asini qo'shish va "xato jurnaliga" yozish uchun qayta yozadi

4. Barcha metodlar bilan ikkala loggerni namoyish eting

## Chiqish namunasi:
\`\`\`
[Console] Hello World
ERROR: [Console] Database connection failed
WARNING: [Console] Low memory

[File] Application started
ERROR: [2024-01-15 10:30:45] [File] Critical failure
WARNING: [File] Deprecated API used
\`\`\``,
            hint1: `Standart metod yaratish uchun metod qaytish turidan oldin 'default' kalit so'zidan foydalaning: default void methodName() { }`,
            hint2: `Standart metodlar bir xil interfeysdagi abstrakt metodlarni chaqirishi mumkin. Qayta yozishda maxsus implementatsiyadan foydalanish uchun hali ham log() metodini chaqirishingiz mumkin.`,
            whyItMatters: `Standart metodlar Java-da muhim muammoni hal qildi: mavjud kodni buzmasdan interfeyslarni qanday rivojlantirish. Java 8 dan oldin interfeysga metod qo'shish barcha amalga oshiruvchi klasslarni buzardi. Standart metodlar yangi funksionallikni qo'shishda orqaga qarab moslikni ta'minlaydi. Bu xususiyat mavjud Java Collections interfeyslariga lambda qo'llab-quvvatlashni qo'shish uchun juda muhim edi.`
        }
    }
};

export default task;
