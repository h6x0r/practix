import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-try-with-resources',
    title: 'Try-With-Resources (AutoCloseable)',
    difficulty: 'medium',
    tags: ['java', 'exceptions', 'resources', 'autocloseable', 'io'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement file operations using try-with-resources for automatic resource management.

Requirements:
1. Create FileProcessor class with methods to read and write files
2. Use try-with-resources for automatic resource cleanup
3. Implement a custom AutoCloseable class (DatabaseConnection)
4. Handle multiple resources in one try-with-resources statement
5. Demonstrate proper exception handling with automatic resource closing

Example:
\`\`\`java
FileProcessor processor = new FileProcessor();
processor.copyFile("input.txt", "output.txt");
// Resources automatically closed, even if exception occurs
\`\`\``,
    initialCode: `import java.io.*;

// TODO: Create DatabaseConnection class implementing AutoCloseable

public class FileProcessor {

    public void readFile(String filename) {
        // TODO: Use try-with-resources to read file
    }

    public void writeFile(String filename, String content) {
        // TODO: Use try-with-resources to write file
    }

    public void copyFile(String source, String destination) {
        // TODO: Use try-with-resources with multiple resources
    }

    public static void main(String[] args) {
        FileProcessor processor = new FileProcessor();

        // Test writing
        processor.writeFile("test.txt", "Hello, Java!");

        // Test reading
        processor.readFile("test.txt");

        // Test copying
        processor.copyFile("test.txt", "copy.txt");
    }
}`,
    solutionCode: `import java.io.*;

// Custom AutoCloseable class
class DatabaseConnection implements AutoCloseable {
    private String connectionId;
    private boolean isOpen;

    public DatabaseConnection(String connectionId) {
        this.connectionId = connectionId;
        this.isOpen = true;
        System.out.println("Opening database connection: " + connectionId);
    }

    public void executeQuery(String query) {
        if (!isOpen) {
            throw new IllegalStateException("Connection is closed");
        }
        System.out.println("Executing query: " + query);
    }

    @Override
    public void close() {
        // Automatically called by try-with-resources
        if (isOpen) {
            System.out.println("Closing database connection: " + connectionId);
            isOpen = false;
        }
    }
}

public class FileProcessor {

    public void readFile(String filename) {
        // Try-with-resources: BufferedReader automatically closed
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            System.out.println("Reading file: " + filename);
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("  " + line);
            }
            // No need to manually close reader
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + filename);
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
        // Reader is automatically closed here, even if exception occurred
    }

    public void writeFile(String filename, String content) {
        // Try-with-resources with BufferedWriter
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            System.out.println("Writing to file: " + filename);
            writer.write(content);
            writer.newLine();
            System.out.println("Write successful");
            // Writer automatically flushed and closed
        } catch (IOException e) {
            System.out.println("Error writing file: " + e.getMessage());
        }
    }

    public void copyFile(String source, String destination) {
        // Multiple resources in one try-with-resources
        // Resources closed in reverse order of declaration
        try (BufferedReader reader = new BufferedReader(new FileReader(source));
             BufferedWriter writer = new BufferedWriter(new FileWriter(destination))) {

            System.out.println("Copying " + source + " to " + destination);
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
            System.out.println("Copy successful");

        } catch (FileNotFoundException e) {
            System.out.println("Source file not found: " + source);
        } catch (IOException e) {
            System.out.println("Error during copy: " + e.getMessage());
        }
        // Both reader and writer automatically closed (writer first, then reader)
    }

    public void demonstrateCustomResource() {
        // Using custom AutoCloseable class
        try (DatabaseConnection conn = new DatabaseConnection("DB-001")) {
            conn.executeQuery("SELECT * FROM users");
            conn.executeQuery("SELECT * FROM orders");
            // Connection automatically closed when exiting try block
        }
        System.out.println("Database operations completed");
    }

    public static void main(String[] args) {
        FileProcessor processor = new FileProcessor();

        System.out.println("=== File Operations Demo ===\n");

        // Test writing
        System.out.println("1. Writing file:");
        processor.writeFile("test.txt", "Hello, Java! Try-with-resources is awesome!");

        // Test reading
        System.out.println("");
        System.out.println("2. Reading file:");
        processor.readFile("test.txt");

        // Test copying
        System.out.println("");
        System.out.println("3. Copying file:");
        processor.copyFile("test.txt", "copy.txt");

        // Test custom resource
        System.out.println("");
        System.out.println("4. Custom AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("");
        System.out.println("=== Demo Complete ===");
    }
}`,
    hint1: `Try-with-resources syntax: try (Resource r = new Resource()) { }. Any class implementing AutoCloseable can be used. The close() method is called automatically when exiting the try block.`,
    hint2: `You can declare multiple resources separated by semicolons. They are closed in reverse order of declaration. This ensures dependent resources are closed properly.`,
    whyItMatters: `Try-with-resources eliminates resource leaks by automatically closing resources. This is crucial for file handles, network connections, and database connections. It makes code cleaner and prevents common bugs caused by forgetting to close resources.

**Production Pattern:**
\`\`\`java
public class DataProcessor {
    public void processData(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath));
             Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(SQL_INSERT)) {

            String line;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                stmt.setString(1, line);
                stmt.addBatch();
                if (++count % 1000 == 0) {
                    stmt.executeBatch();
                }
            }
            stmt.executeBatch();
            conn.commit();
        } // All resources automatically closed
    }
}
\`\`\`

**Practical Benefits:**
- Guaranteed resource release
- Protection from memory leaks
- Cleaner and safer code`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: DatabaseConnection opens and outputs message
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try (DatabaseConnection conn = new DatabaseConnection("TEST-001")) {
            // do nothing, just test opening
        }
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show opening connection message",
            output.contains("Opening") || output.contains("Открытие") || output.contains("Ochish"));
    }
}

// Test2: DatabaseConnection auto-closes and prints close message
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try (DatabaseConnection conn = new DatabaseConnection("TEST-002")) {
            conn.executeQuery("SELECT 1");
        }
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show closing connection message",
            output.contains("Closing") || output.contains("Закрытие") || output.contains("Yopish"));
    }
}

// Test3: executeQuery prints query
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try (DatabaseConnection conn = new DatabaseConnection("TEST-003")) {
            conn.executeQuery("SELECT * FROM users");
        }
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show executing query",
            output.contains("Executing") || output.contains("SELECT") ||
            output.contains("Выполнение") || output.contains("Bajarish"));
    }
}

// Test4: FileProcessor can be instantiated
class Test4 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        assertNotNull("FileProcessor should be created", processor);
    }
}

// Test5: Custom resource demo shows auto-close
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        FileProcessor processor = new FileProcessor();
        processor.demonstrateCustomResource();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should demonstrate auto-close",
            output.contains("Closing") || output.contains("Закрытие") || output.contains("Yopish") ||
            output.contains("Connection opened") || output.contains("closed"));
    }
}

// Test6: main produces output with section headers
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Exception caught = null;
        try {
            FileProcessor.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        if (caught != null) {
            fail("FileProcessor.main should handle exceptions internally, but threw: " + caught.getMessage());
        }
        String output = out.toString();
        assertTrue("Should show demo header",
            output.contains("File Operations Demo") || output.contains("Demo Complete") ||
            output.contains("Демонстрация файловых операций") || output.contains("Fayl operatsiyalari"));
    }
}

// Test7: connection ID is preserved
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try (DatabaseConnection conn = new DatabaseConnection("MY-CONN-007")) {
            conn.executeQuery("TEST");
        }
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show connection ID", output.contains("MY-CONN-007"));
    }
}

// Test8: demo shows numbered steps
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Exception caught = null;
        try {
            FileProcessor.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        if (caught != null) {
            fail("FileProcessor.main should handle exceptions internally, but threw: " + caught.getMessage());
        }
        String output = out.toString();
        assertTrue("Should show numbered steps",
            (output.contains("1.") && output.contains("2.")) ||
            output.contains("Writing") || output.contains("Reading"));
    }
}

// Test9: resources closed in order
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        try (DatabaseConnection conn1 = new DatabaseConnection("FIRST");
             DatabaseConnection conn2 = new DatabaseConnection("SECOND")) {
            conn1.executeQuery("Q1");
            conn2.executeQuery("Q2");
        }
        System.setOut(oldOut);
        String output = out.toString();
        // Both should be closed
        assertTrue("Both connections should be closed",
            output.contains("FIRST") && output.contains("SECOND") &&
            (output.contains("Closing") || output.contains("Закрытие") || output.contains("Yopish")));
    }
}

// Test10: Demo Complete message shown
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        Exception caught = null;
        try {
            FileProcessor.main(new String[]{});
        } catch (Exception e) {
            caught = e;
        }
        System.setOut(oldOut);
        if (caught != null) {
            fail("FileProcessor.main should handle exceptions internally, but threw: " + caught.getMessage());
        }
        String output = out.toString();
        assertTrue("Should show demo complete or custom resource",
            output.contains("Demo Complete") || output.contains("Демонстрация завершена") ||
            output.contains("Namoyish yakunlandi") || output.contains("Custom") || output.contains("AutoCloseable"));
    }
}
`,
    translations: {
        ru: {
            title: 'Try-With-Resources (автозакрываемые ресурсы)',
            solutionCode: `import java.io.*;

// Пользовательский класс AutoCloseable
class DatabaseConnection implements AutoCloseable {
    private String connectionId;
    private boolean isOpen;

    public DatabaseConnection(String connectionId) {
        this.connectionId = connectionId;
        this.isOpen = true;
        System.out.println("Открытие соединения с базой данных: " + connectionId);
    }

    public void executeQuery(String query) {
        if (!isOpen) {
            throw new IllegalStateException("Соединение закрыто");
        }
        System.out.println("Выполнение запроса: " + query);
    }

    @Override
    public void close() {
        // Автоматически вызывается try-with-resources
        if (isOpen) {
            System.out.println("Закрытие соединения с базой данных: " + connectionId);
            isOpen = false;
        }
    }
}

public class FileProcessor {

    public void readFile(String filename) {
        // Try-with-resources: BufferedReader автоматически закрывается
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            System.out.println("Чтение файла: " + filename);
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("  " + line);
            }
            // Не нужно вручную закрывать reader
        } catch (FileNotFoundException e) {
            System.out.println("Файл не найден: " + filename);
        } catch (IOException e) {
            System.out.println("Ошибка чтения файла: " + e.getMessage());
        }
        // Reader автоматически закрывается здесь, даже если произошло исключение
    }

    public void writeFile(String filename, String content) {
        // Try-with-resources с BufferedWriter
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            System.out.println("Запись в файл: " + filename);
            writer.write(content);
            writer.newLine();
            System.out.println("Запись успешна");
            // Writer автоматически сбрасывается и закрывается
        } catch (IOException e) {
            System.out.println("Ошибка записи файла: " + e.getMessage());
        }
    }

    public void copyFile(String source, String destination) {
        // Множественные ресурсы в одном try-with-resources
        // Ресурсы закрываются в обратном порядке объявления
        try (BufferedReader reader = new BufferedReader(new FileReader(source));
             BufferedWriter writer = new BufferedWriter(new FileWriter(destination))) {

            System.out.println("Копирование " + source + " в " + destination);
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
            System.out.println("Копирование успешно");

        } catch (FileNotFoundException e) {
            System.out.println("Исходный файл не найден: " + source);
        } catch (IOException e) {
            System.out.println("Ошибка при копировании: " + e.getMessage());
        }
        // И reader, и writer автоматически закрываются (сначала writer, затем reader)
    }

    public void demonstrateCustomResource() {
        // Использование пользовательского класса AutoCloseable
        try (DatabaseConnection conn = new DatabaseConnection("DB-001")) {
            conn.executeQuery("SELECT * FROM users");
            conn.executeQuery("SELECT * FROM orders");
            // Соединение автоматически закрывается при выходе из блока try
        }
        System.out.println("Операции с базой данных завершены");
    }

    public static void main(String[] args) {
        FileProcessor processor = new FileProcessor();

        System.out.println("=== Демонстрация Операций с Файлами ===\n");

        // Тест записи
        System.out.println("1. Запись файла:");
        processor.writeFile("test.txt", "Привет, Java! Try-with-resources великолепен!");

        // Тест чтения
        System.out.println("");
        System.out.println("2. Чтение файла:");
        processor.readFile("test.txt");

        // Тест копирования
        System.out.println("");
        System.out.println("3. Копирование файла:");
        processor.copyFile("test.txt", "copy.txt");

        // Тест пользовательского ресурса
        System.out.println("");
        System.out.println("4. Пользовательский AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("");
        System.out.println("=== Демонстрация Завершена ===");
    }
}`,
            description: `Реализуйте операции с файлами, используя try-with-resources для автоматического управления ресурсами.

Требования:
1. Создайте класс FileProcessor с методами для чтения и записи файлов
2. Используйте try-with-resources для автоматической очистки ресурсов
3. Реализуйте пользовательский класс AutoCloseable (DatabaseConnection)
4. Обрабатывайте множественные ресурсы в одном операторе try-with-resources
5. Продемонстрируйте правильную обработку исключений с автоматическим закрытием ресурсов

Пример:
\`\`\`java
FileProcessor processor = new FileProcessor();
processor.copyFile("input.txt", "output.txt");
// Ресурсы автоматически закрыты, даже если произошло исключение
\`\`\``,
            hint1: `Синтаксис try-with-resources: try (Resource r = new Resource()) { }. Можно использовать любой класс, реализующий AutoCloseable. Метод close() вызывается автоматически при выходе из блока try.`,
            hint2: `Вы можете объявить множественные ресурсы, разделенные точкой с запятой. Они закрываются в обратном порядке объявления. Это гарантирует правильное закрытие зависимых ресурсов.`,
            whyItMatters: `Try-with-resources устраняет утечки ресурсов путем автоматического закрытия ресурсов. Это критически важно для файловых дескрипторов, сетевых соединений и соединений с базами данных. Это делает код чище и предотвращает распространенные ошибки, вызванные забыванием закрыть ресурсы.

**Продакшен паттерн:**
\`\`\`java
public class DataProcessor {
    public void processData(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath));
             Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(SQL_INSERT)) {

            String line;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                stmt.setString(1, line);
                stmt.addBatch();
                if (++count % 1000 == 0) {
                    stmt.executeBatch();
                }
            }
            stmt.executeBatch();
            conn.commit();
        } // Все ресурсы автоматически закрыты
    }
}
\`\`\`

**Практические преимущества:**
- Гарантированное освобождение ресурсов
- Защита от утечек памяти
- Более чистый и безопасный код`
        },
        uz: {
            title: `Try-With-Resources (avtomatik yopiladigan resurslar)`,
            solutionCode: `import java.io.*;

// Maxsus AutoCloseable klassi
class DatabaseConnection implements AutoCloseable {
    private String connectionId;
    private boolean isOpen;

    public DatabaseConnection(String connectionId) {
        this.connectionId = connectionId;
        this.isOpen = true;
        System.out.println("Ma'lumotlar bazasi ulanishini ochish: " + connectionId);
    }

    public void executeQuery(String query) {
        if (!isOpen) {
            throw new IllegalStateException("Ulanish yopilgan");
        }
        System.out.println("So'rovni bajarish: " + query);
    }

    @Override
    public void close() {
        // Try-with-resources tomonidan avtomatik chaqiriladi
        if (isOpen) {
            System.out.println("Ma'lumotlar bazasi ulanishini yopish: " + connectionId);
            isOpen = false;
        }
    }
}

public class FileProcessor {

    public void readFile(String filename) {
        // Try-with-resources: BufferedReader avtomatik yopiladi
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            System.out.println("Faylni o'qish: " + filename);
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("  " + line);
            }
            // Reader ni qo'lda yopish kerak emas
        } catch (FileNotFoundException e) {
            System.out.println("Fayl topilmadi: " + filename);
        } catch (IOException e) {
            System.out.println("Faylni o'qishda xato: " + e.getMessage());
        }
        // Reader bu yerda avtomatik yopiladi, istisno yuz bergan bo'lsa ham
    }

    public void writeFile(String filename, String content) {
        // BufferedWriter bilan try-with-resources
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            System.out.println("Faylga yozish: " + filename);
            writer.write(content);
            writer.newLine();
            System.out.println("Yozish muvaffaqiyatli");
            // Writer avtomatik tozalanadi va yopiladi
        } catch (IOException e) {
            System.out.println("Faylga yozishda xato: " + e.getMessage());
        }
    }

    public void copyFile(String source, String destination) {
        // Bitta try-with-resources da ko'p resurslar
        // Resurslar e'lon qilish tartibining teskari tartibida yopiladi
        try (BufferedReader reader = new BufferedReader(new FileReader(source));
             BufferedWriter writer = new BufferedWriter(new FileWriter(destination))) {

            System.out.println(source + " dan " + destination + " ga nusxalash");
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
            System.out.println("Nusxalash muvaffaqiyatli");

        } catch (FileNotFoundException e) {
            System.out.println("Manba fayl topilmadi: " + source);
        } catch (IOException e) {
            System.out.println("Nusxalashda xato: " + e.getMessage());
        }
        // Reader ham, writer ham avtomatik yopiladi (avval writer, keyin reader)
    }

    public void demonstrateCustomResource() {
        // Maxsus AutoCloseable klassini ishlatish
        try (DatabaseConnection conn = new DatabaseConnection("DB-001")) {
            conn.executeQuery("SELECT * FROM users");
            conn.executeQuery("SELECT * FROM orders");
            // Try blokidan chiqishda ulanish avtomatik yopiladi
        }
        System.out.println("Ma'lumotlar bazasi amallari yakunlandi");
    }

    public static void main(String[] args) {
        FileProcessor processor = new FileProcessor();

        System.out.println("=== Fayl Amallari Namoyishi ===\n");

        // Yozish testi
        System.out.println("1. Faylga yozish:");
        processor.writeFile("test.txt", "Salom, Java! Try-with-resources ajoyib!");

        // O'qish testi
        System.out.println("");
        System.out.println("2. Faylni o'qish:");
        processor.readFile("test.txt");

        // Nusxalash testi
        System.out.println("");
        System.out.println("3. Faylni nusxalash:");
        processor.copyFile("test.txt", "copy.txt");

        // Maxsus resurs testi
        System.out.println("");
        System.out.println("4. Maxsus AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("");
        System.out.println("=== Namoyish Tugadi ===");
    }
}`,
            description: `Avtomatik resurs boshqaruvi uchun try-with-resources dan foydalanib, fayl operatsiyalarini amalga oshiring.

Talablar:
1. Fayllarni o'qish va yozish metodlari bilan FileProcessor klassini yarating
2. Avtomatik resurslarni tozalash uchun try-with-resources dan foydalaning
3. Maxsus AutoCloseable klassini (DatabaseConnection) yarating
4. Bitta try-with-resources operatorida ko'p resurslarni qayta ishlang
5. Avtomatik resurs yopish bilan to'g'ri istisno qayta ishlashni ko'rsating

Misol:
\`\`\`java
FileProcessor processor = new FileProcessor();
processor.copyFile("input.txt", "output.txt");
// Resurslar avtomatik yopiladi, istisno yuz bergan bo'lsa ham
\`\`\``,
            hint1: `Try-with-resources sintaksisi: try (Resource r = new Resource()) { }. AutoCloseable ni amalga oshiruvchi har qanday klassni ishlatish mumkin. close() metodi try blokidan chiqishda avtomatik chaqiriladi.`,
            hint2: `Ko'p resurslarni nuqta-vergul bilan ajratib e'lon qilishingiz mumkin. Ular e'lon qilish tartibining teskari tartibida yopiladi. Bu bog'liq resurslarning to'g'ri yopilishini ta'minlaydi.`,
            whyItMatters: `Try-with-resources resurslarni avtomatik yopish orqali resurs oqishlarini bartaraf etadi. Bu fayl deskriptorlari, tarmoq ulanishlari va ma'lumotlar bazasi ulanishlari uchun juda muhimdir. Bu kodni tozaroq qiladi va resurslarni yopishni unutish natijasida yuzaga keladigan umumiy xatolarning oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
public class DataProcessor {
    public void processData(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath));
             Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(SQL_INSERT)) {

            String line;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                stmt.setString(1, line);
                stmt.addBatch();
                if (++count % 1000 == 0) {
                    stmt.executeBatch();
                }
            }
            stmt.executeBatch();
            conn.commit();
        } // Barcha resurslar avtomatik yopiladi
    }
}
\`\`\`

**Amaliy foydalari:**
- Resurslarning ozod qilinishi kafolatlangan
- Xotira oqishidan himoya
- Tozaroq va xavfsizroq kod`
        }
    }
};

export default task;
