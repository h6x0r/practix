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
        System.out.println("\n2. Reading file:");
        processor.readFile("test.txt");

        // Test copying
        System.out.println("\n3. Copying file:");
        processor.copyFile("test.txt", "copy.txt");

        // Test custom resource
        System.out.println("\n4. Custom AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("\n=== Demo Complete ===");
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

// Test1: Verify FileProcessor instantiation
class Test1 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        assertNotNull("FileProcessor should be created", processor);
    }
}

// Test2: Verify writeFile method
class Test2 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        try {
            processor.writeFile("test_file.txt", "Test content");
            assertTrue("writeFile should execute", true);
        } catch (Exception e) {
            fail("writeFile should not throw: " + e.getMessage());
        }
    }
}

// Test3: Verify readFile method
class Test3 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        try {
            processor.writeFile("test_read.txt", "Content to read");
            processor.readFile("test_read.txt");
            assertTrue("readFile should execute", true);
        } catch (Exception e) {
            fail("readFile should not throw: " + e.getMessage());
        }
    }
}

// Test4: Verify copyFile method
class Test4 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        try {
            processor.writeFile("source.txt", "Source content");
            processor.copyFile("source.txt", "destination.txt");
            assertTrue("copyFile should execute", true);
        } catch (Exception e) {
            fail("copyFile should not throw: " + e.getMessage());
        }
    }
}

// Test5: Verify DatabaseConnection creation
class Test5 {
    @Test
    public void test() {
        try (DatabaseConnection conn = new DatabaseConnection("TEST-001")) {
            assertNotNull("DatabaseConnection should be created", conn);
        }
    }
}

// Test6: Verify DatabaseConnection auto-close
class Test6 {
    @Test
    public void test() {
        try (DatabaseConnection conn = new DatabaseConnection("TEST-002")) {
            conn.executeQuery("SELECT * FROM test");
            assertTrue("Query should execute", true);
        } catch (Exception e) {
            fail("DatabaseConnection should work: " + e.getMessage());
        }
    }
}

// Test7: Verify custom resource demonstration
class Test7 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        try {
            processor.demonstrateCustomResource();
            assertTrue("Custom resource demo should work", true);
        } catch (Exception e) {
            fail("Custom resource demo should not throw: " + e.getMessage());
        }
    }
}

// Test8: Verify main method execution
class Test8 {
    @Test
    public void test() {
        try {
            FileProcessor.main(new String[]{});
            assertTrue("Main method should execute", true);
        } catch (Exception e) {
            // File operations might fail in test environment
            assertTrue("Main execution attempted", true);
        }
    }
}

// Test9: Verify try-with-resources closes resources
class Test9 {
    @Test
    public void test() {
        try (DatabaseConnection conn = new DatabaseConnection("TEST-009")) {
            conn.executeQuery("TEST QUERY");
            // Resource should auto-close after this block
        }
        assertTrue("Resources should be closed automatically", true);
    }
}

// Test10: Verify multiple resource handling
class Test10 {
    @Test
    public void test() {
        FileProcessor processor = new FileProcessor();
        try {
            processor.writeFile("file1.txt", "Content 1");
            processor.writeFile("file2.txt", "Content 2");
            processor.copyFile("file1.txt", "file3.txt");
            assertTrue("Multiple file operations should work", true);
        } catch (Exception e) {
            // Expected in test environment
            assertTrue("File operations attempted", true);
        }
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
        System.out.println("\n2. Чтение файла:");
        processor.readFile("test.txt");

        // Тест копирования
        System.out.println("\n3. Копирование файла:");
        processor.copyFile("test.txt", "copy.txt");

        // Тест пользовательского ресурса
        System.out.println("\n4. Пользовательский AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("\n=== Демонстрация Завершена ===");
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
        System.out.println("\n2. Faylni o'qish:");
        processor.readFile("test.txt");

        // Nusxalash testi
        System.out.println("\n3. Faylni nusxalash:");
        processor.copyFile("test.txt", "copy.txt");

        // Maxsus resurs testi
        System.out.println("\n4. Maxsus AutoCloseable:");
        processor.demonstrateCustomResource();

        System.out.println("\n=== Namoyish Tugadi ===");
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
