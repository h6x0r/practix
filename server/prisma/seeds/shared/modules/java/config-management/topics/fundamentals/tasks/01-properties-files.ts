import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-properties-files',
    title: 'Properties Files',
    difficulty: 'easy',
    tags: ['java', 'configuration', 'properties', 'io', 'file-handling'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Properties Files

The Properties class in Java provides a way to store and retrieve configuration data as key-value pairs. Properties files use the .properties extension and are commonly used for application configuration, internationalization, and settings management.

## Requirements:
1. Create a Properties object and add configuration entries:
   1.1. Use setProperty() to add key-value pairs
   1.2. Support different data types (strings, numbers, booleans)

2. Save properties to a file:
   2.1. Use store() method with FileOutputStream
   2.2. Add descriptive comments to the properties file

3. Load properties from a file:
   3.1. Use load() method with FileInputStream
   3.2. Handle file not found scenarios gracefully

4. Retrieve and use property values:
   4.1. Use getProperty() with default values
   4.2. Display all properties using propertyNames()

5. Demonstrate practical configuration:
   5.1. Database connection settings
   5.2. Application settings (host, port, timeout)

## Example Output:
\`\`\`
=== Creating Properties ===
Properties created and saved to config.properties

=== Loading Properties ===
Database URL: jdbc:mysql://localhost:3306/mydb
Database User: root
Database Password: secret123
Application Host: localhost
Application Port: 8080
Connection Timeout: 30
Debug Mode: true

=== All Properties ===
db.url=jdbc:mysql://localhost:3306/mydb
db.user=root
db.password=secret123
app.host=localhost
app.port=8080
app.timeout=30
app.debug=true
\`\`\``,
    initialCode: `import java.io.*;
import java.util.*;

public class PropertiesExample {
    public static void main(String[] args) {
        // TODO: Create Properties object and add configuration

        // TODO: Save properties to file

        // TODO: Load properties from file

        // TODO: Retrieve and display property values

        // TODO: Display all properties
    }
}`,
    solutionCode: `import java.io.*;
import java.util.*;

public class PropertiesExample {
    public static void main(String[] args) {
        String filename = "config.properties";

        // Create and save properties
        createProperties(filename);

        // Load and display properties
        loadProperties(filename);
    }

    private static void createProperties(String filename) {
        Properties props = new Properties();

        // Add database configuration
        props.setProperty("db.url", "jdbc:mysql://localhost:3306/mydb");
        props.setProperty("db.user", "root");
        props.setProperty("db.password", "secret123");

        // Add application configuration
        props.setProperty("app.host", "localhost");
        props.setProperty("app.port", "8080");
        props.setProperty("app.timeout", "30");
        props.setProperty("app.debug", "true");

        // Save to file
        try (FileOutputStream out = new FileOutputStream(filename)) {
            props.store(out, "Application Configuration");
            System.out.println("=== Creating Properties ===");
            System.out.println("Properties created and saved to " + filename);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Error saving properties: " + e.getMessage());
        }
    }

    private static void loadProperties(String filename) {
        Properties props = new Properties();

        // Load from file
        try (FileInputStream in = new FileInputStream(filename)) {
            props.load(in);

            System.out.println("=== Loading Properties ===");

            // Retrieve specific properties
            String dbUrl = props.getProperty("db.url");
            String dbUser = props.getProperty("db.user");
            String dbPassword = props.getProperty("db.password");
            String appHost = props.getProperty("app.host");
            String appPort = props.getProperty("app.port", "8080"); // with default
            String appTimeout = props.getProperty("app.timeout");
            String appDebug = props.getProperty("app.debug");

            // Display properties
            System.out.println("Database URL: " + dbUrl);
            System.out.println("Database User: " + dbUser);
            System.out.println("Database Password: " + dbPassword);
            System.out.println("Application Host: " + appHost);
            System.out.println("Application Port: " + appPort);
            System.out.println("Connection Timeout: " + appTimeout);
            System.out.println("Debug Mode: " + appDebug);
            System.out.println();

            // Display all properties
            System.out.println("=== All Properties ===");
            Enumeration<?> names = props.propertyNames();
            while (names.hasMoreElements()) {
                String key = (String) names.nextElement();
                String value = props.getProperty(key);
                System.out.println(key + "=" + value);
            }

        } catch (FileNotFoundException e) {
            System.err.println("Properties file not found: " + filename);
        } catch (IOException e) {
            System.err.println("Error loading properties: " + e.getMessage());
        }
    }
}`,
    hint1: `Use the Properties class with setProperty() to add key-value pairs, store() to save to a file, and load() to read from a file.`,
    hint2: `Remember to use try-with-resources for FileInputStream and FileOutputStream to ensure proper resource cleanup.`,
    whyItMatters: `Properties files are a fundamental way to configure Java applications. They provide a simple, human-readable format for storing configuration data that can be easily modified without recompiling code. This is essential for managing different environments (development, testing, production) and allowing users to customize application behavior. Understanding properties files is crucial for building maintainable and flexible Java applications.

**Production Pattern:**
\`\`\`java
// Loading configuration from properties file
Properties props = new Properties();
try (FileInputStream in = new FileInputStream("config.properties")) {
    props.load(in);

    // Getting values with default settings
    String dbUrl = props.getProperty("db.url");
    String appPort = props.getProperty("app.port", "8080");
    int timeout = Integer.parseInt(props.getProperty("app.timeout", "30"));

    // Using configuration
    System.out.println("Database: " + dbUrl);
    System.out.println("Port: " + appPort);
}
\`\`\`

**Practical Benefits:**
- Change settings without recompiling code
- Simple support for different environments (dev, staging, prod)
- Readable format understandable to non-developers`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.Properties;
import java.io.*;

// Test1: Verify Properties object creation
class Test1 {
    @Test
    public void test() {
        Properties props = new Properties();
        assertNotNull(props);
        assertTrue(props.isEmpty());
    }
}

// Test2: Verify setProperty adds properties
class Test2 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("key", "value");
        assertEquals("value", props.getProperty("key"));
    }
}

// Test3: Verify multiple properties can be set
class Test3 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("db.url", "jdbc:mysql://localhost:3306/mydb");
        props.setProperty("db.user", "root");
        props.setProperty("db.password", "secret");
        assertEquals(3, props.size());
    }
}

// Test4: Verify getProperty returns correct value
class Test4 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("app.host", "localhost");
        props.setProperty("app.port", "8080");
        assertEquals("localhost", props.getProperty("app.host"));
        assertEquals("8080", props.getProperty("app.port"));
    }
}

// Test5: Verify getProperty with default value
class Test5 {
    @Test
    public void test() {
        Properties props = new Properties();
        String value = props.getProperty("missing.key", "default");
        assertEquals("default", value);
    }
}

// Test6: Verify properties can be saved to file
class Test6 {
    @Test
    public void test() throws IOException {
        Properties props = new Properties();
        props.setProperty("test.key", "test.value");

        File tempFile = File.createTempFile("test", ".properties");
        try (FileOutputStream out = new FileOutputStream(tempFile)) {
            props.store(out, "Test properties");
        }

        assertTrue(tempFile.exists());
        assertTrue(tempFile.length() > 0);
        tempFile.delete();
    }
}

// Test7: Verify properties can be loaded from file
class Test7 {
    @Test
    public void test() throws IOException {
        Properties saveProps = new Properties();
        saveProps.setProperty("loaded.key", "loaded.value");

        File tempFile = File.createTempFile("test", ".properties");
        try (FileOutputStream out = new FileOutputStream(tempFile)) {
            saveProps.store(out, "Test");
        }

        Properties loadProps = new Properties();
        try (FileInputStream in = new FileInputStream(tempFile)) {
            loadProps.load(in);
        }

        assertEquals("loaded.value", loadProps.getProperty("loaded.key"));
        tempFile.delete();
    }
}

// Test8: Verify integer property conversion
class Test8 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("app.port", "8080");
        props.setProperty("app.timeout", "30");

        int port = Integer.parseInt(props.getProperty("app.port"));
        int timeout = Integer.parseInt(props.getProperty("app.timeout"));

        assertEquals(8080, port);
        assertEquals(30, timeout);
    }
}

// Test9: Verify boolean property conversion
class Test9 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("app.debug", "true");
        props.setProperty("app.secure", "false");

        boolean debug = Boolean.parseBoolean(props.getProperty("app.debug"));
        boolean secure = Boolean.parseBoolean(props.getProperty("app.secure"));

        assertTrue(debug);
        assertFalse(secure);
    }
}

// Test10: Verify containsKey method
class Test10 {
    @Test
    public void test() {
        Properties props = new Properties();
        props.setProperty("existing.key", "value");

        assertTrue(props.containsKey("existing.key"));
        assertFalse(props.containsKey("non.existing.key"));
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Файлы свойств',
            solutionCode: `import java.io.*;
import java.util.*;

public class PropertiesExample {
    public static void main(String[] args) {
        String filename = "config.properties";

        // Создать и сохранить свойства
        createProperties(filename);

        // Загрузить и отобразить свойства
        loadProperties(filename);
    }

    private static void createProperties(String filename) {
        Properties props = new Properties();

        // Добавить конфигурацию базы данных
        props.setProperty("db.url", "jdbc:mysql://localhost:3306/mydb");
        props.setProperty("db.user", "root");
        props.setProperty("db.password", "secret123");

        // Добавить конфигурацию приложения
        props.setProperty("app.host", "localhost");
        props.setProperty("app.port", "8080");
        props.setProperty("app.timeout", "30");
        props.setProperty("app.debug", "true");

        // Сохранить в файл
        try (FileOutputStream out = new FileOutputStream(filename)) {
            props.store(out, "Application Configuration");
            System.out.println("=== Создание свойств ===");
            System.out.println("Properties created and saved to " + filename);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Error saving properties: " + e.getMessage());
        }
    }

    private static void loadProperties(String filename) {
        Properties props = new Properties();

        // Загрузить из файла
        try (FileInputStream in = new FileInputStream(filename)) {
            props.load(in);

            System.out.println("=== Загрузка свойств ===");

            // Получить конкретные свойства
            String dbUrl = props.getProperty("db.url");
            String dbUser = props.getProperty("db.user");
            String dbPassword = props.getProperty("db.password");
            String appHost = props.getProperty("app.host");
            String appPort = props.getProperty("app.port", "8080"); // со значением по умолчанию
            String appTimeout = props.getProperty("app.timeout");
            String appDebug = props.getProperty("app.debug");

            // Отобразить свойства
            System.out.println("Database URL: " + dbUrl);
            System.out.println("Database User: " + dbUser);
            System.out.println("Database Password: " + dbPassword);
            System.out.println("Application Host: " + appHost);
            System.out.println("Application Port: " + appPort);
            System.out.println("Connection Timeout: " + appTimeout);
            System.out.println("Debug Mode: " + appDebug);
            System.out.println();

            // Отобразить все свойства
            System.out.println("=== Все свойства ===");
            Enumeration<?> names = props.propertyNames();
            while (names.hasMoreElements()) {
                String key = (String) names.nextElement();
                String value = props.getProperty(key);
                System.out.println(key + "=" + value);
            }

        } catch (FileNotFoundException e) {
            System.err.println("Properties file not found: " + filename);
        } catch (IOException e) {
            System.err.println("Error loading properties: " + e.getMessage());
        }
    }
}`,
            description: `# Файлы свойств

Класс Properties в Java предоставляет способ хранения и получения конфигурационных данных в виде пар ключ-значение. Файлы свойств используют расширение .properties и обычно используются для конфигурации приложений, интернационализации и управления настройками.

## Требования:
1. Создайте объект Properties и добавьте записи конфигурации:
   1.1. Используйте setProperty() для добавления пар ключ-значение
   1.2. Поддержка различных типов данных (строки, числа, логические значения)

2. Сохраните свойства в файл:
   2.1. Используйте метод store() с FileOutputStream
   2.2. Добавьте описательные комментарии в файл свойств

3. Загрузите свойства из файла:
   3.1. Используйте метод load() с FileInputStream
   3.2. Корректно обрабатывайте ситуации, когда файл не найден

4. Получите и используйте значения свойств:
   4.1. Используйте getProperty() со значениями по умолчанию
   4.2. Отобразите все свойства используя propertyNames()

5. Продемонстрируйте практическую конфигурацию:
   5.1. Настройки подключения к базе данных
   5.2. Настройки приложения (хост, порт, таймаут)

## Пример вывода:
\`\`\`
=== Creating Properties ===
Properties created and saved to config.properties

=== Loading Properties ===
Database URL: jdbc:mysql://localhost:3306/mydb
Database User: root
Database Password: secret123
Application Host: localhost
Application Port: 8080
Connection Timeout: 30
Debug Mode: true

=== All Properties ===
db.url=jdbc:mysql://localhost:3306/mydb
db.user=root
db.password=secret123
app.host=localhost
app.port=8080
app.timeout=30
app.debug=true
\`\`\``,
            hint1: `Используйте класс Properties с setProperty() для добавления пар ключ-значение, store() для сохранения в файл и load() для чтения из файла.`,
            hint2: `Не забудьте использовать try-with-resources для FileInputStream и FileOutputStream, чтобы обеспечить правильную очистку ресурсов.`,
            whyItMatters: `Файлы свойств являются фундаментальным способом конфигурации Java-приложений. Они предоставляют простой, читаемый человеком формат для хранения конфигурационных данных, которые можно легко изменять без перекомпиляции кода. Это необходимо для управления различными средами (разработка, тестирование, продакшн) и предоставления пользователям возможности настраивать поведение приложения. Понимание файлов свойств имеет решающее значение для создания поддерживаемых и гибких Java-приложений.

**Продакшен паттерн:**
\`\`\`java
// Загрузка конфигурации из properties файла
Properties props = new Properties();
try (FileInputStream in = new FileInputStream("config.properties")) {
    props.load(in);

    // Получение значений с настройками по умолчанию
    String dbUrl = props.getProperty("db.url");
    String appPort = props.getProperty("app.port", "8080");
    int timeout = Integer.parseInt(props.getProperty("app.timeout", "30"));

    // Использование конфигурации
    System.out.println("Database: " + dbUrl);
    System.out.println("Port: " + appPort);
}
\`\`\`

**Практические преимущества:**
- Изменение настроек без перекомпиляции кода
- Простая поддержка различных окружений (dev, staging, prod)
- Читаемый формат, понятный для не-разработчиков`
        },
        uz: {
            title: `Xususiyat fayllari`,
            solutionCode: `import java.io.*;
import java.util.*;

public class PropertiesExample {
    public static void main(String[] args) {
        String filename = "config.properties";

        // Xususiyatlarni yaratish va saqlash
        createProperties(filename);

        // Xususiyatlarni yuklash va ko'rsatish
        loadProperties(filename);
    }

    private static void createProperties(String filename) {
        Properties props = new Properties();

        // Ma'lumotlar bazasi konfiguratsiyasini qo'shish
        props.setProperty("db.url", "jdbc:mysql://localhost:3306/mydb");
        props.setProperty("db.user", "root");
        props.setProperty("db.password", "secret123");

        // Ilova konfiguratsiyasini qo'shish
        props.setProperty("app.host", "localhost");
        props.setProperty("app.port", "8080");
        props.setProperty("app.timeout", "30");
        props.setProperty("app.debug", "true");

        // Faylga saqlash
        try (FileOutputStream out = new FileOutputStream(filename)) {
            props.store(out, "Application Configuration");
            System.out.println("=== Xususiyatlarni yaratish ===");
            System.out.println("Properties created and saved to " + filename);
            System.out.println();
        } catch (IOException e) {
            System.err.println("Error saving properties: " + e.getMessage());
        }
    }

    private static void loadProperties(String filename) {
        Properties props = new Properties();

        // Fayldan yuklash
        try (FileInputStream in = new FileInputStream(filename)) {
            props.load(in);

            System.out.println("=== Xususiyatlarni yuklash ===");

            // Aniq xususiyatlarni olish
            String dbUrl = props.getProperty("db.url");
            String dbUser = props.getProperty("db.user");
            String dbPassword = props.getProperty("db.password");
            String appHost = props.getProperty("app.host");
            String appPort = props.getProperty("app.port", "8080"); // standart qiymat bilan
            String appTimeout = props.getProperty("app.timeout");
            String appDebug = props.getProperty("app.debug");

            // Xususiyatlarni ko'rsatish
            System.out.println("Database URL: " + dbUrl);
            System.out.println("Database User: " + dbUser);
            System.out.println("Database Password: " + dbPassword);
            System.out.println("Application Host: " + appHost);
            System.out.println("Application Port: " + appPort);
            System.out.println("Connection Timeout: " + appTimeout);
            System.out.println("Debug Mode: " + appDebug);
            System.out.println();

            // Barcha xususiyatlarni ko'rsatish
            System.out.println("=== Barcha xususiyatlar ===");
            Enumeration<?> names = props.propertyNames();
            while (names.hasMoreElements()) {
                String key = (String) names.nextElement();
                String value = props.getProperty(key);
                System.out.println(key + "=" + value);
            }

        } catch (FileNotFoundException e) {
            System.err.println("Properties file not found: " + filename);
        } catch (IOException e) {
            System.err.println("Error loading properties: " + e.getMessage());
        }
    }
}`,
            description: `# Xususiyat fayllari

Java-dagi Properties klassi konfiguratsiya ma'lumotlarini kalit-qiymat juftliklari sifatida saqlash va olish usulini taqdim etadi. Xususiyat fayllari .properties kengaytmasidan foydalanadi va odatda ilova konfiguratsiyasi, xalqarolashtirish va sozlamalarni boshqarish uchun ishlatiladi.

## Talablar:
1. Properties obyektini yarating va konfiguratsiya yozuvlarini qo'shing:
   1.1. Kalit-qiymat juftliklarini qo'shish uchun setProperty() dan foydalaning
   1.2. Turli ma'lumot turlarini qo'llab-quvvatlash (satrlar, raqamlar, mantiqiy qiymatlar)

2. Xususiyatlarni faylga saqlang:
   2.1. FileOutputStream bilan store() metodidan foydalaning
   2.2. Xususiyat fayliga tavsiflovchi izohlar qo'shing

3. Xususiyatlarni fayldan yuklang:
   3.1. FileInputStream bilan load() metodidan foydalaning
   3.2. Fayl topilmagan holatlarni to'g'ri boshqaring

4. Xususiyat qiymatlarini oling va ishlating:
   4.1. Standart qiymatlar bilan getProperty() dan foydalaning
   4.2. propertyNames() yordamida barcha xususiyatlarni ko'rsating

5. Amaliy konfiguratsiyani namoyish eting:
   5.1. Ma'lumotlar bazasi ulanish sozlamalari
   5.2. Ilova sozlamalari (host, port, timeout)

## Chiqish namunasi:
\`\`\`
=== Creating Properties ===
Properties created and saved to config.properties

=== Loading Properties ===
Database URL: jdbc:mysql://localhost:3306/mydb
Database User: root
Database Password: secret123
Application Host: localhost
Application Port: 8080
Connection Timeout: 30
Debug Mode: true

=== All Properties ===
db.url=jdbc:mysql://localhost:3306/mydb
db.user=root
db.password=secret123
app.host=localhost
app.port=8080
app.timeout=30
app.debug=true
\`\`\``,
            hint1: `Kalit-qiymat juftliklarini qo'shish uchun setProperty(), faylga saqlash uchun store() va fayldan o'qish uchun load() bilan Properties klassidan foydalaning.`,
            hint2: `Resurslarni to'g'ri tozalashni ta'minlash uchun FileInputStream va FileOutputStream uchun try-with-resources dan foydalanishni unutmang.`,
            whyItMatters: `Xususiyat fayllari Java ilovalarini sozlashning asosiy usulidir. Ular kodni qayta kompilyatsiya qilmasdan osonlikcha o'zgartirilishi mumkin bo'lgan konfiguratsiya ma'lumotlarini saqlash uchun oddiy, inson o'qiy oladigan formatni taqdim etadi. Bu turli muhitlarni (ishlab chiqish, test, ishlab chiqarish) boshqarish va foydalanuvchilarga ilova xatti-harakatini sozlash imkoniyatini berish uchun zarur. Xususiyat fayllarini tushunish qo'llab-quvvatlanadigan va moslashuvchan Java ilovalarini yaratish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Properties faylidan konfiguratsiyani yuklash
Properties props = new Properties();
try (FileInputStream in = new FileInputStream("config.properties")) {
    props.load(in);

    // Standart sozlamalar bilan qiymatlarni olish
    String dbUrl = props.getProperty("db.url");
    String appPort = props.getProperty("app.port", "8080");
    int timeout = Integer.parseInt(props.getProperty("app.timeout", "30"));

    // Konfiguratsiyadan foydalanish
    System.out.println("Database: " + dbUrl);
    System.out.println("Port: " + appPort);
}
\`\`\`

**Amaliy foydalari:**
- Kodni qayta kompilyatsiya qilmasdan sozlamalarni o'zgartirish
- Turli muhitlarni (dev, staging, prod) oson qo'llab-quvvatlash
- Dasturchи bo'lmaganlarga tushunarli o'qiladigan format`
        }
    }
};

export default task;
