import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-singleton-pattern',
    title: 'Singleton Pattern',
    difficulty: 'easy',
    tags: ['java', 'design-patterns', 'creational', 'singleton', 'thread-safety'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Singleton Pattern

The Singleton pattern ensures a class has only one instance and provides a global point of access to it. In Java, there are several ways to implement a thread-safe singleton, including enum-based singletons which are the most robust.

## Requirements:
1. Implement traditional thread-safe singleton:
   1.1. Private constructor
   1.2. Private static instance
   1.3. Public static getInstance() method with double-checked locking

2. Implement enum-based singleton:
   2.1. Single INSTANCE element
   2.2. Methods for business logic

3. Demonstrate usage:
   3.1. Verify same instance is returned
   3.2. Show both implementations work correctly
   3.3. Include thread-safety demonstration

4. Add configuration management example

## Example Output:
\`\`\`
=== Traditional Singleton ===
Database instance 1: DatabaseConnection@1a2b3c
Database instance 2: DatabaseConnection@1a2b3c
Same instance: true
Connected to: jdbc:mysql://localhost:3306/mydb

=== Enum Singleton ===
Config instance 1: AppConfig@4d5e6f
Config instance 2: AppConfig@4d5e6f
Same instance: true
App Name: MyApplication
App Version: 1.0.0
\`\`\``,
    initialCode: `// TODO: Create traditional thread-safe singleton

// TODO: Create enum-based singleton

public class SingletonPattern {
    public static void main(String[] args) {
        // TODO: Test traditional singleton

        // TODO: Test enum singleton
    }
}`,
    solutionCode: `// Traditional thread-safe singleton with double-checked locking
class DatabaseConnection {
    // Volatile ensures visibility across threads
    private static volatile DatabaseConnection instance;
    private String connectionString;

    // Private constructor prevents external instantiation
    private DatabaseConnection() {
        // Simulate expensive initialization
        connectionString = "jdbc:mysql://localhost:3306/mydb";
    }

    // Thread-safe lazy initialization with double-checked locking
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }

    public void connect() {
        System.out.println("Connected to: " + connectionString);
    }
}

// Enum singleton - thread-safe and serialization-safe by default
enum AppConfig {
    INSTANCE;

    private String appName;
    private String version;

    // Constructor called only once when enum is initialized
    AppConfig() {
        appName = "MyApplication";
        version = "1.0.0";
    }

    public String getAppName() {
        return appName;
    }

    public String getVersion() {
        return version;
    }

    public void displayConfig() {
        System.out.println("App Name: " + appName);
        System.out.println("App Version: " + version);
    }
}

public class SingletonPattern {
    public static void main(String[] args) {
        System.out.println("=== Traditional Singleton ===");

        // Get singleton instances
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();

        // Verify same instance
        System.out.println("Database instance 1: " + db1);
        System.out.println("Database instance 2: " + db2);
        System.out.println("Same instance: " + (db1 == db2));
        db1.connect();

        System.out.println("\\n=== Enum Singleton ===");

        // Get enum singleton instances
        AppConfig config1 = AppConfig.INSTANCE;
        AppConfig config2 = AppConfig.INSTANCE;

        // Verify same instance
        System.out.println("Config instance 1: " + config1);
        System.out.println("Config instance 2: " + config2);
        System.out.println("Same instance: " + (config1 == config2));
        config1.displayConfig();
    }
}`,
    hint1: `For thread-safe singleton, use volatile keyword and double-checked locking: if (instance == null) { synchronized { if (instance == null) { instance = new... } } }`,
    hint2: `Enum singleton is the simplest: enum MySingleton { INSTANCE; }. It's thread-safe and serialization-safe by default.`,
    whyItMatters: `Singleton pattern is crucial for managing shared resources like database connections, configuration settings, and logging. Understanding thread-safe implementations prevents bugs in concurrent applications. Enum-based singletons are the modern, preferred approach in Java as they handle serialization and reflection attacks automatically.

**Production Pattern:**
\`\`\`java
// Thread-safe singleton for connection management
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;

    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
}

// Enum singleton - modern approach
enum AppConfig {
    INSTANCE;
    // Automatically thread-safe and protected from serialization
}
\`\`\`

**Practical Benefits:**
- Guarantees single instance of critical resources
- Enum-based singleton protected from reflection attacks
- Double-checked locking optimizes performance
- Simplifies global application state management`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: DatabaseConnection.getInstance() returns non-null
class Test1 {
    @Test
    void testGetInstanceNotNull() {
        DatabaseConnection db = DatabaseConnection.getInstance();
        assertNotNull(db);
    }
}

// Test2: Multiple getInstance() calls return same instance
class Test2 {
    @Test
    void testSameInstance() {
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();
        assertSame(db1, db2);
    }
}

// Test3: AppConfig.INSTANCE returns non-null
class Test3 {
    @Test
    void testEnumSingletonNotNull() {
        AppConfig config = AppConfig.INSTANCE;
        assertNotNull(config);
    }
}

// Test4: Enum singleton returns same instance
class Test4 {
    @Test
    void testEnumSameInstance() {
        AppConfig config1 = AppConfig.INSTANCE;
        AppConfig config2 = AppConfig.INSTANCE;
        assertSame(config1, config2);
    }
}

// Test5: AppConfig has correct app name
class Test5 {
    @Test
    void testAppName() {
        assertEquals("MyApplication", AppConfig.INSTANCE.getAppName());
    }
}

// Test6: AppConfig has version
class Test6 {
    @Test
    void testVersion() {
        assertNotNull(AppConfig.INSTANCE.getVersion());
    }
}

// Test7: DatabaseConnection connect method exists
class Test7 {
    @Test
    void testConnectMethod() {
        DatabaseConnection db = DatabaseConnection.getInstance();
        assertDoesNotThrow(() -> db.connect());
    }
}

// Test8: Traditional singleton is thread-safe
class Test8 {
    @Test
    void testThreadSafety() throws InterruptedException {
        final DatabaseConnection[] instances = new DatabaseConnection[2];
        Thread t1 = new Thread(() -> instances[0] = DatabaseConnection.getInstance());
        Thread t2 = new Thread(() -> instances[1] = DatabaseConnection.getInstance());
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        assertSame(instances[0], instances[1]);
    }
}

// Test9: AppConfig displayConfig doesn't throw
class Test9 {
    @Test
    void testDisplayConfig() {
        assertDoesNotThrow(() -> AppConfig.INSTANCE.displayConfig());
    }
}

// Test10: Version has correct format
class Test10 {
    @Test
    void testVersionFormat() {
        String version = AppConfig.INSTANCE.getVersion();
        assertTrue(version.matches("\\\\d+\\\\.\\\\d+\\\\.\\\\d+"));
    }
}
`,
    translations: {
        ru: {
            title: 'Паттерн Singleton',
            solutionCode: `// Традиционный потокобезопасный singleton с двойной проверкой блокировки
class DatabaseConnection {
    // Volatile обеспечивает видимость между потоками
    private static volatile DatabaseConnection instance;
    private String connectionString;

    // Приватный конструктор предотвращает внешнее создание экземпляров
    private DatabaseConnection() {
        // Имитация дорогостоящей инициализации
        connectionString = "jdbc:mysql://localhost:3306/mydb";
    }

    // Потокобезопасная ленивая инициализация с двойной проверкой блокировки
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }

    public void connect() {
        System.out.println("Connected to: " + connectionString);
    }
}

// Enum singleton - потокобезопасный и безопасный для сериализации по умолчанию
enum AppConfig {
    INSTANCE;

    private String appName;
    private String version;

    // Конструктор вызывается только один раз при инициализации enum
    AppConfig() {
        appName = "MyApplication";
        version = "1.0.0";
    }

    public String getAppName() {
        return appName;
    }

    public String getVersion() {
        return version;
    }

    public void displayConfig() {
        System.out.println("App Name: " + appName);
        System.out.println("App Version: " + version);
    }
}

public class SingletonPattern {
    public static void main(String[] args) {
        System.out.println("=== Традиционный Singleton ===");

        // Получение singleton экземпляров
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();

        // Проверка того же экземпляра
        System.out.println("Database instance 1: " + db1);
        System.out.println("Database instance 2: " + db2);
        System.out.println("Same instance: " + (db1 == db2));
        db1.connect();

        System.out.println("\\n=== Enum Singleton ===");

        // Получение enum singleton экземпляров
        AppConfig config1 = AppConfig.INSTANCE;
        AppConfig config2 = AppConfig.INSTANCE;

        // Проверка того же экземпляра
        System.out.println("Config instance 1: " + config1);
        System.out.println("Config instance 2: " + config2);
        System.out.println("Same instance: " + (config1 == config2));
        config1.displayConfig();
    }
}`,
            description: `# Паттерн Singleton

Паттерн Singleton гарантирует, что класс имеет только один экземпляр и предоставляет глобальную точку доступа к нему. В Java есть несколько способов реализовать потокобезопасный singleton, включая enum-based singleton, которые являются наиболее надежными.

## Требования:
1. Реализуйте традиционный потокобезопасный singleton:
   1.1. Приватный конструктор
   1.2. Приватный статический экземпляр
   1.3. Публичный статический метод getInstance() с двойной проверкой блокировки

2. Реализуйте enum-based singleton:
   2.1. Единственный элемент INSTANCE
   2.2. Методы для бизнес-логики

3. Продемонстрируйте использование:
   3.1. Проверьте, что возвращается тот же экземпляр
   3.2. Покажите, что обе реализации работают корректно
   3.3. Включите демонстрацию потокобезопасности

4. Добавьте пример управления конфигурацией

## Пример вывода:
\`\`\`
=== Traditional Singleton ===
Database instance 1: DatabaseConnection@1a2b3c
Database instance 2: DatabaseConnection@1a2b3c
Same instance: true
Connected to: jdbc:mysql://localhost:3306/mydb

=== Enum Singleton ===
Config instance 1: AppConfig@4d5e6f
Config instance 2: AppConfig@4d5e6f
Same instance: true
App Name: MyApplication
App Version: 1.0.0
\`\`\``,
            hint1: `Для потокобезопасного singleton используйте ключевое слово volatile и двойную проверку блокировки: if (instance == null) { synchronized { if (instance == null) { instance = new... } } }`,
            hint2: `Enum singleton самый простой: enum MySingleton { INSTANCE; }. Он потокобезопасен и безопасен для сериализации по умолчанию.`,
            whyItMatters: `Паттерн Singleton критически важен для управления общими ресурсами, такими как подключения к базе данных, настройки конфигурации и логирование. Понимание потокобезопасных реализаций предотвращает ошибки в параллельных приложениях.

**Продакшен паттерн:**
\`\`\`java
// Потокобезопасный singleton для управления подключениями
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;

    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
}

// Enum singleton - современный подход
enum AppConfig {
    INSTANCE;
    // Автоматически потокобезопасен и защищен от сериализации
}
\`\`\`

**Практические преимущества:**
- Гарантирует единственный экземпляр критичных ресурсов
- Enum-based singleton защищен от атак через рефлексию
- Двойная проверка блокировки оптимизирует производительность
- Упрощает управление глобальным состоянием приложения`
        },
        uz: {
            title: `Singleton namunasi`,
            solutionCode: `// Ikki tomonlama tekshiruvli an'anaviy potok-xavfsiz singleton
class DatabaseConnection {
    // Volatile oqimlar o'rtasida ko'rinishni ta'minlaydi
    private static volatile DatabaseConnection instance;
    private String connectionString;

    // Xususiy konstruktor tashqi yaratishni oldini oladi
    private DatabaseConnection() {
        // Qimmat initsializatsiyani simulyatsiya qilish
        connectionString = "jdbc:mysql://localhost:3306/mydb";
    }

    // Ikki tomonlama tekshiruvli potok-xavfsiz lazy initsializatsiya
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }

    public void connect() {
        System.out.println("Connected to: " + connectionString);
    }
}

// Enum singleton - standart bo'yicha potok-xavfsiz va serializatsiya-xavfsiz
enum AppConfig {
    INSTANCE;

    private String appName;
    private String version;

    // Konstruktor enum ishga tushirilganda faqat bir marta chaqiriladi
    AppConfig() {
        appName = "MyApplication";
        version = "1.0.0";
    }

    public String getAppName() {
        return appName;
    }

    public String getVersion() {
        return version;
    }

    public void displayConfig() {
        System.out.println("App Name: " + appName);
        System.out.println("App Version: " + version);
    }
}

public class SingletonPattern {
    public static void main(String[] args) {
        System.out.println("=== An'anaviy Singleton ===");

        // Singleton misollarini olish
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();

        // Bir xil misolni tekshirish
        System.out.println("Database instance 1: " + db1);
        System.out.println("Database instance 2: " + db2);
        System.out.println("Same instance: " + (db1 == db2));
        db1.connect();

        System.out.println("\\n=== Enum Singleton ===");

        // Enum singleton misollarini olish
        AppConfig config1 = AppConfig.INSTANCE;
        AppConfig config2 = AppConfig.INSTANCE;

        // Bir xil misolni tekshirish
        System.out.println("Config instance 1: " + config1);
        System.out.println("Config instance 2: " + config2);
        System.out.println("Same instance: " + (config1 == config2));
        config1.displayConfig();
    }
}`,
            description: `# Singleton namunasi

Singleton namunasi klass faqat bitta misolga ega ekanligini kafolatlaydi va unga global kirish nuqtasini taqdim etadi. Java-da potok-xavfsiz singleton-ni amalga oshirishning bir necha usullari mavjud, shu jumladan eng mustahkam bo'lgan enum-asoslangan singleton.

## Talablar:
1. An'anaviy potok-xavfsiz singleton-ni amalga oshiring:
   1.1. Xususiy konstruktor
   1.2. Xususiy statik misol
   1.3. Ikki tomonlama tekshiruvli public static getInstance() metodi

2. Enum-asoslangan singleton-ni amalga oshiring:
   2.1. Yagona INSTANCE elementi
   2.2. Biznes mantiq uchun metodlar

3. Foydalanishni namoyish eting:
   3.1. Bir xil misol qaytarilishini tekshiring
   3.2. Ikkala amalga oshirish to'g'ri ishlashini ko'rsating
   3.3. Potok-xavfsizlik namoyishini qo'shing

4. Konfiguratsiyani boshqarish misolini qo'shing

## Chiqish namunasi:
\`\`\`
=== Traditional Singleton ===
Database instance 1: DatabaseConnection@1a2b3c
Database instance 2: DatabaseConnection@1a2b3c
Same instance: true
Connected to: jdbc:mysql://localhost:3306/mydb

=== Enum Singleton ===
Config instance 1: AppConfig@4d5e6f
Config instance 2: AppConfig@4d5e6f
Same instance: true
App Name: MyApplication
App Version: 1.0.0
\`\`\``,
            hint1: `Potok-xavfsiz singleton uchun volatile kalit so'z va ikki tomonlama tekshiruvni ishlating: if (instance == null) { synchronized { if (instance == null) { instance = new... } } }`,
            hint2: `Enum singleton eng oddiy: enum MySingleton { INSTANCE; }. U standart bo'yicha potok-xavfsiz va serializatsiya-xavfsiz.`,
            whyItMatters: `Singleton namunasi ma'lumotlar bazasi ulanishlari, konfiguratsiya sozlamalari va loglar kabi umumiy resurslarni boshqarish uchun juda muhimdir. Potok-xavfsiz amalga oshirishlarni tushunish parallel ilovalarda xatolarning oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ulanishlarni boshqarish uchun potok-xavfsiz singleton
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;

    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
}

// Enum singleton - zamonaviy yondashuv
enum AppConfig {
    INSTANCE;
    // Avtomatik ravishda potok-xavfsiz va serializatsiyadan himoyalangan
}
\`\`\`

**Amaliy foydalari:**
- Muhim resurslarning yagona misolini kafolatlaydi
- Enum-asoslangan singleton reflection hujumlaridan himoyalangan
- Ikki tomonlama tekshiruv samaradorlikni optimallashtiradi
- Global dastur holatini boshqarishni soddalashtiradi`
        }
    }
};

export default task;
