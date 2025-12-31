import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-singleton',
	title: 'Singleton Pattern',
	difficulty: 'easy',
	tags: ['java', 'design-patterns', 'creational', 'singleton'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Singleton pattern in Java - ensure a class has only one instance and provide a global point of access.

**You will implement:**

1. **Thread-safe Singleton** using double-checked locking
2. **Bill Pugh Singleton** using inner static class
3. **Enum Singleton** - the recommended approach

**Example Usage:**

\`\`\`java
DatabaseConnection db1 = DatabaseConnection.getInstance();	// get instance
DatabaseConnection db2 = DatabaseConnection.getInstance();	// get same instance
System.out.println(db1 == db2);	// true - same object

Logger logger = Logger.getInstance();	// lazy initialization via inner class
logger.log("Application started");	// use the singleton

ConfigManager config = ConfigManager.INSTANCE;	// enum singleton access
config.setEnvironment("development");	// modify singleton state
\`\`\``,
	initialCode: `class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    private String connectionString;

    private DatabaseConnection() {
    }

    public static DatabaseConnection getInstance() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    public String getConnectionString() {
    }
}

class Logger {
    private Logger() {}

    public static Logger getInstance() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    public String log(String message) {
    }
}

enum ConfigManager {

    private String environment = "production";

    public String getEnvironment() {
    }

    public void setEnvironment(String env) {
    }
}`,
	solutionCode: `// Thread-safe Singleton with double-checked locking
class DatabaseConnection {	// singleton class for database access
    private static volatile DatabaseConnection instance;	// volatile ensures visibility across threads
    private String connectionString;	// connection configuration

    private DatabaseConnection() {	// private constructor prevents external instantiation
        this.connectionString = "jdbc:mysql://localhost:3306/db";	// initialize connection string
    }

    public static DatabaseConnection getInstance() {	// global access point
        if (instance == null) {	// first check without locking (performance optimization)
            synchronized (DatabaseConnection.class) {	// lock for thread safety
                if (instance == null) {	// second check inside synchronized block
                    instance = new DatabaseConnection();	// create instance only once
                }
            }
        }
        return instance;	// return the singleton instance
    }

    public String getConnectionString() {	// accessor method
        return connectionString;	// return stored connection string
    }
}

// Bill Pugh Singleton (lazy initialization using inner class)
class Logger {	// singleton logger class
    private Logger() {}	// private constructor

    private static class SingletonHelper {	// inner class loaded on first access
        private static final Logger INSTANCE = new Logger();	// instance created when class loads
    }

    public static Logger getInstance() {	// lazy initialization without synchronization
        return SingletonHelper.INSTANCE;	// returns instance from inner class
    }

    public String log(String message) {	// logging method
        return "LOG: " + message;	// format and return log message
    }
}

// Enum Singleton (recommended by Joshua Bloch)
enum ConfigManager {	// enum guarantees single instance
    INSTANCE;	// the singleton instance - JVM ensures uniqueness

    private String environment = "production";	// singleton state

    public String getEnvironment() {	// getter for environment
        return environment;	// return current environment
    }

    public void setEnvironment(String env) {	// setter for environment
        this.environment = env;	// update environment value
    }
}`,
	hint1: `**Understanding Double-Checked Locking:**

The pattern uses two null checks with synchronized block for thread safety:

\`\`\`java
public static DatabaseConnection getInstance() {
    if (instance == null) {	// first check - avoid synchronization overhead
        synchronized (DatabaseConnection.class) {	// lock the class
            if (instance == null) {	// second check - ensure only one creation
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

The \`volatile\` keyword prevents instruction reordering and ensures all threads see the updated value immediately.`,
	hint2: `**Bill Pugh and Enum Singletons:**

\`\`\`java
// Bill Pugh - uses JVM class loading for thread safety
class Logger {
    private static class SingletonHelper {
        private static final Logger INSTANCE = new Logger();
    }

    public static Logger getInstance() {
        return SingletonHelper.INSTANCE;	// inner class loaded on first call
    }
}

// Enum - simplest and safest (recommended by Effective Java)
enum ConfigManager {
    INSTANCE;	// JVM guarantees single instance

    private String value;	// enum can have fields and methods
}
\`\`\`

Enum singleton is serialization-safe and reflection-proof automatically.`,
	whyItMatters: `## Why Singleton Exists

**The Problem: Uncontrolled Instance Creation**

Without Singleton, multiple instances waste resources and cause inconsistency:

\`\`\`java
// ❌ WITHOUT SINGLETON - multiple instances
DatabaseConnection conn1 = new DatabaseConnection();	// opens connection
DatabaseConnection conn2 = new DatabaseConnection();	// opens another connection
DatabaseConnection conn3 = new DatabaseConnection();	// and another...
// 100 threads = 100 connections = resource exhaustion!

// ✅ WITH SINGLETON - controlled single instance
DatabaseConnection conn1 = DatabaseConnection.getInstance();	// creates one connection
DatabaseConnection conn2 = DatabaseConnection.getInstance();	// returns same connection
DatabaseConnection conn3 = DatabaseConnection.getInstance();	// still same connection
// All threads share one connection pool
\`\`\`

---

## Real-World Examples in Java

**1. java.lang.Runtime:**
\`\`\`java
// JVM uses singleton for Runtime
Runtime runtime = Runtime.getRuntime();	// always same instance
runtime.availableProcessors();
\`\`\`

**2. java.awt.Desktop:**
\`\`\`java
// Desktop singleton for system integration
Desktop desktop = Desktop.getDesktop();
desktop.browse(new URI("https://example.com"));
\`\`\`

**3. Spring Framework Beans:**
\`\`\`java
// Spring beans are singleton by default
@Service
public class UserService { }	// one instance per application context
\`\`\`

**4. SLF4J Logger Factory:**
\`\`\`java
// Loggers are typically singletons per class
Logger logger = LoggerFactory.getLogger(MyClass.class);
\`\`\`

---

## Production Pattern: Thread-Safe Configuration Manager

\`\`\`java
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Properties;
import java.io.InputStream;

public final class AppConfig {
    // Volatile for double-checked locking
    private static volatile AppConfig instance;

    // Thread-safe configuration storage
    private final Map<String, String> config;
    private final String environment;

    // Private constructor
    private AppConfig() {
        this.config = new ConcurrentHashMap<>();
        this.environment = loadEnvironment();
        loadConfiguration();
    }

    // Double-checked locking for lazy initialization
    public static AppConfig getInstance() {
        if (instance == null) {
            synchronized (AppConfig.class) {
                if (instance == null) {
                    instance = new AppConfig();
                }
            }
        }
        return instance;
    }

    // Load environment from system property or default
    private String loadEnvironment() {
        return System.getProperty("app.env", "development");
    }

    // Load configuration from properties file
    private void loadConfiguration() {
        String fileName = "config-" + environment + ".properties";
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream(fileName)) {
            if (input != null) {
                Properties props = new Properties();
                props.load(input);
                props.forEach((k, v) -> config.put(k.toString(), v.toString()));
            }
        } catch (Exception e) {
            // Log and use defaults
            setDefaults();
        }
    }

    private void setDefaults() {
        config.put("db.host", "localhost");
        config.put("db.port", "5432");
        config.put("cache.enabled", "true");
    }

    // Thread-safe getters
    public String get(String key) {
        return config.get(key);
    }

    public String get(String key, String defaultValue) {
        return config.getOrDefault(key, defaultValue);
    }

    public int getInt(String key, int defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        return Boolean.parseBoolean(value);
    }

    public String getEnvironment() {
        return environment;
    }

    // Prevent cloning
    @Override
    protected Object clone() throws CloneNotSupportedException {
        throw new CloneNotSupportedException("Singleton cannot be cloned");
    }

    // For testing - reset instance (use with caution)
    static void resetForTesting() {
        instance = null;
    }
}

// Usage
public class Application {
    public static void main(String[] args) {
        AppConfig config = AppConfig.getInstance();

        String dbHost = config.get("db.host", "localhost");
        int dbPort = config.getInt("db.port", 5432);
        boolean cacheEnabled = config.getBoolean("cache.enabled", true);

        System.out.println("Environment: " + config.getEnvironment());
        System.out.println("Database: " + dbHost + ":" + dbPort);
        System.out.println("Cache: " + (cacheEnabled ? "enabled" : "disabled"));
    }
}
\`\`\`

---

## Common Mistakes to Avoid

**1. Not Using Volatile:**
\`\`\`java
// ❌ WRONG - without volatile, threads may see partially constructed object
private static DatabaseConnection instance;	// not volatile!

// ✅ RIGHT - volatile ensures proper visibility
private static volatile DatabaseConnection instance;
\`\`\`

**2. Synchronizing Entire Method:**
\`\`\`java
// ❌ WRONG - synchronizing whole method is inefficient
public static synchronized DatabaseConnection getInstance() {
    if (instance == null) {
        instance = new DatabaseConnection();
    }
    return instance;	// lock held for every call!
}

// ✅ RIGHT - synchronize only when needed
public static DatabaseConnection getInstance() {
    if (instance == null) {	// fast path without lock
        synchronized (DatabaseConnection.class) {
            if (instance == null) {
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

**3. Ignoring Serialization:**
\`\`\`java
// ❌ WRONG - deserialization creates new instance
class BadSingleton implements Serializable {
    private static final BadSingleton INSTANCE = new BadSingleton();
}

// ✅ RIGHT - implement readResolve to preserve singleton
class GoodSingleton implements Serializable {
    private static final GoodSingleton INSTANCE = new GoodSingleton();

    protected Object readResolve() {
        return INSTANCE;	// return existing instance on deserialization
    }
}

// ✅ BEST - use enum (handles serialization automatically)
enum BestSingleton {
    INSTANCE;
}
\`\`\``,
	order: 0,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void databaseConnectionReturnsSameInstance() {
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();
        assertSame(db1, db2, "getInstance should return same instance");
    }
}

class Test2 {
    @Test
    void databaseConnectionNotNull() {
        DatabaseConnection db = DatabaseConnection.getInstance();
        assertNotNull(db, "getInstance should not return null");
    }
}

class Test3 {
    @Test
    void databaseConnectionHasConnectionString() {
        DatabaseConnection db = DatabaseConnection.getInstance();
        assertNotNull(db.getConnectionString(), "Connection string should not be null");
        assertTrue(db.getConnectionString().contains("jdbc"), "Should contain jdbc");
    }
}

class Test4 {
    @Test
    void loggerReturnsSameInstance() {
        Logger log1 = Logger.getInstance();
        Logger log2 = Logger.getInstance();
        assertSame(log1, log2, "Logger getInstance should return same instance");
    }
}

class Test5 {
    @Test
    void loggerLogReturnsFormattedMessage() {
        Logger logger = Logger.getInstance();
        String result = logger.log("Test message");
        assertTrue(result.contains("LOG"), "Log should contain LOG prefix");
        assertTrue(result.contains("Test message"), "Log should contain message");
    }
}

class Test6 {
    @Test
    void configManagerInstanceExists() {
        ConfigManager config = ConfigManager.INSTANCE;
        assertNotNull(config, "ConfigManager.INSTANCE should not be null");
    }
}

class Test7 {
    @Test
    void configManagerEnvironmentDefault() {
        ConfigManager config = ConfigManager.INSTANCE;
        assertNotNull(config.getEnvironment(), "Environment should not be null");
    }
}

class Test8 {
    @Test
    void configManagerSetEnvironment() {
        ConfigManager config = ConfigManager.INSTANCE;
        config.setEnvironment("test");
        assertEquals("test", config.getEnvironment(), "Environment should be settable");
    }
}

class Test9 {
    @Test
    void enumSingletonIsSameReference() {
        ConfigManager c1 = ConfigManager.INSTANCE;
        ConfigManager c2 = ConfigManager.INSTANCE;
        assertSame(c1, c2, "Enum singleton should be same reference");
    }
}

class Test10 {
    @Test
    void multipleCallsThreadSafe() {
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();
        DatabaseConnection db3 = DatabaseConnection.getInstance();
        assertSame(db1, db2, "All calls should return same instance");
        assertSame(db2, db3, "All calls should return same instance");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Singleton (Одиночка)',
			description: `Реализуйте паттерн Singleton на Java — гарантируйте единственный экземпляр класса с глобальной точкой доступа.

**Вы реализуете:**

1. **Потокобезопасный Singleton** с двойной проверкой блокировки
2. **Singleton Билла Пью** с использованием внутреннего статического класса
3. **Enum Singleton** — рекомендуемый подход

**Пример использования:**

\`\`\`java
DatabaseConnection db1 = DatabaseConnection.getInstance();	// получаем экземпляр
DatabaseConnection db2 = DatabaseConnection.getInstance();	// получаем тот же экземпляр
System.out.println(db1 == db2);	// true - один объект

Logger logger = Logger.getInstance();	// ленивая инициализация через внутренний класс
logger.log("Application started");	// используем singleton

ConfigManager config = ConfigManager.INSTANCE;	// доступ к enum singleton
config.setEnvironment("development");	// изменяем состояние singleton
\`\`\``,
			hint1: `**Понимание Double-Checked Locking:**

Паттерн использует две проверки на null с synchronized блоком для потокобезопасности:

\`\`\`java
public static DatabaseConnection getInstance() {
    if (instance == null) {	// первая проверка - избегаем накладных расходов синхронизации
        synchronized (DatabaseConnection.class) {	// блокируем класс
            if (instance == null) {	// вторая проверка - гарантируем единственное создание
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

Ключевое слово \`volatile\` предотвращает переупорядочивание инструкций и гарантирует, что все потоки видят обновлённое значение сразу.`,
			hint2: `**Singleton Билла Пью и Enum:**

\`\`\`java
// Bill Pugh - использует загрузку классов JVM для потокобезопасности
class Logger {
    private static class SingletonHelper {
        private static final Logger INSTANCE = new Logger();
    }

    public static Logger getInstance() {
        return SingletonHelper.INSTANCE;	// внутренний класс загружается при первом вызове
    }
}

// Enum - простейший и безопасный (рекомендован Effective Java)
enum ConfigManager {
    INSTANCE;	// JVM гарантирует единственный экземпляр

    private String value;	// enum может иметь поля и методы
}
\`\`\`

Enum singleton автоматически безопасен для сериализации и защищён от рефлексии.`,
			whyItMatters: `## Почему существует Singleton

**Проблема: Неконтролируемое создание экземпляров**

Без Singleton множество экземпляров тратят ресурсы и вызывают несогласованность:

\`\`\`java
// ❌ БЕЗ SINGLETON - множество экземпляров
DatabaseConnection conn1 = new DatabaseConnection();	// открывает соединение
DatabaseConnection conn2 = new DatabaseConnection();	// открывает ещё одно
DatabaseConnection conn3 = new DatabaseConnection();	// и ещё...
// 100 потоков = 100 соединений = исчерпание ресурсов!

// ✅ С SINGLETON - контролируемый единственный экземпляр
DatabaseConnection conn1 = DatabaseConnection.getInstance();	// создаёт одно соединение
DatabaseConnection conn2 = DatabaseConnection.getInstance();	// возвращает то же соединение
DatabaseConnection conn3 = DatabaseConnection.getInstance();	// всё ещё то же соединение
// Все потоки разделяют один пул соединений
\`\`\`

---

## Примеры из реального мира в Java

**1. java.lang.Runtime:**
\`\`\`java
// JVM использует singleton для Runtime
Runtime runtime = Runtime.getRuntime();	// всегда один экземпляр
runtime.availableProcessors();
\`\`\`

**2. java.awt.Desktop:**
\`\`\`java
// Desktop singleton для системной интеграции
Desktop desktop = Desktop.getDesktop();
desktop.browse(new URI("https://example.com"));
\`\`\`

**3. Spring Framework Beans:**
\`\`\`java
// Spring бины по умолчанию singleton
@Service
public class UserService { }	// один экземпляр на контекст приложения
\`\`\`

**4. SLF4J Logger Factory:**
\`\`\`java
// Логгеры обычно singleton для каждого класса
Logger logger = LoggerFactory.getLogger(MyClass.class);
\`\`\`

---

## Продакшн паттерн: Потокобезопасный Configuration Manager

\`\`\`java
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Properties;
import java.io.InputStream;

public final class AppConfig {
    // Volatile для double-checked locking
    private static volatile AppConfig instance;

    // Потокобезопасное хранилище конфигурации
    private final Map<String, String> config;
    private final String environment;

    // Приватный конструктор
    private AppConfig() {
        this.config = new ConcurrentHashMap<>();
        this.environment = loadEnvironment();
        loadConfiguration();
    }

    // Double-checked locking для ленивой инициализации
    public static AppConfig getInstance() {
        if (instance == null) {
            synchronized (AppConfig.class) {
                if (instance == null) {
                    instance = new AppConfig();
                }
            }
        }
        return instance;
    }

    // Загрузка окружения из системного свойства или значения по умолчанию
    private String loadEnvironment() {
        return System.getProperty("app.env", "development");
    }

    // Загрузка конфигурации из файла properties
    private void loadConfiguration() {
        String fileName = "config-" + environment + ".properties";
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream(fileName)) {
            if (input != null) {
                Properties props = new Properties();
                props.load(input);
                props.forEach((k, v) -> config.put(k.toString(), v.toString()));
            }
        } catch (Exception e) {
            // Логируем и используем значения по умолчанию
            setDefaults();
        }
    }

    private void setDefaults() {
        config.put("db.host", "localhost");
        config.put("db.port", "5432");
        config.put("cache.enabled", "true");
    }

    // Потокобезопасные геттеры
    public String get(String key) {
        return config.get(key);
    }

    public String get(String key, String defaultValue) {
        return config.getOrDefault(key, defaultValue);
    }

    public int getInt(String key, int defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        return Boolean.parseBoolean(value);
    }

    public String getEnvironment() {
        return environment;
    }

    // Предотвращаем клонирование
    @Override
    protected Object clone() throws CloneNotSupportedException {
        throw new CloneNotSupportedException("Singleton cannot be cloned");
    }

    // Для тестирования - сброс экземпляра (использовать осторожно)
    static void resetForTesting() {
        instance = null;
    }
}

// Использование
public class Application {
    public static void main(String[] args) {
        AppConfig config = AppConfig.getInstance();

        String dbHost = config.get("db.host", "localhost");
        int dbPort = config.getInt("db.port", 5432);
        boolean cacheEnabled = config.getBoolean("cache.enabled", true);

        System.out.println("Environment: " + config.getEnvironment());
        System.out.println("Database: " + dbHost + ":" + dbPort);
        System.out.println("Cache: " + (cacheEnabled ? "enabled" : "disabled"));
    }
}
\`\`\`

---

## Распространённые ошибки

**1. Не использование Volatile:**
\`\`\`java
// ❌ НЕПРАВИЛЬНО - без volatile потоки могут видеть частично созданный объект
private static DatabaseConnection instance;	// не volatile!

// ✅ ПРАВИЛЬНО - volatile обеспечивает правильную видимость
private static volatile DatabaseConnection instance;
\`\`\`

**2. Синхронизация всего метода:**
\`\`\`java
// ❌ НЕПРАВИЛЬНО - синхронизация всего метода неэффективна
public static synchronized DatabaseConnection getInstance() {
    if (instance == null) {
        instance = new DatabaseConnection();
    }
    return instance;	// блокировка удерживается при каждом вызове!
}

// ✅ ПРАВИЛЬНО - синхронизируем только когда нужно
public static DatabaseConnection getInstance() {
    if (instance == null) {	// быстрый путь без блокировки
        synchronized (DatabaseConnection.class) {
            if (instance == null) {
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

**3. Игнорирование сериализации:**
\`\`\`java
// ❌ НЕПРАВИЛЬНО - десериализация создаёт новый экземпляр
class BadSingleton implements Serializable {
    private static final BadSingleton INSTANCE = new BadSingleton();
}

// ✅ ПРАВИЛЬНО - реализуем readResolve для сохранения singleton
class GoodSingleton implements Serializable {
    private static final GoodSingleton INSTANCE = new GoodSingleton();

    protected Object readResolve() {
        return INSTANCE;	// возвращаем существующий экземпляр при десериализации
    }
}

// ✅ ЛУЧШЕ - используем enum (обрабатывает сериализацию автоматически)
enum BestSingleton {
    INSTANCE;
}
\`\`\``
		},
		uz: {
			title: 'Singleton (Yagona) Pattern',
			description: `Java da Singleton patternini amalga oshiring — klassning yagona nusxasini global kirish nuqtasi bilan ta'minlang.

**Siz amalga oshirasiz:**

1. **Thread-safe Singleton** ikki marta tekshirish blokirovkasi bilan
2. **Bill Pugh Singleton** ichki statik klass yordamida
3. **Enum Singleton** — tavsiya etilgan yondashuv

**Foydalanish namunasi:**

\`\`\`java
DatabaseConnection db1 = DatabaseConnection.getInstance();	// nusxa olamiz
DatabaseConnection db2 = DatabaseConnection.getInstance();	// xuddi shu nusxani olamiz
System.out.println(db1 == db2);	// true - bitta ob'ekt

Logger logger = Logger.getInstance();	// ichki klass orqali dangasa initsializatsiya
logger.log("Application started");	// singleton dan foydalanamiz

ConfigManager config = ConfigManager.INSTANCE;	// enum singleton kirish
config.setEnvironment("development");	// singleton holatini o'zgartiramiz
\`\`\``,
			hint1: `**Double-Checked Locking ni tushunish:**

Pattern thread xavfsizligi uchun synchronized blok bilan ikkita null tekshirishdan foydalanadi:

\`\`\`java
public static DatabaseConnection getInstance() {
    if (instance == null) {	// birinchi tekshirish - sinxronizatsiya yukini oldini olamiz
        synchronized (DatabaseConnection.class) {	// klassni bloklaymiz
            if (instance == null) {	// ikkinchi tekshirish - faqat bitta yaratishni ta'minlaymiz
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

\`volatile\` kalit so'zi ko'rsatmalarni qayta tartiblashni oldini oladi va barcha threadlar yangilangan qiymatni darhol ko'rishini ta'minlaydi.`,
			hint2: `**Bill Pugh va Enum Singleton:**

\`\`\`java
// Bill Pugh - thread xavfsizligi uchun JVM klass yuklashdan foydalanadi
class Logger {
    private static class SingletonHelper {
        private static final Logger INSTANCE = new Logger();
    }

    public static Logger getInstance() {
        return SingletonHelper.INSTANCE;	// ichki klass birinchi chaqiruvda yuklanadi
    }
}

// Enum - eng oddiy va xavfsiz (Effective Java tomonidan tavsiya etilgan)
enum ConfigManager {
    INSTANCE;	// JVM yagona nusxani kafolatlaydi

    private String value;	// enum maydon va metodlarga ega bo'lishi mumkin
}
\`\`\`

Enum singleton avtomatik ravishda serializatsiya-xavfsiz va refleksiyadan himoyalangan.`,
			whyItMatters: `## Nima uchun Singleton mavjud

**Muammo: Nazorat qilinmagan nusxa yaratish**

Singleton'siz ko'p nusxalar resurslarni sarflaydi va nomuvofiqlikka olib keladi:

\`\`\`java
// ❌ SINGLETON'SIZ - ko'p nusxalar
DatabaseConnection conn1 = new DatabaseConnection();	// ulanish ochadi
DatabaseConnection conn2 = new DatabaseConnection();	// yana bir ulanish ochadi
DatabaseConnection conn3 = new DatabaseConnection();	// va yana...
// 100 thread = 100 ulanish = resurslarning tugashi!

// ✅ SINGLETON BILAN - nazorat qilingan yagona nusxa
DatabaseConnection conn1 = DatabaseConnection.getInstance();	// bitta ulanish yaratadi
DatabaseConnection conn2 = DatabaseConnection.getInstance();	// xuddi shu ulanishni qaytaradi
DatabaseConnection conn3 = DatabaseConnection.getInstance();	// hali ham xuddi shu ulanish
// Barcha threadlar bitta ulanish poolini bo'lishadi
\`\`\`

---

## Java da haqiqiy dunyo misollari

**1. java.lang.Runtime:**
\`\`\`java
// JVM Runtime uchun singleton ishlatadi
Runtime runtime = Runtime.getRuntime();	// doimo bitta nusxa
runtime.availableProcessors();
\`\`\`

**2. java.awt.Desktop:**
\`\`\`java
// Tizim integratsiyasi uchun Desktop singleton
Desktop desktop = Desktop.getDesktop();
desktop.browse(new URI("https://example.com"));
\`\`\`

**3. Spring Framework Beans:**
\`\`\`java
// Spring beanlar default bo'yicha singleton
@Service
public class UserService { }	// dastur konteksti uchun bitta nusxa
\`\`\`

**4. SLF4J Logger Factory:**
\`\`\`java
// Loggerlar odatda har bir klass uchun singleton
Logger logger = LoggerFactory.getLogger(MyClass.class);
\`\`\`

---

## Production Pattern: Thread-Safe Configuration Manager

\`\`\`java
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Properties;
import java.io.InputStream;

public final class AppConfig {
    // Double-checked locking uchun Volatile
    private static volatile AppConfig instance;

    // Thread-xavfsiz konfiguratsiya saqlash
    private final Map<String, String> config;
    private final String environment;

    // Xususiy konstruktor
    private AppConfig() {
        this.config = new ConcurrentHashMap<>();
        this.environment = loadEnvironment();
        loadConfiguration();
    }

    // Dangasa initsializatsiya uchun Double-checked locking
    public static AppConfig getInstance() {
        if (instance == null) {
            synchronized (AppConfig.class) {
                if (instance == null) {
                    instance = new AppConfig();
                }
            }
        }
        return instance;
    }

    // Tizim xususiyatidan yoki default qiymatdan muhitni yuklash
    private String loadEnvironment() {
        return System.getProperty("app.env", "development");
    }

    // Properties faylidan konfiguratsiyani yuklash
    private void loadConfiguration() {
        String fileName = "config-" + environment + ".properties";
        try (InputStream input = getClass().getClassLoader()
                .getResourceAsStream(fileName)) {
            if (input != null) {
                Properties props = new Properties();
                props.load(input);
                props.forEach((k, v) -> config.put(k.toString(), v.toString()));
            }
        } catch (Exception e) {
            // Log va default qiymatlardan foydalanamiz
            setDefaults();
        }
    }

    private void setDefaults() {
        config.put("db.host", "localhost");
        config.put("db.port", "5432");
        config.put("cache.enabled", "true");
    }

    // Thread-xavfsiz getterlar
    public String get(String key) {
        return config.get(key);
    }

    public String get(String key, String defaultValue) {
        return config.getOrDefault(key, defaultValue);
    }

    public int getInt(String key, int defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        String value = config.get(key);
        if (value == null) return defaultValue;
        return Boolean.parseBoolean(value);
    }

    public String getEnvironment() {
        return environment;
    }

    // Klonlashni oldini olamiz
    @Override
    protected Object clone() throws CloneNotSupportedException {
        throw new CloneNotSupportedException("Singleton cannot be cloned");
    }

    // Test uchun - nusxani qayta o'rnatish (ehtiyotkorlik bilan ishlating)
    static void resetForTesting() {
        instance = null;
    }
}

// Foydalanish
public class Application {
    public static void main(String[] args) {
        AppConfig config = AppConfig.getInstance();

        String dbHost = config.get("db.host", "localhost");
        int dbPort = config.getInt("db.port", 5432);
        boolean cacheEnabled = config.getBoolean("cache.enabled", true);

        System.out.println("Environment: " + config.getEnvironment());
        System.out.println("Database: " + dbHost + ":" + dbPort);
        System.out.println("Cache: " + (cacheEnabled ? "enabled" : "disabled"));
    }
}
\`\`\`

---

## Keng tarqalgan xatolar

**1. Volatile ishlatmaslik:**
\`\`\`java
// ❌ NOTO'G'RI - volatile'siz threadlar qisman yaratilgan ob'ektni ko'rishi mumkin
private static DatabaseConnection instance;	// volatile emas!

// ✅ TO'G'RI - volatile to'g'ri ko'rinuvchanlikni ta'minlaydi
private static volatile DatabaseConnection instance;
\`\`\`

**2. Butun metodni sinxronizatsiya qilish:**
\`\`\`java
// ❌ NOTO'G'RI - butun metodni sinxronizatsiya qilish samarasiz
public static synchronized DatabaseConnection getInstance() {
    if (instance == null) {
        instance = new DatabaseConnection();
    }
    return instance;	// har bir chaqiruvda blokirovka ushlab turiladi!
}

// ✅ TO'G'RI - faqat kerak bo'lganda sinxronizatsiya qilamiz
public static DatabaseConnection getInstance() {
    if (instance == null) {	// blokirovkasiz tez yo'l
        synchronized (DatabaseConnection.class) {
            if (instance == null) {
                instance = new DatabaseConnection();
            }
        }
    }
    return instance;
}
\`\`\`

**3. Serializatsiyani e'tiborsiz qoldirish:**
\`\`\`java
// ❌ NOTO'G'RI - deserializatsiya yangi nusxa yaratadi
class BadSingleton implements Serializable {
    private static final BadSingleton INSTANCE = new BadSingleton();
}

// ✅ TO'G'RI - singleton'ni saqlash uchun readResolve ni amalga oshiramiz
class GoodSingleton implements Serializable {
    private static final GoodSingleton INSTANCE = new GoodSingleton();

    protected Object readResolve() {
        return INSTANCE;	// deserializatsiyada mavjud nusxani qaytaramiz
    }
}

// ✅ ENG YAXSHI - enum ishlating (serializatsiyani avtomatik qayta ishlaydi)
enum BestSingleton {
    INSTANCE;
}
\`\`\``
		}
	}
};

export default task;
