import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-hikaricp-setup',
    title: 'HikariCP Setup',
    difficulty: 'medium',
    tags: ['java', 'hikaricp', 'connection-pool', 'jdbc', 'database'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# HikariCP Setup

HikariCP is a fast, simple, and reliable JDBC connection pool. It's the default connection pool in Spring Boot and known for its high performance and low overhead. Learn how to set up and configure HikariCP for your applications.

## Requirements:
1. Create HikariConfig with basic settings:
   1.1. Set JDBC URL, username, password
   1.2. Configure pool name
   1.3. Set initial pool size

2. Initialize HikariDataSource:
   2.1. Create datasource from config
   2.2. Get connections from the pool
   2.3. Return connections properly

3. Demonstrate basic usage:
   3.1. Execute simple queries
   3.2. Show connection reuse
   3.3. Proper resource cleanup

4. Display pool information and connection details

## Example Output:
\`\`\`
=== HikariCP Configuration ===
Pool Name: MyAppPool
JDBC URL: jdbc:h2:mem:testdb
Maximum Pool Size: 10
Connection Timeout: 30000ms

=== Getting Connections ===
Connection 1 obtained: HikariProxyConnection@123
Executing query...
Connection returned to pool

Connection 2 obtained: HikariProxyConnection@123
Connection reused from pool!
Query executed successfully

=== Pool Statistics ===
Active connections: 0
Idle connections: 1
Total connections: 1
\`\`\``,
    initialCode: `// TODO: Import HikariCP classes

public class HikariCPSetup {
    public static void main(String[] args) {
        // TODO: Create HikariConfig

        // TODO: Initialize HikariDataSource

        // TODO: Use connections from pool

        // TODO: Display pool information

        // TODO: Close datasource
    }

    // TODO: Method to execute sample query
}`,
    solutionCode: `import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

public class HikariCPSetup {
    public static void main(String[] args) {
        System.out.println("=== HikariCP Configuration ===");

        // Create HikariConfig
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setPoolName("MyAppPool");
        config.setMaximumPoolSize(10);
        config.setConnectionTimeout(30000);
        config.setIdleTimeout(600000);
        config.setMaxLifetime(1800000);

        System.out.println("Pool Name: " + config.getPoolName());
        System.out.println("JDBC URL: " + config.getJdbcUrl());
        System.out.println("Maximum Pool Size: " + config.getMaximumPoolSize());
        System.out.println("Connection Timeout: " + config.getConnectionTimeout() + "ms");

        // Initialize HikariDataSource
        HikariDataSource dataSource = new HikariDataSource(config);

        System.out.println("\\n=== Getting Connections ===");

        try {
            // Get first connection
            Connection conn1 = dataSource.getConnection();
            System.out.println("Connection 1 obtained: " + conn1.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn1)));

            // Execute query
            System.out.println("Executing query...");
            executeQuery(conn1);

            // Return connection to pool
            conn1.close(); // This doesn't actually close, just returns to pool
            System.out.println("Connection returned to pool\\n");

            // Get second connection - should be reused
            Connection conn2 = dataSource.getConnection();
            System.out.println("Connection 2 obtained: " + conn2.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn2)));
            System.out.println("Connection reused from pool!");

            executeQuery(conn2);
            conn2.close();

            // Display pool statistics
            System.out.println("\\n=== Pool Statistics ===");
            System.out.println("Active connections: " +
                (dataSource.getHikariPoolMXBean().getActiveConnections()));
            System.out.println("Idle connections: " +
                (dataSource.getHikariPoolMXBean().getIdleConnections()));
            System.out.println("Total connections: " +
                (dataSource.getHikariPoolMXBean().getTotalConnections()));
            System.out.println("Threads awaiting connection: " +
                (dataSource.getHikariPoolMXBean().getThreadsAwaitingConnection()));

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Close datasource
            dataSource.close();
            System.out.println("\\nDataSource closed");
        }
    }

    // Execute a simple query
    private static void executeQuery(Connection conn) {
        try {
            Statement stmt = conn.createStatement();
            // Create a simple table if not exists
            stmt.execute("CREATE TABLE IF NOT EXISTS test (id INT PRIMARY KEY, name VARCHAR(50))");
            stmt.execute("INSERT INTO test VALUES (1, 'HikariCP') ON DUPLICATE KEY UPDATE name='HikariCP'");

            ResultSet rs = stmt.executeQuery("SELECT * FROM test");
            while (rs.next()) {
                System.out.println("Query result: ID=" + rs.getInt("id") +
                    ", Name=" + rs.getString("name"));
            }
            rs.close();
            stmt.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}`,
    hint1: `HikariConfig is used to set all configuration properties before creating the HikariDataSource. Set at minimum the JDBC URL, username, and password.`,
    hint2: `When you call close() on a connection obtained from HikariCP, it doesn't actually close the connection - it returns it to the pool for reuse. The pool manages the actual connection lifecycle.`,
    whyItMatters: `HikariCP is the gold standard for JDBC connection pooling in Java. It's extremely fast (often 2-3x faster than alternatives), lightweight, and reliable. Understanding how to properly configure and use HikariCP is essential for building high-performance database applications. Spring Boot uses HikariCP as its default connection pool, making this knowledge valuable for Spring developers.

**Production Pattern:**
\`\`\`java
// HikariCP configuration for production
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
config.setUsername("user");
config.setPassword("password");
config.setMaximumPoolSize(10);
config.setConnectionTimeout(30000);
config.setIdleTimeout(600000);
config.setMaxLifetime(1800000);

HikariDataSource dataSource = new HikariDataSource(config);

// Getting connection (automatically returned to pool on close())
try (Connection conn = dataSource.getConnection()) {
    // Execute queries
}
\`\`\`

**Practical Benefits:**
- 2-3x faster than Apache DBCP and C3P0
- Minimal memory overhead (~130KB for a pool of 10 connections)
- Automatic connection lifecycle management and leak detection`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.sql.Connection;

// Test1: Verify HikariConfig creation
class Test1 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        assertNotNull(config);
        assertEquals("jdbc:h2:mem:testdb", config.getJdbcUrl());
    }
}

// Test2: Verify HikariConfig pool name setting
class Test2 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setPoolName("TestPool");
        assertEquals("TestPool", config.getPoolName());
    }
}

// Test3: Verify HikariConfig maximum pool size
class Test3 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setMaximumPoolSize(10);
        assertEquals(10, config.getMaximumPoolSize());
    }
}

// Test4: Verify HikariDataSource creation
class Test4 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        HikariDataSource dataSource = new HikariDataSource(config);
        assertNotNull(dataSource);
        dataSource.close();
    }
}

// Test5: Verify getting connection from HikariDataSource
class Test5 {
    @Test
    public void test() throws Exception {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        HikariDataSource dataSource = new HikariDataSource(config);
        Connection conn = dataSource.getConnection();
        assertNotNull(conn);
        assertFalse(conn.isClosed());
        conn.close();
        dataSource.close();
    }
}

// Test6: Verify connection timeout setting
class Test6 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setConnectionTimeout(30000);
        assertEquals(30000, config.getConnectionTimeout());
    }
}

// Test7: Verify idle timeout setting
class Test7 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setIdleTimeout(600000);
        assertEquals(600000, config.getIdleTimeout());
    }
}

// Test8: Verify max lifetime setting
class Test8 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setMaxLifetime(1800000);
        assertEquals(1800000, config.getMaxLifetime());
    }
}

// Test9: Verify multiple connections can be obtained
class Test9 {
    @Test
    public void test() throws Exception {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setMaximumPoolSize(5);
        HikariDataSource dataSource = new HikariDataSource(config);

        Connection conn1 = dataSource.getConnection();
        Connection conn2 = dataSource.getConnection();
        assertNotNull(conn1);
        assertNotNull(conn2);

        conn1.close();
        conn2.close();
        dataSource.close();
    }
}

// Test10: Verify datasource close
class Test10 {
    @Test
    public void test() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        HikariDataSource dataSource = new HikariDataSource(config);
        dataSource.close();
        assertTrue(dataSource.isClosed());
    }
}
`,
    order: 2,
    translations: {
        ru: {
            title: 'Настройка HikariCP',
            solutionCode: `import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

public class HikariCPSetup {
    public static void main(String[] args) {
        System.out.println("=== Конфигурация HikariCP ===");

        // Создание HikariConfig
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setPoolName("MyAppPool");
        config.setMaximumPoolSize(10);
        config.setConnectionTimeout(30000);
        config.setIdleTimeout(600000);
        config.setMaxLifetime(1800000);

        System.out.println("Имя пула: " + config.getPoolName());
        System.out.println("JDBC URL: " + config.getJdbcUrl());
        System.out.println("Максимальный размер пула: " + config.getMaximumPoolSize());
        System.out.println("Таймаут соединения: " + config.getConnectionTimeout() + "мс");

        // Инициализация HikariDataSource
        HikariDataSource dataSource = new HikariDataSource(config);

        System.out.println("\\n=== Получение соединений ===");

        try {
            // Получить первое соединение
            Connection conn1 = dataSource.getConnection();
            System.out.println("Соединение 1 получено: " + conn1.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn1)));

            // Выполнить запрос
            System.out.println("Выполнение запроса...");
            executeQuery(conn1);

            // Вернуть соединение в пул
            conn1.close(); // На самом деле не закрывает, а возвращает в пул
            System.out.println("Соединение возвращено в пул\\n");

            // Получить второе соединение - должно быть повторно использовано
            Connection conn2 = dataSource.getConnection();
            System.out.println("Соединение 2 получено: " + conn2.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn2)));
            System.out.println("Соединение повторно использовано из пула!");

            executeQuery(conn2);
            conn2.close();

            // Показать статистику пула
            System.out.println("\\n=== Статистика пула ===");
            System.out.println("Активные соединения: " +
                (dataSource.getHikariPoolMXBean().getActiveConnections()));
            System.out.println("Свободные соединения: " +
                (dataSource.getHikariPoolMXBean().getIdleConnections()));
            System.out.println("Всего соединений: " +
                (dataSource.getHikariPoolMXBean().getTotalConnections()));
            System.out.println("Потоков ожидают соединения: " +
                (dataSource.getHikariPoolMXBean().getThreadsAwaitingConnection()));

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Закрыть datasource
            dataSource.close();
            System.out.println("\\nDataSource закрыт");
        }
    }

    // Выполнить простой запрос
    private static void executeQuery(Connection conn) {
        try {
            Statement stmt = conn.createStatement();
            // Создать простую таблицу если не существует
            stmt.execute("CREATE TABLE IF NOT EXISTS test (id INT PRIMARY KEY, name VARCHAR(50))");
            stmt.execute("INSERT INTO test VALUES (1, 'HikariCP') ON DUPLICATE KEY UPDATE name='HikariCP'");

            ResultSet rs = stmt.executeQuery("SELECT * FROM test");
            while (rs.next()) {
                System.out.println("Результат запроса: ID=" + rs.getInt("id") +
                    ", Имя=" + rs.getString("name"));
            }
            rs.close();
            stmt.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}`,
            description: `# Настройка HikariCP

HikariCP - это быстрый, простой и надёжный пул соединений JDBC. Это пул соединений по умолчанию в Spring Boot, известный своей высокой производительностью и низкими накладными расходами. Узнайте, как настроить и сконфигурировать HikariCP для ваших приложений.

## Требования:
1. Создайте HikariConfig с базовыми настройками:
   1.1. Установите JDBC URL, имя пользователя, пароль
   1.2. Настройте имя пула
   1.3. Установите начальный размер пула

2. Инициализируйте HikariDataSource:
   2.1. Создайте datasource из конфигурации
   2.2. Получите соединения из пула
   2.3. Правильно возвращайте соединения

3. Продемонстрируйте базовое использование:
   3.1. Выполните простые запросы
   3.2. Покажите повторное использование соединений
   3.3. Правильную очистку ресурсов

4. Отобразите информацию о пуле и детали соединений

## Пример вывода:
\`\`\`
=== HikariCP Configuration ===
Pool Name: MyAppPool
JDBC URL: jdbc:h2:mem:testdb
Maximum Pool Size: 10
Connection Timeout: 30000ms

=== Getting Connections ===
Connection 1 obtained: HikariProxyConnection@123
Executing query...
Connection returned to pool

Connection 2 obtained: HikariProxyConnection@123
Connection reused from pool!
Query executed successfully

=== Pool Statistics ===
Active connections: 0
Idle connections: 1
Total connections: 1
\`\`\``,
            hint1: `HikariConfig используется для установки всех свойств конфигурации перед созданием HikariDataSource. Установите как минимум JDBC URL, имя пользователя и пароль.`,
            hint2: `Когда вы вызываете close() на соединении, полученном из HikariCP, оно фактически не закрывается - оно возвращается в пул для повторного использования. Пул управляет фактическим жизненным циклом соединения.`,
            whyItMatters: `HikariCP является золотым стандартом для пулинга соединений JDBC в Java. Он чрезвычайно быстрый (часто в 2-3 раза быстрее альтернатив), лёгкий и надёжный. Понимание того, как правильно настраивать и использовать HikariCP, необходимо для создания высокопроизводительных приложений с базами данных. Spring Boot использует HikariCP как пул соединений по умолчанию, что делает эти знания ценными для Spring-разработчиков.

**Продакшен паттерн:**
\`\`\`java
// Настройка HikariCP для продакшена
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
config.setUsername("user");
config.setPassword("password");
config.setMaximumPoolSize(10);
config.setConnectionTimeout(30000);
config.setIdleTimeout(600000);
config.setMaxLifetime(1800000);

HikariDataSource dataSource = new HikariDataSource(config);

// Получение соединения (автоматически возвращается в пул при close())
try (Connection conn = dataSource.getConnection()) {
    // Выполнение запросов
}
\`\`\`

**Практические преимущества:**
- В 2-3 раза быстрее чем Apache DBCP и C3P0
- Минимальный оверхед памяти (~130KB для пула из 10 соединений)
- Автоматическое управление жизненным циклом соединений и обнаружение утечек`
        },
        uz: {
            title: `HikariCP sozlash`,
            solutionCode: `import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

public class HikariCPSetup {
    public static void main(String[] args) {
        System.out.println("=== HikariCP konfiguratsiyasi ===");

        // HikariConfig yaratish
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb");
        config.setUsername("sa");
        config.setPassword("");
        config.setPoolName("MyAppPool");
        config.setMaximumPoolSize(10);
        config.setConnectionTimeout(30000);
        config.setIdleTimeout(600000);
        config.setMaxLifetime(1800000);

        System.out.println("Pul nomi: " + config.getPoolName());
        System.out.println("JDBC URL: " + config.getJdbcUrl());
        System.out.println("Maksimal pul hajmi: " + config.getMaximumPoolSize());
        System.out.println("Ulanish vaqt cheklovi: " + config.getConnectionTimeout() + "ms");

        // HikariDataSource-ni ishga tushirish
        HikariDataSource dataSource = new HikariDataSource(config);

        System.out.println("\\n=== Ulanishlarni olish ===");

        try {
            // Birinchi ulanishni olish
            Connection conn1 = dataSource.getConnection();
            System.out.println("Ulanish 1 olindi: " + conn1.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn1)));

            // So'rovni bajarish
            System.out.println("So'rov bajarilmoqda...");
            executeQuery(conn1);

            // Ulanishni pulga qaytarish
            conn1.close(); // Aslida yopmaydi, pulga qaytaradi
            System.out.println("Ulanish pulga qaytarildi\\n");

            // Ikkinchi ulanishni olish - qayta ishlatilishi kerak
            Connection conn2 = dataSource.getConnection();
            System.out.println("Ulanish 2 olindi: " + conn2.getClass().getSimpleName() + "@" +
                Integer.toHexString(System.identityHashCode(conn2)));
            System.out.println("Ulanish puldan qayta ishlatildi!");

            executeQuery(conn2);
            conn2.close();

            // Pul statistikasini ko'rsatish
            System.out.println("\\n=== Pul statistikasi ===");
            System.out.println("Faol ulanishlar: " +
                (dataSource.getHikariPoolMXBean().getActiveConnections()));
            System.out.println("Bo'sh ulanishlar: " +
                (dataSource.getHikariPoolMXBean().getIdleConnections()));
            System.out.println("Jami ulanishlar: " +
                (dataSource.getHikariPoolMXBean().getTotalConnections()));
            System.out.println("Ulanish kutayotgan threadlar: " +
                (dataSource.getHikariPoolMXBean().getThreadsAwaitingConnection()));

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // DataSource-ni yopish
            dataSource.close();
            System.out.println("\\nDataSource yopildi");
        }
    }

    // Oddiy so'rovni bajarish
    private static void executeQuery(Connection conn) {
        try {
            Statement stmt = conn.createStatement();
            // Agar mavjud bo'lmasa oddiy jadval yaratish
            stmt.execute("CREATE TABLE IF NOT EXISTS test (id INT PRIMARY KEY, name VARCHAR(50))");
            stmt.execute("INSERT INTO test VALUES (1, 'HikariCP') ON DUPLICATE KEY UPDATE name='HikariCP'");

            ResultSet rs = stmt.executeQuery("SELECT * FROM test");
            while (rs.next()) {
                System.out.println("So'rov natijasi: ID=" + rs.getInt("id") +
                    ", Nomi=" + rs.getString("name"));
            }
            rs.close();
            stmt.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}`,
            description: `# HikariCP sozlash

HikariCP - bu tez, oddiy va ishonchli JDBC ulanish puli. Bu Spring Boot-dagi standart ulanish puli bo'lib, yuqori ishlash va past xarajatlari bilan mashhur. Ilovalaringiz uchun HikariCP-ni qanday sozlash va konfiguratsiya qilishni o'rganing.

## Talablar:
1. Asosiy sozlamalar bilan HikariConfig yarating:
   1.1. JDBC URL, foydalanuvchi nomi, parolni o'rnating
   1.2. Pul nomini sozlang
   1.3. Boshlang'ich pul hajmini o'rnating

2. HikariDataSource-ni ishga tushiring:
   2.1. Konfiguratsiyadan datasource yarating
   2.2. Puldan ulanishlarni oling
   2.3. Ulanishlarni to'g'ri qaytaring

3. Asosiy foydalanishni ko'rsating:
   3.1. Oddiy so'rovlarni bajaring
   3.2. Ulanishlar qayta ishlatilishini ko'rsating
   3.3. To'g'ri resurs tozalash

4. Pul ma'lumotlari va ulanish tafsilotlarini ko'rsating

## Chiqish namunasi:
\`\`\`
=== HikariCP Configuration ===
Pool Name: MyAppPool
JDBC URL: jdbc:h2:mem:testdb
Maximum Pool Size: 10
Connection Timeout: 30000ms

=== Getting Connections ===
Connection 1 obtained: HikariProxyConnection@123
Executing query...
Connection returned to pool

Connection 2 obtained: HikariProxyConnection@123
Connection reused from pool!
Query executed successfully

=== Pool Statistics ===
Active connections: 0
Idle connections: 1
Total connections: 1
\`\`\``,
            hint1: `HikariConfig HikariDataSource yaratishdan oldin barcha konfiguratsiya xususiyatlarini o'rnatish uchun ishlatiladi. Minimal JDBC URL, foydalanuvchi nomi va parolni o'rnating.`,
            hint2: `HikariCP-dan olingan ulanishda close() ni chaqirganingizda, u aslida yopilmaydi - u qayta ishlatish uchun pulga qaytariladi. Pul haqiqiy ulanish hayot siklini boshqaradi.`,
            whyItMatters: `HikariCP Java-da JDBC ulanish pullanishi uchun oltin standartdir. U juda tez (ko'pincha alternativalardan 2-3 marta tezroq), yengil va ishonchli. HikariCP-ni qanday to'g'ri sozlash va ishlatishni tushunish yuqori samarali ma'lumotlar bazasi ilovalarini yaratish uchun zarurdir. Spring Boot HikariCP-ni standart ulanish puli sifatida ishlatadi, bu esa bu bilimlarni Spring dasturchilari uchun qimmatli qiladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ishlab chiqarish uchun HikariCP sozlash
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
config.setUsername("user");
config.setPassword("password");
config.setMaximumPoolSize(10);
config.setConnectionTimeout(30000);
config.setIdleTimeout(600000);
config.setMaxLifetime(1800000);

HikariDataSource dataSource = new HikariDataSource(config);

// Ulanishni olish (close() da avtomatik pulga qaytariladi)
try (Connection conn = dataSource.getConnection()) {
    // So'rovlarni bajarish
}
\`\`\`

**Amaliy foydalari:**
- Apache DBCP va C3P0 dan 2-3 marta tezroq
- Minimal xotira xarajati (~10 ulanishli pul uchun 130KB)
- Avtomatik ulanish hayot siklini boshqarish va oqish aniqlash`
        }
    }
};

export default task;
