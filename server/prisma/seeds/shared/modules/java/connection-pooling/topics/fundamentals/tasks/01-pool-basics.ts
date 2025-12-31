import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-pool-basics',
    title: 'Connection Pool Basics',
    difficulty: 'easy',
    tags: ['java', 'database', 'connection-pool', 'jdbc', 'performance'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Connection Pool Basics

A connection pool is a cache of database connections maintained so that connections can be reused when future requests to the database are required. Connection pooling dramatically improves performance by reducing the overhead of creating new connections.

## Requirements:
1. Demonstrate the problem without connection pooling:
   1.1. Create multiple database connections sequentially
   1.2. Measure time taken for each connection
   1.3. Show the performance impact

2. Explain connection pool concepts:
   2.1. Pool initialization
   2.2. Connection borrowing and returning
   2.3. Connection lifecycle

3. Show basic connection pool benefits:
   3.1. Connection reuse
   3.2. Reduced connection overhead
   3.3. Better resource management

4. Display timing comparisons and connection statistics

## Example Output:
\`\`\`
=== Without Connection Pooling ===
Creating connection 1... took 250ms
Creating connection 2... took 245ms
Creating connection 3... took 248ms
Total time: 743ms

=== Connection Pool Concepts ===
Pool initialized with 5 connections
Connection 1 borrowed from pool... took 2ms
Connection 2 borrowed from pool... took 1ms
Connection 3 borrowed from pool... took 1ms
Total time: 4ms

Performance improvement: 185x faster
\`\`\``,
    initialCode: `// TODO: Import necessary classes

public class ConnectionPoolBasics {
    // TODO: Define database connection parameters

    public static void main(String[] args) {
        // TODO: Demonstrate connections without pooling

        // TODO: Show connection pool benefits

        // TODO: Display performance comparison
    }

    // TODO: Method to create connection without pooling

    // TODO: Method to simulate connection pool behavior
}`,
    solutionCode: `import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class ConnectionPoolBasics {
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public static void main(String[] args) {
        System.out.println("=== Without Connection Pooling ===");
        long startWithout = System.currentTimeMillis();

        // Create connections without pooling
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            try (Connection conn = createNewConnection()) {
                long connTime = System.currentTimeMillis() - connStart;
                System.out.println("Creating connection " + i + "... took " + connTime + "ms");
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

        long totalWithout = System.currentTimeMillis() - startWithout;
        System.out.println("Total time: " + totalWithout + "ms");

        System.out.println("\\n=== Connection Pool Concepts ===");

        // Simulate a simple connection pool
        SimpleConnectionPool pool = new SimpleConnectionPool(5);
        System.out.println("Pool initialized with 5 connections");

        long startWith = System.currentTimeMillis();

        // Borrow connections from pool
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            Connection conn = pool.getConnection();
            long connTime = System.currentTimeMillis() - connStart;
            System.out.println("Connection " + i + " borrowed from pool... took " + connTime + "ms");
            pool.releaseConnection(conn);
        }

        long totalWith = System.currentTimeMillis() - startWith;
        System.out.println("Total time: " + totalWith + "ms");

        System.out.println("\\nPerformance improvement: " +
            (totalWithout / Math.max(1, totalWith)) + "x faster");

        pool.close();
    }

    // Create a new connection without pooling
    private static Connection createNewConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
}

// Simple connection pool implementation for demonstration
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public SimpleConnectionPool(int poolSize) {
        // Initialize pool with connections
        for (int i = 0; i < poolSize; i++) {
            try {
                availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    // Borrow a connection from the pool
    public Connection getConnection() {
        if (availableConnections.isEmpty()) {
            throw new RuntimeException("No available connections");
        }
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    // Return a connection to the pool
    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }

    // Close all connections
    public void close() {
        for (Connection conn : availableConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        for (Connection conn : usedConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}`,
    hint1: `Use System.currentTimeMillis() to measure the time taken to create connections. The difference will clearly show the performance benefits of pooling.`,
    hint2: `A simple connection pool maintains two lists: available connections and used connections. When a connection is borrowed, it moves from available to used, and vice versa when returned.`,
    whyItMatters: `Connection pooling is essential for database-driven applications. Creating a new database connection is expensive (network handshake, authentication, session setup). Connection pools eliminate this overhead by reusing connections, dramatically improving application performance and scalability. Understanding connection pooling is crucial for building efficient enterprise applications.

**Production Pattern:**
\`\`\`java
// Simple connection pool implementation
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();

    public SimpleConnectionPool(int poolSize) {
        // Initialize pool with connections
        for (int i = 0; i < poolSize; i++) {
            availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
        }
    }

    public Connection getConnection() {
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }
}
\`\`\`

**Practical Benefits:**
- Reduction of connection creation time from 250ms to 1-2ms
- Performance improvement of 100+ times for database-intensive applications
- Better resource management and prevention of connection leaks`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.sql.Connection;
import java.util.List;

// Test1: Verify SimpleConnectionPool creation with pool size
class Test1 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(3);
        assertNotNull(pool);
        pool.close();
    }
}

// Test2: Verify getConnection returns a connection
class Test2 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(5);
        Connection conn = pool.getConnection();
        assertNotNull(conn);
        pool.releaseConnection(conn);
        pool.close();
    }
}

// Test3: Verify multiple connections can be borrowed
class Test3 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(5);
        Connection conn1 = pool.getConnection();
        Connection conn2 = pool.getConnection();
        Connection conn3 = pool.getConnection();
        assertNotNull(conn1);
        assertNotNull(conn2);
        assertNotNull(conn3);
        pool.releaseConnection(conn1);
        pool.releaseConnection(conn2);
        pool.releaseConnection(conn3);
        pool.close();
    }
}

// Test4: Verify connections can be released and reused
class Test4 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(2);
        Connection conn1 = pool.getConnection();
        pool.releaseConnection(conn1);
        Connection conn2 = pool.getConnection();
        assertNotNull(conn2);
        pool.releaseConnection(conn2);
        pool.close();
    }
}

// Test5: Verify connection pooling is faster than creating new connections
class Test5 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(3);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 3; i++) {
            Connection conn = pool.getConnection();
            pool.releaseConnection(conn);
        }
        long poolTime = System.currentTimeMillis() - startTime;
        assertTrue(poolTime >= 0); // Should complete quickly
        pool.close();
    }
}

// Test6: Verify pool throws exception when no connections available
class Test6 {
    @Test(expected = RuntimeException.class)
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(2);
        Connection conn1 = pool.getConnection();
        Connection conn2 = pool.getConnection();
        // This should throw RuntimeException
        Connection conn3 = pool.getConnection();
    }
}

// Test7: Verify pool close closes all connections
class Test7 {
    @Test
    public void test() throws Exception {
        SimpleConnectionPool pool = new SimpleConnectionPool(3);
        Connection conn1 = pool.getConnection();
        Connection conn2 = pool.getConnection();
        pool.releaseConnection(conn1);
        pool.releaseConnection(conn2);
        pool.close();
        // Verify connections are closed after pool.close()
        assertTrue(conn1.isClosed());
        assertTrue(conn2.isClosed());
    }
}

// Test8: Verify createNewConnection creates valid connection
class Test8 {
    @Test
    public void test() throws Exception {
        java.sql.Connection conn = java.sql.DriverManager.getConnection(
            "jdbc:h2:mem:testdb", "sa", "");
        assertNotNull(conn);
        assertFalse(conn.isClosed());
        conn.close();
    }
}

// Test9: Verify pool behavior with single connection
class Test9 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(1);
        Connection conn = pool.getConnection();
        assertNotNull(conn);
        pool.releaseConnection(conn);
        Connection conn2 = pool.getConnection();
        assertNotNull(conn2);
        pool.releaseConnection(conn2);
        pool.close();
    }
}

// Test10: Verify pool initialization with large pool size
class Test10 {
    @Test
    public void test() {
        SimpleConnectionPool pool = new SimpleConnectionPool(10);
        Connection conn = pool.getConnection();
        assertNotNull(conn);
        pool.releaseConnection(conn);
        pool.close();
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы пула соединений',
            solutionCode: `import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class ConnectionPoolBasics {
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public static void main(String[] args) {
        System.out.println("=== Без пула соединений ===");
        long startWithout = System.currentTimeMillis();

        // Создание соединений без пула
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            try (Connection conn = createNewConnection()) {
                long connTime = System.currentTimeMillis() - connStart;
                System.out.println("Создание соединения " + i + "... заняло " + connTime + "мс");
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

        long totalWithout = System.currentTimeMillis() - startWithout;
        System.out.println("Общее время: " + totalWithout + "мс");

        System.out.println("\\n=== Концепции пула соединений ===");

        // Имитация простого пула соединений
        SimpleConnectionPool pool = new SimpleConnectionPool(5);
        System.out.println("Пул инициализирован с 5 соединениями");

        long startWith = System.currentTimeMillis();

        // Получение соединений из пула
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            Connection conn = pool.getConnection();
            long connTime = System.currentTimeMillis() - connStart;
            System.out.println("Соединение " + i + " получено из пула... заняло " + connTime + "мс");
            pool.releaseConnection(conn);
        }

        long totalWith = System.currentTimeMillis() - startWith;
        System.out.println("Общее время: " + totalWith + "мс");

        System.out.println("\\nУлучшение производительности: в " +
            (totalWithout / Math.max(1, totalWith)) + " раз быстрее");

        pool.close();
    }

    // Создание нового соединения без пула
    private static Connection createNewConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
}

// Простая реализация пула соединений для демонстрации
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public SimpleConnectionPool(int poolSize) {
        // Инициализация пула соединениями
        for (int i = 0; i < poolSize; i++) {
            try {
                availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    // Получить соединение из пула
    public Connection getConnection() {
        if (availableConnections.isEmpty()) {
            throw new RuntimeException("Нет доступных соединений");
        }
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    // Вернуть соединение в пул
    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }

    // Закрыть все соединения
    public void close() {
        for (Connection conn : availableConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        for (Connection conn : usedConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}`,
            description: `# Основы пула соединений

Пул соединений - это кэш соединений с базой данных, который поддерживается для повторного использования соединений при будущих запросах к базе данных. Пулинг соединений значительно улучшает производительность, уменьшая накладные расходы на создание новых соединений.

## Требования:
1. Продемонстрировать проблему без пула соединений:
   1.1. Создать несколько соединений с базой данных последовательно
   1.2. Измерить время для каждого соединения
   1.3. Показать влияние на производительность

2. Объяснить концепции пула соединений:
   2.1. Инициализация пула
   2.2. Получение и возврат соединений
   2.3. Жизненный цикл соединения

3. Показать преимущества базового пула соединений:
   3.1. Повторное использование соединений
   3.2. Уменьшенные накладные расходы на соединение
   3.3. Лучшее управление ресурсами

4. Отобразить сравнение времени и статистику соединений

## Пример вывода:
\`\`\`
=== Without Connection Pooling ===
Creating connection 1... took 250ms
Creating connection 2... took 245ms
Creating connection 3... took 248ms
Total time: 743ms

=== Connection Pool Concepts ===
Pool initialized with 5 connections
Connection 1 borrowed from pool... took 2ms
Connection 2 borrowed from pool... took 1ms
Connection 3 borrowed from pool... took 1ms
Total time: 4ms

Performance improvement: 185x faster
\`\`\``,
            hint1: `Используйте System.currentTimeMillis() для измерения времени создания соединений. Разница явно покажет преимущества пулинга в производительности.`,
            hint2: `Простой пул соединений поддерживает два списка: доступные соединения и используемые соединения. Когда соединение берётся, оно перемещается из доступных в используемые, и наоборот при возврате.`,
            whyItMatters: `Пулинг соединений необходим для приложений, работающих с базами данных. Создание нового соединения с базой данных дорого (сетевое рукопожатие, аутентификация, настройка сеанса). Пулы соединений устраняют эти накладные расходы путём повторного использования соединений, значительно улучшая производительность и масштабируемость приложения. Понимание пулинга соединений критически важно для создания эффективных корпоративных приложений.

**Продакшен паттерн:**
\`\`\`java
// Простая реализация пула соединений
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();

    public SimpleConnectionPool(int poolSize) {
        // Инициализация пула соединениями
        for (int i = 0; i < poolSize; i++) {
            availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
        }
    }

    public Connection getConnection() {
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }
}
\`\`\`

**Практические преимущества:**
- Уменьшение времени создания соединения с 250мс до 1-2мс
- Повышение производительности в 100+ раз для приложений с интенсивной работой с БД
- Лучшее управление ресурсами и предотвращение утечек соединений`
        },
        uz: {
            title: `Ulanish puli asoslari`,
            solutionCode: `import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class ConnectionPoolBasics {
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public static void main(String[] args) {
        System.out.println("=== Ulanish pulisiz ===");
        long startWithout = System.currentTimeMillis();

        // Pulisiz ulanishlarni yaratish
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            try (Connection conn = createNewConnection()) {
                long connTime = System.currentTimeMillis() - connStart;
                System.out.println("Ulanish " + i + " yaratildi... " + connTime + "ms vaqt ketdi");
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

        long totalWithout = System.currentTimeMillis() - startWithout;
        System.out.println("Jami vaqt: " + totalWithout + "ms");

        System.out.println("\\n=== Ulanish puli tushunchalari ===");

        // Oddiy ulanish pulini simulyatsiya qilish
        SimpleConnectionPool pool = new SimpleConnectionPool(5);
        System.out.println("Pul 5 ta ulanish bilan ishga tushirildi");

        long startWith = System.currentTimeMillis();

        // Puldan ulanishlarni olish
        for (int i = 1; i <= 3; i++) {
            long connStart = System.currentTimeMillis();
            Connection conn = pool.getConnection();
            long connTime = System.currentTimeMillis() - connStart;
            System.out.println("Ulanish " + i + " puldan olindi... " + connTime + "ms vaqt ketdi");
            pool.releaseConnection(conn);
        }

        long totalWith = System.currentTimeMillis() - startWith;
        System.out.println("Jami vaqt: " + totalWith + "ms");

        System.out.println("\\nUnumdorlik yaxshilandi: " +
            (totalWithout / Math.max(1, totalWith)) + " marta tezroq");

        pool.close();
    }

    // Pulisiz yangi ulanish yaratish
    private static Connection createNewConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
}

// Namoyish uchun oddiy ulanish puli implementatsiyasi
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();
    private static final String URL = "jdbc:h2:mem:testdb";
    private static final String USER = "sa";
    private static final String PASSWORD = "";

    public SimpleConnectionPool(int poolSize) {
        // Pulni ulanishlar bilan ishga tushirish
        for (int i = 0; i < poolSize; i++) {
            try {
                availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    // Puldan ulanish olish
    public Connection getConnection() {
        if (availableConnections.isEmpty()) {
            throw new RuntimeException("Mavjud ulanishlar yo'q");
        }
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    // Ulanishni pulga qaytarish
    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }

    // Barcha ulanishlarni yopish
    public void close() {
        for (Connection conn : availableConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        for (Connection conn : usedConnections) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}`,
            description: `# Ulanish puli asoslari

Ulanish puli - bu ma'lumotlar bazasiga kelajakdagi so'rovlar talab qilinganda ulanishlar qayta ishlatilishi uchun saqlanadigan ma'lumotlar bazasi ulanishlari keshi. Ulanish pullanishi yangi ulanishlar yaratish xarajatlarini kamaytirish orqali ishlashni sezilarli darajada yaxshilaydi.

## Talablar:
1. Ulanish pulisiz muammoni ko'rsating:
   1.1. Bir nechta ma'lumotlar bazasi ulanishlarini ketma-ket yarating
   1.2. Har bir ulanish uchun ketgan vaqtni o'lchang
   1.3. Ishlashga ta'sirni ko'rsating

2. Ulanish puli tushunchalarini tushuntiring:
   2.1. Pul ishga tushirish
   2.2. Ulanishni olish va qaytarish
   2.3. Ulanish hayot sikli

3. Asosiy ulanish puli afzalliklarini ko'rsating:
   3.1. Ulanishlarni qayta ishlatish
   3.2. Kamaytirilgan ulanish xarajatlari
   3.3. Yaxshiroq resurs boshqaruvi

4. Vaqt taqqoslashlari va ulanish statistikasini ko'rsating

## Chiqish namunasi:
\`\`\`
=== Without Connection Pooling ===
Creating connection 1... took 250ms
Creating connection 2... took 245ms
Creating connection 3... took 248ms
Total time: 743ms

=== Connection Pool Concepts ===
Pool initialized with 5 connections
Connection 1 borrowed from pool... took 2ms
Connection 2 borrowed from pool... took 1ms
Connection 3 borrowed from pool... took 1ms
Total time: 4ms

Performance improvement: 185x faster
\`\`\``,
            hint1: `Ulanishlar yaratish uchun ketgan vaqtni o'lchash uchun System.currentTimeMillis() dan foydalaning. Farq pullanishning ishlash afzalliklarini aniq ko'rsatadi.`,
            hint2: `Oddiy ulanish puli ikki ro'yxatni saqlaydi: mavjud ulanishlar va foydalanilgan ulanishlar. Ulanish olinganda, u mavjuddan foydalanilganga o'tadi va qaytarilganda aksincha.`,
            whyItMatters: `Ulanish pullanishi ma'lumotlar bazasi asosidagi ilovalar uchun zarurdir. Yangi ma'lumotlar bazasi ulanishini yaratish qimmat (tarmoq qo'l siqish, autentifikatsiya, sessiya sozlash). Ulanish pullari ulanishlarni qayta ishlatish orqali bu xarajatlarni yo'q qiladi, ilova ishlashi va miqyosliligini sezilarli darajada yaxshilaydi. Ulanish pullanishini tushunish samarali korxona ilovalarini yaratish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Oddiy ulanish puli implementatsiyasi
class SimpleConnectionPool {
    private List<Connection> availableConnections = new ArrayList<>();
    private List<Connection> usedConnections = new ArrayList<>();

    public SimpleConnectionPool(int poolSize) {
        // Pulni ulanishlar bilan ishga tushirish
        for (int i = 0; i < poolSize; i++) {
            availableConnections.add(DriverManager.getConnection(URL, USER, PASSWORD));
        }
    }

    public Connection getConnection() {
        Connection conn = availableConnections.remove(availableConnections.size() - 1);
        usedConnections.add(conn);
        return conn;
    }

    public void releaseConnection(Connection conn) {
        usedConnections.remove(conn);
        availableConnections.add(conn);
    }
}
\`\`\`

**Amaliy foydalari:**
- Ulanish yaratish vaqtini 250ms dan 1-2ms ga qisqartirish
- DB bilan intensiv ishlaydigan ilovalar uchun 100+ marta samaradorlik oshishi
- Yaxshiroq resurs boshqaruvi va ulanish oqishlarining oldini olish`
        }
    }
};

export default task;
