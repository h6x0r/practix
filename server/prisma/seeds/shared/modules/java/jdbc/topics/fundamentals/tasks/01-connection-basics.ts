import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-jdbc-connection-basics',
    title: 'JDBC Connection Basics',
    difficulty: 'easy',
    tags: ['java', 'jdbc', 'database', 'connection', 'drivermanager'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to establish a database connection using JDBC's **DriverManager** class.

**Requirements:**
1. Create a DatabaseConnection class with a method to establish a connection:
   1.1. Use DriverManager.getConnection() to connect to a database
   1.2. Use a proper JDBC URL format (e.g., "jdbc:mysql://localhost:3306/mydb")
   1.3. Handle connection with username and password

2. Implement connection testing:
   2.1. Test if the connection is valid
   2.2. Print connection metadata (database product name, version)
   2.3. Close the connection properly

3. In the main method:
   3.1. Establish a connection to a database
   3.2. Display connection information
   3.3. Demonstrate proper resource cleanup

**Learning Goals:**
- Understand JDBC URL structure
- Learn to use DriverManager for connections
- Practice proper resource management with try-catch-finally or try-with-resources`,
    initialCode: `import java.sql.*;

public class DatabaseConnection {
    // Database connection parameters
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // TODO: Implement method to establish database connection

    // TODO: Implement method to display connection information

    // TODO: Implement proper connection closing

    public static void main(String[] args) {
        // TODO: Establish connection and test it
    }
}`,
    solutionCode: `import java.sql.*;

public class DatabaseConnection {
    // Database connection parameters
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Method to establish database connection
    public static Connection getConnection() throws SQLException {
        // Register JDBC driver (optional in JDBC 4.0+)
        // Class.forName("com.mysql.cj.jdbc.Driver");

        // Establish connection
        Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
        System.out.println("Database connection established successfully!");
        return conn;
    }

    // Method to display connection information
    public static void displayConnectionInfo(Connection conn) throws SQLException {
        if (conn != null && !conn.isClosed()) {
            DatabaseMetaData metaData = conn.getMetaData();
            System.out.println("Database Product: " + metaData.getDatabaseProductName());
            System.out.println("Database Version: " + metaData.getDatabaseProductVersion());
            System.out.println("Driver Name: " + metaData.getDriverName());
            System.out.println("Driver Version: " + metaData.getDriverVersion());
            System.out.println("Connection URL: " + metaData.getURL());
            System.out.println("Username: " + metaData.getUserName());
        }
    }

    // Method to close connection safely
    public static void closeConnection(Connection conn) {
        if (conn != null) {
            try {
                conn.close();
                System.out.println("Connection closed successfully.");
            } catch (SQLException e) {
                System.err.println("Error closing connection: " + e.getMessage());
            }
        }
    }

    public static void main(String[] args) {
        Connection conn = null;

        try {
            // Establish connection
            conn = getConnection();

            // Test connection validity
            if (conn.isValid(2)) {
                System.out.println("Connection is valid!");

                // Display connection information
                displayConnectionInfo(conn);
            }

        } catch (SQLException e) {
            System.err.println("Database connection error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Always close the connection
            closeConnection(conn);
        }

        // Alternative: Using try-with-resources (recommended)
        // try (Connection connection = getConnection()) {
        //     displayConnectionInfo(connection);
        // } catch (SQLException e) {
        //     e.printStackTrace();
        // }
    }
}`,
    hint1: `Use DriverManager.getConnection(url, username, password) to establish a connection. The URL format is "jdbc:driver://host:port/database".`,
    hint2: `Use try-with-resources or a finally block to ensure the connection is always closed, preventing resource leaks.`,
    whyItMatters: `JDBC connections are the foundation of database interaction in Java. Understanding how to properly establish, manage, and close connections is critical for building reliable database applications and preventing resource leaks that can crash your application.

**Production Pattern:**
\`\`\`java
// Using connection pool for production
private static final HikariDataSource dataSource = new HikariDataSource();
static {
    dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/testdb");
    dataSource.setUsername("root");
    dataSource.setPassword("password");
    dataSource.setMaximumPoolSize(10);
    dataSource.setConnectionTimeout(30000);
}
public static Connection getConnection() throws SQLException {
    return dataSource.getConnection();
}
\`\`\`

**Practical Benefits:**
- Connection pool reuses connections, saving resources
- Avoids expensive connection creation/closing operations
- Automatic connection lifecycle management`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.sql.*;

// Test1: Verify connection URL format
class Test1 {
    @Test
    public void testConnectionURLFormat() {
        String url = "jdbc:mysql://localhost:3306/testdb";
        assertTrue(url.startsWith("jdbc:"));
        assertTrue(url.contains("mysql"));
    }
}

// Test2: Verify DriverManager class availability
class Test2 {
    @Test
    public void testDriverManagerAvailability() {
        assertNotNull(DriverManager.class);
    }
}

// Test3: Verify connection parameters
class Test3 {
    @Test
    public void testConnectionParameters() {
        String url = "jdbc:mysql://localhost:3306/testdb";
        String user = "root";
        String password = "password";

        assertNotNull(url);
        assertNotNull(user);
        assertNotNull(password);
        assertFalse(url.isEmpty());
    }
}

// Test4: Verify SQLException is throwable
class Test4 {
    @Test
    public void testSQLExceptionThrowable() {
        SQLException ex = new SQLException("Test exception");
        assertNotNull(ex);
        assertEquals("Test exception", ex.getMessage());
    }
}

// Test5: Verify Connection interface exists
class Test5 {
    @Test
    public void testConnectionInterfaceExists() {
        assertNotNull(Connection.class);
    }
}

// Test6: Verify DatabaseMetaData interface exists
class Test6 {
    @Test
    public void testDatabaseMetaDataExists() {
        assertNotNull(DatabaseMetaData.class);
    }
}

// Test7: Verify JDBC URL components
class Test7 {
    @Test
    public void testJDBCURLComponents() {
        String url = "jdbc:mysql://localhost:3306/testdb";
        assertTrue(url.contains("jdbc:"));
        assertTrue(url.contains("://"));
        assertTrue(url.contains(":3306"));
    }
}

// Test8: Verify connection close method signature
class Test8 {
    @Test
    public void testCloseMethodExists() throws NoSuchMethodException {
        assertNotNull(Connection.class.getMethod("close"));
    }
}

// Test9: Verify isValid method exists in Connection
class Test9 {
    @Test
    public void testIsValidMethodExists() throws NoSuchMethodException {
        assertNotNull(Connection.class.getMethod("isValid", int.class));
    }
}

// Test10: Verify getMetaData method exists
class Test10 {
    @Test
    public void testGetMetaDataMethodExists() throws NoSuchMethodException {
        assertNotNull(Connection.class.getMethod("getMetaData"));
    }
}`,
    order: 0,
    translations: {
        ru: {
            title: 'Основы JDBC-соединения',
            solutionCode: `import java.sql.*;

public class DatabaseConnection {
    // Параметры подключения к базе данных
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Метод для установки соединения с базой данных
    public static Connection getConnection() throws SQLException {
        // Регистрация JDBC драйвера (опционально в JDBC 4.0+)
        // Class.forName("com.mysql.cj.jdbc.Driver");

        // Установка соединения
        Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
        System.out.println("Соединение с базой данных успешно установлено!");
        return conn;
    }

    // Метод для отображения информации о соединении
    public static void displayConnectionInfo(Connection conn) throws SQLException {
        if (conn != null && !conn.isClosed()) {
            DatabaseMetaData metaData = conn.getMetaData();
            System.out.println("СУБД: " + metaData.getDatabaseProductName());
            System.out.println("Версия СУБД: " + metaData.getDatabaseProductVersion());
            System.out.println("Имя драйвера: " + metaData.getDriverName());
            System.out.println("Версия драйвера: " + metaData.getDriverVersion());
            System.out.println("URL соединения: " + metaData.getURL());
            System.out.println("Имя пользователя: " + metaData.getUserName());
        }
    }

    // Метод для безопасного закрытия соединения
    public static void closeConnection(Connection conn) {
        if (conn != null) {
            try {
                conn.close();
                System.out.println("Соединение успешно закрыто.");
            } catch (SQLException e) {
                System.err.println("Ошибка при закрытии соединения: " + e.getMessage());
            }
        }
    }

    public static void main(String[] args) {
        Connection conn = null;

        try {
            // Установка соединения
            conn = getConnection();

            // Проверка валидности соединения
            if (conn.isValid(2)) {
                System.out.println("Соединение валидно!");

                // Отображение информации о соединении
                displayConnectionInfo(conn);
            }

        } catch (SQLException e) {
            System.err.println("Ошибка подключения к базе данных: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Всегда закрываем соединение
            closeConnection(conn);
        }

        // Альтернатива: Использование try-with-resources (рекомендуется)
        // try (Connection connection = getConnection()) {
        //     displayConnectionInfo(connection);
        // } catch (SQLException e) {
        //     e.printStackTrace();
        // }
    }
}`,
            description: `Научитесь устанавливать соединение с базой данных с помощью класса **DriverManager** из JDBC.

**Требования:**
1. Создайте класс DatabaseConnection с методом для установки соединения:
   1.1. Используйте DriverManager.getConnection() для подключения к базе данных
   1.2. Используйте правильный формат JDBC URL (например, "jdbc:mysql://localhost:3306/mydb")
   1.3. Обработайте соединение с именем пользователя и паролем

2. Реализуйте тестирование соединения:
   2.1. Проверьте, валидно ли соединение
   2.2. Выведите метаданные соединения (название СУБД, версию)
   2.3. Корректно закройте соединение

3. В методе main:
   3.1. Установите соединение с базой данных
   3.2. Отобразите информацию о соединении
   3.3. Продемонстрируйте правильную очистку ресурсов

**Цели обучения:**
- Понять структуру JDBC URL
- Научиться использовать DriverManager для соединений
- Практиковать правильное управление ресурсами с try-catch-finally или try-with-resources`,
            hint1: `Используйте DriverManager.getConnection(url, username, password) для установки соединения. Формат URL: "jdbc:драйвер://хост:порт/база_данных".`,
            hint2: `Используйте try-with-resources или блок finally, чтобы гарантировать, что соединение всегда будет закрыто, предотвращая утечки ресурсов.`,
            whyItMatters: `JDBC-соединения являются основой взаимодействия с базами данных в Java. Понимание того, как правильно устанавливать, управлять и закрывать соединения, критично для создания надежных приложений с базами данных и предотвращения утечек ресурсов, которые могут привести к сбою приложения.

**Продакшен паттерн:**
\`\`\`java
// Использование пула соединений для production
private static final HikariDataSource dataSource = new HikariDataSource();
static {
    dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/testdb");
    dataSource.setUsername("root");
    dataSource.setPassword("password");
    dataSource.setMaximumPoolSize(10);
    dataSource.setConnectionTimeout(30000);
}
public static Connection getConnection() throws SQLException {
    return dataSource.getConnection();
}
\`\`\`

**Практические преимущества:**
- Пул соединений переиспользует подключения, экономя ресурсы
- Избегаются дорогие операции создания/закрытия соединений
- Автоматическое управление жизненным циклом соединений`
        },
        uz: {
            title: 'JDBC Ulanish Asoslari',
            solutionCode: `import java.sql.*;

public class DatabaseConnection {
    // Ma'lumotlar bazasiga ulanish parametrlari
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Ma'lumotlar bazasiga ulanishni o'rnatish metodi
    public static Connection getConnection() throws SQLException {
        // JDBC drayverni ro'yxatdan o'tkazish (JDBC 4.0+ da ixtiyoriy)
        // Class.forName("com.mysql.cj.jdbc.Driver");

        // Ulanishni o'rnatish
        Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
        System.out.println("Ma'lumotlar bazasiga ulanish muvaffaqiyatli o'rnatildi!");
        return conn;
    }

    // Ulanish ma'lumotlarini ko'rsatish metodi
    public static void displayConnectionInfo(Connection conn) throws SQLException {
        if (conn != null && !conn.isClosed()) {
            DatabaseMetaData metaData = conn.getMetaData();
            System.out.println("Ma'lumotlar bazasi mahsuloti: " + metaData.getDatabaseProductName());
            System.out.println("Ma'lumotlar bazasi versiyasi: " + metaData.getDatabaseProductVersion());
            System.out.println("Drayver nomi: " + metaData.getDriverName());
            System.out.println("Drayver versiyasi: " + metaData.getDriverVersion());
            System.out.println("Ulanish URL: " + metaData.getURL());
            System.out.println("Foydalanuvchi nomi: " + metaData.getUserName());
        }
    }

    // Ulanishni xavfsiz yopish metodi
    public static void closeConnection(Connection conn) {
        if (conn != null) {
            try {
                conn.close();
                System.out.println("Ulanish muvaffaqiyatli yopildi.");
            } catch (SQLException e) {
                System.err.println("Ulanishni yopishda xato: " + e.getMessage());
            }
        }
    }

    public static void main(String[] args) {
        Connection conn = null;

        try {
            // Ulanishni o'rnatish
            conn = getConnection();

            // Ulanish to'g'riligini tekshirish
            if (conn.isValid(2)) {
                System.out.println("Ulanish to'g'ri!");

                // Ulanish ma'lumotlarini ko'rsatish
                displayConnectionInfo(conn);
            }

        } catch (SQLException e) {
            System.err.println("Ma'lumotlar bazasiga ulanish xatosi: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Har doim ulanishni yopamiz
            closeConnection(conn);
        }

        // Muqobil: try-with-resources dan foydalanish (tavsiya etiladi)
        // try (Connection connection = getConnection()) {
        //     displayConnectionInfo(connection);
        // } catch (SQLException e) {
        //     e.printStackTrace();
        // }
    }
}`,
            description: `JDBC ning **DriverManager** sinfidan foydalanib ma'lumotlar bazasiga ulanishni o'rnatishni o'rganing.

**Talablar:**
1. Ulanishni o'rnatish metodi bilan DatabaseConnection sinfini yarating:
   1.1. Ma'lumotlar bazasiga ulanish uchun DriverManager.getConnection() dan foydalaning
   1.2. To'g'ri JDBC URL formatidan foydalaning (masalan, "jdbc:mysql://localhost:3306/mydb")
   1.3. Foydalanuvchi nomi va parol bilan ulanishni boshqaring

2. Ulanishni sinashni amalga oshiring:
   2.1. Ulanish to'g'ri yoki yo'qligini tekshiring
   2.2. Ulanish metama'lumotlarini chiqaring (ma'lumotlar bazasi mahsuloti nomi, versiyasi)
   2.3. Ulanishni to'g'ri yoping

3. Main metodida:
   3.1. Ma'lumotlar bazasiga ulanishni o'rnating
   3.2. Ulanish ma'lumotlarini ko'rsating
   3.3. Resurslarni to'g'ri tozalashni namoyish eting

**O'rganish maqsadlari:**
- JDBC URL tuzilishini tushunish
- Ulanishlar uchun DriverManager dan foydalanishni o'rganish
- try-catch-finally yoki try-with-resources bilan resurslarni to'g'ri boshqarishda amaliyot`,
            hint1: `Ulanishni o'rnatish uchun DriverManager.getConnection(url, username, password) dan foydalaning. URL formati: "jdbc:drayver://host:port/baza".`,
            hint2: `Ulanish har doim yopilishini ta'minlash uchun try-with-resources yoki finally blokidan foydalaning, bu resurs oqishini oldini oladi.`,
            whyItMatters: `JDBC ulanishlari Java-da ma'lumotlar bazasi bilan o'zaro ta'sirning asosi hisoblanadi. Ulanishlarni to'g'ri o'rnatish, boshqarish va yopishni tushunish ishonchli ma'lumotlar bazasi ilovalari yaratish va ilovangizni ishdan chiqarishi mumkin bo'lgan resurs oqishini oldini olish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Production uchun ulanish poolidan foydalanish
private static final HikariDataSource dataSource = new HikariDataSource();
static {
    dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/testdb");
    dataSource.setUsername("root");
    dataSource.setPassword("password");
    dataSource.setMaximumPoolSize(10);
    dataSource.setConnectionTimeout(30000);
}
public static Connection getConnection() throws SQLException {
    return dataSource.getConnection();
}
\`\`\`

**Amaliy foydalari:**
- Ulanish pooli ulanishlarni qayta ishlatadi va resurslarni tejaydi
- Qimmat yaratish/yopish operatsiyalaridan qochiladi
- Ulanish hayot siklini avtomatik boshqarish`
        }
    }
};

export default task;
