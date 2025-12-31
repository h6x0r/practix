import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-jdbc-statement-query',
    title: 'Statement and Query Execution',
    difficulty: 'easy',
    tags: ['java', 'jdbc', 'statement', 'resultset', 'query'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to execute SQL queries using JDBC's **Statement** interface and process results with **ResultSet**.

**Requirements:**
1. Create a QueryExecutor class that executes SELECT queries:
   1.1. Use Statement to execute queries
   1.2. Process ResultSet to retrieve data
   1.3. Handle different data types (String, int, double, etc.)

2. Implement methods to:
   2.1. Execute a SELECT query to retrieve all records
   2.2. Display results in a formatted table
   2.3. Count the total number of records

3. In the main method:
   3.1. Connect to a database
   3.2. Execute a query to retrieve user data (id, name, email, age)
   3.3. Print all records in a readable format
   3.4. Properly close all resources

**Learning Goals:**
- Understand how to create and use Statement objects
- Learn to execute queries with executeQuery()
- Master ResultSet navigation and data retrieval`,
    initialCode: `import java.sql.*;

public class QueryExecutor {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // TODO: Implement method to execute SELECT query

    // TODO: Implement method to display results

    public static void main(String[] args) {
        // TODO: Execute query and display results
    }
}`,
    solutionCode: `import java.sql.*;

public class QueryExecutor {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Method to execute a SELECT query and process results
    public static void executeQuery(String query) {
        Connection conn = null;
        Statement stmt = null;
        ResultSet rs = null;

        try {
            // Establish connection
            conn = DriverManager.getConnection(URL, USER, PASSWORD);
            System.out.println("Connected to database successfully!");

            // Create Statement object
            stmt = conn.createStatement();

            // Execute query
            rs = stmt.executeQuery(query);

            // Display results
            displayResults(rs);

        } catch (SQLException e) {
            System.err.println("Error executing query: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Close resources in reverse order
            closeResources(rs, stmt, conn);
        }
    }

    // Method to display results in a formatted table
    public static void displayResults(ResultSet rs) throws SQLException {
        // Get metadata to determine column count and names
        ResultSetMetaData metaData = rs.getMetaData();
        int columnCount = metaData.getColumnCount();

        // Print column headers
        System.out.println("\\n" + "=".repeat(80));
        for (int i = 1; i <= columnCount; i++) {
            System.out.printf("%-20s", metaData.getColumnName(i));
        }
        System.out.println("\\n" + "=".repeat(80));

        // Iterate through result set
        int rowCount = 0;
        while (rs.next()) {
            for (int i = 1; i <= columnCount; i++) {
                System.out.printf("%-20s", rs.getString(i));
            }
            System.out.println();
            rowCount++;
        }

        System.out.println("=".repeat(80));
        System.out.println("Total records: " + rowCount);
    }

    // Method to count records
    public static int countRecords(String tableName) {
        String query = "SELECT COUNT(*) FROM " + tableName;
        int count = 0;

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            if (rs.next()) {
                count = rs.getInt(1);
            }

        } catch (SQLException e) {
            System.err.println("Error counting records: " + e.getMessage());
        }

        return count;
    }

    // Method to close resources safely
    public static void closeResources(ResultSet rs, Statement stmt, Connection conn) {
        try {
            if (rs != null) rs.close();
            if (stmt != null) stmt.close();
            if (conn != null) conn.close();
            System.out.println("All resources closed successfully.");
        } catch (SQLException e) {
            System.err.println("Error closing resources: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Example 1: Query all users
        String queryAllUsers = "SELECT id, name, email, age FROM users";
        System.out.println("Executing query: " + queryAllUsers);
        executeQuery(queryAllUsers);

        // Example 2: Query with WHERE clause
        String queryAdults = "SELECT id, name, age FROM users WHERE age >= 18";
        System.out.println("\\nExecuting query: " + queryAdults);
        executeQuery(queryAdults);

        // Example 3: Count records
        int totalUsers = countRecords("users");
        System.out.println("\\nTotal users in database: " + totalUsers);

        // Alternative: Using try-with-resources (recommended)
        String query = "SELECT * FROM users LIMIT 5";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            System.out.println("\\nFirst 5 users:");
            while (rs.next()) {
                int id = rs.getInt("id");
                String name = rs.getString("name");
                String email = rs.getString("email");
                int age = rs.getInt("age");

                System.out.printf("ID: %d, Name: %s, Email: %s, Age: %d%n",
                        id, name, email, age);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}`,
    hint1: `Create a Statement using connection.createStatement(), then use statement.executeQuery(sql) to execute SELECT queries. This returns a ResultSet.`,
    hint2: `Use rs.next() to iterate through the ResultSet. Access data with rs.getInt(), rs.getString(), etc. Always close resources using try-with-resources or finally blocks.`,
    whyItMatters: `The Statement interface is the foundation for executing SQL queries in JDBC. Understanding how to execute queries and process results is essential for any database-driven application. Proper resource management prevents memory leaks and database connection exhaustion.

**Production Pattern:**
\`\`\`java
// Pagination pattern for large results
public List<User> getUsersPaginated(int page, int pageSize) {
    String sql = "SELECT * FROM users LIMIT ? OFFSET ?";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement stmt = conn.prepareStatement(sql)) {
        stmt.setInt(1, pageSize);
        stmt.setInt(2, page * pageSize);
        try (ResultSet rs = stmt.executeQuery()) {
            List<User> users = new ArrayList<>();
            while (rs.next()) {
                users.add(mapToUser(rs));
            }
            return users;
        }
    } catch (SQLException e) {
        logger.error("Failed to fetch users", e);
        throw new DataAccessException(e);
    }
}
\`\`\`

**Practical Benefits:**
- Pagination prevents loading millions of rows into memory
- Try-with-resources automatically closes all resources
- PreparedStatement protects against SQL injection`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.sql.*;

// Test1: Verify Statement interface exists
class Test1 {
    @Test
    public void testStatementInterfaceExists() {
        assertNotNull(Statement.class);
    }
}

// Test2: Verify ResultSet interface exists
class Test2 {
    @Test
    public void testResultSetInterfaceExists() {
        assertNotNull(ResultSet.class);
    }
}

// Test3: Verify executeQuery method exists
class Test3 {
    @Test
    public void testExecuteQueryMethodExists() throws NoSuchMethodException {
        assertNotNull(Statement.class.getMethod("executeQuery", String.class));
    }
}

// Test4: Verify ResultSet has next() method
class Test4 {
    @Test
    public void testResultSetNextMethod() throws NoSuchMethodException {
        assertNotNull(ResultSet.class.getMethod("next"));
    }
}

// Test5: Verify ResultSet has getString() method
class Test5 {
    @Test
    public void testResultSetGetStringMethod() throws NoSuchMethodException {
        assertNotNull(ResultSet.class.getMethod("getString", String.class));
    }
}

// Test6: Verify ResultSet has getInt() method
class Test6 {
    @Test
    public void testResultSetGetIntMethod() throws NoSuchMethodException {
        assertNotNull(ResultSet.class.getMethod("getInt", String.class));
    }
}

// Test7: Verify SELECT query syntax
class Test7 {
    @Test
    public void testSelectQuerySyntax() {
        String query = "SELECT id, name FROM users";
        assertTrue(query.toUpperCase().contains("SELECT"));
        assertTrue(query.toUpperCase().contains("FROM"));
    }
}

// Test8: Verify Statement close method exists
class Test8 {
    @Test
    public void testStatementCloseMethod() throws NoSuchMethodException {
        assertNotNull(Statement.class.getMethod("close"));
    }
}

// Test9: Verify ResultSet close method exists
class Test9 {
    @Test
    public void testResultSetCloseMethod() throws NoSuchMethodException {
        assertNotNull(ResultSet.class.getMethod("close"));
    }
}

// Test10: Verify ResultSet metadata method exists
class Test10 {
    @Test
    public void testResultSetMetaDataMethod() throws NoSuchMethodException {
        assertNotNull(ResultSet.class.getMethod("getMetaData"));
    }
}`,
    order: 1,
    translations: {
        ru: {
            title: 'Statement и Выполнение Запросов',
            solutionCode: `import java.sql.*;

public class QueryExecutor {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Метод для выполнения SELECT запроса и обработки результатов
    public static void executeQuery(String query) {
        Connection conn = null;
        Statement stmt = null;
        ResultSet rs = null;

        try {
            // Установка соединения
            conn = DriverManager.getConnection(URL, USER, PASSWORD);
            System.out.println("Успешное подключение к базе данных!");

            // Создание объекта Statement
            stmt = conn.createStatement();

            // Выполнение запроса
            rs = stmt.executeQuery(query);

            // Отображение результатов
            displayResults(rs);

        } catch (SQLException e) {
            System.err.println("Ошибка при выполнении запроса: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Закрытие ресурсов в обратном порядке
            closeResources(rs, stmt, conn);
        }
    }

    // Метод для отображения результатов в форматированной таблице
    public static void displayResults(ResultSet rs) throws SQLException {
        // Получение метаданных для определения количества и имен столбцов
        ResultSetMetaData metaData = rs.getMetaData();
        int columnCount = metaData.getColumnCount();

        // Вывод заголовков столбцов
        System.out.println("\\n" + "=".repeat(80));
        for (int i = 1; i <= columnCount; i++) {
            System.out.printf("%-20s", metaData.getColumnName(i));
        }
        System.out.println("\\n" + "=".repeat(80));

        // Итерация по набору результатов
        int rowCount = 0;
        while (rs.next()) {
            for (int i = 1; i <= columnCount; i++) {
                System.out.printf("%-20s", rs.getString(i));
            }
            System.out.println();
            rowCount++;
        }

        System.out.println("=".repeat(80));
        System.out.println("Всего записей: " + rowCount);
    }

    // Метод для подсчета записей
    public static int countRecords(String tableName) {
        String query = "SELECT COUNT(*) FROM " + tableName;
        int count = 0;

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            if (rs.next()) {
                count = rs.getInt(1);
            }

        } catch (SQLException e) {
            System.err.println("Ошибка при подсчете записей: " + e.getMessage());
        }

        return count;
    }

    // Метод для безопасного закрытия ресурсов
    public static void closeResources(ResultSet rs, Statement stmt, Connection conn) {
        try {
            if (rs != null) rs.close();
            if (stmt != null) stmt.close();
            if (conn != null) conn.close();
            System.out.println("Все ресурсы успешно закрыты.");
        } catch (SQLException e) {
            System.err.println("Ошибка при закрытии ресурсов: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Пример 1: Запрос всех пользователей
        String queryAllUsers = "SELECT id, name, email, age FROM users";
        System.out.println("Выполнение запроса: " + queryAllUsers);
        executeQuery(queryAllUsers);

        // Пример 2: Запрос с условием WHERE
        String queryAdults = "SELECT id, name, age FROM users WHERE age >= 18";
        System.out.println("\\nВыполнение запроса: " + queryAdults);
        executeQuery(queryAdults);

        // Пример 3: Подсчет записей
        int totalUsers = countRecords("users");
        System.out.println("\\nВсего пользователей в базе данных: " + totalUsers);

        // Альтернатива: Использование try-with-resources (рекомендуется)
        String query = "SELECT * FROM users LIMIT 5";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            System.out.println("\\nПервые 5 пользователей:");
            while (rs.next()) {
                int id = rs.getInt("id");
                String name = rs.getString("name");
                String email = rs.getString("email");
                int age = rs.getInt("age");

                System.out.printf("ID: %d, Имя: %s, Email: %s, Возраст: %d%n",
                        id, name, email, age);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}`,
            description: `Научитесь выполнять SQL-запросы с помощью интерфейса **Statement** JDBC и обрабатывать результаты с помощью **ResultSet**.

**Требования:**
1. Создайте класс QueryExecutor, который выполняет SELECT запросы:
   1.1. Используйте Statement для выполнения запросов
   1.2. Обрабатывайте ResultSet для получения данных
   1.3. Работайте с различными типами данных (String, int, double и т.д.)

2. Реализуйте методы для:
   2.1. Выполнения SELECT запроса для получения всех записей
   2.2. Отображения результатов в форматированной таблице
   2.3. Подсчета общего количества записей

3. В методе main:
   3.1. Подключитесь к базе данных
   3.2. Выполните запрос для получения данных пользователей (id, name, email, age)
   3.3. Выведите все записи в читаемом формате
   3.4. Правильно закройте все ресурсы

**Цели обучения:**
- Понять, как создавать и использовать объекты Statement
- Научиться выполнять запросы с помощью executeQuery()
- Освоить навигацию по ResultSet и получение данных`,
            hint1: `Создайте Statement с помощью connection.createStatement(), затем используйте statement.executeQuery(sql) для выполнения SELECT запросов. Это вернет ResultSet.`,
            hint2: `Используйте rs.next() для итерации по ResultSet. Доступ к данным осуществляется через rs.getInt(), rs.getString() и т.д. Всегда закрывайте ресурсы с помощью try-with-resources или блоков finally.`,
            whyItMatters: `Интерфейс Statement является основой для выполнения SQL-запросов в JDBC. Понимание того, как выполнять запросы и обрабатывать результаты, необходимо для любого приложения, работающего с базами данных. Правильное управление ресурсами предотвращает утечки памяти и исчерпание соединений с базой данных.

**Продакшен паттерн:**
\`\`\`java
// Паттерн с пагинацией для больших результатов
public List<User> getUsersPaginated(int page, int pageSize) {
    String sql = "SELECT * FROM users LIMIT ? OFFSET ?";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement stmt = conn.prepareStatement(sql)) {
        stmt.setInt(1, pageSize);
        stmt.setInt(2, page * pageSize);
        try (ResultSet rs = stmt.executeQuery()) {
            List<User> users = new ArrayList<>();
            while (rs.next()) {
                users.add(mapToUser(rs));
            }
            return users;
        }
    } catch (SQLException e) {
        logger.error("Failed to fetch users", e);
        throw new DataAccessException(e);
    }
}
\`\`\`

**Практические преимущества:**
- Пагинация предотвращает загрузку миллионов строк в память
- Try-with-resources автоматически закрывает все ресурсы
- PreparedStatement защищает от SQL-инъекций`
        },
        uz: {
            title: 'Statement va So\'rovlarni Bajarish',
            solutionCode: `import java.sql.*;

public class QueryExecutor {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // SELECT so'rovini bajarish va natijalarni qayta ishlash metodi
    public static void executeQuery(String query) {
        Connection conn = null;
        Statement stmt = null;
        ResultSet rs = null;

        try {
            // Ulanishni o'rnatish
            conn = DriverManager.getConnection(URL, USER, PASSWORD);
            System.out.println("Ma'lumotlar bazasiga muvaffaqiyatli ulandi!");

            // Statement obyektini yaratish
            stmt = conn.createStatement();

            // So'rovni bajarish
            rs = stmt.executeQuery(query);

            // Natijalarni ko'rsatish
            displayResults(rs);

        } catch (SQLException e) {
            System.err.println("So'rovni bajarishda xato: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Resurslarni teskari tartibda yopish
            closeResources(rs, stmt, conn);
        }
    }

    // Natijalarni formatlangan jadvalda ko'rsatish metodi
    public static void displayResults(ResultSet rs) throws SQLException {
        // Ustun soni va nomlarini aniqlash uchun metama'lumotlarni olish
        ResultSetMetaData metaData = rs.getMetaData();
        int columnCount = metaData.getColumnCount();

        // Ustun sarlavhalarini chiqarish
        System.out.println("\\n" + "=".repeat(80));
        for (int i = 1; i <= columnCount; i++) {
            System.out.printf("%-20s", metaData.getColumnName(i));
        }
        System.out.println("\\n" + "=".repeat(80));

        // Natijalar to'plami bo'ylab iteratsiya
        int rowCount = 0;
        while (rs.next()) {
            for (int i = 1; i <= columnCount; i++) {
                System.out.printf("%-20s", rs.getString(i));
            }
            System.out.println();
            rowCount++;
        }

        System.out.println("=".repeat(80));
        System.out.println("Jami yozuvlar: " + rowCount);
    }

    // Yozuvlarni sanash metodi
    public static int countRecords(String tableName) {
        String query = "SELECT COUNT(*) FROM " + tableName;
        int count = 0;

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            if (rs.next()) {
                count = rs.getInt(1);
            }

        } catch (SQLException e) {
            System.err.println("Yozuvlarni sanashda xato: " + e.getMessage());
        }

        return count;
    }

    // Resurslarni xavfsiz yopish metodi
    public static void closeResources(ResultSet rs, Statement stmt, Connection conn) {
        try {
            if (rs != null) rs.close();
            if (stmt != null) stmt.close();
            if (conn != null) conn.close();
            System.out.println("Barcha resurslar muvaffaqiyatli yopildi.");
        } catch (SQLException e) {
            System.err.println("Resurslarni yopishda xato: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // 1-misol: Barcha foydalanuvchilarni so'rash
        String queryAllUsers = "SELECT id, name, email, age FROM users";
        System.out.println("So'rovni bajarish: " + queryAllUsers);
        executeQuery(queryAllUsers);

        // 2-misol: WHERE sharti bilan so'rov
        String queryAdults = "SELECT id, name, age FROM users WHERE age >= 18";
        System.out.println("\\nSo'rovni bajarish: " + queryAdults);
        executeQuery(queryAdults);

        // 3-misol: Yozuvlarni sanash
        int totalUsers = countRecords("users");
        System.out.println("\\nMa'lumotlar bazasidagi jami foydalanuvchilar: " + totalUsers);

        // Muqobil: try-with-resources dan foydalanish (tavsiya etiladi)
        String query = "SELECT * FROM users LIMIT 5";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query)) {

            System.out.println("\\nBirinchi 5 foydalanuvchi:");
            while (rs.next()) {
                int id = rs.getInt("id");
                String name = rs.getString("name");
                String email = rs.getString("email");
                int age = rs.getInt("age");

                System.out.printf("ID: %d, Ism: %s, Email: %s, Yosh: %d%n",
                        id, name, email, age);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}`,
            description: `JDBC ning **Statement** interfeysi yordamida SQL so'rovlarini bajarishni va **ResultSet** bilan natijalarni qayta ishlashni o'rganing.

**Talablar:**
1. SELECT so'rovlarini bajaradigan QueryExecutor sinfini yarating:
   1.1. So'rovlarni bajarish uchun Statement dan foydalaning
   1.2. Ma'lumotlarni olish uchun ResultSet ni qayta ishlang
   1.3. Turli ma'lumot turlarini boshqaring (String, int, double va boshqalar)

2. Quyidagi metodlarni amalga oshiring:
   2.1. Barcha yozuvlarni olish uchun SELECT so'rovini bajarish
   2.2. Natijalarni formatlangan jadvalda ko'rsatish
   2.3. Jami yozuvlar sonini sanash

3. Main metodida:
   3.1. Ma'lumotlar bazasiga ulaning
   3.2. Foydalanuvchi ma'lumotlarini olish uchun so'rov bajaring (id, name, email, age)
   3.3. Barcha yozuvlarni o'qish mumkin formatda chiqaring
   3.4. Barcha resurslarni to'g'ri yoping

**O'rganish maqsadlari:**
- Statement obyektlarini qanday yaratish va ishlatishni tushunish
- executeQuery() yordamida so'rovlarni bajarishni o'rganish
- ResultSet bo'ylab navigatsiya va ma'lumotlarni olishda mohirlik`,
            hint1: `connection.createStatement() yordamida Statement yarating, keyin SELECT so'rovlarini bajarish uchun statement.executeQuery(sql) dan foydalaning. Bu ResultSet qaytaradi.`,
            hint2: `ResultSet bo'ylab iteratsiya qilish uchun rs.next() dan foydalaning. Ma'lumotlarga rs.getInt(), rs.getString() va boshqalar orqali kirish mumkin. Har doim try-with-resources yoki finally bloklari yordamida resurslarni yoping.`,
            whyItMatters: `Statement interfeysi JDBC da SQL so'rovlarini bajarish uchun asos hisoblanadi. So'rovlarni qanday bajarish va natijalarni qayta ishlashni tushunish ma'lumotlar bazasi bilan ishlaydigan har qanday ilova uchun zarur. To'g'ri resurs boshqaruvi xotira oqishi va ma'lumotlar bazasi ulanishlarining tugashini oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Katta natijalar uchun pagination patterni
public List<User> getUsersPaginated(int page, int pageSize) {
    String sql = "SELECT * FROM users LIMIT ? OFFSET ?";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement stmt = conn.prepareStatement(sql)) {
        stmt.setInt(1, pageSize);
        stmt.setInt(2, page * pageSize);
        try (ResultSet rs = stmt.executeQuery()) {
            List<User> users = new ArrayList<>();
            while (rs.next()) {
                users.add(mapToUser(rs));
            }
            return users;
        }
    } catch (SQLException e) {
        logger.error("Failed to fetch users", e);
        throw new DataAccessException(e);
    }
}
\`\`\`

**Amaliy foydalari:**
- Pagination millionlab qatorlarni xotiraga yuklashning oldini oladi
- Try-with-resources barcha resurslarni avtomatik yopadi
- PreparedStatement SQL injeksiyasidan himoya qiladi`
        }
    }
};

export default task;
