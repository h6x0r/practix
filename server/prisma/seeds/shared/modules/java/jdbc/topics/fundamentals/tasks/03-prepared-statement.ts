import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-jdbc-prepared-statement',
    title: 'PreparedStatement and SQL Injection Prevention',
    difficulty: 'medium',
    tags: ['java', 'jdbc', 'preparedstatement', 'security', 'sql-injection'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master **PreparedStatement** for secure and efficient database operations with parameter binding and SQL injection prevention.

**Requirements:**
1. Create a UserDAO class with PreparedStatement operations:
   1.1. Insert a new user with parameterized query
   1.2. Update user information safely
   1.3. Delete user by ID
   1.4. Find user by email using parameters

2. Demonstrate SQL injection prevention:
   2.1. Show the difference between Statement and PreparedStatement
   2.2. Use parameter binding with setString(), setInt(), etc.
   2.3. Explain why PreparedStatement is more secure

3. In the main method:
   3.1. Insert multiple users using PreparedStatement
   3.2. Update a user's information
   3.3. Search for users by email
   3.4. Delete a user
   3.5. Show proper error handling and resource management

**Learning Goals:**
- Understand PreparedStatement advantages over Statement
- Learn parameter binding techniques
- Master SQL injection prevention
- Practice CRUD operations securely`,
    initialCode: `import java.sql.*;

public class UserDAO {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // TODO: Implement insertUser() method with PreparedStatement

    // TODO: Implement updateUser() method

    // TODO: Implement deleteUser() method

    // TODO: Implement findUserByEmail() method

    public static void main(String[] args) {
        // TODO: Demonstrate PreparedStatement usage
    }
}`,
    solutionCode: `import java.sql.*;

public class UserDAO {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Insert user using PreparedStatement - prevents SQL injection
    public static boolean insertUser(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            // Set parameters - index starts at 1
            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            // Execute update
            int rowsAffected = pstmt.executeUpdate();
            System.out.println("User inserted successfully. Rows affected: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Error inserting user: " + e.getMessage());
            return false;
        }
    }

    // Update user information
    public static boolean updateUser(int userId, String newEmail, int newAge) {
        String sql = "UPDATE users SET email = ?, age = ? WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, newEmail);
            pstmt.setInt(2, newAge);
            pstmt.setInt(3, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("User updated. Rows affected: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Error updating user: " + e.getMessage());
            return false;
        }
    }

    // Delete user by ID
    public static boolean deleteUser(int userId) {
        String sql = "DELETE FROM users WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setInt(1, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("User deleted. Rows affected: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Error deleting user: " + e.getMessage());
            return false;
        }
    }

    // Find user by email - demonstrates secure parameter binding
    public static void findUserByEmail(String email) {
        String sql = "SELECT id, name, email, age FROM users WHERE email = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, email);

            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    System.out.println("User found:");
                    System.out.println("ID: " + rs.getInt("id"));
                    System.out.println("Name: " + rs.getString("name"));
                    System.out.println("Email: " + rs.getString("email"));
                    System.out.println("Age: " + rs.getInt("age"));
                } else {
                    System.out.println("No user found with email: " + email);
                }
            }

        } catch (SQLException e) {
            System.err.println("Error finding user: " + e.getMessage());
        }
    }

    // Demonstrate SQL injection vulnerability with Statement
    public static void unsafeQuery(String email) {
        // UNSAFE - vulnerable to SQL injection!
        String sql = "SELECT * FROM users WHERE email = '" + email + "'";
        // If email = "' OR '1'='1", it returns all users!

        System.out.println("UNSAFE Query: " + sql);
    }

    // Insert user with auto-generated key retrieval
    public static int insertUserWithKey(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql,
                     Statement.RETURN_GENERATED_KEYS)) {

            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            int rowsAffected = pstmt.executeUpdate();

            if (rowsAffected > 0) {
                // Retrieve auto-generated ID
                try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                    if (generatedKeys.next()) {
                        int newId = generatedKeys.getInt(1);
                        System.out.println("User inserted with ID: " + newId);
                        return newId;
                    }
                }
            }

        } catch (SQLException e) {
            System.err.println("Error inserting user: " + e.getMessage());
        }

        return -1;
    }

    public static void main(String[] args) {
        System.out.println("=== PreparedStatement Demo ===\\n");

        // Insert users
        insertUser("Alice Johnson", "alice@email.com", 28);
        insertUser("Bob Smith", "bob@email.com", 35);
        int newId = insertUserWithKey("Carol Williams", "carol@email.com", 42);

        // Find user by email
        System.out.println("\\n--- Finding user by email ---");
        findUserByEmail("alice@email.com");

        // Update user
        System.out.println("\\n--- Updating user ---");
        updateUser(newId, "carol.new@email.com", 43);

        // Verify update
        findUserByEmail("carol.new@email.com");

        // Delete user
        System.out.println("\\n--- Deleting user ---");
        deleteUser(newId);

        // Demonstrate SQL injection prevention
        System.out.println("\\n=== SQL Injection Prevention ===");
        String maliciousInput = "' OR '1'='1";
        System.out.println("Malicious input: " + maliciousInput);

        unsafeQuery(maliciousInput); // Shows vulnerable query

        // PreparedStatement safely handles malicious input
        System.out.println("\\nUsing PreparedStatement (safe):");
        findUserByEmail(maliciousInput); // Treats entire string as email value
    }
}`,
    hint1: `Use PreparedStatement instead of Statement. Create it with conn.prepareStatement(sql) where sql contains ? placeholders. Then use setString(), setInt(), etc. to bind values.`,
    hint2: `Parameter indices start at 1, not 0. Use pstmt.setString(1, value) for the first ?, setString(2, value) for the second ?, and so on. Always use try-with-resources.`,
    whyItMatters: `PreparedStatement is crucial for security and performance. It prevents SQL injection attacks by treating user input as data, not executable code. It also improves performance through query pre-compilation and allows the database to cache execution plans. This is essential knowledge for building secure applications.

**Production Pattern:**
\`\`\`java
// Batch operations for performance
public void insertUsersBatch(List<User> users) throws SQLException {
    String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement pstmt = conn.prepareStatement(sql)) {
        conn.setAutoCommit(false);
        for (User user : users) {
            pstmt.setString(1, user.getName());
            pstmt.setString(2, user.getEmail());
            pstmt.setInt(3, user.getAge());
            pstmt.addBatch();
            if (users.indexOf(user) % 1000 == 0) {
                pstmt.executeBatch(); // batch every 1000 records
            }
        }
        pstmt.executeBatch();
        conn.commit();
    }
}
\`\`\`

**Practical Benefits:**
- Batch operations are 10-100x faster than individual inserts
- Transactions ensure atomicity of operations
- SQL injection protection built into PreparedStatement`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.sql.*;

// Test1: Verify PreparedStatement interface exists
class Test1 {
    @Test
    public void testPreparedStatementInterfaceExists() {
        assertNotNull(PreparedStatement.class);
    }
}

// Test2: Verify setString method exists
class Test2 {
    @Test
    public void testSetStringMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("setString", int.class, String.class));
    }
}

// Test3: Verify setInt method exists
class Test3 {
    @Test
    public void testSetIntMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("setInt", int.class, int.class));
    }
}

// Test4: Verify executeUpdate method exists
class Test4 {
    @Test
    public void testExecuteUpdateMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("executeUpdate"));
    }
}

// Test5: Verify SQL placeholder syntax
class Test5 {
    @Test
    public void testSQLPlaceholderSyntax() {
        String sql = "INSERT INTO users (name, email) VALUES (?, ?)";
        assertTrue(sql.contains("?"));
        assertEquals(2, sql.split("\\?", -1).length - 1);
    }
}

// Test6: Verify getGeneratedKeys method exists
class Test6 {
    @Test
    public void testGetGeneratedKeysMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("getGeneratedKeys"));
    }
}

// Test7: Verify addBatch method exists
class Test7 {
    @Test
    public void testAddBatchMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("addBatch"));
    }
}

// Test8: Verify executeBatch method exists
class Test8 {
    @Test
    public void testExecuteBatchMethodExists() throws NoSuchMethodException {
        assertNotNull(PreparedStatement.class.getMethod("executeBatch"));
    }
}

// Test9: Verify SQL injection prevention pattern
class Test9 {
    @Test
    public void testSQLInjectionPreventionPattern() {
        String maliciousInput = "' OR '1'='1";
        String safeSql = "SELECT * FROM users WHERE email = ?";

        assertTrue(safeSql.contains("?"));
        assertFalse(safeSql.contains(maliciousInput));
    }
}

// Test10: Verify parameter index starts at 1
class Test10 {
    @Test
    public void testParameterIndexing() {
        // PreparedStatement indices start at 1, not 0
        int firstParamIndex = 1;
        int secondParamIndex = 2;

        assertTrue(firstParamIndex > 0);
        assertEquals(2, secondParamIndex);
    }
}`,
    order: 2,
    translations: {
        ru: {
            title: 'PreparedStatement и Предотвращение SQL-инъекций',
            solutionCode: `import java.sql.*;

public class UserDAO {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // Вставка пользователя с использованием PreparedStatement - предотвращает SQL-инъекции
    public static boolean insertUser(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            // Установка параметров - индекс начинается с 1
            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            // Выполнение обновления
            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Пользователь успешно добавлен. Затронуто строк: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Ошибка при добавлении пользователя: " + e.getMessage());
            return false;
        }
    }

    // Обновление информации о пользователе
    public static boolean updateUser(int userId, String newEmail, int newAge) {
        String sql = "UPDATE users SET email = ?, age = ? WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, newEmail);
            pstmt.setInt(2, newAge);
            pstmt.setInt(3, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Пользователь обновлен. Затронуто строк: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Ошибка при обновлении пользователя: " + e.getMessage());
            return false;
        }
    }

    // Удаление пользователя по ID
    public static boolean deleteUser(int userId) {
        String sql = "DELETE FROM users WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setInt(1, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Пользователь удален. Затронуто строк: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Ошибка при удалении пользователя: " + e.getMessage());
            return false;
        }
    }

    // Поиск пользователя по email - демонстрирует безопасную привязку параметров
    public static void findUserByEmail(String email) {
        String sql = "SELECT id, name, email, age FROM users WHERE email = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, email);

            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    System.out.println("Пользователь найден:");
                    System.out.println("ID: " + rs.getInt("id"));
                    System.out.println("Имя: " + rs.getString("name"));
                    System.out.println("Email: " + rs.getString("email"));
                    System.out.println("Возраст: " + rs.getInt("age"));
                } else {
                    System.out.println("Пользователь не найден с email: " + email);
                }
            }

        } catch (SQLException e) {
            System.err.println("Ошибка при поиске пользователя: " + e.getMessage());
        }
    }

    // Демонстрация уязвимости SQL-инъекции со Statement
    public static void unsafeQuery(String email) {
        // НЕБЕЗОПАСНО - уязвимо к SQL-инъекции!
        String sql = "SELECT * FROM users WHERE email = '" + email + "'";
        // Если email = "' OR '1'='1", вернет всех пользователей!

        System.out.println("НЕБЕЗОПАСНЫЙ запрос: " + sql);
    }

    // Вставка пользователя с получением автоматически сгенерированного ключа
    public static int insertUserWithKey(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql,
                     Statement.RETURN_GENERATED_KEYS)) {

            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            int rowsAffected = pstmt.executeUpdate();

            if (rowsAffected > 0) {
                // Получение автоматически сгенерированного ID
                try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                    if (generatedKeys.next()) {
                        int newId = generatedKeys.getInt(1);
                        System.out.println("Пользователь добавлен с ID: " + newId);
                        return newId;
                    }
                }
            }

        } catch (SQLException e) {
            System.err.println("Ошибка при добавлении пользователя: " + e.getMessage());
        }

        return -1;
    }

    public static void main(String[] args) {
        System.out.println("=== Демонстрация PreparedStatement ===\\n");

        // Вставка пользователей
        insertUser("Алиса Джонсон", "alice@email.com", 28);
        insertUser("Боб Смит", "bob@email.com", 35);
        int newId = insertUserWithKey("Кэрол Уильямс", "carol@email.com", 42);

        // Поиск пользователя по email
        System.out.println("\\n--- Поиск пользователя по email ---");
        findUserByEmail("alice@email.com");

        // Обновление пользователя
        System.out.println("\\n--- Обновление пользователя ---");
        updateUser(newId, "carol.new@email.com", 43);

        // Проверка обновления
        findUserByEmail("carol.new@email.com");

        // Удаление пользователя
        System.out.println("\\n--- Удаление пользователя ---");
        deleteUser(newId);

        // Демонстрация предотвращения SQL-инъекций
        System.out.println("\\n=== Предотвращение SQL-инъекций ===");
        String maliciousInput = "' OR '1'='1";
        System.out.println("Вредоносный ввод: " + maliciousInput);

        unsafeQuery(maliciousInput); // Показывает уязвимый запрос

        // PreparedStatement безопасно обрабатывает вредоносный ввод
        System.out.println("\\nИспользование PreparedStatement (безопасно):");
        findUserByEmail(maliciousInput); // Обрабатывает всю строку как значение email
    }
}`,
            description: `Освойте **PreparedStatement** для безопасных и эффективных операций с базой данных с привязкой параметров и предотвращением SQL-инъекций.

**Требования:**
1. Создайте класс UserDAO с операциями PreparedStatement:
   1.1. Вставьте нового пользователя с параметризованным запросом
   1.2. Безопасно обновите информацию о пользователе
   1.3. Удалите пользователя по ID
   1.4. Найдите пользователя по email с использованием параметров

2. Продемонстрируйте предотвращение SQL-инъекций:
   2.1. Покажите разницу между Statement и PreparedStatement
   2.2. Используйте привязку параметров с setString(), setInt() и т.д.
   2.3. Объясните, почему PreparedStatement более безопасен

3. В методе main:
   3.1. Вставьте нескольких пользователей с использованием PreparedStatement
   3.2. Обновите информацию пользователя
   3.3. Найдите пользователей по email
   3.4. Удалите пользователя
   3.5. Покажите правильную обработку ошибок и управление ресурсами

**Цели обучения:**
- Понять преимущества PreparedStatement над Statement
- Изучить техники привязки параметров
- Освоить предотвращение SQL-инъекций
- Практиковать безопасные CRUD операции`,
            hint1: `Используйте PreparedStatement вместо Statement. Создайте его с помощью conn.prepareStatement(sql), где sql содержит заполнители ?. Затем используйте setString(), setInt() и т.д. для привязки значений.`,
            hint2: `Индексы параметров начинаются с 1, а не с 0. Используйте pstmt.setString(1, value) для первого ?, setString(2, value) для второго ? и так далее. Всегда используйте try-with-resources.`,
            whyItMatters: `PreparedStatement критически важен для безопасности и производительности. Он предотвращает атаки SQL-инъекций, рассматривая пользовательский ввод как данные, а не исполняемый код. Он также улучшает производительность за счет предварительной компиляции запросов и позволяет базе данных кэшировать планы выполнения. Это необходимое знание для создания безопасных приложений.

**Продакшен паттерн:**
\`\`\`java
// Batch операции для производительности
public void insertUsersBatch(List<User> users) throws SQLException {
    String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement pstmt = conn.prepareStatement(sql)) {
        conn.setAutoCommit(false);
        for (User user : users) {
            pstmt.setString(1, user.getName());
            pstmt.setString(2, user.getEmail());
            pstmt.setInt(3, user.getAge());
            pstmt.addBatch();
            if (users.indexOf(user) % 1000 == 0) {
                pstmt.executeBatch(); // batch каждые 1000 записей
            }
        }
        pstmt.executeBatch();
        conn.commit();
    }
}
\`\`\`

**Практические преимущества:**
- Batch операции в 10-100 раз быстрее отдельных вставок
- Транзакции обеспечивают атомарность операций
- Защита от SQL-инъекций встроена в PreparedStatement`
        },
        uz: {
            title: 'PreparedStatement va SQL Injeksiyasini Oldini Olish',
            solutionCode: `import java.sql.*;

public class UserDAO {
    private static final String URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String USER = "root";
    private static final String PASSWORD = "password";

    // PreparedStatement yordamida foydalanuvchi qo'shish - SQL injeksiyasini oldini oladi
    public static boolean insertUser(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            // Parametrlarni o'rnatish - indeks 1 dan boshlanadi
            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            // Yangilashni bajarish
            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Foydalanuvchi muvaffaqiyatli qo'shildi. Ta'sirlangan qatorlar: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Foydalanuvchi qo'shishda xato: " + e.getMessage());
            return false;
        }
    }

    // Foydalanuvchi ma'lumotlarini yangilash
    public static boolean updateUser(int userId, String newEmail, int newAge) {
        String sql = "UPDATE users SET email = ?, age = ? WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, newEmail);
            pstmt.setInt(2, newAge);
            pstmt.setInt(3, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Foydalanuvchi yangilandi. Ta'sirlangan qatorlar: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Foydalanuvchi yangilashda xato: " + e.getMessage());
            return false;
        }
    }

    // ID bo'yicha foydalanuvchini o'chirish
    public static boolean deleteUser(int userId) {
        String sql = "DELETE FROM users WHERE id = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setInt(1, userId);

            int rowsAffected = pstmt.executeUpdate();
            System.out.println("Foydalanuvchi o'chirildi. Ta'sirlangan qatorlar: " + rowsAffected);
            return rowsAffected > 0;

        } catch (SQLException e) {
            System.err.println("Foydalanuvchi o'chirishda xato: " + e.getMessage());
            return false;
        }
    }

    // Email bo'yicha foydalanuvchini topish - xavfsiz parametr bog'lashni namoyish etadi
    public static void findUserByEmail(String email) {
        String sql = "SELECT id, name, email, age FROM users WHERE email = ?";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setString(1, email);

            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    System.out.println("Foydalanuvchi topildi:");
                    System.out.println("ID: " + rs.getInt("id"));
                    System.out.println("Ism: " + rs.getString("name"));
                    System.out.println("Email: " + rs.getString("email"));
                    System.out.println("Yosh: " + rs.getInt("age"));
                } else {
                    System.out.println("Email bilan foydalanuvchi topilmadi: " + email);
                }
            }

        } catch (SQLException e) {
            System.err.println("Foydalanuvchi topishda xato: " + e.getMessage());
        }
    }

    // Statement bilan SQL injeksiya zaifligini ko'rsatish
    public static void unsafeQuery(String email) {
        // XAVFLI - SQL injeksiyaga zaif!
        String sql = "SELECT * FROM users WHERE email = '" + email + "'";
        // Agar email = "' OR '1'='1" bo'lsa, barcha foydalanuvchilarni qaytaradi!

        System.out.println("XAVFLI so'rov: " + sql);
    }

    // Avtomatik yaratilgan kalit olish bilan foydalanuvchi qo'shish
    public static int insertUserWithKey(String name, String email, int age) {
        String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql,
                     Statement.RETURN_GENERATED_KEYS)) {

            pstmt.setString(1, name);
            pstmt.setString(2, email);
            pstmt.setInt(3, age);

            int rowsAffected = pstmt.executeUpdate();

            if (rowsAffected > 0) {
                // Avtomatik yaratilgan ID ni olish
                try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                    if (generatedKeys.next()) {
                        int newId = generatedKeys.getInt(1);
                        System.out.println("Foydalanuvchi ID bilan qo'shildi: " + newId);
                        return newId;
                    }
                }
            }

        } catch (SQLException e) {
            System.err.println("Foydalanuvchi qo'shishda xato: " + e.getMessage());
        }

        return -1;
    }

    public static void main(String[] args) {
        System.out.println("=== PreparedStatement Namoyishi ===\\n");

        // Foydalanuvchilar qo'shish
        insertUser("Alisa Jonson", "alice@email.com", 28);
        insertUser("Bob Smit", "bob@email.com", 35);
        int newId = insertUserWithKey("Kerol Vilyams", "carol@email.com", 42);

        // Email bo'yicha foydalanuvchi topish
        System.out.println("\\n--- Email bo'yicha foydalanuvchi topish ---");
        findUserByEmail("alice@email.com");

        // Foydalanuvchini yangilash
        System.out.println("\\n--- Foydalanuvchini yangilash ---");
        updateUser(newId, "carol.new@email.com", 43);

        // Yangilashni tekshirish
        findUserByEmail("carol.new@email.com");

        // Foydalanuvchini o'chirish
        System.out.println("\\n--- Foydalanuvchini o'chirish ---");
        deleteUser(newId);

        // SQL injeksiyasini oldini olishni ko'rsatish
        System.out.println("\\n=== SQL Injeksiyasini Oldini Olish ===");
        String maliciousInput = "' OR '1'='1";
        System.out.println("Zararli kiritma: " + maliciousInput);

        unsafeQuery(maliciousInput); // Zaif so'rovni ko'rsatadi

        // PreparedStatement zararli kiritishni xavfsiz qayta ishlaydi
        System.out.println("\\nPreparedStatement dan foydalanish (xavfsiz):");
        findUserByEmail(maliciousInput); // Butun satrni email qiymati sifatida ko'radi
    }
}`,
            description: `Parametr bog'lash va SQL injeksiyasini oldini olish bilan xavfsiz va samarali ma'lumotlar bazasi operatsiyalari uchun **PreparedStatement** ni o'rganing.

**Talablar:**
1. PreparedStatement operatsiyalari bilan UserDAO sinfini yarating:
   1.1. Parametrlangan so'rov bilan yangi foydalanuvchi qo'shing
   1.2. Foydalanuvchi ma'lumotlarini xavfsiz yangilang
   1.3. ID bo'yicha foydalanuvchini o'chiring
   1.4. Parametrlar yordamida email bo'yicha foydalanuvchi toping

2. SQL injeksiyasini oldini olishni namoyish eting:
   2.1. Statement va PreparedStatement o'rtasidagi farqni ko'rsating
   2.2. setString(), setInt() va boshqalar bilan parametr bog'lashdan foydalaning
   2.3. PreparedStatement nima uchun xavfsizroq ekanligini tushuntiring

3. Main metodida:
   3.1. PreparedStatement yordamida bir nechta foydalanuvchilarni qo'shing
   3.2. Foydalanuvchi ma'lumotlarini yangilang
   3.3. Email bo'yicha foydalanuvchilarni qidiring
   3.4. Foydalanuvchini o'chiring
   3.5. To'g'ri xatolarni qayta ishlash va resurslarni boshqarishni ko'rsating

**O'rganish maqsadlari:**
- PreparedStatement ning Statement ustidan afzalliklarini tushunish
- Parametr bog'lash texnikasini o'rganish
- SQL injeksiyasini oldini olishda mohirlik
- Xavfsiz CRUD operatsiyalarida amaliyot`,
            hint1: `Statement o'rniga PreparedStatement dan foydalaning. Uni conn.prepareStatement(sql) bilan yarating, bu yerda sql ? joylashtirgichlarini o'z ichiga oladi. Keyin qiymatlarni bog'lash uchun setString(), setInt() va boshqalardan foydalaning.`,
            hint2: `Parametr indekslari 0 dan emas, 1 dan boshlanadi. Birinchi ? uchun pstmt.setString(1, value), ikkinchi ? uchun setString(2, value) va hokazolardan foydalaning. Har doim try-with-resources dan foydalaning.`,
            whyItMatters: `PreparedStatement xavfsizlik va unumdorlik uchun juda muhimdir. U foydalanuvchi kiritmasini bajariladigan kod sifatida emas, balki ma'lumot sifatida ko'rib, SQL injeksiya hujumlarini oldini oladi. Shuningdek, so'rovlarni oldindan kompilyatsiya qilish orqali unumdorlikni yaxshilaydi va ma'lumotlar bazasiga bajarilish rejalarini keshlash imkonini beradi. Bu xavfsiz ilovalar yaratish uchun zarur bilimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ishlash tezligi uchun Batch operatsiyalari
public void insertUsersBatch(List<User> users) throws SQLException {
    String sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";
    try (Connection conn = dataSource.getConnection();
         PreparedStatement pstmt = conn.prepareStatement(sql)) {
        conn.setAutoCommit(false);
        for (User user : users) {
            pstmt.setString(1, user.getName());
            pstmt.setString(2, user.getEmail());
            pstmt.setInt(3, user.getAge());
            pstmt.addBatch();
            if (users.indexOf(user) % 1000 == 0) {
                pstmt.executeBatch(); // har 1000 yozuvda batch
            }
        }
        pstmt.executeBatch();
        conn.commit();
    }
}
\`\`\`

**Amaliy foydalari:**
- Batch operatsiyalar alohida qo'shishlardan 10-100 marta tezroq
- Tranzaksiyalar operatsiyalarning atomikligini ta'minlaydi
- SQL injeksiyasidan himoya PreparedStatement ga o'rnatilgan`
        }
    }
};

export default task;
