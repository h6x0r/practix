import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-test-lifecycle',
    title: 'Test Lifecycle',
    difficulty: 'medium',
    tags: ['java', 'testing', 'junit', 'lifecycle', 'setup'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master the **JUnit 5 test lifecycle** by implementing setup and teardown methods for a database connection test suite.

**Requirements:**
1. Create a DatabaseConnection class that simulates database operations:
   1.1. connect() - establishes connection
   1.2. disconnect() - closes connection
   1.3. executeQuery(String query) - executes a query
   1.4. isConnected() - checks connection status

2. Create DatabaseConnectionTest using lifecycle annotations:
   2.1. @BeforeAll - runs once before all tests (initialize shared resources)
   2.2. @AfterAll - runs once after all tests (cleanup shared resources)
   2.3. @BeforeEach - runs before each test (setup fresh state)
   2.4. @AfterEach - runs after each test (cleanup after each test)

3. Implement test methods to demonstrate:
   3.1. Connection is established before each test
   3.2. Connection is closed after each test
   3.3. Shared resources are initialized once
   3.4. Tests run in isolation

**Learning Goals:**
- Understand test lifecycle methods
- Learn when to use @BeforeAll vs @BeforeEach
- Practice proper test setup and teardown
- Ensure test isolation and independence`,
    initialCode: `import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DatabaseConnection {
    // TODO: Implement connection state and methods
}

class DatabaseConnectionTest {
    // TODO: Add static resources for @BeforeAll/@AfterAll

    // TODO: Add instance variables for @BeforeEach/@AfterEach

    // TODO: Implement @BeforeAll method

    // TODO: Implement @AfterAll method

    // TODO: Implement @BeforeEach method

    // TODO: Implement @AfterEach method

    // TODO: Write test methods
}`,
    solutionCode: `import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DatabaseConnection {
    private boolean connected = false;
    private String connectionString;

    public DatabaseConnection(String connectionString) {
        this.connectionString = connectionString;
    }

    // Establish database connection
    public void connect() {
        if (!connected) {
            System.out.println("Connecting to: " + connectionString);
            connected = true;
        }
    }

    // Close database connection
    public void disconnect() {
        if (connected) {
            System.out.println("Disconnecting from: " + connectionString);
            connected = false;
        }
    }

    // Execute a query
    public String executeQuery(String query) {
        if (!connected) {
            throw new IllegalStateException("Not connected to database");
        }
        System.out.println("Executing: " + query);
        return "Result for: " + query;
    }

    // Check connection status
    public boolean isConnected() {
        return connected;
    }
}

class DatabaseConnectionTest {
    // Static variable - shared across all tests
    private static String testDatabase;

    // Instance variable - fresh for each test
    private DatabaseConnection connection;

    // Runs once before all tests - initialize shared resources
    @BeforeAll
    static void setupAll() {
        System.out.println("@BeforeAll - Setting up test database");
        testDatabase = "jdbc:test://localhost:5432/testdb";
    }

    // Runs once after all tests - cleanup shared resources
    @AfterAll
    static void tearDownAll() {
        System.out.println("@AfterAll - Cleaning up test database");
        testDatabase = null;
    }

    // Runs before each test - setup fresh state
    @BeforeEach
    void setUp() {
        System.out.println("@BeforeEach - Creating new connection");
        connection = new DatabaseConnection(testDatabase);
        connection.connect();
    }

    // Runs after each test - cleanup
    @AfterEach
    void tearDown() {
        System.out.println("@AfterEach - Closing connection");
        if (connection != null && connection.isConnected()) {
            connection.disconnect();
        }
        connection = null;
    }

    // Test connection is established
    @Test
    void testConnectionIsEstablished() {
        assertTrue(connection.isConnected());
    }

    // Test query execution
    @Test
    void testExecuteQuery() {
        String result = connection.executeQuery("SELECT * FROM users");
        assertNotNull(result);
        assertTrue(result.contains("SELECT"));
    }

    // Test query without connection throws exception
    @Test
    void testExecuteQueryWithoutConnection() {
        connection.disconnect();
        assertThrows(IllegalStateException.class, () -> {
            connection.executeQuery("SELECT * FROM users");
        });
    }

    // Test multiple queries in same test
    @Test
    void testMultipleQueries() {
        String result1 = connection.executeQuery("SELECT * FROM users");
        String result2 = connection.executeQuery("SELECT * FROM orders");

        assertNotNull(result1);
        assertNotNull(result2);
        assertTrue(connection.isConnected());
    }

    // Test connection status
    @Test
    void testConnectionStatus() {
        assertTrue(connection.isConnected());

        connection.disconnect();
        assertFalse(connection.isConnected());

        connection.connect();
        assertTrue(connection.isConnected());
    }
}`,
    hint1: `Use @BeforeAll and @AfterAll for expensive operations that can be shared across tests (like setting up a test database URL). These methods must be static.`,
    hint2: `Use @BeforeEach and @AfterEach for operations that need to run for every test (like creating a fresh connection). This ensures each test starts with a clean state.`,
    whyItMatters: `Understanding test lifecycle is crucial for writing efficient and reliable tests. Proper setup and teardown ensures tests are independent and don't affect each other. Using @BeforeAll for expensive operations improves test performance, while @BeforeEach ensures each test has a fresh, isolated state.

**Production Pattern:**
\`\`\`java
class OrderServiceTest {
    private static DataSource dataSource;
    private OrderService orderService;

    @BeforeAll
    static void initDatabase() {
        dataSource = DatabasePool.create();
    }

    @BeforeEach
    void setUp() {
        orderService = new OrderService(dataSource);
        orderService.clearTestData();
    }
}
\`\`\`

**Practical Benefits:**
- Optimizing test performance by reusing resources
- Guaranteeing test isolation and predictable results`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.Before;
import org.junit.After;

// Test1: Test Before annotation
class Test1 {
    private int value;

    @Before
    public void setUp() {
        value = 10;
    }

    @Test
    public void test() {
        assertEquals(10, value);
    }
}

// Test2: Test After annotation
class Test2 {
    private StringBuilder log;

    @Before
    public void setUp() {
        log = new StringBuilder();
    }

    @Test
    public void test() {
        log.append("test");
        assertEquals("test", log.toString());
    }

    @After
    public void tearDown() {
        log = null;
    }
}

// Test3: Test multiple setup methods
class Test3 {
    private int counter;

    @Before
    public void setUp() {
        counter = 0;
    }

    @Test
    public void test() {
        counter++;
        assertEquals(1, counter);
    }
}

// Test4: Test state isolation between tests
class Test4 {
    private static int globalCounter = 0;
    private int localCounter;

    @Before
    public void setUp() {
        localCounter = 0;
        globalCounter = 0;
    }

    @Test
    public void test() {
        localCounter++;
        assertEquals(1, localCounter);
    }
}

// Test5: Test resource initialization
class Test5 {
    private String resource;

    @Before
    public void setUp() {
        resource = "initialized";
    }

    @Test
    public void test() {
        assertNotNull(resource);
        assertEquals("initialized", resource);
    }
}

// Test6: Test cleanup in After
class Test6 {
    private boolean cleaned;

    @Before
    public void setUp() {
        cleaned = false;
    }

    @Test
    public void test() {
        assertFalse(cleaned);
    }

    @After
    public void tearDown() {
        cleaned = true;
    }
}

// Test7: Test Before sets up fresh state
class Test7 {
    private java.util.List<String> list;

    @Before
    public void setUp() {
        list = new java.util.ArrayList<>();
    }

    @Test
    public void test() {
        assertTrue(list.isEmpty());
        list.add("item");
        assertEquals(1, list.size());
    }
}

// Test8: Test multiple assertions after setup
class Test8 {
    private int[] data;

    @Before
    public void setUp() {
        data = new int[]{1, 2, 3, 4, 5};
    }

    @Test
    public void test() {
        assertEquals(5, data.length);
        assertEquals(1, data[0]);
        assertEquals(5, data[4]);
    }
}

// Test9: Test object initialization
class Test9 {
    private String text;

    @Before
    public void setUp() {
        text = "Hello World";
    }

    @Test
    public void test() {
        assertTrue(text.contains("Hello"));
        assertTrue(text.contains("World"));
    }
}

// Test10: Test complete lifecycle
class Test10 {
    private int setupCount;
    private int teardownCount;

    @Before
    public void setUp() {
        setupCount = 1;
        teardownCount = 0;
    }

    @Test
    public void test() {
        assertEquals(1, setupCount);
        assertEquals(0, teardownCount);
    }

    @After
    public void tearDown() {
        teardownCount = 1;
    }
}
`,
    translations: {
        ru: {
            title: 'Жизненный Цикл Тестов',
            solutionCode: `import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DatabaseConnection {
    private boolean connected = false;
    private String connectionString;

    public DatabaseConnection(String connectionString) {
        this.connectionString = connectionString;
    }

    // Устанавливаем соединение с базой данных
    public void connect() {
        if (!connected) {
            System.out.println("Подключение к: " + connectionString);
            connected = true;
        }
    }

    // Закрываем соединение с базой данных
    public void disconnect() {
        if (connected) {
            System.out.println("Отключение от: " + connectionString);
            connected = false;
        }
    }

    // Выполняем запрос
    public String executeQuery(String query) {
        if (!connected) {
            throw new IllegalStateException("Нет подключения к базе данных");
        }
        System.out.println("Выполнение: " + query);
        return "Результат для: " + query;
    }

    // Проверяем статус соединения
    public boolean isConnected() {
        return connected;
    }
}

class DatabaseConnectionTest {
    // Статическая переменная - общая для всех тестов
    private static String testDatabase;

    // Переменная экземпляра - новая для каждого теста
    private DatabaseConnection connection;

    // Запускается один раз перед всеми тестами - инициализация общих ресурсов
    @BeforeAll
    static void setupAll() {
        System.out.println("@BeforeAll - Настройка тестовой базы данных");
        testDatabase = "jdbc:test://localhost:5432/testdb";
    }

    // Запускается один раз после всех тестов - очистка общих ресурсов
    @AfterAll
    static void tearDownAll() {
        System.out.println("@AfterAll - Очистка тестовой базы данных");
        testDatabase = null;
    }

    // Запускается перед каждым тестом - настройка свежего состояния
    @BeforeEach
    void setUp() {
        System.out.println("@BeforeEach - Создание нового соединения");
        connection = new DatabaseConnection(testDatabase);
        connection.connect();
    }

    // Запускается после каждого теста - очистка
    @AfterEach
    void tearDown() {
        System.out.println("@AfterEach - Закрытие соединения");
        if (connection != null && connection.isConnected()) {
            connection.disconnect();
        }
        connection = null;
    }

    // Тест установки соединения
    @Test
    void testConnectionIsEstablished() {
        assertTrue(connection.isConnected());
    }

    // Тест выполнения запроса
    @Test
    void testExecuteQuery() {
        String result = connection.executeQuery("SELECT * FROM users");
        assertNotNull(result);
        assertTrue(result.contains("SELECT"));
    }

    // Тест запроса без соединения выбрасывает исключение
    @Test
    void testExecuteQueryWithoutConnection() {
        connection.disconnect();
        assertThrows(IllegalStateException.class, () -> {
            connection.executeQuery("SELECT * FROM users");
        });
    }

    // Тест множественных запросов в одном тесте
    @Test
    void testMultipleQueries() {
        String result1 = connection.executeQuery("SELECT * FROM users");
        String result2 = connection.executeQuery("SELECT * FROM orders");

        assertNotNull(result1);
        assertNotNull(result2);
        assertTrue(connection.isConnected());
    }

    // Тест статуса соединения
    @Test
    void testConnectionStatus() {
        assertTrue(connection.isConnected());

        connection.disconnect();
        assertFalse(connection.isConnected());

        connection.connect();
        assertTrue(connection.isConnected());
    }
}`,
            description: `Освойте **жизненный цикл тестов JUnit 5**, реализовав методы настройки и очистки для набора тестов подключения к базе данных.

**Требования:**
1. Создайте класс DatabaseConnection, имитирующий операции с базой данных:
   1.1. connect() - устанавливает соединение
   1.2. disconnect() - закрывает соединение
   1.3. executeQuery(String query) - выполняет запрос
   1.4. isConnected() - проверяет статус соединения

2. Создайте DatabaseConnectionTest используя аннотации жизненного цикла:
   2.1. @BeforeAll - запускается один раз перед всеми тестами (инициализация общих ресурсов)
   2.2. @AfterAll - запускается один раз после всех тестов (очистка общих ресурсов)
   2.3. @BeforeEach - запускается перед каждым тестом (настройка свежего состояния)
   2.4. @AfterEach - запускается после каждого теста (очистка после каждого теста)

3. Реализуйте тестовые методы, демонстрирующие:
   3.1. Соединение устанавливается перед каждым тестом
   3.2. Соединение закрывается после каждого теста
   3.3. Общие ресурсы инициализируются один раз
   3.4. Тесты выполняются изолированно

**Цели обучения:**
- Понять методы жизненного цикла тестов
- Научиться, когда использовать @BeforeAll vs @BeforeEach
- Практиковаться в правильной настройке и очистке тестов
- Обеспечить изоляцию и независимость тестов`,
            hint1: `Используйте @BeforeAll и @AfterAll для дорогостоящих операций, которые могут быть общими для всех тестов (например, настройка URL тестовой базы данных). Эти методы должны быть статическими.`,
            hint2: `Используйте @BeforeEach и @AfterEach для операций, которые должны выполняться для каждого теста (например, создание нового соединения). Это гарантирует, что каждый тест начинается с чистого состояния.`,
            whyItMatters: `Понимание жизненного цикла тестов имеет решающее значение для написания эффективных и надежных тестов. Правильная настройка и очистка обеспечивает независимость тестов и отсутствие влияния друг на друга. Использование @BeforeAll для дорогостоящих операций повышает производительность тестов, в то время как @BeforeEach гарантирует, что каждый тест имеет свежее, изолированное состояние.

**Продакшен паттерн:**
\`\`\`java
class OrderServiceTest {
    private static DataSource dataSource;
    private OrderService orderService;

    @BeforeAll
    static void initDatabase() {
        dataSource = DatabasePool.create();
    }

    @BeforeEach
    void setUp() {
        orderService = new OrderService(dataSource);
        orderService.clearTestData();
    }
}
\`\`\`

**Практические преимущества:**
- Оптимизация производительности тестов за счет переиспользования ресурсов
- Гарантия изоляции тестов и предсказуемых результатов`
        },
        uz: {
            title: 'Test Hayot Tsikli',
            solutionCode: `import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class DatabaseConnection {
    private boolean connected = false;
    private String connectionString;

    public DatabaseConnection(String connectionString) {
        this.connectionString = connectionString;
    }

    // Ma'lumotlar bazasiga ulanish
    public void connect() {
        if (!connected) {
            System.out.println("Ulanish: " + connectionString);
            connected = true;
        }
    }

    // Ma'lumotlar bazasidan uzilish
    public void disconnect() {
        if (connected) {
            System.out.println("Uzilish: " + connectionString);
            connected = false;
        }
    }

    // So'rov bajarish
    public String executeQuery(String query) {
        if (!connected) {
            throw new IllegalStateException("Ma'lumotlar bazasiga ulanmagan");
        }
        System.out.println("Bajarilmoqda: " + query);
        return "Natija: " + query;
    }

    // Ulanish holatini tekshirish
    public boolean isConnected() {
        return connected;
    }
}

class DatabaseConnectionTest {
    // Statik o'zgaruvchi - barcha testlar uchun umumiy
    private static String testDatabase;

    // Nusxa o'zgaruvchisi - har bir test uchun yangi
    private DatabaseConnection connection;

    // Barcha testlardan oldin bir marta ishga tushadi - umumiy resurslarni ishga tushirish
    @BeforeAll
    static void setupAll() {
        System.out.println("@BeforeAll - Test ma'lumotlar bazasini sozlash");
        testDatabase = "jdbc:test://localhost:5432/testdb";
    }

    // Barcha testlardan keyin bir marta ishga tushadi - umumiy resurslarni tozalash
    @AfterAll
    static void tearDownAll() {
        System.out.println("@AfterAll - Test ma'lumotlar bazasini tozalash");
        testDatabase = null;
    }

    // Har bir testdan oldin ishga tushadi - yangi holatni sozlash
    @BeforeEach
    void setUp() {
        System.out.println("@BeforeEach - Yangi ulanish yaratish");
        connection = new DatabaseConnection(testDatabase);
        connection.connect();
    }

    // Har bir testdan keyin ishga tushadi - tozalash
    @AfterEach
    void tearDown() {
        System.out.println("@AfterEach - Ulanishni yopish");
        if (connection != null && connection.isConnected()) {
            connection.disconnect();
        }
        connection = null;
    }

    // Ulanish o'rnatilgan testi
    @Test
    void testConnectionIsEstablished() {
        assertTrue(connection.isConnected());
    }

    // So'rov bajarish testi
    @Test
    void testExecuteQuery() {
        String result = connection.executeQuery("SELECT * FROM users");
        assertNotNull(result);
        assertTrue(result.contains("SELECT"));
    }

    // Ulanishsiz so'rov istisno chiqaradi testi
    @Test
    void testExecuteQueryWithoutConnection() {
        connection.disconnect();
        assertThrows(IllegalStateException.class, () -> {
            connection.executeQuery("SELECT * FROM users");
        });
    }

    // Bir testda bir nechta so'rovlar testi
    @Test
    void testMultipleQueries() {
        String result1 = connection.executeQuery("SELECT * FROM users");
        String result2 = connection.executeQuery("SELECT * FROM orders");

        assertNotNull(result1);
        assertNotNull(result2);
        assertTrue(connection.isConnected());
    }

    // Ulanish holati testi
    @Test
    void testConnectionStatus() {
        assertTrue(connection.isConnected());

        connection.disconnect();
        assertFalse(connection.isConnected());

        connection.connect();
        assertTrue(connection.isConnected());
    }
}`,
            description: `Ma'lumotlar bazasi ulanishi test to'plami uchun sozlash va tozalash metodlarini amalga oshirish orqali **JUnit 5 test hayot tsiklini** o'rganing.

**Talablar:**
1. Ma'lumotlar bazasi operatsiyalarini taqlid qiladigan DatabaseConnection sinfini yarating:
   1.1. connect() - ulanish o'rnatadi
   1.2. disconnect() - ulanishni yopadi
   1.3. executeQuery(String query) - so'rov bajaradi
   1.4. isConnected() - ulanish holatini tekshiradi

2. Hayot tsikli annotatsiyalaridan foydalangan holda DatabaseConnectionTest yarating:
   2.1. @BeforeAll - barcha testlardan oldin bir marta ishga tushadi (umumiy resurslarni ishga tushirish)
   2.2. @AfterAll - barcha testlardan keyin bir marta ishga tushadi (umumiy resurslarni tozalash)
   2.3. @BeforeEach - har bir testdan oldin ishga tushadi (yangi holatni sozlash)
   2.4. @AfterEach - har bir testdan keyin ishga tushadi (har bir testdan keyin tozalash)

3. Quyidagilarni ko'rsatadigan test metodlarini amalga oshiring:
   3.1. Har bir testdan oldin ulanish o'rnatiladi
   3.2. Har bir testdan keyin ulanish yopiladi
   3.3. Umumiy resurslar bir marta ishga tushiriladi
   3.4. Testlar alohida bajariladi

**O'rganish maqsadlari:**
- Test hayot tsikli metodlarini tushunish
- @BeforeAll va @BeforeEach dan qachon foydalanishni o'rganish
- To'g'ri test sozlash va tozalashda amaliyot
- Test ajratilishi va mustaqilligini ta'minlash`,
            hint1: `Barcha testlarda umumiy bo'lishi mumkin bo'lgan qimmat operatsiyalar uchun @BeforeAll va @AfterAll dan foydalaning (masalan, test ma'lumotlar bazasi URL manzilini sozlash). Bu metodlar statik bo'lishi kerak.`,
            hint2: `Har bir test uchun bajarilishi kerak bo'lgan operatsiyalar uchun @BeforeEach va @AfterEach dan foydalaning (masalan, yangi ulanish yaratish). Bu har bir testning toza holatdan boshlanishini ta'minlaydi.`,
            whyItMatters: `Test hayot tsiklini tushunish samarali va ishonchli testlar yozish uchun juda muhim. To'g'ri sozlash va tozalash testlarning mustaqilligini va bir-biriga ta'sir qilmasligini ta'minlaydi. Qimmat operatsiyalar uchun @BeforeAll dan foydalanish test samaradorligini oshiradi, @BeforeEach esa har bir testning yangi, ajratilgan holatga ega ekanligini kafolatlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
class OrderServiceTest {
    private static DataSource dataSource;
    private OrderService orderService;

    @BeforeAll
    static void initDatabase() {
        dataSource = DatabasePool.create();
    }

    @BeforeEach
    void setUp() {
        orderService = new OrderService(dataSource);
        orderService.clearTestData();
    }
}
\`\`\`

**Amaliy foydalari:**
- Resurslarni qayta ishlatish orqali test samaradorligini optimallashtirish
- Testlarning ajratilishi va bashorat qilinadigan natijalarni kafolatlash`
        }
    }
};

export default task;
