import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-generic-patterns',
    title: 'Generic Design Patterns',
    difficulty: 'medium',
    tags: ['java', 'generics', 'design-patterns', 'factory', 'repository'],
    estimatedTime: '40m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master common design patterns using Java generics.

**Requirements:**
1. Implement a Generic Factory pattern with Factory<T>
2. Create a Generic Repository pattern for data access
3. Implement a Builder pattern with generics for type safety
4. Create a type-safe heterogeneous container (map with different value types)
5. Implement a Generic Singleton pattern
6. Demonstrate all patterns with practical examples

Generic design patterns leverage type parameters to create reusable, type-safe implementations of common software patterns.`,
    initialCode: `import java.util.*;
import java.util.function.Supplier;

public class GenericPatterns {
    // 1. Generic Factory pattern
    // - Create Factory<T> interface
    // - Implement concrete factories

    // 2. Generic Repository pattern
    // - Create Repository<T, ID> interface
    // - Implement InMemoryRepository

    // 3. Generic Builder pattern
    // - Create type-safe builder

    // 4. Type-safe heterogeneous container
    // - Store different types in same container

    // 5. Generic Singleton
    // - Thread-safe generic singleton

    public static void main(String[] args) {
        // Demonstrate all patterns
    }
}`,
    solutionCode: `import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

public class GenericPatterns {

    // PATTERN 1: Generic Factory
    interface Factory<T> {
        T create();
    }

    static class User {
        private final String name;

        public User(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{name='" + name + "'}";
        }
    }

    static class Product {
        private final String id;
        private final double price;

        public Product(String id, double price) {
            this.id = id;
            this.price = price;
        }

        @Override
        public String toString() {
            return "Product{id='" + id + "', price=" + price + "}";
        }
    }

    // Concrete factories
    static class UserFactory implements Factory<User> {
        private int counter = 0;

        @Override
        public User create() {
            return new User("User" + (++counter));
        }
    }

    static class ProductFactory implements Factory<Product> {
        private int counter = 0;

        @Override
        public Product create() {
            return new Product("PROD" + (++counter), 99.99);
        }
    }

    // PATTERN 2: Generic Repository
    interface Repository<T, ID> {
        void save(T entity);
        Optional<T> findById(ID id);
        List<T> findAll();
        void delete(ID id);
        boolean exists(ID id);
    }

    static class InMemoryRepository<T, ID> implements Repository<T, ID> {
        private final Map<ID, T> storage = new HashMap<>();

        @Override
        public void save(T entity) {
            // Assume entity has getId method via reflection or interface
            // For simplicity, we'll use a different approach
        }

        public void save(ID id, T entity) {
            storage.put(id, entity);
        }

        @Override
        public Optional<T> findById(ID id) {
            return Optional.ofNullable(storage.get(id));
        }

        @Override
        public List<T> findAll() {
            return new ArrayList<>(storage.values());
        }

        @Override
        public void delete(ID id) {
            storage.remove(id);
        }

        @Override
        public boolean exists(ID id) {
            return storage.containsKey(id);
        }

        public int size() {
            return storage.size();
        }
    }

    // PATTERN 3: Generic Builder
    static class Query<T> {
        private final Class<T> type;
        private String filter;
        private Integer limit;
        private String orderBy;

        private Query(Class<T> type) {
            this.type = type;
        }

        public static <T> Builder<T> builder(Class<T> type) {
            return new Builder<>(type);
        }

        static class Builder<T> {
            private final Query<T> query;

            private Builder(Class<T> type) {
                this.query = new Query<>(type);
            }

            public Builder<T> filter(String filter) {
                query.filter = filter;
                return this;
            }

            public Builder<T> limit(int limit) {
                query.limit = limit;
                return this;
            }

            public Builder<T> orderBy(String field) {
                query.orderBy = field;
                return this;
            }

            public Query<T> build() {
                return query;
            }
        }

        @Override
        public String toString() {
            return "Query{type=" + type.getSimpleName() +
                    ", filter='" + filter + "'" +
                    ", limit=" + limit +
                    ", orderBy='" + orderBy + "'}";
        }
    }

    // PATTERN 4: Type-safe Heterogeneous Container
    static class TypeSafeMap {
        private final Map<Class<?>, Object> map = new HashMap<>();

        // Type-safe put
        public <T> void put(Class<T> type, T instance) {
            map.put(type, type.cast(instance));
        }

        // Type-safe get
        public <T> T get(Class<T> type) {
            return type.cast(map.get(type));
        }

        public <T> boolean contains(Class<T> type) {
            return map.containsKey(type);
        }

        public int size() {
            return map.size();
        }
    }

    // PATTERN 5: Generic Singleton
    static class Singleton<T> {
        private static final Map<Class<?>, Object> instances = new ConcurrentHashMap<>();

        @SuppressWarnings("unchecked")
        public static <T> T getInstance(Class<T> type, Supplier<T> supplier) {
            return (T) instances.computeIfAbsent(type, k -> supplier.get());
        }

        public static <T> boolean hasInstance(Class<T> type) {
            return instances.containsKey(type);
        }
    }

    // Example service for singleton
    static class DatabaseService {
        private final String connectionString;

        public DatabaseService() {
            this.connectionString = "jdbc:db://localhost:5432";
            System.out.println("DatabaseService created");
        }

        public String getConnectionString() {
            return connectionString;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Generic Factory Pattern ===");
        Factory<User> userFactory = new UserFactory();
        Factory<Product> productFactory = new ProductFactory();

        User user1 = userFactory.create();
        User user2 = userFactory.create();
        Product product1 = productFactory.create();

        System.out.println("Created: " + user1);
        System.out.println("Created: " + user2);
        System.out.println("Created: " + product1);

        System.out.println("\\n=== Generic Repository Pattern ===");
        InMemoryRepository<User, Integer> userRepo = new InMemoryRepository<>();
        userRepo.save(1, user1);
        userRepo.save(2, user2);

        System.out.println("User count: " + userRepo.size());
        System.out.println("Find user 1: " + userRepo.findById(1));
        System.out.println("All users: " + userRepo.findAll());
        System.out.println("User 3 exists: " + userRepo.exists(3));

        System.out.println("\\n=== Generic Builder Pattern ===");
        Query<User> userQuery = Query.builder(User.class)
                .filter("age > 18")
                .limit(10)
                .orderBy("name")
                .build();

        Query<Product> productQuery = Query.builder(Product.class)
                .filter("price < 100")
                .limit(5)
                .build();

        System.out.println("User query: " + userQuery);
        System.out.println("Product query: " + productQuery);

        System.out.println("\\n=== Type-safe Heterogeneous Container ===");
        TypeSafeMap container = new TypeSafeMap();

        // Store different types in same container
        container.put(String.class, "Hello Generics");
        container.put(Integer.class, 42);
        container.put(Double.class, 3.14159);
        container.put(User.class, user1);

        // Type-safe retrieval
        String str = container.get(String.class);
        Integer num = container.get(Integer.class);
        Double dbl = container.get(Double.class);
        User usr = container.get(User.class);

        System.out.println("String: " + str);
        System.out.println("Integer: " + num);
        System.out.println("Double: " + dbl);
        System.out.println("User: " + usr);
        System.out.println("Container size: " + container.size());

        System.out.println("\\n=== Generic Singleton Pattern ===");
        // First call creates instance
        DatabaseService db1 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        // Second call returns same instance
        DatabaseService db2 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        System.out.println("db1 connection: " + db1.getConnectionString());
        System.out.println("db2 connection: " + db2.getConnectionString());
        System.out.println("Same instance? " + (db1 == db2));
        System.out.println("Has instance? " + Singleton.hasInstance(DatabaseService.class));

        System.out.println("\\n=== Summary ===");
        System.out.println("1. Factory<T>: Type-safe object creation");
        System.out.println("2. Repository<T, ID>: Generic data access layer");
        System.out.println("3. Builder<T>: Fluent type-safe construction");
        System.out.println("4. TypeSafeMap: Heterogeneous container with type safety");
        System.out.println("5. Singleton<T>: Generic singleton with type preservation");
    }
}`,
    hint1: `Generic patterns combine design patterns with type parameters. Factory<T> creates type T, Repository<T, ID> manages type T with identifier ID.`,
    hint2: `Type-safe heterogeneous containers use Class<T> as keys to maintain type safety while storing different types. Builder pattern with generics ensures type safety throughout the building process.`,
    whyItMatters: `Generic design patterns are essential for creating reusable, maintainable libraries and frameworks.

**Production Pattern:**
\`\`\`java
// Generic Repository pattern
interface Repository<T, ID> {
    void save(T entity);
    Optional<T> findById(ID id);
    List<T> findAll();
}

// Generic Factory pattern
interface Factory<T> {
    T create();
}

// Usage
Repository<User, Integer> userRepo = new InMemoryRepository<>();
userRepo.save(new User("Alice"));
Optional<User> user = userRepo.findById(1);
\`\`\`

**Practical Benefits:**
- Spring Framework uses these patterns
- Hibernate, JPA use Repository<T, ID>
- Type-safe object creation
- Critical for senior developers`,
    order: 6,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;

// Test1: Verify UserFactory creates User objects
class Test1 {
    @Test
    public void test() {
        GenericPatterns.UserFactory factory = new GenericPatterns.UserFactory();
        GenericPatterns.User user = factory.create();
        assertNotNull(user);
    }
}

// Test2: Verify ProductFactory creates Product objects
class Test2 {
    @Test
    public void test() {
        GenericPatterns.ProductFactory factory = new GenericPatterns.ProductFactory();
        GenericPatterns.Product product = factory.create();
        assertNotNull(product);
    }
}

// Test3: Verify InMemoryRepository save and findById
class Test3 {
    @Test
    public void test() {
        GenericPatterns.InMemoryRepository<String, Integer> repo = new GenericPatterns.InMemoryRepository<>();
        repo.save(1, "Test");
        Optional<String> result = repo.findById(1);
        assertTrue(result.isPresent());
        assertEquals("Test", result.get());
    }
}

// Test4: Verify InMemoryRepository exists method
class Test4 {
    @Test
    public void test() {
        GenericPatterns.InMemoryRepository<String, Integer> repo = new GenericPatterns.InMemoryRepository<>();
        repo.save(1, "Test");
        assertTrue(repo.exists(1));
        assertFalse(repo.exists(2));
    }
}

// Test5: Verify InMemoryRepository findAll method
class Test5 {
    @Test
    public void test() {
        GenericPatterns.InMemoryRepository<String, Integer> repo = new GenericPatterns.InMemoryRepository<>();
        repo.save(1, "A");
        repo.save(2, "B");
        assertEquals(2, repo.findAll().size());
    }
}

// Test6: Verify Query Builder pattern
class Test6 {
    @Test
    public void test() {
        GenericPatterns.Query<String> query = GenericPatterns.Query.builder(String.class)
            .filter("test")
            .limit(10)
            .build();
        assertNotNull(query);
    }
}

// Test7: Verify TypeSafeMap put and get
class Test7 {
    @Test
    public void test() {
        GenericPatterns.TypeSafeMap map = new GenericPatterns.TypeSafeMap();
        map.put(String.class, "Hello");
        String result = map.get(String.class);
        assertEquals("Hello", result);
    }
}

// Test8: Verify TypeSafeMap with multiple types
class Test8 {
    @Test
    public void test() {
        GenericPatterns.TypeSafeMap map = new GenericPatterns.TypeSafeMap();
        map.put(String.class, "Hello");
        map.put(Integer.class, 42);
        assertEquals("Hello", map.get(String.class));
        assertEquals(Integer.valueOf(42), map.get(Integer.class));
    }
}

// Test9: Verify Singleton pattern returns same instance
class Test9 {
    @Test
    public void test() {
        GenericPatterns.DatabaseService db1 = GenericPatterns.Singleton.getInstance(
            GenericPatterns.DatabaseService.class,
            GenericPatterns.DatabaseService::new
        );
        GenericPatterns.DatabaseService db2 = GenericPatterns.Singleton.getInstance(
            GenericPatterns.DatabaseService.class,
            GenericPatterns.DatabaseService::new
        );
        assertSame(db1, db2);
    }
}

// Test10: Verify InMemoryRepository delete method
class Test10 {
    @Test
    public void test() {
        GenericPatterns.InMemoryRepository<String, Integer> repo = new GenericPatterns.InMemoryRepository<>();
        repo.save(1, "Test");
        repo.delete(1);
        assertFalse(repo.exists(1));
    }
}`,
    translations: {
        ru: {
            title: 'Шаблоны проектирования с обобщениями',
            solutionCode: `import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

public class GenericPatterns {

    // ШАБЛОН 1: Обобщенная фабрика
    interface Factory<T> {
        T create();
    }

    static class User {
        private final String name;

        public User(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{name='" + name + "'}";
        }
    }

    static class Product {
        private final String id;
        private final double price;

        public Product(String id, double price) {
            this.id = id;
            this.price = price;
        }

        @Override
        public String toString() {
            return "Product{id='" + id + "', price=" + price + "}";
        }
    }

    // Конкретные фабрики
    static class UserFactory implements Factory<User> {
        private int counter = 0;

        @Override
        public User create() {
            return new User("User" + (++counter));
        }
    }

    static class ProductFactory implements Factory<Product> {
        private int counter = 0;

        @Override
        public Product create() {
            return new Product("PROD" + (++counter), 99.99);
        }
    }

    // ШАБЛОН 2: Обобщенный репозиторий
    interface Repository<T, ID> {
        void save(T entity);
        Optional<T> findById(ID id);
        List<T> findAll();
        void delete(ID id);
        boolean exists(ID id);
    }

    static class InMemoryRepository<T, ID> implements Repository<T, ID> {
        private final Map<ID, T> storage = new HashMap<>();

        @Override
        public void save(T entity) {
            // Предполагаем, что entity имеет метод getId через рефлексию или интерфейс
            // Для простоты используем другой подход
        }

        public void save(ID id, T entity) {
            storage.put(id, entity);
        }

        @Override
        public Optional<T> findById(ID id) {
            return Optional.ofNullable(storage.get(id));
        }

        @Override
        public List<T> findAll() {
            return new ArrayList<>(storage.values());
        }

        @Override
        public void delete(ID id) {
            storage.remove(id);
        }

        @Override
        public boolean exists(ID id) {
            return storage.containsKey(id);
        }

        public int size() {
            return storage.size();
        }
    }

    // ШАБЛОН 3: Обобщенный строитель
    static class Query<T> {
        private final Class<T> type;
        private String filter;
        private Integer limit;
        private String orderBy;

        private Query(Class<T> type) {
            this.type = type;
        }

        public static <T> Builder<T> builder(Class<T> type) {
            return new Builder<>(type);
        }

        static class Builder<T> {
            private final Query<T> query;

            private Builder(Class<T> type) {
                this.query = new Query<>(type);
            }

            public Builder<T> filter(String filter) {
                query.filter = filter;
                return this;
            }

            public Builder<T> limit(int limit) {
                query.limit = limit;
                return this;
            }

            public Builder<T> orderBy(String field) {
                query.orderBy = field;
                return this;
            }

            public Query<T> build() {
                return query;
            }
        }

        @Override
        public String toString() {
            return "Query{type=" + type.getSimpleName() +
                    ", filter='" + filter + "'" +
                    ", limit=" + limit +
                    ", orderBy='" + orderBy + "'}";
        }
    }

    // ШАБЛОН 4: Типобезопасный гетерогенный контейнер
    static class TypeSafeMap {
        private final Map<Class<?>, Object> map = new HashMap<>();

        // Типобезопасное добавление
        public <T> void put(Class<T> type, T instance) {
            map.put(type, type.cast(instance));
        }

        // Типобезопасное получение
        public <T> T get(Class<T> type) {
            return type.cast(map.get(type));
        }

        public <T> boolean contains(Class<T> type) {
            return map.containsKey(type);
        }

        public int size() {
            return map.size();
        }
    }

    // ШАБЛОН 5: Обобщенный одиночка
    static class Singleton<T> {
        private static final Map<Class<?>, Object> instances = new ConcurrentHashMap<>();

        @SuppressWarnings("unchecked")
        public static <T> T getInstance(Class<T> type, Supplier<T> supplier) {
            return (T) instances.computeIfAbsent(type, k -> supplier.get());
        }

        public static <T> boolean hasInstance(Class<T> type) {
            return instances.containsKey(type);
        }
    }

    // Пример сервиса для одиночки
    static class DatabaseService {
        private final String connectionString;

        public DatabaseService() {
            this.connectionString = "jdbc:db://localhost:5432";
            System.out.println("DatabaseService создан");
        }

        public String getConnectionString() {
            return connectionString;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Шаблон Обобщенная фабрика ===");
        Factory<User> userFactory = new UserFactory();
        Factory<Product> productFactory = new ProductFactory();

        User user1 = userFactory.create();
        User user2 = userFactory.create();
        Product product1 = productFactory.create();

        System.out.println("Создан: " + user1);
        System.out.println("Создан: " + user2);
        System.out.println("Создан: " + product1);

        System.out.println("\\n=== Шаблон Обобщенный репозиторий ===");
        InMemoryRepository<User, Integer> userRepo = new InMemoryRepository<>();
        userRepo.save(1, user1);
        userRepo.save(2, user2);

        System.out.println("Количество пользователей: " + userRepo.size());
        System.out.println("Найти пользователя 1: " + userRepo.findById(1));
        System.out.println("Все пользователи: " + userRepo.findAll());
        System.out.println("Пользователь 3 существует: " + userRepo.exists(3));

        System.out.println("\\n=== Шаблон Обобщенный строитель ===");
        Query<User> userQuery = Query.builder(User.class)
                .filter("age > 18")
                .limit(10)
                .orderBy("name")
                .build();

        Query<Product> productQuery = Query.builder(Product.class)
                .filter("price < 100")
                .limit(5)
                .build();

        System.out.println("Запрос пользователей: " + userQuery);
        System.out.println("Запрос продуктов: " + productQuery);

        System.out.println("\\n=== Типобезопасный гетерогенный контейнер ===");
        TypeSafeMap container = new TypeSafeMap();

        // Храним разные типы в одном контейнере
        container.put(String.class, "Hello Generics");
        container.put(Integer.class, 42);
        container.put(Double.class, 3.14159);
        container.put(User.class, user1);

        // Типобезопасное извлечение
        String str = container.get(String.class);
        Integer num = container.get(Integer.class);
        Double dbl = container.get(Double.class);
        User usr = container.get(User.class);

        System.out.println("String: " + str);
        System.out.println("Integer: " + num);
        System.out.println("Double: " + dbl);
        System.out.println("User: " + usr);
        System.out.println("Размер контейнера: " + container.size());

        System.out.println("\\n=== Шаблон Обобщенный одиночка ===");
        // Первый вызов создает экземпляр
        DatabaseService db1 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        // Второй вызов возвращает тот же экземпляр
        DatabaseService db2 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        System.out.println("Соединение db1: " + db1.getConnectionString());
        System.out.println("Соединение db2: " + db2.getConnectionString());
        System.out.println("Тот же экземпляр? " + (db1 == db2));
        System.out.println("Есть экземпляр? " + Singleton.hasInstance(DatabaseService.class));

        System.out.println("\\n=== Резюме ===");
        System.out.println("1. Factory<T>: Типобезопасное создание объектов");
        System.out.println("2. Repository<T, ID>: Обобщенный слой доступа к данным");
        System.out.println("3. Builder<T>: Гибкая типобезопасная конструкция");
        System.out.println("4. TypeSafeMap: Гетерогенный контейнер с безопасностью типов");
        System.out.println("5. Singleton<T>: Обобщенный одиночка с сохранением типа");
    }
}`,
            description: `Освойте распространенные шаблоны проектирования с использованием обобщений Java.

**Требования:**
1. Реализуйте шаблон Обобщенная фабрика с Factory<T>
2. Создайте шаблон Обобщенный репозиторий для доступа к данным
3. Реализуйте шаблон Строитель с обобщениями для безопасности типов
4. Создайте типобезопасный гетерогенный контейнер (map с разными типами значений)
5. Реализуйте шаблон Обобщенный одиночка
6. Продемонстрируйте все шаблоны с практическими примерами

Обобщенные шаблоны проектирования используют параметры типа для создания повторно используемых, типобезопасных реализаций распространенных программных шаблонов.`,
            hint1: `Обобщенные шаблоны сочетают шаблоны проектирования с параметрами типа. Factory<T> создает тип T, Repository<T, ID> управляет типом T с идентификатором ID.`,
            hint2: `Типобезопасные гетерогенные контейнеры используют Class<T> в качестве ключей для поддержания безопасности типов при хранении разных типов. Шаблон Строитель с обобщениями обеспечивает безопасность типов на протяжении всего процесса построения.`,
            whyItMatters: `Обобщенные шаблоны проектирования необходимы для создания повторно используемых, поддерживаемых библиотек и фреймворков.

**Продакшен паттерн:**
\`\`\`java
// Generic Repository паттерн
interface Repository<T, ID> {
    void save(T entity);
    Optional<T> findById(ID id);
    List<T> findAll();
}

// Generic Factory паттерн
interface Factory<T> {
    T create();
}

// Использование
Repository<User, Integer> userRepo = new InMemoryRepository<>();
userRepo.save(new User("Alice"));
Optional<User> user = userRepo.findById(1);
\`\`\`

**Практические преимущества:**
- Spring Framework использует эти паттерны
- Hibernate, JPA используют Repository<T, ID>
- Типобезопасное создание объектов
- Критично для senior разработчиков`
        },
        uz: {
            title: 'Umumiy dizayn naqshlari',
            solutionCode: `import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

public class GenericPatterns {

    // NAQSH 1: Umumiy fabrika
    interface Factory<T> {
        T create();
    }

    static class User {
        private final String name;

        public User(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{name='" + name + "'}";
        }
    }

    static class Product {
        private final String id;
        private final double price;

        public Product(String id, double price) {
            this.id = id;
            this.price = price;
        }

        @Override
        public String toString() {
            return "Product{id='" + id + "', price=" + price + "}";
        }
    }

    // Aniq fabrikalar
    static class UserFactory implements Factory<User> {
        private int counter = 0;

        @Override
        public User create() {
            return new User("User" + (++counter));
        }
    }

    static class ProductFactory implements Factory<Product> {
        private int counter = 0;

        @Override
        public Product create() {
            return new Product("PROD" + (++counter), 99.99);
        }
    }

    // NAQSH 2: Umumiy repository
    interface Repository<T, ID> {
        void save(T entity);
        Optional<T> findById(ID id);
        List<T> findAll();
        void delete(ID id);
        boolean exists(ID id);
    }

    static class InMemoryRepository<T, ID> implements Repository<T, ID> {
        private final Map<ID, T> storage = new HashMap<>();

        @Override
        public void save(T entity) {
            // Entity reflection yoki interfeys orqali getId metodiga ega deb faraz qilamiz
            // Soddalik uchun boshqa yondashuvdan foydalanamiz
        }

        public void save(ID id, T entity) {
            storage.put(id, entity);
        }

        @Override
        public Optional<T> findById(ID id) {
            return Optional.ofNullable(storage.get(id));
        }

        @Override
        public List<T> findAll() {
            return new ArrayList<>(storage.values());
        }

        @Override
        public void delete(ID id) {
            storage.remove(id);
        }

        @Override
        public boolean exists(ID id) {
            return storage.containsKey(id);
        }

        public int size() {
            return storage.size();
        }
    }

    // NAQSH 3: Umumiy builder
    static class Query<T> {
        private final Class<T> type;
        private String filter;
        private Integer limit;
        private String orderBy;

        private Query(Class<T> type) {
            this.type = type;
        }

        public static <T> Builder<T> builder(Class<T> type) {
            return new Builder<>(type);
        }

        static class Builder<T> {
            private final Query<T> query;

            private Builder(Class<T> type) {
                this.query = new Query<>(type);
            }

            public Builder<T> filter(String filter) {
                query.filter = filter;
                return this;
            }

            public Builder<T> limit(int limit) {
                query.limit = limit;
                return this;
            }

            public Builder<T> orderBy(String field) {
                query.orderBy = field;
                return this;
            }

            public Query<T> build() {
                return query;
            }
        }

        @Override
        public String toString() {
            return "Query{type=" + type.getSimpleName() +
                    ", filter='" + filter + "'" +
                    ", limit=" + limit +
                    ", orderBy='" + orderBy + "'}";
        }
    }

    // NAQSH 4: Tur-xavfsiz geterogen konteyner
    static class TypeSafeMap {
        private final Map<Class<?>, Object> map = new HashMap<>();

        // Tur-xavfsiz qo'shish
        public <T> void put(Class<T> type, T instance) {
            map.put(type, type.cast(instance));
        }

        // Tur-xavfsiz olish
        public <T> T get(Class<T> type) {
            return type.cast(map.get(type));
        }

        public <T> boolean contains(Class<T> type) {
            return map.containsKey(type);
        }

        public int size() {
            return map.size();
        }
    }

    // NAQSH 5: Umumiy singleton
    static class Singleton<T> {
        private static final Map<Class<?>, Object> instances = new ConcurrentHashMap<>();

        @SuppressWarnings("unchecked")
        public static <T> T getInstance(Class<T> type, Supplier<T> supplier) {
            return (T) instances.computeIfAbsent(type, k -> supplier.get());
        }

        public static <T> boolean hasInstance(Class<T> type) {
            return instances.containsKey(type);
        }
    }

    // Singleton uchun misol xizmat
    static class DatabaseService {
        private final String connectionString;

        public DatabaseService() {
            this.connectionString = "jdbc:db://localhost:5432";
            System.out.println("DatabaseService yaratildi");
        }

        public String getConnectionString() {
            return connectionString;
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Umumiy Fabrika Naqshi ===");
        Factory<User> userFactory = new UserFactory();
        Factory<Product> productFactory = new ProductFactory();

        User user1 = userFactory.create();
        User user2 = userFactory.create();
        Product product1 = productFactory.create();

        System.out.println("Yaratildi: " + user1);
        System.out.println("Yaratildi: " + user2);
        System.out.println("Yaratildi: " + product1);

        System.out.println("\\n=== Umumiy Repository Naqshi ===");
        InMemoryRepository<User, Integer> userRepo = new InMemoryRepository<>();
        userRepo.save(1, user1);
        userRepo.save(2, user2);

        System.out.println("Foydalanuvchilar soni: " + userRepo.size());
        System.out.println("Foydalanuvchi 1 ni topish: " + userRepo.findById(1));
        System.out.println("Barcha foydalanuvchilar: " + userRepo.findAll());
        System.out.println("Foydalanuvchi 3 mavjud: " + userRepo.exists(3));

        System.out.println("\\n=== Umumiy Builder Naqshi ===");
        Query<User> userQuery = Query.builder(User.class)
                .filter("age > 18")
                .limit(10)
                .orderBy("name")
                .build();

        Query<Product> productQuery = Query.builder(Product.class)
                .filter("price < 100")
                .limit(5)
                .build();

        System.out.println("Foydalanuvchi so'rovi: " + userQuery);
        System.out.println("Mahsulot so'rovi: " + productQuery);

        System.out.println("\\n=== Tur-xavfsiz Geterogen Konteyner ===");
        TypeSafeMap container = new TypeSafeMap();

        // Bir konteynerda turli turlarni saqlaymiz
        container.put(String.class, "Hello Generics");
        container.put(Integer.class, 42);
        container.put(Double.class, 3.14159);
        container.put(User.class, user1);

        // Tur-xavfsiz olish
        String str = container.get(String.class);
        Integer num = container.get(Integer.class);
        Double dbl = container.get(Double.class);
        User usr = container.get(User.class);

        System.out.println("String: " + str);
        System.out.println("Integer: " + num);
        System.out.println("Double: " + dbl);
        System.out.println("User: " + usr);
        System.out.println("Konteyner o'lchami: " + container.size());

        System.out.println("\\n=== Umumiy Singleton Naqshi ===");
        // Birinchi chaqiruv nusxani yaratadi
        DatabaseService db1 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        // Ikkinchi chaqiruv xuddi shu nusxani qaytaradi
        DatabaseService db2 = Singleton.getInstance(
                DatabaseService.class,
                DatabaseService::new
        );

        System.out.println("db1 ulanish: " + db1.getConnectionString());
        System.out.println("db2 ulanish: " + db2.getConnectionString());
        System.out.println("Bir xil nusxa? " + (db1 == db2));
        System.out.println("Nusxa bor? " + Singleton.hasInstance(DatabaseService.class));

        System.out.println("\\n=== Xulosa ===");
        System.out.println("1. Factory<T>: Tur-xavfsiz obyekt yaratish");
        System.out.println("2. Repository<T, ID>: Umumiy ma'lumotlarga kirish qatlami");
        System.out.println("3. Builder<T>: Moslashuvchan tur-xavfsiz qurilish");
        System.out.println("4. TypeSafeMap: Tur xavfsizligi bilan geterogen konteyner");
        System.out.println("5. Singleton<T>: Tur saqlanishi bilan umumiy singleton");
    }
}`,
            description: `Java umumiy tiplaridan foydalangan holda keng tarqalgan dizayn naqshlarini o'zlashtirang.

**Talablar:**
1. Factory<T> bilan Umumiy Fabrika naqshini amalga oshiring
2. Ma'lumotlarga kirish uchun Umumiy Repository naqshini yarating
3. Tur xavfsizligi uchun umumiy tiplar bilan Builder naqshini amalga oshiring
4. Tur-xavfsiz geterogen konteyner yarating (turli qiymat turlari bilan map)
5. Umumiy Singleton naqshini amalga oshiring
6. Barcha naqshlarni amaliy misollar bilan ko'rsating

Umumiy dizayn naqshlari umumiy dasturiy naqshlarning qayta foydalaniladigan, tur-xavfsiz amalga oshirilishini yaratish uchun tur parametrlaridan foydalanadi.`,
            hint1: `Umumiy naqshlar dizayn naqshlarini tur parametrlari bilan birlashtiradi. Factory<T> T turini yaratadi, Repository<T, ID> ID identifikatori bilan T turini boshqaradi.`,
            hint2: `Tur-xavfsiz geterogen konteynerlar turli turlarni saqlaganda tur xavfsizligini saqlab qolish uchun Class<T> ni kalitlar sifatida ishlatadi. Umumiy tiplar bilan Builder naqshi qurilish jarayoni davomida tur xavfsizligini ta'minlaydi.`,
            whyItMatters: `Umumiy dizayn naqshlari qayta foydalaniladigan, boshqariladigan kutubxonalar va freymvorklar yaratish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Generic Repository naqshi
interface Repository<T, ID> {
    void save(T entity);
    Optional<T> findById(ID id);
    List<T> findAll();
}

// Generic Factory naqshi
interface Factory<T> {
    T create();
}

// Foydalanish
Repository<User, Integer> userRepo = new InMemoryRepository<>();
userRepo.save(new User("Alice"));
Optional<User> user = userRepo.findById(1);
\`\`\`

**Amaliy foydalari:**
- Spring Framework bu naqshlardan foydalanadi
- Hibernate, JPA Repository<T, ID> dan foydalanadi
- Tur-xavfsiz obyekt yaratish
- Senior dasturchilari uchun muhim`
        }
    }
};

export default task;
