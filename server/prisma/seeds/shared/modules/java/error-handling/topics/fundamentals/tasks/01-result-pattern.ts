import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-result-pattern',
    title: 'Result/Either Pattern',
    difficulty: 'medium',
    tags: ['java', 'error-handling', 'result-pattern', 'functional'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement the Result/Either pattern for error handling without exceptions.

Requirements:
1. Create a generic Result<T> class with success and error states
2. Implement static factory methods: success() and failure()
3. Add map(), flatMap(), and orElse() methods
4. Create a UserService that uses Result instead of throwing exceptions
5. Handle errors functionally without try-catch blocks

Example:
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Welcome " + user.getName())
    .orElse("User not found");
\`\`\``,
    initialCode: `// TODO: Create Result<T> class with success and error states

class User {
    private String email;
    private String name;

    public User(String email, String name) {
        this.email = email;
        this.name = name;
    }

    public String getEmail() { return email; }
    public String getName() { return name; }
}

class UserService {
    // TODO: Implement findUser using Result pattern
    public Result<User> findUser(String email) {
        // TODO: Return Result.success() or Result.failure()
        return null;
    }

    // TODO: Implement createUser using Result pattern
    public Result<User> createUser(String email, String name) {
        // TODO: Validate and return Result
        return null;
    }
}

public class ResultPatternDemo {
    public static void main(String[] args) {
        UserService service = new UserService();

        // TODO: Use Result pattern to handle user operations
        Result<User> result = service.findUser("john@example.com");

        // TODO: Map and handle result functionally
    }
}`,
    solutionCode: `// Generic Result class for error handling without exceptions
class Result<T> {
    private final T value;
    private final String error;
    private final boolean success;

    // Private constructor to enforce factory methods
    private Result(T value, String error, boolean success) {
        this.value = value;
        this.error = error;
        this.success = success;
    }

    // Factory method for successful result
    public static <T> Result<T> success(T value) {
        return new Result<>(value, null, true);
    }

    // Factory method for failed result
    public static <T> Result<T> failure(String error) {
        return new Result<>(null, error, false);
    }

    // Check if result is successful
    public boolean isSuccess() {
        return success;
    }

    // Get value (throws if failed)
    public T getValue() {
        if (!success) {
            throw new IllegalStateException("Cannot get value from failed result");
        }
        return value;
    }

    // Get error message
    public String getError() {
        return error;
    }

    // Transform value if successful
    public <U> Result<U> map(java.util.function.Function<T, U> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return Result.success(mapper.apply(value));
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Chain operations that return Result
    public <U> Result<U> flatMap(java.util.function.Function<T, Result<U>> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return mapper.apply(value);
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Get value or default
    public T orElse(T defaultValue) {
        return success ? value : defaultValue;
    }

    // Get value or compute from error
    public T orElseGet(java.util.function.Function<String, T> errorHandler) {
        return success ? value : errorHandler.apply(error);
    }
}

class User {
    private String email;
    private String name;

    public User(String email, String name) {
        this.email = email;
        this.name = name;
    }

    public String getEmail() { return email; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return "User{email='" + email + "', name='" + name + "'}";
    }
}

class UserService {
    // Simulate user database
    private final java.util.Map<String, User> users = new java.util.HashMap<>();

    public UserService() {
        // Add sample users
        users.put("john@example.com", new User("john@example.com", "John Doe"));
        users.put("jane@example.com", new User("jane@example.com", "Jane Smith"));
    }

    // Find user using Result pattern
    public Result<User> findUser(String email) {
        if (email == null || email.trim().isEmpty()) {
            return Result.failure("Email cannot be empty");
        }

        User user = users.get(email);
        if (user == null) {
            return Result.failure("User not found: " + email);
        }

        return Result.success(user);
    }

    // Create user using Result pattern
    public Result<User> createUser(String email, String name) {
        // Validate email
        if (email == null || !email.contains("@")) {
            return Result.failure("Invalid email format");
        }

        // Validate name
        if (name == null || name.trim().isEmpty()) {
            return Result.failure("Name cannot be empty");
        }

        // Check if user already exists
        if (users.containsKey(email)) {
            return Result.failure("User already exists: " + email);
        }

        // Create and store user
        User user = new User(email, name);
        users.put(email, user);
        return Result.success(user);
    }
}

public class ResultPatternDemo {
    public static void main(String[] args) {
        UserService service = new UserService();

        // Example 1: Successful user lookup
        System.out.println("=== Example 1: Find existing user ===");
        Result<User> result1 = service.findUser("john@example.com");
        String message1 = result1
            .map(user -> "Welcome, " + user.getName() + "!")
            .orElse("User not found");
        System.out.println(message1);

        // Example 2: User not found
        System.out.println("\\n=== Example 2: Find non-existent user ===");
        Result<User> result2 = service.findUser("unknown@example.com");
        if (!result2.isSuccess()) {
            System.out.println("Error: " + result2.getError());
        }

        // Example 3: Chain operations with flatMap
        System.out.println("\\n=== Example 3: Chain operations ===");
        String greeting = service.findUser("jane@example.com")
            .flatMap(user -> Result.success("Hello, " + user.getName()))
            .map(String::toUpperCase)
            .orElse("No greeting available");
        System.out.println(greeting);

        // Example 4: Create new user
        System.out.println("\\n=== Example 4: Create new user ===");
        Result<User> result3 = service.createUser("bob@example.com", "Bob Wilson");
        result3.map(user -> {
            System.out.println("Created: " + user);
            return user;
        });

        // Example 5: Validation error
        System.out.println("\\n=== Example 5: Validation error ===");
        Result<User> result4 = service.createUser("invalid-email", "Test User");
        if (!result4.isSuccess()) {
            System.out.println("Validation failed: " + result4.getError());
        }

        // Example 6: Handle with orElseGet
        System.out.println("\\n=== Example 6: Custom error handling ===");
        User user = service.findUser("missing@example.com")
            .orElseGet(error -> new User("default@example.com", "Guest"));
        System.out.println("Resolved user: " + user);
    }
}`,
    hint1: `The Result class should have private fields for value, error, and success flag. Use static factory methods to create instances and prevent invalid states.`,
    hint2: `Implement map() to transform the value if successful, and flatMap() to chain operations that return Result. Both methods should propagate errors automatically.`,
    whyItMatters: `The Result pattern provides type-safe error handling without exceptions. It makes error handling explicit in the type system, improves code readability, and enables functional composition of operations. This pattern is widely used in modern Java applications and functional programming.

**Production Pattern:**
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Welcome, " + user.getName())
    .orElse("User not found");
\`\`\`

**Practical Benefits:**
- Type-safe error handling in the type system
- Functional composition of operations with map() and flatMap()
- Widely used in modern Java applications`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify Result.success creates successful result
class Test1 {
    @Test
    public void test() {
        Result<String> result = Result.success("test value");
        assertTrue(result.isSuccess());
        assertEquals("test value", result.getValue());
    }
}

// Test2: Verify Result.failure creates failed result
class Test2 {
    @Test
    public void test() {
        Result<String> result = Result.failure("error message");
        assertFalse(result.isSuccess());
        assertEquals("error message", result.getError());
    }
}

// Test3: Verify map() transforms successful result
class Test3 {
    @Test
    public void test() {
        Result<String> result = Result.success("hello");
        Result<String> mapped = result.map(String::toUpperCase);
        assertTrue(mapped.isSuccess());
        assertEquals("HELLO", mapped.getValue());
    }
}

// Test4: Verify map() propagates error
class Test4 {
    @Test
    public void test() {
        Result<String> result = Result.failure("error");
        Result<Integer> mapped = result.map(String::length);
        assertFalse(mapped.isSuccess());
        assertEquals("error", mapped.getError());
    }
}

// Test5: Verify orElse() returns value for success
class Test5 {
    @Test
    public void test() {
        Result<String> result = Result.success("value");
        String output = result.orElse("default");
        assertEquals("value", output);
    }
}

// Test6: Verify orElse() returns default for failure
class Test6 {
    @Test
    public void test() {
        Result<String> result = Result.failure("error");
        String output = result.orElse("default");
        assertEquals("default", output);
    }
}

// Test7: Verify UserService findUser success
class Test7 {
    @Test
    public void test() {
        UserService service = new UserService();
        Result<User> result = service.findUser("john@example.com");
        assertTrue(result.isSuccess());
        assertEquals("john@example.com", result.getValue().getEmail());
    }
}

// Test8: Verify UserService findUser failure
class Test8 {
    @Test
    public void test() {
        UserService service = new UserService();
        Result<User> result = service.findUser("unknown@example.com");
        assertFalse(result.isSuccess());
        assertTrue(result.getError().contains("not found"));
    }
}

// Test9: Verify UserService createUser success
class Test9 {
    @Test
    public void test() {
        UserService service = new UserService();
        Result<User> result = service.createUser("new@example.com", "New User");
        assertTrue(result.isSuccess());
        assertEquals("new@example.com", result.getValue().getEmail());
        assertEquals("New User", result.getValue().getName());
    }
}

// Test10: Verify UserService createUser validation
class Test10 {
    @Test
    public void test() {
        UserService service = new UserService();

        // Test invalid email
        Result<User> result1 = service.createUser("invalid-email", "Test");
        assertFalse(result1.isSuccess());
        assertTrue(result1.getError().contains("email"));

        // Test empty name
        Result<User> result2 = service.createUser("test@example.com", "");
        assertFalse(result2.isSuccess());
        assertTrue(result2.getError().contains("Name"));
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Паттерн Result/Either',
            solutionCode: `// Универсальный класс Result для обработки ошибок без исключений
class Result<T> {
    private final T value;
    private final String error;
    private final boolean success;

    // Приватный конструктор для обеспечения фабричных методов
    private Result(T value, String error, boolean success) {
        this.value = value;
        this.error = error;
        this.success = success;
    }

    // Фабричный метод для успешного результата
    public static <T> Result<T> success(T value) {
        return new Result<>(value, null, true);
    }

    // Фабричный метод для неудачного результата
    public static <T> Result<T> failure(String error) {
        return new Result<>(null, error, false);
    }

    // Проверка успешности результата
    public boolean isSuccess() {
        return success;
    }

    // Получение значения (выбрасывает исключение при ошибке)
    public T getValue() {
        if (!success) {
            throw new IllegalStateException("Невозможно получить значение из неудачного результата");
        }
        return value;
    }

    // Получение сообщения об ошибке
    public String getError() {
        return error;
    }

    // Преобразование значения при успехе
    public <U> Result<U> map(java.util.function.Function<T, U> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return Result.success(mapper.apply(value));
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Цепочка операций, возвращающих Result
    public <U> Result<U> flatMap(java.util.function.Function<T, Result<U>> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return mapper.apply(value);
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Получение значения или значения по умолчанию
    public T orElse(T defaultValue) {
        return success ? value : defaultValue;
    }

    // Получение значения или вычисление из ошибки
    public T orElseGet(java.util.function.Function<String, T> errorHandler) {
        return success ? value : errorHandler.apply(error);
    }
}

class User {
    private String email;
    private String name;

    public User(String email, String name) {
        this.email = email;
        this.name = name;
    }

    public String getEmail() { return email; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return "User{email='" + email + "', name='" + name + "'}";
    }
}

class UserService {
    // Имитация базы данных пользователей
    private final java.util.Map<String, User> users = new java.util.HashMap<>();

    public UserService() {
        // Добавление примеров пользователей
        users.put("john@example.com", new User("john@example.com", "Иван Иванов"));
        users.put("jane@example.com", new User("jane@example.com", "Елена Смирнова"));
    }

    // Поиск пользователя с использованием паттерна Result
    public Result<User> findUser(String email) {
        if (email == null || email.trim().isEmpty()) {
            return Result.failure("Email не может быть пустым");
        }

        User user = users.get(email);
        if (user == null) {
            return Result.failure("Пользователь не найден: " + email);
        }

        return Result.success(user);
    }

    // Создание пользователя с использованием паттерна Result
    public Result<User> createUser(String email, String name) {
        // Проверка email
        if (email == null || !email.contains("@")) {
            return Result.failure("Неверный формат email");
        }

        // Проверка имени
        if (name == null || name.trim().isEmpty()) {
            return Result.failure("Имя не может быть пустым");
        }

        // Проверка существования пользователя
        if (users.containsKey(email)) {
            return Result.failure("Пользователь уже существует: " + email);
        }

        // Создание и сохранение пользователя
        User user = new User(email, name);
        users.put(email, user);
        return Result.success(user);
    }
}

public class ResultPatternDemo {
    public static void main(String[] args) {
        UserService service = new UserService();

        // Пример 1: Успешный поиск пользователя
        System.out.println("=== Пример 1: Поиск существующего пользователя ===");
        Result<User> result1 = service.findUser("john@example.com");
        String message1 = result1
            .map(user -> "Добро пожаловать, " + user.getName() + "!")
            .orElse("Пользователь не найден");
        System.out.println(message1);

        // Пример 2: Пользователь не найден
        System.out.println("\\n=== Пример 2: Поиск несуществующего пользователя ===");
        Result<User> result2 = service.findUser("unknown@example.com");
        if (!result2.isSuccess()) {
            System.out.println("Ошибка: " + result2.getError());
        }

        // Пример 3: Цепочка операций с flatMap
        System.out.println("\\n=== Пример 3: Цепочка операций ===");
        String greeting = service.findUser("jane@example.com")
            .flatMap(user -> Result.success("Привет, " + user.getName()))
            .map(String::toUpperCase)
            .orElse("Приветствие недоступно");
        System.out.println(greeting);

        // Пример 4: Создание нового пользователя
        System.out.println("\\n=== Пример 4: Создание нового пользователя ===");
        Result<User> result3 = service.createUser("bob@example.com", "Борис Петров");
        result3.map(user -> {
            System.out.println("Создан: " + user);
            return user;
        });

        // Пример 5: Ошибка валидации
        System.out.println("\\n=== Пример 5: Ошибка валидации ===");
        Result<User> result4 = service.createUser("invalid-email", "Тестовый Пользователь");
        if (!result4.isSuccess()) {
            System.out.println("Валидация не прошла: " + result4.getError());
        }

        // Пример 6: Обработка с orElseGet
        System.out.println("\\n=== Пример 6: Пользовательская обработка ошибок ===");
        User user = service.findUser("missing@example.com")
            .orElseGet(error -> new User("default@example.com", "Гость"));
        System.out.println("Разрешенный пользователь: " + user);
    }
}`,
            description: `Реализуйте паттерн Result/Either для обработки ошибок без исключений.

Требования:
1. Создайте универсальный класс Result<T> с состояниями успеха и ошибки
2. Реализуйте статические фабричные методы: success() и failure()
3. Добавьте методы map(), flatMap() и orElse()
4. Создайте UserService, использующий Result вместо выбрасывания исключений
5. Обрабатывайте ошибки функционально без блоков try-catch

Пример:
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Добро пожаловать " + user.getName())
    .orElse("Пользователь не найден");
\`\`\``,
            hint1: `Класс Result должен иметь приватные поля для значения, ошибки и флага успеха. Используйте статические фабричные методы для создания экземпляров и предотвращения недопустимых состояний.`,
            hint2: `Реализуйте map() для преобразования значения при успехе, и flatMap() для цепочки операций, возвращающих Result. Оба метода должны автоматически распространять ошибки.`,
            whyItMatters: `Паттерн Result обеспечивает типобезопасную обработку ошибок без исключений. Он делает обработку ошибок явной в системе типов, улучшает читаемость кода и позволяет функциональную композицию операций.

**Продакшен паттерн:**
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Добро пожаловать, " + user.getName())
    .orElse("Пользователь не найден");
\`\`\`

**Практические преимущества:**
- Типобезопасная обработка ошибок в системе типов
- Функциональная композиция операций с map() и flatMap()
- Широко используется в современных Java-приложениях`
        },
        uz: {
            title: `Result/Either Patterni`,
            solutionCode: `// Istisnolarsiz xatolarni qayta ishlash uchun universal Result klassi
class Result<T> {
    private final T value;
    private final String error;
    private final boolean success;

    // Fabrika metodlarini ta'minlash uchun privat konstruktor
    private Result(T value, String error, boolean success) {
        this.value = value;
        this.error = error;
        this.success = success;
    }

    // Muvaffaqiyatli natija uchun fabrika metodi
    public static <T> Result<T> success(T value) {
        return new Result<>(value, null, true);
    }

    // Muvaffaqiyatsiz natija uchun fabrika metodi
    public static <T> Result<T> failure(String error) {
        return new Result<>(null, error, false);
    }

    // Natijaning muvaffaqiyatligini tekshirish
    public boolean isSuccess() {
        return success;
    }

    // Qiymatni olish (xato bo'lsa istisno tashlaydi)
    public T getValue() {
        if (!success) {
            throw new IllegalStateException("Muvaffaqiyatsiz natijadan qiymat olish mumkin emas");
        }
        return value;
    }

    // Xato xabarini olish
    public String getError() {
        return error;
    }

    // Muvaffaqiyat bo'lsa qiymatni o'zgartirish
    public <U> Result<U> map(java.util.function.Function<T, U> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return Result.success(mapper.apply(value));
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Result qaytaruvchi operatsiyalar zanjiri
    public <U> Result<U> flatMap(java.util.function.Function<T, Result<U>> mapper) {
        if (!success) {
            return Result.failure(error);
        }
        try {
            return mapper.apply(value);
        } catch (Exception e) {
            return Result.failure(e.getMessage());
        }
    }

    // Qiymat yoki standart qiymatni olish
    public T orElse(T defaultValue) {
        return success ? value : defaultValue;
    }

    // Qiymat yoki xatodan hisoblashni olish
    public T orElseGet(java.util.function.Function<String, T> errorHandler) {
        return success ? value : errorHandler.apply(error);
    }
}

class User {
    private String email;
    private String name;

    public User(String email, String name) {
        this.email = email;
        this.name = name;
    }

    public String getEmail() { return email; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return "User{email='" + email + "', name='" + name + "'}";
    }
}

class UserService {
    // Foydalanuvchilar bazasini taqlid qilish
    private final java.util.Map<String, User> users = new java.util.HashMap<>();

    public UserService() {
        // Namuna foydalanuvchilarini qo'shish
        users.put("john@example.com", new User("john@example.com", "Jovohir Karimov"));
        users.put("jane@example.com", new User("jane@example.com", "Jamila Usmonova"));
    }

    // Result patternidan foydalanib foydalanuvchini topish
    public Result<User> findUser(String email) {
        if (email == null || email.trim().isEmpty()) {
            return Result.failure("Email bo'sh bo'lishi mumkin emas");
        }

        User user = users.get(email);
        if (user == null) {
            return Result.failure("Foydalanuvchi topilmadi: " + email);
        }

        return Result.success(user);
    }

    // Result patternidan foydalanib foydalanuvchi yaratish
    public Result<User> createUser(String email, String name) {
        // Email tekshirish
        if (email == null || !email.contains("@")) {
            return Result.failure("Email formati noto'g'ri");
        }

        // Ism tekshirish
        if (name == null || name.trim().isEmpty()) {
            return Result.failure("Ism bo'sh bo'lishi mumkin emas");
        }

        // Foydalanuvchi mavjudligini tekshirish
        if (users.containsKey(email)) {
            return Result.failure("Foydalanuvchi allaqachon mavjud: " + email);
        }

        // Foydalanuvchini yaratish va saqlash
        User user = new User(email, name);
        users.put(email, user);
        return Result.success(user);
    }
}

public class ResultPatternDemo {
    public static void main(String[] args) {
        UserService service = new UserService();

        // Misol 1: Muvaffaqiyatli foydalanuvchini topish
        System.out.println("=== Misol 1: Mavjud foydalanuvchini topish ===");
        Result<User> result1 = service.findUser("john@example.com");
        String message1 = result1
            .map(user -> "Xush kelibsiz, " + user.getName() + "!")
            .orElse("Foydalanuvchi topilmadi");
        System.out.println(message1);

        // Misol 2: Foydalanuvchi topilmadi
        System.out.println("\\n=== Misol 2: Mavjud bo'lmagan foydalanuvchini topish ===");
        Result<User> result2 = service.findUser("unknown@example.com");
        if (!result2.isSuccess()) {
            System.out.println("Xato: " + result2.getError());
        }

        // Misol 3: flatMap bilan operatsiyalar zanjiri
        System.out.println("\\n=== Misol 3: Operatsiyalar zanjiri ===");
        String greeting = service.findUser("jane@example.com")
            .flatMap(user -> Result.success("Salom, " + user.getName()))
            .map(String::toUpperCase)
            .orElse("Salomlashish mavjud emas");
        System.out.println(greeting);

        // Misol 4: Yangi foydalanuvchi yaratish
        System.out.println("\\n=== Misol 4: Yangi foydalanuvchi yaratish ===");
        Result<User> result3 = service.createUser("bob@example.com", "Bobur Rashidov");
        result3.map(user -> {
            System.out.println("Yaratildi: " + user);
            return user;
        });

        // Misol 5: Validatsiya xatosi
        System.out.println("\\n=== Misol 5: Validatsiya xatosi ===");
        Result<User> result4 = service.createUser("invalid-email", "Test Foydalanuvchi");
        if (!result4.isSuccess()) {
            System.out.println("Validatsiya muvaffaqiyatsiz: " + result4.getError());
        }

        // Misol 6: orElseGet bilan qayta ishlash
        System.out.println("\\n=== Misol 6: Maxsus xatolarni qayta ishlash ===");
        User user = service.findUser("missing@example.com")
            .orElseGet(error -> new User("default@example.com", "Mehmon"));
        System.out.println("Hal qilingan foydalanuvchi: " + user);
    }
}`,
            description: `Istisnolarsiz xatolarni qayta ishlash uchun Result/Either patternini amalga oshiring.

Talablar:
1. Muvaffaqiyat va xato holatlari bilan umumiy Result<T> klassini yarating
2. Statik fabrika metodlarini amalga oshiring: success() va failure()
3. map(), flatMap() va orElse() metodlarini qo'shing
4. Istisnolarni tashlash o'rniga Result ishlatadigan UserService yarating
5. Xatolarni try-catch bloklarisiz funktsional tarzda qayta ishlang

Misol:
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Xush kelibsiz " + user.getName())
    .orElse("Foydalanuvchi topilmadi");
\`\`\``,
            hint1: `Result klassi qiymat, xato va muvaffaqiyat belgisi uchun privat maydonlarga ega bo'lishi kerak. Nusxalar yaratish va noto'g'ri holatlarni oldini olish uchun statik fabrika metodlaridan foydalaning.`,
            hint2: `Muvaffaqiyat bo'lsa qiymatni o'zgartirish uchun map() va Result qaytaruvchi operatsiyalar zanjiri uchun flatMap() ni amalga oshiring. Ikkala metod ham xatolarni avtomatik tarzda tarqatishi kerak.`,
            whyItMatters: `Result patterni istisnolarsiz tip-xavfsiz xatolarni qayta ishlashni ta'minlaydi. U tip tizimida xatolarni qayta ishlashni aniq qiladi, kod o'qilishini yaxshilaydi va operatsiyalarning funktsional kompozitsiyasini yoqadi.

**Ishlab chiqarish patterni:**
\`\`\`java
Result<User> result = userService.findUser("john@example.com");
String message = result
    .map(user -> "Xush kelibsiz, " + user.getName())
    .orElse("Foydalanuvchi topilmadi");
\`\`\`

**Amaliy foydalari:**
- Tip tizimida tip-xavfsiz xatolarni qayta ishlash
- map() va flatMap() bilan operatsiyalarning funktsional kompozitsiyasi
- Zamonaviy Java ilovalarida keng qo'llaniladi`
        }
    }
};

export default task;
