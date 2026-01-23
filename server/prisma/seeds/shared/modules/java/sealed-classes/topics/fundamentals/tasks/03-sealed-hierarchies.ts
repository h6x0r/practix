import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-sealed-hierarchies',
    title: 'Sealed Type Hierarchies',
    difficulty: 'medium',
    tags: ['java', 'sealed', 'hierarchies', 'design', 'type-safety'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn to design complex type hierarchies using sealed classes.

**Requirements:**
1. Create a sealed interface Result<T> that permits Success<T> and Failure
2. Create a final record Success<T> with T value field
3. Create a sealed class Failure that permits NetworkError, ValidationError, and DatabaseError
4. Create final classes for each error type with appropriate fields
5. Add methods to Result: boolean isSuccess(), boolean isFailure()
6. Create generic methods to handle Result: map(), flatMap(), getOrDefault()
7. Demonstrate usage with a simulated API call that returns Result<String>

Sealed hierarchies allow you to model complex domain logic with nested sealed types for maximum type safety.`,
    initialCode: `// Create sealed interface Result<T> that permits Success, Failure

// Create final record Success<T> with T value

// Create sealed class Failure that permits NetworkError, ValidationError, DatabaseError

// Create final class NetworkError with String url, int statusCode

// Create final class ValidationError with String field, String message

// Create final class DatabaseError with String query, String errorCode

public class SealedHierarchies {
    // Create method: Result<String> fetchUser(int id)
    // Simulate API call returning Result

    // Create method: String handleResult(Result<String> result)
    // Use pattern matching to handle all cases

    public static void main(String[] args) {
        // Test with different scenarios
    }
}`,
    solutionCode: `// Sealed interface with generic type parameter
sealed interface Result<T> permits Success, Failure {
    boolean isSuccess();
    boolean isFailure();
}

// Generic record for successful results
final record Success<T>(T value) implements Result<T> {
    @Override
    public boolean isSuccess() {
        return true;
    }

    @Override
    public boolean isFailure() {
        return false;
    }
}

// Sealed class for failures with nested hierarchy
sealed class Failure implements Result {
    @Override
    public boolean isSuccess() {
        return false;
    }

    @Override
    public boolean isFailure() {
        return true;
    }
}

// Network error with connection details
final class NetworkError extends Failure {
    private final String url;
    private final int statusCode;

    public NetworkError(String url, int statusCode) {
        this.url = url;
        this.statusCode = statusCode;
    }

    public String getUrl() {
        return url;
    }

    public int getStatusCode() {
        return statusCode;
    }
}

// Validation error with field details
final class ValidationError extends Failure {
    private final String field;
    private final String message;

    public ValidationError(String field, String message) {
        this.field = field;
        this.message = message;
    }

    public String getField() {
        return field;
    }

    public String getMessage() {
        return message;
    }
}

// Database error with query details
final class DatabaseError extends Failure {
    private final String query;
    private final String errorCode;

    public DatabaseError(String query, String errorCode) {
        this.query = query;
        this.errorCode = errorCode;
    }

    public String getQuery() {
        return query;
    }

    public String getErrorCode() {
        return errorCode;
    }
}

public class SealedHierarchies {
    // Simulate API call with different outcomes
    static Result<String> fetchUser(int id) {
        return switch (id) {
            case 1 -> new Success<>("Alice");
            case 2 -> new NetworkError("https://api.example.com/users/2", 404);
            case 3 -> new ValidationError("id", "Invalid user ID format");
            case 4 -> new DatabaseError("SELECT * FROM users WHERE id = 4", "DB_CONNECTION_LOST");
            default -> new Success<>("Unknown User");
        };
    }

    // Exhaustive pattern matching with sealed hierarchy
    static String handleResult(Result<String> result) {
        // Compiler knows all possible types
        if (result instanceof Success<String> success) {
            return "Success: Found user " + success.value();
        } else if (result instanceof NetworkError networkError) {
            return "Network Error: " + networkError.getStatusCode() + " at " + networkError.getUrl();
        } else if (result instanceof ValidationError validationError) {
            return "Validation Error in field '" + validationError.getField() + "': " + validationError.getMessage();
        } else if (result instanceof DatabaseError dbError) {
            return "Database Error (" + dbError.getErrorCode() + "): " + dbError.getQuery();
        }
        // No default needed - compiler ensures exhaustiveness
        throw new IllegalStateException("Unexpected result type");
    }

    public static void main(String[] args) {
        // Test all scenarios
        System.out.println(handleResult(fetchUser(1)));	// Success
        System.out.println(handleResult(fetchUser(2)));	// Network error
        System.out.println(handleResult(fetchUser(3)));	// Validation error
        System.out.println(handleResult(fetchUser(4)));	// Database error

        // Type safety: all cases must be handled
        Result<String> result = fetchUser(1);
        if (result.isSuccess() && result instanceof Success<String> success) {
            System.out.println("");
            System.out.println("User data: " + success.value());
        }
    }
}`,
    hint1: `Use nested sealed types: sealed interface Result permits Success, Failure; then sealed class Failure permits specific error types.`,
    hint2: `Generic sealed types work like regular generics: sealed interface Result<T> permits Success, Failure. Success should be generic: record Success<T>(T value).`,
    whyItMatters: `Sealed hierarchies are essential for error handling, state machines, and domain modeling. They provide compile-time guarantees that all cases are handled, eliminating entire classes of runtime errors common in traditional OOP hierarchies.

**Production Pattern:**
\`\`\`java
sealed interface OrderStatus permits Draft, Submitted, Processing, Completed, Cancelled {}
final record Draft() implements OrderStatus {}
final record Submitted(LocalDateTime at) implements OrderStatus {}
final record Processing(String warehouseId) implements OrderStatus {}
final record Completed(LocalDateTime at, String trackingNumber) implements OrderStatus {}
final record Cancelled(String reason) implements OrderStatus {}

public String getStatusMessage(OrderStatus status) {
    return switch (status) {
        case Draft d -> "Order is in draft";
        case Submitted s -> "Submitted at " + s.at();
        case Processing p -> "Processing in warehouse " + p.warehouseId();
        case Completed c -> "Completed: " + c.trackingNumber();
        case Cancelled c -> "Cancelled: " + c.reason();
    };
}
\`\`\`

**Practical Benefits:**
- Impossible to forget handling order status
- Each status can have unique data`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Define sealed hierarchy for Result pattern
sealed interface Result<T> permits Success, Failure {
    boolean isSuccess();
}

final class Success<T> implements Result<T> {
    private final T value;
    Success(T value) { this.value = value; }
    public boolean isSuccess() { return true; }
    public T getValue() { return value; }
}

final class Failure<T> implements Result<T> {
    private final String error;
    Failure(String error) { this.error = error; }
    public boolean isSuccess() { return false; }
    public String getError() { return error; }
}

// Test1: Test Success creation
class Test1 {
    @Test
    public void test() {
        Result<Integer> result = new Success<>(42);
        assertTrue(result.isSuccess());
    }
}

// Test2: Test Failure creation
class Test2 {
    @Test
    public void test() {
        Result<String> result = new Failure<>("Error occurred");
        assertFalse(result.isSuccess());
    }
}

// Test3: Test Success value retrieval
class Test3 {
    @Test
    public void test() {
        Success<String> success = new Success<>("Hello");
        assertEquals("Hello", success.getValue());
    }
}

// Test4: Test Failure error retrieval
class Test4 {
    @Test
    public void test() {
        Failure<Integer> failure = new Failure<>("Not found");
        assertEquals("Not found", failure.getError());
    }
}

// Test5: Test instanceof with Success
class Test5 {
    @Test
    public void test() {
        Result<Integer> result = new Success<>(100);
        assertTrue(result instanceof Success);
        assertFalse(result instanceof Failure);
    }
}

// Test6: Test instanceof with Failure
class Test6 {
    @Test
    public void test() {
        Result<String> result = new Failure<>("Failed");
        assertTrue(result instanceof Failure);
        assertFalse(result instanceof Success);
    }
}

// Test7: Test Success with different types
class Test7 {
    @Test
    public void test() {
        Success<Integer> intSuccess = new Success<>(42);
        Success<String> strSuccess = new Success<>("Success");
        assertEquals(Integer.valueOf(42), intSuccess.getValue());
        assertEquals("Success", strSuccess.getValue());
    }
}

// Test8: Test Failure with different messages
class Test8 {
    @Test
    public void test() {
        Failure<Integer> f1 = new Failure<>("Error 1");
        Failure<String> f2 = new Failure<>("Error 2");
        assertEquals("Error 1", f1.getError());
        assertEquals("Error 2", f2.getError());
    }
}

// Test9: Test Result pattern matching
class Test9 {
    @Test
    public void test() {
        Result<Integer> success = new Success<>(10);
        Result<Integer> failure = new Failure<>("Failed");
        assertTrue(success.isSuccess());
        assertFalse(failure.isSuccess());
    }
}

// Test10: Test multiple Results
class Test10 {
    @Test
    public void test() {
        Result<String>[] results = new Result[] {
            new Success<>("OK"),
            new Failure<>("Error"),
            new Success<>("Done")
        };
        assertEquals(3, results.length);
        assertTrue(results[0].isSuccess());
        assertFalse(results[1].isSuccess());
        assertTrue(results[2].isSuccess());
    }
}
`,
    translations: {
        ru: {
            title: 'Иерархии запечатанных типов',
            solutionCode: `// Запечатанный интерфейс с параметром обобщенного типа
sealed interface Result<T> permits Success, Failure {
    boolean isSuccess();
    boolean isFailure();
}

// Обобщенная запись для успешных результатов
final record Success<T>(T value) implements Result<T> {
    @Override
    public boolean isSuccess() {
        return true;
    }

    @Override
    public boolean isFailure() {
        return false;
    }
}

// Запечатанный класс для ошибок с вложенной иерархией
sealed class Failure implements Result {
    @Override
    public boolean isSuccess() {
        return false;
    }

    @Override
    public boolean isFailure() {
        return true;
    }
}

// Сетевая ошибка с деталями соединения
final class NetworkError extends Failure {
    private final String url;
    private final int statusCode;

    public NetworkError(String url, int statusCode) {
        this.url = url;
        this.statusCode = statusCode;
    }

    public String getUrl() {
        return url;
    }

    public int getStatusCode() {
        return statusCode;
    }
}

// Ошибка валидации с деталями поля
final class ValidationError extends Failure {
    private final String field;
    private final String message;

    public ValidationError(String field, String message) {
        this.field = field;
        this.message = message;
    }

    public String getField() {
        return field;
    }

    public String getMessage() {
        return message;
    }
}

// Ошибка базы данных с деталями запроса
final class DatabaseError extends Failure {
    private final String query;
    private final String errorCode;

    public DatabaseError(String query, String errorCode) {
        this.query = query;
        this.errorCode = errorCode;
    }

    public String getQuery() {
        return query;
    }

    public String getErrorCode() {
        return errorCode;
    }
}

public class SealedHierarchies {
    // Имитация API вызова с разными исходами
    static Result<String> fetchUser(int id) {
        return switch (id) {
            case 1 -> new Success<>("Alice");
            case 2 -> new NetworkError("https://api.example.com/users/2", 404);
            case 3 -> new ValidationError("id", "Неверный формат ID пользователя");
            case 4 -> new DatabaseError("SELECT * FROM users WHERE id = 4", "DB_CONNECTION_LOST");
            default -> new Success<>("Неизвестный пользователь");
        };
    }

    // Исчерпывающее сопоставление с образцом с запечатанной иерархией
    static String handleResult(Result<String> result) {
        // Компилятор знает все возможные типы
        if (result instanceof Success<String> success) {
            return "Успех: Найден пользователь " + success.value();
        } else if (result instanceof NetworkError networkError) {
            return "Сетевая ошибка: " + networkError.getStatusCode() + " по адресу " + networkError.getUrl();
        } else if (result instanceof ValidationError validationError) {
            return "Ошибка валидации в поле '" + validationError.getField() + "': " + validationError.getMessage();
        } else if (result instanceof DatabaseError dbError) {
            return "Ошибка базы данных (" + dbError.getErrorCode() + "): " + dbError.getQuery();
        }
        // default не нужен - компилятор обеспечивает исчерпываемость
        throw new IllegalStateException("Неожиданный тип результата");
    }

    public static void main(String[] args) {
        // Тестируем все сценарии
        System.out.println(handleResult(fetchUser(1)));	// Успех
        System.out.println(handleResult(fetchUser(2)));	// Сетевая ошибка
        System.out.println(handleResult(fetchUser(3)));	// Ошибка валидации
        System.out.println(handleResult(fetchUser(4)));	// Ошибка базы данных

        // Безопасность типов: все случаи должны быть обработаны
        Result<String> result = fetchUser(1);
        if (result.isSuccess() && result instanceof Success<String> success) {
            System.out.println("");
            System.out.println("Данные пользователя: " + success.value());
        }
    }
}`,
            description: `Научитесь проектировать сложные иерархии типов с использованием запечатанных классов.

**Требования:**
1. Создайте запечатанный интерфейс Result<T>, который разрешает Success<T> и Failure
2. Создайте финальную запись Success<T> с полем value типа T
3. Создайте запечатанный класс Failure, который разрешает NetworkError, ValidationError и DatabaseError
4. Создайте финальные классы для каждого типа ошибки с соответствующими полями
5. Добавьте методы в Result: boolean isSuccess(), boolean isFailure()
6. Создайте обобщенные методы для обработки Result: map(), flatMap(), getOrDefault()
7. Продемонстрируйте использование с имитацией API вызова, возвращающего Result<String>

Запечатанные иерархии позволяют моделировать сложную доменную логику с вложенными запечатанными типами для максимальной безопасности типов.`,
            hint1: `Используйте вложенные запечатанные типы: sealed interface Result permits Success, Failure; затем sealed class Failure permits конкретные типы ошибок.`,
            hint2: `Обобщенные запечатанные типы работают как обычные обобщения: sealed interface Result<T> permits Success, Failure. Success должен быть обобщенным: record Success<T>(T value).`,
            whyItMatters: `Запечатанные иерархии необходимы для обработки ошибок, конечных автоматов и моделирования доменов. Они обеспечивают гарантии времени компиляции, что все случаи обработаны, устраняя целые классы ошибок времени выполнения, характерные для традиционных иерархий ООП.

**Продакшен паттерн:**
\`\`\`java
sealed interface OrderStatus permits Draft, Submitted, Processing, Completed, Cancelled {}
final record Draft() implements OrderStatus {}
final record Submitted(LocalDateTime at) implements OrderStatus {}
final record Processing(String warehouseId) implements OrderStatus {}
final record Completed(LocalDateTime at, String trackingNumber) implements OrderStatus {}
final record Cancelled(String reason) implements OrderStatus {}

public String getStatusMessage(OrderStatus status) {
    return switch (status) {
        case Draft d -> "Order is in draft";
        case Submitted s -> "Submitted at " + s.at();
        case Processing p -> "Processing in warehouse " + p.warehouseId();
        case Completed c -> "Completed: " + c.trackingNumber();
        case Cancelled c -> "Cancelled: " + c.reason();
    };
}
\`\`\`

**Практические преимущества:**
- Невозможно забыть обработать статус заказа
- Каждый статус может иметь уникальные данные`
        },
        uz: {
            title: 'Muhrlangan tur iyerarxiyalari',
            solutionCode: `// Umumiy tur parametri bilan muhrlangan interfeys
sealed interface Result<T> permits Success, Failure {
    boolean isSuccess();
    boolean isFailure();
}

// Muvaffaqiyatli natijalar uchun umumiy record
final record Success<T>(T value) implements Result<T> {
    @Override
    public boolean isSuccess() {
        return true;
    }

    @Override
    public boolean isFailure() {
        return false;
    }
}

// Ichki iyerarxiyali xatolar uchun muhrlangan klass
sealed class Failure implements Result {
    @Override
    public boolean isSuccess() {
        return false;
    }

    @Override
    public boolean isFailure() {
        return true;
    }
}

// Ulanish tafsilotlari bilan tarmoq xatosi
final class NetworkError extends Failure {
    private final String url;
    private final int statusCode;

    public NetworkError(String url, int statusCode) {
        this.url = url;
        this.statusCode = statusCode;
    }

    public String getUrl() {
        return url;
    }

    public int getStatusCode() {
        return statusCode;
    }
}

// Maydon tafsilotlari bilan tasdiqlash xatosi
final class ValidationError extends Failure {
    private final String field;
    private final String message;

    public ValidationError(String field, String message) {
        this.field = field;
        this.message = message;
    }

    public String getField() {
        return field;
    }

    public String getMessage() {
        return message;
    }
}

// So'rov tafsilotlari bilan ma'lumotlar bazasi xatosi
final class DatabaseError extends Failure {
    private final String query;
    private final String errorCode;

    public DatabaseError(String query, String errorCode) {
        this.query = query;
        this.errorCode = errorCode;
    }

    public String getQuery() {
        return query;
    }

    public String getErrorCode() {
        return errorCode;
    }
}

public class SealedHierarchies {
    // Turli natijalar bilan API chaqiruvini simulyatsiya qilish
    static Result<String> fetchUser(int id) {
        return switch (id) {
            case 1 -> new Success<>("Alice");
            case 2 -> new NetworkError("https://api.example.com/users/2", 404);
            case 3 -> new ValidationError("id", "Foydalanuvchi ID formati noto'g'ri");
            case 4 -> new DatabaseError("SELECT * FROM users WHERE id = 4", "DB_CONNECTION_LOST");
            default -> new Success<>("Noma'lum foydalanuvchi");
        };
    }

    // Muhrlangan iyerarxiya bilan to'liq pattern matching
    static String handleResult(Result<String> result) {
        // Kompilyator barcha mumkin bo'lgan turlarni biladi
        if (result instanceof Success<String> success) {
            return "Muvaffaqiyat: Foydalanuvchi topildi " + success.value();
        } else if (result instanceof NetworkError networkError) {
            return "Tarmoq xatosi: " + networkError.getStatusCode() + " manzilda " + networkError.getUrl();
        } else if (result instanceof ValidationError validationError) {
            return "'" + validationError.getField() + "' maydonida tasdiqlash xatosi: " + validationError.getMessage();
        } else if (result instanceof DatabaseError dbError) {
            return "Ma'lumotlar bazasi xatosi (" + dbError.getErrorCode() + "): " + dbError.getQuery();
        }
        // default kerak emas - kompilyator to'liqlikni ta'minlaydi
        throw new IllegalStateException("Kutilmagan natija turi");
    }

    public static void main(String[] args) {
        // Barcha stsenariylarni sinovdan o'tkazamiz
        System.out.println(handleResult(fetchUser(1)));	// Muvaffaqiyat
        System.out.println(handleResult(fetchUser(2)));	// Tarmoq xatosi
        System.out.println(handleResult(fetchUser(3)));	// Tasdiqlash xatosi
        System.out.println(handleResult(fetchUser(4)));	// Ma'lumotlar bazasi xatosi

        // Tur xavfsizligi: barcha holatlar qayta ishlanishi kerak
        Result<String> result = fetchUser(1);
        if (result.isSuccess() && result instanceof Success<String> success) {
            System.out.println("");
            System.out.println("Foydalanuvchi ma'lumotlari: " + success.value());
        }
    }
}`,
            description: `Muhrlangan klasslar yordamida murakkab tur iyerarxiyalarini loyihalashni o'rganing.

**Talablar:**
1. Success<T> va Failure ga ruxsat beruvchi muhrlangan Result<T> interfeysini yarating
2. T value maydoni bilan final Success<T> recordini yarating
3. NetworkError, ValidationError va DatabaseError ga ruxsat beruvchi muhrlangan Failure klassini yarating
4. Har bir xato turi uchun mos maydonlar bilan final klasslar yarating
5. Result ga metodlar qo'shing: boolean isSuccess(), boolean isFailure()
6. Result ni qayta ishlash uchun umumiy metodlar yarating: map(), flatMap(), getOrDefault()
7. Result<String> qaytaruvchi simulyatsiya qilingan API chaqiruvi bilan foydalanishni ko'rsating

Muhrlangan iyerarxiyalar maksimal tur xavfsizligi uchun ichki muhrlangan turlar bilan murakkab domen mantiqini modellashtirish imkonini beradi.`,
            hint1: `Ichki muhrlangan turlardan foydalaning: sealed interface Result permits Success, Failure; keyin sealed class Failure permits aniq xato turlari.`,
            hint2: `Umumiy muhrlangan turlar oddiy umumiy turlar kabi ishlaydi: sealed interface Result<T> permits Success, Failure. Success umumiy bo'lishi kerak: record Success<T>(T value).`,
            whyItMatters: `Muhrlangan iyerarxiyalar xatolarni qayta ishlash, holat mashinalari va domen modellash uchun zarurdir. Ular kompilyatsiya vaqtida barcha holatlar qayta ishlanganligini kafolatlaydi va an'anaviy OOP iyerarxiyalarida keng tarqalgan runtime xatolarining butun sinflarini yo'q qiladi.

**Ishlab chiqarish patterni:**
\`\`\`java
sealed interface OrderStatus permits Draft, Submitted, Processing, Completed, Cancelled {}
final record Draft() implements OrderStatus {}
final record Submitted(LocalDateTime at) implements OrderStatus {}
final record Processing(String warehouseId) implements OrderStatus {}
final record Completed(LocalDateTime at, String trackingNumber) implements OrderStatus {}
final record Cancelled(String reason) implements OrderStatus {}

public String getStatusMessage(OrderStatus status) {
    return switch (status) {
        case Draft d -> "Order is in draft";
        case Submitted s -> "Submitted at " + s.at();
        case Processing p -> "Processing in warehouse " + p.warehouseId();
        case Completed c -> "Completed: " + c.trackingNumber();
        case Cancelled c -> "Cancelled: " + c.reason();
    };
}
\`\`\`

**Amaliy foydalari:**
- Buyurtma holatini qayta ishlashni unutish mumkin emas
- Har bir holat o'ziga xos ma'lumotlarga ega bo'lishi mumkin`
        }
    }
};

export default task;
