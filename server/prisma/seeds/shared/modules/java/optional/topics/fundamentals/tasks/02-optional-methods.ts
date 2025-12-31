import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-optional-methods',
    title: 'Optional Methods',
    difficulty: 'easy',
    tags: ['java', 'optional', 'java8', 'methods', 'null-safety'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Optional Methods

Optional provides several methods to check for values, retrieve them, and provide defaults. Understanding these methods is crucial for safe and effective null-handling. Each method serves a specific purpose in the Optional API.

## Requirements:
1. Demonstrate core Optional methods:
   1.1. \`isPresent()\` - Returns true if value exists
   1.2. \`isEmpty()\` - Returns true if no value (Java 11+)
   1.3. \`get()\` - Returns value or throws exception
   1.4. \`ifPresent(Consumer)\` - Executes action if value exists

2. Show default value methods:
   2.1. \`orElse(T)\` - Returns value or default
   2.2. \`orElseGet(Supplier)\` - Returns value or computed default
   2.3. \`orElseThrow()\` - Returns value or throws exception
   2.4. \`orElseThrow(Supplier)\` - Returns value or throws custom exception

3. Demonstrate the difference between \`orElse\` and \`orElseGet\`:
   3.1. \`orElse\` always evaluates the default
   3.2. \`orElseGet\` only evaluates if Optional is empty

4. Show safe value retrieval patterns

## Example Output:
\`\`\`
=== Checking for Values ===
Has value: true
Is empty: false
Value with get(): Java

=== Conditional Actions ===
Value exists: JAVA

=== Default Values ===
With value: Java
orElse with empty: Default
orElseGet with empty: Computed Default

=== Throwing Exceptions ===
Value: Java
Empty throws: NoSuchElementException
\`\`\``,
    initialCode: `// TODO: Import Optional and other classes

public class OptionalMethods {
    public static void main(String[] args) {
        // TODO: Create Optionals for testing

        // TODO: Demonstrate isPresent() and isEmpty()

        // TODO: Use ifPresent() for conditional actions

        // TODO: Show orElse() and orElseGet()

        // TODO: Demonstrate orElseThrow()
    }
}`,
    solutionCode: `import java.util.Optional;
import java.util.NoSuchElementException;

public class OptionalMethods {
    public static void main(String[] args) {
        System.out.println("=== Checking for Values ===");

        Optional<String> withValue = Optional.of("Java");
        Optional<String> empty = Optional.empty();

        // Check if value is present
        System.out.println("Has value: " + withValue.isPresent());
        System.out.println("Is empty: " + empty.isEmpty());

        // Get value (careful: throws exception if empty)
        if (withValue.isPresent()) {
            System.out.println("Value with get(): " + withValue.get());
        }

        System.out.println("\\n=== Conditional Actions ===");

        // Execute action if value is present
        withValue.ifPresent(value ->
            System.out.println("Value exists: " + value.toUpperCase())
        );

        // Nothing happens with empty Optional
        empty.ifPresent(value ->
            System.out.println("This won't print")
        );

        System.out.println("\\n=== Default Values ===");

        // orElse: return value or default
        System.out.println("With value: " + withValue.orElse("Default"));
        System.out.println("orElse with empty: " + empty.orElse("Default"));

        // orElseGet: return value or compute default (lazy)
        System.out.println("orElseGet with empty: " +
            empty.orElseGet(() -> "Computed Default"));

        // Demonstrate difference between orElse and orElseGet
        System.out.println("\\n=== orElse vs orElseGet ===");

        Optional<String> opt = Optional.of("Present");

        // orElse always evaluates (not efficient)
        String result1 = opt.orElse(expensiveOperation());

        // orElseGet only evaluates if empty (efficient)
        String result2 = opt.orElseGet(() -> expensiveOperation());

        System.out.println("\\n=== Throwing Exceptions ===");

        // orElseThrow: return value or throw NoSuchElementException
        try {
            System.out.println("Value: " + withValue.orElseThrow());
        } catch (NoSuchElementException e) {
            System.out.println("Exception: " + e.getMessage());
        }

        // Throw exception with empty Optional
        try {
            empty.orElseThrow();
        } catch (NoSuchElementException e) {
            System.out.println("Empty throws: " + e.getClass().getSimpleName());
        }

        // Custom exception
        try {
            empty.orElseThrow(() ->
                new IllegalStateException("Value must be present")
            );
        } catch (IllegalStateException e) {
            System.out.println("Custom exception: " + e.getMessage());
        }

        System.out.println("\\n=== Safe Retrieval Pattern ===");

        // Preferred pattern: use orElse or orElseGet
        String safe = empty.orElse("Safe Default");
        System.out.println("Safe retrieval: " + safe);
    }

    private static String expensiveOperation() {
        System.out.println("Expensive operation called!");
        return "Expensive Result";
    }
}`,
    hint1: `Never use get() without checking isPresent() first. Prefer using orElse(), orElseGet(), or orElseThrow() instead.`,
    hint2: `Use orElseGet() instead of orElse() when the default value is expensive to compute. orElseGet() only computes the default if the Optional is empty.`,
    whyItMatters: `Proper use of Optional methods prevents NullPointerException and makes code more expressive. Understanding the difference between orElse and orElseGet is crucial for performance. The ifPresent() method enables clean conditional logic without null checks.

**Production Pattern:**
\`\`\`java
// Safe value retrieval with defaults
String config = getConfig("timeout")
    .orElse("30");  // Simple default

// Lazy computation for expensive defaults
User user = findUserById(id)
    .orElseGet(() -> createDefaultUser());  // Only computed if absent

// Throwing business exceptions
Order order = findOrder(orderId)
    .orElseThrow(() -> new OrderNotFoundException(orderId));

// Conditional actions without null checks
findUser(userId).ifPresent(user -> {
    sendEmail(user);
    logActivity(user);
});
\`\`\`

**Practical Benefits:**
- orElseGet avoids unnecessary computation for defaults
- orElseThrow provides clear business exception context
- ifPresent eliminates explicit null checks in code`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Optional;
import java.util.NoSuchElementException;

// Test 1: isPresent returns true for value
class Test1 {
    @Test
    void testIsPresentWithValue() {
        Optional<String> opt = Optional.of("Java");
        assertTrue(opt.isPresent());
    }
}

// Test 2: isEmpty returns true for empty
class Test2 {
    @Test
    void testIsEmptyWithEmpty() {
        Optional<String> opt = Optional.empty();
        assertTrue(opt.isEmpty());
    }
}

// Test 3: orElse returns value when present
class Test3 {
    @Test
    void testOrElseWithValue() {
        Optional<String> opt = Optional.of("Java");
        assertEquals("Java", opt.orElse("Default"));
    }
}

// Test 4: orElse returns default when empty
class Test4 {
    @Test
    void testOrElseWithEmpty() {
        Optional<String> opt = Optional.empty();
        assertEquals("Default", opt.orElse("Default"));
    }
}

// Test 5: orElseGet returns computed default when empty
class Test5 {
    @Test
    void testOrElseGetWithEmpty() {
        Optional<String> opt = Optional.empty();
        assertEquals("Computed", opt.orElseGet(() -> "Computed"));
    }
}

// Test 6: orElseThrow throws when empty
class Test6 {
    @Test
    void testOrElseThrowWithEmpty() {
        Optional<String> opt = Optional.empty();
        assertThrows(NoSuchElementException.class, () -> opt.orElseThrow());
    }
}

// Test 7: orElseThrow returns value when present
class Test7 {
    @Test
    void testOrElseThrowWithValue() {
        Optional<String> opt = Optional.of("Java");
        assertEquals("Java", opt.orElseThrow());
    }
}

// Test 8: ifPresent executes when value present
class Test8 {
    @Test
    void testIfPresentExecutes() {
        Optional<String> opt = Optional.of("Java");
        StringBuilder result = new StringBuilder();
        opt.ifPresent(v -> result.append(v.toUpperCase()));
        assertEquals("JAVA", result.toString());
    }
}

// Test 9: ifPresent does nothing when empty
class Test9 {
    @Test
    void testIfPresentDoesNothing() {
        Optional<String> opt = Optional.empty();
        StringBuilder result = new StringBuilder("unchanged");
        opt.ifPresent(v -> result.setLength(0));
        assertEquals("unchanged", result.toString());
    }
}

// Test 10: orElseThrow with custom exception
class Test10 {
    @Test
    void testOrElseThrowCustomException() {
        Optional<String> opt = Optional.empty();
        assertThrows(IllegalStateException.class, () ->
            opt.orElseThrow(() -> new IllegalStateException("Missing"))
        );
    }
}`,
    translations: {
        ru: {
            title: 'Методы Optional',
            solutionCode: `import java.util.Optional;
import java.util.NoSuchElementException;

public class OptionalMethods {
    public static void main(String[] args) {
        System.out.println("=== Проверка наличия значений ===");

        Optional<String> withValue = Optional.of("Java");
        Optional<String> empty = Optional.empty();

        // Проверка наличия значения
        System.out.println("Has value: " + withValue.isPresent());
        System.out.println("Is empty: " + empty.isEmpty());

        // Получение значения (осторожно: бросает исключение если пусто)
        if (withValue.isPresent()) {
            System.out.println("Value with get(): " + withValue.get());
        }

        System.out.println("\\n=== Условные действия ===");

        // Выполнение действия если значение присутствует
        withValue.ifPresent(value ->
            System.out.println("Value exists: " + value.toUpperCase())
        );

        // Ничего не происходит с пустым Optional
        empty.ifPresent(value ->
            System.out.println("This won't print")
        );

        System.out.println("\\n=== Значения по умолчанию ===");

        // orElse: вернуть значение или значение по умолчанию
        System.out.println("With value: " + withValue.orElse("Default"));
        System.out.println("orElse with empty: " + empty.orElse("Default"));

        // orElseGet: вернуть значение или вычислить по умолчанию (ленивое)
        System.out.println("orElseGet with empty: " +
            empty.orElseGet(() -> "Computed Default"));

        // Демонстрация разницы между orElse и orElseGet
        System.out.println("\\n=== orElse vs orElseGet ===");

        Optional<String> opt = Optional.of("Present");

        // orElse всегда вычисляет (не эффективно)
        String result1 = opt.orElse(expensiveOperation());

        // orElseGet вычисляет только если пусто (эффективно)
        String result2 = opt.orElseGet(() -> expensiveOperation());

        System.out.println("\\n=== Выброс исключений ===");

        // orElseThrow: вернуть значение или выбросить NoSuchElementException
        try {
            System.out.println("Value: " + withValue.orElseThrow());
        } catch (NoSuchElementException e) {
            System.out.println("Exception: " + e.getMessage());
        }

        // Выброс исключения с пустым Optional
        try {
            empty.orElseThrow();
        } catch (NoSuchElementException e) {
            System.out.println("Empty throws: " + e.getClass().getSimpleName());
        }

        // Пользовательское исключение
        try {
            empty.orElseThrow(() ->
                new IllegalStateException("Value must be present")
            );
        } catch (IllegalStateException e) {
            System.out.println("Custom exception: " + e.getMessage());
        }

        System.out.println("\\n=== Паттерн безопасного получения ===");

        // Предпочтительный паттерн: использовать orElse или orElseGet
        String safe = empty.orElse("Safe Default");
        System.out.println("Safe retrieval: " + safe);
    }

    private static String expensiveOperation() {
        System.out.println("Expensive operation called!");
        return "Expensive Result";
    }
}`,
            description: `# Методы Optional

Optional предоставляет несколько методов для проверки значений, их извлечения и предоставления значений по умолчанию. Понимание этих методов критически важно для безопасной и эффективной обработки null. Каждый метод служит определенной цели в API Optional.

## Требования:
1. Продемонстрируйте основные методы Optional:
   1.1. \`isPresent()\` - Возвращает true если значение существует
   1.2. \`isEmpty()\` - Возвращает true если нет значения (Java 11+)
   1.3. \`get()\` - Возвращает значение или бросает исключение
   1.4. \`ifPresent(Consumer)\` - Выполняет действие если значение существует

2. Покажите методы значений по умолчанию:
   2.1. \`orElse(T)\` - Возвращает значение или значение по умолчанию
   2.2. \`orElseGet(Supplier)\` - Возвращает значение или вычисленное по умолчанию
   2.3. \`orElseThrow()\` - Возвращает значение или бросает исключение
   2.4. \`orElseThrow(Supplier)\` - Возвращает значение или бросает пользовательское исключение

3. Продемонстрируйте разницу между \`orElse\` и \`orElseGet\`:
   3.1. \`orElse\` всегда вычисляет значение по умолчанию
   3.2. \`orElseGet\` вычисляет только если Optional пуст

4. Покажите паттерны безопасного извлечения значений

## Пример вывода:
\`\`\`
=== Checking for Values ===
Has value: true
Is empty: false
Value with get(): Java

=== Conditional Actions ===
Value exists: JAVA

=== Default Values ===
With value: Java
orElse with empty: Default
orElseGet with empty: Computed Default

=== Throwing Exceptions ===
Value: Java
Empty throws: NoSuchElementException
\`\`\``,
            hint1: `Никогда не используйте get() без предварительной проверки isPresent(). Предпочитайте использовать orElse(), orElseGet() или orElseThrow().`,
            hint2: `Используйте orElseGet() вместо orElse(), когда значение по умолчанию дорого вычислять. orElseGet() вычисляет значение только если Optional пуст.`,
            whyItMatters: `Правильное использование методов Optional предотвращает NullPointerException и делает код более выразительным. Понимание разницы между orElse и orElseGet критично для производительности. Метод ifPresent() обеспечивает чистую условную логику без проверок на null.

**Продакшен паттерн:**
\`\`\`java
// Безопасное получение значений со значениями по умолчанию
String config = getConfig("timeout")
    .orElse("30");  // Простое значение по умолчанию

// Ленивое вычисление для дорогих значений по умолчанию
User user = findUserById(id)
    .orElseGet(() -> createDefaultUser());  // Вычисляется только если отсутствует

// Выброс бизнес-исключений
Order order = findOrder(orderId)
    .orElseThrow(() -> new OrderNotFoundException(orderId));

// Условные действия без проверок на null
findUser(userId).ifPresent(user -> {
    sendEmail(user);
    logActivity(user);
});
\`\`\`

**Практические преимущества:**
- orElseGet избегает ненужных вычислений для значений по умолчанию
- orElseThrow предоставляет четкий контекст бизнес-исключений
- ifPresent устраняет явные проверки на null в коде`
        },
        uz: {
            title: `Optional metodlari`,
            solutionCode: `import java.util.Optional;
import java.util.NoSuchElementException;

public class OptionalMethods {
    public static void main(String[] args) {
        System.out.println("=== Qiymatlar mavjudligini tekshirish ===");

        Optional<String> withValue = Optional.of("Java");
        Optional<String> empty = Optional.empty();

        // Qiymat mavjudligini tekshirish
        System.out.println("Has value: " + withValue.isPresent());
        System.out.println("Is empty: " + empty.isEmpty());

        // Qiymatni olish (ehtiyot: bo'sh bo'lsa istisno tashlaydi)
        if (withValue.isPresent()) {
            System.out.println("Value with get(): " + withValue.get());
        }

        System.out.println("\\n=== Shartli harakatlar ===");

        // Qiymat mavjud bo'lsa harakatni bajarish
        withValue.ifPresent(value ->
            System.out.println("Value exists: " + value.toUpperCase())
        );

        // Bo'sh Optional bilan hech narsa bo'lmaydi
        empty.ifPresent(value ->
            System.out.println("This won't print")
        );

        System.out.println("\\n=== Standart qiymatlar ===");

        // orElse: qiymat yoki standart qiymatni qaytarish
        System.out.println("With value: " + withValue.orElse("Default"));
        System.out.println("orElse with empty: " + empty.orElse("Default"));

        // orElseGet: qiymat yoki hisoblangan standartni qaytarish (kech)
        System.out.println("orElseGet with empty: " +
            empty.orElseGet(() -> "Computed Default"));

        // orElse va orElseGet orasidagi farqni ko'rsatish
        System.out.println("\\n=== orElse vs orElseGet ===");

        Optional<String> opt = Optional.of("Present");

        // orElse har doim hisoblaydi (samarali emas)
        String result1 = opt.orElse(expensiveOperation());

        // orElseGet faqat bo'sh bo'lsa hisoblaydi (samarali)
        String result2 = opt.orElseGet(() -> expensiveOperation());

        System.out.println("\\n=== Istisno tashlash ===");

        // orElseThrow: qiymat yoki NoSuchElementException tashlash
        try {
            System.out.println("Value: " + withValue.orElseThrow());
        } catch (NoSuchElementException e) {
            System.out.println("Exception: " + e.getMessage());
        }

        // Bo'sh Optional bilan istisno tashlash
        try {
            empty.orElseThrow();
        } catch (NoSuchElementException e) {
            System.out.println("Empty throws: " + e.getClass().getSimpleName());
        }

        // Maxsus istisno
        try {
            empty.orElseThrow(() ->
                new IllegalStateException("Value must be present")
            );
        } catch (IllegalStateException e) {
            System.out.println("Custom exception: " + e.getMessage());
        }

        System.out.println("\\n=== Xavfsiz olish naqshi ===");

        // Afzal naqsh: orElse yoki orElseGet dan foydalanish
        String safe = empty.orElse("Safe Default");
        System.out.println("Safe retrieval: " + safe);
    }

    private static String expensiveOperation() {
        System.out.println("Expensive operation called!");
        return "Expensive Result";
    }
}`,
            description: `# Optional metodlari

Optional qiymatlarni tekshirish, ularni olish va standart qiymatlarni taqdim etish uchun bir nechta metodlarni taqdim etadi. Bu metodlarni tushunish xavfsiz va samarali null bilan ishlash uchun juda muhim. Har bir metod Optional API da ma'lum bir maqsadga xizmat qiladi.

## Talablar:
1. Optional asosiy metodlarini namoyish eting:
   1.1. \`isPresent()\` - Qiymat mavjud bo'lsa true qaytaradi
   1.2. \`isEmpty()\` - Qiymat yo'q bo'lsa true qaytaradi (Java 11+)
   1.3. \`get()\` - Qiymat yoki istisno qaytaradi
   1.4. \`ifPresent(Consumer)\` - Qiymat mavjud bo'lsa harakatni bajaradi

2. Standart qiymat metodlarini ko'rsating:
   2.1. \`orElse(T)\` - Qiymat yoki standart qaytaradi
   2.2. \`orElseGet(Supplier)\` - Qiymat yoki hisoblangan standart qaytaradi
   2.3. \`orElseThrow()\` - Qiymat yoki istisno tashlaydi
   2.4. \`orElseThrow(Supplier)\` - Qiymat yoki maxsus istisno tashlaydi

3. \`orElse\` va \`orElseGet\` orasidagi farqni ko'rsating:
   3.1. \`orElse\` har doim standartni hisoblaydi
   3.2. \`orElseGet\` faqat Optional bo'sh bo'lsa hisoblaydi

4. Xavfsiz qiymat olish naqshlarini ko'rsating

## Chiqish namunasi:
\`\`\`
=== Checking for Values ===
Has value: true
Is empty: false
Value with get(): Java

=== Conditional Actions ===
Value exists: JAVA

=== Default Values ===
With value: Java
orElse with empty: Default
orElseGet with empty: Computed Default

=== Throwing Exceptions ===
Value: Java
Empty throws: NoSuchElementException
\`\`\``,
            hint1: `Hech qachon isPresent() ni tekshirmasdan get() dan foydalanmang. O'rniga orElse(), orElseGet() yoki orElseThrow() dan foydalaning.`,
            hint2: `Standart qiymatni hisoblash qimmat bo'lsa, orElse() o'rniga orElseGet() dan foydalaning. orElseGet() faqat Optional bo'sh bo'lsa hisoblaydi.`,
            whyItMatters: `Optional metodlarini to'g'ri ishlatish NullPointerException oldini oladi va kodni yanada ifodali qiladi. orElse va orElseGet o'rtasidagi farqni tushunish samaradorlik uchun juda muhim. ifPresent() metodi null tekshiruvisiz toza shartli mantiqni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Standart qiymatlar bilan xavfsiz qiymat olish
String config = getConfig("timeout")
    .orElse("30");  // Oddiy standart qiymat

// Qimmat standart qiymatlar uchun kech hisoblash
User user = findUserById(id)
    .orElseGet(() -> createDefaultUser());  // Faqat yo'q bo'lganda hisoblanadi

// Biznes istisnolarini tashlash
Order order = findOrder(orderId)
    .orElseThrow(() -> new OrderNotFoundException(orderId));

// Null tekshiruvisiz shartli harakatlar
findUser(userId).ifPresent(user -> {
    sendEmail(user);
    logActivity(user);
});
\`\`\`

**Amaliy foydalari:**
- orElseGet standart qiymatlar uchun keraksiz hisoblashlardan qochadi
- orElseThrow biznes istisnolarining aniq kontekstini beradi
- ifPresent kodda aniq null tekshiruvlarini bartaraf etadi`
        }
    }
};

export default task;
