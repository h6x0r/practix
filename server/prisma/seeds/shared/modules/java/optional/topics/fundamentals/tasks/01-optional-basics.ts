import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-optional-basics',
    title: 'Optional Basics',
    difficulty: 'easy',
    tags: ['java', 'optional', 'java8', 'null-safety', 'basics'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Optional Basics

The Optional class is a container object that may or may not contain a non-null value. It's designed to help prevent NullPointerException and make null-handling more explicit and safer. Optional provides methods to create instances with or without values.

## Requirements:
1. Create Optional instances using different methods:
   1.1. \`Optional.of(value)\` - Creates Optional with non-null value
   1.2. \`Optional.ofNullable(value)\` - Creates Optional that may be null
   1.3. \`Optional.empty()\` - Creates empty Optional

2. Demonstrate when to use each creation method:
   2.1. Use \`of()\` when value is guaranteed to be non-null
   2.2. Use \`ofNullable()\` when value might be null
   2.3. Use \`empty()\` to explicitly represent absence

3. Show what happens with null values:
   3.1. \`Optional.of(null)\` throws NullPointerException
   3.2. \`Optional.ofNullable(null)\` creates empty Optional

4. Compare Optional instances with equals()

## Example Output:
\`\`\`
=== Creating Optionals ===
Optional.of(): Optional[Java]
Optional.ofNullable() with value: Optional[Python]
Optional.ofNullable() with null: Optional.empty
Optional.empty(): Optional.empty

=== Using Optional Values ===
Has value: true
Value: Java

Empty Optional:
Has value: false

=== Comparing Optionals ===
Same value optionals are equal: true
Empty optionals are equal: true
\`\`\``,
    initialCode: `// TODO: Import Optional

public class OptionalBasics {
    public static void main(String[] args) {
        // TODO: Create Optional with of()

        // TODO: Create Optional with ofNullable()

        // TODO: Create empty Optional

        // TODO: Demonstrate null handling

        // TODO: Compare Optional instances
    }
}`,
    solutionCode: `import java.util.Optional;

public class OptionalBasics {
    public static void main(String[] args) {
        System.out.println("=== Creating Optionals ===");

        // Create Optional with non-null value
        Optional<String> optionalOf = Optional.of("Java");
        System.out.println("Optional.of(): " + optionalOf);

        // Create Optional that may be null (safe)
        Optional<String> optionalOfNullable = Optional.ofNullable("Python");
        System.out.println("Optional.ofNullable() with value: " + optionalOfNullable);

        // Create Optional with null value (safe)
        Optional<String> optionalNull = Optional.ofNullable(null);
        System.out.println("Optional.ofNullable() with null: " + optionalNull);

        // Create empty Optional explicitly
        Optional<String> optionalEmpty = Optional.empty();
        System.out.println("Optional.empty(): " + optionalEmpty);

        System.out.println("\\n=== Using Optional Values ===");

        // Check if Optional has value
        if (optionalOf.isPresent()) {
            System.out.println("Has value: " + optionalOf.isPresent());
            System.out.println("Value: " + optionalOf.get());
        }

        System.out.println("\\nEmpty Optional:");
        System.out.println("Has value: " + optionalEmpty.isPresent());

        System.out.println("\\n=== Comparing Optionals ===");

        // Compare Optionals
        Optional<String> opt1 = Optional.of("Test");
        Optional<String> opt2 = Optional.of("Test");
        System.out.println("Same value optionals are equal: " + opt1.equals(opt2));

        Optional<String> empty1 = Optional.empty();
        Optional<String> empty2 = Optional.empty();
        System.out.println("Empty optionals are equal: " + empty1.equals(empty2));

        // Demonstrate NullPointerException with of()
        try {
            Optional<String> willThrow = Optional.of(null);
        } catch (NullPointerException e) {
            System.out.println("\\nOptional.of(null) throws: " + e.getClass().getSimpleName());
        }
    }
}`,
    hint1: `Use Optional.of() only when you're certain the value is not null. For potentially null values, always use Optional.ofNullable().`,
    hint2: `Optional.empty() is equivalent to Optional.ofNullable(null) but more explicit about intent. Use it when you want to clearly indicate absence of value.`,
    whyItMatters: `Optional helps eliminate NullPointerException by making null-handling explicit. It forces developers to think about the absence of values and handle them properly. This leads to more robust, self-documenting code and is a fundamental pattern in modern Java development.

**Production Pattern:**
\`\`\`java
// Creating Optional - choosing the right method
Optional<String> guaranteed = Optional.of("never null");
Optional<String> maybeNull = Optional.ofNullable(getValue());
Optional<String> empty = Optional.empty();

// Checking for value presence
if (optional.isPresent()) {
    String value = optional.get();
}

// Optional.of(null) throws NullPointerException
// Optional.ofNullable(null) returns Optional.empty
\`\`\`

**Practical Benefits:**
- Makes null-handling explicit and mandatory
- Prevents NullPointerException at compile time
- Self-documenting code - Optional signals possible absence
- Fundamental pattern in Stream API and modern Java libraries`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Optional;

// Test 1: Optional.of creates Optional with value
class Test1 {
    @Test
    void testOptionalOfCreatesWithValue() {
        Optional<String> opt = Optional.of("Java");
        assertTrue(opt.isPresent());
        assertEquals("Java", opt.get());
    }
}

// Test 2: Optional.ofNullable with value works
class Test2 {
    @Test
    void testOptionalOfNullableWithValue() {
        Optional<String> opt = Optional.ofNullable("Python");
        assertTrue(opt.isPresent());
        assertEquals("Python", opt.get());
    }
}

// Test 3: Optional.ofNullable with null creates empty
class Test3 {
    @Test
    void testOptionalOfNullableWithNull() {
        Optional<String> opt = Optional.ofNullable(null);
        assertTrue(opt.isEmpty());
    }
}

// Test 4: Optional.empty creates empty Optional
class Test4 {
    @Test
    void testOptionalEmptyCreatesEmpty() {
        Optional<String> opt = Optional.empty();
        assertTrue(opt.isEmpty());
        assertFalse(opt.isPresent());
    }
}

// Test 5: Optional.of(null) throws NullPointerException
class Test5 {
    @Test
    void testOptionalOfNullThrows() {
        assertThrows(NullPointerException.class, () -> {
            Optional.of(null);
        });
    }
}

// Test 6: isPresent returns true for value
class Test6 {
    @Test
    void testIsPresentReturnsTrue() {
        Optional<String> opt = Optional.of("Test");
        assertTrue(opt.isPresent());
    }
}

// Test 7: isPresent returns false for empty
class Test7 {
    @Test
    void testIsPresentReturnsFalse() {
        Optional<String> opt = Optional.empty();
        assertFalse(opt.isPresent());
    }
}

// Test 8: Same value Optionals are equal
class Test8 {
    @Test
    void testSameValueOptionalEquals() {
        Optional<String> opt1 = Optional.of("Test");
        Optional<String> opt2 = Optional.of("Test");
        assertEquals(opt1, opt2);
    }
}

// Test 9: Empty Optionals are equal
class Test9 {
    @Test
    void testEmptyOptionalEquals() {
        Optional<String> opt1 = Optional.empty();
        Optional<String> opt2 = Optional.empty();
        assertEquals(opt1, opt2);
    }
}

// Test 10: get() returns value when present
class Test10 {
    @Test
    void testGetReturnsValue() {
        Optional<Integer> opt = Optional.of(42);
        assertEquals(42, opt.get());
    }
}`,
    translations: {
        ru: {
            title: 'Основы Optional',
            solutionCode: `import java.util.Optional;

public class OptionalBasics {
    public static void main(String[] args) {
        System.out.println("=== Создание Optional ===");

        // Создание Optional с не-null значением
        Optional<String> optionalOf = Optional.of("Java");
        System.out.println("Optional.of(): " + optionalOf);

        // Создание Optional, который может быть null (безопасно)
        Optional<String> optionalOfNullable = Optional.ofNullable("Python");
        System.out.println("Optional.ofNullable() with value: " + optionalOfNullable);

        // Создание Optional с null значением (безопасно)
        Optional<String> optionalNull = Optional.ofNullable(null);
        System.out.println("Optional.ofNullable() with null: " + optionalNull);

        // Явное создание пустого Optional
        Optional<String> optionalEmpty = Optional.empty();
        System.out.println("Optional.empty(): " + optionalEmpty);

        System.out.println("\\n=== Использование Optional значений ===");

        // Проверка наличия значения в Optional
        if (optionalOf.isPresent()) {
            System.out.println("Has value: " + optionalOf.isPresent());
            System.out.println("Value: " + optionalOf.get());
        }

        System.out.println("\\nEmpty Optional:");
        System.out.println("Has value: " + optionalEmpty.isPresent());

        System.out.println("\\n=== Сравнение Optional ===");

        // Сравнение Optional
        Optional<String> opt1 = Optional.of("Test");
        Optional<String> opt2 = Optional.of("Test");
        System.out.println("Same value optionals are equal: " + opt1.equals(opt2));

        Optional<String> empty1 = Optional.empty();
        Optional<String> empty2 = Optional.empty();
        System.out.println("Empty optionals are equal: " + empty1.equals(empty2));

        // Демонстрация NullPointerException с of()
        try {
            Optional<String> willThrow = Optional.of(null);
        } catch (NullPointerException e) {
            System.out.println("\\nOptional.of(null) throws: " + e.getClass().getSimpleName());
        }
    }
}`,
            description: `# Основы Optional

Класс Optional - это контейнерный объект, который может содержать или не содержать не-null значение. Он разработан для предотвращения NullPointerException и делает обработку null более явной и безопасной. Optional предоставляет методы для создания экземпляров со значениями или без них.

## Требования:
1. Создайте экземпляры Optional различными способами:
   1.1. \`Optional.of(value)\` - Создает Optional с не-null значением
   1.2. \`Optional.ofNullable(value)\` - Создает Optional, который может быть null
   1.3. \`Optional.empty()\` - Создает пустой Optional

2. Продемонстрируйте, когда использовать каждый метод создания:
   2.1. Используйте \`of()\`, когда значение гарантированно не-null
   2.2. Используйте \`ofNullable()\`, когда значение может быть null
   2.3. Используйте \`empty()\` для явного представления отсутствия

3. Покажите, что происходит с null значениями:
   3.1. \`Optional.of(null)\` бросает NullPointerException
   3.2. \`Optional.ofNullable(null)\` создает пустой Optional

4. Сравните экземпляры Optional с помощью equals()

## Пример вывода:
\`\`\`
=== Creating Optionals ===
Optional.of(): Optional[Java]
Optional.ofNullable() with value: Optional[Python]
Optional.ofNullable() with null: Optional.empty
Optional.empty(): Optional.empty

=== Using Optional Values ===
Has value: true
Value: Java

Empty Optional:
Has value: false

=== Comparing Optionals ===
Same value optionals are equal: true
Empty optionals are equal: true
\`\`\``,
            hint1: `Используйте Optional.of() только когда вы уверены, что значение не null. Для потенциально null значений всегда используйте Optional.ofNullable().`,
            hint2: `Optional.empty() эквивалентен Optional.ofNullable(null), но более явно указывает на намерение. Используйте его, когда хотите четко указать на отсутствие значения.`,
            whyItMatters: `Optional помогает устранить NullPointerException, делая обработку null явной. Он заставляет разработчиков думать об отсутствии значений и правильно их обрабатывать. Это приводит к более надежному, самодокументирующемуся коду и является фундаментальным паттерном в современной разработке на Java.

**Продакшен паттерн:**
\`\`\`java
// Создание Optional - выбор правильного метода
Optional<String> guaranteed = Optional.of("never null");
Optional<String> maybeNull = Optional.ofNullable(getValue());
Optional<String> empty = Optional.empty();

// Проверка наличия значения
if (optional.isPresent()) {
    String value = optional.get();
}

// Optional.of(null) бросает NullPointerException
// Optional.ofNullable(null) возвращает Optional.empty
\`\`\`

**Практические преимущества:**
- Делает null-обработку явной и обязательной
- Предотвращает NullPointerException во время компиляции
- Самодокументирующийся код - Optional сигнализирует о возможном отсутствии
- Фундаментальный паттерн в Stream API и современных Java библиотеках`
        },
        uz: {
            title: `Optional asoslari`,
            solutionCode: `import java.util.Optional;

public class OptionalBasics {
    public static void main(String[] args) {
        System.out.println("=== Optional yaratish ===");

        // Null bo'lmagan qiymat bilan Optional yaratish
        Optional<String> optionalOf = Optional.of("Java");
        System.out.println("Optional.of(): " + optionalOf);

        // Null bo'lishi mumkin bo'lgan Optional yaratish (xavfsiz)
        Optional<String> optionalOfNullable = Optional.ofNullable("Python");
        System.out.println("Optional.ofNullable() with value: " + optionalOfNullable);

        // Null qiymat bilan Optional yaratish (xavfsiz)
        Optional<String> optionalNull = Optional.ofNullable(null);
        System.out.println("Optional.ofNullable() with null: " + optionalNull);

        // Bo'sh Optional yaratish
        Optional<String> optionalEmpty = Optional.empty();
        System.out.println("Optional.empty(): " + optionalEmpty);

        System.out.println("\\n=== Optional qiymatlardan foydalanish ===");

        // Optional qiymat borligini tekshirish
        if (optionalOf.isPresent()) {
            System.out.println("Has value: " + optionalOf.isPresent());
            System.out.println("Value: " + optionalOf.get());
        }

        System.out.println("\\nEmpty Optional:");
        System.out.println("Has value: " + optionalEmpty.isPresent());

        System.out.println("\\n=== Optional taqqoslash ===");

        // Optionalni taqqoslash
        Optional<String> opt1 = Optional.of("Test");
        Optional<String> opt2 = Optional.of("Test");
        System.out.println("Same value optionals are equal: " + opt1.equals(opt2));

        Optional<String> empty1 = Optional.empty();
        Optional<String> empty2 = Optional.empty();
        System.out.println("Empty optionals are equal: " + empty1.equals(empty2));

        // of() bilan NullPointerException namoyishi
        try {
            Optional<String> willThrow = Optional.of(null);
        } catch (NullPointerException e) {
            System.out.println("\\nOptional.of(null) throws: " + e.getClass().getSimpleName());
        }
    }
}`,
            description: `# Optional asoslari

Optional klassi null bo'lmagan qiymatni o'z ichiga olishi yoki olmasligi mumkin bo'lgan konteyner obyekti. U NullPointerException oldini olish va null bilan ishlashni aniqroq va xavfsizroq qilish uchun mo'ljallangan. Optional qiymatlar bilan yoki ularsiz ekzemplyar yaratish usullarini taqdim etadi.

## Talablar:
1. Turli usullar bilan Optional ekzemplyarlarini yarating:
   1.1. \`Optional.of(value)\` - Null bo'lmagan qiymat bilan Optional yaratadi
   1.2. \`Optional.ofNullable(value)\` - Null bo'lishi mumkin bo'lgan Optional yaratadi
   1.3. \`Optional.empty()\` - Bo'sh Optional yaratadi

2. Har bir yaratish usulini qachon ishlatishni namoyish eting:
   2.1. Qiymat kafolatlangan null bo'lmaganda \`of()\` dan foydalaning
   2.2. Qiymat null bo'lishi mumkin bo'lsa \`ofNullable()\` dan foydalaning
   2.3. Yo'qligini aniq ko'rsatish uchun \`empty()\` dan foydalaning

3. Null qiymatlar bilan nima sodir bo'lishini ko'rsating:
   3.1. \`Optional.of(null)\` NullPointerException tashlaydi
   3.2. \`Optional.ofNullable(null)\` bo'sh Optional yaratadi

4. Optional ekzemplyarlarini equals() bilan solishtiring

## Chiqish namunasi:
\`\`\`
=== Creating Optionals ===
Optional.of(): Optional[Java]
Optional.ofNullable() with value: Optional[Python]
Optional.ofNullable() with null: Optional.empty
Optional.empty(): Optional.empty

=== Using Optional Values ===
Has value: true
Value: Java

Empty Optional:
Has value: false

=== Comparing Optionals ===
Same value optionals are equal: true
Empty optionals are equal: true
\`\`\``,
            hint1: `Optional.of() ni faqat qiymat null emasligiga ishonchingiz komil bo'lganda ishlating. Null bo'lishi mumkin bo'lgan qiymatlar uchun doim Optional.ofNullable() dan foydalaning.`,
            hint2: `Optional.empty() Optional.ofNullable(null) ga ekvivalent, lekin niyatni aniqroq bildiradi. Qiymat yo'qligini aniq ko'rsatmoqchi bo'lsangiz, undan foydalaning.`,
            whyItMatters: `Optional null bilan ishlashni aniq qilish orqali NullPointerException ni bartaraf etishga yordam beradi. U dasturchilarga qiymat yo'qligi haqida o'ylashga va ularni to'g'ri ishlashga majbur qiladi. Bu yanada ishonchli, o'z-o'zini hujjatlashtiradigan kodga olib keladi va zamonaviy Java dasturlashda fundamental naqsh hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Optional yaratish - to'g'ri metodini tanlash
Optional<String> guaranteed = Optional.of("never null");
Optional<String> maybeNull = Optional.ofNullable(getValue());
Optional<String> empty = Optional.empty();

// Qiymat mavjudligini tekshirish
if (optional.isPresent()) {
    String value = optional.get();
}

// Optional.of(null) NullPointerException tashlaydi
// Optional.ofNullable(null) Optional.empty qaytaradi
\`\`\`

**Amaliy foydalari:**
- Null bilan ishlashni aniq va majburiy qiladi
- Kompilyatsiya vaqtida NullPointerException oldini oladi
- O'z-o'zini hujjatlashtiradigan kod - Optional yo'qolish imkoniyatini bildiradi
- Stream API va zamonaviy Java kutubxonalarida fundamental naqsh`
        }
    }
};

export default task;
