import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-caffeine-basics',
    title: 'Caffeine Cache Basics',
    difficulty: 'easy',
    tags: ['java', 'caffeine', 'caching', 'performance', 'basics'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Caffeine Cache Basics

Caffeine is a high-performance, near-optimal caching library for Java. It provides an in-memory cache with excellent performance characteristics and a clean API. Understanding basic cache operations like creating a cache, putting values, and retrieving them is fundamental to using Caffeine effectively.

## Requirements:
1. Create a Caffeine cache using the builder pattern:
   1.1. Use \`Caffeine.newBuilder()\` to start building
   1.2. Use \`.build()\` to create the cache instance
   1.3. Store key-value pairs (String keys and String values)

2. Implement basic cache operations:
   2.1. \`put(key, value)\` - Store a value in the cache
   2.2. \`getIfPresent(key)\` - Retrieve a value (returns null if not present)
   2.3. \`get(key, mappingFunction)\` - Get or compute a value
   2.4. \`invalidate(key)\` - Remove an entry from the cache

3. Demonstrate cache behavior:
   3.1. Show that \`getIfPresent()\` returns null for missing keys
   3.2. Use \`get()\` with a mapping function to compute missing values
   3.3. Display cache contents after various operations

4. Handle null values safely with proper checks

## Example Output:
\`\`\`
=== Basic Cache Operations ===
Stored: Java -> High-performance language
Retrieved: High-performance language

=== Cache Miss ===
Missing key returns: null

=== Get with Compute ===
Python computed: Interpreted language
Kotlin computed: JVM language

=== Cache State ===
Java: High-performance language
Python: Interpreted language
Kotlin: JVM language

=== Invalidation ===
After invalidating 'Java': null
Remaining entries: 2
\`\`\``,
    initialCode: `// TODO: Import Caffeine and Cache

public class CaffeineBasics {
    public static void main(String[] args) {
        // TODO: Create a Caffeine cache

        // TODO: Put values into cache

        // TODO: Get values from cache

        // TODO: Get with compute function

        // TODO: Invalidate entries
    }
}`,
    solutionCode: `import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class CaffeineBasics {
    public static void main(String[] args) {
        System.out.println("=== Basic Cache Operations ===");

        // Create a simple cache
        Cache<String, String> cache = Caffeine.newBuilder()
                .build();

        // Put a value into the cache
        cache.put("Java", "High-performance language");
        System.out.println("Stored: Java -> High-performance language");

        // Get a value from the cache
        String value = cache.getIfPresent("Java");
        System.out.println("Retrieved: " + value);

        System.out.println("\\n=== Cache Miss ===");

        // Try to get a non-existent value
        String missing = cache.getIfPresent("Ruby");
        System.out.println("Missing key returns: " + missing);

        System.out.println("\\n=== Get with Compute ===");

        // Get or compute a value
        String python = cache.get("Python", key -> {
            System.out.println("Computing value for: " + key);
            return "Interpreted language";
        });
        System.out.println("Python computed: " + python);

        // Get existing value (won't compute)
        String kotlin = cache.get("Kotlin", key -> {
            System.out.println("Computing value for: " + key);
            return "JVM language";
        });
        System.out.println("Kotlin computed: " + kotlin);

        System.out.println("\\n=== Cache State ===");

        // Display all cached values
        cache.asMap().forEach((key, val) ->
                System.out.println(key + ": " + val)
        );

        System.out.println("\\n=== Invalidation ===");

        // Remove an entry
        cache.invalidate("Java");
        System.out.println("After invalidating 'Java': " + cache.getIfPresent("Java"));
        System.out.println("Remaining entries: " + cache.estimatedSize());
    }
}`,
    hint1: `Use Caffeine.newBuilder().build() to create a simple cache. The cache requires type parameters for keys and values, like Cache<String, String>.`,
    hint2: `The get() method with a mapping function will only compute the value if the key is not already in the cache. This is useful for lazy loading patterns.`,
    whyItMatters: `Caffeine is one of the fastest and most efficient caching libraries available for Java. Understanding its basic operations is essential for improving application performance by reducing expensive computations and database calls. Proper caching can dramatically reduce response times and system load.

**Production Pattern:**
\`\`\`java
// Caching expensive computation results
public class ProductService {
    private final Cache<String, Product> productCache;
    private final ProductRepository repository;

    public ProductService(ProductRepository repository) {
        this.repository = repository;
        this.productCache = Caffeine.newBuilder()
            .maximumSize(10_000)
            .expireAfterWrite(Duration.ofMinutes(10))
            .recordStats()
            .build();
    }

    public Product getProduct(String id) {
        return productCache.get(id, key -> {
            // Load from DB only when not in cache
            return repository.findById(key);
        });
    }

    public void invalidateProduct(String id) {
        productCache.invalidate(id);
    }
}
\`\`\`

**Practical Benefits:**
- Reduces response time by 90%+ for frequently requested data
- Decreases database load
- Built-in statistics for monitoring cache effectiveness`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

// Test1: Verify Cache creation
class Test1 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        assertNotNull(cache);
    }
}

// Test2: Verify putting value into cache
class Test2 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        cache.put("key", "value");
        assertEquals("value", cache.getIfPresent("key"));
    }
}

// Test3: Verify getIfPresent returns null for missing key
class Test3 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        String result = cache.getIfPresent("missing");
        assertNull(result);
    }
}

// Test4: Verify get with compute function
class Test4 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        String result = cache.get("key", k -> "computed-" + k);
        assertEquals("computed-key", result);
    }
}

// Test5: Verify multiple entries can be cached
class Test5 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");

        assertEquals("value1", cache.getIfPresent("key1"));
        assertEquals("value2", cache.getIfPresent("key2"));
        assertEquals("value3", cache.getIfPresent("key3"));
    }
}

// Test6: Verify invalidate removes entry
class Test6 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        cache.put("key", "value");
        assertEquals("value", cache.getIfPresent("key"));

        cache.invalidate("key");
        assertNull(cache.getIfPresent("key"));
    }
}

// Test7: Verify invalidateAll clears cache
class Test7 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        cache.put("key1", "value1");
        cache.put("key2", "value2");

        cache.invalidateAll();
        assertNull(cache.getIfPresent("key1"));
        assertNull(cache.getIfPresent("key2"));
    }
}

// Test8: Verify estimatedSize returns count
class Test8 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");

        assertEquals(3, cache.estimatedSize());
    }
}

// Test9: Verify cache with maximum size
class Test9 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder()
                .maximumSize(2)
                .build();

        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");

        // Cache should have evicted oldest entry
        assertTrue(cache.estimatedSize() <= 2);
    }
}

// Test10: Verify get computes value only once
class Test10 {
    @Test
    public void test() {
        Cache<String, String> cache = Caffeine.newBuilder().build();

        String result1 = cache.get("key", k -> "computed");
        String result2 = cache.get("key", k -> "should-not-compute");

        assertEquals("computed", result1);
        assertEquals("computed", result2); // Should return cached value
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы кэша Caffeine',
            solutionCode: `import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class CaffeineBasics {
    public static void main(String[] args) {
        System.out.println("=== Основные операции с кэшем ===");

        // Создание простого кэша
        Cache<String, String> cache = Caffeine.newBuilder()
                .build();

        // Помещение значения в кэш
        cache.put("Java", "High-performance language");
        System.out.println("Stored: Java -> High-performance language");

        // Получение значения из кэша
        String value = cache.getIfPresent("Java");
        System.out.println("Retrieved: " + value);

        System.out.println("\\n=== Промах кэша ===");

        // Попытка получить несуществующее значение
        String missing = cache.getIfPresent("Ruby");
        System.out.println("Missing key returns: " + missing);

        System.out.println("\\n=== Получение с вычислением ===");

        // Получение или вычисление значения
        String python = cache.get("Python", key -> {
            System.out.println("Computing value for: " + key);
            return "Interpreted language";
        });
        System.out.println("Python computed: " + python);

        // Получение существующего значения (не будет вычислять)
        String kotlin = cache.get("Kotlin", key -> {
            System.out.println("Computing value for: " + key);
            return "JVM language";
        });
        System.out.println("Kotlin computed: " + kotlin);

        System.out.println("\\n=== Состояние кэша ===");

        // Отображение всех кэшированных значений
        cache.asMap().forEach((key, val) ->
                System.out.println(key + ": " + val)
        );

        System.out.println("\\n=== Инвалидация ===");

        // Удаление записи
        cache.invalidate("Java");
        System.out.println("After invalidating 'Java': " + cache.getIfPresent("Java"));
        System.out.println("Remaining entries: " + cache.estimatedSize());
    }
}`,
            description: `# Основы кэша Caffeine

Caffeine - это высокопроизводительная, близкая к оптимальной библиотека кэширования для Java. Она предоставляет кэш в памяти с отличными характеристиками производительности и чистым API. Понимание базовых операций кэширования, таких как создание кэша, помещение значений и их извлечение, является основой для эффективного использования Caffeine.

## Требования:
1. Создайте кэш Caffeine используя паттерн builder:
   1.1. Используйте \`Caffeine.newBuilder()\` для начала построения
   1.2. Используйте \`.build()\` для создания экземпляра кэша
   1.3. Храните пары ключ-значение (String ключи и String значения)

2. Реализуйте базовые операции с кэшем:
   2.1. \`put(key, value)\` - Сохранение значения в кэше
   2.2. \`getIfPresent(key)\` - Извлечение значения (возвращает null, если отсутствует)
   2.3. \`get(key, mappingFunction)\` - Получение или вычисление значения
   2.4. \`invalidate(key)\` - Удаление записи из кэша

3. Продемонстрируйте поведение кэша:
   3.1. Покажите, что \`getIfPresent()\` возвращает null для отсутствующих ключей
   3.2. Используйте \`get()\` с функцией отображения для вычисления отсутствующих значений
   3.3. Отобразите содержимое кэша после различных операций

4. Безопасно обрабатывайте null значения с помощью проверок

## Пример вывода:
\`\`\`
=== Basic Cache Operations ===
Stored: Java -> High-performance language
Retrieved: High-performance language

=== Cache Miss ===
Missing key returns: null

=== Get with Compute ===
Python computed: Interpreted language
Kotlin computed: JVM language

=== Cache State ===
Java: High-performance language
Python: Interpreted language
Kotlin: JVM language

=== Invalidation ===
After invalidating 'Java': null
Remaining entries: 2
\`\`\``,
            hint1: `Используйте Caffeine.newBuilder().build() для создания простого кэша. Кэш требует параметры типа для ключей и значений, например Cache<String, String>.`,
            hint2: `Метод get() с функцией отображения будет вычислять значение только если ключ еще не находится в кэше. Это полезно для паттернов ленивой загрузки.`,
            whyItMatters: `Caffeine - одна из самых быстрых и эффективных библиотек кэширования, доступных для Java. Понимание её базовых операций необходимо для улучшения производительности приложения путем сокращения дорогостоящих вычислений и обращений к базе данных. Правильное кэширование может значительно сократить время отклика и нагрузку на систему.

**Продакшен паттерн:**
\`\`\`java
// Кэширование результатов дорогих вычислений
public class ProductService {
    private final Cache<String, Product> productCache;
    private final ProductRepository repository;

    public ProductService(ProductRepository repository) {
        this.repository = repository;
        this.productCache = Caffeine.newBuilder()
            .maximumSize(10_000)
            .expireAfterWrite(Duration.ofMinutes(10))
            .recordStats()
            .build();
    }

    public Product getProduct(String id) {
        return productCache.get(id, key -> {
            // Загрузка из БД только при отсутствии в кэше
            return repository.findById(key);
        });
    }

    public void invalidateProduct(String id) {
        productCache.invalidate(id);
    }
}
\`\`\`

**Практические преимущества:**
- Сокращение времени отклика на 90%+ для часто запрашиваемых данных
- Снижение нагрузки на базу данных
- Встроенная статистика для мониторинга эффективности кэша`
        },
        uz: {
            title: `Caffeine kesh asoslari`,
            solutionCode: `import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class CaffeineBasics {
    public static void main(String[] args) {
        System.out.println("=== Asosiy kesh operatsiyalari ===");

        // Oddiy kesh yaratish
        Cache<String, String> cache = Caffeine.newBuilder()
                .build();

        // Keshga qiymat qo'yish
        cache.put("Java", "High-performance language");
        System.out.println("Stored: Java -> High-performance language");

        // Keshdan qiymat olish
        String value = cache.getIfPresent("Java");
        System.out.println("Retrieved: " + value);

        System.out.println("\\n=== Kesh missasi ===");

        // Mavjud bo'lmagan qiymatni olishga urinish
        String missing = cache.getIfPresent("Ruby");
        System.out.println("Missing key returns: " + missing);

        System.out.println("\\n=== Hisoblash bilan olish ===");

        // Qiymatni olish yoki hisoblash
        String python = cache.get("Python", key -> {
            System.out.println("Computing value for: " + key);
            return "Interpreted language";
        });
        System.out.println("Python computed: " + python);

        // Mavjud qiymatni olish (hisoblash bo'lmaydi)
        String kotlin = cache.get("Kotlin", key -> {
            System.out.println("Computing value for: " + key);
            return "JVM language";
        });
        System.out.println("Kotlin computed: " + kotlin);

        System.out.println("\\n=== Kesh holati ===");

        // Barcha keshlangan qiymatlarni ko'rsatish
        cache.asMap().forEach((key, val) ->
                System.out.println(key + ": " + val)
        );

        System.out.println("\\n=== Invalidatsiya ===");

        // Yozuvni o'chirish
        cache.invalidate("Java");
        System.out.println("After invalidating 'Java': " + cache.getIfPresent("Java"));
        System.out.println("Remaining entries: " + cache.estimatedSize());
    }
}`,
            description: `# Caffeine kesh asoslari

Caffeine - Java uchun yuqori samarali, deyarli optimal keshlash kutubxonasi. U ajoyib ishlash xususiyatlari va toza API bilan xotiradagi keshni taqdim etadi. Kesh yaratish, qiymatlarni joylashtirish va ularni olish kabi asosiy kesh operatsiyalarini tushunish Caffeine-ni samarali ishlatish uchun asosdir.

## Talablar:
1. Builder naqshidan foydalanib Caffeine kesh yarating:
   1.1. Qurishni boshlash uchun \`Caffeine.newBuilder()\` dan foydalaning
   1.2. Kesh nusxasini yaratish uchun \`.build()\` dan foydalaning
   1.3. Kalit-qiymat juftliklarini saqlang (String kalitlar va String qiymatlar)

2. Asosiy kesh operatsiyalarini amalga oshiring:
   2.1. \`put(key, value)\` - Keshga qiymat saqlash
   2.2. \`getIfPresent(key)\` - Qiymatni olish (mavjud bo'lmasa null qaytaradi)
   2.3. \`get(key, mappingFunction)\` - Qiymatni olish yoki hisoblash
   2.4. \`invalidate(key)\` - Keshdan yozuvni o'chirish

3. Kesh xatti-harakatini namoyish eting:
   3.1. \`getIfPresent()\` mavjud bo'lmagan kalitlar uchun null qaytarishini ko'rsating
   3.2. Mavjud bo'lmagan qiymatlarni hisoblash uchun mapping funktsiyasi bilan \`get()\` dan foydalaning
   3.3. Turli operatsiyalardan keyin kesh mazmunini ko'rsating

4. Null qiymatlarni tekshirish bilan xavfsiz ishlating

## Chiqish namunasi:
\`\`\`
=== Basic Cache Operations ===
Stored: Java -> High-performance language
Retrieved: High-performance language

=== Cache Miss ===
Missing key returns: null

=== Get with Compute ===
Python computed: Interpreted language
Kotlin computed: JVM language

=== Cache State ===
Java: High-performance language
Python: Interpreted language
Kotlin: JVM language

=== Invalidation ===
After invalidating 'Java': null
Remaining entries: 2
\`\`\``,
            hint1: `Oddiy kesh yaratish uchun Caffeine.newBuilder().build() dan foydalaning. Kesh kalitlar va qiymatlar uchun tip parametrlarini talab qiladi, masalan Cache<String, String>.`,
            hint2: `Mapping funktsiyasi bilan get() metodi qiymatni faqat kalit hali keshda bo'lmasa hisoblab chiqadi. Bu lazy loading naqshlari uchun foydalidir.`,
            whyItMatters: `Caffeine Java uchun mavjud bo'lgan eng tez va samarali keshlash kutubxonalaridan biridir. Uning asosiy operatsiyalarini tushunish qimmat hisob-kitoblar va ma'lumotlar bazasi murojatlarini qisqartirish orqali dastur ishlashini yaxshilash uchun juda muhimdir. To'g'ri keshlash javob vaqti va tizim yukini sezilarli darajada qisqartirishi mumkin.

**Ishlab chiqarish patterni:**
\`\`\`java
// Qimmat hisob-kitoblar natijalarini keshlash
public class ProductService {
    private final Cache<String, Product> productCache;
    private final ProductRepository repository;

    public ProductService(ProductRepository repository) {
        this.repository = repository;
        this.productCache = Caffeine.newBuilder()
            .maximumSize(10_000)
            .expireAfterWrite(Duration.ofMinutes(10))
            .recordStats()
            .build();
    }

    public Product getProduct(String id) {
        return productCache.get(id, key -> {
            // Faqat keshda bo'lmaganda ma'lumotlar bazasidan yuklash
            return repository.findById(key);
        });
    }

    public void invalidateProduct(String id) {
        productCache.invalidate(id);
    }
}
\`\`\`

**Amaliy foydalari:**
- Tez-tez so'raladigan ma'lumotlar uchun javob vaqtini 90%+ qisqartirish
- Ma'lumotlar bazasi yukini kamaytirish
- Kesh samaradorligini monitoring qilish uchun o'rnatilgan statistika`
        }
    }
};

export default task;
