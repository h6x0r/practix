import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-cache-loading',
    title: 'Cache Loading Strategies',
    difficulty: 'medium',
    tags: ['java', 'caffeine', 'caching', 'loading', 'cache-loader'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Cache Loading Strategies

LoadingCache automatically loads values when they're not present in the cache. This eliminates the need to check for null values and manually compute missing entries. CacheLoader defines the logic for loading values, making cache access uniform and error-free.

## Requirements:
1. Create a LoadingCache with a CacheLoader:
   1.1. Implement CacheLoader interface with load() method
   1.2. Use \`Caffeine.newBuilder().build(loader)\` to create LoadingCache
   1.3. Define the loading logic for missing cache entries

2. Demonstrate LoadingCache operations:
   2.1. Use \`get(key)\` - automatically loads missing values
   2.2. Use \`getAll(keys)\` - batch load multiple keys
   2.3. Show that values are loaded only once per key

3. Handle loading exceptions:
   3.1. Show what happens when loading fails
   3.2. Catch and handle exceptions from the loader

4. Compare LoadingCache with regular Cache:
   4.1. Highlight the difference in null handling
   4.2. Show automatic computation vs manual computation

## Example Output:
\`\`\`
=== LoadingCache with CacheLoader ===
Loading: user:1
User loaded: User{id=1, name='Alice'}
Cached retrieval: User{id=1, name='Alice'}

Loading: user:2
User loaded: User{id=2, name='Bob'}

=== Batch Loading ===
Loading: user:3
Loading: user:4
Loading: user:5
Loaded 3 users at once

=== Cache Hit ===
Retrieved from cache (no loading): User{id=1, name='Alice'}

=== Load All Users ===
Current cache size: 5
All users: [user:1, user:2, user:3, user:4, user:5]
\`\`\``,
    initialCode: `// TODO: Import Caffeine, LoadingCache, and CacheLoader

public class CacheLoading {
    // TODO: Define User class

    public static void main(String[] args) {
        // TODO: Create CacheLoader

        // TODO: Create LoadingCache

        // TODO: Load single values

        // TODO: Batch load multiple values

        // TODO: Display cache contents
    }
}`,
    solutionCode: `import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;

import java.util.List;
import java.util.Map;

public class CacheLoading {
    static class User {
        int id;
        String name;

        User(int id, String name) {
            this.id = id;
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{id=" + id + ", name='" + name + "'}";
        }
    }

    public static void main(String[] args) {
        System.out.println("=== LoadingCache with CacheLoader ===");

        // Create a CacheLoader that loads users
        CacheLoader<String, User> loader = key -> {
            System.out.println("Loading: " + key);
            // Simulate loading from database
            int id = Integer.parseInt(key.split(":")[1]);
            String[] names = {"Alice", "Bob", "Charlie", "Diana", "Eve"};
            return new User(id, names[id - 1]);
        };

        // Create LoadingCache
        LoadingCache<String, User> cache = Caffeine.newBuilder()
                .maximumSize(100)
                .build(loader);

        // Get will automatically load if not present
        User user1 = cache.get("user:1");
        System.out.println("User loaded: " + user1);

        // Second access won't reload
        User user1Again = cache.get("user:1");
        System.out.println("Cached retrieval: " + user1Again);

        // Load another user
        User user2 = cache.get("user:2");
        System.out.println("\\nUser loaded: " + user2);

        System.out.println("\\n=== Batch Loading ===");

        // Load multiple users at once
        Map<String, User> users = cache.getAll(
                List.of("user:3", "user:4", "user:5")
        );
        System.out.println("Loaded " + users.size() + " users at once");

        System.out.println("\\n=== Cache Hit ===");

        // This won't trigger loading (already cached)
        User cachedUser = cache.get("user:1");
        System.out.println("Retrieved from cache (no loading): " + cachedUser);

        System.out.println("\\n=== Load All Users ===");

        // Display cache statistics
        System.out.println("Current cache size: " + cache.estimatedSize());
        System.out.println("All users: " + cache.asMap().keySet());
    }
}`,
    hint1: `CacheLoader's load() method is called automatically when a key is not found in the cache. This method should contain the logic to fetch or compute the value.`,
    hint2: `LoadingCache.get() never returns null (unless your loader returns null). If the value isn't cached, it calls the loader synchronously. Use getAll() for efficient batch loading.`,
    whyItMatters: `LoadingCache simplifies caching logic by eliminating boilerplate null checks and manual value computation. It ensures consistent loading behavior across your application and reduces the risk of cache-related bugs. This pattern is essential for building scalable, maintainable caching layers.

**Production Pattern:**
\`\`\`java
// Automatic user data loading
public class UserCache {
    private final LoadingCache<Long, User> cache;
    private final UserRepository repository;

    public UserCache(UserRepository repository) {
        this.repository = repository;
        this.cache = Caffeine.newBuilder()
            .maximumSize(100_000)
            .expireAfterWrite(Duration.ofMinutes(30))
            .refreshAfterWrite(Duration.ofMinutes(5))
            .build(id -> {
                // Automatic loading from DB
                return repository.findById(id)
                    .orElseThrow(() -> new UserNotFoundException(id));
            });
    }

    public User getUser(Long id) {
        return cache.get(id);
    }

    public Map<Long, User> getUsers(List<Long> ids) {
        return cache.getAll(ids);
    }

    public void invalidate(Long id) {
        cache.invalidate(id);
    }
}
\`\`\`

**Practical Benefits:**
- Complete elimination of null-checks in application code
- Automatic background data refresh
- Efficient batch loading for multiple requests`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.util.concurrent.TimeUnit;

// Test1: Verify LoadingCache creation with loader
class Test1 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .build(key -> "loaded-" + key);
        assertNotNull(cache);
    }
}

// Test2: Verify automatic loading on cache miss
class Test2 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .build(key -> "loaded-" + key);

        String value = cache.get("testKey");
        assertEquals("loaded-testKey", value);
    }
}

// Test3: Verify cache returns same value on subsequent calls
class Test3 {
    @Test
    public void test() {
        LoadingCache<String, Integer> cache = Caffeine.newBuilder()
                .build(key -> (int) (Math.random() * 1000));

        Integer value1 = cache.get("key");
        Integer value2 = cache.get("key");

        assertEquals(value1, value2); // Should be same cached value
    }
}

// Test4: Verify expireAfterWrite configuration
class Test4 {
    @Test
    public void test() throws InterruptedException {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .expireAfterWrite(100, TimeUnit.MILLISECONDS)
                .build(key -> "value-" + key);

        cache.get("key");
        Thread.sleep(150);
        // Value should be reloaded after expiration
        String value = cache.get("key");
        assertNotNull(value);
    }
}

// Test5: Verify maximumSize eviction
class Test5 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .maximumSize(2)
                .build(key -> "value-" + key);

        cache.get("key1");
        cache.get("key2");
        cache.get("key3");

        assertTrue(cache.estimatedSize() <= 2);
    }
}

// Test6: Verify getAll batch loading
class Test6 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .build(key -> "value-" + key);

        java.util.List<String> keys = java.util.Arrays.asList("key1", "key2", "key3");
        java.util.Map<String, String> values = cache.getAll(keys);

        assertEquals(3, values.size());
        assertEquals("value-key1", values.get("key1"));
        assertEquals("value-key2", values.get("key2"));
        assertEquals("value-key3", values.get("key3"));
    }
}

// Test7: Verify refresh functionality
class Test7 {
    @Test
    public void test() {
        LoadingCache<String, Long> cache = Caffeine.newBuilder()
                .build(key -> System.currentTimeMillis());

        Long value1 = cache.get("key");
        cache.refresh("key");
        Long value2 = cache.get("key");

        // Values should be different due to refresh
        assertTrue(value2 >= value1);
    }
}

// Test8: Verify recordStats and stats retrieval
class Test8 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .recordStats()
                .build(key -> "value-" + key);

        cache.get("key1");
        cache.get("key1"); // Hit
        cache.get("key2"); // Miss

        assertNotNull(cache.stats());
        assertTrue(cache.stats().hitCount() >= 1);
    }
}

// Test9: Verify loader exception handling
class Test9 {
    @Test
    public void test() {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .build(key -> {
                    if (key.equals("error")) {
                        throw new RuntimeException("Test error");
                    }
                    return "value-" + key;
                });

        String validValue = cache.get("valid");
        assertEquals("value-valid", validValue);

        try {
            cache.get("error");
            fail("Should have thrown exception");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("Test error"));
        }
    }
}

// Test10: Verify expireAfterAccess configuration
class Test10 {
    @Test
    public void test() throws InterruptedException {
        LoadingCache<String, String> cache = Caffeine.newBuilder()
                .expireAfterAccess(100, TimeUnit.MILLISECONDS)
                .build(key -> "value-" + key);

        cache.get("key");
        Thread.sleep(50);
        cache.get("key"); // Access resets timer
        Thread.sleep(50);
        cache.get("key"); // Access again

        // Should still be cached since we kept accessing it
        assertEquals("value-key", cache.getIfPresent("key"));
    }
}
`,
    order: 2,
    translations: {
        ru: {
            title: 'Стратегии загрузки кэша',
            solutionCode: `import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;

import java.util.List;
import java.util.Map;

public class CacheLoading {
    static class User {
        int id;
        String name;

        User(int id, String name) {
            this.id = id;
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{id=" + id + ", name='" + name + "'}";
        }
    }

    public static void main(String[] args) {
        System.out.println("=== LoadingCache с CacheLoader ===");

        // Создание CacheLoader для загрузки пользователей
        CacheLoader<String, User> loader = key -> {
            System.out.println("Loading: " + key);
            // Имитация загрузки из базы данных
            int id = Integer.parseInt(key.split(":")[1]);
            String[] names = {"Alice", "Bob", "Charlie", "Diana", "Eve"};
            return new User(id, names[id - 1]);
        };

        // Создание LoadingCache
        LoadingCache<String, User> cache = Caffeine.newBuilder()
                .maximumSize(100)
                .build(loader);

        // Get автоматически загрузит, если отсутствует
        User user1 = cache.get("user:1");
        System.out.println("User loaded: " + user1);

        // Второй доступ не будет перезагружать
        User user1Again = cache.get("user:1");
        System.out.println("Cached retrieval: " + user1Again);

        // Загрузка другого пользователя
        User user2 = cache.get("user:2");
        System.out.println("\\nUser loaded: " + user2);

        System.out.println("\\n=== Пакетная загрузка ===");

        // Загрузка нескольких пользователей одновременно
        Map<String, User> users = cache.getAll(
                List.of("user:3", "user:4", "user:5")
        );
        System.out.println("Loaded " + users.size() + " users at once");

        System.out.println("\\n=== Попадание в кэш ===");

        // Это не вызовет загрузку (уже в кэше)
        User cachedUser = cache.get("user:1");
        System.out.println("Retrieved from cache (no loading): " + cachedUser);

        System.out.println("\\n=== Загрузка всех пользователей ===");

        // Отображение статистики кэша
        System.out.println("Current cache size: " + cache.estimatedSize());
        System.out.println("All users: " + cache.asMap().keySet());
    }
}`,
            description: `# Стратегии загрузки кэша

LoadingCache автоматически загружает значения, когда они отсутствуют в кэше. Это устраняет необходимость проверки на null и ручного вычисления отсутствующих записей. CacheLoader определяет логику загрузки значений, делая доступ к кэшу единообразным и безошибочным.

## Требования:
1. Создайте LoadingCache с CacheLoader:
   1.1. Реализуйте интерфейс CacheLoader с методом load()
   1.2. Используйте \`Caffeine.newBuilder().build(loader)\` для создания LoadingCache
   1.3. Определите логику загрузки для отсутствующих записей кэша

2. Продемонстрируйте операции LoadingCache:
   2.1. Используйте \`get(key)\` - автоматически загружает отсутствующие значения
   2.2. Используйте \`getAll(keys)\` - пакетная загрузка нескольких ключей
   2.3. Покажите, что значения загружаются только один раз для каждого ключа

3. Обработайте исключения загрузки:
   3.1. Покажите, что происходит при неудачной загрузке
   3.2. Перехватывайте и обрабатывайте исключения от загрузчика

4. Сравните LoadingCache с обычным Cache:
   4.1. Выделите разницу в обработке null
   4.2. Покажите автоматическое вычисление против ручного вычисления

## Пример вывода:
\`\`\`
=== LoadingCache with CacheLoader ===
Loading: user:1
User loaded: User{id=1, name='Alice'}
Cached retrieval: User{id=1, name='Alice'}

Loading: user:2
User loaded: User{id=2, name='Bob'}

=== Batch Loading ===
Loading: user:3
Loading: user:4
Loading: user:5
Loaded 3 users at once

=== Cache Hit ===
Retrieved from cache (no loading): User{id=1, name='Alice'}

=== Load All Users ===
Current cache size: 5
All users: [user:1, user:2, user:3, user:4, user:5]
\`\`\``,
            hint1: `Метод load() CacheLoader вызывается автоматически, когда ключ не найден в кэше. Этот метод должен содержать логику для получения или вычисления значения.`,
            hint2: `LoadingCache.get() никогда не возвращает null (если только ваш загрузчик не возвращает null). Если значение не закэшировано, он вызывает загрузчик синхронно. Используйте getAll() для эффективной пакетной загрузки.`,
            whyItMatters: `LoadingCache упрощает логику кэширования, устраняя шаблонные проверки на null и ручное вычисление значений. Он обеспечивает согласованное поведение загрузки во всем приложении и снижает риск ошибок, связанных с кэшем. Этот паттерн необходим для создания масштабируемых, поддерживаемых слоев кэширования.

**Продакшен паттерн:**
\`\`\`java
// Автоматическая загрузка данных пользователей
public class UserCache {
    private final LoadingCache<Long, User> cache;
    private final UserRepository repository;

    public UserCache(UserRepository repository) {
        this.repository = repository;
        this.cache = Caffeine.newBuilder()
            .maximumSize(100_000)
            .expireAfterWrite(Duration.ofMinutes(30))
            .refreshAfterWrite(Duration.ofMinutes(5))
            .build(id -> {
                // Автоматическая загрузка из БД
                return repository.findById(id)
                    .orElseThrow(() -> new UserNotFoundException(id));
            });
    }

    public User getUser(Long id) {
        return cache.get(id);
    }

    public Map<Long, User> getUsers(List<Long> ids) {
        return cache.getAll(ids);
    }

    public void invalidate(Long id) {
        cache.invalidate(id);
    }
}
\`\`\`

**Практические преимущества:**
- Полное устранение null-проверок в коде приложения
- Автоматическое обновление данных в фоне (refresh)
- Эффективная пакетная загрузка для множественных запросов`
        },
        uz: {
            title: `Kesh yuklash strategiyalari`,
            solutionCode: `import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;

import java.util.List;
import java.util.Map;

public class CacheLoading {
    static class User {
        int id;
        String name;

        User(int id, String name) {
            this.id = id;
            this.name = name;
        }

        @Override
        public String toString() {
            return "User{id=" + id + ", name='" + name + "'}";
        }
    }

    public static void main(String[] args) {
        System.out.println("=== CacheLoader bilan LoadingCache ===");

        // Foydalanuvchilarni yuklash uchun CacheLoader yaratish
        CacheLoader<String, User> loader = key -> {
            System.out.println("Loading: " + key);
            // Ma'lumotlar bazasidan yuklashni taqlid qilish
            int id = Integer.parseInt(key.split(":")[1]);
            String[] names = {"Alice", "Bob", "Charlie", "Diana", "Eve"};
            return new User(id, names[id - 1]);
        };

        // LoadingCache yaratish
        LoadingCache<String, User> cache = Caffeine.newBuilder()
                .maximumSize(100)
                .build(loader);

        // Get mavjud bo'lmasa avtomatik yuklaydi
        User user1 = cache.get("user:1");
        System.out.println("User loaded: " + user1);

        // Ikkinchi kirish qayta yuklamaydi
        User user1Again = cache.get("user:1");
        System.out.println("Cached retrieval: " + user1Again);

        // Boshqa foydalanuvchini yuklash
        User user2 = cache.get("user:2");
        System.out.println("\\nUser loaded: " + user2);

        System.out.println("\\n=== Paketli yuklash ===");

        // Bir vaqtning o'zida bir nechta foydalanuvchini yuklash
        Map<String, User> users = cache.getAll(
                List.of("user:3", "user:4", "user:5")
        );
        System.out.println("Loaded " + users.size() + " users at once");

        System.out.println("\\n=== Kesh topilishi ===");

        // Bu yuklashni chaqirmaydi (allaqachon keshda)
        User cachedUser = cache.get("user:1");
        System.out.println("Retrieved from cache (no loading): " + cachedUser);

        System.out.println("\\n=== Barcha foydalanuvchilarni yuklash ===");

        // Kesh statistikasini ko'rsatish
        System.out.println("Current cache size: " + cache.estimatedSize());
        System.out.println("All users: " + cache.asMap().keySet());
    }
}`,
            description: `# Kesh yuklash strategiyalari

LoadingCache keshda mavjud bo'lmaganda qiymatlarni avtomatik yuklaydi. Bu null tekshiruvlar va qo'lda yo'qolgan yozuvlarni hisoblash zaruriyatini bartaraf etadi. CacheLoader qiymatlarni yuklash mantiqini belgilaydi, keshga kirishni bir xil va xatosiz qiladi.

## Talablar:
1. CacheLoader bilan LoadingCache yarating:
   1.1. load() metodi bilan CacheLoader interfeysini amalga oshiring
   1.2. LoadingCache yaratish uchun \`Caffeine.newBuilder().build(loader)\` dan foydalaning
   1.3. Mavjud bo'lmagan kesh yozuvlari uchun yuklash mantiqini belgilang

2. LoadingCache operatsiyalarini namoyish eting:
   2.1. \`get(key)\` dan foydalaning - mavjud bo'lmagan qiymatlarni avtomatik yuklaydi
   2.2. \`getAll(keys)\` dan foydalaning - bir nechta kalitlarni paketli yuklash
   2.3. Qiymatlar har bir kalit uchun faqat bir marta yuklanishini ko'rsating

3. Yuklash istisnolarini ishlating:
   3.1. Yuklash muvaffaqiyatsiz bo'lganda nima sodir bo'lishini ko'rsating
   3.2. Yuklash moslamadan istisnolarni ushlang va ishlating

4. LoadingCache ni oddiy Cache bilan solishtiring:
   4.1. Null ishlov berishdagi farqni ta'kidlang
   4.2. Avtomatik hisoblashni qo'lda hisoblashga qarshi ko'rsating

## Chiqish namunasi:
\`\`\`
=== LoadingCache with CacheLoader ===
Loading: user:1
User loaded: User{id=1, name='Alice'}
Cached retrieval: User{id=1, name='Alice'}

Loading: user:2
User loaded: User{id=2, name='Bob'}

=== Batch Loading ===
Loading: user:3
Loading: user:4
Loading: user:5
Loaded 3 users at once

=== Cache Hit ===
Retrieved from cache (no loading): User{id=1, name='Alice'}

=== Load All Users ===
Current cache size: 5
All users: [user:1, user:2, user:3, user:4, user:5]
\`\`\``,
            hint1: `CacheLoader ning load() metodi kalit keshda topilmaganda avtomatik chaqiriladi. Bu metod qiymatni olish yoki hisoblash mantiqini o'z ichiga olishi kerak.`,
            hint2: `LoadingCache.get() hech qachon null qaytarmaydi (agar yuklash moslamangiz null qaytarmasa). Agar qiymat keshlangan bo'lmasa, u yuklash moslamani sinxron tarzda chaqiradi. Samarali paketli yuklash uchun getAll() dan foydalaning.`,
            whyItMatters: `LoadingCache keshlash mantiqini null tekshiruvlar va qo'lda qiymatni hisoblash shablonlarini yo'q qilish orqali soddalashtradi. U butun dasturda izchil yuklash xatti-harakatini ta'minlaydi va kesh bilan bog'liq xatolar xavfini kamaytiradi. Bu naqsh miqyoslanadigan, saqlanadiganr keshlash qatlamlarini yaratish uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Foydalanuvchi ma'lumotlarini avtomatik yuklash
public class UserCache {
    private final LoadingCache<Long, User> cache;
    private final UserRepository repository;

    public UserCache(UserRepository repository) {
        this.repository = repository;
        this.cache = Caffeine.newBuilder()
            .maximumSize(100_000)
            .expireAfterWrite(Duration.ofMinutes(30))
            .refreshAfterWrite(Duration.ofMinutes(5))
            .build(id -> {
                // Ma'lumotlar bazasidan avtomatik yuklash
                return repository.findById(id)
                    .orElseThrow(() -> new UserNotFoundException(id));
            });
    }

    public User getUser(Long id) {
        return cache.get(id);
    }

    public Map<Long, User> getUsers(List<Long> ids) {
        return cache.getAll(ids);
    }

    public void invalidate(Long id) {
        cache.invalidate(id);
    }
}
\`\`\`

**Amaliy foydalari:**
- Dastur kodida null-tekshirishlarni butunlay yo'q qilish
- Fonda ma'lumotlarni avtomatik yangilash (refresh)
- Ko'p so'rovlar uchun samarali paketli yuklash`
        }
    }
};

export default task;
