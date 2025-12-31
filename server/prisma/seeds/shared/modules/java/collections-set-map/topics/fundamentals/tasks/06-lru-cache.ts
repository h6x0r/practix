import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-lru-cache',
	title: 'Implementing LRU Cache',
	difficulty: 'hard',
	tags: ['java', 'collections', 'cache', 'lru'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement an **LRUCache** class using LinkedHashMap with access-order eviction.

**Requirements:**
1. Create \`LRUCache<K, V>\` with maximum capacity
2. Implement \`put(K key, V value)\` - stores value, evicts LRU if at capacity
3. Implement \`get(K key)\` - returns value and marks as recently used
4. Implement \`size()\` to return current cache size
5. Automatically evict least recently used entry when capacity exceeded

**Example:**
\`\`\`java
LRUCache<String, Integer> cache = new LRUCache<>(3);
cache.put("a", 1);
cache.put("b", 2);
cache.put("c", 3);
cache.get("a");        // Access "a", makes it most recent
cache.put("d", 4);     // Evicts "b" (least recently used)
System.out.println(cache.get("b")); // null (evicted)
System.out.println(cache.get("a")); // 1 (still present)
\`\`\`

**Key Concepts:**
- Use LinkedHashMap with accessOrder=true
- Override removeEldestEntry() for automatic eviction
- get() and put() both update access order`,
	initialCode: `import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> {
    // TODO: Declare fields for map and capacity

    public LRUCache(int capacity) {
        // TODO: Initialize LinkedHashMap with access-order
    }

    public void put(K key, V value) {
        // TODO: Store value, auto-eviction handled by removeEldestEntry
    }

    public V get(K key) {
        // TODO: Return value (null if not present)
        return null;
    }

    public int size() {
        // TODO: Return current size
        return 0;
    }
}`,
	solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> {
    private final int capacity;
    private final LinkedHashMap<K, V> cache;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        // accessOrder=true for LRU behavior
        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > capacity;  // Auto-evict when exceeds
            }
        };
    }

    public void put(K key, V value) {
        cache.put(key, value);  // Auto-evicts if needed
    }

    public V get(K key) {
        return cache.get(key);  // Returns null if not present, updates access order
    }

    public int size() {
        return cache.size();
    }
}`,
	hint1: `Use LinkedHashMap constructor with accessOrder=true: new LinkedHashMap<>(capacity, 0.75f, true)`,
	hint2: `Override removeEldestEntry() to return true when size() > capacity. This enables automatic LRU eviction.`,
	whyItMatters: `LRU Cache is a fundamental caching pattern used everywhere from CPU caches to web applications.

**Why LRU Cache:**
- **Memory Bounded:** Never exceeds configured capacity
- **Automatic Eviction:** Removes least recently used items
- **Hot Data Priority:** Frequently accessed items stay cached

**Production Use Cases:**
\`\`\`java
// Database query cache
LRUCache<String, QueryResult> queryCache = new LRUCache<>(1000);
QueryResult result = queryCache.get(sqlHash);
if (result == null) {
    result = executeQuery(sql);
    queryCache.put(sqlHash, result);
}

// Session cache
LRUCache<String, Session> sessions = new LRUCache<>(10000);

// Image thumbnail cache
LRUCache<String, BufferedImage> thumbnails = new LRUCache<>(500);
\`\`\`

**Alternative Implementations:**
- Caffeine library - more features, better performance
- Guava Cache - configurable expiration, statistics
- Redis - distributed caching

**LRU Variants:**
- LRU-K: Considers K-th access time
- SLRU: Segmented LRU (probation + protected)
- ARC: Adaptive Replacement Cache

**Production Pattern:**
\`\`\`java
// Computation result cache
LRUCache<String, ComputationResult> resultCache = new LRUCache<>(1000);
ComputationResult result = resultCache.get(inputHash);
if (result == null) {
    result = performExpensiveComputation(input);
    resultCache.put(inputHash, result);
}

// Memory-bounded session cache
LRUCache<String, UserSession> sessions = new LRUCache<>(10000);
sessions.put(sessionId, session);  // Old sessions auto-evicted
\`\`\`

**Practical Benefits:**
- Automatic memory management with LRU eviction
- Simple implementation without external libraries
- Excellent performance for local caching`,
	order: 5,
	testCode: `import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: LRUCache can be created with capacity
class Test1 {
    @Test
    void testCreation() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        assertNotNull(cache);
    }
}

// Test2: put() and get() work correctly
class Test2 {
    @Test
    void testPutGet() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        cache.put("key", 42);
        assertEquals(42, cache.get("key"));
    }
}

// Test3: size() returns current size
class Test3 {
    @Test
    void testSize() {
        LRUCache<String, Integer> cache = new LRUCache<>(5);
        cache.put("a", 1);
        cache.put("b", 2);
        assertEquals(2, cache.size());
    }
}

// Test4: Evicts LRU when capacity exceeded
class Test4 {
    @Test
    void testEviction() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.put("d", 4);
        assertNull(cache.get("a"));
        assertEquals(3, cache.size());
    }
}

// Test5: get() updates access order
class Test5 {
    @Test
    void testGetUpdatesOrder() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.get("a");
        cache.put("d", 4);
        assertNull(cache.get("b"));
        assertEquals(1, cache.get("a"));
    }
}

// Test6: get() returns null for missing key
class Test6 {
    @Test
    void testGetMissing() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        assertNull(cache.get("nonexistent"));
    }
}

// Test7: Capacity of 1 works correctly
class Test7 {
    @Test
    void testCapacityOne() {
        LRUCache<String, Integer> cache = new LRUCache<>(1);
        cache.put("a", 1);
        cache.put("b", 2);
        assertNull(cache.get("a"));
        assertEquals(2, cache.get("b"));
    }
}

// Test8: Update existing key doesn't count as new entry
class Test8 {
    @Test
    void testUpdateExisting() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.put("a", 100);
        assertEquals(3, cache.size());
        assertEquals(100, cache.get("a"));
    }
}

// Test9: Recent access prevents eviction
class Test9 {
    @Test
    void testRecentAccessPreventsEviction() {
        LRUCache<String, Integer> cache = new LRUCache<>(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.get("a");
        cache.get("b");
        cache.put("d", 4);
        assertNull(cache.get("c"));
        assertNotNull(cache.get("a"));
        assertNotNull(cache.get("b"));
    }
}

// Test10: Works with different types
class Test10 {
    @Test
    void testDifferentTypes() {
        LRUCache<Integer, String> cache = new LRUCache<>(2);
        cache.put(1, "one");
        cache.put(2, "two");
        assertEquals("one", cache.get(1));
        assertEquals("two", cache.get(2));
    }
}
`,
	translations: {
		ru: {
			title: 'Реализация LRU Кэша',
			description: `Реализуйте класс **LRUCache** с использованием LinkedHashMap с вытеснением по порядку доступа.

**Требования:**
1. Создайте \`LRUCache<K, V>\` с максимальной вместимостью
2. Реализуйте \`put(K key, V value)\` - сохраняет значение, вытесняет LRU при достижении вместимости
3. Реализуйте \`get(K key)\` - возвращает значение и отмечает как недавно использованное
4. Реализуйте \`size()\` для возврата текущего размера кэша
5. Автоматически вытесняйте наименее недавно использованную запись при превышении вместимости

**Пример:**
\`\`\`java
LRUCache<String, Integer> cache = new LRUCache<>(3);
cache.put("a", 1);
cache.put("b", 2);
cache.put("c", 3);
cache.get("a");        // Доступ к "a", делает его самым недавним
cache.put("d", 4);     // Вытесняет "b" (наименее недавно использованный)
System.out.println(cache.get("b")); // null (вытеснен)
\`\`\`

**Ключевые концепции:**
- Используйте LinkedHashMap с accessOrder=true
- Переопределите removeEldestEntry() для автоматического вытеснения
- get() и put() обновляют порядок доступа`,
			hint1: `Используйте конструктор LinkedHashMap с accessOrder=true: new LinkedHashMap<>(capacity, 0.75f, true)`,
			hint2: `Переопределите removeEldestEntry() чтобы возвращать true когда size() > capacity.`,
			whyItMatters: `LRU Кэш - фундаментальный паттерн кэширования, используемый повсюду от кэшей CPU до веб-приложений.

**Почему LRU Кэш:**
- **Ограничение памяти:** Никогда не превышает заданную вместимость
- **Автоматическое вытеснение:** Удаляет наименее недавно использованные элементы
- **Приоритет горячих данных:** Часто используемые элементы остаются в кэше

**Использование в production:**
- Кэш запросов к базе данных
- Кэш сессий
- Кэш миниатюр изображений

**Альтернативные реализации:**
- Caffeine - больше возможностей
- Guava Cache - настраиваемое истечение
- Redis - распределенное кэширование

**Продакшен паттерн:**
\`\`\`java
// Кэш результатов вычислений
LRUCache<String, ComputationResult> resultCache = new LRUCache<>(1000);
ComputationResult result = resultCache.get(inputHash);
if (result == null) {
    result = performExpensiveComputation(input);
    resultCache.put(inputHash, result);
}

// Кэш сессий с ограничением памяти
LRUCache<String, UserSession> sessions = new LRUCache<>(10000);
sessions.put(sessionId, session);  // Старые сессии вытесняются автоматически
\`\`\`

**Практические преимущества:**
- Автоматическое управление памятью с вытеснением LRU
- Простая реализация без внешних библиотек
- Отличная производительность для локального кэширования`,
			solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> {
    private final int capacity;
    private final LinkedHashMap<K, V> cache;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        // accessOrder=true для LRU поведения
        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > capacity;  // Автоматическое вытеснение при превышении
            }
        };
    }

    public void put(K key, V value) {
        cache.put(key, value);  // Автоматически вытесняет при необходимости
    }

    public V get(K key) {
        return cache.get(key);  // Возвращает null если отсутствует, обновляет порядок доступа
    }

    public int size() {
        return cache.size();
    }
}`
		},
		uz: {
			title: 'LRU Keshni Amalga Oshirish',
			description: `Kirish tartibi bo'yicha chiqarish bilan LinkedHashMap ishlatib **LRUCache** klassini amalga oshiring.

**Talablar:**
1. Maksimal sig'im bilan \`LRUCache<K, V>\` yarating
2. \`put(K key, V value)\` - qiymatni saqlaydi, sig'imga yetganda LRU ni chiqaradi
3. \`get(K key)\` - qiymatni qaytaradi va yaqinda ishlatilgan deb belgilaydi
4. Joriy kesh o'lchamini qaytarish uchun \`size()\` ni amalga oshiring
5. Sig'im oshganda eng kam yaqinda ishlatilgan yozuvni avtomatik chiqaring

**Misol:**
\`\`\`java
LRUCache<String, Integer> cache = new LRUCache<>(3);
cache.put("a", 1);
cache.put("b", 2);
cache.put("c", 3);
cache.get("a");        // "a" ga kirish, uni eng yaqin qiladi
cache.put("d", 4);     // "b" ni chiqaradi (eng kam yaqinda ishlatilgan)
System.out.println(cache.get("b")); // null (chiqarilgan)
\`\`\`

**Asosiy tushunchalar:**
- accessOrder=true bilan LinkedHashMap dan foydalaning
- Avtomatik chiqarish uchun removeEldestEntry() ni qayta belgilang
- get() va put() ikkalasi ham kirish tartibini yangilaydi`,
			hint1: `accessOrder=true bilan LinkedHashMap konstruktoridan foydalaning: new LinkedHashMap<>(capacity, 0.75f, true)`,
			hint2: `size() > capacity bo'lganda true qaytarish uchun removeEldestEntry() ni qayta belgilang.`,
			whyItMatters: `LRU Kesh CPU keshlaridan web ilovalarigacha hamma joyda ishlatiladigan asosiy keshlash namunasi.

**Nima uchun LRU Kesh:**
- **Xotira chegaralangan:** Sozlangan sig'imdan hech qachon oshmaydi
- **Avtomatik chiqarish:** Eng kam yaqinda ishlatilgan elementlarni o'chiradi
- **Issiq ma'lumotlar ustuvorligi:** Tez-tez ishlatiladigan elementlar keshda qoladi

**Production'da foydalanish:**
- Ma'lumotlar bazasi so'rovlari keshi
- Sessiya keshi
- Rasm miniatyuralari keshi

**Muqobil amalga oshirishlar:**
- Caffeine - ko'proq imkoniyatlar
- Guava Cache - sozlanadigan muddati tugashi
- Redis - taqsimlangan keshlash

**Ishlab chiqarish patterni:**
\`\`\`java
// Hisoblash natijalari keshi
LRUCache<String, ComputationResult> resultCache = new LRUCache<>(1000);
ComputationResult result = resultCache.get(inputHash);
if (result == null) {
    result = performExpensiveComputation(input);
    resultCache.put(inputHash, result);
}

// Xotira cheklangan sessiya keshi
LRUCache<String, UserSession> sessions = new LRUCache<>(10000);
sessions.put(sessionId, session);  // Eski sessiyalar avtomatik chiqariladi
\`\`\`

**Amaliy foydalari:**
- LRU chiqarish bilan avtomatik xotira boshqaruvi
- Tashqi kutubxonalarsiz oddiy implementatsiya
- Lokal keshlash uchun ajoyib samaradorlik`,
			solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> {
    private final int capacity;
    private final LinkedHashMap<K, V> cache;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        // LRU xatti-harakati uchun accessOrder=true
        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > capacity;  // Oshganda avtomatik chiqarish
            }
        };
    }

    public void put(K key, V value) {
        cache.put(key, value);  // Kerak bo'lsa avtomatik chiqaradi
    }

    public V get(K key) {
        return cache.get(key);  // Yo'q bo'lsa null qaytaradi, kirish tartibini yangilaydi
    }

    public int size() {
        return cache.size();
    }
}`
		}
	}
};

export default task;
