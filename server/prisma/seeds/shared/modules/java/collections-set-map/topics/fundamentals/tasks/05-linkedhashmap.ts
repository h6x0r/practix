import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-linkedhashmap',
	title: 'LinkedHashMap and Insertion Order',
	difficulty: 'medium',
	tags: ['java', 'collections', 'linkedhashmap'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement an **OrderedCache** class using LinkedHashMap to maintain insertion order.

**Requirements:**
1. Create \`OrderedCache<K, V>\` with LinkedHashMap storage
2. Implement \`put(K key, V value)\` to store key-value pair
3. Implement \`get(K key)\` to retrieve value
4. Implement \`keys()\` to return keys in insertion order
5. Implement \`oldest()\` to return the first inserted key

**Example:**
\`\`\`java
OrderedCache<String, Integer> cache = new OrderedCache<>();
cache.put("first", 1);
cache.put("second", 2);
cache.put("third", 3);
System.out.println(cache.keys());   // [first, second, third]
System.out.println(cache.oldest()); // first
cache.put("first", 10);             // Update existing
System.out.println(cache.keys());   // [first, second, third] - order preserved
\`\`\`

**Key Concepts:**
- LinkedHashMap maintains insertion order (or access order)
- O(1) for get, put, containsKey operations
- Iteration follows insertion order unlike HashMap`,
	initialCode: `import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class OrderedCache<K, V> {
    // TODO: Declare LinkedHashMap field

    public OrderedCache() {
        // TODO: Initialize LinkedHashMap
    }

    public void put(K key, V value) {
        // TODO: Store key-value pair
    }

    public V get(K key) {
        // TODO: Retrieve value
        return null;
    }

    public List<K> keys() {
        // TODO: Return keys in insertion order
        return null;
    }

    public K oldest() {
        // TODO: Return first inserted key
        return null;
    }
}`,
	solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class OrderedCache<K, V> {
    private LinkedHashMap<K, V> cache;

    public OrderedCache() {
        this.cache = new LinkedHashMap<>();
    }

    public void put(K key, V value) {
        cache.put(key, value);
    }

    public V get(K key) {
        return cache.get(key);
    }

    public List<K> keys() {
        return new ArrayList<>(cache.keySet()); // Insertion order
    }

    public K oldest() {
        if (cache.isEmpty()) return null;
        return cache.keySet().iterator().next(); // First element
    }
}`,
	hint1: `LinkedHashMap maintains insertion order by default. keySet() and values() iterate in insertion order.`,
	hint2: `Use keySet().iterator().next() to get the first (oldest) key. Create a new ArrayList from keySet() for ordered list.`,
	whyItMatters: `LinkedHashMap combines HashMap speed with predictable iteration order.

**Why LinkedHashMap:**
- **Predictable Order:** Iterates in insertion order (or access order)
- **O(1) Operations:** Same performance as HashMap
- **LRU Support:** Can be configured for access-order (for LRU caches)

**Production Use Cases:**
\`\`\`java
// Ordered configuration
Map<String, String> config = new LinkedHashMap<>();
config.put("host", "localhost");
config.put("port", "8080");
// Always outputs in same order when serialized

// Recent items
LinkedHashMap<String, Action> recentActions = new LinkedHashMap<>();
// Iteration shows oldest to newest

// Simple LRU cache
LinkedHashMap<K, V> lruCache = new LinkedHashMap<>(16, 0.75f, true) {
    @Override
    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > MAX_SIZE;
    }
};
\`\`\`

**Constructor Options:**
- \`new LinkedHashMap<>()\` - insertion order
- \`new LinkedHashMap<>(capacity, loadFactor, accessOrder)\` - access order when true

**Production Pattern:**
\`\`\`java
// JSON-like serialization with order preservation
Map<String, Object> jsonObject = new LinkedHashMap<>();
jsonObject.put("id", userId);
jsonObject.put("name", userName);
jsonObject.put("email", userEmail);
// Always outputs in same order: id, name, email

// Fixed-size command history
Map<String, Command> commandHistory = new LinkedHashMap<>(MAX_HISTORY, 0.75f, true) {
    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > MAX_HISTORY;
    }
};
\`\`\`

**Practical Benefits:**
- Predictable serialization for API responses
- Command history with order preservation
- Basic LRU cache implementation`,
	order: 4,
	testCode: `import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: OrderedCache can be created
class Test1 {
    @Test
    void testCreation() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        assertNotNull(cache);
    }
}

// Test2: put() and get() work correctly
class Test2 {
    @Test
    void testPutGet() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("key", 42);
        assertEquals(42, cache.get("key"));
    }
}

// Test3: keys() returns insertion order
class Test3 {
    @Test
    void testKeysOrder() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("first", 1);
        cache.put("second", 2);
        cache.put("third", 3);
        List<String> keys = cache.keys();
        assertEquals("first", keys.get(0));
        assertEquals("second", keys.get(1));
        assertEquals("third", keys.get(2));
    }
}

// Test4: oldest() returns first inserted key
class Test4 {
    @Test
    void testOldest() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("first", 1);
        cache.put("second", 2);
        assertEquals("first", cache.oldest());
    }
}

// Test5: Update preserves insertion order
class Test5 {
    @Test
    void testUpdatePreservesOrder() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("first", 1);
        cache.put("second", 2);
        cache.put("first", 100);
        List<String> keys = cache.keys();
        assertEquals("first", keys.get(0));
    }
}

// Test6: get() returns null for missing key
class Test6 {
    @Test
    void testGetMissing() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        assertNull(cache.get("nonexistent"));
    }
}

// Test7: oldest() returns null for empty cache
class Test7 {
    @Test
    void testOldestEmpty() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        assertNull(cache.oldest());
    }
}

// Test8: keys() returns empty list for empty cache
class Test8 {
    @Test
    void testKeysEmpty() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        assertTrue(cache.keys().isEmpty());
    }
}

// Test9: Multiple values stored correctly
class Test9 {
    @Test
    void testMultipleValues() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        assertEquals(1, cache.get("a"));
        assertEquals(2, cache.get("b"));
        assertEquals(3, cache.get("c"));
    }
}

// Test10: keys() size matches entries
class Test10 {
    @Test
    void testKeysSize() {
        OrderedCache<String, Integer> cache = new OrderedCache<>();
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        assertEquals(3, cache.keys().size());
    }
}
`,
	translations: {
		ru: {
			title: 'LinkedHashMap и Порядок Вставки',
			description: `Реализуйте класс **OrderedCache** с использованием LinkedHashMap для сохранения порядка вставки.

**Требования:**
1. Создайте \`OrderedCache<K, V>\` с хранилищем LinkedHashMap
2. Реализуйте \`put(K key, V value)\` для хранения пары ключ-значение
3. Реализуйте \`get(K key)\` для получения значения
4. Реализуйте \`keys()\` для возврата ключей в порядке вставки
5. Реализуйте \`oldest()\` для возврата первого вставленного ключа

**Пример:**
\`\`\`java
OrderedCache<String, Integer> cache = new OrderedCache<>();
cache.put("first", 1);
cache.put("second", 2);
cache.put("third", 3);
System.out.println(cache.keys());   // [first, second, third]
System.out.println(cache.oldest()); // first
\`\`\`

**Ключевые концепции:**
- LinkedHashMap сохраняет порядок вставки (или доступа)
- O(1) для get, put, containsKey операций
- Итерация следует порядку вставки в отличие от HashMap`,
			hint1: `LinkedHashMap по умолчанию сохраняет порядок вставки. keySet() и values() итерируют в порядке вставки.`,
			hint2: `Используйте keySet().iterator().next() для получения первого (старейшего) ключа.`,
			whyItMatters: `LinkedHashMap сочетает скорость HashMap с предсказуемым порядком итерации.

**Почему LinkedHashMap:**
- **Предсказуемый порядок:** Итерирует в порядке вставки (или доступа)
- **O(1) операции:** Такая же производительность как HashMap
- **Поддержка LRU:** Можно настроить для access-order

**Использование в production:**
- Упорядоченная конфигурация
- Список недавних действий
- Простой LRU кэш

**Опции конструктора:**
- \`new LinkedHashMap<>()\` - порядок вставки
- \`new LinkedHashMap<>(capacity, loadFactor, accessOrder)\` - порядок доступа

**Продакшен паттерн:**
\`\`\`java
// JSON-подобная сериализация с сохранением порядка
Map<String, Object> jsonObject = new LinkedHashMap<>();
jsonObject.put("id", userId);
jsonObject.put("name", userName);
jsonObject.put("email", userEmail);
// Всегда выводится в том же порядке: id, name, email

// История команд фиксированного размера
Map<String, Command> commandHistory = new LinkedHashMap<>(MAX_HISTORY, 0.75f, true) {
    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > MAX_HISTORY;
    }
};
\`\`\`

**Практические преимущества:**
- Предсказуемая сериализация для API ответов
- История команд с сохранением порядка
- Базовая реализация LRU кэша`,
			solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class OrderedCache<K, V> {
    private LinkedHashMap<K, V> cache;

    public OrderedCache() {
        this.cache = new LinkedHashMap<>();
    }

    public void put(K key, V value) {
        cache.put(key, value);
    }

    public V get(K key) {
        return cache.get(key);
    }

    public List<K> keys() {
        return new ArrayList<>(cache.keySet()); // Порядок вставки
    }

    public K oldest() {
        if (cache.isEmpty()) return null;
        return cache.keySet().iterator().next(); // Первый элемент
    }
}`
		},
		uz: {
			title: 'LinkedHashMap va Qo\'shish Tartibi',
			description: `Qo'shish tartibini saqlash uchun LinkedHashMap ishlatib **OrderedCache** klassini amalga oshiring.

**Talablar:**
1. LinkedHashMap saqlash bilan \`OrderedCache<K, V>\` yarating
2. Kalit-qiymat juftini saqlash uchun \`put(K key, V value)\` ni amalga oshiring
3. Qiymatni olish uchun \`get(K key)\` ni amalga oshiring
4. Kalitlarni qo'shish tartibida qaytarish uchun \`keys()\` ni amalga oshiring
5. Birinchi qo'shilgan kalitni qaytarish uchun \`oldest()\` ni amalga oshiring

**Misol:**
\`\`\`java
OrderedCache<String, Integer> cache = new OrderedCache<>();
cache.put("first", 1);
cache.put("second", 2);
cache.put("third", 3);
System.out.println(cache.keys());   // [first, second, third]
System.out.println(cache.oldest()); // first
\`\`\`

**Asosiy tushunchalar:**
- LinkedHashMap qo'shish tartibini saqlaydi (yoki kirish tartibini)
- get, put, containsKey uchun O(1)
- HashMap dan farqli o'laroq iteratsiya qo'shish tartibiga amal qiladi`,
			hint1: `LinkedHashMap sukut bo'yicha qo'shish tartibini saqlaydi. keySet() va values() qo'shish tartibida iteratsiya qiladi.`,
			hint2: `Birinchi (eng eski) kalitni olish uchun keySet().iterator().next() dan foydalaning.`,
			whyItMatters: `LinkedHashMap HashMap tezligini bashorat qilinadigan iteratsiya tartibi bilan birlashtiradi.

**Nima uchun LinkedHashMap:**
- **Bashorat qilinadigan tartib:** Qo'shish tartibida iteratsiya qiladi
- **O(1) operatsiyalar:** HashMap bilan bir xil unumdorlik
- **LRU qo'llab-quvvatlash:** Kirish tartibi uchun sozlanishi mumkin

**Production'da foydalanish:**
- Tartiblangan konfiguratsiya
- Yaqinda amalga oshirilgan harakatlar ro'yxati
- Oddiy LRU kesh

**Konstruktor variantlari:**
- \`new LinkedHashMap<>()\` - qo'shish tartibi
- \`new LinkedHashMap<>(capacity, loadFactor, accessOrder)\` - kirish tartibi

**Ishlab chiqarish patterni:**
\`\`\`java
// Tartibni saqlagan holda JSON-ga o'xshash serializatsiya
Map<String, Object> jsonObject = new LinkedHashMap<>();
jsonObject.put("id", userId);
jsonObject.put("name", userName);
jsonObject.put("email", userEmail);
// Doimo bir xil tartibda chiqadi: id, name, email

// Qat'iy o'lchamdagi buyruqlar tarixi
Map<String, Command> commandHistory = new LinkedHashMap<>(MAX_HISTORY, 0.75f, true) {
    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > MAX_HISTORY;
    }
};
\`\`\`

**Amaliy foydalari:**
- API javoblari uchun bashorat qilinadigan serializatsiya
- Tartibni saqlaydigan buyruqlar tarixi
- LRU keshning asosiy implementatsiyasi`,
			solutionCode: `import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class OrderedCache<K, V> {
    private LinkedHashMap<K, V> cache;

    public OrderedCache() {
        this.cache = new LinkedHashMap<>();
    }

    public void put(K key, V value) {
        cache.put(key, value);
    }

    public V get(K key) {
        return cache.get(key);
    }

    public List<K> keys() {
        return new ArrayList<>(cache.keySet()); // Qo'shish tartibi
    }

    public K oldest() {
        if (cache.isEmpty()) return null;
        return cache.keySet().iterator().next(); // Birinchi element
    }
}`
		}
	}
};

export default task;
