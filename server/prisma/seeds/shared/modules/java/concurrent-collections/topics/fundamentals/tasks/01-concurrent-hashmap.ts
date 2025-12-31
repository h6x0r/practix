import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-concurrent-hashmap',
    title: 'ConcurrentHashMap',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'collections', 'thread-safe'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn ConcurrentHashMap for thread-safe map operations.

**Requirements:**
1. Create a ConcurrentHashMap to store product inventory
2. Add products using put(): "laptop"->10, "mouse"->50, "keyboard"->30
3. Use putIfAbsent() to add "monitor"->15 (only if absent)
4. Use computeIfAbsent() to get "tablet" or set it to 20
5. Increment "laptop" count by 5 using compute()
6. Remove "mouse" if count equals 50 using remove(key, value)
7. Print all entries using forEach()
8. Demonstrate thread-safety by updating from multiple threads

ConcurrentHashMap provides thread-safe operations without locking the entire map, allowing high concurrency.`,
    initialCode: `import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentHashMapDemo {
    public static void main(String[] args) {
        // Create a ConcurrentHashMap for product inventory

        // Add products using put()

        // Use putIfAbsent() to add "monitor"->15

        // Use computeIfAbsent() for "tablet"->20

        // Increment "laptop" count by 5 using compute()

        // Remove "mouse" if count equals 50

        // Print all entries using forEach()

        // Demonstrate thread-safety with multiple threads
    }
}`,
    solutionCode: `import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentHashMapDemo {
    public static void main(String[] args) throws InterruptedException {
        // Create a ConcurrentHashMap for product inventory
        ConcurrentHashMap<String, Integer> inventory = new ConcurrentHashMap<>();

        // Add products using put()
        inventory.put("laptop", 10);
        inventory.put("mouse", 50);
        inventory.put("keyboard", 30);
        System.out.println("Initial inventory: " + inventory);

        // Use putIfAbsent() to add "monitor"->15 (only if absent)
        Integer previous = inventory.putIfAbsent("monitor", 15);
        System.out.println("Added monitor, previous value: " + previous);

        // Try putIfAbsent() again - should return existing value
        previous = inventory.putIfAbsent("monitor", 20);
        System.out.println("Tried to add monitor again, existing value: " + previous);

        // Use computeIfAbsent() to get "tablet" or set it to 20
        Integer tabletCount = inventory.computeIfAbsent("tablet", k -> 20);
        System.out.println("Tablet count (computed): " + tabletCount);

        // Increment "laptop" count by 5 using compute()
        inventory.compute("laptop", (k, v) -> v == null ? 5 : v + 5);
        System.out.println("Laptop count after increment: " + inventory.get("laptop"));

        // Remove "mouse" if count equals 50 using remove(key, value)
        boolean removed = inventory.remove("mouse", 50);
        System.out.println("Mouse removed (count was 50): " + removed);

        // Print all entries using forEach()
        System.out.println("\nCurrent inventory:");
        inventory.forEach((product, count) ->
            System.out.println("  " + product + ": " + count)
        );

        // Demonstrate thread-safety with multiple threads
        System.out.println("\nTesting concurrent updates...");
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("Final laptop count after concurrent updates: " + inventory.get("laptop"));
        System.out.println("Expected: 215 (15 + 100 + 100)");
    }
}`,
    hint1: `putIfAbsent() returns null if the key was absent, or the existing value if present. Use it for conditional insertion.`,
    hint2: `computeIfAbsent() takes a function that computes the value if absent. compute() always applies the function, allowing atomic updates.`,
    whyItMatters: `ConcurrentHashMap is essential for building thread-safe caches and shared data structures in multi-threaded applications, providing better performance than synchronized HashMap.

**Production Pattern:**
\`\`\`java
// Application configuration cache with thread-safe updates
public class ConfigCache {
    private final ConcurrentHashMap<String, String> cache = new ConcurrentHashMap<>();

    public String getConfig(String key, Supplier<String> loader) {
        // Atomic computation of value if absent
        return cache.computeIfAbsent(key, k -> loader.get());
    }

    public void updateMetric(String metricName, int increment) {
        // Atomic update of metric counter
        cache.compute(metricName, (k, v) -> {
            int current = v == null ? 0 : Integer.parseInt(v);
            return String.valueOf(current + increment);
        });
    }
}
\`\`\`

**Practical Benefits:**
- High performance under concurrent access without locking entire map
- Atomic operations (computeIfAbsent, compute) for safe updates
- Ideal for caches, counters, shared configurations in web applications`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.*;

// Test1: Verify ConcurrentHashMap class exists
class Test1 {
    @Test
    public void testConcurrentHashMapExists() {
        assertNotNull(ConcurrentHashMap.class);
    }
}

// Test2: Verify basic put and get operations
class Test2 {
    @Test
    public void testPutAndGet() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("key1", 100);

        assertEquals(Integer.valueOf(100), map.get("key1"));
    }
}

// Test3: Verify putIfAbsent method
class Test3 {
    @Test
    public void testPutIfAbsent() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

        Integer result1 = map.putIfAbsent("key1", 100);
        assertNull(result1); // Returns null if key was absent

        Integer result2 = map.putIfAbsent("key1", 200);
        assertEquals(Integer.valueOf(100), result2); // Returns existing value
    }
}

// Test4: Verify computeIfAbsent method
class Test4 {
    @Test
    public void testComputeIfAbsent() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

        Integer value = map.computeIfAbsent("key1", k -> 42);
        assertEquals(Integer.valueOf(42), value);
        assertEquals(Integer.valueOf(42), map.get("key1"));
    }
}

// Test5: Verify compute method
class Test5 {
    @Test
    public void testCompute() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("counter", 10);

        map.compute("counter", (k, v) -> v == null ? 1 : v + 1);
        assertEquals(Integer.valueOf(11), map.get("counter"));
    }
}

// Test6: Verify remove with value check
class Test6 {
    @Test
    public void testRemoveWithValue() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("key1", 100);

        boolean removed1 = map.remove("key1", 100);
        assertTrue(removed1);

        boolean removed2 = map.remove("key2", 100);
        assertFalse(removed2); // Key doesn't exist
    }
}

// Test7: Verify concurrent modifications
class Test7 {
    @Test
    public void testConcurrentModifications() throws InterruptedException {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("counter", 0);

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                map.compute("counter", (k, v) -> v + 1);
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                map.compute("counter", (k, v) -> v + 1);
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        assertEquals(Integer.valueOf(200), map.get("counter"));
    }
}

// Test8: Verify size method
class Test8 {
    @Test
    public void testSize() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("key1", 1);
        map.put("key2", 2);
        map.put("key3", 3);

        assertEquals(3, map.size());
    }
}

// Test9: Verify isEmpty method
class Test9 {
    @Test
    public void testIsEmpty() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        assertTrue(map.isEmpty());

        map.put("key1", 1);
        assertFalse(map.isEmpty());
    }
}

// Test10: Verify forEach operation
class Test10 {
    @Test
    public void testForEach() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("a", 1);
        map.put("b", 2);
        map.put("c", 3);

        int[] sum = {0};
        map.forEach((k, v) -> sum[0] += v);

        assertEquals(6, sum[0]);
    }
}`,
    order: 0,
    translations: {
        ru: {
            title: 'ConcurrentHashMap',
            solutionCode: `import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentHashMapDemo {
    public static void main(String[] args) throws InterruptedException {
        // Создаем ConcurrentHashMap для инвентаря продуктов
        ConcurrentHashMap<String, Integer> inventory = new ConcurrentHashMap<>();

        // Добавляем продукты используя put()
        inventory.put("laptop", 10);
        inventory.put("mouse", 50);
        inventory.put("keyboard", 30);
        System.out.println("Начальный инвентарь: " + inventory);

        // Используем putIfAbsent() для добавления "monitor"->15 (только если отсутствует)
        Integer previous = inventory.putIfAbsent("monitor", 15);
        System.out.println("Добавлен monitor, предыдущее значение: " + previous);

        // Пробуем putIfAbsent() снова - должно вернуть существующее значение
        previous = inventory.putIfAbsent("monitor", 20);
        System.out.println("Попытка добавить monitor снова, существующее значение: " + previous);

        // Используем computeIfAbsent() для получения "tablet" или установки 20
        Integer tabletCount = inventory.computeIfAbsent("tablet", k -> 20);
        System.out.println("Количество tablet (вычислено): " + tabletCount);

        // Увеличиваем количество "laptop" на 5 используя compute()
        inventory.compute("laptop", (k, v) -> v == null ? 5 : v + 5);
        System.out.println("Количество laptop после увеличения: " + inventory.get("laptop"));

        // Удаляем "mouse" если количество равно 50 используя remove(key, value)
        boolean removed = inventory.remove("mouse", 50);
        System.out.println("Mouse удален (количество было 50): " + removed);

        // Выводим все записи используя forEach()
        System.out.println("\nТекущий инвентарь:");
        inventory.forEach((product, count) ->
            System.out.println("  " + product + ": " + count)
        );

        // Демонстрируем потокобезопасность с несколькими потоками
        System.out.println("\nТестирование конкурентных обновлений...");
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("Финальное количество laptop после конкурентных обновлений: " + inventory.get("laptop"));
        System.out.println("Ожидается: 215 (15 + 100 + 100)");
    }
}`,
            description: `Изучите ConcurrentHashMap для потокобезопасных операций с картой.

**Требования:**
1. Создайте ConcurrentHashMap для хранения инвентаря продуктов
2. Добавьте продукты используя put(): "laptop"->10, "mouse"->50, "keyboard"->30
3. Используйте putIfAbsent() для добавления "monitor"->15 (только если отсутствует)
4. Используйте computeIfAbsent() для получения "tablet" или установки 20
5. Увеличьте количество "laptop" на 5 используя compute()
6. Удалите "mouse" если количество равно 50 используя remove(key, value)
7. Выведите все записи используя forEach()
8. Продемонстрируйте потокобезопасность обновлением из нескольких потоков

ConcurrentHashMap обеспечивает потокобезопасные операции без блокировки всей карты, позволяя высокую конкурентность.`,
            hint1: `putIfAbsent() возвращает null если ключ отсутствовал, или существующее значение если присутствует. Используйте для условной вставки.`,
            hint2: `computeIfAbsent() принимает функцию, которая вычисляет значение если отсутствует. compute() всегда применяет функцию, позволяя атомарные обновления.`,
            whyItMatters: `ConcurrentHashMap необходим для построения потокобезопасных кэшей и общих структур данных в многопоточных приложениях, обеспечивая лучшую производительность чем синхронизированный HashMap.

**Продакшен паттерн:**
\`\`\`java
// Кэш конфигурации приложения с потокобезопасными обновлениями
public class ConfigCache {
    private final ConcurrentHashMap<String, String> cache = new ConcurrentHashMap<>();

    public String getConfig(String key, Supplier<String> loader) {
        // Атомарное вычисление значения если отсутствует
        return cache.computeIfAbsent(key, k -> loader.get());
    }

    public void updateMetric(String metricName, int increment) {
        // Атомарное обновление счетчика метрик
        cache.compute(metricName, (k, v) -> {
            int current = v == null ? 0 : Integer.parseInt(v);
            return String.valueOf(current + increment);
        });
    }
}
\`\`\`

**Практические преимущества:**
- Высокая производительность при конкурентном доступе без блокировки всей карты
- Атомарные операции (computeIfAbsent, compute) для безопасных обновлений
- Идеален для кэшей, счетчиков, разделяемых конфигураций в веб-приложениях`
        },
        uz: {
            title: 'ConcurrentHashMap',
            solutionCode: `import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentHashMapDemo {
    public static void main(String[] args) throws InterruptedException {
        // Mahsulot inventari uchun ConcurrentHashMap yaratamiz
        ConcurrentHashMap<String, Integer> inventory = new ConcurrentHashMap<>();

        // put() yordamida mahsulotlarni qo'shamiz
        inventory.put("laptop", 10);
        inventory.put("mouse", 50);
        inventory.put("keyboard", 30);
        System.out.println("Boshlang'ich inventar: " + inventory);

        // "monitor"->15 qo'shish uchun putIfAbsent() dan foydalanamiz (faqat yo'q bo'lsa)
        Integer previous = inventory.putIfAbsent("monitor", 15);
        System.out.println("Monitor qo'shildi, oldingi qiymat: " + previous);

        // putIfAbsent() ni yana sinab ko'ramiz - mavjud qiymatni qaytarishi kerak
        previous = inventory.putIfAbsent("monitor", 20);
        System.out.println("Monitor yana qo'shishga urinish, mavjud qiymat: " + previous);

        // "tablet" ni olish yoki 20 qo'yish uchun computeIfAbsent() dan foydalanamiz
        Integer tabletCount = inventory.computeIfAbsent("tablet", k -> 20);
        System.out.println("Tablet soni (hisoblangan): " + tabletCount);

        // compute() yordamida "laptop" sonini 5 ga oshiramiz
        inventory.compute("laptop", (k, v) -> v == null ? 5 : v + 5);
        System.out.println("Oshirilgandan keyin laptop soni: " + inventory.get("laptop"));

        // Agar son 50 ga teng bo'lsa "mouse" ni o'chiramiz remove(key, value) yordamida
        boolean removed = inventory.remove("mouse", 50);
        System.out.println("Mouse o'chirildi (soni 50 edi): " + removed);

        // forEach() yordamida barcha yozuvlarni chiqaramiz
        System.out.println("\nJoriy inventar:");
        inventory.forEach((product, count) ->
            System.out.println("  " + product + ": " + count)
        );

        // Bir nechta oqim bilan thread-safety ni ko'rsatamiz
        System.out.println("\nKonkurrent yangilanishlarni sinash...");
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100; i++) {
                inventory.compute("laptop", (k, v) -> v + 1);
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("Konkurrent yangilanishlardan keyin yakuniy laptop soni: " + inventory.get("laptop"));
        System.out.println("Kutilgan: 215 (15 + 100 + 100)");
    }
}`,
            description: `Thread-safe map operatsiyalari uchun ConcurrentHashMap ni o'rganing.

**Talablar:**
1. Mahsulot inventari uchun ConcurrentHashMap yarating
2. put() yordamida mahsulotlarni qo'shing: "laptop"->10, "mouse"->50, "keyboard"->30
3. "monitor"->15 qo'shish uchun putIfAbsent() dan foydalaning (faqat yo'q bo'lsa)
4. "tablet" ni olish yoki 20 o'rnatish uchun computeIfAbsent() dan foydalaning
5. compute() yordamida "laptop" sonini 5 ga oshiring
6. Agar son 50 ga teng bo'lsa remove(key, value) yordamida "mouse" ni o'chiring
7. forEach() yordamida barcha yozuvlarni chiqaring
8. Bir nechta oqimdan yangilash orqali thread-safety ni ko'rsating

ConcurrentHashMap butun mapni qulflamasdan thread-safe operatsiyalarni ta'minlaydi va yuqori konkurrentlikka imkon beradi.`,
            hint1: `putIfAbsent() kalit yo'q bo'lsa null qaytaradi, yoki mavjud qiymatni qaytaradi. Shartli qo'shish uchun foydalaning.`,
            hint2: `computeIfAbsent() yo'q bo'lsa qiymatni hisoblaydigan funksiyani qabul qiladi. compute() har doim funksiyani qo'llaydi va atom yangilanishga imkon beradi.`,
            whyItMatters: `ConcurrentHashMap ko'p oqimli ilovalarda thread-safe keshlar va umumiy ma'lumot tuzilmalarini qurish uchun zarur bo'lib, sinxronlangan HashMap dan yaxshiroq ishlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Oqim xavfsiz yangilanishlari bilan dastur konfiguratsiya keshi
public class ConfigCache {
    private final ConcurrentHashMap<String, String> cache = new ConcurrentHashMap<>();

    public String getConfig(String key, Supplier<String> loader) {
        // Yo'q bo'lsa qiymatni atom tarzda hisoblash
        return cache.computeIfAbsent(key, k -> loader.get());
    }

    public void updateMetric(String metricName, int increment) {
        // Metrika schyotchigini atom tarzda yangilash
        cache.compute(metricName, (k, v) -> {
            int current = v == null ? 0 : Integer.parseInt(v);
            return String.valueOf(current + increment);
        });
    }
}
\`\`\`

**Amaliy foydalari:**
- Butun mapni qulflamasdan konkurrent kirishda yuqori ishlash
- Xavfsiz yangilanishlar uchun atom operatsiyalar (computeIfAbsent, compute)
- Veb-ilovalarda keshlar, schyotchiklar, umumiy konfiguratsiyalar uchun ideal`
        }
    }
};

export default task;
