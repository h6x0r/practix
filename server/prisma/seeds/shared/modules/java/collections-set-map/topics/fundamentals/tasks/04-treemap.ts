import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-treemap',
	title: 'TreeMap and Sorted Maps',
	difficulty: 'medium',
	tags: ['java', 'collections', 'treemap', 'sorting'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **RangeMap** class using TreeMap for range-based lookups.

**Requirements:**
1. Create \`RangeMap<K extends Comparable<K>, V>\` with TreeMap storage
2. Implement \`put(K key, V value)\` to store key-value pair
3. Implement \`getFloor(K key)\` to get value for largest key <= given key
4. Implement \`getCeiling(K key)\` to get value for smallest key >= given key
5. Implement \`rangeValues(K from, K to)\` to get all values in range

**Example:**
\`\`\`java
RangeMap<Integer, String> map = new RangeMap<>();
map.put(10, "ten");
map.put(20, "twenty");
map.put(30, "thirty");
System.out.println(map.getFloor(15));   // ten (key 10)
System.out.println(map.getCeiling(15)); // twenty (key 20)
System.out.println(map.rangeValues(10, 25)); // [ten, twenty]
\`\`\`

**Key Concepts:**
- TreeMap maintains keys in sorted order
- NavigableMap provides floor, ceiling, lower, higher operations
- O(log n) for all operations`,
	initialCode: `import java.util.TreeMap;
import java.util.NavigableMap;
import java.util.List;
import java.util.ArrayList;

public class RangeMap<K extends Comparable<K>, V> {
    // TODO: Declare TreeMap field

    public RangeMap() {
        // TODO: Initialize TreeMap
    }

    public void put(K key, V value) {
        // TODO: Store key-value pair
    }

    public V getFloor(K key) {
        // TODO: Return value for largest key <= given key
        return null;
    }

    public V getCeiling(K key) {
        // TODO: Return value for smallest key >= given key
        return null;
    }

    public List<V> rangeValues(K from, K to) {
        // TODO: Return all values in range [from, to)
        return null;
    }
}`,
	solutionCode: `import java.util.TreeMap;
import java.util.NavigableMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class RangeMap<K extends Comparable<K>, V> {
    private NavigableMap<K, V> map;

    public RangeMap() {
        this.map = new TreeMap<>();
    }

    public void put(K key, V value) {
        map.put(key, value);
    }

    public V getFloor(K key) {
        Map.Entry<K, V> entry = map.floorEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public V getCeiling(K key) {
        Map.Entry<K, V> entry = map.ceilingEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public List<V> rangeValues(K from, K to) {
        return new ArrayList<>(map.subMap(from, to).values());
    }
}`,
	hint1: `Use TreeMap's floorEntry(key) and ceilingEntry(key) methods to find the nearest entries.`,
	hint2: `Use subMap(from, to) to get a view of entries in range, then extract values() into a new ArrayList.`,
	whyItMatters: `TreeMap is essential for ordered key-value storage with range query capabilities.

**Why TreeMap:**
- **Sorted Keys:** Always maintained in order
- **Range Queries:** Efficient subMap, headMap, tailMap
- **NavigableMap API:** floor, ceiling, higher, lower entries

**Production Use Cases:**
\`\`\`java
// IP range lookup
TreeMap<Long, String> ipRanges = new TreeMap<>();
ipRanges.put(ipToLong("10.0.0.0"), "internal");
String network = ipRanges.floorEntry(clientIP).getValue();

// Version compatibility
TreeMap<Version, Config> configs = new TreeMap<>();
Config compatible = configs.floorEntry(clientVersion).getValue();

// Time-series data
TreeMap<Instant, Double> readings = new TreeMap<>();
Double latestValue = readings.lastEntry().getValue();
\`\`\`

**TreeMap vs HashMap:**
- HashMap: O(1) but unordered keys
- TreeMap: O(log n) but sorted keys with range operations

**Production Pattern:**
\`\`\`java
// IP addresses to regions with efficient lookup
TreeMap<Long, String> ipToRegion = new TreeMap<>();
ipToRegion.put(ipToLong("192.168.0.0"), "internal");
ipToRegion.put(ipToLong("10.0.0.0"), "vpn");
String region = ipToRegion.floorEntry(clientIpLong).getValue();

// API versions with backward compatibility
TreeMap<Version, ApiHandler> apiVersions = new TreeMap<>();
ApiHandler handler = apiVersions.floorEntry(requestVersion).getValue();
\`\`\`

**Practical Benefits:**
- Efficient range lookup for IP addresses and versions
- floor/ceiling for finding compatible versions
- Automatic key sorting without overhead`,
	order: 3,
	testCode: `import java.util.TreeMap;
import java.util.NavigableMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: RangeMap can be created
class Test1 {
    @Test
    void testCreation() {
        RangeMap<Integer, String> map = new RangeMap<>();
        assertNotNull(map);
    }
}

// Test2: put() stores key-value pair
class Test2 {
    @Test
    void testPut() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        assertEquals("ten", map.getFloor(10));
    }
}

// Test3: getFloor() returns value for largest key <= given
class Test3 {
    @Test
    void testGetFloor() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        map.put(30, "thirty");
        assertEquals("ten", map.getFloor(15));
    }
}

// Test4: getCeiling() returns value for smallest key >= given
class Test4 {
    @Test
    void testGetCeiling() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        map.put(30, "thirty");
        assertEquals("twenty", map.getCeiling(15));
    }
}

// Test5: rangeValues() returns values in range
class Test5 {
    @Test
    void testRangeValues() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        map.put(30, "thirty");
        List<String> values = map.rangeValues(10, 25);
        assertEquals(2, values.size());
        assertTrue(values.contains("ten"));
        assertTrue(values.contains("twenty"));
    }
}

// Test6: getFloor() returns null if no floor exists
class Test6 {
    @Test
    void testGetFloorNull() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(20, "twenty");
        assertNull(map.getFloor(10));
    }
}

// Test7: getCeiling() returns null if no ceiling exists
class Test7 {
    @Test
    void testGetCeilingNull() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        assertNull(map.getCeiling(20));
    }
}

// Test8: Exact key match works for floor
class Test8 {
    @Test
    void testExactFloor() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        assertEquals("twenty", map.getFloor(20));
    }
}

// Test9: Exact key match works for ceiling
class Test9 {
    @Test
    void testExactCeiling() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        assertEquals("ten", map.getCeiling(10));
    }
}

// Test10: rangeValues() excludes upper bound
class Test10 {
    @Test
    void testRangeExcludesUpperBound() {
        RangeMap<Integer, String> map = new RangeMap<>();
        map.put(10, "ten");
        map.put(20, "twenty");
        map.put(30, "thirty");
        List<String> values = map.rangeValues(10, 30);
        assertEquals(2, values.size());
        assertFalse(values.contains("thirty"));
    }
}
`,
	translations: {
		ru: {
			title: 'TreeMap и Отсортированные Карты',
			description: `Реализуйте класс **RangeMap** с использованием TreeMap для поиска по диапазону.

**Требования:**
1. Создайте \`RangeMap<K extends Comparable<K>, V>\` с хранилищем TreeMap
2. Реализуйте \`put(K key, V value)\` для хранения пары ключ-значение
3. Реализуйте \`getFloor(K key)\` для получения значения наибольшего ключа <= данного
4. Реализуйте \`getCeiling(K key)\` для получения значения наименьшего ключа >= данного
5. Реализуйте \`rangeValues(K from, K to)\` для получения всех значений в диапазоне

**Пример:**
\`\`\`java
RangeMap<Integer, String> map = new RangeMap<>();
map.put(10, "ten");
map.put(20, "twenty");
System.out.println(map.getFloor(15));   // ten
System.out.println(map.getCeiling(15)); // twenty
\`\`\`

**Ключевые концепции:**
- TreeMap поддерживает ключи в отсортированном порядке
- NavigableMap предоставляет floor, ceiling операции
- O(log n) для всех операций`,
			hint1: `Используйте методы TreeMap floorEntry(key) и ceilingEntry(key) для поиска ближайших записей.`,
			hint2: `Используйте subMap(from, to) для получения записей в диапазоне, затем извлеките values().`,
			whyItMatters: `TreeMap необходим для упорядоченного хранения ключ-значение с возможностью запросов диапазона.

**Почему TreeMap:**
- **Отсортированные ключи:** Всегда поддерживаются в порядке
- **Запросы диапазона:** Эффективные subMap, headMap, tailMap
- **NavigableMap API:** floor, ceiling, higher, lower записи

**Использование в production:**
- Поиск диапазонов IP
- Совместимость версий
- Данные временных рядов

**TreeMap vs HashMap:**
- HashMap: O(1) но без порядка
- TreeMap: O(log n) но с сортировкой и операциями диапазона

**Продакшен паттерн:**
\`\`\`java
// IP-адреса к регионам с эффективным поиском
TreeMap<Long, String> ipToRegion = new TreeMap<>();
ipToRegion.put(ipToLong("192.168.0.0"), "internal");
ipToRegion.put(ipToLong("10.0.0.0"), "vpn");
String region = ipToRegion.floorEntry(clientIpLong).getValue();

// Версии API с обратной совместимостью
TreeMap<Version, ApiHandler> apiVersions = new TreeMap<>();
ApiHandler handler = apiVersions.floorEntry(requestVersion).getValue();
\`\`\`

**Практические преимущества:**
- Эффективный range lookup для IP-адресов и версий
- floor/ceiling для поиска совместимых версий
- Автоматическая сортировка ключей без дополнительных затрат`,
			solutionCode: `import java.util.TreeMap;
import java.util.NavigableMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class RangeMap<K extends Comparable<K>, V> {
    private NavigableMap<K, V> map;

    public RangeMap() {
        this.map = new TreeMap<>();
    }

    public void put(K key, V value) {
        map.put(key, value);
    }

    public V getFloor(K key) {
        Map.Entry<K, V> entry = map.floorEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public V getCeiling(K key) {
        Map.Entry<K, V> entry = map.ceilingEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public List<V> rangeValues(K from, K to) {
        return new ArrayList<>(map.subMap(from, to).values());
    }
}`
		},
		uz: {
			title: 'TreeMap va Tartiblangan Xaritalar',
			description: `Diapazon bo'yicha qidirish uchun TreeMap ishlatib **RangeMap** klassini amalga oshiring.

**Talablar:**
1. TreeMap saqlash bilan \`RangeMap<K extends Comparable<K>, V>\` yarating
2. Kalit-qiymat juftini saqlash uchun \`put(K key, V value)\` ni amalga oshiring
3. Berilgan kalitdan <= eng katta kalit qiymatini olish uchun \`getFloor(K key)\` ni amalga oshiring
4. Berilgan kalitdan >= eng kichik kalit qiymatini olish uchun \`getCeiling(K key)\` ni amalga oshiring
5. Diapazon ichidagi barcha qiymatlarni olish uchun \`rangeValues(K from, K to)\` ni amalga oshiring

**Misol:**
\`\`\`java
RangeMap<Integer, String> map = new RangeMap<>();
map.put(10, "ten");
map.put(20, "twenty");
System.out.println(map.getFloor(15));   // ten
System.out.println(map.getCeiling(15)); // twenty
\`\`\`

**Asosiy tushunchalar:**
- TreeMap kalitlarni tartiblangan holatda saqlaydi
- NavigableMap floor, ceiling operatsiyalarini taqdim etadi
- Barcha operatsiyalar uchun O(log n)`,
			hint1: `Eng yaqin yozuvlarni topish uchun TreeMap ning floorEntry(key) va ceilingEntry(key) metodlaridan foydalaning.`,
			hint2: `Diapazon yozuvlarini olish uchun subMap(from, to) dan, so'ng values() ni yangi ArrayList ga chiqaring.`,
			whyItMatters: `TreeMap diapazon so'rovlari imkoniyati bilan tartiblangan kalit-qiymat saqlash uchun zarur.

**Nima uchun TreeMap:**
- **Tartiblangan kalitlar:** Doimo tartibda saqlanadi
- **Diapazon so'rovlari:** Samarali subMap, headMap, tailMap
- **NavigableMap API:** floor, ceiling, higher, lower yozuvlar

**Production'da foydalanish:**
- IP diapazon qidirish
- Versiya mosligi
- Vaqt seriyalari ma'lumotlari

**TreeMap vs HashMap:**
- HashMap: O(1) lekin tartibsiz
- TreeMap: O(log n) lekin tartiblangan

**Ishlab chiqarish patterni:**
\`\`\`java
// Samarali qidiruv bilan IP-manzillardan hududlarga
TreeMap<Long, String> ipToRegion = new TreeMap<>();
ipToRegion.put(ipToLong("192.168.0.0"), "internal");
ipToRegion.put(ipToLong("10.0.0.0"), "vpn");
String region = ipToRegion.floorEntry(clientIpLong).getValue();

// Orqaga moslik bilan API versiyalari
TreeMap<Version, ApiHandler> apiVersions = new TreeMap<>();
ApiHandler handler = apiVersions.floorEntry(requestVersion).getValue();
\`\`\`

**Amaliy foydalari:**
- IP-manzillar va versiyalar uchun samarali range lookup
- Mos versiyalarni qidirish uchun floor/ceiling
- Qo'shimcha xarajatlarsiz kalitlarni avtomatik saralash`,
			solutionCode: `import java.util.TreeMap;
import java.util.NavigableMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class RangeMap<K extends Comparable<K>, V> {
    private NavigableMap<K, V> map;

    public RangeMap() {
        this.map = new TreeMap<>();
    }

    public void put(K key, V value) {
        map.put(key, value);
    }

    public V getFloor(K key) {
        Map.Entry<K, V> entry = map.floorEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public V getCeiling(K key) {
        Map.Entry<K, V> entry = map.ceilingEntry(key);
        return entry != null ? entry.getValue() : null;
    }

    public List<V> rangeValues(K from, K to) {
        return new ArrayList<>(map.subMap(from, to).values());
    }
}`
		}
	}
};

export default task;
