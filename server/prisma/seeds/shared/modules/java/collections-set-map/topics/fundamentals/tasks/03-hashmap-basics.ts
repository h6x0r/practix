import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-hashmap-basics',
	title: 'HashMap Key-Value Operations',
	difficulty: 'easy',
	tags: ['java', 'collections', 'hashmap'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **FrequencyCounter** class using HashMap to count element occurrences.

**Requirements:**
1. Create \`FrequencyCounter<T>\` with HashMap to store counts
2. Implement \`add(T element)\` to increment count for element
3. Implement \`count(T element)\` to get count for element (0 if not present)
4. Implement \`mostFrequent()\` to return element with highest count

**Example:**
\`\`\`java
FrequencyCounter<String> counter = new FrequencyCounter<>();
counter.add("apple");
counter.add("banana");
counter.add("apple");
System.out.println(counter.count("apple"));  // 2
System.out.println(counter.count("orange")); // 0
System.out.println(counter.mostFrequent());  // apple
\`\`\`

**Key Concepts:**
- HashMap stores key-value pairs with O(1) access
- Use getOrDefault() for safe value retrieval
- Keys must properly implement hashCode() and equals()`,
	initialCode: `import java.util.HashMap;
import java.util.Map;

public class FrequencyCounter<T> {
    // TODO: Declare HashMap field

    public FrequencyCounter() {
        // TODO: Initialize HashMap
    }

    public void add(T element) {
        // TODO: Increment count for element
    }

    public int count(T element) {
        // TODO: Return count for element (0 if not present)
        return 0;
    }

    public T mostFrequent() {
        // TODO: Return element with highest count
        return null;
    }
}`,
	solutionCode: `import java.util.HashMap;
import java.util.Map;

public class FrequencyCounter<T> {
    private Map<T, Integer> counts;

    public FrequencyCounter() {
        this.counts = new HashMap<>();
    }

    public void add(T element) {
        counts.merge(element, 1, Integer::sum);  // Increment or set to 1
    }

    public int count(T element) {
        return counts.getOrDefault(element, 0);  // Safe retrieval
    }

    public T mostFrequent() {
        T result = null;
        int maxCount = 0;
        for (Map.Entry<T, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }
}`,
	hint1: `Use HashMap.merge(key, 1, Integer::sum) to safely increment counts - it handles both new and existing keys.`,
	hint2: `Use getOrDefault(key, 0) to return 0 for missing keys instead of null. Iterate entrySet() for mostFrequent.`,
	whyItMatters: `HashMap is the most versatile collection for key-value associations with fast operations.

**Why HashMap:**
- **O(1) Operations:** Constant time for get, put, containsKey
- **Flexible Keys:** Any object can be a key (with proper equals/hashCode)
- **Null Support:** Allows one null key and multiple null values

**Production Use Cases:**
\`\`\`java
// Configuration
Map<String, String> config = new HashMap<>();
config.put("timeout", "30s");

// Caching
Map<String, User> userCache = new HashMap<>();
userCache.computeIfAbsent(userId, this::loadUser);

// Counting
Map<String, Integer> wordCount = new HashMap<>();
wordCount.merge(word, 1, Integer::sum);
\`\`\`

**Important Methods:**
- \`getOrDefault(key, default)\` - safe retrieval
- \`computeIfAbsent(key, mapper)\` - lazy computation
- \`merge(key, value, remapping)\` - combine values

**Production Pattern:**
\`\`\`java
// Cache with lazy loading
Map<String, User> userCache = new HashMap<>();
User user = userCache.computeIfAbsent(userId, id -> loadUserFromDB(id));

// Data aggregation
Map<String, Integer> wordCount = new HashMap<>();
for (String word : document.split("\\s+")) {
    wordCount.merge(word, 1, Integer::sum);  // Increment counter
}

// Configuration with safe defaults
Map<String, String> config = new HashMap<>();
int timeout = Integer.parseInt(config.getOrDefault("timeout", "30"));
\`\`\`

**Practical Benefits:**
- computeIfAbsent for caching with lazy loading
- merge for elegant counting and aggregation
- getOrDefault for safe handling of missing keys`,
	order: 2,
	testCode: `import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: FrequencyCounter can be created
class Test1 {
    @Test
    void testCreation() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        assertNotNull(counter);
    }
}

// Test2: add() increments count
class Test2 {
    @Test
    void testAdd() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("apple");
        assertEquals(1, counter.count("apple"));
    }
}

// Test3: Multiple adds increment count
class Test3 {
    @Test
    void testMultipleAdds() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("apple");
        counter.add("apple");
        counter.add("apple");
        assertEquals(3, counter.count("apple"));
    }
}

// Test4: count() returns 0 for missing element
class Test4 {
    @Test
    void testCountMissing() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("apple");
        assertEquals(0, counter.count("banana"));
    }
}

// Test5: mostFrequent() returns most common element
class Test5 {
    @Test
    void testMostFrequent() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("apple");
        counter.add("banana");
        counter.add("apple");
        assertEquals("apple", counter.mostFrequent());
    }
}

// Test6: Works with Integer type
class Test6 {
    @Test
    void testWithIntegers() {
        FrequencyCounter<Integer> counter = new FrequencyCounter<>();
        counter.add(1);
        counter.add(2);
        counter.add(1);
        assertEquals(2, counter.count(1));
        assertEquals(1, counter.count(2));
    }
}

// Test7: mostFrequent() returns null for empty counter
class Test7 {
    @Test
    void testMostFrequentEmpty() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        assertNull(counter.mostFrequent());
    }
}

// Test8: Different elements have separate counts
class Test8 {
    @Test
    void testSeparateCounts() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("a");
        counter.add("b");
        counter.add("b");
        counter.add("c");
        counter.add("c");
        counter.add("c");
        assertEquals(1, counter.count("a"));
        assertEquals(2, counter.count("b"));
        assertEquals(3, counter.count("c"));
    }
}

// Test9: mostFrequent() updates with new adds
class Test9 {
    @Test
    void testMostFrequentUpdates() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        counter.add("a");
        counter.add("a");
        assertEquals("a", counter.mostFrequent());
        counter.add("b");
        counter.add("b");
        counter.add("b");
        assertEquals("b", counter.mostFrequent());
    }
}

// Test10: Large number of elements
class Test10 {
    @Test
    void testLargeCount() {
        FrequencyCounter<String> counter = new FrequencyCounter<>();
        for (int i = 0; i < 100; i++) {
            counter.add("item");
        }
        assertEquals(100, counter.count("item"));
    }
}
`,
	translations: {
		ru: {
			title: 'Операции Ключ-Значение в HashMap',
			description: `Реализуйте класс **FrequencyCounter** с использованием HashMap для подсчета вхождений элементов.

**Требования:**
1. Создайте \`FrequencyCounter<T>\` с HashMap для хранения счетчиков
2. Реализуйте \`add(T element)\` для инкремента счетчика элемента
3. Реализуйте \`count(T element)\` для получения счетчика (0 если отсутствует)
4. Реализуйте \`mostFrequent()\` для возврата элемента с максимальным счетчиком

**Пример:**
\`\`\`java
FrequencyCounter<String> counter = new FrequencyCounter<>();
counter.add("apple");
counter.add("banana");
counter.add("apple");
System.out.println(counter.count("apple"));  // 2
System.out.println(counter.mostFrequent());  // apple
\`\`\`

**Ключевые концепции:**
- HashMap хранит пары ключ-значение с O(1) доступом
- Используйте getOrDefault() для безопасного получения значения
- Ключи должны правильно реализовывать hashCode() и equals()`,
			hint1: `Используйте HashMap.merge(key, 1, Integer::sum) для безопасного инкремента - обрабатывает новые и существующие ключи.`,
			hint2: `Используйте getOrDefault(key, 0) для возврата 0 для отсутствующих ключей. Итерируйте entrySet() для mostFrequent.`,
			whyItMatters: `HashMap - самая универсальная коллекция для ассоциаций ключ-значение с быстрыми операциями.

**Почему HashMap:**
- **O(1) операции:** Константное время для get, put, containsKey
- **Гибкие ключи:** Любой объект может быть ключом
- **Поддержка null:** Допускает один null ключ и множество null значений

**Использование в production:**
- Конфигурация приложения
- Кэширование данных
- Подсчет частоты

**Важные методы:**
- getOrDefault(), computeIfAbsent(), merge()

**Продакшен паттерн:**
\`\`\`java
// Кэш с ленивой загрузкой
Map<String, User> userCache = new HashMap<>();
User user = userCache.computeIfAbsent(userId, id -> loadUserFromDB(id));

// Агрегация данных
Map<String, Integer> wordCount = new HashMap<>();
for (String word : document.split("\\s+")) {
    wordCount.merge(word, 1, Integer::sum);  // Инкремент счетчика
}

// Конфигурация с безопасными значениями по умолчанию
Map<String, String> config = new HashMap<>();
int timeout = Integer.parseInt(config.getOrDefault("timeout", "30"));
\`\`\`

**Практические преимущества:**
- computeIfAbsent для кэширования с ленивой загрузкой
- merge для элегантного подсчета и агрегации
- getOrDefault для безопасной работы с отсутствующими ключами`,
			solutionCode: `import java.util.HashMap;
import java.util.Map;

public class FrequencyCounter<T> {
    private Map<T, Integer> counts;

    public FrequencyCounter() {
        this.counts = new HashMap<>();
    }

    public void add(T element) {
        counts.merge(element, 1, Integer::sum);  // Инкремент или установка в 1
    }

    public int count(T element) {
        return counts.getOrDefault(element, 0);  // Безопасное получение
    }

    public T mostFrequent() {
        T result = null;
        int maxCount = 0;
        for (Map.Entry<T, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }
}`
		},
		uz: {
			title: 'HashMap Kalit-Qiymat Operatsiyalari',
			description: `Element takrorlanishlarini hisoblash uchun HashMap ishlatib **FrequencyCounter** klassini amalga oshiring.

**Talablar:**
1. Hisoblagichlarni saqlash uchun HashMap bilan \`FrequencyCounter<T>\` yarating
2. Element hisoblagichini oshirish uchun \`add(T element)\` ni amalga oshiring
3. Element hisoblagichini olish uchun \`count(T element)\` ni amalga oshiring (0 agar yo'q bo'lsa)
4. Eng yuqori hisoblagichli elementni qaytarish uchun \`mostFrequent()\` ni amalga oshiring

**Misol:**
\`\`\`java
FrequencyCounter<String> counter = new FrequencyCounter<>();
counter.add("apple");
counter.add("banana");
counter.add("apple");
System.out.println(counter.count("apple"));  // 2
System.out.println(counter.mostFrequent());  // apple
\`\`\`

**Asosiy tushunchalar:**
- HashMap O(1) kirish bilan kalit-qiymat juftlarini saqlaydi
- Xavfsiz qiymat olish uchun getOrDefault() dan foydalaning
- Kalitlar hashCode() va equals() ni to'g'ri amalga oshirishi kerak`,
			hint1: `Xavfsiz oshirish uchun HashMap.merge(key, 1, Integer::sum) dan foydalaning - yangi va mavjud kalitlarni boshqaradi.`,
			hint2: `Yo'q kalitlar uchun null o'rniga 0 qaytarish uchun getOrDefault(key, 0) dan foydalaning.`,
			whyItMatters: `HashMap tez operatsiyalar bilan kalit-qiymat bog'lanishlari uchun eng ko'p qirrali kolleksiya.

**Nima uchun HashMap:**
- **O(1) operatsiyalar:** get, put, containsKey uchun doimiy vaqt
- **Moslashuvchan kalitlar:** Har qanday obyekt kalit bo'lishi mumkin
- **null qo'llab-quvvatlash:** Bitta null kalit va ko'p null qiymatlariga ruxsat beradi

**Production'da foydalanish:**
- Konfiguratsiya
- Ma'lumotlarni keshlash
- Chastotani hisoblash

**Muhim metodlar:**
- getOrDefault(), computeIfAbsent(), merge()

**Ishlab chiqarish patterni:**
\`\`\`java
// Lazy loading bilan kesh
Map<String, User> userCache = new HashMap<>();
User user = userCache.computeIfAbsent(userId, id -> loadUserFromDB(id));

// Ma'lumotlar agregatsiyasi
Map<String, Integer> wordCount = new HashMap<>();
for (String word : document.split("\\s+")) {
    wordCount.merge(word, 1, Integer::sum);  // Hisoblagichni oshirish
}

// Xavfsiz standart qiymatlar bilan konfiguratsiya
Map<String, String> config = new HashMap<>();
int timeout = Integer.parseInt(config.getOrDefault("timeout", "30"));
\`\`\`

**Amaliy foydalari:**
- Lazy loading bilan keshlash uchun computeIfAbsent
- Nafis hisoblash va agregatsiya uchun merge
- Yo'q kalitlar bilan xavfsiz ishlash uchun getOrDefault`,
			solutionCode: `import java.util.HashMap;
import java.util.Map;

public class FrequencyCounter<T> {
    private Map<T, Integer> counts;

    public FrequencyCounter() {
        this.counts = new HashMap<>();
    }

    public void add(T element) {
        counts.merge(element, 1, Integer::sum);  // Oshirish yoki 1 ga o'rnatish
    }

    public int count(T element) {
        return counts.getOrDefault(element, 0);  // Xavfsiz olish
    }

    public T mostFrequent() {
        T result = null;
        int maxCount = 0;
        for (Map.Entry<T, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }
}`
		}
	}
};

export default task;
