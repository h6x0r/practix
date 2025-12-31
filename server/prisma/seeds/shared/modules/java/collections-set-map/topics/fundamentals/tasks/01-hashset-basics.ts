import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-hashset-basics',
	title: 'HashSet Fundamentals',
	difficulty: 'easy',
	tags: ['java', 'collections', 'hashset'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **UniqueCounter** class using HashSet to track unique elements.

**Requirements:**
1. Create \`UniqueCounter<T>\` class with a HashSet to store unique elements
2. Implement \`add(T element)\` method that returns true if element was new
3. Implement \`count()\` method to return number of unique elements
4. Implement \`contains(T element)\` method to check existence

**Example:**
\`\`\`java
UniqueCounter<String> counter = new UniqueCounter<>();
System.out.println(counter.add("Apple"));  // true
System.out.println(counter.add("Banana")); // true
System.out.println(counter.add("Apple"));  // false (duplicate)
System.out.println(counter.count());       // 2
System.out.println(counter.contains("Apple")); // true
\`\`\`

**Key Concepts:**
- HashSet does not allow duplicates
- HashSet provides O(1) time for add, remove, contains operations
- For custom objects, must override equals() and hashCode()`,
	initialCode: `import java.util.HashSet;
import java.util.Set;

public class UniqueCounter<T> {
    // TODO: Declare HashSet field

    public UniqueCounter() {
        // TODO: Initialize HashSet
    }

    public boolean add(T element) {
        // TODO: Add element and return true if new
        return false;
    }

    public int count() {
        // TODO: Return number of unique elements
        return 0;
    }

    public boolean contains(T element) {
        // TODO: Check if element exists
        return false;
    }
}`,
	solutionCode: `import java.util.HashSet;
import java.util.Set;

public class UniqueCounter<T> {
    private Set<T> elements;              // HashSet to store unique elements

    public UniqueCounter() {
        this.elements = new HashSet<>();  // Initialize empty HashSet
    }

    public boolean add(T element) {
        return elements.add(element);     // Returns false if duplicate
    }

    public int count() {
        return elements.size();           // O(1) operation
    }

    public boolean contains(T element) {
        return elements.contains(element); // O(1) lookup
    }
}`,
	hint1: `HashSet.add() returns a boolean indicating whether the element was added (true) or was already present (false).`,
	hint2: `Use HashSet.size() to get the count and HashSet.contains() for existence check - both are O(1) operations.`,
	whyItMatters: `HashSet is the go-to collection for tracking unique elements with fast operations.

**Why HashSet:**
- **No Duplicates:** Automatically rejects duplicate elements
- **O(1) Performance:** Constant time for add, remove, contains operations
- **Memory Efficient:** Only stores unique values, no index overhead

**Production Use Cases:**
\`\`\`java
// Deduplication
Set<String> uniqueEmails = new HashSet<>(allEmails);

// Membership check
Set<String> blockedIPs = new HashSet<>(loadBlocked());
if (blockedIPs.contains(requestIP)) { reject(); }

// Counting unique visitors
Set<String> uniqueVisitors = new HashSet<>();
uniqueVisitors.add(userId);
\`\`\`

**Important:** For custom objects, you MUST override equals() and hashCode() for HashSet to work correctly. Without these, each object is considered unique even with same field values.

**Production Pattern:**
\`\`\`java
// Filtering duplicates from data stream
Set<String> seenIds = new HashSet<>();
for (Record record : dataStream) {
    if (seenIds.add(record.getId())) {  // add returns false for duplicate
        processRecord(record);
    }
}

// Access control
Set<String> allowedUsers = new HashSet<>(loadAllowedUsers());
if (allowedUsers.contains(userId)) {
    grantAccess();
}
\`\`\`

**Practical Benefits:**
- Instant deduplication without sorting
- O(1) membership check for access control
- Automatic rejection of duplicates`,
	order: 0,
	testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify UniqueCounter creation
class Test1 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        assertNotNull("UniqueCounter should be created", counter);
        assertEquals("Initial count should be 0", 0, counter.count());
    }
}

// Test2: Verify adding unique elements
class Test2 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        assertTrue("Adding Apple should return true", counter.add("Apple"));
        assertTrue("Adding Banana should return true", counter.add("Banana"));
        assertEquals("Count should be 2", 2, counter.count());
    }
}

// Test3: Verify duplicate detection
class Test3 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        assertTrue("First add should return true", counter.add("Apple"));
        assertFalse("Duplicate add should return false", counter.add("Apple"));
        assertEquals("Count should be 1", 1, counter.count());
    }
}

// Test4: Verify contains method
class Test4 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        counter.add("Apple");
        assertTrue("Should contain Apple", counter.contains("Apple"));
        assertFalse("Should not contain Banana", counter.contains("Banana"));
    }
}

// Test5: Verify count method
class Test5 {
    @Test
    public void test() {
        UniqueCounter<Integer> counter = new UniqueCounter<>();
        counter.add(1);
        counter.add(2);
        counter.add(3);
        counter.add(2); // duplicate
        assertEquals("Count should be 3", 3, counter.count());
    }
}

// Test6: Verify with different types
class Test6 {
    @Test
    public void test() {
        UniqueCounter<Integer> intCounter = new UniqueCounter<>();
        assertTrue("Adding 100 should work", intCounter.add(100));
        assertTrue("Adding 200 should work", intCounter.add(200));
        assertEquals("Count should be 2", 2, intCounter.count());
    }
}

// Test7: Verify multiple duplicates
class Test7 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        counter.add("A");
        counter.add("A");
        counter.add("A");
        assertEquals("Count should be 1 despite multiple adds", 1, counter.count());
    }
}

// Test8: Verify empty counter
class Test8 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        assertEquals("Empty counter should have count 0", 0, counter.count());
        assertFalse("Empty counter should not contain anything", counter.contains("test"));
    }
}

// Test9: Verify large dataset
class Test9 {
    @Test
    public void test() {
        UniqueCounter<Integer> counter = new UniqueCounter<>();
        for (int i = 0; i < 100; i++) {
            counter.add(i);
        }
        assertEquals("Should have 100 unique elements", 100, counter.count());
    }
}

// Test10: Verify mixed operations
class Test10 {
    @Test
    public void test() {
        UniqueCounter<String> counter = new UniqueCounter<>();
        assertTrue("Add Apple", counter.add("Apple"));
        assertTrue("Add Banana", counter.add("Banana"));
        assertFalse("Add duplicate Apple", counter.add("Apple"));
        assertTrue("Contains Apple", counter.contains("Apple"));
        assertTrue("Contains Banana", counter.contains("Banana"));
        assertFalse("Does not contain Orange", counter.contains("Orange"));
        assertEquals("Final count is 2", 2, counter.count());
    }
}
`,
	translations: {
		ru: {
			title: 'Основы HashSet',
			description: `Реализуйте класс **UniqueCounter** с использованием HashSet для отслеживания уникальных элементов.

**Требования:**
1. Создайте класс \`UniqueCounter<T>\` с HashSet для хранения уникальных элементов
2. Реализуйте метод \`add(T element)\`, возвращающий true если элемент новый
3. Реализуйте метод \`count()\` для возврата количества уникальных элементов
4. Реализуйте метод \`contains(T element)\` для проверки существования

**Пример:**
\`\`\`java
UniqueCounter<String> counter = new UniqueCounter<>();
System.out.println(counter.add("Apple"));  // true
System.out.println(counter.add("Banana")); // true
System.out.println(counter.add("Apple"));  // false (дубликат)
System.out.println(counter.count());       // 2
\`\`\`

**Ключевые концепции:**
- HashSet не допускает дубликаты
- HashSet обеспечивает O(1) время для add, remove, contains
- Для пользовательских объектов нужно переопределить equals() и hashCode()`,
			hint1: `HashSet.add() возвращает boolean, показывающий был ли элемент добавлен (true) или уже существовал (false).`,
			hint2: `Используйте HashSet.size() для подсчета и HashSet.contains() для проверки существования - обе операции O(1).`,
			whyItMatters: `HashSet - основная коллекция для отслеживания уникальных элементов с быстрыми операциями.

**Почему HashSet:**
- **Нет дубликатов:** Автоматически отклоняет повторяющиеся элементы
- **O(1) производительность:** Константное время для add, remove, contains
- **Эффективность памяти:** Хранит только уникальные значения

**Использование в production:**
- Дедупликация данных
- Проверка членства (blocked IPs, allowed users)
- Подсчет уникальных посетителей

**Важно:** Для пользовательских объектов НЕОБХОДИМО переопределить equals() и hashCode().

**Продакшен паттерн:**
\`\`\`java
// Фильтрация дубликатов из потока данных
Set<String> seenIds = new HashSet<>();
for (Record record : dataStream) {
    if (seenIds.add(record.getId())) {  // add возвращает false для дубликата
        processRecord(record);
    }
}

// Проверка доступа
Set<String> allowedUsers = new HashSet<>(loadAllowedUsers());
if (allowedUsers.contains(userId)) {
    grantAccess();
}
\`\`\`

**Практические преимущества:**
- Мгновенная дедупликация без сортировки
- O(1) проверка членства для контроля доступа
- Автоматическое отклонение дубликатов`,
			solutionCode: `import java.util.HashSet;
import java.util.Set;

public class UniqueCounter<T> {
    private Set<T> elements;              // HashSet для хранения уникальных элементов

    public UniqueCounter() {
        this.elements = new HashSet<>();  // Инициализация пустого HashSet
    }

    public boolean add(T element) {
        return elements.add(element);     // Возвращает false для дубликата
    }

    public int count() {
        return elements.size();           // O(1) операция
    }

    public boolean contains(T element) {
        return elements.contains(element); // O(1) поиск
    }
}`
		},
		uz: {
			title: 'HashSet Asoslari',
			description: `Unikal elementlarni kuzatish uchun HashSet ishlatib **UniqueCounter** klassini amalga oshiring.

**Talablar:**
1. Unikal elementlarni saqlash uchun HashSet bilan \`UniqueCounter<T>\` klassini yarating
2. Element yangi bo'lsa true qaytaruvchi \`add(T element)\` metodini amalga oshiring
3. Unikal elementlar sonini qaytaruvchi \`count()\` metodini amalga oshiring
4. Mavjudlikni tekshiruvchi \`contains(T element)\` metodini amalga oshiring

**Misol:**
\`\`\`java
UniqueCounter<String> counter = new UniqueCounter<>();
System.out.println(counter.add("Apple"));  // true
System.out.println(counter.add("Banana")); // true
System.out.println(counter.add("Apple"));  // false (dublikat)
System.out.println(counter.count());       // 2
\`\`\`

**Asosiy tushunchalar:**
- HashSet dublikatlarga ruxsat bermaydi
- HashSet add, remove, contains uchun O(1) vaqtni ta'minlaydi
- Maxsus obyektlar uchun equals() va hashCode() ni qayta belgilash kerak`,
			hint1: `HashSet.add() element qo'shildi (true) yoki allaqachon mavjud edi (false) ko'rsatuvchi boolean qaytaradi.`,
			hint2: `Hisoblash uchun HashSet.size() va mavjudlikni tekshirish uchun HashSet.contains() dan foydalaning - ikkala operatsiya ham O(1).`,
			whyItMatters: `HashSet tez operatsiyalar bilan unikal elementlarni kuzatish uchun asosiy kolleksiya.

**Nima uchun HashSet:**
- **Dublikatsiz:** Takroriy elementlarni avtomatik rad etadi
- **O(1) unumdorlik:** add, remove, contains uchun doimiy vaqt
- **Xotira samaradorligi:** Faqat unikal qiymatlarni saqlaydi

**Production'da foydalanish:**
- Ma'lumotlarni deduplikatsiya qilish
- A'zolikni tekshirish (bloklangan IP lar, ruxsat berilgan foydalanuvchilar)
- Unikal tashrifchilarni hisoblash

**Muhim:** Maxsus obyektlar uchun equals() va hashCode() ni qayta belgilash SHART.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ma'lumotlar oqimidan dublikatlarni filtrlash
Set<String> seenIds = new HashSet<>();
for (Record record : dataStream) {
    if (seenIds.add(record.getId())) {  // add dublikat uchun false qaytaradi
        processRecord(record);
    }
}

// Kirish huquqini tekshirish
Set<String> allowedUsers = new HashSet<>(loadAllowedUsers());
if (allowedUsers.contains(userId)) {
    grantAccess();
}
\`\`\`

**Amaliy foydalari:**
- Saralashsiz bir zumda deduplikatsiya
- Kirish nazorati uchun O(1) a'zolik tekshiruvi
- Dublikatlarni avtomatik rad etish`,
			solutionCode: `import java.util.HashSet;
import java.util.Set;

public class UniqueCounter<T> {
    private Set<T> elements;              // Unikal elementlarni saqlash uchun HashSet

    public UniqueCounter() {
        this.elements = new HashSet<>();  // Bo'sh HashSet ni ishga tushirish
    }

    public boolean add(T element) {
        return elements.add(element);     // Dublikat uchun false qaytaradi
    }

    public int count() {
        return elements.size();           // O(1) operatsiya
    }

    public boolean contains(T element) {
        return elements.contains(element); // O(1) qidiruv
    }
}`
		}
	}
};

export default task;
