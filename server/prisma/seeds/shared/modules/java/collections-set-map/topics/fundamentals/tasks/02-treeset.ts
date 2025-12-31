import { Task } from '../../../../types';

export const task: Task = {
	slug: 'java-treeset',
	title: 'TreeSet and Natural Ordering',
	difficulty: 'easy',
	tags: ['java', 'collections', 'treeset', 'sorting'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a **SortedUniqueList** class using TreeSet to maintain sorted unique elements.

**Requirements:**
1. Create \`SortedUniqueList<T extends Comparable<T>>\` with TreeSet storage
2. Implement \`add(T element)\` to add element maintaining sorted order
3. Implement \`first()\` and \`last()\` to get min and max elements
4. Implement \`range(T from, T to)\` to get elements in range

**Example:**
\`\`\`java
SortedUniqueList<Integer> list = new SortedUniqueList<>();
list.add(30);
list.add(10);
list.add(20);
System.out.println(list.first()); // 10
System.out.println(list.last());  // 30
System.out.println(list.range(10, 25)); // [10, 20]
\`\`\`

**Key Concepts:**
- TreeSet maintains elements in sorted order
- Elements must implement Comparable or use Comparator
- O(log n) for add, remove, contains operations`,
	initialCode: `import java.util.TreeSet;
import java.util.SortedSet;

public class SortedUniqueList<T extends Comparable<T>> {
    // TODO: Declare TreeSet field

    public SortedUniqueList() {
        // TODO: Initialize TreeSet
    }

    public void add(T element) {
        // TODO: Add element
    }

    public T first() {
        // TODO: Return smallest element
        return null;
    }

    public T last() {
        // TODO: Return largest element
        return null;
    }

    public SortedSet<T> range(T from, T to) {
        // TODO: Return elements in range [from, to)
        return null;
    }
}`,
	solutionCode: `import java.util.TreeSet;
import java.util.SortedSet;

public class SortedUniqueList<T extends Comparable<T>> {
    private TreeSet<T> elements;

    public SortedUniqueList() {
        this.elements = new TreeSet<>();  // Natural ordering
    }

    public void add(T element) {
        elements.add(element);            // O(log n) insertion
    }

    public T first() {
        return elements.first();          // Smallest element
    }

    public T last() {
        return elements.last();           // Largest element
    }

    public SortedSet<T> range(T from, T to) {
        return elements.subSet(from, to); // Range view [from, to)
    }
}`,
	hint1: `TreeSet automatically maintains sorted order using natural ordering (Comparable) or a provided Comparator.`,
	hint2: `Use TreeSet.first() and last() for min/max, and subSet(from, to) for range queries - to is exclusive.`,
	whyItMatters: `TreeSet provides sorted storage with efficient operations for ordered data.

**Why TreeSet:**
- **Auto-Sorting:** Elements always in sorted order without explicit sort calls
- **Range Operations:** Efficient subSet, headSet, tailSet for range queries
- **NavigableSet:** Methods like floor(), ceiling(), higher(), lower()

**Production Use Cases:**
\`\`\`java
// Leaderboard (top scores)
TreeSet<Score> leaderboard = new TreeSet<>();
Score topScore = leaderboard.first();

// Time-based events
TreeSet<Event> timeline = new TreeSet<>();
Set<Event> thisWeek = timeline.subSet(startOfWeek, endOfWeek);

// Available time slots
TreeSet<TimeSlot> slots = new TreeSet<>();
TimeSlot nextAvailable = slots.ceiling(now);
\`\`\`

**TreeSet vs HashSet:**
- HashSet: O(1) but unordered
- TreeSet: O(log n) but always sorted
- Use TreeSet when you need sorted iteration or range queries

**Production Pattern:**
\`\`\`java
// Leaderboard with automatic sorting
TreeSet<Score> topScores = new TreeSet<>(
    Comparator.comparingInt(Score::getPoints).reversed());
topScores.add(newScore);
Score highestScore = topScores.first();  // Always highest score

// Time slots with fast lookup
TreeSet<TimeSlot> availableSlots = new TreeSet<>();
TimeSlot nextFree = availableSlots.ceiling(currentTime);  // Next available
\`\`\`

**Practical Benefits:**
- Automatic sorting without explicit sort() calls
- Efficient nearest neighbor search with ceiling/floor
- Perfect for leaderboards and time-based data`,
	order: 1,
	testCode: `import java.util.TreeSet;
import java.util.SortedSet;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: SortedUniqueList can be created
class Test1 {
    @Test
    void testCreation() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        assertNotNull(list);
    }
}

// Test2: add() stores elements
class Test2 {
    @Test
    void testAdd() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(10);
        list.add(20);
        assertEquals(10, list.first());
    }
}

// Test3: first() returns smallest element
class Test3 {
    @Test
    void testFirst() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(30);
        list.add(10);
        list.add(20);
        assertEquals(10, list.first());
    }
}

// Test4: last() returns largest element
class Test4 {
    @Test
    void testLast() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(30);
        list.add(10);
        list.add(20);
        assertEquals(30, list.last());
    }
}

// Test5: range() returns elements in range
class Test5 {
    @Test
    void testRange() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(10);
        list.add(20);
        list.add(30);
        list.add(40);
        SortedSet<Integer> range = list.range(15, 35);
        assertEquals(2, range.size());
        assertTrue(range.contains(20));
        assertTrue(range.contains(30));
    }
}

// Test6: Duplicates are not added
class Test6 {
    @Test
    void testNoDuplicates() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(10);
        list.add(10);
        list.add(10);
        assertEquals(10, list.first());
        assertEquals(10, list.last());
    }
}

// Test7: Works with String type
class Test7 {
    @Test
    void testWithStrings() {
        SortedUniqueList<String> list = new SortedUniqueList<>();
        list.add("banana");
        list.add("apple");
        list.add("cherry");
        assertEquals("apple", list.first());
        assertEquals("cherry", list.last());
    }
}

// Test8: range() with exact boundaries
class Test8 {
    @Test
    void testRangeExact() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(10);
        list.add(20);
        list.add(30);
        SortedSet<Integer> range = list.range(10, 30);
        assertTrue(range.contains(10));
        assertTrue(range.contains(20));
        assertFalse(range.contains(30));
    }
}

// Test9: Empty range returns empty set
class Test9 {
    @Test
    void testEmptyRange() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(10);
        list.add(20);
        SortedSet<Integer> range = list.range(100, 200);
        assertTrue(range.isEmpty());
    }
}

// Test10: Elements maintain sorted order
class Test10 {
    @Test
    void testSortedOrder() {
        SortedUniqueList<Integer> list = new SortedUniqueList<>();
        list.add(5);
        list.add(1);
        list.add(3);
        list.add(4);
        list.add(2);
        assertEquals(1, list.first());
        assertEquals(5, list.last());
    }
}
`,
	translations: {
		ru: {
			title: 'TreeSet и Естественная Сортировка',
			description: `Реализуйте класс **SortedUniqueList** с использованием TreeSet для хранения отсортированных уникальных элементов.

**Требования:**
1. Создайте \`SortedUniqueList<T extends Comparable<T>>\` с хранилищем TreeSet
2. Реализуйте \`add(T element)\` для добавления с сохранением порядка
3. Реализуйте \`first()\` и \`last()\` для получения мин и макс элементов
4. Реализуйте \`range(T from, T to)\` для получения элементов в диапазоне

**Пример:**
\`\`\`java
SortedUniqueList<Integer> list = new SortedUniqueList<>();
list.add(30);
list.add(10);
list.add(20);
System.out.println(list.first()); // 10
System.out.println(list.last());  // 30
\`\`\`

**Ключевые концепции:**
- TreeSet поддерживает элементы в отсортированном порядке
- Элементы должны реализовывать Comparable или использовать Comparator
- O(log n) для add, remove, contains операций`,
			hint1: `TreeSet автоматически поддерживает отсортированный порядок используя естественный порядок (Comparable).`,
			hint2: `Используйте TreeSet.first() и last() для мин/макс, и subSet(from, to) для запросов диапазона.`,
			whyItMatters: `TreeSet обеспечивает отсортированное хранение с эффективными операциями.

**Почему TreeSet:**
- **Авто-сортировка:** Элементы всегда в отсортированном порядке
- **Операции диапазона:** Эффективные subSet, headSet, tailSet
- **NavigableSet:** Методы floor(), ceiling(), higher(), lower()

**Использование в production:**
- Лидерборды (топ результаты)
- События по времени
- Доступные временные слоты

**TreeSet vs HashSet:**
- HashSet: O(1) но без порядка
- TreeSet: O(log n) но всегда отсортирован

**Продакшен паттерн:**
\`\`\`java
// Лидерборд с автоматической сортировкой
TreeSet<Score> topScores = new TreeSet<>(
    Comparator.comparingInt(Score::getPoints).reversed());
topScores.add(newScore);
Score highestScore = topScores.first();  // Всегда наивысший балл

// Временные интервалы с быстрым поиском
TreeSet<TimeSlot> availableSlots = new TreeSet<>();
TimeSlot nextFree = availableSlots.ceiling(currentTime);  // Следующий доступный
\`\`\`

**Практические преимущества:**
- Автоматическая сортировка без явных вызовов sort()
- Эффективный поиск nearest neighbor с ceiling/floor
- Идеально для топ-рейтингов и временных данных`,
			solutionCode: `import java.util.TreeSet;
import java.util.SortedSet;

public class SortedUniqueList<T extends Comparable<T>> {
    private TreeSet<T> elements;

    public SortedUniqueList() {
        this.elements = new TreeSet<>();  // Естественный порядок
    }

    public void add(T element) {
        elements.add(element);            // O(log n) вставка
    }

    public T first() {
        return elements.first();          // Наименьший элемент
    }

    public T last() {
        return elements.last();           // Наибольший элемент
    }

    public SortedSet<T> range(T from, T to) {
        return elements.subSet(from, to); // Диапазон [from, to)
    }
}`
		},
		uz: {
			title: 'TreeSet va Tabiiy Tartiblash',
			description: `Tartiblangan unikal elementlarni saqlash uchun TreeSet ishlatib **SortedUniqueList** klassini amalga oshiring.

**Talablar:**
1. TreeSet saqlash bilan \`SortedUniqueList<T extends Comparable<T>>\` yarating
2. Tartibni saqlab qo'shish uchun \`add(T element)\` ni amalga oshiring
3. Min va max elementlarni olish uchun \`first()\` va \`last()\` ni amalga oshiring
4. Diapazon elementlarini olish uchun \`range(T from, T to)\` ni amalga oshiring

**Misol:**
\`\`\`java
SortedUniqueList<Integer> list = new SortedUniqueList<>();
list.add(30);
list.add(10);
list.add(20);
System.out.println(list.first()); // 10
System.out.println(list.last());  // 30
\`\`\`

**Asosiy tushunchalar:**
- TreeSet elementlarni tartiblangan holatda saqlaydi
- Elementlar Comparable ni amalga oshirishi yoki Comparator ishlatishi kerak
- add, remove, contains uchun O(log n)`,
			hint1: `TreeSet tabiiy tartib (Comparable) yoki berilgan Comparator yordamida avtomatik tartiblangan holatni saqlaydi.`,
			hint2: `Min/max uchun TreeSet.first() va last(), diapazon so'rovlari uchun subSet(from, to) dan foydalaning.`,
			whyItMatters: `TreeSet tartiblangan ma'lumotlar uchun samarali operatsiyalar bilan tartiblangan saqlashni ta'minlaydi.

**Nima uchun TreeSet:**
- **Avto-tartiblash:** Elementlar doimo tartiblangan holatda
- **Diapazon operatsiyalari:** Samarali subSet, headSet, tailSet
- **NavigableSet:** floor(), ceiling(), higher(), lower() metodlari

**Production'da foydalanish:**
- Liderlik jadvallari
- Vaqt bo'yicha hodisalar
- Mavjud vaqt slotlari

**TreeSet vs HashSet:**
- HashSet: O(1) lekin tartibsiz
- TreeSet: O(log n) lekin doimo tartiblangan

**Ishlab chiqarish patterni:**
\`\`\`java
// Avtomatik saralash bilan liderlik jadvali
TreeSet<Score> topScores = new TreeSet<>(
    Comparator.comparingInt(Score::getPoints).reversed());
topScores.add(newScore);
Score highestScore = topScores.first();  // Doimo eng yuqori ball

// Tez qidiruv bilan vaqt intervallari
TreeSet<TimeSlot> availableSlots = new TreeSet<>();
TimeSlot nextFree = availableSlots.ceiling(currentTime);  // Keyingi mavjud
\`\`\`

**Amaliy foydalari:**
- sort() ni aniq chaqirmasdan avtomatik saralash
- ceiling/floor bilan samarali nearest neighbor qidiruv
- Top reytinglar va vaqt ma'lumotlari uchun ideal`,
			solutionCode: `import java.util.TreeSet;
import java.util.SortedSet;

public class SortedUniqueList<T extends Comparable<T>> {
    private TreeSet<T> elements;

    public SortedUniqueList() {
        this.elements = new TreeSet<>();  // Tabiiy tartib
    }

    public void add(T element) {
        elements.add(element);            // O(log n) qo'shish
    }

    public T first() {
        return elements.first();          // Eng kichik element
    }

    public T last() {
        return elements.last();           // Eng katta element
    }

    public SortedSet<T> range(T from, T to) {
        return elements.subSet(from, to); // Diapazon [from, to)
    }
}`
		}
	}
};

export default task;
