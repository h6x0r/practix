import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-atomic-arrays',
    title: 'Atomic Arrays',
    difficulty: 'medium',
    tags: ['java', 'concurrency', 'atomic', 'arrays'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn AtomicIntegerArray and AtomicReferenceArray for thread-safe array operations.

**Requirements:**
1. Create an AtomicIntegerArray of size 5, initialized with values [10, 20, 30, 40, 50]
2. Get and print the value at index 2
3. Use set(1, 25) to change index 1 to 25
4. Use getAndAdd(3, 5) to add 5 to index 3 and print the old value
5. Use compareAndSet(2, 30, 35) to change index 2 from 30 to 35
6. Create an AtomicReferenceArray<String> with ["A", "B", "C"]
7. Use getAndSet(1, "X") to replace "B" with "X"
8. Print both arrays

Atomic arrays provide thread-safe operations on array elements without locking the entire array.`,
    initialCode: `import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class AtomicArraysDemo {
    public static void main(String[] args) {
        // Create AtomicIntegerArray with [10, 20, 30, 40, 50]

        // Get value at index 2

        // Set index 1 to 25

        // Add 5 to index 3 and get old value

        // compareAndSet at index 2

        // Create AtomicReferenceArray with ["A", "B", "C"]

        // Replace "B" with "X" at index 1

        // Print both arrays
    }
}`,
    solutionCode: `import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class AtomicArraysDemo {
    public static void main(String[] args) {
        // Create AtomicIntegerArray with [10, 20, 30, 40, 50]
        int[] initialValues = {10, 20, 30, 40, 50};
        AtomicIntegerArray intArray = new AtomicIntegerArray(initialValues);
        System.out.println("Initial integer array:");
        printIntArray(intArray);

        // Get value at index 2
        int value = intArray.get(2);
        System.out.println("\\nValue at index 2: " + value);

        // Set index 1 to 25
        intArray.set(1, 25);
        System.out.println("After setting index 1 to 25:");
        printIntArray(intArray);

        // Add 5 to index 3 and get old value
        int oldValue = intArray.getAndAdd(3, 5);
        System.out.println("\\ngetAndAdd(3, 5) returned: " + oldValue);
        System.out.println("After adding 5 to index 3:");
        printIntArray(intArray);

        // compareAndSet at index 2 (30 -> 35)
        boolean success = intArray.compareAndSet(2, 30, 35);
        System.out.println("\\ncompareAndSet(2, 30, 35) success: " + success);
        System.out.println("After compareAndSet:");
        printIntArray(intArray);

        // Create AtomicReferenceArray with ["A", "B", "C"]
        String[] strings = {"A", "B", "C"};
        AtomicReferenceArray<String> refArray = new AtomicReferenceArray<>(strings);
        System.out.println("\\nInitial reference array:");
        printRefArray(refArray);

        // Replace "B" with "X" at index 1
        String oldString = refArray.getAndSet(1, "X");
        System.out.println("\\ngetAndSet(1, \\"X\\") returned: " + oldString);
        System.out.println("After replacing at index 1:");
        printRefArray(refArray);
    }

    private static void printIntArray(AtomicIntegerArray array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }

    private static void printRefArray(AtomicReferenceArray<String> array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }
}`,
    hint1: `AtomicIntegerArray can be initialized with an int[] array. All operations work on individual array elements atomically.`,
    hint2: `getAndAdd() adds a delta to the element and returns the previous value. compareAndSet() works on individual array indices.`,
    whyItMatters: `Atomic arrays enable fine-grained thread-safe operations on array elements, allowing concurrent access without locking the entire array, which improves performance in multi-threaded scenarios.

**Production Pattern:**
\`\`\`java
// Thread-safe counter by categories
public class CategoryMetrics {
    private final AtomicIntegerArray metrics;
    private final String[] categoryNames;

    public CategoryMetrics(String[] categories) {
        this.categoryNames = categories;
        this.metrics = new AtomicIntegerArray(categories.length);
    }

    public void incrementCategory(int categoryIndex) {
        if (categoryIndex >= 0 && categoryIndex < metrics.length()) {
            metrics.incrementAndGet(categoryIndex);
        }
    }

    public int getCategoryCount(int categoryIndex) {
        return metrics.get(categoryIndex);
    }

    public Map<String, Integer> getAllMetrics() {
        Map<String, Integer> result = new HashMap<>();
        for (int i = 0; i < categoryNames.length; i++) {
            result.put(categoryNames[i], metrics.get(i));
        }
        return result;
    }
}
\`\`\`

**Practical Benefits:**
- Parallel updates of different elements without conflicts
- Excellent scalability for multi-threaded metrics
- No need to synchronize the entire array`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

// Test1: Test AtomicIntegerArray initialization
class Test1 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(5);
        assertEquals(5, array.length());
        assertEquals(0, array.get(0));
    }
}

// Test2: Test AtomicIntegerArray set and get
class Test2 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(3);
        array.set(1, 42);
        assertEquals(42, array.get(1));
    }
}

// Test3: Test AtomicIntegerArray incrementAndGet
class Test3 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(new int[]{10, 20, 30});
        assertEquals(11, array.incrementAndGet(0));
        assertEquals(11, array.get(0));
    }
}

// Test4: Test AtomicIntegerArray getAndIncrement
class Test4 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(new int[]{10, 20, 30});
        assertEquals(10, array.getAndIncrement(0));
        assertEquals(11, array.get(0));
    }
}

// Test5: Test AtomicIntegerArray addAndGet
class Test5 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(new int[]{10, 20, 30});
        assertEquals(15, array.addAndGet(0, 5));
        assertEquals(15, array.get(0));
    }
}

// Test6: Test AtomicIntegerArray compareAndSet
class Test6 {
    @Test
    public void test() {
        AtomicIntegerArray array = new AtomicIntegerArray(new int[]{10, 20, 30});
        assertTrue(array.compareAndSet(1, 20, 25));
        assertEquals(25, array.get(1));
    }
}

// Test7: Test AtomicReferenceArray initialization
class Test7 {
    @Test
    public void test() {
        AtomicReferenceArray<String> array = new AtomicReferenceArray<>(3);
        assertEquals(3, array.length());
        assertNull(array.get(0));
    }
}

// Test8: Test AtomicReferenceArray set and get
class Test8 {
    @Test
    public void test() {
        AtomicReferenceArray<String> array = new AtomicReferenceArray<>(3);
        array.set(0, "Hello");
        assertEquals("Hello", array.get(0));
    }
}

// Test9: Test AtomicReferenceArray compareAndSet
class Test9 {
    @Test
    public void test() {
        AtomicReferenceArray<String> array = new AtomicReferenceArray<>(new String[]{"A", "B", "C"});
        assertTrue(array.compareAndSet(1, "B", "X"));
        assertEquals("X", array.get(1));
    }
}

// Test10: Test AtomicReferenceArray getAndSet
class Test10 {
    @Test
    public void test() {
        AtomicReferenceArray<String> array = new AtomicReferenceArray<>(new String[]{"A", "B", "C"});
        assertEquals("B", array.getAndSet(1, "Y"));
        assertEquals("Y", array.get(1));
    }
}
`,
    translations: {
        ru: {
            title: 'Атомарные массивы',
            solutionCode: `import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class AtomicArraysDemo {
    public static void main(String[] args) {
        // Создаем AtomicIntegerArray с [10, 20, 30, 40, 50]
        int[] initialValues = {10, 20, 30, 40, 50};
        AtomicIntegerArray intArray = new AtomicIntegerArray(initialValues);
        System.out.println("Начальный массив целых чисел:");
        printIntArray(intArray);

        // Получаем значение по индексу 2
        int value = intArray.get(2);
        System.out.println("\\nЗначение по индексу 2: " + value);

        // Устанавливаем индекс 1 в 25
        intArray.set(1, 25);
        System.out.println("После установки индекса 1 в 25:");
        printIntArray(intArray);

        // Добавляем 5 к индексу 3 и получаем старое значение
        int oldValue = intArray.getAndAdd(3, 5);
        System.out.println("\\ngetAndAdd(3, 5) вернул: " + oldValue);
        System.out.println("После добавления 5 к индексу 3:");
        printIntArray(intArray);

        // compareAndSet по индексу 2 (30 -> 35)
        boolean success = intArray.compareAndSet(2, 30, 35);
        System.out.println("\\ncompareAndSet(2, 30, 35) успех: " + success);
        System.out.println("После compareAndSet:");
        printIntArray(intArray);

        // Создаем AtomicReferenceArray с ["A", "B", "C"]
        String[] strings = {"A", "B", "C"};
        AtomicReferenceArray<String> refArray = new AtomicReferenceArray<>(strings);
        System.out.println("\\nНачальный массив ссылок:");
        printRefArray(refArray);

        // Заменяем "B" на "X" по индексу 1
        String oldString = refArray.getAndSet(1, "X");
        System.out.println("\\ngetAndSet(1, \\"X\\") вернул: " + oldString);
        System.out.println("После замены по индексу 1:");
        printRefArray(refArray);
    }

    private static void printIntArray(AtomicIntegerArray array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }

    private static void printRefArray(AtomicReferenceArray<String> array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }
}`,
            description: `Изучите AtomicIntegerArray и AtomicReferenceArray для потокобезопасных операций с массивами.

**Требования:**
1. Создайте AtomicIntegerArray размером 5, инициализированный значениями [10, 20, 30, 40, 50]
2. Получите и выведите значение по индексу 2
3. Используйте set(1, 25) для изменения индекса 1 на 25
4. Используйте getAndAdd(3, 5) для добавления 5 к индексу 3 и выведите старое значение
5. Используйте compareAndSet(2, 30, 35) для изменения индекса 2 с 30 на 35
6. Создайте AtomicReferenceArray<String> с ["A", "B", "C"]
7. Используйте getAndSet(1, "X") для замены "B" на "X"
8. Выведите оба массива

Атомарные массивы предоставляют потокобезопасные операции над элементами массива без блокировки всего массива.`,
            hint1: `AtomicIntegerArray можно инициализировать массивом int[]. Все операции работают с отдельными элементами массива атомарно.`,
            hint2: `getAndAdd() добавляет дельту к элементу и возвращает предыдущее значение. compareAndSet() работает с отдельными индексами массива.`,
            whyItMatters: `Атомарные массивы обеспечивают детальные потокобезопасные операции над элементами массива, позволяя одновременный доступ без блокировки всего массива, что улучшает производительность в многопоточных сценариях.

**Продакшен паттерн:**
\`\`\`java
// Потокобезопасный счетчик по категориям
public class CategoryMetrics {
    private final AtomicIntegerArray metrics;
    private final String[] categoryNames;

    public CategoryMetrics(String[] categories) {
        this.categoryNames = categories;
        this.metrics = new AtomicIntegerArray(categories.length);
    }

    public void incrementCategory(int categoryIndex) {
        if (categoryIndex >= 0 && categoryIndex < metrics.length()) {
            metrics.incrementAndGet(categoryIndex);
        }
    }

    public int getCategoryCount(int categoryIndex) {
        return metrics.get(categoryIndex);
    }

    public Map<String, Integer> getAllMetrics() {
        Map<String, Integer> result = new HashMap<>();
        for (int i = 0; i < categoryNames.length; i++) {
            result.put(categoryNames[i], metrics.get(i));
        }
        return result;
    }
}
\`\`\`

**Практические преимущества:**
- Параллельное обновление разных элементов без конфликтов
- Отличная масштабируемость для многопоточных метрик
- Нет необходимости в синхронизации всего массива`
        },
        uz: {
            title: 'Atomik massivlar',
            solutionCode: `import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class AtomicArraysDemo {
    public static void main(String[] args) {
        // [10, 20, 30, 40, 50] bilan AtomicIntegerArray yaratamiz
        int[] initialValues = {10, 20, 30, 40, 50};
        AtomicIntegerArray intArray = new AtomicIntegerArray(initialValues);
        System.out.println("Boshlang'ich butun sonlar massivi:");
        printIntArray(intArray);

        // 2-indeksdagi qiymatni olamiz
        int value = intArray.get(2);
        System.out.println("\\n2-indeksdagi qiymat: " + value);

        // 1-indeksni 25 ga o'rnatamiz
        intArray.set(1, 25);
        System.out.println("1-indeksni 25 ga o'rnatgandan keyin:");
        printIntArray(intArray);

        // 3-indeksga 5 qo'shamiz va eski qiymatni olamiz
        int oldValue = intArray.getAndAdd(3, 5);
        System.out.println("\\ngetAndAdd(3, 5) qaytardi: " + oldValue);
        System.out.println("3-indeksga 5 qo'shgandan keyin:");
        printIntArray(intArray);

        // 2-indeksda compareAndSet (30 -> 35)
        boolean success = intArray.compareAndSet(2, 30, 35);
        System.out.println("\\ncompareAndSet(2, 30, 35) muvaffaqiyat: " + success);
        System.out.println("compareAndSet dan keyin:");
        printIntArray(intArray);

        // ["A", "B", "C"] bilan AtomicReferenceArray yaratamiz
        String[] strings = {"A", "B", "C"};
        AtomicReferenceArray<String> refArray = new AtomicReferenceArray<>(strings);
        System.out.println("\\nBoshlang'ich havola massivi:");
        printRefArray(refArray);

        // 1-indeksda "B" ni "X" ga almashtiramiz
        String oldString = refArray.getAndSet(1, "X");
        System.out.println("\\ngetAndSet(1, \\"X\\") qaytardi: " + oldString);
        System.out.println("1-indeksda almashtirgandan keyin:");
        printRefArray(refArray);
    }

    private static void printIntArray(AtomicIntegerArray array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }

    private static void printRefArray(AtomicReferenceArray<String> array) {
        for (int i = 0; i < array.length(); i++) {
            System.out.print(array.get(i) + " ");
        }
        System.out.println();
    }
}`,
            description: `Thread-xavfsiz massiv operatsiyalari uchun AtomicIntegerArray va AtomicReferenceArray ni o'rganing.

**Talablar:**
1. [10, 20, 30, 40, 50] qiymatlari bilan 5 o'lchamli AtomicIntegerArray yarating
2. 2-indeksdagi qiymatni oling va chiqaring
3. 1-indeksni 25 ga o'zgartirish uchun set(1, 25) dan foydalaning
4. 3-indeksga 5 qo'shish va eski qiymatni chiqarish uchun getAndAdd(3, 5) dan foydalaning
5. 2-indeksni 30 dan 35 ga o'zgartirish uchun compareAndSet(2, 30, 35) dan foydalaning
6. ["A", "B", "C"] bilan AtomicReferenceArray<String> yarating
7. "B" ni "X" ga almashtirish uchun getAndSet(1, "X") dan foydalaning
8. Ikkala massivni ham chiqaring

Atomik massivlar butun massivni bloklashsiz massiv elementlari ustida thread-xavfsiz operatsiyalarni taqdim etadi.`,
            hint1: `AtomicIntegerArray int[] massivi bilan ishga tushirilishi mumkin. Barcha operatsiyalar alohida massiv elementlari bilan atomik ravishda ishlaydi.`,
            hint2: `getAndAdd() elementga deltani qo'shadi va oldingi qiymatni qaytaradi. compareAndSet() alohida massiv indekslari bilan ishlaydi.`,
            whyItMatters: `Atomik massivlar massiv elementlari ustida nozik thread-xavfsiz operatsiyalarni ta'minlaydi, butun massivni bloklashsiz bir vaqtda kirishga imkon beradi, bu ko'p oqimli stsenariylarda ishlashni yaxshilaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Kategoriyalar bo'yicha thread-xavfsiz hisoblagich
public class CategoryMetrics {
    private final AtomicIntegerArray metrics;
    private final String[] categoryNames;

    public CategoryMetrics(String[] categories) {
        this.categoryNames = categories;
        this.metrics = new AtomicIntegerArray(categories.length);
    }

    public void incrementCategory(int categoryIndex) {
        if (categoryIndex >= 0 && categoryIndex < metrics.length()) {
            metrics.incrementAndGet(categoryIndex);
        }
    }

    public int getCategoryCount(int categoryIndex) {
        return metrics.get(categoryIndex);
    }

    public Map<String, Integer> getAllMetrics() {
        Map<String, Integer> result = new HashMap<>();
        for (int i = 0; i < categoryNames.length; i++) {
            result.put(categoryNames[i], metrics.get(i));
        }
        return result;
    }
}
\`\`\`

**Amaliy foydalari:**
- Konfliktlarsiz turli elementlarni parallel yangilash
- Ko'p oqimli metrikalar uchun ajoyib miqyoslanish
- Butun massivni sinxronlash kerak emas`
        }
    }
};

export default task;
