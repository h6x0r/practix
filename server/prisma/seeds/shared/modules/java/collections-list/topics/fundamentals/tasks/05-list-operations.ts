import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-list-operations',
    title: 'Advanced List Operations',
    difficulty: 'medium',
    tags: ['java', 'collections', 'list', 'operations'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master advanced list operations for data manipulation.

**Requirements:**
1. Create a list of numbers from 1 to 10
2. Use subList() to create a view of elements from index 3 to 7
3. Modify the subList and observe changes in the original list
4. Use replaceAll() to square all numbers
5. Create another list and use retainAll() to keep only common elements
6. Use removeIf() to remove all numbers greater than 50
7. Create an immutable copy using List.copyOf()
8. Use addAll() to merge two lists

SubList creates a view, not a copy - changes affect the original list.`,
    initialCode: `import java.util.ArrayList;
import java.util.List;

public class ListOperations {
    public static void main(String[] args) {
        // Create a list of numbers from 1 to 10

        // Use subList() to create a view

        // Modify the subList

        // Use replaceAll() to square all numbers

        // Use retainAll() to keep common elements

        // Use removeIf() to remove numbers > 50

        // Create immutable copy with List.copyOf()

        // Use addAll() to merge lists
    }
}`,
    solutionCode: `import java.util.ArrayList;
import java.util.List;

public class ListOperations {
    public static void main(String[] args) {
        // Create a list of numbers from 1 to 10
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            numbers.add(i);
        }
        System.out.println("Original list: " + numbers);

        // Use subList() to create a view of elements from index 3 to 7
        List<Integer> subList = numbers.subList(3, 7);
        System.out.println("SubList (index 3-6): " + subList);

        // Modify the subList and observe changes in original list
        subList.set(0, 100); // Changes original list!
        System.out.println("After modifying subList:");
        System.out.println("  SubList: " + subList);
        System.out.println("  Original: " + numbers);

        // Use replaceAll() to square all numbers
        List<Integer> squareList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        squareList.replaceAll(n -> n * n);
        System.out.println("\nAfter replaceAll (squaring): " + squareList);

        // Create another list and use retainAll() to keep common elements
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("\nList1: " + list1);
        System.out.println("List2: " + list2);
        list1.retainAll(list2); // Keep only elements present in list2
        System.out.println("After retainAll (common elements): " + list1);

        // Use removeIf() to remove all numbers greater than 50
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("\nBefore removeIf: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("After removeIf (n > 50): " + filterList);

        // Create an immutable copy using List.copyOf()
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("\nImmutable copy: " + immutableList);
        try {
            immutableList.add(4); // This will throw exception
        } catch (UnsupportedOperationException e) {
            System.out.println("Cannot modify immutable list!");
        }

        // Use addAll() to merge two lists
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("\nListA: " + listA);
        System.out.println("ListB: " + listB);
        listA.addAll(listB); // Merge listB into listA
        System.out.println("After addAll: " + listA);

        // Can also add at specific index
        List<Integer> listC = new ArrayList<>(List.of(1, 5, 6));
        listC.addAll(1, List.of(2, 3, 4)); // Insert at index 1
        System.out.println("After addAll at index 1: " + listC);
    }
}`,
    hint1: `subList() returns a view that's backed by the original list. Changes to the subList affect the original.`,
    hint2: `replaceAll() takes a UnaryOperator, removeIf() takes a Predicate. List.copyOf() creates an immutable copy.`,
    whyItMatters: `These advanced operations are essential for efficient data manipulation. Understanding subList views, bulk operations, and immutability helps write cleaner, more efficient code.

**Production Pattern:**
\`\`\`java
// Batch processing with subList
List<Transaction> allTransactions = loadTransactions();
int batchSize = 100;
for (int i = 0; i < allTransactions.size(); i += batchSize) {
    int end = Math.min(i + batchSize, allTransactions.size());
    List<Transaction> batch = allTransactions.subList(i, end);
    processBatch(batch);  // Process batch
}

// Removing invalid records
List<Record> records = database.getRecords();
records.removeIf(r -> !r.isValid() || r.isExpired());

// Immutable copy for thread safety
List<String> mutableConfig = loadConfig();
List<String> immutableConfig = List.copyOf(mutableConfig);
shareWithOtherThreads(immutableConfig);  // Safe sharing
\`\`\`

**Practical Benefits:**
- Efficient batch processing of large data volumes
- Concise filtering with removeIf instead of loops
- Thread safety through immutable collections`,
    order: 4,
    testCode: `import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: subList creates a view of the original list
class Test1 {
    @Test
    void testSubListView() {
        List<Integer> numbers = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        List<Integer> sub = numbers.subList(1, 4);
        assertEquals(3, sub.size());
        assertEquals(List.of(2, 3, 4), sub);
    }
}

// Test2: Modifying subList affects original list
class Test2 {
    @Test
    void testSubListModifiesOriginal() {
        List<Integer> numbers = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        List<Integer> sub = numbers.subList(1, 4);
        sub.set(0, 100);
        assertEquals(100, numbers.get(1));
    }
}

// Test3: replaceAll applies function to all elements
class Test3 {
    @Test
    void testReplaceAll() {
        List<Integer> numbers = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        numbers.replaceAll(n -> n * n);
        assertEquals(List.of(1, 4, 9, 16, 25), numbers);
    }
}

// Test4: retainAll keeps only common elements
class Test4 {
    @Test
    void testRetainAll() {
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        List<Integer> list2 = List.of(3, 4, 5, 6, 7);
        list1.retainAll(list2);
        assertEquals(List.of(3, 4, 5), list1);
    }
}

// Test5: removeIf removes elements matching predicate
class Test5 {
    @Test
    void testRemoveIf() {
        List<Integer> numbers = new ArrayList<>(List.of(10, 20, 55, 70, 25));
        numbers.removeIf(n -> n > 50);
        assertEquals(List.of(10, 20, 25), numbers);
    }
}

// Test6: List.copyOf creates immutable copy
class Test6 {
    @Test
    void testCopyOfImmutable() {
        List<Integer> mutable = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutable = List.copyOf(mutable);
        assertThrows(UnsupportedOperationException.class, () -> immutable.add(4));
    }
}

// Test7: addAll merges two lists
class Test7 {
    @Test
    void testAddAll() {
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> list2 = List.of(4, 5, 6);
        list1.addAll(list2);
        assertEquals(List.of(1, 2, 3, 4, 5, 6), list1);
    }
}

// Test8: addAll at index inserts at position
class Test8 {
    @Test
    void testAddAllAtIndex() {
        List<Integer> list = new ArrayList<>(List.of(1, 5, 6));
        list.addAll(1, List.of(2, 3, 4));
        assertEquals(List.of(1, 2, 3, 4, 5, 6), list);
    }
}

// Test9: clear() removes all elements
class Test9 {
    @Test
    void testClear() {
        List<Integer> numbers = new ArrayList<>(List.of(1, 2, 3));
        numbers.clear();
        assertTrue(numbers.isEmpty());
    }
}

// Test10: subList clear removes range from original
class Test10 {
    @Test
    void testSubListClear() {
        List<Integer> numbers = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        numbers.subList(1, 4).clear();
        assertEquals(List.of(1, 5), numbers);
    }
}
`,
    translations: {
        ru: {
            title: 'Продвинутые операции со списками',
            solutionCode: `import java.util.ArrayList;
import java.util.List;

public class ListOperations {
    public static void main(String[] args) {
        // Создаем список чисел от 1 до 10
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            numbers.add(i);
        }
        System.out.println("Исходный список: " + numbers);

        // Используем subList() для создания представления элементов с индекса 3 до 7
        List<Integer> subList = numbers.subList(3, 7);
        System.out.println("Подсписок (индексы 3-6): " + subList);

        // Изменяем подсписок и наблюдаем изменения в исходном списке
        subList.set(0, 100); // Изменяет исходный список!
        System.out.println("После изменения подсписка:");
        System.out.println("  Подсписок: " + subList);
        System.out.println("  Исходный: " + numbers);

        // Используем replaceAll() для возведения всех чисел в квадрат
        List<Integer> squareList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        squareList.replaceAll(n -> n * n);
        System.out.println("\nПосле replaceAll (возведение в квадрат): " + squareList);

        // Создаем другой список и используем retainAll() для сохранения общих элементов
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("\nСписок1: " + list1);
        System.out.println("Список2: " + list2);
        list1.retainAll(list2); // Оставляем только элементы, присутствующие в list2
        System.out.println("После retainAll (общие элементы): " + list1);

        // Используем removeIf() для удаления всех чисел больше 50
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("\nДо removeIf: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("После removeIf (n > 50): " + filterList);

        // Создаем неизменяемую копию используя List.copyOf()
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("\nНеизменяемая копия: " + immutableList);
        try {
            immutableList.add(4); // Это вызовет исключение
        } catch (UnsupportedOperationException e) {
            System.out.println("Нельзя изменять неизменяемый список!");
        }

        // Используем addAll() для объединения двух списков
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("\nСписокA: " + listA);
        System.out.println("СписокB: " + listB);
        listA.addAll(listB); // Объединяем listB в listA
        System.out.println("После addAll: " + listA);

        // Можно также добавить по конкретному индексу
        List<Integer> listC = new ArrayList<>(List.of(1, 5, 6));
        listC.addAll(1, List.of(2, 3, 4)); // Вставляем по индексу 1
        System.out.println("После addAll по индексу 1: " + listC);
    }
}`,
            description: `Освойте продвинутые операции со списками для манипуляции данными.

**Требования:**
1. Создайте список чисел от 1 до 10
2. Используйте subList() для создания представления элементов с индекса 3 до 7
3. Измените подсписок и наблюдайте изменения в исходном списке
4. Используйте replaceAll() для возведения всех чисел в квадрат
5. Создайте другой список и используйте retainAll() для сохранения только общих элементов
6. Используйте removeIf() для удаления всех чисел больше 50
7. Создайте неизменяемую копию используя List.copyOf()
8. Используйте addAll() для объединения двух списков

SubList создает представление, а не копию - изменения влияют на исходный список.`,
            hint1: `subList() возвращает представление, связанное с исходным списком. Изменения подсписка влияют на исходный.`,
            hint2: `replaceAll() принимает UnaryOperator, removeIf() принимает Predicate. List.copyOf() создает неизменяемую копию.`,
            whyItMatters: `Эти продвинутые операции необходимы для эффективной манипуляции данными. Понимание представлений subList, массовых операций и неизменяемости помогает писать более чистый и эффективный код.

**Продакшен паттерн:**
\`\`\`java
// Пакетная обработка с subList
List<Transaction> allTransactions = loadTransactions();
int batchSize = 100;
for (int i = 0; i < allTransactions.size(); i += batchSize) {
    int end = Math.min(i + batchSize, allTransactions.size());
    List<Transaction> batch = allTransactions.subList(i, end);
    processBatch(batch);  // Обработка батча
}

// Удаление недействительных записей
List<Record> records = database.getRecords();
records.removeIf(r -> !r.isValid() || r.isExpired());

// Неизменяемая копия для потокобезопасности
List<String> mutableConfig = loadConfig();
List<String> immutableConfig = List.copyOf(mutableConfig);
shareWithOtherThreads(immutableConfig);  // Безопасная передача
\`\`\`

**Практические преимущества:**
- Эффективная пакетная обработка больших объемов данных
- Лаконичная фильтрация с removeIf вместо циклов
- Потокобезопасность через неизменяемые коллекции`
        },
        uz: {
            title: 'Ilg\'or Ro\'yxat Operatsiyalari',
            solutionCode: `import java.util.ArrayList;
import java.util.List;

public class ListOperations {
    public static void main(String[] args) {
        // 1 dan 10 gacha sonlar ro'yxatini yaratamiz
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            numbers.add(i);
        }
        System.out.println("Dastlabki ro'yxat: " + numbers);

        // 3 dan 7 gacha indekslar uchun subList() yordamida ko'rinish yaratamiz
        List<Integer> subList = numbers.subList(3, 7);
        System.out.println("Qism ro'yxat (3-6 indekslar): " + subList);

        // Qism ro'yxatni o'zgartiramiz va asl ro'yxatdagi o'zgarishlarni kuzatamiz
        subList.set(0, 100); // Asl ro'yxatni o'zgartiradi!
        System.out.println("Qism ro'yxatni o'zgartirgandan keyin:");
        System.out.println("  Qism ro'yxat: " + subList);
        System.out.println("  Asl ro'yxat: " + numbers);

        // Barcha sonlarni kvadratga ko'tarish uchun replaceAll() dan foydalanamiz
        List<Integer> squareList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        squareList.replaceAll(n -> n * n);
        System.out.println("\nreplaceAll dan keyin (kvadratga): " + squareList);

        // Boshqa ro'yxat yaratamiz va umumiy elementlarni saqlash uchun retainAll() dan foydalanamiz
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("\nRo'yxat1: " + list1);
        System.out.println("Ro'yxat2: " + list2);
        list1.retainAll(list2); // Faqat list2 da mavjud elementlarni saqlaymiz
        System.out.println("retainAll dan keyin (umumiy elementlar): " + list1);

        // 50 dan katta sonlarni o'chirish uchun removeIf() dan foydalanamiz
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("\nremoveIf dan oldin: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("removeIf dan keyin (n > 50): " + filterList);

        // List.copyOf() yordamida o'zgarmas nusxa yaratamiz
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("\nO'zgarmas nusxa: " + immutableList);
        try {
            immutableList.add(4); // Bu xato chiqaradi
        } catch (UnsupportedOperationException e) {
            System.out.println("O'zgarmas ro'yxatni o'zgartirib bo'lmaydi!");
        }

        // Ikki ro'yxatni birlashtirish uchun addAll() dan foydalanamiz
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("\nRo'yxatA: " + listA);
        System.out.println("Ro'yxatB: " + listB);
        listA.addAll(listB); // listB ni listA ga birlashtirish
        System.out.println("addAll dan keyin: " + listA);

        // Muayyan indeksga ham qo'shish mumkin
        List<Integer> listC = new ArrayList<>(List.of(1, 5, 6));
        listC.addAll(1, List.of(2, 3, 4)); // 1-indeksga kiritish
        System.out.println("1-indeksga addAll dan keyin: " + listC);
    }
}`,
            description: `Ma'lumotlarni boshqarish uchun ilg'or ro'yxat operatsiyalarini o'rganing.

**Talablar:**
1. 1 dan 10 gacha sonlar ro'yxatini yarating
2. 3 dan 7 gacha indekslar uchun subList() yordamida ko'rinish yarating
3. Qism ro'yxatni o'zgartiring va asl ro'yxatdagi o'zgarishlarni kuzating
4. Barcha sonlarni kvadratga ko'tarish uchun replaceAll() dan foydalaning
5. Boshqa ro'yxat yarating va faqat umumiy elementlarni saqlash uchun retainAll() dan foydalaning
6. 50 dan katta barcha sonlarni o'chirish uchun removeIf() dan foydalaning
7. List.copyOf() yordamida o'zgarmas nusxa yarating
8. Ikki ro'yxatni birlashtirish uchun addAll() dan foydalaning

SubList nusxa emas, ko'rinish yaratadi - o'zgarishlar asl ro'yxatga ta'sir qiladi.`,
            hint1: `subList() asl ro'yxat bilan bog'langan ko'rinishni qaytaradi. Qism ro'yxatdagi o'zgarishlar asl ro'yxatga ta'sir qiladi.`,
            hint2: `replaceAll() UnaryOperator qabul qiladi, removeIf() Predicate qabul qiladi. List.copyOf() o'zgarmas nusxa yaratadi.`,
            whyItMatters: `Bu ilg'or operatsiyalar samarali ma'lumotlarni boshqarish uchun zarur. subList ko'rinishlari, ommaviy operatsiyalar va o'zgarmaslikni tushunish toza va samarali kod yozishga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// subList bilan paketli qayta ishlash
List<Transaction> allTransactions = loadTransactions();
int batchSize = 100;
for (int i = 0; i < allTransactions.size(); i += batchSize) {
    int end = Math.min(i + batchSize, allTransactions.size());
    List<Transaction> batch = allTransactions.subList(i, end);
    processBatch(batch);  // Paketni qayta ishlash
}

// Yaroqsiz yozuvlarni o'chirish
List<Record> records = database.getRecords();
records.removeIf(r -> !r.isValid() || r.isExpired());

// Thread xavfsizligi uchun o'zgarmas nusxa
List<String> mutableConfig = loadConfig();
List<String> immutableConfig = List.copyOf(mutableConfig);
shareWithOtherThreads(immutableConfig);  // Xavfsiz uzatish
\`\`\`

**Amaliy foydalari:**
- Katta hajmdagi ma'lumotlarni samarali paketli qayta ishlash
- Tsikllar o'rniga removeIf bilan qisqa filtrlash
- O'zgarmas kolleksiyalar orqali thread xavfsizligi`
        }
    }
};

export default task;
