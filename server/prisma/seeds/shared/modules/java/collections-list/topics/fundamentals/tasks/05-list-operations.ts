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
        System.out.println("");
        System.out.println("After replaceAll (squaring): " + squareList);

        // Create another list and use retainAll() to keep common elements
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("");
        System.out.println("List1: " + list1);
        System.out.println("List2: " + list2);
        list1.retainAll(list2); // Keep only elements present in list2
        System.out.println("After retainAll (common elements): " + list1);

        // Use removeIf() to remove all numbers greater than 50
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("");
        System.out.println("Before removeIf: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("After removeIf (n > 50): " + filterList);

        // Create an immutable copy using List.copyOf()
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("");
        System.out.println("Immutable copy: " + immutableList);
        try {
            immutableList.add(4); // This will throw exception
        } catch (UnsupportedOperationException e) {
            System.out.println("Cannot modify immutable list!");
        }

        // Use addAll() to merge two lists
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("");
        System.out.println("ListA: " + listA);
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
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show List operations demo
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show List operations demo",
            output.contains("List") || output.contains("subList") ||
            output.contains("Operations") || output.contains("indexOf"));
    }
}

// Test2: Output should show subList operation
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention subList",
            output.contains("sublist") || output.contains("view") ||
            output.contains("подсписок") || output.contains("qism ro'yxat"));
    }
}

// Test3: Output should show subList affects original
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention original list change",
            output.contains("original") || output.contains("change") ||
            output.contains("оригинал") || output.contains("o'zgaradi"));
    }
}

// Test4: Output should show replaceAll squaring
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention replaceAll or squaring",
            output.contains("replaceall") || output.contains("square") ||
            output.contains("квадрат") || output.contains("kvadrat"));
    }
}

// Test5: Output should show retainAll common elements
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention retainAll or common",
            output.contains("retainall") || output.contains("retain") || output.contains("common") ||
            output.contains("общие") || output.contains("umumiy"));
    }
}

// Test6: Output should show removeIf operation
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention removeIf",
            output.contains("removeif") || output.contains("> 50") ||
            output.contains("больше 50") || output.contains("50 dan katta"));
    }
}

// Test7: Output should show immutable copy
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention immutable or copyOf",
            output.contains("immutable") || output.contains("copyof") ||
            output.contains("неизменяем") || output.contains("o'zgarmas"));
    }
}

// Test8: Output should show addAll merge
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention addAll or merge",
            output.contains("addall") || output.contains("merge") ||
            output.contains("объедин") || output.contains("birlashtirildi"));
    }
}

// Test9: Output should show numbers 1-10
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain numbers", output.contains("1") && output.contains("10"));
    }
}

// Test10: Output should show squared values
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        // After squaring: 16 = 4^2, 25 = 5^2, 36 = 6^2, etc.
        assertTrue("Should show squared values",
            output.contains("16") || output.contains("25") || output.contains("36"));
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
        System.out.println("");
        System.out.println("После replaceAll (возведение в квадрат): " + squareList);

        // Создаем другой список и используем retainAll() для сохранения общих элементов
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("");
        System.out.println("Список1: " + list1);
        System.out.println("Список2: " + list2);
        list1.retainAll(list2); // Оставляем только элементы, присутствующие в list2
        System.out.println("После retainAll (общие элементы): " + list1);

        // Используем removeIf() для удаления всех чисел больше 50
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("");
        System.out.println("До removeIf: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("После removeIf (n > 50): " + filterList);

        // Создаем неизменяемую копию используя List.copyOf()
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("");
        System.out.println("Неизменяемая копия: " + immutableList);
        try {
            immutableList.add(4); // Это вызовет исключение
        } catch (UnsupportedOperationException e) {
            System.out.println("Нельзя изменять неизменяемый список!");
        }

        // Используем addAll() для объединения двух списков
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("");
        System.out.println("СписокA: " + listA);
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
        System.out.println("");
        System.out.println("replaceAll dan keyin (kvadratga): " + squareList);

        // Boshqa ro'yxat yaratamiz va umumiy elementlarni saqlash uchun retainAll() dan foydalanamiz
        List<Integer> list1 = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6));
        List<Integer> list2 = List.of(4, 5, 6, 7, 8);
        System.out.println("");
        System.out.println("Ro'yxat1: " + list1);
        System.out.println("Ro'yxat2: " + list2);
        list1.retainAll(list2); // Faqat list2 da mavjud elementlarni saqlaymiz
        System.out.println("retainAll dan keyin (umumiy elementlar): " + list1);

        // 50 dan katta sonlarni o'chirish uchun removeIf() dan foydalanamiz
        List<Integer> filterList = new ArrayList<>(List.of(10, 30, 55, 70, 25, 100, 5));
        System.out.println("");
        System.out.println("removeIf dan oldin: " + filterList);
        filterList.removeIf(n -> n > 50);
        System.out.println("removeIf dan keyin (n > 50): " + filterList);

        // List.copyOf() yordamida o'zgarmas nusxa yaratamiz
        List<Integer> mutableList = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> immutableList = List.copyOf(mutableList);
        System.out.println("");
        System.out.println("O'zgarmas nusxa: " + immutableList);
        try {
            immutableList.add(4); // Bu xato chiqaradi
        } catch (UnsupportedOperationException e) {
            System.out.println("O'zgarmas ro'yxatni o'zgartirib bo'lmaydi!");
        }

        // Ikki ro'yxatni birlashtirish uchun addAll() dan foydalanamiz
        List<Integer> listA = new ArrayList<>(List.of(1, 2, 3));
        List<Integer> listB = List.of(4, 5, 6);
        System.out.println("");
        System.out.println("Ro'yxatA: " + listA);
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
