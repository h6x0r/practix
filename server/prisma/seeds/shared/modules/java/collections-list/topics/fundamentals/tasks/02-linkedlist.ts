import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-linkedlist-deque',
    title: 'LinkedList and Deque Operations',
    difficulty: 'easy',
    tags: ['java', 'collections', 'list', 'linkedlist', 'deque'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn LinkedList as both List and Deque.

**Requirements:**
1. Create a LinkedList to store integers
2. Use addFirst() to add elements 1, 2, 3 at the front
3. Use addLast() to add elements 7, 8, 9 at the end
4. Use add() to append element 10
5. Print the list
6. Remove the first and last elements using removeFirst() and removeLast()
7. Use getFirst() and getLast() to print the current first and last elements
8. Explain when to use LinkedList vs ArrayList

LinkedList implements both List and Deque interfaces, making it ideal for queue and stack operations.`,
    initialCode: `import java.util.LinkedList;

public class LinkedListDeque {
    public static void main(String[] args) {
        // Create a LinkedList to store integers

        // Use addFirst() to add 1, 2, 3 at the front

        // Use addLast() to add 7, 8, 9 at the end

        // Use add() to append 10

        // Print the list

        // Remove first and last elements

        // Print first and last elements

        // When to use LinkedList vs ArrayList?
    }
}`,
    solutionCode: `import java.util.LinkedList;

public class LinkedListDeque {
    public static void main(String[] args) {
        // Create a LinkedList to store integers
        LinkedList<Integer> numbers = new LinkedList<>();

        // Use addFirst() to add 1, 2, 3 at the front
        // Note: each addFirst adds to the front, so order will be reversed
        numbers.addFirst(1);
        numbers.addFirst(2);
        numbers.addFirst(3);
        System.out.println("After addFirst: " + numbers);

        // Use addLast() to add 7, 8, 9 at the end
        numbers.addLast(7);
        numbers.addLast(8);
        numbers.addLast(9);
        System.out.println("After addLast: " + numbers);

        // Use add() to append 10 (equivalent to addLast)
        numbers.add(10);
        System.out.println("After add(10): " + numbers);

        // Remove first and last elements
        Integer first = numbers.removeFirst();
        Integer last = numbers.removeLast();
        System.out.println("Removed first: " + first + ", last: " + last);
        System.out.println("After removal: " + numbers);

        // Print first and last elements
        System.out.println("Current first: " + numbers.getFirst());
        System.out.println("Current last: " + numbers.getLast());

        // When to use LinkedList vs ArrayList?
        System.out.println("");
        System.out.println("LinkedList vs ArrayList:");
        System.out.println("Use LinkedList when:");
        System.out.println("  - Frequent insertions/deletions at beginning or end");
        System.out.println("  - Implementing queue or deque operations");
        System.out.println("  - Memory overhead is acceptable");
        System.out.println("");
        System.out.println("Use ArrayList when:");
        System.out.println("  - Random access (get by index) is frequent");
        System.out.println("  - Memory efficiency is important");
        System.out.println("  - Mostly appending to the end");
    }
}`,
    hint1: `LinkedList has special methods for both ends: addFirst/addLast, removeFirst/removeLast, getFirst/getLast.`,
    hint2: `addFirst() adds to the front, so adding 1, 2, 3 results in [3, 2, 1]. addLast() adds to the end normally.`,
    whyItMatters: `LinkedList is essential when you need efficient insertions/deletions at both ends, such as implementing queues, stacks, or deques. Understanding when to choose it over ArrayList is crucial for performance.

**Production Pattern:**
\`\`\`java
// Task queue implementation with fast insertion
LinkedList<Task> taskQueue = new LinkedList<>();
taskQueue.addLast(newTask);  // O(1) add to end
Task next = taskQueue.removeFirst();  // O(1) remove from start

// Bidirectional log processing
LinkedList<LogEntry> recentLogs = new LinkedList<>();
recentLogs.addFirst(criticalLog);  // Critical logs to front
if (recentLogs.size() > MAX_LOGS) {
    recentLogs.removeLast();  // Remove old ones
}
\`\`\`

**Practical Benefits:**
- Fast add/remove from both ends without array reconstruction
- Ideal for implementing queues and stacks
- Does not require contiguous memory block`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show LinkedList/Deque demo
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show LinkedList/Deque demo",
            output.contains("LinkedList") || output.contains("Deque") ||
            output.contains("addFirst") || output.contains("First"));
    }
}

// Test2: Output should show addFirst result
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention addFirst", output.contains("addfirst") || output.contains("front") || output.contains("начало"));
    }
}

// Test3: Output should show addLast result
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention addLast", output.contains("addlast") || output.contains("end") || output.contains("конец"));
    }
}

// Test4: Output should show numbers 7, 8, 9 added
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 7", output.contains("7"));
        assertTrue("Should contain 8", output.contains("8"));
        assertTrue("Should contain 9", output.contains("9"));
    }
}

// Test5: Output should show add(10)
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 10", output.contains("10"));
    }
}

// Test6: Output should show removeFirst/removeLast
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention removal", output.contains("remov") || output.contains("удален") || output.contains("o'chirildi"));
    }
}

// Test7: Output should show getFirst/getLast results
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention first element", output.contains("first") || output.contains("первый") || output.contains("birinchi"));
        assertTrue("Should mention last element", output.contains("last") || output.contains("последний") || output.contains("oxirgi"));
    }
}

// Test8: Output should contain comparison LinkedList vs ArrayList
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention LinkedList", output.contains("LinkedList"));
        assertTrue("Should mention ArrayList", output.contains("ArrayList"));
    }
}

// Test9: Output should mention when to use LinkedList
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention insertion/deletion or queue",
            output.contains("insert") || output.contains("delet") || output.contains("queue") ||
            output.contains("вставк") || output.contains("удален") || output.contains("очередь"));
    }
}

// Test10: Output should mention when to use ArrayList
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        LinkedListDeque.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention random access or index",
            output.contains("random") || output.contains("access") || output.contains("index") ||
            output.contains("произвольн") || output.contains("индекс"));
    }
}
`,
    translations: {
        ru: {
            title: 'LinkedList и операции Deque',
            solutionCode: `import java.util.LinkedList;

public class LinkedListDeque {
    public static void main(String[] args) {
        // Создаем LinkedList для хранения целых чисел
        LinkedList<Integer> numbers = new LinkedList<>();

        // Используем addFirst() для добавления 1, 2, 3 в начало
        // Примечание: каждый addFirst добавляет в начало, поэтому порядок будет обратным
        numbers.addFirst(1);
        numbers.addFirst(2);
        numbers.addFirst(3);
        System.out.println("После addFirst: " + numbers);

        // Используем addLast() для добавления 7, 8, 9 в конец
        numbers.addLast(7);
        numbers.addLast(8);
        numbers.addLast(9);
        System.out.println("После addLast: " + numbers);

        // Используем add() для добавления 10 (эквивалент addLast)
        numbers.add(10);
        System.out.println("После add(10): " + numbers);

        // Удаляем первый и последний элементы
        Integer first = numbers.removeFirst();
        Integer last = numbers.removeLast();
        System.out.println("Удалены первый: " + first + ", последний: " + last);
        System.out.println("После удаления: " + numbers);

        // Выводим первый и последний элементы
        System.out.println("Текущий первый: " + numbers.getFirst());
        System.out.println("Текущий последний: " + numbers.getLast());

        // Когда использовать LinkedList вместо ArrayList?
        System.out.println("");
        System.out.println("LinkedList vs ArrayList:");
        System.out.println("Используйте LinkedList когда:");
        System.out.println("  - Частые вставки/удаления в начале или конце");
        System.out.println("  - Реализация очереди или двусторонней очереди");
        System.out.println("  - Дополнительные затраты памяти допустимы");
        System.out.println("");
        System.out.println("Используйте ArrayList когда:");
        System.out.println("  - Частый произвольный доступ (get по индексу)");
        System.out.println("  - Важна эффективность памяти");
        System.out.println("  - Преимущественно добавление в конец");
    }
}`,
            description: `Изучите LinkedList как List и Deque.

**Требования:**
1. Создайте LinkedList для хранения целых чисел
2. Используйте addFirst() для добавления элементов 1, 2, 3 в начало
3. Используйте addLast() для добавления элементов 7, 8, 9 в конец
4. Используйте add() для добавления элемента 10
5. Выведите список
6. Удалите первый и последний элементы используя removeFirst() и removeLast()
7. Используйте getFirst() и getLast() для вывода текущих первого и последнего элементов
8. Объясните, когда использовать LinkedList вместо ArrayList

LinkedList реализует интерфейсы List и Deque, что делает его идеальным для операций с очередями и стеками.`,
            hint1: `LinkedList имеет специальные методы для обоих концов: addFirst/addLast, removeFirst/removeLast, getFirst/getLast.`,
            hint2: `addFirst() добавляет в начало, поэтому добавление 1, 2, 3 дает [3, 2, 1]. addLast() добавляет в конец обычным образом.`,
            whyItMatters: `LinkedList необходим, когда нужны эффективные вставки/удаления с обоих концов, например, при реализации очередей, стеков или двусторонних очередей. Понимание, когда выбрать его вместо ArrayList, критически важно для производительности.

**Продакшен паттерн:**
\`\`\`java
// Реализация очереди задач с быстрой вставкой
LinkedList<Task> taskQueue = new LinkedList<>();
taskQueue.addLast(newTask);  // O(1) добавление в конец
Task next = taskQueue.removeFirst();  // O(1) удаление из начала

// Двусторонняя обработка логов
LinkedList<LogEntry> recentLogs = new LinkedList<>();
recentLogs.addFirst(criticalLog);  // Критичные логи в начало
if (recentLogs.size() > MAX_LOGS) {
    recentLogs.removeLast();  // Удаляем старые
}
\`\`\`

**Практические преимущества:**
- Быстрое добавление/удаление с обоих концов без перестройки массива
- Идеален для реализации очередей и стеков
- Не требует непрерывного блока памяти`
        },
        uz: {
            title: 'LinkedList va Deque Operatsiyalari',
            solutionCode: `import java.util.LinkedList;

public class LinkedListDeque {
    public static void main(String[] args) {
        // Butun sonlarni saqlash uchun LinkedList yaratamiz
        LinkedList<Integer> numbers = new LinkedList<>();

        // addFirst() yordamida 1, 2, 3 ni boshiga qo'shamiz
        // Eslatma: har bir addFirst boshiga qo'shadi, shuning uchun tartib teskari bo'ladi
        numbers.addFirst(1);
        numbers.addFirst(2);
        numbers.addFirst(3);
        System.out.println("addFirst dan keyin: " + numbers);

        // addLast() yordamida 7, 8, 9 ni oxiriga qo'shamiz
        numbers.addLast(7);
        numbers.addLast(8);
        numbers.addLast(9);
        System.out.println("addLast dan keyin: " + numbers);

        // add() yordamida 10 ni qo'shamiz (addLast ga teng)
        numbers.add(10);
        System.out.println("add(10) dan keyin: " + numbers);

        // Birinchi va oxirgi elementlarni o'chiramiz
        Integer first = numbers.removeFirst();
        Integer last = numbers.removeLast();
        System.out.println("O'chirildi birinchi: " + first + ", oxirgi: " + last);
        System.out.println("O'chirilgandan keyin: " + numbers);

        // Birinchi va oxirgi elementlarni chiqaramiz
        System.out.println("Hozirgi birinchi: " + numbers.getFirst());
        System.out.println("Hozirgi oxirgi: " + numbers.getLast());

        // LinkedList va ArrayList qachon ishlatiladi?
        System.out.println("");
        System.out.println("LinkedList vs ArrayList:");
        System.out.println("LinkedList dan foydalaning:");
        System.out.println("  - Boshi yoki oxirida tez-tez qo'shish/o'chirish");
        System.out.println("  - Navbat yoki ikki tomonlama navbat amalga oshirish");
        System.out.println("  - Xotira sarfi qabul qilinadigan bo'lsa");
        System.out.println("");
        System.out.println("ArrayList dan foydalaning:");
        System.out.println("  - Tasodifiy kirish (indeks bo'yicha get) tez-tez");
        System.out.println("  - Xotira samaradorligi muhim");
        System.out.println("  - Asosan oxiriga qo'shish");
    }
}`,
            description: `LinkedList ni List va Deque sifatida o'rganing.

**Talablar:**
1. Butun sonlarni saqlash uchun LinkedList yarating
2. addFirst() yordamida 1, 2, 3 elementlarni boshiga qo'shing
3. addLast() yordamida 7, 8, 9 elementlarni oxiriga qo'shing
4. add() yordamida 10 elementni qo'shing
5. Ro'yxatni chiqaring
6. removeFirst() va removeLast() yordamida birinchi va oxirgi elementlarni o'chiring
7. getFirst() va getLast() yordamida hozirgi birinchi va oxirgi elementlarni chiqaring
8. LinkedList va ArrayList qachon ishlatilishini tushuntiring

LinkedList List va Deque interfeyslarini amalga oshiradi, bu uni navbat va stek operatsiyalari uchun ideal qiladi.`,
            hint1: `LinkedList ikki uch uchun maxsus metodlarga ega: addFirst/addLast, removeFirst/removeLast, getFirst/getLast.`,
            hint2: `addFirst() boshiga qo'shadi, shuning uchun 1, 2, 3 qo'shish [3, 2, 1] ni beradi. addLast() oddiy tarzda oxiriga qo'shadi.`,
            whyItMatters: `LinkedList ikki uchdan ham samarali qo'shish/o'chirish kerak bo'lganda zarur, masalan, navbatlar, steklar yoki ikki tomonlama navbatlarni amalga oshirishda. Qachon ArrayList o'rniga uni tanlashni tushunish samaradorlik uchun muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
// Tez qo'shish bilan vazifalar navbatini amalga oshirish
LinkedList<Task> taskQueue = new LinkedList<>();
taskQueue.addLast(newTask);  // O(1) oxiriga qo'shish
Task next = taskQueue.removeFirst();  // O(1) boshidan o'chirish

// Loglarni ikki tomonlama qayta ishlash
LinkedList<LogEntry> recentLogs = new LinkedList<>();
recentLogs.addFirst(criticalLog);  // Muhim loglar boshiga
if (recentLogs.size() > MAX_LOGS) {
    recentLogs.removeLast();  // Eskilarini o'chirish
}
\`\`\`

**Amaliy foydalari:**
- Massivni qayta qurishsiz ikki uchdan tez qo'shish/o'chirish
- Navbat va steklar uchun ideal
- Uzluksiz xotira bloki talab qilmaydi`
        }
    }
};

export default task;
