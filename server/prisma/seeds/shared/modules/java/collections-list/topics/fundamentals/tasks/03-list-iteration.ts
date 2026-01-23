import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-list-iteration',
    title: 'List Iteration Methods',
    difficulty: 'easy',
    tags: ['java', 'collections', 'list', 'iteration', 'iterator'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master different ways to iterate over lists.

**Requirements:**
1. Create a list with numbers 1 through 5
2. Iterate using enhanced for-each loop
3. Iterate using Iterator and print each element
4. Iterate using ListIterator and print index and element
5. Remove all even numbers using Iterator.remove()
6. Try to demonstrate why removing during for-each fails
7. Show iterating backwards with ListIterator

Understanding safe iteration and removal is crucial for avoiding ConcurrentModificationException.`,
    initialCode: `import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

public class ListIteration {
    public static void main(String[] args) {
        // Create a list with numbers 1 through 5

        // Iterate using enhanced for-each loop

        // Iterate using Iterator

        // Iterate using ListIterator with index

        // Remove even numbers using Iterator

        // Show why removing during for-each fails

        // Iterate backwards with ListIterator
    }
}`,
    solutionCode: `import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.List;

public class ListIteration {
    public static void main(String[] args) {
        // Create a list with numbers 1 through 5
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            numbers.add(i);
        }

        // Iterate using enhanced for-each loop
        System.out.println("Enhanced for-each loop:");
        for (Integer num : numbers) {
            System.out.print(num + " ");
        }
        System.out.println();

        // Iterate using Iterator
        System.out.println("");
        System.out.println("Using Iterator:");
        Iterator<Integer> iterator = numbers.iterator();
        while (iterator.hasNext()) {
            Integer num = iterator.next();
            System.out.print(num + " ");
        }
        System.out.println();

        // Iterate using ListIterator with index
        System.out.println("");
        System.out.println("Using ListIterator with index:");
        ListIterator<Integer> listIterator = numbers.listIterator();
        while (listIterator.hasNext()) {
            int index = listIterator.nextIndex();
            Integer num = listIterator.next();
            System.out.println("Index " + index + ": " + num);
        }

        // Remove even numbers using Iterator (SAFE way)
        System.out.println("");
        System.out.println("Removing even numbers using Iterator:");
        Iterator<Integer> removeIterator = numbers.iterator();
        while (removeIterator.hasNext()) {
            Integer num = removeIterator.next();
            if (num % 2 == 0) {
                removeIterator.remove(); // Safe removal
            }
        }
        System.out.println("After removal: " + numbers);

        // Show why removing during for-each fails
        System.out.println("");
        System.out.println("Demonstrating ConcurrentModificationException:");
        List<Integer> testList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        try {
            for (Integer num : testList) {
                if (num % 2 == 0) {
                    testList.remove(num); // This throws exception!
                }
            }
        } catch (Exception e) {
            System.out.println("Exception caught: " + e.getClass().getSimpleName());
            System.out.println("Cannot modify list during for-each!");
        }

        // Iterate backwards with ListIterator
        System.out.println("");
        System.out.println("Iterating backwards:");
        ListIterator<Integer> backwardIterator = numbers.listIterator(numbers.size());
        while (backwardIterator.hasPrevious()) {
            Integer num = backwardIterator.previous();
            System.out.print(num + " ");
        }
        System.out.println();
    }
}`,
    hint1: `Iterator has hasNext(), next(), and remove() methods. Use Iterator.remove() for safe removal during iteration.`,
    hint2: `ListIterator extends Iterator and adds hasPrevious(), previous(), nextIndex(), and previousIndex() methods.`,
    whyItMatters: `Proper iteration is fundamental to working with collections. Using the wrong approach can lead to ConcurrentModificationException. Iterator.remove() is the only safe way to remove elements during iteration.

**Production Pattern:**
\`\`\`java
// Safe removal during event processing
Iterator<Event> it = events.iterator();
while (it.hasNext()) {
    Event event = it.next();
    if (event.isExpired()) {
        it.remove();  // Safe removal
        event.cleanup();
    }
}

// Backward iteration for reverse processing
ListIterator<Command> lit = commands.listIterator(commands.size());
while (lit.hasPrevious()) {
    Command cmd = lit.previous();
    cmd.undo();  // Undo in reverse order
}
\`\`\`

**Practical Benefits:**
- Prevents ConcurrentModificationException in production code
- Safe modification of collections during traversal
- Supports bidirectional iteration for complex algorithms`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show List iteration demo
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show List iteration demo",
            output.contains("Iteration") || output.contains("for-each") ||
            output.contains("Iterator") || output.contains("loop"));
    }
}

// Test2: Output should show enhanced for-each loop
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention for-each",
            output.contains("for-each") || output.contains("foreach") || output.contains("enhanced") ||
            output.contains("расширенн") || output.contains("kengaytirilgan"));
    }
}

// Test3: Output should show numbers 1-5
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 1", output.contains("1"));
        assertTrue("Should contain 5", output.contains("5"));
    }
}

// Test4: Output should show Iterator usage
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention Iterator", output.contains("Iterator") || output.contains("iterator"));
    }
}

// Test5: Output should show ListIterator with index
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention ListIterator", output.contains("ListIterator"));
    }
}

// Test6: Output should show removing even numbers
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention removing even numbers",
            output.contains("remov") || output.contains("even") ||
            output.contains("удален") || output.contains("чётн") || output.contains("juft"));
    }
}

// Test7: Output should mention ConcurrentModificationException or for-each fail
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should explain for-each modification issue",
            output.contains("concurrent") || output.contains("fail") || output.contains("exception") ||
            output.contains("cannot") || output.contains("нельзя") || output.contains("ошибк"));
    }
}

// Test8: Output should show backwards iteration
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention backwards iteration",
            output.contains("backward") || output.contains("reverse") || output.contains("previous") ||
            output.contains("обратн") || output.contains("teskari"));
    }
}

// Test9: Output should show index during iteration
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention index",
            output.contains("index") || output.contains("[0]") || output.contains("индекс"));
    }
}

// Test10: After removing evens, should show only odds
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListIteration.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show result with odd numbers [1, 3, 5]",
            output.contains("1") && output.contains("3") && output.contains("5"));
    }
}
`,
    translations: {
        ru: {
            title: 'Методы итерации по списку',
            solutionCode: `import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.List;

public class ListIteration {
    public static void main(String[] args) {
        // Создаем список с числами от 1 до 5
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            numbers.add(i);
        }

        // Итерируем используя расширенный for-each цикл
        System.out.println("Расширенный for-each цикл:");
        for (Integer num : numbers) {
            System.out.print(num + " ");
        }
        System.out.println();

        // Итерируем используя Iterator
        System.out.println("");
        System.out.println("Использование Iterator:");
        Iterator<Integer> iterator = numbers.iterator();
        while (iterator.hasNext()) {
            Integer num = iterator.next();
            System.out.print(num + " ");
        }
        System.out.println();

        // Итерируем используя ListIterator с индексом
        System.out.println("");
        System.out.println("Использование ListIterator с индексом:");
        ListIterator<Integer> listIterator = numbers.listIterator();
        while (listIterator.hasNext()) {
            int index = listIterator.nextIndex();
            Integer num = listIterator.next();
            System.out.println("Индекс " + index + ": " + num);
        }

        // Удаляем четные числа используя Iterator (БЕЗОПАСНЫЙ способ)
        System.out.println("");
        System.out.println("Удаление четных чисел через Iterator:");
        Iterator<Integer> removeIterator = numbers.iterator();
        while (removeIterator.hasNext()) {
            Integer num = removeIterator.next();
            if (num % 2 == 0) {
                removeIterator.remove(); // Безопасное удаление
            }
        }
        System.out.println("После удаления: " + numbers);

        // Показываем почему удаление в for-each не работает
        System.out.println("");
        System.out.println("Демонстрация ConcurrentModificationException:");
        List<Integer> testList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        try {
            for (Integer num : testList) {
                if (num % 2 == 0) {
                    testList.remove(num); // Это вызывает исключение!
                }
            }
        } catch (Exception e) {
            System.out.println("Исключение поймано: " + e.getClass().getSimpleName());
            System.out.println("Нельзя изменять список во время for-each!");
        }

        // Итерируем в обратном порядке с ListIterator
        System.out.println("");
        System.out.println("Итерация в обратном порядке:");
        ListIterator<Integer> backwardIterator = numbers.listIterator(numbers.size());
        while (backwardIterator.hasPrevious()) {
            Integer num = backwardIterator.previous();
            System.out.print(num + " ");
        }
        System.out.println();
    }
}`,
            description: `Освойте различные способы итерации по спискам.

**Требования:**
1. Создайте список с числами от 1 до 5
2. Итерируйте используя расширенный for-each цикл
3. Итерируйте используя Iterator и выводите каждый элемент
4. Итерируйте используя ListIterator и выводите индекс и элемент
5. Удалите все четные числа используя Iterator.remove()
6. Попробуйте продемонстрировать, почему удаление в for-each не работает
7. Покажите обратную итерацию с ListIterator

Понимание безопасной итерации и удаления критически важно для избежания ConcurrentModificationException.`,
            hint1: `Iterator имеет методы hasNext(), next() и remove(). Используйте Iterator.remove() для безопасного удаления во время итерации.`,
            hint2: `ListIterator расширяет Iterator и добавляет методы hasPrevious(), previous(), nextIndex() и previousIndex().`,
            whyItMatters: `Правильная итерация фундаментальна для работы с коллекциями. Использование неправильного подхода может привести к ConcurrentModificationException. Iterator.remove() - единственный безопасный способ удалять элементы во время итерации.

**Продакшен паттерн:**
\`\`\`java
// Безопасное удаление при обработке событий
Iterator<Event> it = events.iterator();
while (it.hasNext()) {
    Event event = it.next();
    if (event.isExpired()) {
        it.remove();  // Безопасное удаление
        event.cleanup();
    }
}

// Обратная итерация для обработки в обратном порядке
ListIterator<Command> lit = commands.listIterator(commands.size());
while (lit.hasPrevious()) {
    Command cmd = lit.previous();
    cmd.undo();  // Отмена в обратном порядке
}
\`\`\`

**Практические преимущества:**
- Предотвращение ConcurrentModificationException в продакшен коде
- Безопасная модификация коллекций во время обхода
- Поддержка двунаправленной итерации для сложных алгоритмов`
        },
        uz: {
            title: 'Ro\'yxat bo\'ylab Iteratsiya Metodlari',
            solutionCode: `import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.List;

public class ListIteration {
    public static void main(String[] args) {
        // 1 dan 5 gacha sonlar bilan ro'yxat yaratamiz
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            numbers.add(i);
        }

        // Kengaytirilgan for-each sikl yordamida iteratsiya
        System.out.println("Kengaytirilgan for-each sikl:");
        for (Integer num : numbers) {
            System.out.print(num + " ");
        }
        System.out.println();

        // Iterator yordamida iteratsiya
        System.out.println("");
        System.out.println("Iterator dan foydalanish:");
        Iterator<Integer> iterator = numbers.iterator();
        while (iterator.hasNext()) {
            Integer num = iterator.next();
            System.out.print(num + " ");
        }
        System.out.println();

        // ListIterator yordamida indeks bilan iteratsiya
        System.out.println("");
        System.out.println("ListIterator indeks bilan:");
        ListIterator<Integer> listIterator = numbers.listIterator();
        while (listIterator.hasNext()) {
            int index = listIterator.nextIndex();
            Integer num = listIterator.next();
            System.out.println("Indeks " + index + ": " + num);
        }

        // Iterator yordamida juft sonlarni o'chirish (XAVFSIZ usul)
        System.out.println("");
        System.out.println("Iterator yordamida juft sonlarni o'chirish:");
        Iterator<Integer> removeIterator = numbers.iterator();
        while (removeIterator.hasNext()) {
            Integer num = removeIterator.next();
            if (num % 2 == 0) {
                removeIterator.remove(); // Xavfsiz o'chirish
            }
        }
        System.out.println("O'chirilgandan keyin: " + numbers);

        // for-each da o'chirish nima uchun ishlamasligini ko'rsatamiz
        System.out.println("");
        System.out.println("ConcurrentModificationException namoyishi:");
        List<Integer> testList = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        try {
            for (Integer num : testList) {
                if (num % 2 == 0) {
                    testList.remove(num); // Bu xato chiqaradi!
                }
            }
        } catch (Exception e) {
            System.out.println("Xatolik ushlandi: " + e.getClass().getSimpleName());
            System.out.println("for-each paytida ro'yxatni o'zgartirish mumkin emas!");
        }

        // ListIterator bilan teskari iteratsiya
        System.out.println("");
        System.out.println("Teskari iteratsiya:");
        ListIterator<Integer> backwardIterator = numbers.listIterator(numbers.size());
        while (backwardIterator.hasPrevious()) {
            Integer num = backwardIterator.previous();
            System.out.print(num + " ");
        }
        System.out.println();
    }
}`,
            description: `Ro'yxatlar bo'ylab iteratsiya qilishning turli usullarini o'rganing.

**Talablar:**
1. 1 dan 5 gacha sonlar bilan ro'yxat yarating
2. Kengaytirilgan for-each sikl yordamida iteratsiya qiling
3. Iterator yordamida iteratsiya qiling va har bir elementni chiqaring
4. ListIterator yordamida iteratsiya qiling va indeks va elementni chiqaring
5. Iterator.remove() yordamida barcha juft sonlarni o'chiring
6. for-each da o'chirish nima uchun ishlamasligini ko'rsating
7. ListIterator bilan teskari iteratsiyani ko'rsating

Xavfsiz iteratsiya va o'chirishni tushunish ConcurrentModificationException dan qochish uchun muhim.`,
            hint1: `Iterator da hasNext(), next() va remove() metodlari bor. Iteratsiya paytida xavfsiz o'chirish uchun Iterator.remove() dan foydalaning.`,
            hint2: `ListIterator Iterator ni kengaytiradi va hasPrevious(), previous(), nextIndex() va previousIndex() metodlarini qo'shadi.`,
            whyItMatters: `To'g'ri iteratsiya kolleksiyalar bilan ishlash uchun asosdir. Noto'g'ri usuldan foydalanish ConcurrentModificationException ga olib kelishi mumkin. Iterator.remove() - iteratsiya paytida elementlarni o'chirishning yagona xavfsiz usuli.

**Ishlab chiqarish patterni:**
\`\`\`java
// Hodisalarni qayta ishlashda xavfsiz o'chirish
Iterator<Event> it = events.iterator();
while (it.hasNext()) {
    Event event = it.next();
    if (event.isExpired()) {
        it.remove();  // Xavfsiz o'chirish
        event.cleanup();
    }
}

// Teskari tartibda qayta ishlash uchun teskari iteratsiya
ListIterator<Command> lit = commands.listIterator(commands.size());
while (lit.hasPrevious()) {
    Command cmd = lit.previous();
    cmd.undo();  // Teskari tartibda bekor qilish
}
\`\`\`

**Amaliy foydalari:**
- Production kodda ConcurrentModificationException dan saqlash
- Aylanish paytida kolleksiyalarni xavfsiz o'zgartirish
- Murakkab algoritmlar uchun ikki yo'nalishli iteratsiyani qo'llab-quvvatlash`
        }
    }
};

export default task;
