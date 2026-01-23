import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-control-flow-statements',
    title: 'Control Flow Statements',
    difficulty: 'easy',
    tags: ['java', 'syntax', 'control-flow', 'loops', 'conditionals'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Java control flow: conditionals, loops, and flow control statements.

**Requirements:**
1. Create a \`demonstrateConditionals(int score)\` method using if-else statements to grade scores
2. Implement a \`demonstrateLoops()\` method showing for, while, and do-while loops
3. Create a \`demonstrateFlowControl()\` method using break, continue, and labeled statements
4. Implement a \`nestedLoopsExample()\` method to print a multiplication table

**Concepts to Cover:**
- if, else if, else statements
- for loop (traditional and enhanced)
- while and do-while loops
- break and continue statements
- Labeled statements
- Nested loops

**Example Output:**
\`\`\`
Score: 85 - Grade: B
For loop: 1 2 3 4 5
While loop: 5 4 3 2 1
...\`\`\``,
    initialCode: `public class ControlFlowDemo {

    public static void demonstrateConditionals(int score) {
        // TODO: Use if-else to assign grades
        // 90-100: A, 80-89: B, 70-79: C, 60-69: D, <60: F


    }

    public static void demonstrateLoops() {
        // TODO: Demonstrate for, while, and do-while loops


    }

    public static void demonstrateFlowControl() {
        // TODO: Show break, continue, and labeled statements


    }

    public static void nestedLoopsExample() {
        // TODO: Print a 5x5 multiplication table


    }

    public static void main(String[] args) {
        demonstrateConditionals(85);
        demonstrateLoops();
        demonstrateFlowControl();
        nestedLoopsExample();
    }
}`,
    solutionCode: `public class ControlFlowDemo {

    public static void demonstrateConditionals(int score) {
        System.out.println("=== Conditional Statements ===");
        System.out.print("Score: " + score + " - Grade: ");

        // If-else chain for grading
        if (score >= 90 && score <= 100) {
            System.out.println("A (Excellent)");
        } else if (score >= 80) {
            System.out.println("B (Good)");
        } else if (score >= 70) {
            System.out.println("C (Average)");
        } else if (score >= 60) {
            System.out.println("D (Below Average)");
        } else if (score >= 0) {
            System.out.println("F (Fail)");
        } else {
            System.out.println("Invalid score!");
        }

        // Ternary operator
        String result = (score >= 60) ? "Pass" : "Fail";
        System.out.println("Result: " + result);

        // Nested if
        if (score >= 0 && score <= 100) {
            if (score >= 90) {
                System.out.println("Honor Roll!");
            }
        }
    }

    public static void demonstrateLoops() {
        System.out.println("");
        System.out.println("=== Loop Statements ===");

        // Traditional for loop
        System.out.print("For loop (1-5): ");
        for (int i = 1; i <= 5; i++) {
            System.out.print(i + " ");
        }
        System.out.println();

        // While loop
        System.out.print("While loop (5-1): ");
        int counter = 5;
        while (counter > 0) {
            System.out.print(counter + " ");
            counter--;
        }
        System.out.println();

        // Do-while loop (executes at least once)
        System.out.print("Do-while loop: ");
        int num = 1;
        do {
            System.out.print(num + " ");
            num++;
        } while (num <= 5);
        System.out.println();

        // Enhanced for loop (for-each)
        System.out.print("Enhanced for loop: ");
        int[] numbers = {10, 20, 30, 40, 50};
        for (int n : numbers) {
            System.out.print(n + " ");
        }
        System.out.println();
    }

    public static void demonstrateFlowControl() {
        System.out.println("");
        System.out.println("=== Flow Control (break, continue) ===");

        // Break statement
        System.out.print("Break at 5: ");
        for (int i = 1; i <= 10; i++) {
            if (i == 5) {
                break; // Exit loop when i is 5
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Continue statement
        System.out.print("Skip even numbers: ");
        for (int i = 1; i <= 10; i++) {
            if (i % 2 == 0) {
                continue; // Skip even numbers
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Labeled break (for nested loops)
        System.out.println("");
        System.out.println("Labeled break example:");
        outer: // Label for outer loop
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                if (i == 2 && j == 2) {
                    System.out.println("Breaking outer loop at i=" + i + ", j=" + j);
                    break outer; // Break out of outer loop
                }
                System.out.println("i=" + i + ", j=" + j);
            }
        }
    }

    public static void nestedLoopsExample() {
        System.out.println("");
        System.out.println("=== Multiplication Table (5x5) ===");

        // Print header
        System.out.print("    ");
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%4d", i);
        }
        System.out.println("");
        System.out.println("    --------------------");

        // Print multiplication table
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%2d |", i);
            for (int j = 1; j <= 5; j++) {
                System.out.printf("%4d", i * j);
            }
            System.out.println();
        }

        // Pattern printing with nested loops
        System.out.println("");
        System.out.println("=== Triangle Pattern ===");
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print("* ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        demonstrateConditionals(85);
        demonstrateLoops();
        demonstrateFlowControl();
        nestedLoopsExample();
    }
}`,
    hint1: `Use if-else chains for multiple conditions. Remember: for loops are best when you know iteration count, while loops when you don't.`,
    hint2: `Break exits the loop entirely, continue skips to the next iteration. Labeled breaks can exit nested loops directly to a specific level.`,
    whyItMatters: `Control flow is the backbone of program logic. Mastering conditionals and loops enables you to implement algorithms, process data, and handle complex business logic. These patterns appear in virtually every Java application.

**Production Pattern:**
\`\`\`java
// Processing batch data with error control
public void processBatch(List<Task> tasks) {
    for (int i = 0; i < tasks.size(); i++) {
        Task task = tasks.get(i);
        if (task.isPriority()) {
            // Process priority tasks first
            if (!processTask(task)) {
                continue; // Skip and continue
            }
        }
        // Process remaining tasks
    }
}
\`\`\`

**Practical Benefits:**
- Precise execution control in complex scenarios
- Efficient error handling with continue/break
- Flexibility in implementing business rules`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: demonstrateConditionals should show grade A for score 95
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateConditionals(95);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 'A' for score 95", output.contains("A"));
    }
}

// Test2: demonstrateConditionals should show grade B for score 85
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateConditionals(85);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 'B' for score 85", output.contains("B"));
    }
}

// Test3: demonstrateConditionals should show grade F for score 45
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateConditionals(45);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain 'F' for score 45", output.contains("F"));
    }
}

// Test4: demonstrateLoops should show for loop output
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateLoops();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should contain 'for'", output.contains("for"));
    }
}

// Test5: demonstrateLoops should show while loop output
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateLoops();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should contain 'while'", output.contains("while"));
    }
}

// Test6: demonstrateLoops should output numbers
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateLoops();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain '1' in loop output", output.contains("1"));
        assertTrue("Should contain '5' in loop output", output.contains("5"));
    }
}

// Test7: demonstrateFlowControl should show break statement
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateFlowControl();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should contain 'break'", output.contains("break"));
    }
}

// Test8: demonstrateFlowControl should show continue or skip behavior
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.demonstrateFlowControl();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show skipping or continue behavior",
            output.contains("skip") || output.contains("continue") ||
            output.contains("пропуск") || output.contains("o'tkazib"));
    }
}

// Test9: nestedLoopsExample should show multiplication table
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.nestedLoopsExample();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain multiplication result (e.g., 25 for 5x5)", output.contains("25"));
    }
}

// Test10: nestedLoopsExample should show pattern or table
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ControlFlowDemo.nestedLoopsExample();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain asterisks (*) for pattern or table dividers",
            output.contains("*") || output.contains("-") || output.contains("|"));
    }
}
`,
    translations: {
        ru: {
            title: 'Операторы управления потоком',
            solutionCode: `public class ControlFlowDemo {

    public static void demonstrateConditionals(int score) {
        System.out.println("=== Условные операторы ===");
        System.out.print("Балл: " + score + " - Оценка: ");

        // Цепочка if-else для выставления оценок
        if (score >= 90 && score <= 100) {
            System.out.println("A (Отлично)");
        } else if (score >= 80) {
            System.out.println("B (Хорошо)");
        } else if (score >= 70) {
            System.out.println("C (Средне)");
        } else if (score >= 60) {
            System.out.println("D (Ниже среднего)");
        } else if (score >= 0) {
            System.out.println("F (Неудовлетворительно)");
        } else {
            System.out.println("Неверный балл!");
        }

        // Тернарный оператор
        String result = (score >= 60) ? "Сдано" : "Не сдано";
        System.out.println("Результат: " + result);

        // Вложенный if
        if (score >= 0 && score <= 100) {
            if (score >= 90) {
                System.out.println("Доска почета!");
            }
        }
    }

    public static void demonstrateLoops() {
        System.out.println("");
        System.out.println("=== Операторы циклов ===");

        // Традиционный цикл for
        System.out.print("Цикл for (1-5): ");
        for (int i = 1; i <= 5; i++) {
            System.out.print(i + " ");
        }
        System.out.println();

        // Цикл while
        System.out.print("Цикл while (5-1): ");
        int counter = 5;
        while (counter > 0) {
            System.out.print(counter + " ");
            counter--;
        }
        System.out.println();

        // Цикл do-while (выполняется минимум один раз)
        System.out.print("Цикл do-while: ");
        int num = 1;
        do {
            System.out.print(num + " ");
            num++;
        } while (num <= 5);
        System.out.println();

        // Расширенный цикл for (for-each)
        System.out.print("Расширенный цикл for: ");
        int[] numbers = {10, 20, 30, 40, 50};
        for (int n : numbers) {
            System.out.print(n + " ");
        }
        System.out.println();
    }

    public static void demonstrateFlowControl() {
        System.out.println("");
        System.out.println("=== Управление потоком (break, continue) ===");

        // Оператор break
        System.out.print("Прерывание на 5: ");
        for (int i = 1; i <= 10; i++) {
            if (i == 5) {
                break; // Выход из цикла когда i равно 5
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Оператор continue
        System.out.print("Пропуск четных чисел: ");
        for (int i = 1; i <= 10; i++) {
            if (i % 2 == 0) {
                continue; // Пропускаем четные числа
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Метка break (для вложенных циклов)
        System.out.println("");
        System.out.println("Пример с меткой break:");
        outer: // Метка для внешнего цикла
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                if (i == 2 && j == 2) {
                    System.out.println("Прерывание внешнего цикла при i=" + i + ", j=" + j);
                    break outer; // Выход из внешнего цикла
                }
                System.out.println("i=" + i + ", j=" + j);
            }
        }
    }

    public static void nestedLoopsExample() {
        System.out.println("");
        System.out.println("=== Таблица умножения (5x5) ===");

        // Печать заголовка
        System.out.print("    ");
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%4d", i);
        }
        System.out.println("");
        System.out.println("    --------------------");

        // Печать таблицы умножения
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%2d |", i);
            for (int j = 1; j <= 5; j++) {
                System.out.printf("%4d", i * j);
            }
            System.out.println();
        }

        // Печать узора с вложенными циклами
        System.out.println("");
        System.out.println("=== Треугольный узор ===");
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print("* ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        demonstrateConditionals(85);
        demonstrateLoops();
        demonstrateFlowControl();
        nestedLoopsExample();
    }
}`,
            description: `Освойте управление потоком в Java: условные операторы, циклы и операторы управления.

**Требования:**
1. Создайте метод \`demonstrateConditionals(int score)\`, использующий операторы if-else для выставления оценок
2. Реализуйте метод \`demonstrateLoops()\`, показывающий циклы for, while и do-while
3. Создайте метод \`demonstrateFlowControl()\`, использующий break, continue и операторы с метками
4. Реализуйте метод \`nestedLoopsExample()\` для печати таблицы умножения

**Концепции для изучения:**
- Операторы if, else if, else
- Цикл for (традиционный и расширенный)
- Циклы while и do-while
- Операторы break и continue
- Операторы с метками
- Вложенные циклы

**Пример вывода:**
\`\`\`
Балл: 85 - Оценка: B
Цикл for: 1 2 3 4 5
Цикл while: 5 4 3 2 1
...\`\`\``,
            hint1: `Используйте цепочки if-else для множественных условий. Помните: циклы for лучше когда вы знаете количество итераций, while когда не знаете.`,
            hint2: `Break полностью выходит из цикла, continue пропускает к следующей итерации. Break с меткой может выйти из вложенных циклов напрямую на определенный уровень.`,
            whyItMatters: `Управление потоком - это основа программной логики. Освоение условных операторов и циклов позволяет реализовывать алгоритмы, обрабатывать данные и обрабатывать сложную бизнес-логику. Эти паттерны встречаются практически в каждом Java-приложении.

**Продакшен паттерн:**
\`\`\`java
// Обработка пакетных данных с контролем ошибок
public void processBatch(List<Task> tasks) {
    for (int i = 0; i < tasks.size(); i++) {
        Task task = tasks.get(i);
        if (task.isPriority()) {
            // Обработка приоритетных задач первыми
            if (!processTask(task)) {
                continue; // Пропустить и продолжить
            }
        }
        // Обработка остальных задач
    }
}
\`\`\`

**Практические преимущества:**
- Точный контроль выполнения в сложных сценариях
- Эффективная обработка ошибок с continue/break
- Гибкость в реализации бизнес-правил`
        },
        uz: {
            title: `Boshqaruv oqimi operatorlari`,
            solutionCode: `public class ControlFlowDemo {

    public static void demonstrateConditionals(int score) {
        System.out.println("=== Shartli operatorlar ===");
        System.out.print("Ball: " + score + " - Baho: ");

        // Baho berish uchun if-else zanjiri
        if (score >= 90 && score <= 100) {
            System.out.println("A (A'lo)");
        } else if (score >= 80) {
            System.out.println("B (Yaxshi)");
        } else if (score >= 70) {
            System.out.println("C (O'rta)");
        } else if (score >= 60) {
            System.out.println("D (O'rtadan past)");
        } else if (score >= 0) {
            System.out.println("F (Yomon)");
        } else {
            System.out.println("Noto'g'ri ball!");
        }

        // Uchlik operator
        String result = (score >= 60) ? "O'tdi" : "O'tmadi";
        System.out.println("Natija: " + result);

        // Ichki if
        if (score >= 0 && score <= 100) {
            if (score >= 90) {
                System.out.println("Faxriy taxta!");
            }
        }
    }

    public static void demonstrateLoops() {
        System.out.println("");
        System.out.println("=== Sikl operatorlari ===");

        // An'anaviy for sikli
        System.out.print("For sikli (1-5): ");
        for (int i = 1; i <= 5; i++) {
            System.out.print(i + " ");
        }
        System.out.println();

        // While sikli
        System.out.print("While sikli (5-1): ");
        int counter = 5;
        while (counter > 0) {
            System.out.print(counter + " ");
            counter--;
        }
        System.out.println();

        // Do-while sikli (kamida bir marta bajariladi)
        System.out.print("Do-while sikli: ");
        int num = 1;
        do {
            System.out.print(num + " ");
            num++;
        } while (num <= 5);
        System.out.println();

        // Kengaytirilgan for sikli (for-each)
        System.out.print("Kengaytirilgan for sikli: ");
        int[] numbers = {10, 20, 30, 40, 50};
        for (int n : numbers) {
            System.out.print(n + " ");
        }
        System.out.println();
    }

    public static void demonstrateFlowControl() {
        System.out.println("");
        System.out.println("=== Oqimni boshqarish (break, continue) ===");

        // Break operatori
        System.out.print("5 da to'xtatish: ");
        for (int i = 1; i <= 10; i++) {
            if (i == 5) {
                break; // i 5 ga teng bo'lganda sikldan chiqish
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Continue operatori
        System.out.print("Juft sonlarni o'tkazib yuborish: ");
        for (int i = 1; i <= 10; i++) {
            if (i % 2 == 0) {
                continue; // Juft sonlarni o'tkazib yuboramiz
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // Yorliqli break (ichki sikllar uchun)
        System.out.println("");
        System.out.println("Yorliqli break misoli:");
        outer: // Tashqi sikl uchun yorliq
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                if (i == 2 && j == 2) {
                    System.out.println("Tashqi siklni to'xtatish i=" + i + ", j=" + j);
                    break outer; // Tashqi sikldan chiqish
                }
                System.out.println("i=" + i + ", j=" + j);
            }
        }
    }

    public static void nestedLoopsExample() {
        System.out.println("");
        System.out.println("=== Ko'paytirish jadvali (5x5) ===");

        // Sarlavhani chop etish
        System.out.print("    ");
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%4d", i);
        }
        System.out.println("");
        System.out.println("    --------------------");

        // Ko'paytirish jadvalini chop etish
        for (int i = 1; i <= 5; i++) {
            System.out.printf("%2d |", i);
            for (int j = 1; j <= 5; j++) {
                System.out.printf("%4d", i * j);
            }
            System.out.println();
        }

        // Ichki sikllar bilan naqsh chop etish
        System.out.println("");
        System.out.println("=== Uchburchak naqshi ===");
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print("* ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        demonstrateConditionals(85);
        demonstrateLoops();
        demonstrateFlowControl();
        nestedLoopsExample();
    }
}`,
            description: `Java da boshqaruv oqimini o'zlashtiring: shartli operatorlar, sikllar va boshqaruv operatorlari.

**Talablar:**
1. Baholarni berish uchun if-else operatorlaridan foydalanadigan \`demonstrateConditionals(int score)\` metodini yarating
2. for, while va do-while siklarini ko'rsatadigan \`demonstrateLoops()\` metodini yarating
3. break, continue va yorliqli operatorlarni ishlatadigan \`demonstrateFlowControl()\` metodini yarating
4. Ko'paytirish jadvalini chop etish uchun \`nestedLoopsExample()\` metodini yarating

**O'rganish uchun tushunchalar:**
- if, else if, else operatorlari
- for sikli (an'anaviy va kengaytirilgan)
- while va do-while siklari
- break va continue operatorlari
- Yorliqli operatorlar
- Ichki sikllar

**Chiqish namunasi:**
\`\`\`
Ball: 85 - Baho: B
For sikli: 1 2 3 4 5
While sikli: 5 4 3 2 1
...\`\`\``,
            hint1: `Ko'p shartlar uchun if-else zanjirlaridan foydalaning. Esda tuting: for siklari iteratsiya sonini bilganingizda, while siklari bilmaganingizda yaxshiroq.`,
            hint2: `Break sikldan butunlay chiqadi, continue keyingi iteratsiyaga o'tadi. Yorliqli break ichki siklardan to'g'ridan-to'g'ri ma'lum bir darajaga chiqishi mumkin.`,
            whyItMatters: `Boshqaruv oqimi dastur mantiqining asosi hisoblanadi. Shartli operatorlar va siklarni o'zlashtirib olish algoritmlarni amalga oshirish, ma'lumotlarni qayta ishlash va murakkab biznes mantiqini boshqarish imkonini beradi. Bu naqshlar deyarli har bir Java ilovasida uchraydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Xatolarni nazorat qilish bilan paketli ma'lumotlarni qayta ishlash
public void processBatch(List<Task> tasks) {
    for (int i = 0; i < tasks.size(); i++) {
        Task task = tasks.get(i);
        if (task.isPriority()) {
            // Birinchi navbatda ustuvor vazifalarni qayta ishlash
            if (!processTask(task)) {
                continue; // O'tkazib yuborish va davom etish
            }
        }
        // Qolgan vazifalarni qayta ishlash
    }
}
\`\`\`

**Amaliy foydalari:**
- Murakkab stsenariylarda aniq bajarilish nazorati
- continue/break bilan samarali xatolarni qayta ishlash
- Biznes qoidalarini amalga oshirishda moslashuvchanlik`
        }
    }
};

export default task;
