import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-try-catch-basics',
    title: 'Try-Catch-Finally Blocks',
    difficulty: 'easy',
    tags: ['java', 'exceptions', 'try-catch', 'finally'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a SafeDivision class that safely performs division operations with proper exception handling.

Requirements:
1. Implement divide() method that takes two integers and returns their division result
2. Handle ArithmeticException when dividing by zero
3. Use a finally block to log completion
4. Return a default value (0) when exception occurs

Example:
\`\`\`java
SafeDivision sd = new SafeDivision();
sd.divide(10, 2);  // Returns: 5
sd.divide(10, 0);  // Returns: 0, catches exception
\`\`\``,
    initialCode: `public class SafeDivision {

    public int divide(int numerator, int denominator) {
        // TODO: Implement safe division with try-catch-finally
        return 0;
    }

    public static void main(String[] args) {
        SafeDivision sd = new SafeDivision();

        System.out.println("10 / 2 = " + sd.divide(10, 2));
        System.out.println("10 / 0 = " + sd.divide(10, 0));
        System.out.println("15 / 3 = " + sd.divide(15, 3));
    }
}`,
    solutionCode: `public class SafeDivision {

    public int divide(int numerator, int denominator) {
        try {
            // Attempt the division operation
            int result = numerator / denominator;
            System.out.println("Division successful: " + result);
            return result;
        } catch (ArithmeticException e) {
            // Handle division by zero
            System.out.println("Error: Cannot divide by zero - " + e.getMessage());
            return 0;
        } finally {
            // This block always executes
            System.out.println("Division operation completed");
        }
    }

    public static void main(String[] args) {
        SafeDivision sd = new SafeDivision();

        System.out.println("10 / 2 = " + sd.divide(10, 2));
        System.out.println("10 / 0 = " + sd.divide(10, 0));
        System.out.println("15 / 3 = " + sd.divide(15, 3));
    }
}`,
    hint1: `Use a try block to wrap the division operation, a catch block to handle ArithmeticException, and a finally block for cleanup code.`,
    hint2: `The finally block executes whether an exception is thrown or not - perfect for logging or cleanup operations.`,
    whyItMatters: `Try-catch-finally is the foundation of exception handling in Java. Understanding this pattern is essential for writing robust applications that can gracefully handle errors without crashing.

**Production Pattern:**
\`\`\`java
try {
    int result = performOperation();
    logger.info("Operation completed successfully");
    return result;
} catch (SpecificException e) {
    logger.error("Operation failed", e);
    metricsService.incrementErrorCounter("operation_error");
    return defaultValue;
} finally {
    cleanup();
}
\`\`\`

**Practical Benefits:**
- Structured error handling with logging
- Metrics for performance monitoring
- Guaranteed resource cleanup`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: successful division returns correct result
class Test1 {
    @Test
    public void test() {
        SafeDivision sd = new SafeDivision();
        int result = sd.divide(10, 2);
        assertEquals("10 / 2 should equal 5", 5, result);
    }
}

// Test2: division by zero returns 0
class Test2 {
    @Test
    public void test() {
        SafeDivision sd = new SafeDivision();
        int result = sd.divide(10, 0);
        assertEquals("Division by zero should return 0", 0, result);
    }
}

// Test3: successful division prints success message
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision sd = new SafeDivision();
        sd.divide(10, 2);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should print success message",
            output.contains("successful") || output.contains("5") ||
            output.contains("успешно") || output.contains("muvaffaqiyatli"));
    }
}

// Test4: division by zero prints error message
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision sd = new SafeDivision();
        sd.divide(10, 0);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should print error message for division by zero",
            output.contains("Error") || output.contains("zero") ||
            output.contains("Ошибка") || output.contains("ноль") ||
            output.contains("Xato") || output.contains("nol"));
    }
}

// Test5: finally block always executes (prints completed message)
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision sd = new SafeDivision();
        sd.divide(10, 2);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Finally should print completed message",
            output.contains("completed") || output.contains("завершен") || output.contains("yakunlandi"));
    }
}

// Test6: finally executes even on exception
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision sd = new SafeDivision();
        sd.divide(10, 0);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Finally should execute even on error",
            output.contains("completed") || output.contains("завершен") || output.contains("yakunlandi"));
    }
}

// Test7: negative number division works
class Test7 {
    @Test
    public void test() {
        SafeDivision sd = new SafeDivision();
        assertEquals("-10 / 2 should equal -5", -5, sd.divide(-10, 2));
        assertEquals("10 / -2 should equal -5", -5, sd.divide(10, -2));
    }
}

// Test8: main method produces expected output
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Main should print 10 / 2 result",
            output.contains("10 / 2") || output.contains("5"));
    }
}

// Test9: main handles division by zero gracefully
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SafeDivision.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Main should show error handling for 10 / 0",
            output.contains("10 / 0") || output.contains("Error") ||
            output.contains("Ошибка") || output.contains("Xato"));
    }
}

// Test10: multiple divisions work correctly
class Test10 {
    @Test
    public void test() {
        SafeDivision sd = new SafeDivision();
        assertEquals("20 / 5 = 4", 4, sd.divide(20, 5));
        assertEquals("21 / 3 = 7", 7, sd.divide(21, 3));
        assertEquals("100 / 1 = 100", 100, sd.divide(100, 1));
        assertEquals("0 / 5 = 0", 0, sd.divide(0, 5));
    }
}
`,
    translations: {
        ru: {
            title: 'Блоки Try-Catch-Finally',
            solutionCode: `public class SafeDivision {

    public int divide(int numerator, int denominator) {
        try {
            // Попытка выполнить операцию деления
            int result = numerator / denominator;
            System.out.println("Деление выполнено успешно: " + result);
            return result;
        } catch (ArithmeticException e) {
            // Обработка деления на ноль
            System.out.println("Ошибка: Невозможно делить на ноль - " + e.getMessage());
            return 0;
        } finally {
            // Этот блок выполняется всегда
            System.out.println("Операция деления завершена");
        }
    }

    public static void main(String[] args) {
        SafeDivision sd = new SafeDivision();

        System.out.println("10 / 2 = " + sd.divide(10, 2));
        System.out.println("10 / 0 = " + sd.divide(10, 0));
        System.out.println("15 / 3 = " + sd.divide(15, 3));
    }
}`,
            description: `Создайте класс SafeDivision, который безопасно выполняет операции деления с правильной обработкой исключений.

Требования:
1. Реализуйте метод divide(), который принимает два целых числа и возвращает результат их деления
2. Обработайте ArithmeticException при делении на ноль
3. Используйте блок finally для логирования завершения
4. Возвращайте значение по умолчанию (0) при возникновении исключения

Пример:
\`\`\`java
SafeDivision sd = new SafeDivision();
sd.divide(10, 2);  // Возвращает: 5
sd.divide(10, 0);  // Возвращает: 0, перехватывает исключение
\`\`\``,
            hint1: `Используйте блок try для оборачивания операции деления, блок catch для обработки ArithmeticException и блок finally для кода очистки.`,
            hint2: `Блок finally выполняется независимо от того, было ли выброшено исключение - идеально подходит для логирования или операций очистки.`,
            whyItMatters: `Try-catch-finally - это основа обработки исключений в Java. Понимание этого паттерна необходимо для написания надежных приложений, которые могут корректно обрабатывать ошибки без сбоев.

**Продакшен паттерн:**
\`\`\`java
try {
    int result = performOperation();
    logger.info("Operation completed successfully");
    return result;
} catch (SpecificException e) {
    logger.error("Operation failed", e);
    metricsService.incrementErrorCounter("operation_error");
    return defaultValue;
} finally {
    cleanup();
}
\`\`\`

**Практические преимущества:**
- Структурированная обработка ошибок с логированием
- Метрики для мониторинга производительности
- Гарантированная очистка ресурсов`
        },
        uz: {
            title: `Try-Catch-Finally Bloklari`,
            solutionCode: `public class SafeDivision {

    public int divide(int numerator, int denominator) {
        try {
            // Bo'lish amalini bajarish urinishi
            int result = numerator / denominator;
            System.out.println("Bo'lish muvaffaqiyatli bajarildi: " + result);
            return result;
        } catch (ArithmeticException e) {
            // Nolga bo'lishni qayta ishlash
            System.out.println("Xato: Nolga bo'lish mumkin emas - " + e.getMessage());
            return 0;
        } finally {
            // Bu blok har doim bajariladi
            System.out.println("Bo'lish amali yakunlandi");
        }
    }

    public static void main(String[] args) {
        SafeDivision sd = new SafeDivision();

        System.out.println("10 / 2 = " + sd.divide(10, 2));
        System.out.println("10 / 0 = " + sd.divide(10, 0));
        System.out.println("15 / 3 = " + sd.divide(15, 3));
    }
}`,
            description: `SafeDivision klassini yarating, u bo'lish amallarini to'g'ri istisno qayta ishlash bilan xavfsiz bajaradi.

Talablar:
1. divide() metodini yarating, u ikkita butun sonni qabul qilib, ularning bo'linish natijasini qaytaradi
2. Nolga bo'lganda ArithmeticException ni qayta ishlang
3. Tugashni jurnallashtirish uchun finally blokidan foydalaning
4. Istisno yuzaga kelganda standart qiymat (0) qaytaring

Misol:
\`\`\`java
SafeDivision sd = new SafeDivision();
sd.divide(10, 2);  // Qaytaradi: 5
sd.divide(10, 0);  // Qaytaradi: 0, istisnoni ushlab qoladi
\`\`\``,
            hint1: `Bo'lish amalini o'rash uchun try blokidan, ArithmeticException ni qayta ishlash uchun catch blokidan va tozalash kodi uchun finally blokidan foydalaning.`,
            hint2: `Finally bloki istisno tashlangan yoki tashlanmaganidan qat'i nazar bajariladi - jurnallashtirish yoki tozalash amallari uchun juda mos.`,
            whyItMatters: `Try-catch-finally Java dasturlashda istisnolarni qayta ishlashning asosi hisoblanadi. Ushbu naqshni tushunish ishonchli ilovalar yozish uchun zarur bo'lib, ular xatolarni to'g'ri qayta ishlab, nosozliksiz ishlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
try {
    int result = performOperation();
    logger.info("Operatsiya muvaffaqiyatli yakunlandi");
    return result;
} catch (SpecificException e) {
    logger.error("Operatsiya muvaffaqiyatsiz", e);
    metricsService.incrementErrorCounter("operation_error");
    return defaultValue;
} finally {
    cleanup();
}
\`\`\`

**Amaliy foydalari:**
- Jurnallashtirish bilan tuzilgan xatolarni qayta ishlash
- Unumdorlikni monitoring qilish uchun metrikalar
- Resurslarni tozalash kafolatlanadi`
        }
    }
};

export default task;
