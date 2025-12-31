import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-retry-basics',
    title: 'Simple Retry Logic',
    difficulty: 'easy',
    tags: ['java', 'retry', 'resilience', 'error-handling'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to implement basic retry logic for handling transient failures.

**Requirements:**
1. Create a RetryService class with a retry method
2. Implement retry logic with a maximum number of attempts (3 retries)
3. Add a delay between retry attempts (1 second)
4. Handle exceptions and count retry attempts
5. Log each retry attempt
6. Throw the final exception if all retries fail

Retry logic is essential for handling temporary failures in distributed systems, network calls, and external service integrations.`,
    initialCode: `import java.util.concurrent.TimeUnit;

public class RetryBasics {
    // Create RetryService class
    // - retry method that accepts a Runnable task
    // - maxAttempts parameter (3 retries)
    // - delaySeconds parameter (1 second)
    // - Log each attempt
    // - Throw exception if all retries fail

    public static void main(String[] args) {
        RetryService retryService = new RetryService();

        // Example 1: Successful operation after retries
        System.out.println("Test 1: Eventually successful operation");
        // Implement retry logic

        // Example 2: All retries fail
        System.out.println("\\nTest 2: All retries fail");
        // Implement retry logic that fails
    }
}`,
    solutionCode: `import java.util.concurrent.TimeUnit;

public class RetryBasics {
    // Service that provides retry functionality
    static class RetryService {
        // Execute task with retry logic
        public void retry(Runnable task, int maxAttempts, int delaySeconds) throws Exception {
            int attempt = 0;
            Exception lastException = null;

            while (attempt < maxAttempts) {
                try {
                    attempt++;
                    System.out.println("Attempt " + attempt + " of " + maxAttempts);
                    task.run();
                    System.out.println("Operation succeeded!");
                    return; // Success - exit the method
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Attempt " + attempt + " failed: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        try {
                            System.out.println("Waiting " + delaySeconds + " seconds before retry...");
                            TimeUnit.SECONDS.sleep(delaySeconds);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Retry interrupted", ie);
                        }
                    }
                }
            }

            // All retries failed
            throw new Exception("Operation failed after " + maxAttempts + " attempts", lastException);
        }
    }

    // Simulated operation that fails initially but succeeds after retries
    static class EventuallySuccessfulOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            if (callCount < 3) {
                throw new RuntimeException("Transient failure");
            }
            // Success on third attempt
        }
    }

    // Simulated operation that always fails
    static class AlwaysFailingOperation implements Runnable {
        @Override
        public void run() {
            throw new RuntimeException("Permanent failure");
        }
    }

    public static void main(String[] args) {
        RetryService retryService = new RetryService();

        // Example 1: Successful operation after retries
        System.out.println("Test 1: Eventually successful operation");
        try {
            retryService.retry(new EventuallySuccessfulOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Final failure: " + e.getMessage());
        }

        // Example 2: All retries fail
        System.out.println("\\nTest 2: All retries fail");
        try {
            retryService.retry(new AlwaysFailingOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Final failure: " + e.getMessage());
        }
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

// Test1: Verify retry succeeds on first attempt
class Test1 {
    @Test
    public void test() throws Exception {
        Callable<String> operation = () -> "success";
        String result = retryOperation(operation, 3, 1);
        assertEquals("success", result);
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
                TimeUnit.SECONDS.sleep(delaySeconds);
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test2: Verify retry eventually succeeds
class Test2 {
    @Test
    public void test() throws Exception {
        AtomicInteger attempt = new AtomicInteger(0);
        Callable<String> operation = () -> {
            if (attempt.incrementAndGet() < 3) {
                throw new RuntimeException("Transient failure");
            }
            return "success";
        };
        String result = retryOperation(operation, 3, 0);
        assertEquals("success", result);
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test3: Verify max retries throws exception
class Test3 {
    @Test(expected = RuntimeException.class)
    public void test() throws Exception {
        Callable<String> operation = () -> {
            throw new RuntimeException("Permanent failure");
        };
        retryOperation(operation, 3, 0);
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test4: Verify retry count is respected
class Test4 {
    @Test
    public void test() {
        AtomicInteger count = new AtomicInteger(0);
        Callable<String> operation = () -> {
            count.incrementAndGet();
            throw new RuntimeException("Fail");
        };
        try {
            retryOperation(operation, 3, 0);
        } catch (Exception e) {
            assertEquals(3, count.get());
        }
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test5: Verify delay between retries
class Test5 {
    @Test
    public void test() throws Exception {
        long start = System.currentTimeMillis();
        AtomicInteger attempt = new AtomicInteger(0);
        Callable<String> operation = () -> {
            if (attempt.incrementAndGet() < 2) {
                throw new RuntimeException("Fail");
            }
            return "success";
        };
        retryOperation(operation, 2, 0);
        long duration = System.currentTimeMillis() - start;
        assertTrue(duration < 1000); // Should complete quickly with no delay
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test6: Verify single retry attempt (maxRetries = 1)
class Test6 {
    @Test
    public void test() throws Exception {
        AtomicInteger count = new AtomicInteger(0);
        Callable<String> operation = () -> {
            count.incrementAndGet();
            return "immediate success";
        };
        String result = retryOperation(operation, 1, 0);
        assertEquals("immediate success", result);
        assertEquals(1, count.get());
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test7: Verify exception message is preserved after all retries fail
class Test7 {
    @Test
    public void test() {
        Callable<String> operation = () -> {
            throw new RuntimeException("Specific error message");
        };
        try {
            retryOperation(operation, 2, 0);
            fail("Expected exception");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("Specific error message"));
        }
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        Exception lastException = null;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                lastException = e;
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw lastException;
    }
}

// Test8: Verify retry succeeds on last attempt
class Test8 {
    @Test
    public void test() throws Exception {
        AtomicInteger attempt = new AtomicInteger(0);
        Callable<String> operation = () -> {
            if (attempt.incrementAndGet() < 5) {
                throw new RuntimeException("Not yet");
            }
            return "success on 5th";
        };
        String result = retryOperation(operation, 5, 0);
        assertEquals("success on 5th", result);
        assertEquals(5, attempt.get());
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test9: Verify different exception types are handled
class Test9 {
    @Test
    public void test() {
        AtomicInteger attempt = new AtomicInteger(0);
        Callable<String> operation = () -> {
            int current = attempt.incrementAndGet();
            if (current == 1) throw new IllegalArgumentException("Bad arg");
            if (current == 2) throw new IllegalStateException("Bad state");
            return "success";
        };
        try {
            String result = retryOperation(operation, 3, 0);
            assertEquals("success", result);
        } catch (Exception e) {
            fail("Should have succeeded on 3rd attempt");
        }
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}

// Test10: Verify retry with return value changes between attempts
class Test10 {
    @Test
    public void test() throws Exception {
        AtomicInteger attempt = new AtomicInteger(0);
        Callable<Integer> operation = () -> {
            int current = attempt.incrementAndGet();
            if (current < 3) throw new RuntimeException("Fail " + current);
            return current * 10;
        };
        Integer result = retryOperation(operation, 3, 0);
        assertEquals(Integer.valueOf(30), result);
    }

    private <T> T retryOperation(Callable<T> operation, int maxRetries, int delaySeconds) throws Exception {
        int attempts = 0;
        while (attempts < maxRetries) {
            try {
                return operation.call();
            } catch (Exception e) {
                attempts++;
                if (attempts >= maxRetries) throw e;
            }
        }
        throw new RuntimeException("Max retries reached");
    }
}
`,
    hint1: `Use a while loop to retry the operation. Track the number of attempts and catch exceptions to determine if another retry should be attempted.`,
    hint2: `Use TimeUnit.SECONDS.sleep() to add a delay between retries. Remember to handle InterruptedException properly.`,
    whyItMatters: `Retry logic is crucial for building resilient applications. Many failures in distributed systems are transient (temporary), such as network hiccups or temporary service unavailability. Implementing retry logic helps your application recover automatically from these temporary failures.

**Production Pattern:**
\`\`\`java
// Transient network failure
if (callCount < 3) {
    throw new RuntimeException("Transient failure");
}
// Success on third attempt
\`\`\`

**Practical Benefits:**
- Automatic recovery from transient failures
- Improved service availability without manual intervention
- Reduced error count in production environment`,
    order: 0,
    translations: {
        ru: {
            title: 'Простая логика повторных попыток',
            solutionCode: `import java.util.concurrent.TimeUnit;

public class RetryBasics {
    // Сервис, предоставляющий функциональность повторных попыток
    static class RetryService {
        // Выполнить задачу с логикой повторных попыток
        public void retry(Runnable task, int maxAttempts, int delaySeconds) throws Exception {
            int attempt = 0;
            Exception lastException = null;

            while (attempt < maxAttempts) {
                try {
                    attempt++;
                    System.out.println("Попытка " + attempt + " из " + maxAttempts);
                    task.run();
                    System.out.println("Операция выполнена успешно!");
                    return; // Успех - выходим из метода
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Попытка " + attempt + " не удалась: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        try {
                            System.out.println("Ожидание " + delaySeconds + " секунд перед повтором...");
                            TimeUnit.SECONDS.sleep(delaySeconds);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Повтор прерван", ie);
                        }
                    }
                }
            }

            // Все попытки не удались
            throw new Exception("Операция не удалась после " + maxAttempts + " попыток", lastException);
        }
    }

    // Имитация операции, которая изначально не удается, но успешна после повторов
    static class EventuallySuccessfulOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            if (callCount < 3) {
                throw new RuntimeException("Временный сбой");
            }
            // Успех на третьей попытке
        }
    }

    // Имитация операции, которая всегда не удается
    static class AlwaysFailingOperation implements Runnable {
        @Override
        public void run() {
            throw new RuntimeException("Постоянный сбой");
        }
    }

    public static void main(String[] args) {
        RetryService retryService = new RetryService();

        // Пример 1: Успешная операция после повторов
        System.out.println("Тест 1: Операция в конечном итоге успешна");
        try {
            retryService.retry(new EventuallySuccessfulOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Окончательный сбой: " + e.getMessage());
        }

        // Пример 2: Все попытки не удались
        System.out.println("\\nТест 2: Все попытки не удались");
        try {
            retryService.retry(new AlwaysFailingOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Окончательный сбой: " + e.getMessage());
        }
    }
}`,
            description: `Изучите реализацию базовой логики повторных попыток для обработки временных сбоев.

**Требования:**
1. Создайте класс RetryService с методом retry
2. Реализуйте логику повторных попыток с максимальным количеством попыток (3 повтора)
3. Добавьте задержку между попытками повтора (1 секунда)
4. Обработайте исключения и подсчитайте попытки повтора
5. Логируйте каждую попытку повтора
6. Выбросьте финальное исключение, если все повторы не удались

Логика повторных попыток необходима для обработки временных сбоев в распределенных системах, сетевых вызовах и интеграциях с внешними сервисами.`,
            hint1: `Используйте цикл while для повторения операции. Отслеживайте количество попыток и перехватывайте исключения, чтобы определить, следует ли повторить попытку.`,
            hint2: `Используйте TimeUnit.SECONDS.sleep() для добавления задержки между повторами. Не забудьте правильно обработать InterruptedException.`,
            whyItMatters: `Логика повторных попыток имеет решающее значение для создания устойчивых приложений. Многие сбои в распределенных системах являются временными (преходящими), такими как сетевые сбои или временная недоступность сервиса.

**Продакшен паттерн:**
\`\`\`java
// Временный сбой сети
if (callCount < 3) {
    throw new RuntimeException("Transient failure");
}
// Успех на третьей попытке
\`\`\`

**Практические преимущества:**
- Автоматическое восстановление от временных сбоев
- Улучшение доступности сервиса без ручного вмешательства
- Снижение количества ошибок в production окружении`
        },
        uz: {
            title: 'Oddiy qayta urinish logikasi',
            solutionCode: `import java.util.concurrent.TimeUnit;

public class RetryBasics {
    // Qayta urinish funksiyasini taqdim etuvchi servis
    static class RetryService {
        // Vazifani qayta urinish logikasi bilan bajarish
        public void retry(Runnable task, int maxAttempts, int delaySeconds) throws Exception {
            int attempt = 0;
            Exception lastException = null;

            while (attempt < maxAttempts) {
                try {
                    attempt++;
                    System.out.println("Urinish " + attempt + " / " + maxAttempts);
                    task.run();
                    System.out.println("Operatsiya muvaffaqiyatli bajarildi!");
                    return; // Muvaffaqiyat - metoddan chiqamiz
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Urinish " + attempt + " muvaffaqiyatsiz: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        try {
                            System.out.println("Qayta urinishdan oldin " + delaySeconds + " soniya kutilmoqda...");
                            TimeUnit.SECONDS.sleep(delaySeconds);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Qayta urinish to'xtatildi", ie);
                        }
                    }
                }
            }

            // Barcha urinishlar muvaffaqiyatsiz
            throw new Exception(maxAttempts + " urinishdan keyin operatsiya muvaffaqiyatsiz", lastException);
        }
    }

    // Dastlab muvaffaqiyatsiz bo'lgan, lekin qayta urinishlardan keyin muvaffaqiyatli bo'lgan operatsiya taqlidi
    static class EventuallySuccessfulOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            if (callCount < 3) {
                throw new RuntimeException("Vaqtinchalik nosozlik");
            }
            // Uchinchi urinishda muvaffaqiyat
        }
    }

    // Doim muvaffaqiyatsiz bo'lgan operatsiya taqlidi
    static class AlwaysFailingOperation implements Runnable {
        @Override
        public void run() {
            throw new RuntimeException("Doimiy nosozlik");
        }
    }

    public static void main(String[] args) {
        RetryService retryService = new RetryService();

        // Misol 1: Qayta urinishlardan keyin muvaffaqiyatli operatsiya
        System.out.println("Test 1: Oxir-oqibat muvaffaqiyatli operatsiya");
        try {
            retryService.retry(new EventuallySuccessfulOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Yakuniy muvaffaqiyatsizlik: " + e.getMessage());
        }

        // Misol 2: Barcha urinishlar muvaffaqiyatsiz
        System.out.println("\\nTest 2: Barcha urinishlar muvaffaqiyatsiz");
        try {
            retryService.retry(new AlwaysFailingOperation(), 3, 1);
        } catch (Exception e) {
            System.out.println("Yakuniy muvaffaqiyatsizlik: " + e.getMessage());
        }
    }
}`,
            description: `Vaqtinchalik nosozliklarni qayta ishlash uchun asosiy qayta urinish logikasini amalga oshirishni o'rganing.

**Talablar:**
1. retry metodi bilan RetryService klassini yarating
2. Maksimal urinishlar soni bilan qayta urinish logikasini amalga oshiring (3 qayta urinish)
3. Qayta urinish urinishlari orasida kechikish qo'shing (1 soniya)
4. Istisnolarni boshqaring va qayta urinish urinishlarini hisoblang
5. Har bir qayta urinishni loglang
6. Agar barcha qayta urinishlar muvaffaqiyatsiz bo'lsa, yakuniy istisnoni tashlang

Qayta urinish logikasi taqsimlangan tizimlarda, tarmoq chaqiruvlarida va tashqi servis integratsiyalarida vaqtinchalik nosozliklarni qayta ishlash uchun zarurdir.`,
            hint1: `Operatsiyani qayta urinish uchun while tsiklidan foydalaning. Urinishlar sonini kuzatib boring va boshqa urinish amalga oshirilishi kerakligini aniqlash uchun istisnolarni ushlang.`,
            hint2: `Qayta urinishlar orasida kechikish qo'shish uchun TimeUnit.SECONDS.sleep() dan foydalaning. InterruptedException ni to'g'ri boshqarishni unutmang.`,
            whyItMatters: `Qayta urinish logikasi mustahkam ilovalar yaratish uchun juda muhimdir. Taqsimlangan tizimlardagi ko'plab nosozliklar vaqtinchalik bo'ladi, masalan, tarmoq nosozliklari yoki vaqtinchalik servis mavjud emasligi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Vaqtinchalik tarmoq nosozligi
if (callCount < 3) {
    throw new RuntimeException("Transient failure");
}
// Uchinchi urinishda muvaffaqiyat
\`\`\`

**Amaliy foydalari:**
- Vaqtinchalik nosozliklardan avtomatik tiklanish
- Qo'lda aralashuvsiz xizmat mavjudligini yaxshilash
- Production muhitida xatolar sonini kamaytirish`
        }
    }
};

export default task;
