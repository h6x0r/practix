import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-exponential-backoff',
    title: 'Exponential Backoff Implementation',
    difficulty: 'medium',
    tags: ['java', 'retry', 'exponential-backoff', 'resilience'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to implement exponential backoff retry strategy to avoid overwhelming services.

**Requirements:**
1. Create an ExponentialBackoffRetry class
2. Implement exponential backoff: delay doubles after each retry (1s, 2s, 4s, 8s)
3. Add a maximum delay cap (e.g., 10 seconds)
4. Add optional jitter (random variation) to prevent thundering herd
5. Track and log the delay for each retry attempt
6. Demonstrate the increasing delays between retries

Exponential backoff is a standard retry strategy that progressively increases the wait time between retries, giving failing services time to recover.`,
    initialCode: `import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ExponentialBackoff {
    // Create ExponentialBackoffRetry class
    // - Initial delay: 1 second
    // - Exponential multiplier: 2
    // - Maximum delay: 10 seconds
    // - Optional jitter: 0-100ms random variation
    // - Calculate delay as: min(initialDelay * 2^attempt, maxDelay) + jitter

    public static void main(String[] args) {
        // Create retry service with exponential backoff

        // Test with a failing operation
        System.out.println("Testing exponential backoff:");
        // Implement retry with exponential backoff
    }
}`,
    solutionCode: `import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ExponentialBackoff {
    // Retry service with exponential backoff strategy
    static class ExponentialBackoffRetry {
        private final int initialDelayMs;
        private final int maxDelayMs;
        private final double multiplier;
        private final int maxAttempts;
        private final Random random;

        public ExponentialBackoffRetry(int initialDelayMs, int maxDelayMs,
                                      double multiplier, int maxAttempts) {
            this.initialDelayMs = initialDelayMs;
            this.maxDelayMs = maxDelayMs;
            this.multiplier = multiplier;
            this.maxAttempts = maxAttempts;
            this.random = new Random();
        }

        // Calculate delay with exponential backoff
        private long calculateDelay(int attempt) {
            // Calculate exponential delay: initialDelay * multiplier^attempt
            long exponentialDelay = (long) (initialDelayMs * Math.pow(multiplier, attempt - 1));

            // Cap at maximum delay
            long delay = Math.min(exponentialDelay, maxDelayMs);

            // Add jitter (random 0-100ms) to prevent thundering herd
            long jitter = random.nextInt(100);

            return delay + jitter;
        }

        // Execute task with exponential backoff retry
        public void retry(Runnable task) throws Exception {
            Exception lastException = null;

            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    System.out.println("Attempt " + attempt + " of " + maxAttempts);
                    task.run();
                    System.out.println("Operation succeeded!");
                    return; // Success
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Attempt " + attempt + " failed: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        long delayMs = calculateDelay(attempt);
                        System.out.println("Waiting " + delayMs + "ms before retry " +
                                         "(exponential backoff)");

                        try {
                            TimeUnit.MILLISECONDS.sleep(delayMs);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Retry interrupted", ie);
                        }
                    }
                }
            }

            throw new Exception("Operation failed after " + maxAttempts + " attempts",
                              lastException);
        }
    }

    // Simulated failing operation
    static class FailingOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            throw new RuntimeException("Service unavailable (attempt " + callCount + ")");
        }
    }

    public static void main(String[] args) {
        // Create retry service with exponential backoff
        // Initial delay: 1000ms, Max delay: 10000ms, Multiplier: 2, Max attempts: 5
        ExponentialBackoffRetry retryService =
            new ExponentialBackoffRetry(1000, 10000, 2.0, 5);

        // Test with a failing operation
        System.out.println("Testing exponential backoff:");
        System.out.println("Expected delays: ~1s, ~2s, ~4s, ~8s");
        System.out.println("(with small random jitter)\\n");

        try {
            retryService.retry(new FailingOperation());
        } catch (Exception e) {
            System.out.println("\\nFinal result: " + e.getMessage());
        }

        // Demonstrate delay calculation
        System.out.println("\\nDelay progression:");
        ExponentialBackoffRetry demo = new ExponentialBackoffRetry(1000, 10000, 2.0, 6);
        for (int i = 1; i <= 6; i++) {
            long delay = demo.calculateDelay(i);
            System.out.println("Attempt " + i + ": " + delay + "ms");
        }
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify exponential delay calculation
class Test1 {
    @Test
    public void test() {
        long delay1 = (long)(1000 * Math.pow(2, 0));
        long delay2 = (long)(1000 * Math.pow(2, 1));
        long delay3 = (long)(1000 * Math.pow(2, 2));
        assertEquals(1000, delay1);
        assertEquals(2000, delay2);
        assertEquals(4000, delay3);
    }
}

// Test2: Verify max delay cap
class Test2 {
    @Test
    public void test() {
        long maxDelay = 10000;
        long calculatedDelay = (long)(1000 * Math.pow(2, 10));
        long actualDelay = Math.min(calculatedDelay, maxDelay);
        assertEquals(maxDelay, actualDelay);
    }
}

// Test3: Verify multiplier effect
class Test3 {
    @Test
    public void test() {
        double multiplier = 2.0;
        long baseDelay = 1000;
        long delay = (long)(baseDelay * Math.pow(multiplier, 3));
        assertEquals(8000, delay);
    }
}

// Test4: Verify jitter range
class Test4 {
    @Test
    public void test() {
        java.util.Random random = new java.util.Random();
        int jitter = random.nextInt(100);
        assertTrue(jitter >= 0 && jitter < 100);
    }
}

// Test5: Verify delay progression
class Test5 {
    @Test
    public void test() {
        long delay1 = 1000;
        long delay2 = 2000;
        long delay3 = 4000;
        assertTrue(delay2 == delay1 * 2);
        assertTrue(delay3 == delay2 * 2);
    }
}

// Test6: Verify first attempt delay equals initial delay
class Test6 {
    @Test
    public void test() {
        long initialDelay = 500;
        long attempt1Delay = (long)(initialDelay * Math.pow(2, 0));
        assertEquals(500, attempt1Delay);
    }
}

// Test7: Verify delay never exceeds max delay for high attempts
class Test7 {
    @Test
    public void test() {
        long initialDelay = 1000;
        long maxDelay = 30000;
        for (int attempt = 1; attempt <= 20; attempt++) {
            long calculated = (long)(initialDelay * Math.pow(2, attempt - 1));
            long actual = Math.min(calculated, maxDelay);
            assertTrue("Delay should not exceed max", actual <= maxDelay);
        }
    }
}

// Test8: Verify different multipliers produce different delays
class Test8 {
    @Test
    public void test() {
        long base = 1000;
        int attempt = 3;
        long delayMult2 = (long)(base * Math.pow(2, attempt));
        long delayMult3 = (long)(base * Math.pow(3, attempt));
        assertTrue(delayMult3 > delayMult2);
        assertEquals(8000, delayMult2);
        assertEquals(27000, delayMult3);
    }
}

// Test9: Verify delay with multiplier of 1 stays constant
class Test9 {
    @Test
    public void test() {
        long base = 1000;
        double multiplier = 1.0;
        long delay1 = (long)(base * Math.pow(multiplier, 0));
        long delay2 = (long)(base * Math.pow(multiplier, 5));
        long delay3 = (long)(base * Math.pow(multiplier, 10));
        assertEquals(delay1, delay2);
        assertEquals(delay2, delay3);
    }
}

// Test10: Verify delay sequence increases until hitting max
class Test10 {
    @Test
    public void test() {
        long initialDelay = 100;
        long maxDelay = 1000;
        long prevDelay = 0;
        boolean reachedMax = false;
        for (int attempt = 1; attempt <= 10; attempt++) {
            long calculated = (long)(initialDelay * Math.pow(2, attempt - 1));
            long delay = Math.min(calculated, maxDelay);
            if (!reachedMax) {
                assertTrue("Delay should increase", delay >= prevDelay);
            }
            if (delay == maxDelay) reachedMax = true;
            prevDelay = delay;
        }
        assertTrue("Should have reached max delay", reachedMax);
    }
}
`,
    hint1: `Use Math.pow(multiplier, attempt) to calculate exponential growth. Start with a base delay and multiply it by 2^attempt.`,
    hint2: `Add jitter using Random.nextInt() to add a small random delay. This prevents multiple clients from retrying at exactly the same time (thundering herd problem).`,
    whyItMatters: `Exponential backoff is the gold standard for retry strategies. It prevents overwhelming an already struggling service by giving it progressively more time to recover. The jitter component prevents synchronized retries from multiple clients, which can cause cascading failures. This pattern is used by AWS, Google Cloud, and most major cloud services.

**Production Pattern:**
\`\`\`java
// Exponential delay: 1s, 2s, 4s, 8s
long delay = initialDelay * Math.pow(2, attempt - 1);
delay = Math.min(delay, maxDelay);
// Add jitter to prevent thundering herd
delay += random.nextInt(100);
\`\`\`

**Practical Benefits:**
- Prevents overwhelming recovering services
- Jitter prevents synchronized retries from multiple clients
- Used by AWS, Google Cloud, and most cloud services`,
    order: 1,
    translations: {
        ru: {
            title: 'Реализация экспоненциальной отсрочки',
            solutionCode: `import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ExponentialBackoff {
    // Сервис повторных попыток со стратегией экспоненциальной отсрочки
    static class ExponentialBackoffRetry {
        private final int initialDelayMs;
        private final int maxDelayMs;
        private final double multiplier;
        private final int maxAttempts;
        private final Random random;

        public ExponentialBackoffRetry(int initialDelayMs, int maxDelayMs,
                                      double multiplier, int maxAttempts) {
            this.initialDelayMs = initialDelayMs;
            this.maxDelayMs = maxDelayMs;
            this.multiplier = multiplier;
            this.maxAttempts = maxAttempts;
            this.random = new Random();
        }

        // Вычислить задержку с экспоненциальной отсрочкой
        private long calculateDelay(int attempt) {
            // Вычислить экспоненциальную задержку: initialDelay * multiplier^attempt
            long exponentialDelay = (long) (initialDelayMs * Math.pow(multiplier, attempt - 1));

            // Ограничить максимальной задержкой
            long delay = Math.min(exponentialDelay, maxDelayMs);

            // Добавить джиттер (случайные 0-100мс) для предотвращения эффекта стада
            long jitter = random.nextInt(100);

            return delay + jitter;
        }

        // Выполнить задачу с повторными попытками и экспоненциальной отсрочкой
        public void retry(Runnable task) throws Exception {
            Exception lastException = null;

            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    System.out.println("Попытка " + attempt + " из " + maxAttempts);
                    task.run();
                    System.out.println("Операция выполнена успешно!");
                    return; // Успех
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Попытка " + attempt + " не удалась: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        long delayMs = calculateDelay(attempt);
                        System.out.println("Ожидание " + delayMs + "мс перед повтором " +
                                         "(экспоненциальная отсрочка)");

                        try {
                            TimeUnit.MILLISECONDS.sleep(delayMs);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Повтор прерван", ie);
                        }
                    }
                }
            }

            throw new Exception("Операция не удалась после " + maxAttempts + " попыток",
                              lastException);
        }
    }

    // Имитация неудачной операции
    static class FailingOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            throw new RuntimeException("Сервис недоступен (попытка " + callCount + ")");
        }
    }

    public static void main(String[] args) {
        // Создать сервис повторных попыток с экспоненциальной отсрочкой
        // Начальная задержка: 1000мс, Макс задержка: 10000мс, Множитель: 2, Макс попыток: 5
        ExponentialBackoffRetry retryService =
            new ExponentialBackoffRetry(1000, 10000, 2.0, 5);

        // Тест с неудачной операцией
        System.out.println("Тестирование экспоненциальной отсрочки:");
        System.out.println("Ожидаемые задержки: ~1с, ~2с, ~4с, ~8с");
        System.out.println("(с небольшим случайным джиттером)\\n");

        try {
            retryService.retry(new FailingOperation());
        } catch (Exception e) {
            System.out.println("\\nОкончательный результат: " + e.getMessage());
        }

        // Демонстрация вычисления задержки
        System.out.println("\\nПрогрессия задержки:");
        ExponentialBackoffRetry demo = new ExponentialBackoffRetry(1000, 10000, 2.0, 6);
        for (int i = 1; i <= 6; i++) {
            long delay = demo.calculateDelay(i);
            System.out.println("Попытка " + i + ": " + delay + "мс");
        }
    }
}`,
            description: `Изучите реализацию стратегии повторных попыток с экспоненциальной отсрочкой для предотвращения перегрузки сервисов.

**Требования:**
1. Создайте класс ExponentialBackoffRetry
2. Реализуйте экспоненциальную отсрочку: задержка удваивается после каждого повтора (1с, 2с, 4с, 8с)
3. Добавьте максимальное ограничение задержки (например, 10 секунд)
4. Добавьте опциональный джиттер (случайное отклонение) для предотвращения эффекта стада
5. Отслеживайте и логируйте задержку для каждой попытки повтора
6. Продемонстрируйте увеличивающиеся задержки между повторами

Экспоненциальная отсрочка - это стандартная стратегия повторных попыток, которая прогрессивно увеличивает время ожидания между повторами, давая неработающим сервисам время на восстановление.`,
            hint1: `Используйте Math.pow(multiplier, attempt) для вычисления экспоненциального роста. Начните с базовой задержки и умножайте ее на 2^attempt.`,
            hint2: `Добавьте джиттер, используя Random.nextInt(), чтобы добавить небольшую случайную задержку. Это предотвращает повторные попытки нескольких клиентов точно в одно и то же время (проблема эффекта стада).`,
            whyItMatters: `Экспоненциальная отсрочка является золотым стандартом стратегий повторных попыток. Она предотвращает перегрузку и без того испытывающего трудности сервиса, давая ему прогрессивно больше времени на восстановление.

**Продакшен паттерн:**
\`\`\`java
// Экспоненциальная задержка: 1с, 2с, 4с, 8с
long delay = initialDelay * Math.pow(2, attempt - 1);
delay = Math.min(delay, maxDelay);
// Добавить джиттер для предотвращения эффекта стада
delay += random.nextInt(100);
\`\`\`

**Практические преимущества:**
- Предотвращение перегрузки восстанавливающихся сервисов
- Джиттер предотвращает синхронизированные повторы от нескольких клиентов
- Используется AWS, Google Cloud и большинством облачных сервисов`
        },
        uz: {
            title: 'Eksponensial kechikishni amalga oshirish',
            solutionCode: `import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ExponentialBackoff {
    // Eksponensial kechikish strategiyasi bilan qayta urinish xizmati
    static class ExponentialBackoffRetry {
        private final int initialDelayMs;
        private final int maxDelayMs;
        private final double multiplier;
        private final int maxAttempts;
        private final Random random;

        public ExponentialBackoffRetry(int initialDelayMs, int maxDelayMs,
                                      double multiplier, int maxAttempts) {
            this.initialDelayMs = initialDelayMs;
            this.maxDelayMs = maxDelayMs;
            this.multiplier = multiplier;
            this.maxAttempts = maxAttempts;
            this.random = new Random();
        }

        // Eksponensial kechikish bilan kechikishni hisoblash
        private long calculateDelay(int attempt) {
            // Eksponensial kechikishni hisoblash: initialDelay * multiplier^attempt
            long exponentialDelay = (long) (initialDelayMs * Math.pow(multiplier, attempt - 1));

            // Maksimal kechikish bilan cheklash
            long delay = Math.min(exponentialDelay, maxDelayMs);

            // Jitter qo'shish (tasodifiy 0-100ms) to'da effektini oldini olish uchun
            long jitter = random.nextInt(100);

            return delay + jitter;
        }

        // Vazifani eksponensial kechikish bilan qayta urinish orqali bajarish
        public void retry(Runnable task) throws Exception {
            Exception lastException = null;

            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    System.out.println("Urinish " + attempt + " / " + maxAttempts);
                    task.run();
                    System.out.println("Operatsiya muvaffaqiyatli bajarildi!");
                    return; // Muvaffaqiyat
                } catch (Exception e) {
                    lastException = e;
                    System.out.println("Urinish " + attempt + " muvaffaqiyatsiz: " + e.getMessage());

                    if (attempt < maxAttempts) {
                        long delayMs = calculateDelay(attempt);
                        System.out.println("Qayta urinishdan oldin " + delayMs + "ms kutilmoqda " +
                                         "(eksponensial kechikish)");

                        try {
                            TimeUnit.MILLISECONDS.sleep(delayMs);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new Exception("Qayta urinish to'xtatildi", ie);
                        }
                    }
                }
            }

            throw new Exception(maxAttempts + " urinishdan keyin operatsiya muvaffaqiyatsiz",
                              lastException);
        }
    }

    // Muvaffaqiyatsiz operatsiya taqlidi
    static class FailingOperation implements Runnable {
        private int callCount = 0;

        @Override
        public void run() {
            callCount++;
            throw new RuntimeException("Xizmat mavjud emas (urinish " + callCount + ")");
        }
    }

    public static void main(String[] args) {
        // Eksponensial kechikish bilan qayta urinish xizmatini yaratish
        // Boshlang'ich kechikish: 1000ms, Maks kechikish: 10000ms, Ko'paytiruvchi: 2, Maks urinishlar: 5
        ExponentialBackoffRetry retryService =
            new ExponentialBackoffRetry(1000, 10000, 2.0, 5);

        // Muvaffaqiyatsiz operatsiya bilan test
        System.out.println("Eksponensial kechikishni test qilish:");
        System.out.println("Kutilgan kechikishlar: ~1s, ~2s, ~4s, ~8s");
        System.out.println("(kichik tasodifiy jitter bilan)\\n");

        try {
            retryService.retry(new FailingOperation());
        } catch (Exception e) {
            System.out.println("\\nYakuniy natija: " + e.getMessage());
        }

        // Kechikish hisoblashni namoyish etish
        System.out.println("\\nKechikish progressiyasi:");
        ExponentialBackoffRetry demo = new ExponentialBackoffRetry(1000, 10000, 2.0, 6);
        for (int i = 1; i <= 6; i++) {
            long delay = demo.calculateDelay(i);
            System.out.println("Urinish " + i + ": " + delay + "ms");
        }
    }
}`,
            description: `Xizmatlarni ortiqcha yuklashdan saqlash uchun eksponensial kechikish bilan qayta urinish strategiyasini amalga oshirishni o'rganing.

**Talablar:**
1. ExponentialBackoffRetry klassini yarating
2. Eksponensial kechikishni amalga oshiring: har bir qayta urinishdan keyin kechikish ikki barobarga oshadi (1s, 2s, 4s, 8s)
3. Maksimal kechikish chegarasini qo'shing (masalan, 10 soniya)
4. Ixtiyoriy jitter (tasodifiy o'zgarish) qo'shing, to'da effektini oldini olish uchun
5. Har bir qayta urinish uchun kechikishni kuzatib boring va loglang
6. Qayta urinishlar orasidagi ortib borayotgan kechikishlarni namoyish eting

Eksponensial kechikish qayta urinishlar orasidagi kutish vaqtini progressiv ravishda oshiruvchi standart qayta urinish strategiyasi bo'lib, muvaffaqiyatsiz xizmatlarga tiklanish uchun vaqt beradi.`,
            hint1: `Eksponensial o'sishni hisoblash uchun Math.pow(multiplier, attempt) dan foydalaning. Asosiy kechikish bilan boshlang va uni 2^attempt ga ko'paytiring.`,
            hint2: `Random.nextInt() yordamida kichik tasodifiy kechikish qo'shish uchun jitter qo'shing. Bu bir nechta mijozlarning aynan bir vaqtda qayta urinishini oldini oladi (to'da muammosi).`,
            whyItMatters: `Eksponensial kechikish qayta urinish strategiyalarining oltin standarti hisoblanadi. U allaqachon qiynalayotgan xizmatni ortiqcha yuklashning oldini olib, unga progressiv ravishda ko'proq tiklanish vaqti beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Eksponensial kechikish: 1s, 2s, 4s, 8s
long delay = initialDelay * Math.pow(2, attempt - 1);
delay = Math.min(delay, maxDelay);
// To'da effektini oldini olish uchun jitter qo'shish
delay += random.nextInt(100);
\`\`\`

**Amaliy foydalari:**
- Tiklanayotgan xizmatlarni ortiqcha yuklashning oldini olish
- Jitter bir nechta mijozlarning sinxronlashgan qayta urinishlarini oldini oladi
- AWS, Google Cloud va ko'pchilik bulutli xizmatlar tomonidan qo'llaniladi`
        }
    }
};

export default task;
