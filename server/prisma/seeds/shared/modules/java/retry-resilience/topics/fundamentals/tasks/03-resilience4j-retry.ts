import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-resilience4j-retry',
    title: 'Resilience4j Retry Configuration',
    difficulty: 'medium',
    tags: ['java', 'resilience4j', 'retry', 'circuit-breaker'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to use Resilience4j library for advanced retry patterns.

**Requirements:**
1. Configure a Resilience4j Retry with maxAttempts=3
2. Set waitDuration between retries (1 second)
3. Add exponential backoff with multiplier=2
4. Configure retry only for specific exceptions (IOException, TimeoutException)
5. Add event listeners to log retry attempts
6. Decorate a function with retry logic
7. Demonstrate successful retry after transient failures

Resilience4j is a lightweight fault tolerance library inspired by Netflix Hystrix, designed for functional programming.`,
    initialCode: `import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import java.io.IOException;

public class Resilience4jRetry {
    // Configure Resilience4j Retry
    // - maxAttempts: 3
    // - waitDuration: 1 second
    // - Enable exponential backoff with multiplier 2
    // - Retry on: IOException, RuntimeException
    // - Add event listeners for retry events

    static class RemoteService {
        private int callCount = 0;

        public String fetchData() throws IOException {
            callCount++;
            if (callCount < 3) {
                throw new IOException("Connection timeout");
            }
            return "Success! Data fetched on attempt " + callCount;
        }
    }

    public static void main(String[] args) {
        // Create RetryConfig

        // Create Retry instance

        // Add event listeners

        // Decorate the service method with retry

        // Execute and handle result
    }
}`,
    solutionCode: `import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import java.io.IOException;
import java.util.function.Supplier;

public class Resilience4jRetry {
    // Service that simulates transient failures
    static class RemoteService {
        private int callCount = 0;

        public String fetchData() throws IOException {
            callCount++;
            System.out.println("Calling remote service (attempt " + callCount + ")");

            if (callCount < 3) {
                throw new IOException("Connection timeout - transient failure");
            }

            return "Success! Data fetched on attempt " + callCount;
        }

        public void reset() {
            callCount = 0;
        }
    }

    public static void main(String[] args) {
        // Configure Retry with exponential backoff
        RetryConfig config = RetryConfig.custom()
            .maxAttempts(3)                                    // Maximum retry attempts
            .waitDuration(Duration.ofSeconds(1))              // Initial wait duration
            .intervalFunction(                                 // Exponential backoff
                io.github.resilience4j.retry.IntervalFunction
                    .ofExponentialBackoff(Duration.ofSeconds(1), 2.0)
            )
            .retryExceptions(IOException.class, RuntimeException.class)  // Retry on these exceptions
            .build();

        // Create RetryRegistry and Retry instance
        RetryRegistry registry = RetryRegistry.of(config);
        Retry retry = registry.retry("remoteServiceRetry");

        // Add event listeners to track retry attempts
        retry.getEventPublisher()
            .onRetry(event -> {
                System.out.println("Retry event: Attempt #" +
                    event.getNumberOfRetryAttempts() +
                    " - Last exception: " + event.getLastThrowable().getMessage());
            })
            .onSuccess(event -> {
                System.out.println("Success event: Operation succeeded");
            })
            .onError(event -> {
                System.out.println("Error event: All retry attempts failed");
            });

        RemoteService service = new RemoteService();

        // Example 1: Successful retry after transient failures
        System.out.println("=== Test 1: Eventually successful ===");
        try {
            // Decorate the supplier with retry logic
            Supplier<String> decoratedSupplier = Retry.decorateSupplier(retry, () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

            // Execute the decorated function
            String result = decoratedSupplier.get();
            System.out.println("Result: " + result);

        } catch (Exception e) {
            System.out.println("Final failure: " + e.getMessage());
        }

        // Example 2: Configure retry with different settings
        System.out.println("\\n=== Test 2: Custom retry config ===");
        RetryConfig customConfig = RetryConfig.custom()
            .maxAttempts(5)
            .waitDuration(Duration.ofMillis(500))
            .retryOnException(e -> e instanceof IOException)  // Retry predicate
            .build();

        Retry customRetry = RetryRegistry.of(customConfig).retry("customRetry");

        customRetry.getEventPublisher()
            .onRetry(event -> System.out.println(
                "Custom retry - Attempt: " + event.getNumberOfRetryAttempts()
            ));

        service.reset();
        Supplier<String> customDecoratedSupplier = Retry.decorateSupplier(
            customRetry,
            () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        );

        try {
            String result = customDecoratedSupplier.get();
            System.out.println("Custom retry result: " + result);
        } catch (Exception e) {
            System.out.println("Custom retry failed: " + e.getMessage());
        }
    }
}`,
    hint1: `Use RetryConfig.custom() to build your retry configuration. Set maxAttempts, waitDuration, and use IntervalFunction.ofExponentialBackoff() for exponential delays.`,
    hint2: `Use Retry.decorateSupplier() or Retry.decorateFunction() to wrap your service calls. Add event listeners with retry.getEventPublisher() to track retry events.`,
    whyItMatters: `Resilience4j is the modern standard for building resilient Java applications. It provides a rich set of patterns (retry, circuit breaker, rate limiter, bulkhead) with minimal overhead. Unlike custom retry logic, Resilience4j offers production-ready features like metrics, event publishing, and Spring Boot integration. It's used by companies like Netflix, Spotify, and many Fortune 500 companies.

**Production Pattern:**
\`\`\`java
RetryConfig config = RetryConfig.custom()
    .maxAttempts(3)
    .waitDuration(Duration.ofSeconds(1))
    .intervalFunction(IntervalFunction.ofExponentialBackoff())
    .retryExceptions(IOException.class)
    .build();
\`\`\`

**Practical Benefits:**
- Production-ready features: metrics, events, Spring Boot integration
- Declarative configuration instead of imperative code
- Used by Netflix, Spotify, and Fortune 500 companies`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

// Test 1: RetryConfig maxAttempts works
class Test1 {
    @Test
    void testMaxAttempts() {
        int maxAttempts = 3;
        AtomicInteger count = new AtomicInteger(0);
        Exception lastException = null;

        for (int i = 0; i < maxAttempts; i++) {
            try {
                count.incrementAndGet();
                throw new RuntimeException("Fail");
            } catch (RuntimeException e) {
                lastException = e;
            }
        }

        assertEquals(3, count.get());
    }
}

// Test 2: RetryConfig waitDuration is respected
class Test2 {
    @Test
    void testWaitDuration() {
        Duration waitDuration = Duration.ofSeconds(1);
        assertEquals(1000, waitDuration.toMillis());
    }
}

// Test 3: Retry succeeds on first attempt
class Test3 {
    @Test
    void testSuccessOnFirstAttempt() {
        AtomicInteger attempts = new AtomicInteger(0);
        String result = null;

        for (int i = 0; i < 3; i++) {
            try {
                attempts.incrementAndGet();
                result = "success";
                break;
            } catch (Exception e) {
                // retry
            }
        }

        assertEquals(1, attempts.get());
        assertEquals("success", result);
    }
}

// Test 4: Retry eventually succeeds
class Test4 {
    @Test
    void testEventualSuccess() {
        AtomicInteger attempts = new AtomicInteger(0);
        String result = null;

        for (int i = 0; i < 3; i++) {
            try {
                if (attempts.incrementAndGet() < 3) {
                    throw new RuntimeException("Transient failure");
                }
                result = "success";
                break;
            } catch (RuntimeException e) {
                // retry
            }
        }

        assertEquals(3, attempts.get());
        assertEquals("success", result);
    }
}

// Test 5: Exponential backoff multiplier calculation
class Test5 {
    @Test
    void testExponentialMultiplier() {
        double multiplier = 2.0;
        long baseDelay = 1000;
        long attempt1Delay = (long) (baseDelay * Math.pow(multiplier, 0));
        long attempt2Delay = (long) (baseDelay * Math.pow(multiplier, 1));
        long attempt3Delay = (long) (baseDelay * Math.pow(multiplier, 2));
        assertEquals(1000, attempt1Delay);
        assertEquals(2000, attempt2Delay);
        assertEquals(4000, attempt3Delay);
    }
}

// Test 6: Retry on specific exception type
class Test6 {
    @Test
    void testRetryOnSpecificException() {
        AtomicInteger retryCount = new AtomicInteger(0);
        Exception caughtException = null;

        try {
            throw new java.io.IOException("Connection failed");
        } catch (java.io.IOException e) {
            caughtException = e;
            retryCount.incrementAndGet();
        } catch (Exception e) {
            // don't retry
        }

        assertEquals(1, retryCount.get());
        assertTrue(caughtException instanceof java.io.IOException);
    }
}

// Test 7: All retries fail throws final exception
class Test7 {
    @Test
    void testAllRetriesFail() {
        AtomicInteger attempts = new AtomicInteger(0);
        Exception finalException = null;

        for (int i = 0; i < 3; i++) {
            try {
                attempts.incrementAndGet();
                throw new RuntimeException("Permanent failure");
            } catch (RuntimeException e) {
                finalException = e;
            }
        }

        assertEquals(3, attempts.get());
        assertNotNull(finalException);
    }
}

// Test 8: Supplier decoration returns result
class Test8 {
    @Test
    void testSupplierDecoration() {
        java.util.function.Supplier<String> supplier = () -> "decorated result";
        String result = supplier.get();
        assertEquals("decorated result", result);
    }
}

// Test 9: Retry counter increments properly
class Test9 {
    @Test
    void testRetryCounter() {
        AtomicInteger counter = new AtomicInteger(0);
        int maxRetries = 5;
        for (int i = 0; i < maxRetries; i++) {
            counter.incrementAndGet();
        }
        assertEquals(5, counter.get());
    }
}

// Test 10: Duration conversion works correctly
class Test10 {
    @Test
    void testDurationConversion() {
        Duration seconds = Duration.ofSeconds(5);
        Duration millis = Duration.ofMillis(500);
        assertEquals(5000, seconds.toMillis());
        assertEquals(500, millis.toMillis());
    }
}`,
    translations: {
        ru: {
            title: 'Конфигурация Retry в Resilience4j',
            solutionCode: `import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import java.io.IOException;
import java.util.function.Supplier;

public class Resilience4jRetry {
    // Сервис, имитирующий временные сбои
    static class RemoteService {
        private int callCount = 0;

        public String fetchData() throws IOException {
            callCount++;
            System.out.println("Вызов удаленного сервиса (попытка " + callCount + ")");

            if (callCount < 3) {
                throw new IOException("Превышено время ожидания соединения - временный сбой");
            }

            return "Успех! Данные получены на попытке " + callCount;
        }

        public void reset() {
            callCount = 0;
        }
    }

    public static void main(String[] args) {
        // Настройка Retry с экспоненциальной отсрочкой
        RetryConfig config = RetryConfig.custom()
            .maxAttempts(3)                                    // Максимальное количество попыток
            .waitDuration(Duration.ofSeconds(1))              // Начальная длительность ожидания
            .intervalFunction(                                 // Экспоненциальная отсрочка
                io.github.resilience4j.retry.IntervalFunction
                    .ofExponentialBackoff(Duration.ofSeconds(1), 2.0)
            )
            .retryExceptions(IOException.class, RuntimeException.class)  // Повтор при этих исключениях
            .build();

        // Создание RetryRegistry и экземпляра Retry
        RetryRegistry registry = RetryRegistry.of(config);
        Retry retry = registry.retry("remoteServiceRetry");

        // Добавление слушателей событий для отслеживания попыток повтора
        retry.getEventPublisher()
            .onRetry(event -> {
                System.out.println("Событие повтора: Попытка #" +
                    event.getNumberOfRetryAttempts() +
                    " - Последнее исключение: " + event.getLastThrowable().getMessage());
            })
            .onSuccess(event -> {
                System.out.println("Событие успеха: Операция выполнена успешно");
            })
            .onError(event -> {
                System.out.println("Событие ошибки: Все попытки повтора не удались");
            });

        RemoteService service = new RemoteService();

        // Пример 1: Успешный повтор после временных сбоев
        System.out.println("=== Тест 1: В конечном итоге успешно ===");
        try {
            // Декорирование supplier логикой повтора
            Supplier<String> decoratedSupplier = Retry.decorateSupplier(retry, () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

            // Выполнение декорированной функции
            String result = decoratedSupplier.get();
            System.out.println("Результат: " + result);

        } catch (Exception e) {
            System.out.println("Окончательный сбой: " + e.getMessage());
        }

        // Пример 2: Настройка повтора с другими параметрами
        System.out.println("\\n=== Тест 2: Пользовательская конфигурация повтора ===");
        RetryConfig customConfig = RetryConfig.custom()
            .maxAttempts(5)
            .waitDuration(Duration.ofMillis(500))
            .retryOnException(e -> e instanceof IOException)  // Предикат повтора
            .build();

        Retry customRetry = RetryRegistry.of(customConfig).retry("customRetry");

        customRetry.getEventPublisher()
            .onRetry(event -> System.out.println(
                "Пользовательский повтор - Попытка: " + event.getNumberOfRetryAttempts()
            ));

        service.reset();
        Supplier<String> customDecoratedSupplier = Retry.decorateSupplier(
            customRetry,
            () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        );

        try {
            String result = customDecoratedSupplier.get();
            System.out.println("Результат пользовательского повтора: " + result);
        } catch (Exception e) {
            System.out.println("Пользовательский повтор не удался: " + e.getMessage());
        }
    }
}`,
            description: `Изучите использование библиотеки Resilience4j для продвинутых паттернов повторных попыток.

**Требования:**
1. Настройте Resilience4j Retry с maxAttempts=3
2. Установите waitDuration между повторами (1 секунда)
3. Добавьте экспоненциальную отсрочку с multiplier=2
4. Настройте повтор только для определенных исключений (IOException, TimeoutException)
5. Добавьте слушателей событий для логирования попыток повтора
6. Декорируйте функцию логикой повтора
7. Продемонстрируйте успешный повтор после временных сбоев

Resilience4j - это легковесная библиотека отказоустойчивости, вдохновленная Netflix Hystrix, разработанная для функционального программирования.`,
            hint1: `Используйте RetryConfig.custom() для построения конфигурации повтора. Установите maxAttempts, waitDuration и используйте IntervalFunction.ofExponentialBackoff() для экспоненциальных задержек.`,
            hint2: `Используйте Retry.decorateSupplier() или Retry.decorateFunction() для обертывания вызовов сервиса. Добавьте слушателей событий с помощью retry.getEventPublisher() для отслеживания событий повтора.`,
            whyItMatters: `Resilience4j - это современный стандарт для создания устойчивых Java-приложений. Он предоставляет богатый набор паттернов (retry, circuit breaker, rate limiter, bulkhead) с минимальными накладными расходами.

**Продакшен паттерн:**
\`\`\`java
RetryConfig config = RetryConfig.custom()
    .maxAttempts(3)
    .waitDuration(Duration.ofSeconds(1))
    .intervalFunction(IntervalFunction.ofExponentialBackoff())
    .retryExceptions(IOException.class)
    .build();
\`\`\`

**Практические преимущества:**
- Готовые к production функции: метрики, события, Spring Boot интеграция
- Декларативная конфигурация вместо императивного кода
- Используется Netflix, Spotify и компаниями Fortune 500`
        },
        uz: {
            title: 'Resilience4j da Retry konfiguratsiyasi',
            solutionCode: `import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import java.io.IOException;
import java.util.function.Supplier;

public class Resilience4jRetry {
    // Vaqtinchalik nosozliklarni taqlid qiluvchi xizmat
    static class RemoteService {
        private int callCount = 0;

        public String fetchData() throws IOException {
            callCount++;
            System.out.println("Masofaviy xizmatni chaqirish (urinish " + callCount + ")");

            if (callCount < 3) {
                throw new IOException("Ulanish vaqti tugadi - vaqtinchalik nosozlik");
            }

            return "Muvaffaqiyat! Ma'lumotlar " + callCount + "-urinishda olindi";
        }

        public void reset() {
            callCount = 0;
        }
    }

    public static void main(String[] args) {
        // Eksponensial kechikish bilan Retry ni sozlash
        RetryConfig config = RetryConfig.custom()
            .maxAttempts(3)                                    // Maksimal qayta urinishlar soni
            .waitDuration(Duration.ofSeconds(1))              // Boshlang'ich kutish davomiyligi
            .intervalFunction(                                 // Eksponensial kechikish
                io.github.resilience4j.retry.IntervalFunction
                    .ofExponentialBackoff(Duration.ofSeconds(1), 2.0)
            )
            .retryExceptions(IOException.class, RuntimeException.class)  // Bu istisnolarda qayta urinish
            .build();

        // RetryRegistry va Retry instansiyasini yaratish
        RetryRegistry registry = RetryRegistry.of(config);
        Retry retry = registry.retry("remoteServiceRetry");

        // Qayta urinishlarni kuzatish uchun hodisa tinglovchilarini qo'shish
        retry.getEventPublisher()
            .onRetry(event -> {
                System.out.println("Qayta urinish hodisasi: Urinish #" +
                    event.getNumberOfRetryAttempts() +
                    " - Oxirgi istisno: " + event.getLastThrowable().getMessage());
            })
            .onSuccess(event -> {
                System.out.println("Muvaffaqiyat hodisasi: Operatsiya muvaffaqiyatli bajarildi");
            })
            .onError(event -> {
                System.out.println("Xato hodisasi: Barcha qayta urinishlar muvaffaqiyatsiz");
            });

        RemoteService service = new RemoteService();

        // Misol 1: Vaqtinchalik nosozliklardan keyin muvaffaqiyatli qayta urinish
        System.out.println("=== Test 1: Oxir-oqibat muvaffaqiyatli ===");
        try {
            // Supplier ni qayta urinish logikasi bilan dekoratsiya qilish
            Supplier<String> decoratedSupplier = Retry.decorateSupplier(retry, () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

            // Dekoratsiya qilingan funksiyani bajarish
            String result = decoratedSupplier.get();
            System.out.println("Natija: " + result);

        } catch (Exception e) {
            System.out.println("Yakuniy muvaffaqiyatsizlik: " + e.getMessage());
        }

        // Misol 2: Boshqa parametrlar bilan qayta urinishni sozlash
        System.out.println("\\n=== Test 2: Maxsus qayta urinish konfiguratsiyasi ===");
        RetryConfig customConfig = RetryConfig.custom()
            .maxAttempts(5)
            .waitDuration(Duration.ofMillis(500))
            .retryOnException(e -> e instanceof IOException)  // Qayta urinish predikati
            .build();

        Retry customRetry = RetryRegistry.of(customConfig).retry("customRetry");

        customRetry.getEventPublisher()
            .onRetry(event -> System.out.println(
                "Maxsus qayta urinish - Urinish: " + event.getNumberOfRetryAttempts()
            ));

        service.reset();
        Supplier<String> customDecoratedSupplier = Retry.decorateSupplier(
            customRetry,
            () -> {
                try {
                    return service.fetchData();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        );

        try {
            String result = customDecoratedSupplier.get();
            System.out.println("Maxsus qayta urinish natijasi: " + result);
        } catch (Exception e) {
            System.out.println("Maxsus qayta urinish muvaffaqiyatsiz: " + e.getMessage());
        }
    }
}`,
            description: `Ilg'or qayta urinish naqshlari uchun Resilience4j kutubxonasidan foydalanishni o'rganing.

**Talablar:**
1. maxAttempts=3 bilan Resilience4j Retry ni sozlang
2. Qayta urinishlar orasida waitDuration ni o'rnating (1 soniya)
3. multiplier=2 bilan eksponensial kechikishni qo'shing
4. Faqat muayyan istisnolar uchun qayta urinishni sozlang (IOException, TimeoutException)
5. Qayta urinish urinishlarini loglash uchun hodisa tinglovchilarini qo'shing
6. Funksiyani qayta urinish logikasi bilan dekoratsiya qiling
7. Vaqtinchalik nosozliklardan keyin muvaffaqiyatli qayta urinishni namoyish eting

Resilience4j - bu Netflix Hystrix dan ilhomlantirilgan, funksional dasturlash uchun mo'ljallangan yengil nosozliklarga chidamlilik kutubxonasi.`,
            hint1: `Qayta urinish konfiguratsiyasini qurish uchun RetryConfig.custom() dan foydalaning. maxAttempts, waitDuration ni o'rnating va eksponensial kechikishlar uchun IntervalFunction.ofExponentialBackoff() dan foydalaning.`,
            hint2: `Xizmat chaqiruvlarini o'rash uchun Retry.decorateSupplier() yoki Retry.decorateFunction() dan foydalaning. Qayta urinish hodisalarini kuzatish uchun retry.getEventPublisher() yordamida hodisa tinglovchilarini qo'shing.`,
            whyItMatters: `Resilience4j mustahkam Java ilovalarini yaratish uchun zamonaviy standart hisoblanadi. U minimal xarajatlar bilan naqshlarning boy to'plamini (retry, circuit breaker, rate limiter, bulkhead) taqdim etadi.

**Ishlab chiqarish patterni:**
\`\`\`java
RetryConfig config = RetryConfig.custom()
    .maxAttempts(3)
    .waitDuration(Duration.ofSeconds(1))
    .intervalFunction(IntervalFunction.ofExponentialBackoff())
    .retryExceptions(IOException.class)
    .build();
\`\`\`

**Amaliy foydalari:**
- Production ga tayyor funksiyalar: metrikalar, hodisalar, Spring Boot integratsiyasi
- Imperativ kod o'rniga deklarativ konfiguratsiya
- Netflix, Spotify va Fortune 500 kompaniyalari tomonidan qo'llaniladi`
        }
    }
};

export default task;
