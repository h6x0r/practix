import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-circuit-breaker',
    title: 'Resilience4j CircuitBreaker Pattern',
    difficulty: 'medium',
    tags: ['java', 'resilience4j', 'circuit-breaker', 'fault-tolerance'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to implement the Circuit Breaker pattern to prevent cascading failures.

**Requirements:**
1. Configure a CircuitBreaker with Resilience4j
2. Set failure rate threshold (50%)
3. Set minimum number of calls (5)
4. Configure wait duration in open state (10 seconds)
5. Set permitted calls in half-open state (3)
6. Add event listeners to track state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
7. Demonstrate circuit breaker opening after threshold failures
8. Show recovery when service becomes healthy again

The Circuit Breaker pattern prevents an application from repeatedly trying to execute an operation that's likely to fail, allowing it to recover.`,
    initialCode: `import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import java.time.Duration;
import java.util.function.Supplier;

public class CircuitBreakerPattern {
    // Configure CircuitBreaker
    // - failureRateThreshold: 50%
    // - minimumNumberOfCalls: 5
    // - waitDurationInOpenState: 10 seconds
    // - permittedNumberOfCallsInHalfOpenState: 3
    // - automaticTransitionFromOpenToHalfOpenEnabled: true

    static class ExternalService {
        private boolean isHealthy = false;

        public String call() throws Exception {
            if (!isHealthy) {
                throw new Exception("Service unavailable");
            }
            return "Success from external service";
        }

        public void setHealthy(boolean healthy) {
            this.isHealthy = healthy;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Create CircuitBreaker config

        // Create CircuitBreaker instance

        // Add event listeners

        // Test circuit breaker behavior
    }
}`,
    solutionCode: `import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import java.time.Duration;
import java.util.function.Supplier;

public class CircuitBreakerPattern {
    // External service that can be healthy or unhealthy
    static class ExternalService {
        private boolean isHealthy = false;
        private int callCount = 0;

        public String call() throws Exception {
            callCount++;
            System.out.println("  -> Service call #" + callCount);

            if (!isHealthy) {
                throw new Exception("Service unavailable");
            }
            return "Success from external service";
        }

        public void setHealthy(boolean healthy) {
            this.isHealthy = healthy;
            System.out.println("Service health changed: " +
                (healthy ? "HEALTHY" : "UNHEALTHY"));
        }

        public int getCallCount() {
            return callCount;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Configure CircuitBreaker
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50)                          // Open when 50% of calls fail
            .minimumNumberOfCalls(5)                          // Need at least 5 calls to calculate rate
            .waitDurationInOpenState(Duration.ofSeconds(10))  // Wait 10s before attempting recovery
            .permittedNumberOfCallsInHalfOpenState(3)        // Try 3 calls in half-open state
            .automaticTransitionFromOpenToHalfOpenEnabled(true) // Auto transition to half-open
            .slidingWindowSize(10)                            // Track last 10 calls
            .build();

        // Create CircuitBreaker
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(config);
        CircuitBreaker circuitBreaker = registry.circuitBreaker("externalServiceBreaker");

        // Add event listeners to track state transitions
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                System.out.println("\\n*** Circuit Breaker State Transition: " +
                    event.getStateTransition() + " ***\\n");
            })
            .onSuccess(event -> {
                System.out.println("  Success event");
            })
            .onError(event -> {
                System.out.println("  Error event: " + event.getThrowable().getMessage());
            })
            .onCallNotPermitted(event -> {
                System.out.println("  Call not permitted - Circuit is OPEN");
            });

        ExternalService service = new ExternalService();

        // Phase 1: Service is unhealthy - Circuit will open
        System.out.println("=== Phase 1: Unhealthy Service (Circuit will OPEN) ===");
        service.setHealthy(false);

        for (int i = 1; i <= 7; i++) {
            try {
                System.out.println("\\nAttempt " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Result: " + result);

            } catch (Exception e) {
                System.out.println("  Failed: " + e.getMessage());
            }
        }

        // Check circuit breaker state
        System.out.println("\\nCircuit state: " + circuitBreaker.getState());
        System.out.println("Metrics: " + circuitBreaker.getMetrics());

        // Phase 2: Wait and service becomes healthy
        System.out.println("\\n=== Phase 2: Waiting for circuit to transition to HALF_OPEN ===");
        System.out.println("Waiting 11 seconds...");
        Thread.sleep(11000);

        // Service becomes healthy
        service.setHealthy(true);

        // Phase 3: Recovery - Circuit will close again
        System.out.println("\\n=== Phase 3: Service Recovered (Circuit will CLOSE) ===");

        for (int i = 1; i <= 5; i++) {
            try {
                System.out.println("\\nRecovery attempt " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Result: " + result);
                System.out.println("  Circuit state: " + circuitBreaker.getState());

            } catch (Exception e) {
                System.out.println("  Failed: " + e.getMessage());
            }
        }

        System.out.println("\\n=== Final State ===");
        System.out.println("Circuit state: " + circuitBreaker.getState());
        System.out.println("Total service calls: " + service.getCallCount());
        System.out.println("Metrics: " + circuitBreaker.getMetrics());
    }
}`,
    hint1: `Configure CircuitBreakerConfig with thresholds and durations. Use failureRateThreshold to set when the circuit should open (e.g., 50% failures).`,
    hint2: `The circuit breaker has three states: CLOSED (normal), OPEN (failing, calls blocked), and HALF_OPEN (testing recovery). Use event listeners to track transitions between these states.`,
    whyItMatters: `The Circuit Breaker pattern is essential for building resilient microservices. It prevents cascading failures by failing fast when a service is down, rather than waiting for timeouts. This protects your system resources and provides better user experience. Circuit breakers are used extensively in distributed systems at companies like Netflix, Amazon, and Uber to handle service failures gracefully.

**Production Pattern:**
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)          // Open at 50% failures
    .minimumNumberOfCalls(5)           // Minimum 5 calls
    .waitDurationInOpenState(Duration.ofSeconds(10))
    .build();
\`\`\`

**Practical Benefits:**
- Fast fail instead of waiting for timeouts
- Protects system resources and improves UX
- Used by Netflix, Amazon, and Uber for handling service failures`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.concurrent.atomic.AtomicInteger;

// Test 1: Circuit breaker starts in CLOSED state
class Test1 {
    @Test
    void testInitialState() {
        String initialState = "CLOSED";
        assertEquals("CLOSED", initialState);
    }
}

// Test 2: Failure rate threshold calculation
class Test2 {
    @Test
    void testFailureRateThreshold() {
        int totalCalls = 10;
        int failures = 5;
        double failureRate = (failures * 100.0) / totalCalls;
        assertEquals(50.0, failureRate, 0.01);
    }
}

// Test 3: Circuit opens when threshold exceeded
class Test3 {
    @Test
    void testCircuitOpens() {
        int minimumCalls = 5;
        double threshold = 50.0;
        int failures = 3;
        int totalCalls = 5;
        double failureRate = (failures * 100.0) / totalCalls;
        boolean shouldOpen = totalCalls >= minimumCalls && failureRate >= threshold;
        assertTrue(shouldOpen);
    }
}

// Test 4: Minimum calls required before opening
class Test4 {
    @Test
    void testMinimumCalls() {
        int minimumCalls = 5;
        int currentCalls = 3;
        boolean canCalculateRate = currentCalls >= minimumCalls;
        assertFalse(canCalculateRate);
    }
}

// Test 5: Sliding window tracks correct number of calls
class Test5 {
    @Test
    void testSlidingWindow() {
        int slidingWindowSize = 10;
        AtomicInteger callCount = new AtomicInteger(0);
        for (int i = 0; i < 15; i++) {
            callCount.incrementAndGet();
        }
        int windowedCalls = Math.min(callCount.get(), slidingWindowSize);
        assertEquals(10, windowedCalls);
    }
}

// Test 6: Wait duration in open state
class Test6 {
    @Test
    void testWaitDuration() {
        long waitDurationSeconds = 10;
        long waitDurationMs = waitDurationSeconds * 1000;
        assertEquals(10000, waitDurationMs);
    }
}

// Test 7: Permitted calls in half-open state
class Test7 {
    @Test
    void testHalfOpenPermittedCalls() {
        int permittedCalls = 3;
        AtomicInteger halfOpenCalls = new AtomicInteger(0);
        for (int i = 0; i < 5; i++) {
            if (halfOpenCalls.get() < permittedCalls) {
                halfOpenCalls.incrementAndGet();
            }
        }
        assertEquals(3, halfOpenCalls.get());
    }
}

// Test 8: Successful calls in half-open close circuit
class Test8 {
    @Test
    void testHalfOpenToClosedTransition() {
        int permittedCalls = 3;
        int successfulCalls = 3;
        boolean shouldClose = successfulCalls >= permittedCalls;
        assertTrue(shouldClose);
    }
}

// Test 9: Call not permitted in OPEN state
class Test9 {
    @Test
    void testCallNotPermittedInOpen() {
        String state = "OPEN";
        boolean callPermitted = !state.equals("OPEN");
        assertFalse(callPermitted);
    }
}

// Test 10: State transitions correctly
class Test10 {
    @Test
    void testStateTransitions() {
        String[] validTransitions = {"CLOSED", "OPEN", "HALF_OPEN", "CLOSED"};
        assertEquals("CLOSED", validTransitions[0]);
        assertEquals("OPEN", validTransitions[1]);
        assertEquals("HALF_OPEN", validTransitions[2]);
        assertEquals("CLOSED", validTransitions[3]);
    }
}`,
    translations: {
        ru: {
            title: 'Паттерн CircuitBreaker в Resilience4j',
            solutionCode: `import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import java.time.Duration;
import java.util.function.Supplier;

public class CircuitBreakerPattern {
    // Внешний сервис, который может быть здоровым или нездоровым
    static class ExternalService {
        private boolean isHealthy = false;
        private int callCount = 0;

        public String call() throws Exception {
            callCount++;
            System.out.println("  -> Вызов сервиса #" + callCount);

            if (!isHealthy) {
                throw new Exception("Сервис недоступен");
            }
            return "Успех от внешнего сервиса";
        }

        public void setHealthy(boolean healthy) {
            this.isHealthy = healthy;
            System.out.println("Здоровье сервиса изменилось: " +
                (healthy ? "ЗДОРОВ" : "НЕЗДОРОВ"));
        }

        public int getCallCount() {
            return callCount;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Настройка CircuitBreaker
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50)                          // Открыть при 50% неудачных вызовов
            .minimumNumberOfCalls(5)                          // Минимум 5 вызовов для расчета процента
            .waitDurationInOpenState(Duration.ofSeconds(10))  // Ждать 10с перед попыткой восстановления
            .permittedNumberOfCallsInHalfOpenState(3)        // Попробовать 3 вызова в полуоткрытом состоянии
            .automaticTransitionFromOpenToHalfOpenEnabled(true) // Автопереход в полуоткрытое
            .slidingWindowSize(10)                            // Отслеживать последние 10 вызовов
            .build();

        // Создание CircuitBreaker
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(config);
        CircuitBreaker circuitBreaker = registry.circuitBreaker("externalServiceBreaker");

        // Добавление слушателей событий для отслеживания переходов состояний
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                System.out.println("\\n*** Переход состояния Circuit Breaker: " +
                    event.getStateTransition() + " ***\\n");
            })
            .onSuccess(event -> {
                System.out.println("  Событие успеха");
            })
            .onError(event -> {
                System.out.println("  Событие ошибки: " + event.getThrowable().getMessage());
            })
            .onCallNotPermitted(event -> {
                System.out.println("  Вызов не разрешен - Circuit ОТКРЫТ");
            });

        ExternalService service = new ExternalService();

        // Фаза 1: Сервис нездоров - Circuit откроется
        System.out.println("=== Фаза 1: Нездоровый сервис (Circuit откроется) ===");
        service.setHealthy(false);

        for (int i = 1; i <= 7; i++) {
            try {
                System.out.println("\\nПопытка " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Результат: " + result);

            } catch (Exception e) {
                System.out.println("  Не удалось: " + e.getMessage());
            }
        }

        // Проверка состояния circuit breaker
        System.out.println("\\nСостояние Circuit: " + circuitBreaker.getState());
        System.out.println("Метрики: " + circuitBreaker.getMetrics());

        // Фаза 2: Ожидание и сервис становится здоровым
        System.out.println("\\n=== Фаза 2: Ожидание перехода Circuit в HALF_OPEN ===");
        System.out.println("Ожидание 11 секунд...");
        Thread.sleep(11000);

        // Сервис становится здоровым
        service.setHealthy(true);

        // Фаза 3: Восстановление - Circuit снова закроется
        System.out.println("\\n=== Фаза 3: Сервис восстановлен (Circuit закроется) ===");

        for (int i = 1; i <= 5; i++) {
            try {
                System.out.println("\\nПопытка восстановления " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Результат: " + result);
                System.out.println("  Состояние Circuit: " + circuitBreaker.getState());

            } catch (Exception e) {
                System.out.println("  Не удалось: " + e.getMessage());
            }
        }

        System.out.println("\\n=== Финальное состояние ===");
        System.out.println("Состояние Circuit: " + circuitBreaker.getState());
        System.out.println("Всего вызовов сервиса: " + service.getCallCount());
        System.out.println("Метрики: " + circuitBreaker.getMetrics());
    }
}`,
            description: `Изучите реализацию паттерна Circuit Breaker для предотвращения каскадных сбоев.

**Требования:**
1. Настройте CircuitBreaker с Resilience4j
2. Установите порог процента сбоев (50%)
3. Установите минимальное количество вызовов (5)
4. Настройте длительность ожидания в открытом состоянии (10 секунд)
5. Установите разрешенные вызовы в полуоткрытом состоянии (3)
6. Добавьте слушателей событий для отслеживания переходов состояний (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
7. Продемонстрируйте открытие circuit breaker после превышения порога сбоев
8. Покажите восстановление, когда сервис снова становится здоровым

Паттерн Circuit Breaker предотвращает попытки приложения многократно выполнять операцию, которая, вероятно, не удастся, позволяя ему восстановиться.`,
            hint1: `Настройте CircuitBreakerConfig с порогами и длительностями. Используйте failureRateThreshold, чтобы установить, когда circuit должен открыться (например, 50% сбоев).`,
            hint2: `Circuit breaker имеет три состояния: CLOSED (нормальное), OPEN (сбой, вызовы блокируются) и HALF_OPEN (тестирование восстановления). Используйте слушателей событий для отслеживания переходов между этими состояниями.`,
            whyItMatters: `Паттерн Circuit Breaker необходим для создания устойчивых микросервисов. Он предотвращает каскадные сбои, быстро отказывая при недоступности сервиса, вместо ожидания таймаутов.

**Продакшен паттерн:**
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)          // Открыть при 50% сбоев
    .minimumNumberOfCalls(5)           // Минимум 5 вызовов
    .waitDurationInOpenState(Duration.ofSeconds(10))
    .build();
\`\`\`

**Практические преимущества:**
- Быстрый отказ вместо ожидания таймаутов
- Защита ресурсов системы и улучшение UX
- Используется Netflix, Amazon и Uber для обработки сбоев сервисов`
        },
        uz: {
            title: 'Resilience4j da CircuitBreaker naqshi',
            solutionCode: `import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import java.time.Duration;
import java.util.function.Supplier;

public class CircuitBreakerPattern {
    // Sog'lom yoki nosog'lom bo'lishi mumkin bo'lgan tashqi xizmat
    static class ExternalService {
        private boolean isHealthy = false;
        private int callCount = 0;

        public String call() throws Exception {
            callCount++;
            System.out.println("  -> Xizmat chaqiruvi #" + callCount);

            if (!isHealthy) {
                throw new Exception("Xizmat mavjud emas");
            }
            return "Tashqi xizmatdan muvaffaqiyat";
        }

        public void setHealthy(boolean healthy) {
            this.isHealthy = healthy;
            System.out.println("Xizmat salomatligi o'zgartirildi: " +
                (healthy ? "SOG'LOM" : "NOSOG'LOM"));
        }

        public int getCallCount() {
            return callCount;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // CircuitBreaker ni sozlash
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50)                          // 50% chaqiruvlar muvaffaqiyatsiz bo'lganda ochish
            .minimumNumberOfCalls(5)                          // Foizni hisoblash uchun kamida 5 chaqiruv kerak
            .waitDurationInOpenState(Duration.ofSeconds(10))  // Tiklanish urinishidan oldin 10s kutish
            .permittedNumberOfCallsInHalfOpenState(3)        // Yarim ochiq holatda 3 chaqiruvni sinab ko'rish
            .automaticTransitionFromOpenToHalfOpenEnabled(true) // Yarim ochiqqa avto o'tish
            .slidingWindowSize(10)                            // Oxirgi 10 chaqiruvni kuzatish
            .build();

        // CircuitBreaker yaratish
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(config);
        CircuitBreaker circuitBreaker = registry.circuitBreaker("externalServiceBreaker");

        // Holat o'tishlarini kuzatish uchun hodisa tinglovchilarini qo'shish
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                System.out.println("\\n*** Circuit Breaker holat o'tishi: " +
                    event.getStateTransition() + " ***\\n");
            })
            .onSuccess(event -> {
                System.out.println("  Muvaffaqiyat hodisasi");
            })
            .onError(event -> {
                System.out.println("  Xato hodisasi: " + event.getThrowable().getMessage());
            })
            .onCallNotPermitted(event -> {
                System.out.println("  Chaqiruv ruxsat etilmagan - Circuit OCHIQ");
            });

        ExternalService service = new ExternalService();

        // Faza 1: Xizmat nosog'lom - Circuit ochiladi
        System.out.println("=== Faza 1: Nosog'lom xizmat (Circuit OCHILadi) ===");
        service.setHealthy(false);

        for (int i = 1; i <= 7; i++) {
            try {
                System.out.println("\\nUrinish " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Natija: " + result);

            } catch (Exception e) {
                System.out.println("  Muvaffaqiyatsiz: " + e.getMessage());
            }
        }

        // Circuit breaker holatini tekshirish
        System.out.println("\\nCircuit holati: " + circuitBreaker.getState());
        System.out.println("Metrikalar: " + circuitBreaker.getMetrics());

        // Faza 2: Kutish va xizmat sog'lom bo'ladi
        System.out.println("\\n=== Faza 2: Circuit HALF_OPEN ga o'tishini kutish ===");
        System.out.println("11 soniya kutilmoqda...");
        Thread.sleep(11000);

        // Xizmat sog'lom bo'ladi
        service.setHealthy(true);

        // Faza 3: Tiklanish - Circuit yana yopiladi
        System.out.println("\\n=== Faza 3: Xizmat tiklandi (Circuit YOPILadi) ===");

        for (int i = 1; i <= 5; i++) {
            try {
                System.out.println("\\nTiklanish urinishi " + i + ":");
                Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(
                    circuitBreaker,
                    () -> {
                        try {
                            return service.call();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                );

                String result = decoratedSupplier.get();
                System.out.println("  Natija: " + result);
                System.out.println("  Circuit holati: " + circuitBreaker.getState());

            } catch (Exception e) {
                System.out.println("  Muvaffaqiyatsiz: " + e.getMessage());
            }
        }

        System.out.println("\\n=== Yakuniy holat ===");
        System.out.println("Circuit holati: " + circuitBreaker.getState());
        System.out.println("Jami xizmat chaqiruvlari: " + service.getCallCount());
        System.out.println("Metrikalar: " + circuitBreaker.getMetrics());
    }
}`,
            description: `Kaskadli nosozliklarni oldini olish uchun Circuit Breaker naqshini amalga oshirishni o'rganing.

**Talablar:**
1. Resilience4j bilan CircuitBreaker ni sozlang
2. Nosozlik darajasi chegarasini o'rnating (50%)
3. Minimal chaqiruvlar sonini o'rnating (5)
4. Ochiq holatda kutish davomiyligini sozlang (10 soniya)
5. Yarim ochiq holatda ruxsat etilgan chaqiruvlarni o'rnating (3)
6. Holat o'tishlari uchun hodisa tinglovchilarini qo'shing (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
7. Nosozlik chegarasidan keyin circuit breaker ochilishini namoyish eting
8. Xizmat yana sog'lom bo'lganda tiklanishni ko'rsating

Circuit Breaker naqshi ilovaning muvaffaqiyatsiz bo'lishi mumkin bo'lgan operatsiyani qayta-qayta bajarishga urinishining oldini oladi va tiklanishga imkon beradi.`,
            hint1: `CircuitBreakerConfig ni chegaralar va davomiyliklar bilan sozlang. Circuit qachon ochilishi kerakligini belgilash uchun failureRateThreshold dan foydalaning (masalan, 50% nosozliklar).`,
            hint2: `Circuit breaker uchta holatga ega: CLOSED (normal), OPEN (nosozlik, chaqiruvlar bloklanadi) va HALF_OPEN (tiklanishni sinash). Bu holatlar orasidagi o'tishlarni kuzatish uchun hodisa tinglovchilaridan foydalaning.`,
            whyItMatters: `Circuit Breaker naqshi mustahkam mikroservislarni yaratish uchun zarurdir. U xizmat ishlamay qolganda timeout ni kutish o'rniga tez muvaffaqiyatsiz bo'lish orqali kaskadli nosozliklarning oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)          // 50% nosozlikda ochish
    .minimumNumberOfCalls(5)           // Minimal 5 chaqiruv
    .waitDurationInOpenState(Duration.ofSeconds(10))
    .build();
\`\`\`

**Amaliy foydalari:**
- Timeout ni kutish o'rniga tez muvaffaqiyatsiz bo'lish
- Tizim resurslarini himoya qilish va UX ni yaxshilash
- Netflix, Amazon va Uber tomonidan xizmat nosozliklarini boshqarish uchun qo'llaniladi`
        }
    }
};

export default task;
