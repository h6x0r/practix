import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-timers',
    title: 'Timer Metrics',
    difficulty: 'medium',
    tags: ['java', 'metrics', 'micrometer', 'timer'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Timer metrics for measuring durations.

**Requirements:**
1. Create a MeterRegistry
2. Create a Timer named "api.requests"
3. Use timer.record() with a Runnable to measure a simulated operation
4. Simulate an operation that takes 100ms using Thread.sleep()
5. Record the timer 3 times with different durations
6. Use Timer.Sample to manually start and stop timing
7. Print timer statistics: count, totalTime, max, mean
8. Create a timer with tags for different endpoints
9. Record metrics for "/users" and "/orders" endpoints
10. Compare statistics between endpoints

Timer measures both the frequency and duration of events. It's perfect for tracking request latencies, method execution times, and task durations.`,
    initialCode: `import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.TimeUnit;

public class TimerMetrics {
    public static void main(String[] args) throws InterruptedException {
        // Create a MeterRegistry

        // Create a Timer

        // Record operations

        // Use Timer.Sample for manual timing

        // Print statistics
    }
}`,
    solutionCode: `import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.TimeUnit;

public class TimerMetrics {
    public static void main(String[] args) throws InterruptedException {
        // Create a MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Create a Timer named "api.requests"
        Timer timer = registry.timer("api.requests");

        // Use timer.record() with a Runnable to measure a simulated operation
        System.out.println("Recording operation 1...");
        timer.record(() -> {
            try {
                // Simulate an operation that takes 100ms
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Record the timer 3 times with different durations
        System.out.println("Recording operation 2...");
        timer.record(() -> {
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        System.out.println("Recording operation 3...");
        timer.record(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Use Timer.Sample to manually start and stop timing
        System.out.println("\\nUsing Timer.Sample for manual timing...");
        Timer.Sample sample = Timer.start(registry);
        Thread.sleep(120);
        sample.stop(timer);

        // Print timer statistics: count, totalTime, max, mean
        System.out.println("\\n=== Timer Statistics ===");
        System.out.println("Count: " + timer.count());
        System.out.println("Total Time: " + String.format("%.2f", timer.totalTime(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("Max: " + String.format("%.2f", timer.max(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("Mean: " + String.format("%.2f", timer.mean(TimeUnit.MILLISECONDS)) + "ms");

        // Create a timer with tags for different endpoints
        Timer usersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/users")
            .description("Request timer for /users endpoint")
            .register(registry);

        Timer ordersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/orders")
            .description("Request timer for /orders endpoint")
            .register(registry);

        // Record metrics for "/users" and "/orders" endpoints
        System.out.println("\\n=== Recording endpoint metrics ===");

        // Simulate /users requests (faster)
        for (int i = 0; i < 5; i++) {
            usersTimer.record(() -> {
                try {
                    Thread.sleep(50 + (long)(Math.random() * 50));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // Simulate /orders requests (slower)
        for (int i = 0; i < 5; i++) {
            ordersTimer.record(() -> {
                try {
                    Thread.sleep(100 + (long)(Math.random() * 100));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // Compare statistics between endpoints
        System.out.println("\\n=== Endpoint Comparison ===");
        System.out.println("/users endpoint:");
        System.out.println("  Count: " + usersTimer.count());
        System.out.println("  Mean: " + String.format("%.2f", usersTimer.mean(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("  Max: " + String.format("%.2f", usersTimer.max(TimeUnit.MILLISECONDS)) + "ms");

        System.out.println("\\n/orders endpoint:");
        System.out.println("  Count: " + ordersTimer.count());
        System.out.println("  Mean: " + String.format("%.2f", ordersTimer.mean(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("  Max: " + String.format("%.2f", ordersTimer.max(TimeUnit.MILLISECONDS)) + "ms");
    }
}`,
    hint1: `Use timer.record(Runnable) to automatically measure the duration of a block of code. Timer.Sample gives you manual control over start and stop.`,
    hint2: `Timer provides rich statistics including count(), totalTime(), max(), and mean(). Use tags to track metrics for different scenarios separately.`,
    whyItMatters: `Timers are essential for performance monitoring. They help you identify slow operations, track SLA compliance, and understand system behavior under load. Proper timer usage is critical for production observability.

**Production Pattern:**
\`\`\`java
@Service
public class PaymentService {
    private final Timer paymentTimer;

    public PaymentService(MeterRegistry registry) {
        this.paymentTimer = Timer.builder("payment.processing.time")
            .description("Payment processing duration")
            .tags("service", "payment")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);
    }

    public Payment processPayment(PaymentRequest request) {
        return paymentTimer.record(() -> {
            // Process payment
            return paymentGateway.charge(request);
        });
    }
}
\`\`\`

**Practical Benefits:**
- Measure execution time of critical operations
- Track SLA compliance (99% < 500ms)
- Identify performance bottlenecks`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.TimeUnit;

// Test 1: MeterRegistry is created
class Test1 {
    @Test
    void testMeterRegistryCreated() {
        MeterRegistry registry = new SimpleMeterRegistry();
        assertNotNull(registry);
    }
}

// Test 2: Timer is created with name
class Test2 {
    @Test
    void testTimerCreatedWithName() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("api.requests");
        assertNotNull(timer);
        assertEquals("api.requests", timer.getId().getName());
    }
}

// Test 3: Timer.record() executes Runnable
class Test3 {
    @Test
    void testTimerRecordRunnable() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        boolean[] executed = {false};
        timer.record(() -> executed[0] = true);
        assertTrue(executed[0]);
    }
}

// Test 4: Timer count increments
class Test4 {
    @Test
    void testTimerCountIncrements() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        timer.record(() -> {});
        timer.record(() -> {});
        assertEquals(2, timer.count());
    }
}

// Test 5: Timer records duration
class Test5 {
    @Test
    void testTimerRecordsDuration() throws InterruptedException {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        timer.record(() -> {
            try { Thread.sleep(50); } catch (InterruptedException e) {}
        });
        assertTrue(timer.totalTime(TimeUnit.MILLISECONDS) >= 40);
    }
}

// Test 6: Timer.Sample works correctly
class Test6 {
    @Test
    void testTimerSample() throws InterruptedException {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        Timer.Sample sample = Timer.start(registry);
        Thread.sleep(30);
        sample.stop(timer);
        assertEquals(1, timer.count());
    }
}

// Test 7: Timer with tags is created
class Test7 {
    @Test
    void testTimerWithTags() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/users")
            .register(registry);
        assertTrue(timer.getId().getTags().stream()
            .anyMatch(t -> t.getKey().equals("endpoint") && t.getValue().equals("/users")));
    }
}

// Test 8: Timer max returns maximum duration
class Test8 {
    @Test
    void testTimerMax() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        timer.record(100, TimeUnit.MILLISECONDS);
        timer.record(200, TimeUnit.MILLISECONDS);
        assertEquals(200.0, timer.max(TimeUnit.MILLISECONDS), 1.0);
    }
}

// Test 9: Timer mean calculates correctly
class Test9 {
    @Test
    void testTimerMean() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer = registry.timer("test.timer");
        timer.record(100, TimeUnit.MILLISECONDS);
        timer.record(200, TimeUnit.MILLISECONDS);
        assertEquals(150.0, timer.mean(TimeUnit.MILLISECONDS), 1.0);
    }
}

// Test 10: Multiple timers are independent
class Test10 {
    @Test
    void testMultipleTimersIndependent() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Timer timer1 = registry.timer("timer1");
        Timer timer2 = registry.timer("timer2");
        timer1.record(100, TimeUnit.MILLISECONDS);
        timer1.record(100, TimeUnit.MILLISECONDS);
        timer2.record(50, TimeUnit.MILLISECONDS);
        assertEquals(2, timer1.count());
        assertEquals(1, timer2.count());
    }
}`,
    translations: {
        ru: {
            title: 'Метрики Timer',
            solutionCode: `import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.TimeUnit;

public class TimerMetrics {
    public static void main(String[] args) throws InterruptedException {
        // Создаем MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Создаем Timer с именем "api.requests"
        Timer timer = registry.timer("api.requests");

        // Используем timer.record() с Runnable для измерения симулированной операции
        System.out.println("Запись операции 1...");
        timer.record(() -> {
            try {
                // Симулируем операцию, которая занимает 100мс
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Записываем таймер 3 раза с разными длительностями
        System.out.println("Запись операции 2...");
        timer.record(() -> {
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        System.out.println("Запись операции 3...");
        timer.record(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Используем Timer.Sample для ручного измерения времени
        System.out.println("\\nИспользование Timer.Sample для ручного измерения...");
        Timer.Sample sample = Timer.start(registry);
        Thread.sleep(120);
        sample.stop(timer);

        // Выводим статистику таймера: count, totalTime, max, mean
        System.out.println("\\n=== Статистика таймера ===");
        System.out.println("Количество: " + timer.count());
        System.out.println("Общее время: " + String.format("%.2f", timer.totalTime(TimeUnit.MILLISECONDS)) + "мс");
        System.out.println("Максимум: " + String.format("%.2f", timer.max(TimeUnit.MILLISECONDS)) + "мс");
        System.out.println("Среднее: " + String.format("%.2f", timer.mean(TimeUnit.MILLISECONDS)) + "мс");

        // Создаем таймер с тегами для разных конечных точек
        Timer usersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/users")
            .description("Таймер запросов для конечной точки /users")
            .register(registry);

        Timer ordersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/orders")
            .description("Таймер запросов для конечной точки /orders")
            .register(registry);

        // Записываем метрики для конечных точек "/users" и "/orders"
        System.out.println("\\n=== Запись метрик конечных точек ===");

        // Симулируем запросы к /users (быстрее)
        for (int i = 0; i < 5; i++) {
            usersTimer.record(() -> {
                try {
                    Thread.sleep(50 + (long)(Math.random() * 50));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // Симулируем запросы к /orders (медленнее)
        for (int i = 0; i < 5; i++) {
            ordersTimer.record(() -> {
                try {
                    Thread.sleep(100 + (long)(Math.random() * 100));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // Сравниваем статистику между конечными точками
        System.out.println("\\n=== Сравнение конечных точек ===");
        System.out.println("Конечная точка /users:");
        System.out.println("  Количество: " + usersTimer.count());
        System.out.println("  Среднее: " + String.format("%.2f", usersTimer.mean(TimeUnit.MILLISECONDS)) + "мс");
        System.out.println("  Максимум: " + String.format("%.2f", usersTimer.max(TimeUnit.MILLISECONDS)) + "мс");

        System.out.println("\\nКонечная точка /orders:");
        System.out.println("  Количество: " + ordersTimer.count());
        System.out.println("  Среднее: " + String.format("%.2f", ordersTimer.mean(TimeUnit.MILLISECONDS)) + "мс");
        System.out.println("  Максимум: " + String.format("%.2f", ordersTimer.max(TimeUnit.MILLISECONDS)) + "мс");
    }
}`,
            description: `Освойте метрики Timer для измерения длительности.

**Требования:**
1. Создайте MeterRegistry
2. Создайте Timer с именем "api.requests"
3. Используйте timer.record() с Runnable для измерения симулированной операции
4. Симулируйте операцию, которая занимает 100мс используя Thread.sleep()
5. Запишите таймер 3 раза с разными длительностями
6. Используйте Timer.Sample для ручного запуска и остановки измерения
7. Выведите статистику таймера: count, totalTime, max, mean
8. Создайте таймер с тегами для разных конечных точек
9. Запишите метрики для конечных точек "/users" и "/orders"
10. Сравните статистику между конечными точками

Timer измеряет как частоту, так и длительность событий. Идеально подходит для отслеживания задержек запросов, времени выполнения методов и длительности задач.`,
            hint1: `Используйте timer.record(Runnable) для автоматического измерения длительности блока кода. Timer.Sample дает вам ручной контроль над запуском и остановкой.`,
            hint2: `Timer предоставляет богатую статистику, включая count(), totalTime(), max() и mean(). Используйте теги для отдельного отслеживания метрик для разных сценариев.`,
            whyItMatters: `Таймеры необходимы для мониторинга производительности. Они помогают выявлять медленные операции, отслеживать соблюдение SLA и понимать поведение системы под нагрузкой. Правильное использование таймеров критически важно для наблюдаемости в продакшене.

**Продакшен паттерн:**
\`\`\`java
@Service
public class PaymentService {
    private final Timer paymentTimer;

    public PaymentService(MeterRegistry registry) {
        this.paymentTimer = Timer.builder("payment.processing.time")
            .description("Payment processing duration")
            .tags("service", "payment")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);
    }

    public Payment processPayment(PaymentRequest request) {
        return paymentTimer.record(() -> {
            // Обработка платежа
            return paymentGateway.charge(request);
        });
    }
}
\`\`\`

**Практические преимущества:**
- Измерение времени выполнения критических операций
- Отслеживание соблюдения SLA (99% < 500ms)
- Выявление узких мест производительности`
        },
        uz: {
            title: 'Timer Metrikalari',
            solutionCode: `import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.TimeUnit;

public class TimerMetrics {
    public static void main(String[] args) throws InterruptedException {
        // MeterRegistry yaratamiz
        MeterRegistry registry = new SimpleMeterRegistry();

        // "api.requests" nomi bilan Timer yaratamiz
        Timer timer = registry.timer("api.requests");

        // Simulyatsiya qilingan operatsiyani o'lchash uchun timer.record() ni Runnable bilan ishlatamiz
        System.out.println("Operatsiya 1 yozilmoqda...");
        timer.record(() -> {
            try {
                // 100ms vaqt oladigan operatsiyani simulyatsiya qilamiz
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Timerni turli davomiylikda 3 marta yozamiz
        System.out.println("Operatsiya 2 yozilmoqda...");
        timer.record(() -> {
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        System.out.println("Operatsiya 3 yozilmoqda...");
        timer.record(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        // Qo'lda vaqtni o'lchash uchun Timer.Sample dan foydalanamiz
        System.out.println("\\nQo'lda o'lchash uchun Timer.Sample ishlatilmoqda...");
        Timer.Sample sample = Timer.start(registry);
        Thread.sleep(120);
        sample.stop(timer);

        // Timer statistikasini chiqaramiz: count, totalTime, max, mean
        System.out.println("\\n=== Timer statistikasi ===");
        System.out.println("Soni: " + timer.count());
        System.out.println("Umumiy vaqt: " + String.format("%.2f", timer.totalTime(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("Maksimum: " + String.format("%.2f", timer.max(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("O'rtacha: " + String.format("%.2f", timer.mean(TimeUnit.MILLISECONDS)) + "ms");

        // Turli endpoint'lar uchun teglar bilan timer yaratamiz
        Timer usersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/users")
            .description("/users endpoint uchun so'rov taymer")
            .register(registry);

        Timer ordersTimer = Timer.builder("endpoint.requests")
            .tag("endpoint", "/orders")
            .description("/orders endpoint uchun so'rov taymer")
            .register(registry);

        // "/users" va "/orders" endpoint'lari uchun metrikalarni yozamiz
        System.out.println("\\n=== Endpoint metrikalari yozilmoqda ===");

        // /users so'rovlarini simulyatsiya qilamiz (tezroq)
        for (int i = 0; i < 5; i++) {
            usersTimer.record(() -> {
                try {
                    Thread.sleep(50 + (long)(Math.random() * 50));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // /orders so'rovlarini simulyatsiya qilamiz (sekinroq)
        for (int i = 0; i < 5; i++) {
            ordersTimer.record(() -> {
                try {
                    Thread.sleep(100 + (long)(Math.random() * 100));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // Endpoint'lar o'rtasida statistikani solishtiramiz
        System.out.println("\\n=== Endpoint'larni solishtirish ===");
        System.out.println("/users endpoint:");
        System.out.println("  Soni: " + usersTimer.count());
        System.out.println("  O'rtacha: " + String.format("%.2f", usersTimer.mean(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("  Maksimum: " + String.format("%.2f", usersTimer.max(TimeUnit.MILLISECONDS)) + "ms");

        System.out.println("\\n/orders endpoint:");
        System.out.println("  Soni: " + ordersTimer.count());
        System.out.println("  O'rtacha: " + String.format("%.2f", ordersTimer.mean(TimeUnit.MILLISECONDS)) + "ms");
        System.out.println("  Maksimum: " + String.format("%.2f", ordersTimer.max(TimeUnit.MILLISECONDS)) + "ms");
    }
}`,
            description: `Davomiylikni o'lchash uchun Timer metrikalarini o'zlashtirig.

**Talablar:**
1. MeterRegistry yarating
2. "api.requests" nomi bilan Timer yarating
3. Simulyatsiya qilingan operatsiyani o'lchash uchun timer.record() ni Runnable bilan ishlating
4. Thread.sleep() yordamida 100ms vaqt oladigan operatsiyani simulyatsiya qiling
5. Timerni turli davomiylikda 3 marta yozing
6. Qo'lda boshlash va to'xtatish uchun Timer.Sample dan foydalaning
7. Timer statistikasini chiqaring: count, totalTime, max, mean
8. Turli endpoint'lar uchun teglar bilan timer yarating
9. "/users" va "/orders" endpoint'lari uchun metrikalarni yozing
10. Endpoint'lar o'rtasida statistikani solishtiring

Timer hodisalarning chastotasini ham, davomiyligini ham o'lchaydi. So'rov kechikishlari, metod bajarilish vaqti va vazifa davomiyligini kuzatish uchun ideal.`,
            hint1: `Kod bloki davomiyligini avtomatik o'lchash uchun timer.record(Runnable) dan foydalaning. Timer.Sample boshlash va to'xtatish ustidan qo'lda nazorat beradi.`,
            hint2: `Timer count(), totalTime(), max() va mean() ni o'z ichiga olgan boy statistikani taqdim etadi. Turli stsenariylar uchun metrikalarni alohida kuzatish uchun teglardan foydalaning.`,
            whyItMatters: `Timerlar ishlash monitoringi uchun zarur. Ular sekin operatsiyalarni aniqlashga, SLA bajarilishini kuzatishga va yuklanish ostida tizim xatti-harakatlarini tushunishga yordam beradi. Timerni to'g'ri ishlatish ishlab chiqarishda kuzatish uchun juda muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class PaymentService {
    private final Timer paymentTimer;

    public PaymentService(MeterRegistry registry) {
        this.paymentTimer = Timer.builder("payment.processing.time")
            .description("To'lov qayta ishlash davomiyligi")
            .tags("service", "payment")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);
    }

    public Payment processPayment(PaymentRequest request) {
        return paymentTimer.record(() -> {
            // To'lovni qayta ishlash
            return paymentGateway.charge(request);
        });
    }
}
\`\`\`

**Amaliy foydalari:**
- Muhim operatsiyalar bajarilish vaqtini o'lchash
- SLA bajarilishini kuzatish (99% < 500ms)
- Ishlash tor joylarini aniqlash`
        }
    }
};

export default task;
