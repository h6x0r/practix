import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-micrometer-basics',
    title: 'Micrometer Basics',
    difficulty: 'easy',
    tags: ['java', 'metrics', 'micrometer', 'monitoring'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn Micrometer basics for application metrics.

**Requirements:**
1. Create a SimpleMeterRegistry
2. Create a Counter with name "requests.total"
3. Increment the counter 5 times
4. Create a Counter with tags: "endpoint=/api", "method=GET"
5. Increment the tagged counter 3 times
6. Print all metrics using registry.getMeters()
7. Print counter values using counter.count()
8. Demonstrate tag filtering

Micrometer is a vendor-neutral application metrics facade that supports multiple monitoring systems like Prometheus, Grafana, and New Relic.`,
    initialCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class MicrometerBasics {
    public static void main(String[] args) {
        // Create a SimpleMeterRegistry

        // Create a Counter with name "requests.total"

        // Increment the counter 5 times

        // Create a Counter with tags

        // Increment the tagged counter 3 times

        // Print all metrics

        // Print counter values
    }
}`,
    solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class MicrometerBasics {
    public static void main(String[] args) {
        // Create a SimpleMeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Create a Counter with name "requests.total"
        Counter requestCounter = registry.counter("requests.total");

        // Increment the counter 5 times
        for (int i = 0; i < 5; i++) {
            requestCounter.increment();
        }
        System.out.println("Total requests: " + requestCounter.count());

        // Create a Counter with tags: "endpoint=/api", "method=GET"
        Counter apiCounter = Counter.builder("api.requests")
            .tag("endpoint", "/api")
            .tag("method", "GET")
            .register(registry);

        // Increment the tagged counter 3 times
        for (int i = 0; i < 3; i++) {
            apiCounter.increment();
        }
        System.out.println("API requests: " + apiCounter.count());

        // Print all metrics using registry.getMeters()
        System.out.println("\\nAll Metrics:");
        registry.getMeters().forEach(meter -> {
            System.out.println("  - " + meter.getId().getName() +
                             " [" + meter.getId().getTags() + "]");
        });

        // Print counter values
        System.out.println("\\nCounter Values:");
        System.out.println("requests.total: " + requestCounter.count());
        System.out.println("api.requests (GET /api): " + apiCounter.count());

        // Demonstrate tag filtering
        System.out.println("\\nMetrics with 'api' prefix:");
        registry.find("api.requests").counters()
            .forEach(c -> System.out.println("  - " + c.getId()));
    }
}`,
    hint1: `Use SimpleMeterRegistry as your MeterRegistry implementation. Counter.builder() allows you to add tags to your metrics.`,
    hint2: `The registry.getMeters() method returns all registered metrics. Use registry.find() to search for specific metrics by name.`,
    whyItMatters: `Micrometer provides a vendor-neutral way to collect application metrics. Understanding the MeterRegistry and basic Counter usage is fundamental for implementing effective monitoring.

**Production Pattern:**
\`\`\`java
@Service
public class MetricsService {
    private final MeterRegistry registry;
    private final Counter requestCounter;

    public MetricsService(MeterRegistry registry) {
        this.registry = registry;
        this.requestCounter = registry.counter("http.requests",
            "application", "payment-service");
    }

    public void recordRequest() {
        requestCounter.increment();
    }
}
\`\`\`

**Practical Benefits:**
- Centralized metrics collection for monitoring
- Integration with Prometheus, Grafana, and other systems
- Real-time traffic analysis capabilities`,
    order: 0,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

// Test 1: SimpleMeterRegistry is created
class Test1 {
    @Test
    void testSimpleMeterRegistryCreated() {
        MeterRegistry registry = new SimpleMeterRegistry();
        assertNotNull(registry);
    }
}

// Test 2: Counter is created with name
class Test2 {
    @Test
    void testCounterCreatedWithName() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = registry.counter("requests.total");
        assertNotNull(counter);
        assertEquals("requests.total", counter.getId().getName());
    }
}

// Test 3: Counter increments correctly
class Test3 {
    @Test
    void testCounterIncrements() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = registry.counter("requests.total");
        counter.increment();
        counter.increment();
        assertEquals(2.0, counter.count(), 0.01);
    }
}

// Test 4: Counter increments 5 times
class Test4 {
    @Test
    void testCounterIncrementsMultiple() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = registry.counter("requests.total");
        for (int i = 0; i < 5; i++) {
            counter.increment();
        }
        assertEquals(5.0, counter.count(), 0.01);
    }
}

// Test 5: Counter with tags is created
class Test5 {
    @Test
    void testCounterWithTags() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = Counter.builder("api.requests")
            .tag("endpoint", "/api")
            .tag("method", "GET")
            .register(registry);
        assertNotNull(counter);
    }
}

// Test 6: Counter tags are correct
class Test6 {
    @Test
    void testCounterTagsCorrect() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = Counter.builder("api.requests")
            .tag("method", "GET")
            .register(registry);
        assertTrue(counter.getId().getTags().stream()
            .anyMatch(t -> t.getKey().equals("method") && t.getValue().equals("GET")));
    }
}

// Test 7: Registry getMeters returns all meters
class Test7 {
    @Test
    void testRegistryGetMeters() {
        MeterRegistry registry = new SimpleMeterRegistry();
        registry.counter("counter1");
        registry.counter("counter2");
        assertTrue(registry.getMeters().size() >= 2);
    }
}

// Test 8: Registry find returns counter
class Test8 {
    @Test
    void testRegistryFind() {
        MeterRegistry registry = new SimpleMeterRegistry();
        registry.counter("test.counter");
        Counter found = registry.find("test.counter").counter();
        assertNotNull(found);
    }
}

// Test 9: Multiple counters are independent
class Test9 {
    @Test
    void testMultipleCountersIndependent() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter c1 = registry.counter("counter1");
        Counter c2 = registry.counter("counter2");
        c1.increment();
        c1.increment();
        c2.increment();
        assertEquals(2.0, c1.count(), 0.01);
        assertEquals(1.0, c2.count(), 0.01);
    }
}

// Test 10: Same name returns same counter
class Test10 {
    @Test
    void testSameNameReturnsSameCounter() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter c1 = registry.counter("my.counter");
        Counter c2 = registry.counter("my.counter");
        c1.increment();
        assertEquals(1.0, c2.count(), 0.01);
    }
}`,
    translations: {
        ru: {
            title: 'Основы Micrometer',
            solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class MicrometerBasics {
    public static void main(String[] args) {
        // Создаем SimpleMeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Создаем Counter с именем "requests.total"
        Counter requestCounter = registry.counter("requests.total");

        // Увеличиваем счетчик 5 раз
        for (int i = 0; i < 5; i++) {
            requestCounter.increment();
        }
        System.out.println("Всего запросов: " + requestCounter.count());

        // Создаем Counter с тегами: "endpoint=/api", "method=GET"
        Counter apiCounter = Counter.builder("api.requests")
            .tag("endpoint", "/api")
            .tag("method", "GET")
            .register(registry);

        // Увеличиваем счетчик с тегами 3 раза
        for (int i = 0; i < 3; i++) {
            apiCounter.increment();
        }
        System.out.println("API запросов: " + apiCounter.count());

        // Выводим все метрики через registry.getMeters()
        System.out.println("\\nВсе метрики:");
        registry.getMeters().forEach(meter -> {
            System.out.println("  - " + meter.getId().getName() +
                             " [" + meter.getId().getTags() + "]");
        });

        // Выводим значения счетчиков
        System.out.println("\\nЗначения счетчиков:");
        System.out.println("requests.total: " + requestCounter.count());
        System.out.println("api.requests (GET /api): " + apiCounter.count());

        // Демонстрируем фильтрацию по тегам
        System.out.println("\\nМетрики с префиксом 'api':");
        registry.find("api.requests").counters()
            .forEach(c -> System.out.println("  - " + c.getId()));
    }
}`,
            description: `Изучите основы Micrometer для метрик приложения.

**Требования:**
1. Создайте SimpleMeterRegistry
2. Создайте Counter с именем "requests.total"
3. Увеличьте счетчик 5 раз
4. Создайте Counter с тегами: "endpoint=/api", "method=GET"
5. Увеличьте счетчик с тегами 3 раза
6. Выведите все метрики используя registry.getMeters()
7. Выведите значения счетчиков используя counter.count()
8. Продемонстрируйте фильтрацию по тегам

Micrometer - это универсальный фасад для метрик приложений, поддерживающий множество систем мониторинга, таких как Prometheus, Grafana и New Relic.`,
            hint1: `Используйте SimpleMeterRegistry в качестве реализации MeterRegistry. Counter.builder() позволяет добавлять теги к метрикам.`,
            hint2: `Метод registry.getMeters() возвращает все зарегистрированные метрики. Используйте registry.find() для поиска конкретных метрик по имени.`,
            whyItMatters: `Micrometer предоставляет универсальный способ сбора метрик приложения. Понимание MeterRegistry и базового использования Counter имеет основополагающее значение для реализации эффективного мониторинга.

**Продакшен паттерн:**
\`\`\`java
@Service
public class MetricsService {
    private final MeterRegistry registry;
    private final Counter requestCounter;

    public MetricsService(MeterRegistry registry) {
        this.registry = registry;
        this.requestCounter = registry.counter("http.requests",
            "application", "payment-service");
    }

    public void recordRequest() {
        requestCounter.increment();
    }
}
\`\`\`

**Практические преимущества:**
- Централизованный сбор метрик для мониторинга
- Интеграция с Prometheus, Grafana и другими системами
- Возможность анализа трафика в реальном времени`
        },
        uz: {
            title: 'Micrometer Asoslari',
            solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class MicrometerBasics {
    public static void main(String[] args) {
        // SimpleMeterRegistry yaratamiz
        MeterRegistry registry = new SimpleMeterRegistry();

        // "requests.total" nomi bilan Counter yaratamiz
        Counter requestCounter = registry.counter("requests.total");

        // Hisoblagichni 5 marta oshiramiz
        for (int i = 0; i < 5; i++) {
            requestCounter.increment();
        }
        System.out.println("Jami so'rovlar: " + requestCounter.count());

        // Teglar bilan Counter yaratamiz: "endpoint=/api", "method=GET"
        Counter apiCounter = Counter.builder("api.requests")
            .tag("endpoint", "/api")
            .tag("method", "GET")
            .register(registry);

        // Tegli hisoblagichni 3 marta oshiramiz
        for (int i = 0; i < 3; i++) {
            apiCounter.increment();
        }
        System.out.println("API so'rovlar: " + apiCounter.count());

        // registry.getMeters() orqali barcha metrikalarni chiqaramiz
        System.out.println("\\nBarcha metrikalar:");
        registry.getMeters().forEach(meter -> {
            System.out.println("  - " + meter.getId().getName() +
                             " [" + meter.getId().getTags() + "]");
        });

        // Hisoblagich qiymatlarini chiqaramiz
        System.out.println("\\nHisoblagich qiymatlari:");
        System.out.println("requests.total: " + requestCounter.count());
        System.out.println("api.requests (GET /api): " + apiCounter.count());

        // Teglar bo'yicha filtrlashni ko'rsatamiz
        System.out.println("\\n'api' prefiksi bilan metrikalar:");
        registry.find("api.requests").counters()
            .forEach(c -> System.out.println("  - " + c.getId()));
    }
}`,
            description: `Dastur metrikalari uchun Micrometer asoslarini o'rganing.

**Talablar:**
1. SimpleMeterRegistry yarating
2. "requests.total" nomi bilan Counter yarating
3. Hisoblagichni 5 marta oshiring
4. Teglar bilan Counter yarating: "endpoint=/api", "method=GET"
5. Tegli hisoblagichni 3 marta oshiring
6. registry.getMeters() yordamida barcha metrikalarni chiqaring
7. counter.count() yordamida hisoblagich qiymatlarini chiqaring
8. Teglar bo'yicha filtrlashni ko'rsating

Micrometer - bu Prometheus, Grafana va New Relic kabi ko'plab monitoring tizimlarini qo'llab-quvvatlaydigan vendor-neutral dastur metrikalari interfeysi.`,
            hint1: `MeterRegistry implementatsiyasi sifatida SimpleMeterRegistry dan foydalaning. Counter.builder() metrikalarga teglar qo'shishga imkon beradi.`,
            hint2: `registry.getMeters() metodi barcha ro'yxatga olingan metrikalarni qaytaradi. registry.find() dan nom bo'yicha aniq metrikalarni qidirish uchun foydalaning.`,
            whyItMatters: `Micrometer dastur metrikalarini yig'ishning universal usulini taqdim etadi. MeterRegistry va Counter asosiy foydalanishini tushunish samarali monitoringni amalga oshirish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class MetricsService {
    private final MeterRegistry registry;
    private final Counter requestCounter;

    public MetricsService(MeterRegistry registry) {
        this.registry = registry;
        this.requestCounter = registry.counter("http.requests",
            "application", "payment-service");
    }

    public void recordRequest() {
        requestCounter.increment();
    }
}
\`\`\`

**Amaliy foydalari:**
- Monitoring uchun markazlashgan metrikalar yig'ish
- Prometheus, Grafana va boshqa tizimlar bilan integratsiya
- Real vaqt rejimida trafik tahlili imkoniyati`
        }
    }
};

export default task;
