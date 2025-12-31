import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-counters',
    title: 'Counter Metrics',
    difficulty: 'easy',
    tags: ['java', 'metrics', 'micrometer', 'counter'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Counter metrics for tracking events.

**Requirements:**
1. Create a MeterRegistry
2. Create counters for: "orders.created", "orders.failed", "orders.cancelled"
3. Simulate 10 order creations (increment orders.created)
4. Simulate 2 failed orders (increment orders.failed)
5. Simulate 1 cancelled order (increment orders.cancelled)
6. Create a counter that increments by custom amounts
7. Use Counter.increment(double amount) to add 5.5 to a "revenue" counter
8. Print all counter values and calculate success rate

Counter is a cumulative metric that represents a single monotonically increasing value. It's ideal for counting events like requests, errors, or completed tasks.`,
    initialCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class CounterMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry

        // Create counters for orders

        // Simulate order operations

        // Create revenue counter

        // Print results and calculate success rate
    }
}`,
    solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class CounterMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Create counters for: "orders.created", "orders.failed", "orders.cancelled"
        Counter ordersCreated = registry.counter("orders.created");
        Counter ordersFailed = registry.counter("orders.failed");
        Counter ordersCancelled = registry.counter("orders.cancelled");

        // Simulate 10 order creations
        System.out.println("Creating 10 orders...");
        for (int i = 0; i < 10; i++) {
            ordersCreated.increment();
        }

        // Simulate 2 failed orders
        System.out.println("Processing 2 failed orders...");
        for (int i = 0; i < 2; i++) {
            ordersFailed.increment();
        }

        // Simulate 1 cancelled order
        System.out.println("Processing 1 cancelled order...");
        ordersCancelled.increment();

        // Create a counter that increments by custom amounts
        Counter revenueCounter = registry.counter("revenue.total");

        // Use Counter.increment(double amount) to add 5.5
        System.out.println("\\nAdding revenue...");
        revenueCounter.increment(5.5);
        revenueCounter.increment(10.25);
        revenueCounter.increment(3.75);

        // Print all counter values
        System.out.println("\\n=== Order Statistics ===");
        System.out.println("Orders Created: " + ordersCreated.count());
        System.out.println("Orders Failed: " + ordersFailed.count());
        System.out.println("Orders Cancelled: " + ordersCancelled.count());
        System.out.println("Total Revenue: $" + String.format("%.2f", revenueCounter.count()));

        // Calculate success rate
        double totalOrders = ordersCreated.count();
        double failedOrders = ordersFailed.count() + ordersCancelled.count();
        double successRate = ((totalOrders - failedOrders) / totalOrders) * 100;
        System.out.println("\\nSuccess Rate: " + String.format("%.1f%%", successRate));
    }
}`,
    hint1: `Use registry.counter(name) to create simple counters. Call increment() without arguments to add 1, or increment(amount) to add a custom value.`,
    hint2: `Counter.count() returns the current value as a double. You can use multiple counters to track different aspects of your system.`,
    whyItMatters: `Counters are the most fundamental metric type. They're essential for tracking events, errors, and throughput in production systems. Understanding how to use counters effectively is crucial for observability.

**Production Pattern:**
\`\`\`java
@RestController
public class OrderController {
    private final Counter ordersCreated;
    private final Counter ordersFailed;

    public OrderController(MeterRegistry registry) {
        this.ordersCreated = registry.counter("orders.created");
        this.ordersFailed = registry.counter("orders.failed",
            "reason", "validation");
    }

    @PostMapping("/orders")
    public Order createOrder(@RequestBody OrderDto dto) {
        try {
            Order order = orderService.create(dto);
            ordersCreated.increment();
            return order;
        } catch (ValidationException e) {
            ordersFailed.increment();
            throw e;
        }
    }
}
\`\`\`

**Practical Benefits:**
- Track number of operations and errors
- Calculate success rate (SLA)
- Early detection of system anomalies`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

// Test 1: Orders created counter works
class Test1 {
    @Test
    void testOrdersCreatedCounter() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter ordersCreated = registry.counter("orders.created");
        for (int i = 0; i < 10; i++) {
            ordersCreated.increment();
        }
        assertEquals(10.0, ordersCreated.count(), 0.01);
    }
}

// Test 2: Orders failed counter works
class Test2 {
    @Test
    void testOrdersFailedCounter() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter ordersFailed = registry.counter("orders.failed");
        ordersFailed.increment();
        ordersFailed.increment();
        assertEquals(2.0, ordersFailed.count(), 0.01);
    }
}

// Test 3: Orders cancelled counter works
class Test3 {
    @Test
    void testOrdersCancelledCounter() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter ordersCancelled = registry.counter("orders.cancelled");
        ordersCancelled.increment();
        assertEquals(1.0, ordersCancelled.count(), 0.01);
    }
}

// Test 4: Counter increment with amount
class Test4 {
    @Test
    void testCounterIncrementWithAmount() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter revenue = registry.counter("revenue.total");
        revenue.increment(5.5);
        assertEquals(5.5, revenue.count(), 0.01);
    }
}

// Test 5: Multiple increments with amounts
class Test5 {
    @Test
    void testMultipleIncrementsWithAmounts() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter revenue = registry.counter("revenue.total");
        revenue.increment(5.5);
        revenue.increment(10.25);
        revenue.increment(3.75);
        assertEquals(19.5, revenue.count(), 0.01);
    }
}

// Test 6: Counter starts at zero
class Test6 {
    @Test
    void testCounterStartsAtZero() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = registry.counter("new.counter");
        assertEquals(0.0, counter.count(), 0.01);
    }
}

// Test 7: Counter is monotonically increasing
class Test7 {
    @Test
    void testCounterMonotonicallyIncreasing() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = registry.counter("test.counter");
        double prev = counter.count();
        for (int i = 0; i < 5; i++) {
            counter.increment();
            assertTrue(counter.count() > prev);
            prev = counter.count();
        }
    }
}

// Test 8: Success rate calculation
class Test8 {
    @Test
    void testSuccessRateCalculation() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter created = registry.counter("orders.created");
        Counter failed = registry.counter("orders.failed");

        for (int i = 0; i < 10; i++) created.increment();
        for (int i = 0; i < 2; i++) failed.increment();

        double successRate = ((created.count() - failed.count()) / created.count()) * 100;
        assertEquals(80.0, successRate, 0.01);
    }
}

// Test 9: Counter with description
class Test9 {
    @Test
    void testCounterWithDescription() {
        MeterRegistry registry = new SimpleMeterRegistry();
        Counter counter = Counter.builder("documented.counter")
            .description("A counter with description")
            .register(registry);
        assertNotNull(counter);
    }
}

// Test 10: All counters are tracked
class Test10 {
    @Test
    void testAllCountersTracked() {
        MeterRegistry registry = new SimpleMeterRegistry();
        registry.counter("orders.created");
        registry.counter("orders.failed");
        registry.counter("orders.cancelled");
        registry.counter("revenue.total");
        assertEquals(4, registry.getMeters().size());
    }
}`,
    translations: {
        ru: {
            title: 'Метрики Counter',
            solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class CounterMetrics {
    public static void main(String[] args) {
        // Создаем MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Создаем счетчики для: "orders.created", "orders.failed", "orders.cancelled"
        Counter ordersCreated = registry.counter("orders.created");
        Counter ordersFailed = registry.counter("orders.failed");
        Counter ordersCancelled = registry.counter("orders.cancelled");

        // Симулируем создание 10 заказов
        System.out.println("Создание 10 заказов...");
        for (int i = 0; i < 10; i++) {
            ordersCreated.increment();
        }

        // Симулируем 2 неудачных заказа
        System.out.println("Обработка 2 неудачных заказов...");
        for (int i = 0; i < 2; i++) {
            ordersFailed.increment();
        }

        // Симулируем 1 отмененный заказ
        System.out.println("Обработка 1 отмененного заказа...");
        ordersCancelled.increment();

        // Создаем счетчик с пользовательским приращением
        Counter revenueCounter = registry.counter("revenue.total");

        // Используем Counter.increment(double amount) для добавления 5.5
        System.out.println("\\nДобавление выручки...");
        revenueCounter.increment(5.5);
        revenueCounter.increment(10.25);
        revenueCounter.increment(3.75);

        // Выводим все значения счетчиков
        System.out.println("\\n=== Статистика заказов ===");
        System.out.println("Создано заказов: " + ordersCreated.count());
        System.out.println("Неудачных заказов: " + ordersFailed.count());
        System.out.println("Отмененных заказов: " + ordersCancelled.count());
        System.out.println("Общая выручка: $" + String.format("%.2f", revenueCounter.count()));

        // Вычисляем процент успеха
        double totalOrders = ordersCreated.count();
        double failedOrders = ordersFailed.count() + ordersCancelled.count();
        double successRate = ((totalOrders - failedOrders) / totalOrders) * 100;
        System.out.println("\\nПроцент успеха: " + String.format("%.1f%%", successRate));
    }
}`,
            description: `Освойте метрики Counter для отслеживания событий.

**Требования:**
1. Создайте MeterRegistry
2. Создайте счетчики для: "orders.created", "orders.failed", "orders.cancelled"
3. Симулируйте создание 10 заказов (увеличьте orders.created)
4. Симулируйте 2 неудачных заказа (увеличьте orders.failed)
5. Симулируйте 1 отмененный заказ (увеличьте orders.cancelled)
6. Создайте счетчик с пользовательским приращением
7. Используйте Counter.increment(double amount) чтобы добавить 5.5 к счетчику "revenue"
8. Выведите все значения счетчиков и вычислите процент успеха

Counter - это кумулятивная метрика, представляющая единственное монотонно возрастающее значение. Идеально подходит для подсчета событий, таких как запросы, ошибки или выполненные задачи.`,
            hint1: `Используйте registry.counter(name) для создания простых счетчиков. Вызывайте increment() без аргументов чтобы добавить 1, или increment(amount) для добавления пользовательского значения.`,
            hint2: `Counter.count() возвращает текущее значение как double. Вы можете использовать несколько счетчиков для отслеживания различных аспектов вашей системы.`,
            whyItMatters: `Счетчики - самый фундаментальный тип метрик. Они необходимы для отслеживания событий, ошибок и пропускной способности в производственных системах. Понимание эффективного использования счетчиков критически важно для наблюдаемости.

**Продакшен паттерн:**
\`\`\`java
@RestController
public class OrderController {
    private final Counter ordersCreated;
    private final Counter ordersFailed;

    public OrderController(MeterRegistry registry) {
        this.ordersCreated = registry.counter("orders.created");
        this.ordersFailed = registry.counter("orders.failed",
            "reason", "validation");
    }

    @PostMapping("/orders")
    public Order createOrder(@RequestBody OrderDto dto) {
        try {
            Order order = orderService.create(dto);
            ordersCreated.increment();
            return order;
        } catch (ValidationException e) {
            ordersFailed.increment();
            throw e;
        }
    }
}
\`\`\`

**Практические преимущества:**
- Отслеживание количества операций и ошибок
- Расчет процента успешных операций (SLA)
- Раннее обнаружение аномалий в системе`
        },
        uz: {
            title: 'Counter Metrikalari',
            solutionCode: `import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

public class CounterMetrics {
    public static void main(String[] args) {
        // MeterRegistry yaratamiz
        MeterRegistry registry = new SimpleMeterRegistry();

        // Buyurtmalar uchun hisoblagichlar yaratamiz: "orders.created", "orders.failed", "orders.cancelled"
        Counter ordersCreated = registry.counter("orders.created");
        Counter ordersFailed = registry.counter("orders.failed");
        Counter ordersCancelled = registry.counter("orders.cancelled");

        // 10 ta buyurtma yaratishni simulyatsiya qilamiz
        System.out.println("10 ta buyurtma yaratilmoqda...");
        for (int i = 0; i < 10; i++) {
            ordersCreated.increment();
        }

        // 2 ta muvaffaqiyatsiz buyurtmani simulyatsiya qilamiz
        System.out.println("2 ta muvaffaqiyatsiz buyurtma qayta ishlanmoqda...");
        for (int i = 0; i < 2; i++) {
            ordersFailed.increment();
        }

        // 1 ta bekor qilingan buyurtmani simulyatsiya qilamiz
        System.out.println("1 ta bekor qilingan buyurtma qayta ishlanmoqda...");
        ordersCancelled.increment();

        // Maxsus miqdorda oshiriladigan hisoblagich yaratamiz
        Counter revenueCounter = registry.counter("revenue.total");

        // 5.5 qo'shish uchun Counter.increment(double amount) dan foydalanamiz
        System.out.println("\\nDaromad qo'shilmoqda...");
        revenueCounter.increment(5.5);
        revenueCounter.increment(10.25);
        revenueCounter.increment(3.75);

        // Barcha hisoblagich qiymatlarini chiqaramiz
        System.out.println("\\n=== Buyurtmalar statistikasi ===");
        System.out.println("Yaratilgan buyurtmalar: " + ordersCreated.count());
        System.out.println("Muvaffaqiyatsiz buyurtmalar: " + ordersFailed.count());
        System.out.println("Bekor qilingan buyurtmalar: " + ordersCancelled.count());
        System.out.println("Jami daromad: $" + String.format("%.2f", revenueCounter.count()));

        // Muvaffaqiyat foizini hisoblaymiz
        double totalOrders = ordersCreated.count();
        double failedOrders = ordersFailed.count() + ordersCancelled.count();
        double successRate = ((totalOrders - failedOrders) / totalOrders) * 100;
        System.out.println("\\nMuvaffaqiyat foizi: " + String.format("%.1f%%", successRate));
    }
}`,
            description: `Hodisalarni kuzatish uchun Counter metrikalarini o'zlashtirig.

**Talablar:**
1. MeterRegistry yarating
2. Quyidagilar uchun hisoblagichlar yarating: "orders.created", "orders.failed", "orders.cancelled"
3. 10 ta buyurtma yaratishni simulyatsiya qiling (orders.created ni oshiring)
4. 2 ta muvaffaqiyatsiz buyurtmani simulyatsiya qiling (orders.failed ni oshiring)
5. 1 ta bekor qilingan buyurtmani simulyatsiya qiling (orders.cancelled ni oshiring)
6. Maxsus miqdorda oshiriladigan hisoblagich yarating
7. "revenue" hisoblagichiga 5.5 qo'shish uchun Counter.increment(double amount) dan foydalaning
8. Barcha hisoblagich qiymatlarini chiqaring va muvaffaqiyat foizini hisoblang

Counter - bu yagona monoton o'sadigan qiymatni ifodalovchi kumulyativ metrika. So'rovlar, xatolar yoki bajarilgan vazifalar kabi hodisalarni hisoblash uchun ideal.`,
            hint1: `Oddiy hisoblagichlar yaratish uchun registry.counter(name) dan foydalaning. 1 qo'shish uchun argumentsiz increment() ni chaqiring yoki maxsus qiymat qo'shish uchun increment(amount) ni ishlatng.`,
            hint2: `Counter.count() joriy qiymatni double sifatida qaytaradi. Tizimingizning turli jihatlarini kuzatish uchun bir nechta hisoblagichlardan foydalanishingiz mumkin.`,
            whyItMatters: `Hisoblagichlar eng asosiy metrika turi. Ular ishlab chiqarish tizimlarida hodisalar, xatolar va o'tkazuvchanlikni kuzatish uchun zarur. Hisoblagichlardan samarali foydalanishni tushunish kuzatish uchun juda muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
@RestController
public class OrderController {
    private final Counter ordersCreated;
    private final Counter ordersFailed;

    public OrderController(MeterRegistry registry) {
        this.ordersCreated = registry.counter("orders.created");
        this.ordersFailed = registry.counter("orders.failed",
            "reason", "validation");
    }

    @PostMapping("/orders")
    public Order createOrder(@RequestBody OrderDto dto) {
        try {
            Order order = orderService.create(dto);
            ordersCreated.increment();
            return order;
        } catch (ValidationException e) {
            ordersFailed.increment();
            throw e;
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Operatsiyalar va xatolar sonini kuzatish
- Muvaffaqiyatli operatsiyalar foizini hisoblash (SLA)
- Tizimdagi anomaliyalarni erta aniqlash`
        }
    }
};

export default task;
