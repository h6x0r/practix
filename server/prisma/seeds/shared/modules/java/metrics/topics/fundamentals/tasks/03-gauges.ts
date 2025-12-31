import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-gauges',
    title: 'Gauge Metrics',
    difficulty: 'medium',
    tags: ['java', 'metrics', 'micrometer', 'gauge'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Gauge metrics for tracking current values.

**Requirements:**
1. Create a MeterRegistry
2. Create a List to track active connections
3. Register a Gauge that reports list.size() as "connections.active"
4. Simulate adding 5 connections to the list
5. Print the gauge value
6. Remove 2 connections from the list
7. Print the updated gauge value
8. Create a Gauge that tracks a custom object's property
9. Create a Temperature class with celsius field
10. Register a Gauge that converts celsius to fahrenheit

Gauge represents a current value that can go up or down, like active connections, memory usage, or queue size. Unlike counters, gauges can decrease.`,
    initialCode: `import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.ArrayList;
import java.util.List;

public class GaugeMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry

        // Create a List to track active connections

        // Register a Gauge for list.size()

        // Simulate adding connections

        // Create Temperature class and gauge
    }

    static class Temperature {
        // Implement Temperature class
    }
}`,
    solutionCode: `import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.ArrayList;
import java.util.List;

public class GaugeMetrics {
    public static void main(String[] args) {
        // Create a MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Create a List to track active connections
        List<String> activeConnections = new ArrayList<>();

        // Register a Gauge that reports list.size() as "connections.active"
        Gauge.builder("connections.active", activeConnections, List::size)
            .description("Number of active connections")
            .register(registry);

        // Simulate adding 5 connections to the list
        System.out.println("Adding 5 connections...");
        for (int i = 1; i <= 5; i++) {
            activeConnections.add("Connection-" + i);
        }

        // Print the gauge value
        Gauge gauge = registry.find("connections.active").gauge();
        System.out.println("Active connections: " + gauge.value());

        // Remove 2 connections from the list
        System.out.println("\\nRemoving 2 connections...");
        activeConnections.remove(0);
        activeConnections.remove(0);

        // Print the updated gauge value
        System.out.println("Active connections: " + gauge.value());

        // Create a Gauge that tracks a custom object's property
        Temperature temp = new Temperature(25.0);

        // Register a Gauge that converts celsius to fahrenheit
        Gauge.builder("temperature.fahrenheit", temp, t -> t.getCelsius() * 9.0 / 5.0 + 32.0)
            .description("Current temperature in Fahrenheit")
            .register(registry);

        System.out.println("\\nTemperature: " + temp.getCelsius() + "°C");
        Gauge tempGauge = registry.find("temperature.fahrenheit").gauge();
        System.out.println("Temperature: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Update temperature
        temp.setCelsius(30.0);
        System.out.println("\\nUpdated temperature: " + temp.getCelsius() + "°C");
        System.out.println("Temperature: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Print all metrics
        System.out.println("\\n=== All Gauges ===");
        registry.find("connections.active").gauge().getId();
        System.out.println("connections.active: " + gauge.value());
        System.out.println("temperature.fahrenheit: " + String.format("%.1f", tempGauge.value()));
    }

    static class Temperature {
        private double celsius;

        public Temperature(double celsius) {
            this.celsius = celsius;
        }

        public double getCelsius() {
            return celsius;
        }

        public void setCelsius(double celsius) {
            this.celsius = celsius;
        }
    }
}`,
    hint1: `Use Gauge.builder(name, object, valueFunction) to create a gauge. The valueFunction is called each time the gauge value is read.`,
    hint2: `Gauges don't store values - they compute them on demand by calling the provided function. Make sure the object you're tracking remains in scope.`,
    whyItMatters: `Gauges are essential for monitoring dynamic values like memory usage, queue sizes, and active connections. Understanding how gauges work helps you track the current state of your system effectively.

**Production Pattern:**
\`\`\`java
@Component
public class ConnectionPoolMetrics {
    private final DataSource dataSource;

    public ConnectionPoolMetrics(MeterRegistry registry,
                                 HikariDataSource dataSource) {
        this.dataSource = dataSource;

        Gauge.builder("db.connections.active",
                dataSource, ds -> ds.getHikariPoolMXBean().getActiveConnections())
            .description("Active database connections")
            .register(registry);

        Gauge.builder("db.connections.idle",
                dataSource, ds -> ds.getHikariPoolMXBean().getIdleConnections())
            .description("Idle database connections")
            .register(registry);
    }
}
\`\`\`

**Practical Benefits:**
- Monitor connection pool resources
- Detect connection leaks
- Optimize pool size based on metrics`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.*;

// Test 1: Gauge is created with list size
class Test1 {
    @Test
    void testGaugeCreatedWithListSize() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        Gauge.builder("connections.active", list, List::size).register(registry);
        Gauge gauge = registry.find("connections.active").gauge();
        assertNotNull(gauge);
    }
}

// Test 2: Gauge reflects list size
class Test2 {
    @Test
    void testGaugeReflectsListSize() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        Gauge.builder("connections.active", list, List::size).register(registry);
        Gauge gauge = registry.find("connections.active").gauge();
        assertEquals(0.0, gauge.value(), 0.01);
    }
}

// Test 3: Gauge updates when list grows
class Test3 {
    @Test
    void testGaugeUpdatesWhenListGrows() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        Gauge.builder("connections.active", list, List::size).register(registry);
        Gauge gauge = registry.find("connections.active").gauge();

        list.add("conn1");
        list.add("conn2");
        assertEquals(2.0, gauge.value(), 0.01);
    }
}

// Test 4: Gauge updates when list shrinks
class Test4 {
    @Test
    void testGaugeUpdatesWhenListShrinks() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        list.add("conn1");
        list.add("conn2");
        list.add("conn3");
        Gauge.builder("connections.active", list, List::size).register(registry);
        Gauge gauge = registry.find("connections.active").gauge();

        list.remove(0);
        assertEquals(2.0, gauge.value(), 0.01);
    }
}

// Test 5: Gauge with 5 items
class Test5 {
    @Test
    void testGaugeWithFiveItems() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        Gauge.builder("connections.active", list, List::size).register(registry);
        Gauge gauge = registry.find("connections.active").gauge();

        for (int i = 0; i < 5; i++) {
            list.add("Connection-" + i);
        }
        assertEquals(5.0, gauge.value(), 0.01);
    }
}

// Test 6: Gauge with custom function
class Test6 {
    @Test
    void testGaugeWithCustomFunction() {
        MeterRegistry registry = new SimpleMeterRegistry();
        double[] value = {25.0};
        Gauge.builder("temperature", value, v -> v[0] * 9.0 / 5.0 + 32.0).register(registry);
        Gauge gauge = registry.find("temperature").gauge();
        assertEquals(77.0, gauge.value(), 0.01);
    }
}

// Test 7: Gauge value changes dynamically
class Test7 {
    @Test
    void testGaugeValueChangesDynamically() {
        MeterRegistry registry = new SimpleMeterRegistry();
        double[] value = {20.0};
        Gauge.builder("value", value, v -> v[0]).register(registry);
        Gauge gauge = registry.find("value").gauge();

        assertEquals(20.0, gauge.value(), 0.01);
        value[0] = 30.0;
        assertEquals(30.0, gauge.value(), 0.01);
    }
}

// Test 8: Gauge with description
class Test8 {
    @Test
    void testGaugeWithDescription() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        Gauge.builder("connections.active", list, List::size)
            .description("Number of active connections")
            .register(registry);
        Gauge gauge = registry.find("connections.active").gauge();
        assertNotNull(gauge.getId().getDescription());
    }
}

// Test 9: Multiple gauges work independently
class Test9 {
    @Test
    void testMultipleGaugesIndependent() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list1 = new ArrayList<>();
        List<String> list2 = new ArrayList<>();

        Gauge.builder("gauge1", list1, List::size).register(registry);
        Gauge.builder("gauge2", list2, List::size).register(registry);

        list1.add("a");
        list2.add("b");
        list2.add("c");

        Gauge g1 = registry.find("gauge1").gauge();
        Gauge g2 = registry.find("gauge2").gauge();

        assertEquals(1.0, g1.value(), 0.01);
        assertEquals(2.0, g2.value(), 0.01);
    }
}

// Test 10: Gauge can decrease
class Test10 {
    @Test
    void testGaugeCanDecrease() {
        MeterRegistry registry = new SimpleMeterRegistry();
        List<String> list = new ArrayList<>();
        list.add("a");
        list.add("b");
        list.add("c");
        Gauge.builder("gauge", list, List::size).register(registry);
        Gauge gauge = registry.find("gauge").gauge();

        assertEquals(3.0, gauge.value(), 0.01);
        list.clear();
        assertEquals(0.0, gauge.value(), 0.01);
    }
}`,
    translations: {
        ru: {
            title: 'Метрики Gauge',
            solutionCode: `import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.ArrayList;
import java.util.List;

public class GaugeMetrics {
    public static void main(String[] args) {
        // Создаем MeterRegistry
        MeterRegistry registry = new SimpleMeterRegistry();

        // Создаем List для отслеживания активных соединений
        List<String> activeConnections = new ArrayList<>();

        // Регистрируем Gauge, который сообщает list.size() как "connections.active"
        Gauge.builder("connections.active", activeConnections, List::size)
            .description("Количество активных соединений")
            .register(registry);

        // Симулируем добавление 5 соединений в список
        System.out.println("Добавление 5 соединений...");
        for (int i = 1; i <= 5; i++) {
            activeConnections.add("Connection-" + i);
        }

        // Выводим значение gauge
        Gauge gauge = registry.find("connections.active").gauge();
        System.out.println("Активных соединений: " + gauge.value());

        // Удаляем 2 соединения из списка
        System.out.println("\\nУдаление 2 соединений...");
        activeConnections.remove(0);
        activeConnections.remove(0);

        // Выводим обновленное значение gauge
        System.out.println("Активных соединений: " + gauge.value());

        // Создаем Gauge, который отслеживает свойство пользовательского объекта
        Temperature temp = new Temperature(25.0);

        // Регистрируем Gauge, который конвертирует цельсии в фаренгейты
        Gauge.builder("temperature.fahrenheit", temp, t -> t.getCelsius() * 9.0 / 5.0 + 32.0)
            .description("Текущая температура в фаренгейтах")
            .register(registry);

        System.out.println("\\nТемпература: " + temp.getCelsius() + "°C");
        Gauge tempGauge = registry.find("temperature.fahrenheit").gauge();
        System.out.println("Температура: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Обновляем температуру
        temp.setCelsius(30.0);
        System.out.println("\\nОбновленная температура: " + temp.getCelsius() + "°C");
        System.out.println("Температура: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Выводим все метрики
        System.out.println("\\n=== Все Gauge ===");
        registry.find("connections.active").gauge().getId();
        System.out.println("connections.active: " + gauge.value());
        System.out.println("temperature.fahrenheit: " + String.format("%.1f", tempGauge.value()));
    }

    static class Temperature {
        private double celsius;

        public Temperature(double celsius) {
            this.celsius = celsius;
        }

        public double getCelsius() {
            return celsius;
        }

        public void setCelsius(double celsius) {
            this.celsius = celsius;
        }
    }
}`,
            description: `Освойте метрики Gauge для отслеживания текущих значений.

**Требования:**
1. Создайте MeterRegistry
2. Создайте List для отслеживания активных соединений
3. Зарегистрируйте Gauge, который сообщает list.size() как "connections.active"
4. Симулируйте добавление 5 соединений в список
5. Выведите значение gauge
6. Удалите 2 соединения из списка
7. Выведите обновленное значение gauge
8. Создайте Gauge, который отслеживает свойство пользовательского объекта
9. Создайте класс Temperature с полем celsius
10. Зарегистрируйте Gauge, который конвертирует цельсии в фаренгейты

Gauge представляет текущее значение, которое может увеличиваться или уменьшаться, например активные соединения, использование памяти или размер очереди. В отличие от счетчиков, gauge может уменьшаться.`,
            hint1: `Используйте Gauge.builder(name, object, valueFunction) для создания gauge. valueFunction вызывается каждый раз при чтении значения gauge.`,
            hint2: `Gauge не хранят значения - они вычисляют их по требованию, вызывая предоставленную функцию. Убедитесь, что отслеживаемый объект остается в области видимости.`,
            whyItMatters: `Gauge необходимы для мониторинга динамических значений, таких как использование памяти, размеры очередей и активные соединения. Понимание работы gauge помогает эффективно отслеживать текущее состояние системы.

**Продакшен паттерн:**
\`\`\`java
@Component
public class ConnectionPoolMetrics {
    private final DataSource dataSource;

    public ConnectionPoolMetrics(MeterRegistry registry,
                                 HikariDataSource dataSource) {
        this.dataSource = dataSource;

        Gauge.builder("db.connections.active",
                dataSource, ds -> ds.getHikariPoolMXBean().getActiveConnections())
            .description("Active database connections")
            .register(registry);

        Gauge.builder("db.connections.idle",
                dataSource, ds -> ds.getHikariPoolMXBean().getIdleConnections())
            .description("Idle database connections")
            .register(registry);
    }
}
\`\`\`

**Практические преимущества:**
- Мониторинг ресурсов пула соединений
- Обнаружение утечек соединений
- Оптимизация размера пула на основе метрик`
        },
        uz: {
            title: 'Gauge Metrikalari',
            solutionCode: `import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.ArrayList;
import java.util.List;

public class GaugeMetrics {
    public static void main(String[] args) {
        // MeterRegistry yaratamiz
        MeterRegistry registry = new SimpleMeterRegistry();

        // Faol ulanishlarni kuzatish uchun List yaratamiz
        List<String> activeConnections = new ArrayList<>();

        // list.size() ni "connections.active" sifatida xabar qiluvchi Gauge ro'yxatdan o'tkazamiz
        Gauge.builder("connections.active", activeConnections, List::size)
            .description("Faol ulanishlar soni")
            .register(registry);

        // Ro'yxatga 5 ta ulanish qo'shishni simulyatsiya qilamiz
        System.out.println("5 ta ulanish qo'shilmoqda...");
        for (int i = 1; i <= 5; i++) {
            activeConnections.add("Connection-" + i);
        }

        // Gauge qiymatini chiqaramiz
        Gauge gauge = registry.find("connections.active").gauge();
        System.out.println("Faol ulanishlar: " + gauge.value());

        // Ro'yxatdan 2 ta ulanishni o'chiramiz
        System.out.println("\\n2 ta ulanish o'chirilmoqda...");
        activeConnections.remove(0);
        activeConnections.remove(0);

        // Yangilangan gauge qiymatini chiqaramiz
        System.out.println("Faol ulanishlar: " + gauge.value());

        // Maxsus obyektning xususiyatini kuzatuvchi Gauge yaratamiz
        Temperature temp = new Temperature(25.0);

        // Selsiyni farengeytga aylantiruvchi Gauge ro'yxatdan o'tkazamiz
        Gauge.builder("temperature.fahrenheit", temp, t -> t.getCelsius() * 9.0 / 5.0 + 32.0)
            .description("Joriy harorat Farengeyt bo'yicha")
            .register(registry);

        System.out.println("\\nHarorat: " + temp.getCelsius() + "°C");
        Gauge tempGauge = registry.find("temperature.fahrenheit").gauge();
        System.out.println("Harorat: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Haroratni yangilaymiz
        temp.setCelsius(30.0);
        System.out.println("\\nYangilangan harorat: " + temp.getCelsius() + "°C");
        System.out.println("Harorat: " + String.format("%.1f", tempGauge.value()) + "°F");

        // Barcha metrikalarni chiqaramiz
        System.out.println("\\n=== Barcha Gauge'lar ===");
        registry.find("connections.active").gauge().getId();
        System.out.println("connections.active: " + gauge.value());
        System.out.println("temperature.fahrenheit: " + String.format("%.1f", tempGauge.value()));
    }

    static class Temperature {
        private double celsius;

        public Temperature(double celsius) {
            this.celsius = celsius;
        }

        public double getCelsius() {
            return celsius;
        }

        public void setCelsius(double celsius) {
            this.celsius = celsius;
        }
    }
}`,
            description: `Joriy qiymatlarni kuzatish uchun Gauge metrikalarini o'zlashtirig.

**Talablar:**
1. MeterRegistry yarating
2. Faol ulanishlarni kuzatish uchun List yarating
3. list.size() ni "connections.active" sifatida xabar qiluvchi Gauge ro'yxatdan o'tkazing
4. Ro'yxatga 5 ta ulanish qo'shishni simulyatsiya qiling
5. Gauge qiymatini chiqaring
6. Ro'yxatdan 2 ta ulanishni o'chiring
7. Yangilangan gauge qiymatini chiqaring
8. Maxsus obyektning xususiyatini kuzatuvchi Gauge yarating
9. celsius maydoni bilan Temperature klassini yarating
10. Selsiyni farengeytga aylantiruvchi Gauge ro'yxatdan o'tkazing

Gauge faol ulanishlar, xotira ishlatilishi yoki navbat hajmi kabi ortishi yoki kamayishi mumkin bo'lgan joriy qiymatni ifodalaydi. Hisoblagichlardan farqli o'laroq, gauge kamayishi mumkin.`,
            hint1: `Gauge yaratish uchun Gauge.builder(name, object, valueFunction) dan foydalaning. valueFunction har safar gauge qiymati o'qilganda chaqiriladi.`,
            hint2: `Gauge'lar qiymatlarni saqlamaydi - ular taqdim etilgan funksiyani chaqirish orqali ularni talab bo'yicha hisoblaydi. Kuzatayotgan obyekt ko'rinish doirasida qolishiga ishonch hosil qiling.`,
            whyItMatters: `Gauge'lar xotira ishlatilishi, navbat hajmlari va faol ulanishlar kabi dinamik qiymatlarni kuzatish uchun zarur. Gauge'lar qanday ishlashini tushunish tizimingizning joriy holatini samarali kuzatishga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Component
public class ConnectionPoolMetrics {
    private final DataSource dataSource;

    public ConnectionPoolMetrics(MeterRegistry registry,
                                 HikariDataSource dataSource) {
        this.dataSource = dataSource;

        Gauge.builder("db.connections.active",
                dataSource, ds -> ds.getHikariPoolMXBean().getActiveConnections())
            .description("Faol ma'lumotlar bazasi ulanishlari")
            .register(registry);

        Gauge.builder("db.connections.idle",
                dataSource, ds -> ds.getHikariPoolMXBean().getIdleConnections())
            .description("Bo'sh ma'lumotlar bazasi ulanishlari")
            .register(registry);
    }
}
\`\`\`

**Amaliy foydalari:**
- Ulanishlar puli resurslarini monitoring qilish
- Ulanish oqib ketishini aniqlash
- Metrikalar asosida pul hajmini optimallashtirish`
        }
    }
};

export default task;
