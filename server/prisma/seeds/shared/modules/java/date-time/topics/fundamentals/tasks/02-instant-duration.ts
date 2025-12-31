import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-instant-duration',
    title: 'Instant, Duration, Period',
    difficulty: 'easy',
    tags: ['java', 'date-time', 'instant', 'duration', 'period', 'temporal'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Instant, Duration, Period

Instant represents a point in time on the timeline (epoch timestamp), Duration represents time-based amounts (hours, minutes, seconds), and Period represents date-based amounts (years, months, days).

## Requirements:
1. Work with Instant:
   1. Get current instant (epoch timestamp)
   2. Create instant from epoch seconds
   3. Convert to/from LocalDateTime
   4. Calculate time differences

2. Use Duration for time-based amounts:
   1. Create durations (hours, minutes, seconds)
   2. Calculate duration between times
   3. Add/subtract durations
   4. Get duration components

3. Use Period for date-based amounts:
   1. Create periods (years, months, days)
   2. Calculate period between dates
   3. Add/subtract periods
   4. Get period components

4. Demonstrate practical use cases (elapsed time, age calculation)

## Example Output:
\`\`\`
=== Instant Examples ===
Current instant: 2024-12-10T14:30:45.123Z
Epoch seconds: 1702217445
From epoch: 2024-12-10T14:30:45Z
Time difference: 3600 seconds

=== Duration Examples ===
2 hours: PT2H
90 minutes: PT1H30M
Between times: PT5H30M
Total seconds: 19800
Hours: 5, Minutes: 30

=== Period Examples ===
1 year: P1Y
6 months: P6M
Between dates: P2Y3M15D
Years: 2, Months: 3, Days: 15
Age: 25 years old
\`\`\``,
    initialCode: `// TODO: Import java.time classes

public class InstantDurationPeriod {
    public static void main(String[] args) {
        // TODO: Work with Instant

        // TODO: Work with Duration

        // TODO: Work with Period
    }
}`,
    solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;

public class InstantDurationPeriod {
    public static void main(String[] args) {
        System.out.println("=== Instant Examples ===");

        // Get current instant (point in time)
        Instant now = Instant.now();
        System.out.println("Current instant: " + now);

        // Get epoch seconds
        long epochSeconds = now.getEpochSecond();
        System.out.println("Epoch seconds: " + epochSeconds);

        // Create instant from epoch
        Instant fromEpoch = Instant.ofEpochSecond(epochSeconds);
        System.out.println("From epoch: " + fromEpoch);

        // Calculate time difference
        Instant oneHourAgo = now.minus(1, ChronoUnit.HOURS);
        long secondsDiff = ChronoUnit.SECONDS.between(oneHourAgo, now);
        System.out.println("Time difference: " + secondsDiff + " seconds");

        System.out.println("\\n=== Duration Examples ===");

        // Create durations
        Duration twoHours = Duration.ofHours(2);
        System.out.println("2 hours: " + twoHours);

        Duration ninetyMinutes = Duration.ofMinutes(90);
        System.out.println("90 minutes: " + ninetyMinutes);

        // Duration between times
        LocalTime startTime = LocalTime.of(9, 0);
        LocalTime endTime = LocalTime.of(14, 30);
        Duration timeDiff = Duration.between(startTime, endTime);
        System.out.println("Between times: " + timeDiff);

        // Get duration components
        long totalSeconds = timeDiff.getSeconds();
        long hours = timeDiff.toHours();
        long minutes = timeDiff.toMinutes() % 60;
        System.out.println("Total seconds: " + totalSeconds);
        System.out.println("Hours: " + hours + ", Minutes: " + minutes);

        System.out.println("\\n=== Period Examples ===");

        // Create periods
        Period oneYear = Period.ofYears(1);
        System.out.println("1 year: " + oneYear);

        Period sixMonths = Period.ofMonths(6);
        System.out.println("6 months: " + sixMonths);

        // Period between dates
        LocalDate startDate = LocalDate.of(2020, 3, 15);
        LocalDate endDate = LocalDate.of(2022, 6, 30);
        Period dateDiff = Period.between(startDate, endDate);
        System.out.println("Between dates: " + dateDiff);

        // Get period components
        System.out.println("Years: " + dateDiff.getYears() +
            ", Months: " + dateDiff.getMonths() +
            ", Days: " + dateDiff.getDays());

        // Calculate age
        LocalDate birthDate = LocalDate.of(1999, 5, 15);
        LocalDate today = LocalDate.now();
        Period age = Period.between(birthDate, today);
        System.out.println("Age: " + age.getYears() + " years old");

        // Add period to date
        LocalDate futureDate = today.plus(Period.ofMonths(3));
        System.out.println("3 months from now: " + futureDate);
    }
}`,
    hint1: `Instant represents a point in time (timestamp). Use Instant.now() for current time and Duration.between() for time differences.`,
    hint2: `Duration is for time-based amounts (hours, minutes, seconds), while Period is for date-based amounts (years, months, days). Use the appropriate one based on your needs.`,
    whyItMatters: `Instant, Duration, and Period are essential for measuring and calculating time intervals. Instant provides machine-readable timestamps, Duration handles time-based calculations (like elapsed time), and Period handles date-based calculations (like age). These classes are crucial for performance monitoring, scheduling, and business logic involving time spans.

**Production Pattern:**
\`\`\`java
// Performance measurement
Instant start = Instant.now();
performHeavyOperation();
Instant end = Instant.now();
Duration elapsed = Duration.between(start, end);
System.out.println("Operation took: " + elapsed.toMillis() + "ms");

// Age calculation
LocalDate birthDate = LocalDate.of(1990, 5, 15);
Period age = Period.between(birthDate, LocalDate.now());
System.out.println("Age: " + age.getYears() + " years");
\`\`\`

**Practical Benefits:**
- Accurate performance measurement with nanosecond precision
- Business logic calculations (age, expiration dates)
- Type safety: Duration for time, Period for dates`,
    order: 2,
    testCode: `import java.time.*;
import java.time.temporal.ChronoUnit;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: Instant.now() returns current instant
class Test1 {
    @Test
    void testInstantNow() {
        Instant now = Instant.now();
        assertNotNull(now);
        assertTrue(now.getEpochSecond() > 0);
    }
}

// Test2: Instant.ofEpochSecond() creates from epoch
class Test2 {
    @Test
    void testInstantOfEpoch() {
        long epoch = 1702217445;
        Instant instant = Instant.ofEpochSecond(epoch);
        assertEquals(epoch, instant.getEpochSecond());
    }
}

// Test3: Duration.ofHours() creates hour duration
class Test3 {
    @Test
    void testDurationOfHours() {
        Duration twoHours = Duration.ofHours(2);
        assertEquals(2, twoHours.toHours());
        assertEquals(7200, twoHours.getSeconds());
    }
}

// Test4: Duration.ofMinutes() creates minute duration
class Test4 {
    @Test
    void testDurationOfMinutes() {
        Duration ninetyMinutes = Duration.ofMinutes(90);
        assertEquals(90, ninetyMinutes.toMinutes());
        assertEquals(1, ninetyMinutes.toHours());
    }
}

// Test5: Duration.between() calculates time difference
class Test5 {
    @Test
    void testDurationBetween() {
        LocalTime start = LocalTime.of(9, 0);
        LocalTime end = LocalTime.of(14, 30);
        Duration duration = Duration.between(start, end);
        assertEquals(5, duration.toHours());
        assertEquals(30, duration.toMinutes() % 60);
    }
}

// Test6: Period.ofYears() creates year period
class Test6 {
    @Test
    void testPeriodOfYears() {
        Period oneYear = Period.ofYears(1);
        assertEquals(1, oneYear.getYears());
        assertEquals(0, oneYear.getMonths());
    }
}

// Test7: Period.between() calculates date difference
class Test7 {
    @Test
    void testPeriodBetween() {
        LocalDate start = LocalDate.of(2020, 1, 1);
        LocalDate end = LocalDate.of(2022, 6, 15);
        Period period = Period.between(start, end);
        assertEquals(2, period.getYears());
        assertEquals(5, period.getMonths());
    }
}

// Test8: Duration components are correct
class Test8 {
    @Test
    void testDurationComponents() {
        Duration duration = Duration.ofSeconds(3661);
        assertEquals(1, duration.toHours());
        assertEquals(61, duration.toMinutes() - 60);
    }
}

// Test9: Period addition works correctly
class Test9 {
    @Test
    void testPeriodAddition() {
        LocalDate today = LocalDate.of(2024, 1, 15);
        LocalDate future = today.plus(Period.ofMonths(3));
        assertEquals(LocalDate.of(2024, 4, 15), future);
    }
}

// Test10: ChronoUnit calculates between instants
class Test10 {
    @Test
    void testChronoUnitBetween() {
        Instant now = Instant.now();
        Instant later = now.plus(1, ChronoUnit.HOURS);
        long seconds = ChronoUnit.SECONDS.between(now, later);
        assertEquals(3600, seconds);
    }
}
`,
    translations: {
        ru: {
            title: 'Instant, Duration, Period',
            solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;

public class InstantDurationPeriod {
    public static void main(String[] args) {
        System.out.println("=== Примеры Instant ===");

        // Получить текущий момент времени
        Instant now = Instant.now();
        System.out.println("Current instant: " + now);

        // Получить эпоху в секундах
        long epochSeconds = now.getEpochSecond();
        System.out.println("Epoch seconds: " + epochSeconds);

        // Создать момент из эпохи
        Instant fromEpoch = Instant.ofEpochSecond(epochSeconds);
        System.out.println("From epoch: " + fromEpoch);

        // Вычислить разницу во времени
        Instant oneHourAgo = now.minus(1, ChronoUnit.HOURS);
        long secondsDiff = ChronoUnit.SECONDS.between(oneHourAgo, now);
        System.out.println("Time difference: " + secondsDiff + " seconds");

        System.out.println("\\n=== Примеры Duration ===");

        // Создать длительности
        Duration twoHours = Duration.ofHours(2);
        System.out.println("2 hours: " + twoHours);

        Duration ninetyMinutes = Duration.ofMinutes(90);
        System.out.println("90 minutes: " + ninetyMinutes);

        // Длительность между временами
        LocalTime startTime = LocalTime.of(9, 0);
        LocalTime endTime = LocalTime.of(14, 30);
        Duration timeDiff = Duration.between(startTime, endTime);
        System.out.println("Between times: " + timeDiff);

        // Получить компоненты длительности
        long totalSeconds = timeDiff.getSeconds();
        long hours = timeDiff.toHours();
        long minutes = timeDiff.toMinutes() % 60;
        System.out.println("Total seconds: " + totalSeconds);
        System.out.println("Hours: " + hours + ", Minutes: " + minutes);

        System.out.println("\\n=== Примеры Period ===");

        // Создать периоды
        Period oneYear = Period.ofYears(1);
        System.out.println("1 year: " + oneYear);

        Period sixMonths = Period.ofMonths(6);
        System.out.println("6 months: " + sixMonths);

        // Период между датами
        LocalDate startDate = LocalDate.of(2020, 3, 15);
        LocalDate endDate = LocalDate.of(2022, 6, 30);
        Period dateDiff = Period.between(startDate, endDate);
        System.out.println("Between dates: " + dateDiff);

        // Получить компоненты периода
        System.out.println("Years: " + dateDiff.getYears() +
            ", Months: " + dateDiff.getMonths() +
            ", Days: " + dateDiff.getDays());

        // Вычислить возраст
        LocalDate birthDate = LocalDate.of(1999, 5, 15);
        LocalDate today = LocalDate.now();
        Period age = Period.between(birthDate, today);
        System.out.println("Age: " + age.getYears() + " years old");

        // Добавить период к дате
        LocalDate futureDate = today.plus(Period.ofMonths(3));
        System.out.println("3 months from now: " + futureDate);
    }
}`,
            description: `# Instant, Duration, Period

Instant представляет точку во времени на временной шкале (метка времени эпохи), Duration представляет временные величины (часы, минуты, секунды), а Period представляет величины на основе дат (годы, месяцы, дни).

## Требования:
1. Работа с Instant:
   1. Получить текущий момент времени (метка времени эпохи)
   2. Создать момент из секунд эпохи
   3. Преобразовать в/из LocalDateTime
   4. Вычислить разницу во времени

2. Использовать Duration для временных величин:
   1. Создать длительности (часы, минуты, секунды)
   2. Вычислить длительность между временами
   3. Добавить/вычесть длительности
   4. Получить компоненты длительности

3. Использовать Period для величин на основе дат:
   1. Создать периоды (годы, месяцы, дни)
   2. Вычислить период между датами
   3. Добавить/вычесть периоды
   4. Получить компоненты периода

4. Продемонстрировать практические случаи использования (прошедшее время, вычисление возраста)

## Пример вывода:
\`\`\`
=== Instant Examples ===
Current instant: 2024-12-10T14:30:45.123Z
Epoch seconds: 1702217445
From epoch: 2024-12-10T14:30:45Z
Time difference: 3600 seconds

=== Duration Examples ===
2 hours: PT2H
90 minutes: PT1H30M
Between times: PT5H30M
Total seconds: 19800
Hours: 5, Minutes: 30

=== Period Examples ===
1 year: P1Y
6 months: P6M
Between dates: P2Y3M15D
Years: 2, Months: 3, Days: 15
Age: 25 years old
\`\`\``,
            hint1: `Instant представляет точку во времени (метка времени). Используйте Instant.now() для текущего времени и Duration.between() для разницы во времени.`,
            hint2: `Duration предназначен для временных величин (часы, минуты, секунды), а Period для величин на основе дат (годы, месяцы, дни). Используйте подходящий в зависимости от ваших потребностей.`,
            whyItMatters: `Instant, Duration и Period необходимы для измерения и вычисления временных интервалов. Instant предоставляет машиночитаемые метки времени, Duration обрабатывает вычисления на основе времени (например, прошедшее время), а Period обрабатывает вычисления на основе дат (например, возраст). Эти классы имеют решающее значение для мониторинга производительности, планирования и бизнес-логики, связанной с временными промежутками.

**Продакшен паттерн:**
\`\`\`java
// Измерение производительности
Instant start = Instant.now();
performHeavyOperation();
Instant end = Instant.now();
Duration elapsed = Duration.between(start, end);
System.out.println("Операция заняла: " + elapsed.toMillis() + "ms");

// Вычисление возраста
LocalDate birthDate = LocalDate.of(1990, 5, 15);
Period age = Period.between(birthDate, LocalDate.now());
System.out.println("Возраст: " + age.getYears() + " лет");
\`\`\`

**Практические преимущества:**
- Точное измерение производительности с наносекундной точностью
- Расчеты на основе бизнес-логики (возраст, срок действия)
- Безопасность типов: Duration для времени, Period для дат`
        },
        uz: {
            title: `Instant, Duration, Period`,
            solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;

public class InstantDurationPeriod {
    public static void main(String[] args) {
        System.out.println("=== Instant namunalari ===");

        // Joriy vaqt nuqtasini olish
        Instant now = Instant.now();
        System.out.println("Current instant: " + now);

        // Epoch soniyalarini olish
        long epochSeconds = now.getEpochSecond();
        System.out.println("Epoch seconds: " + epochSeconds);

        // Epochdan vaqt nuqtasini yaratish
        Instant fromEpoch = Instant.ofEpochSecond(epochSeconds);
        System.out.println("From epoch: " + fromEpoch);

        // Vaqt farqini hisoblash
        Instant oneHourAgo = now.minus(1, ChronoUnit.HOURS);
        long secondsDiff = ChronoUnit.SECONDS.between(oneHourAgo, now);
        System.out.println("Time difference: " + secondsDiff + " seconds");

        System.out.println("\\n=== Duration namunalari ===");

        // Davomiyliklarni yaratish
        Duration twoHours = Duration.ofHours(2);
        System.out.println("2 hours: " + twoHours);

        Duration ninetyMinutes = Duration.ofMinutes(90);
        System.out.println("90 minutes: " + ninetyMinutes);

        // Vaqtlar orasidagi davomiylik
        LocalTime startTime = LocalTime.of(9, 0);
        LocalTime endTime = LocalTime.of(14, 30);
        Duration timeDiff = Duration.between(startTime, endTime);
        System.out.println("Between times: " + timeDiff);

        // Davomiylik komponentlarini olish
        long totalSeconds = timeDiff.getSeconds();
        long hours = timeDiff.toHours();
        long minutes = timeDiff.toMinutes() % 60;
        System.out.println("Total seconds: " + totalSeconds);
        System.out.println("Hours: " + hours + ", Minutes: " + minutes);

        System.out.println("\\n=== Period namunalari ===");

        // Davrlarni yaratish
        Period oneYear = Period.ofYears(1);
        System.out.println("1 year: " + oneYear);

        Period sixMonths = Period.ofMonths(6);
        System.out.println("6 months: " + sixMonths);

        // Sanalar orasidagi davr
        LocalDate startDate = LocalDate.of(2020, 3, 15);
        LocalDate endDate = LocalDate.of(2022, 6, 30);
        Period dateDiff = Period.between(startDate, endDate);
        System.out.println("Between dates: " + dateDiff);

        // Davr komponentlarini olish
        System.out.println("Years: " + dateDiff.getYears() +
            ", Months: " + dateDiff.getMonths() +
            ", Days: " + dateDiff.getDays());

        // Yoshni hisoblash
        LocalDate birthDate = LocalDate.of(1999, 5, 15);
        LocalDate today = LocalDate.now();
        Period age = Period.between(birthDate, today);
        System.out.println("Age: " + age.getYears() + " years old");

        // Sanaga davrni qo'shish
        LocalDate futureDate = today.plus(Period.ofMonths(3));
        System.out.println("3 months from now: " + futureDate);
    }
}`,
            description: `# Instant, Duration, Period

Instant vaqt chizig'idagi vaqt nuqtasini (epoch timestamp) ifodalaydi, Duration vaqtga asoslangan miqdorlarni (soat, daqiqa, soniya) ifodalaydi va Period sanaga asoslangan miqdorlarni (yil, oy, kun) ifodalaydi.

## Talablar:
1. Instant bilan ishlash:
   1. Joriy vaqt nuqtasini olish (epoch timestamp)
   2. Epoch soniyalaridan vaqt nuqtasini yaratish
   3. LocalDateTime ga/dan konvertatsiya qilish
   4. Vaqt farqini hisoblash

2. Vaqtga asoslangan miqdorlar uchun Duration dan foydalanish:
   1. Davomiyliklarni yaratish (soat, daqiqa, soniya)
   2. Vaqtlar orasidagi davomiylikni hisoblash
   3. Davomiyliklarni qo'shish/ayirish
   4. Davomiylik komponentlarini olish

3. Sanaga asoslangan miqdorlar uchun Period dan foydalanish:
   1. Davrlarni yaratish (yil, oy, kun)
   2. Sanalar orasidagi davrni hisoblash
   3. Davrlarni qo'shish/ayirish
   4. Davr komponentlarini olish

4. Amaliy foydalanish holatlarini ko'rsatish (o'tgan vaqt, yoshni hisoblash)

## Chiqish namunasi:
\`\`\`
=== Instant Examples ===
Current instant: 2024-12-10T14:30:45.123Z
Epoch seconds: 1702217445
From epoch: 2024-12-10T14:30:45Z
Time difference: 3600 seconds

=== Duration Examples ===
2 hours: PT2H
90 minutes: PT1H30M
Between times: PT5H30M
Total seconds: 19800
Hours: 5, Minutes: 30

=== Period Examples ===
1 year: P1Y
6 months: P6M
Between dates: P2Y3M15D
Years: 2, Months: 3, Days: 15
Age: 25 years old
\`\`\``,
            hint1: `Instant vaqt nuqtasini (timestamp) ifodalaydi. Joriy vaqt uchun Instant.now() va vaqt farqi uchun Duration.between() dan foydalaning.`,
            hint2: `Duration vaqtga asoslangan miqdorlar (soat, daqiqa, soniya) uchun, Period esa sanaga asoslangan miqdorlar (yil, oy, kun) uchun. Ehtiyojlaringizga qarab tegishli birini ishlating.`,
            whyItMatters: `Instant, Duration va Period vaqt intervallarini o'lchash va hisoblash uchun zarurdir. Instant mashinaga o'qiladigan vaqt belgilarini taqdim etadi, Duration vaqtga asoslangan hisoblarni (masalan, o'tgan vaqt) boshqaradi va Period sanaga asoslangan hisoblarni (masalan, yosh) boshqaradi. Bu klasslar ishlash monitoringi, rejalashtirish va vaqt oralig'i bilan bog'liq biznes mantiq uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ishlashni o'lchash
Instant start = Instant.now();
performHeavyOperation();
Instant end = Instant.now();
Duration elapsed = Duration.between(start, end);
System.out.println("Operatsiya vaqti: " + elapsed.toMillis() + "ms");

// Yoshni hisoblash
LocalDate birthDate = LocalDate.of(1990, 5, 15);
Period age = Period.between(birthDate, LocalDate.now());
System.out.println("Yosh: " + age.getYears() + " yil");
\`\`\`

**Amaliy foydalari:**
- Nanosekundli aniqlik bilan ishlashni o'lchash
- Biznes mantiqiga asoslangan hisoblar (yosh, amal qilish muddati)
- Tur xavfsizligi: vaqt uchun Duration, sanalar uchun Period`
        }
    }
};

export default task;
