import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-modern-switch-expressions',
    title: 'Modern Switch Expressions (Java 14+)',
    difficulty: 'medium',
    tags: ['java', 'syntax', 'switch', 'expressions', 'pattern-matching'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn modern Java switch expressions with arrow syntax and pattern matching.

**Requirements:**
1. Create a \`classicSwitch(String day)\` method using traditional switch statements
2. Implement a \`modernSwitchExpression(String day)\` method using switch expressions with arrow syntax
3. Create a \`switchWithYield(int month)\` method demonstrating the yield keyword
4. Implement a \`switchPatternMatching(Object obj)\` method showing basic pattern matching

**Modern Switch Features:**
- Arrow syntax (->)
- Switch expressions that return values
- Multiple case labels
- yield keyword for complex logic
- Pattern matching (Java 17+)
- No fall-through by default

**Example Output:**
\`\`\`
Monday -> Weekday
Season for month 3: Spring
Type: Integer with value 42
\`\`\``,
    initialCode: `public class SwitchExpressionsDemo {

    public static void classicSwitch(String day) {
        // TODO: Use traditional switch with break statements


    }

    public static String modernSwitchExpression(String day) {
        // TODO: Use modern switch expression with arrow syntax


    }

    public static String switchWithYield(int month) {
        // TODO: Use yield keyword for complex logic


    }

    public static void switchPatternMatching(Object obj) {
        // TODO: Demonstrate pattern matching with switch


    }

    public static void main(String[] args) {
        classicSwitch("Monday");
        System.out.println(modernSwitchExpression("Saturday"));
        System.out.println(switchWithYield(3));
        switchPatternMatching(42);
        switchPatternMatching("Hello");
        switchPatternMatching(3.14);
    }
}`,
    solutionCode: `public class SwitchExpressionsDemo {

    public static void classicSwitch(String day) {
        System.out.println("=== Classic Switch Statement ===");
        String dayType;

        // Traditional switch with break statements
        switch (day) {
            case "Monday":
            case "Tuesday":
            case "Wednesday":
            case "Thursday":
            case "Friday":
                dayType = "Weekday";
                break;
            case "Saturday":
            case "Sunday":
                dayType = "Weekend";
                break;
            default:
                dayType = "Invalid day";
                break;
        }

        System.out.println(day + " is a " + dayType);
    }

    public static String modernSwitchExpression(String day) {
        System.out.println("");
        System.out.println("=== Modern Switch Expression ===");

        // Modern switch expression with arrow syntax (no break needed)
        String dayType = switch (day) {
            case "Monday", "Tuesday", "Wednesday", "Thursday", "Friday" -> "Weekday";
            case "Saturday", "Sunday" -> "Weekend";
            default -> "Invalid day";
        };

        System.out.println(day + " -> " + dayType);
        return dayType;
    }

    public static String switchWithYield(int month) {
        System.out.println("");
        System.out.println("=== Switch with Yield Keyword ===");

        // Using yield for complex logic within switch
        String season = switch (month) {
            case 12, 1, 2 -> {
                System.out.println("Cold months");
                yield "Winter";
            }
            case 3, 4, 5 -> {
                System.out.println("Blooming months");
                yield "Spring";
            }
            case 6, 7, 8 -> {
                System.out.println("Hot months");
                yield "Summer";
            }
            case 9, 10, 11 -> {
                System.out.println("Falling leaves");
                yield "Fall";
            }
            default -> {
                System.out.println("Invalid month");
                yield "Unknown";
            }
        };

        System.out.println("Season for month " + month + ": " + season);
        return season;
    }

    public static void switchPatternMatching(Object obj) {
        System.out.println("");
        System.out.println("=== Pattern Matching with Switch ===");

        // Pattern matching in switch (Java 17+ preview, Java 21 final)
        String result = switch (obj) {
            case Integer i -> "Integer with value: " + i;
            case String s -> "String with length: " + s.length() + " (\"" + s + "\")";
            case Double d -> "Double with value: " + d;
            case null -> "Null value";
            default -> "Unknown type: " + obj.getClass().getSimpleName();
        };

        System.out.println("Type: " + result);
    }

    // Additional example: Calculator using switch expressions
    public static double calculator(double a, double b, String operator) {
        System.out.println("");
        System.out.println("=== Calculator Example ===");

        double result = switch (operator) {
            case "+" -> a + b;
            case "-" -> a - b;
            case "*" -> a * b;
            case "/" -> {
                if (b == 0) {
                    System.out.println("Error: Division by zero");
                    yield 0.0;
                }
                yield a / b;
            }
            case "%" -> a % b;
            default -> {
                System.out.println("Unknown operator: " + operator);
                yield 0.0;
            }
        };

        System.out.println(a + " " + operator + " " + b + " = " + result);
        return result;
    }

    public static void main(String[] args) {
        classicSwitch("Monday");
        modernSwitchExpression("Saturday");
        switchWithYield(3);
        switchPatternMatching(42);
        switchPatternMatching("Hello");
        switchPatternMatching(3.14);
        calculator(10, 5, "+");
        calculator(10, 0, "/");
    }
}`,
    hint1: `Modern switch expressions use -> instead of : and don't need break statements. They can return values directly, making code more concise.`,
    hint2: `Use yield keyword when you need multiple statements in a case block. Pattern matching allows you to test the type and extract values in one step.`,
    whyItMatters: `Modern switch expressions eliminate common bugs like fall-through errors and make code more readable. Pattern matching reduces boilerplate code and makes type checking safer. These features represent Java's evolution toward more expressive and safer syntax.

**Production Pattern:**
\`\`\`java
// Processing different event types with pattern matching
public String processEvent(Event event) {
    return switch (event) {
        case OrderCreated e -> {
            notifyUser(e.getUserId());
            yield "Order " + e.getOrderId() + " created";
        }
        case PaymentProcessed e -> handlePayment(e);
        case ShipmentDispatched e -> trackShipment(e);
        default -> "Unknown event type";
    };
}
\`\`\`

**Practical Benefits:**
- Safe handling of polymorphic types without instanceof
- Compact code for state machines and event handling
- Compiler checks for exhaustive case coverage`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: Test modern switch with weekday
class Test1 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.modernSwitchExpression("Monday");
        assertEquals("Weekday", result);
    }
}

// Test2: Test modern switch with weekend
class Test2 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.modernSwitchExpression("Saturday");
        assertEquals("Weekend", result);
    }
}

// Test3: Test switch with yield for Winter
class Test3 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.switchWithYield(12);
        assertEquals("Winter", result);
    }
}

// Test4: Test switch with yield for Spring
class Test4 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.switchWithYield(3);
        assertEquals("Spring", result);
    }
}

// Test5: Test switch with yield for Summer
class Test5 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.switchWithYield(7);
        assertEquals("Summer", result);
    }
}

// Test6: Test switch with yield for Fall
class Test6 {
    @Test
    public void test() {
        String result = SwitchExpressionsDemo.switchWithYield(10);
        assertEquals("Fall", result);
    }
}

// Test7: Test pattern matching with Integer
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SwitchExpressionsDemo.switchPatternMatching(42);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Integer' or 'Целое' or 'Butun'",
            output.contains("Integer") || output.contains("Целое") || output.contains("Butun"));
        assertTrue("Output should contain '42'", output.contains("42"));
    }
}

// Test8: Test pattern matching with String
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        SwitchExpressionsDemo.switchPatternMatching("Hello");
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'String' or 'Строка' or 'String'",
            output.contains("String") || output.contains("Строка"));
        assertTrue("Output should contain 'Hello'", output.contains("Hello"));
    }
}

// Test9: Test calculator with addition
class Test9 {
    @Test
    public void test() {
        double result = SwitchExpressionsDemo.calculator(10, 5, "+");
        assertEquals(15.0, result, 0.001);
    }
}

// Test10: Test calculator with division by zero
class Test10 {
    @Test
    public void test() {
        double result = SwitchExpressionsDemo.calculator(10, 0, "/");
        assertEquals(0.0, result, 0.001);
    }
}
`,
    translations: {
        ru: {
            title: 'Современные switch-выражения (Java 14+)',
            solutionCode: `public class SwitchExpressionsDemo {

    public static void classicSwitch(String day) {
        System.out.println("=== Классический оператор Switch ===");
        String dayType;

        // Традиционный switch с операторами break
        switch (day) {
            case "Monday":
            case "Tuesday":
            case "Wednesday":
            case "Thursday":
            case "Friday":
                dayType = "Будний день";
                break;
            case "Saturday":
            case "Sunday":
                dayType = "Выходной";
                break;
            default:
                dayType = "Неверный день";
                break;
        }

        System.out.println(day + " это " + dayType);
    }

    public static String modernSwitchExpression(String day) {
        System.out.println("");
        System.out.println("=== Современное switch-выражение ===");

        // Современное switch-выражение со стрелочным синтаксисом (break не нужен)
        String dayType = switch (day) {
            case "Monday", "Tuesday", "Wednesday", "Thursday", "Friday" -> "Будний день";
            case "Saturday", "Sunday" -> "Выходной";
            default -> "Неверный день";
        };

        System.out.println(day + " -> " + dayType);
        return dayType;
    }

    public static String switchWithYield(int month) {
        System.out.println("");
        System.out.println("=== Switch с ключевым словом Yield ===");

        // Использование yield для сложной логики внутри switch
        String season = switch (month) {
            case 12, 1, 2 -> {
                System.out.println("Холодные месяцы");
                yield "Зима";
            }
            case 3, 4, 5 -> {
                System.out.println("Месяцы цветения");
                yield "Весна";
            }
            case 6, 7, 8 -> {
                System.out.println("Жаркие месяцы");
                yield "Лето";
            }
            case 9, 10, 11 -> {
                System.out.println("Падающие листья");
                yield "Осень";
            }
            default -> {
                System.out.println("Неверный месяц");
                yield "Неизвестно";
            }
        };

        System.out.println("Сезон для месяца " + month + ": " + season);
        return season;
    }

    public static void switchPatternMatching(Object obj) {
        System.out.println("");
        System.out.println("=== Сопоставление с образцом в Switch ===");

        // Сопоставление с образцом в switch (Java 17+ preview, Java 21 final)
        String result = switch (obj) {
            case Integer i -> "Целое число со значением: " + i;
            case String s -> "Строка с длиной: " + s.length() + " (\"" + s + "\")";
            case Double d -> "Число с плавающей точкой: " + d;
            case null -> "Нулевое значение";
            default -> "Неизвестный тип: " + obj.getClass().getSimpleName();
        };

        System.out.println("Тип: " + result);
    }

    // Дополнительный пример: Калькулятор используя switch-выражения
    public static double calculator(double a, double b, String operator) {
        System.out.println("");
        System.out.println("=== Пример калькулятора ===");

        double result = switch (operator) {
            case "+" -> a + b;
            case "-" -> a - b;
            case "*" -> a * b;
            case "/" -> {
                if (b == 0) {
                    System.out.println("Ошибка: Деление на ноль");
                    yield 0.0;
                }
                yield a / b;
            }
            case "%" -> a % b;
            default -> {
                System.out.println("Неизвестный оператор: " + operator);
                yield 0.0;
            }
        };

        System.out.println(a + " " + operator + " " + b + " = " + result);
        return result;
    }

    public static void main(String[] args) {
        classicSwitch("Monday");
        modernSwitchExpression("Saturday");
        switchWithYield(3);
        switchPatternMatching(42);
        switchPatternMatching("Hello");
        switchPatternMatching(3.14);
        calculator(10, 5, "+");
        calculator(10, 0, "/");
    }
}`,
            description: `Изучите современные switch-выражения Java со стрелочным синтаксисом и сопоставлением с образцом.

**Требования:**
1. Создайте метод \`classicSwitch(String day)\`, использующий традиционные switch-операторы
2. Реализуйте метод \`modernSwitchExpression(String day)\`, использующий switch-выражения со стрелочным синтаксисом
3. Создайте метод \`switchWithYield(int month)\`, демонстрирующий ключевое слово yield
4. Реализуйте метод \`switchPatternMatching(Object obj)\`, показывающий базовое сопоставление с образцом

**Возможности современного Switch:**
- Стрелочный синтаксис (->)
- Switch-выражения, возвращающие значения
- Множественные метки case
- Ключевое слово yield для сложной логики
- Сопоставление с образцом (Java 17+)
- Отсутствие провала по умолчанию

**Пример вывода:**
\`\`\`
Monday -> Будний день
Сезон для месяца 3: Весна
Тип: Целое число со значением 42
\`\`\``,
            hint1: `Современные switch-выражения используют -> вместо : и не требуют операторов break. Они могут возвращать значения напрямую, делая код более лаконичным.`,
            hint2: `Используйте ключевое слово yield, когда вам нужно несколько операторов в блоке case. Сопоставление с образцом позволяет проверить тип и извлечь значения за один шаг.`,
            whyItMatters: `Современные switch-выражения устраняют распространенные ошибки, такие как провалы, и делают код более читаемым. Сопоставление с образцом уменьшает шаблонный код и делает проверку типов безопаснее. Эти функции представляют эволюцию Java к более выразительному и безопасному синтаксису.

**Продакшен паттерн:**
\`\`\`java
// Обработка различных типов событий с pattern matching
public String processEvent(Event event) {
    return switch (event) {
        case OrderCreated e -> {
            notifyUser(e.getUserId());
            yield "Order " + e.getOrderId() + " created";
        }
        case PaymentProcessed e -> handlePayment(e);
        case ShipmentDispatched e -> trackShipment(e);
        default -> "Unknown event type";
    };
}
\`\`\`

**Практические преимущества:**
- Безопасная обработка полиморфных типов без instanceof
- Компактный код для state machines и event handling
- Компилятор проверяет полноту покрытия случаев`
        },
        uz: {
            title: `Zamonaviy switch ifodalari (Java 14+)`,
            solutionCode: `public class SwitchExpressionsDemo {

    public static void classicSwitch(String day) {
        System.out.println("=== Klassik Switch operatori ===");
        String dayType;

        // An'anaviy switch break operatorlari bilan
        switch (day) {
            case "Monday":
            case "Tuesday":
            case "Wednesday":
            case "Thursday":
            case "Friday":
                dayType = "Ish kuni";
                break;
            case "Saturday":
            case "Sunday":
                dayType = "Dam olish kuni";
                break;
            default:
                dayType = "Noto'g'ri kun";
                break;
        }

        System.out.println(day + " bu " + dayType);
    }

    public static String modernSwitchExpression(String day) {
        System.out.println("");
        System.out.println("=== Zamonaviy switch ifodasi ===");

        // Zamonaviy switch ifodasi o'q sintaksisi bilan (break kerak emas)
        String dayType = switch (day) {
            case "Monday", "Tuesday", "Wednesday", "Thursday", "Friday" -> "Ish kuni";
            case "Saturday", "Sunday" -> "Dam olish kuni";
            default -> "Noto'g'ri kun";
        };

        System.out.println(day + " -> " + dayType);
        return dayType;
    }

    public static String switchWithYield(int month) {
        System.out.println("");
        System.out.println("=== Yield kalit so'zi bilan Switch ===");

        // Switch ichida murakkab mantiq uchun yield dan foydalanish
        String season = switch (month) {
            case 12, 1, 2 -> {
                System.out.println("Sovuq oylar");
                yield "Qish";
            }
            case 3, 4, 5 -> {
                System.out.println("Gullaydigan oylar");
                yield "Bahor";
            }
            case 6, 7, 8 -> {
                System.out.println("Issiq oylar");
                yield "Yoz";
            }
            case 9, 10, 11 -> {
                System.out.println("Barglar to'kilmoqda");
                yield "Kuz";
            }
            default -> {
                System.out.println("Noto'g'ri oy");
                yield "Noma'lum";
            }
        };

        System.out.println(month + "-oy uchun fasl: " + season);
        return season;
    }

    public static void switchPatternMatching(Object obj) {
        System.out.println("");
        System.out.println("=== Switch da naqsh moslashtirish ===");

        // Switch da naqsh moslashtirish (Java 17+ preview, Java 21 final)
        String result = switch (obj) {
            case Integer i -> "Butun son qiymati: " + i;
            case String s -> "String uzunligi: " + s.length() + " (\"" + s + "\")";
            case Double d -> "Kasr son qiymati: " + d;
            case null -> "Null qiymat";
            default -> "Noma'lum tur: " + obj.getClass().getSimpleName();
        };

        System.out.println("Tur: " + result);
    }

    // Qo'shimcha misol: Switch ifodalari yordamida kalkulyator
    public static double calculator(double a, double b, String operator) {
        System.out.println("");
        System.out.println("=== Kalkulyator misoli ===");

        double result = switch (operator) {
            case "+" -> a + b;
            case "-" -> a - b;
            case "*" -> a * b;
            case "/" -> {
                if (b == 0) {
                    System.out.println("Xato: Nolga bo'lish");
                    yield 0.0;
                }
                yield a / b;
            }
            case "%" -> a % b;
            default -> {
                System.out.println("Noma'lum operator: " + operator);
                yield 0.0;
            }
        };

        System.out.println(a + " " + operator + " " + b + " = " + result);
        return result;
    }

    public static void main(String[] args) {
        classicSwitch("Monday");
        modernSwitchExpression("Saturday");
        switchWithYield(3);
        switchPatternMatching(42);
        switchPatternMatching("Hello");
        switchPatternMatching(3.14);
        calculator(10, 5, "+");
        calculator(10, 0, "/");
    }
}`,
            description: `O'q sintaksisi va naqsh moslashtirish bilan zamonaviy Java switch ifodalari.

**Talablar:**
1. An'anaviy switch operatorlaridan foydalanadigan \`classicSwitch(String day)\` metodini yarating
2. O'q sintaksisi bilan switch ifodalari ishlatadigan \`modernSwitchExpression(String day)\` metodini yarating
3. yield kalit so'zini ko'rsatadigan \`switchWithYield(int month)\` metodini yarating
4. Asosiy naqsh moslashtirish ko'rsatadigan \`switchPatternMatching(Object obj)\` metodini yarating

**Zamonaviy Switch imkoniyatlari:**
- O'q sintaksisi (->)
- Qiymat qaytaradigan switch ifodalari
- Ko'p case yorliqlari
- Murakkab mantiq uchun yield kalit so'zi
- Naqsh moslashtirish (Java 17+)
- Standart holatda tushib ketish yo'q

**Chiqish namunasi:**
\`\`\`
Monday -> Ish kuni
3-oy uchun fasl: Bahor
Tur: Butun son qiymati 42
\`\`\``,
            hint1: `Zamonaviy switch ifodalari : o'rniga -> dan foydalanadi va break operatorlari talab qilmaydi. Ular qiymatlarni to'g'ridan-to'g'ri qaytarishi mumkin, bu kodni qisqaroq qiladi.`,
            hint2: `Case blokida bir nechta operatorlar kerak bo'lganda yield kalit so'zidan foydalaning. Naqsh moslashtirish turni tekshirish va qiymatlarni bir qadamda olish imkonini beradi.`,
            whyItMatters: `Zamonaviy switch ifodalari tushib ketish kabi keng tarqalgan xatolarni bartaraf qiladi va kodni o'qishni osonlashtiradi. Naqsh moslashtirish shablon kodini kamaytiradi va tur tekshirishni xavfsizroq qiladi. Bu xususiyatlar Java ning yanada ifodali va xavfsiz sintaksisga evolyutsiyasini ifodalaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Naqsh moslashtirish bilan turli hodisalar turlarini qayta ishlash
public String processEvent(Event event) {
    return switch (event) {
        case OrderCreated e -> {
            notifyUser(e.getUserId());
            yield "Order " + e.getOrderId() + " created";
        }
        case PaymentProcessed e -> handlePayment(e);
        case ShipmentDispatched e -> trackShipment(e);
        default -> "Unknown event type";
    };
}
\`\`\`

**Amaliy foydalari:**
- instanceof siz polimorfik turlarni xavfsiz qayta ishlash
- State machines va event handling uchun ixcham kod
- Kompilyator holatlar qamrovining to'liqligini tekshiradi`
        }
    }
};

export default task;
