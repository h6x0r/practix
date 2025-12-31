import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-static-methods-interfaces',
    title: 'Static Methods in Interfaces',
    difficulty: 'easy',
    tags: ['java', 'interfaces', 'static-methods', 'java8'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Static Methods in Interfaces

Java 8 also introduced static methods in interfaces. These are utility methods that belong to the interface itself, not to implementing classes. They cannot be overridden.

## Requirements:
1. Create a \`MathOperations\` interface with:
   1.1. Abstract method: \`double calculate()\`
   1.2. Static method: \`boolean isValid(double value)\` - checks if not NaN or Infinity
   1.3. Static method: \`double safeCalculate(MathOperations operation, double defaultValue)\` - returns result or default if invalid

2. Create an \`Addition\` class:
   2.1. Implements \`MathOperations\`
   2.2. Has two double fields
   2.3. Implements \`calculate()\` to return sum

3. Create a \`Division\` class:
   3.1. Implements \`MathOperations\`
   3.2. Has dividend and divisor fields
   3.3. Implements \`calculate()\` to return division result

4. In main(), demonstrate:
   4.1. Valid calculations
   4.2. Using static validation method
   4.3. Using static safe calculation method with invalid operation (division by zero)

## Example Output:
\`\`\`
Addition: 10.5 + 5.3 = 15.80
Is valid? true

Division: 20.0 / 4.0 = 5.00
Is valid? true

Division by zero: 10.0 / 0.0 = Infinity
Is valid? false
Safe result: 0.00
\`\`\``,
    initialCode: `// TODO: Create MathOperations interface with static methods

// TODO: Create Addition class

// TODO: Create Division class

public class StaticMethods {
    public static void main(String[] args) {
        // TODO: Test valid operations

        // TODO: Test static validation

        // TODO: Test safe calculation with division by zero
    }
}`,
    solutionCode: `// Interface with abstract and static methods
interface MathOperations {
    // Abstract method to be implemented
    double calculate();

    // Static utility method - belongs to interface, not instances
    static boolean isValid(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    // Static method that uses abstract method through parameter
    static double safeCalculate(MathOperations operation, double defaultValue) {
        double result = operation.calculate();
        return isValid(result) ? result : defaultValue;
    }
}

// Addition implements the interface
class Addition implements MathOperations {
    private double a;
    private double b;

    public Addition(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public double calculate() {
        return a + b;
    }

    @Override
    public String toString() {
        return String.format("%.1f + %.1f", a, b);
    }
}

// Division implements the interface
class Division implements MathOperations {
    private double dividend;
    private double divisor;

    public Division(double dividend, double divisor) {
        this.dividend = dividend;
        this.divisor = divisor;
    }

    @Override
    public double calculate() {
        return dividend / divisor;
    }

    @Override
    public String toString() {
        return String.format("%.1f / %.1f", dividend, divisor);
    }
}

public class StaticMethods {
    public static void main(String[] args) {
        // Test valid addition
        MathOperations add = new Addition(10.5, 5.3);
        double addResult = add.calculate();
        System.out.printf("Addition: %s = %.2f%n", add, addResult);
        System.out.println("Is valid? " + MathOperations.isValid(addResult));

        System.out.println();

        // Test valid division
        MathOperations div = new Division(20.0, 4.0);
        double divResult = div.calculate();
        System.out.printf("Division: %s = %.2f%n", div, divResult);
        System.out.println("Is valid? " + MathOperations.isValid(divResult));

        System.out.println();

        // Test division by zero with static safe method
        MathOperations divByZero = new Division(10.0, 0.0);
        double invalidResult = divByZero.calculate();
        System.out.printf("Division by zero: %s = %s%n", divByZero, invalidResult);
        System.out.println("Is valid? " + MathOperations.isValid(invalidResult));

        // Use static safe calculation
        double safeResult = MathOperations.safeCalculate(divByZero, 0.0);
        System.out.printf("Safe result: %.2f%n", safeResult);
    }
}`,
    hint1: `Static methods in interfaces are called using the interface name: InterfaceName.staticMethod()`,
    hint2: `Static methods cannot access instance methods directly, but they can work with interface instances passed as parameters.`,
    whyItMatters: `Static methods in interfaces provide a place for utility methods closely related to the interface without needing a separate utility class. They're commonly used for factory methods, validators, and helper functions. For example, the Comparator interface has static methods like comparing() and naturalOrder() that create comparator instances.`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test 1: Addition implements MathOperations
class Test1 {
    @Test
    void testAdditionImplementsMathOperations() {
        MathOperations op = new Addition(5.0, 3.0);
        assertNotNull(op);
        assertTrue(op instanceof MathOperations);
    }
}

// Test 2: Division implements MathOperations
class Test2 {
    @Test
    void testDivisionImplementsMathOperations() {
        MathOperations op = new Division(10.0, 2.0);
        assertNotNull(op);
        assertTrue(op instanceof MathOperations);
    }
}

// Test 3: Addition calculate returns sum
class Test3 {
    @Test
    void testAdditionCalculate() {
        Addition add = new Addition(10.5, 5.3);
        assertEquals(15.8, add.calculate(), 0.01);
    }
}

// Test 4: Division calculate returns quotient
class Test4 {
    @Test
    void testDivisionCalculate() {
        Division div = new Division(20.0, 4.0);
        assertEquals(5.0, div.calculate(), 0.01);
    }
}

// Test 5: Static isValid returns true for valid number
class Test5 {
    @Test
    void testIsValidForValidNumber() {
        assertTrue(MathOperations.isValid(10.0));
        assertTrue(MathOperations.isValid(0.0));
        assertTrue(MathOperations.isValid(-5.0));
    }
}

// Test 6: Static isValid returns false for Infinity
class Test6 {
    @Test
    void testIsValidForInfinity() {
        assertFalse(MathOperations.isValid(Double.POSITIVE_INFINITY));
        assertFalse(MathOperations.isValid(Double.NEGATIVE_INFINITY));
    }
}

// Test 7: Static isValid returns false for NaN
class Test7 {
    @Test
    void testIsValidForNaN() {
        assertFalse(MathOperations.isValid(Double.NaN));
    }
}

// Test 8: Division by zero produces Infinity
class Test8 {
    @Test
    void testDivisionByZero() {
        Division divByZero = new Division(10.0, 0.0);
        double result = divByZero.calculate();
        assertTrue(Double.isInfinite(result));
    }
}

// Test 9: Static safeCalculate returns default for invalid result
class Test9 {
    @Test
    void testSafeCalculateWithInvalidResult() {
        Division divByZero = new Division(10.0, 0.0);
        double safeResult = MathOperations.safeCalculate(divByZero, -1.0);
        assertEquals(-1.0, safeResult, 0.01);
    }
}

// Test 10: Static safeCalculate returns result for valid result
class Test10 {
    @Test
    void testSafeCalculateWithValidResult() {
        Addition add = new Addition(5.0, 3.0);
        double safeResult = MathOperations.safeCalculate(add, 0.0);
        assertEquals(8.0, safeResult, 0.01);
    }
}`,
    translations: {
        ru: {
            title: 'Статические методы в интерфейсах',
            solutionCode: `// Интерфейс с абстрактными и статическими методами
interface MathOperations {
    // Абстрактный метод для реализации
    double calculate();

    // Статический вспомогательный метод - принадлежит интерфейсу, а не экземплярам
    static boolean isValid(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    // Статический метод, который использует абстрактный метод через параметр
    static double safeCalculate(MathOperations operation, double defaultValue) {
        double result = operation.calculate();
        return isValid(result) ? result : defaultValue;
    }
}

// Addition реализует интерфейс
class Addition implements MathOperations {
    private double a;
    private double b;

    public Addition(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public double calculate() {
        return a + b;
    }

    @Override
    public String toString() {
        return String.format("%.1f + %.1f", a, b);
    }
}

// Division реализует интерфейс
class Division implements MathOperations {
    private double dividend;
    private double divisor;

    public Division(double dividend, double divisor) {
        this.dividend = dividend;
        this.divisor = divisor;
    }

    @Override
    public double calculate() {
        return dividend / divisor;
    }

    @Override
    public String toString() {
        return String.format("%.1f / %.1f", dividend, divisor);
    }
}

public class StaticMethods {
    public static void main(String[] args) {
        // Тестируем корректное сложение
        MathOperations add = new Addition(10.5, 5.3);
        double addResult = add.calculate();
        System.out.printf("Addition: %s = %.2f%n", add, addResult);
        System.out.println("Is valid? " + MathOperations.isValid(addResult));

        System.out.println();

        // Тестируем корректное деление
        MathOperations div = new Division(20.0, 4.0);
        double divResult = div.calculate();
        System.out.printf("Division: %s = %.2f%n", div, divResult);
        System.out.println("Is valid? " + MathOperations.isValid(divResult));

        System.out.println();

        // Тестируем деление на ноль со статическим безопасным методом
        MathOperations divByZero = new Division(10.0, 0.0);
        double invalidResult = divByZero.calculate();
        System.out.printf("Division by zero: %s = %s%n", divByZero, invalidResult);
        System.out.println("Is valid? " + MathOperations.isValid(invalidResult));

        // Используем статический безопасный расчет
        double safeResult = MathOperations.safeCalculate(divByZero, 0.0);
        System.out.printf("Safe result: %.2f%n", safeResult);
    }
}`,
            description: `# Статические методы в интерфейсах

Java 8 также представила статические методы в интерфейсах. Это вспомогательные методы, которые принадлежат самому интерфейсу, а не реализующим классам. Они не могут быть переопределены.

## Требования:
1. Создайте интерфейс \`MathOperations\` с:
   1.1. Абстрактным методом: \`double calculate()\`
   1.2. Статическим методом: \`boolean isValid(double value)\` - проверяет, что не NaN или Infinity
   1.3. Статическим методом: \`double safeCalculate(MathOperations operation, double defaultValue)\` - возвращает результат или значение по умолчанию, если невалидно

2. Создайте класс \`Addition\`:
   2.1. Реализует \`MathOperations\`
   2.2. Имеет два поля типа double
   2.3. Реализует \`calculate()\` для возврата суммы

3. Создайте класс \`Division\`:
   3.1. Реализует \`MathOperations\`
   3.2. Имеет поля dividend и divisor
   3.3. Реализует \`calculate()\` для возврата результата деления

4. В main() продемонстрируйте:
   4.1. Корректные вычисления
   4.2. Использование статического метода валидации
   4.3. Использование статического метода безопасного вычисления с некорректной операцией (деление на ноль)

## Пример вывода:
\`\`\`
Addition: 10.5 + 5.3 = 15.80
Is valid? true

Division: 20.0 / 4.0 = 5.00
Is valid? true

Division by zero: 10.0 / 0.0 = Infinity
Is valid? false
Safe result: 0.00
\`\`\``,
            hint1: `Статические методы в интерфейсах вызываются с использованием имени интерфейса: InterfaceName.staticMethod()`,
            hint2: `Статические методы не могут напрямую обращаться к методам экземпляра, но могут работать с экземплярами интерфейса, переданными в качестве параметров.`,
            whyItMatters: `Статические методы в интерфейсах предоставляют место для вспомогательных методов, тесно связанных с интерфейсом, без необходимости отдельного вспомогательного класса. Они обычно используются для фабричных методов, валидаторов и вспомогательных функций. Например, интерфейс Comparator имеет статические методы, такие как comparing() и naturalOrder(), которые создают экземпляры компаратора.`
        },
        uz: {
            title: `Interfeyslarда statik metodlar`,
            solutionCode: `// Abstrakt va statik metodlar bilan interfeys
interface MathOperations {
    // Amalga oshirilishi kerak bo'lgan abstrakt metod
    double calculate();

    // Statik yordamchi metod - misollar emas, interfeysga tegishli
    static boolean isValid(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    // Parametr orqali abstrakt metoddan foydalanadigan statik metod
    static double safeCalculate(MathOperations operation, double defaultValue) {
        double result = operation.calculate();
        return isValid(result) ? result : defaultValue;
    }
}

// Addition interfeyсni amalga oshiradi
class Addition implements MathOperations {
    private double a;
    private double b;

    public Addition(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public double calculate() {
        return a + b;
    }

    @Override
    public String toString() {
        return String.format("%.1f + %.1f", a, b);
    }
}

// Division interfeyсni amalga oshiradi
class Division implements MathOperations {
    private double dividend;
    private double divisor;

    public Division(double dividend, double divisor) {
        this.dividend = dividend;
        this.divisor = divisor;
    }

    @Override
    public double calculate() {
        return dividend / divisor;
    }

    @Override
    public String toString() {
        return String.format("%.1f / %.1f", dividend, divisor);
    }
}

public class StaticMethods {
    public static void main(String[] args) {
        // To'g'ri qo'shishni sinovdan o'tkazamiz
        MathOperations add = new Addition(10.5, 5.3);
        double addResult = add.calculate();
        System.out.printf("Addition: %s = %.2f%n", add, addResult);
        System.out.println("Is valid? " + MathOperations.isValid(addResult));

        System.out.println();

        // To'g'ri bo'lishni sinovdan o'tkazamiz
        MathOperations div = new Division(20.0, 4.0);
        double divResult = div.calculate();
        System.out.printf("Division: %s = %.2f%n", div, divResult);
        System.out.println("Is valid? " + MathOperations.isValid(divResult));

        System.out.println();

        // Nolga bo'lishni statik xavfsiz metod bilan sinovdan o'tkazamiz
        MathOperations divByZero = new Division(10.0, 0.0);
        double invalidResult = divByZero.calculate();
        System.out.printf("Division by zero: %s = %s%n", divByZero, invalidResult);
        System.out.println("Is valid? " + MathOperations.isValid(invalidResult));

        // Statik xavfsiz hisoblashdan foydalanamiz
        double safeResult = MathOperations.safeCalculate(divByZero, 0.0);
        System.out.printf("Safe result: %.2f%n", safeResult);
    }
}`,
            description: `# Interfeyslarда statik metodlar

Java 8 interfeyslarда statik metodlarni ham kiritdi. Bu interfeyѕning o'ziga tegishli yordamchi metodlar bo'lib, amalga oshiruvchi klasslarga emas. Ular qayta yozib bo'lmaydi.

## Talablar:
1. \`MathOperations\` interfeysini yarating:
   1.1. Abstrakt metod: \`double calculate()\`
   1.2. Statik metod: \`boolean isValid(double value)\` - NaN yoki Infinity emasligini tekshiradi
   1.3. Statik metod: \`double safeCalculate(MathOperations operation, double defaultValue)\` - natijani yoki noto'g'ri bo'lsa standart qiymatni qaytaradi

2. \`Addition\` klassini yarating:
   2.1. \`MathOperations\` ni amalga oshiradi
   2.2. Ikkita double maydoniga ega
   2.3. Yig'indini qaytarish uchun \`calculate()\` ni amalga oshiradi

3. \`Division\` klassini yarating:
   3.1. \`MathOperations\` ni amalga oshiradi
   3.2. dividend va divisor maydonlariga ega
   3.3. Bo'lish natijasini qaytarish uchun \`calculate()\` ni amalga oshiradi

4. main() da namoyish eting:
   4.1. To'g'ri hisoblashlar
   4.2. Statik tekshirish metodidan foydalanish
   4.3. Noto'g'ri operatsiya bilan statik xavfsiz hisoblash metodidan foydalanish (nolga bo'lish)

## Chiqish namunasi:
\`\`\`
Addition: 10.5 + 5.3 = 15.80
Is valid? true

Division: 20.0 / 4.0 = 5.00
Is valid? true

Division by zero: 10.0 / 0.0 = Infinity
Is valid? false
Safe result: 0.00
\`\`\``,
            hint1: `Interfeyslarдаgi statik metodlar interfeys nomi yordamida chaqiriladi: InterfaceName.staticMethod()`,
            hint2: `Statik metodlar misol metodlariga to'g'ridan-to'g'ri murojaat qila olmaydi, lekin parametr sifatida uzatilgan interfeys misollari bilan ishlashi mumkin.`,
            whyItMatters: `Interfeyslarдаgi statik metodlar alohida yordamchi klass kerak bo'lmasdan, interfeys bilan chambarchas bog'liq yordamchi metodlar uchun joy beradi. Ular odatda fabrika metodlari, validatorlar va yordamchi funksiyalar uchun ishlatiladi. Masalan, Comparator interfeysi qiyoslash misollari yaratadigan comparing() va naturalOrder() kabi statik metodlarga ega.`
        }
    }
};

export default task;
