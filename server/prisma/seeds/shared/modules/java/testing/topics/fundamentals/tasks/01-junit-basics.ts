import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-junit-basics',
    title: 'JUnit Basics',
    difficulty: 'easy',
    tags: ['java', 'testing', 'junit', 'unit-testing'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Write your first unit tests using **JUnit 5** to test a simple Calculator class.

**Requirements:**
1. Create a Calculator class with the following methods:
   1.1. add(int a, int b) - returns sum
   1.2. subtract(int a, int b) - returns difference
   1.3. multiply(int a, int b) - returns product
   1.4. divide(int a, int b) - returns quotient

2. Create a CalculatorTest class with test methods:
   2.1. Use @Test annotation for each test method
   2.2. Test each calculator method with different inputs
   2.3. Use basic assertions (assertEquals)
   2.4. Include at least 5 test cases

3. Test edge cases:
   3.1. Division by zero (should throw ArithmeticException)
   3.2. Negative numbers
   3.3. Zero values

**Learning Goals:**
- Understand the structure of a JUnit test
- Learn to use @Test annotation
- Practice writing basic assertions
- Understand test naming conventions`,
    initialCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Calculator {
    // TODO: Implement add method

    // TODO: Implement subtract method

    // TODO: Implement multiply method

    // TODO: Implement divide method
}

class CalculatorTest {
    // TODO: Create test methods with @Test annotation
    // TODO: Test add, subtract, multiply, divide methods
    // TODO: Test edge cases
}`,
    solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Calculator {
    // Add two numbers
    public int add(int a, int b) {
        return a + b;
    }

    // Subtract two numbers
    public int subtract(int a, int b) {
        return a - b;
    }

    // Multiply two numbers
    public int multiply(int a, int b) {
        return a * b;
    }

    // Divide two numbers
    public int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        return a / b;
    }
}

class CalculatorTest {
    // Test addition with positive numbers
    @Test
    void testAddPositiveNumbers() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }

    // Test addition with negative numbers
    @Test
    void testAddNegativeNumbers() {
        Calculator calc = new Calculator();
        assertEquals(-5, calc.add(-2, -3));
    }

    // Test subtraction
    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.subtract(10, 5));
    }

    // Test multiplication
    @Test
    void testMultiply() {
        Calculator calc = new Calculator();
        assertEquals(20, calc.multiply(4, 5));
    }

    // Test division
    @Test
    void testDivide() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.divide(20, 5));
    }

    // Test division by zero throws exception
    @Test
    void testDivideByZero() {
        Calculator calc = new Calculator();
        assertThrows(ArithmeticException.class, () -> {
            calc.divide(10, 0);
        });
    }

    // Test with zero values
    @Test
    void testMultiplyByZero() {
        Calculator calc = new Calculator();
        assertEquals(0, calc.multiply(5, 0));
    }
}`,
    hint1: `Start by implementing the Calculator class with basic arithmetic operations. Each method should perform one operation and return the result.`,
    hint2: `In your test class, create a new Calculator instance in each test method. Use assertEquals(expected, actual) to verify the results.`,
    whyItMatters: `Unit testing is essential for ensuring code quality and preventing bugs. JUnit is the industry-standard testing framework for Java, used in professional development to verify that individual units of code work correctly. Writing tests helps you catch bugs early and makes refactoring safer.

**Production Pattern:**
\`\`\`java
@Test
void testCalculateDiscount() {
    PriceCalculator calculator = new PriceCalculator();
    assertEquals(90.0, calculator.applyDiscount(100.0, 10));
    assertEquals(0.0, calculator.applyDiscount(0.0, 10));
}
\`\`\`

**Practical Benefits:**
- Quick detection of regressions when code changes
- Documentation of expected system behavior`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Test basic assertion
class Test1 {
    @Test
    public void test() {
        assertEquals(5, 2 + 3);
    }
}

// Test2: Test assertTrue
class Test2 {
    @Test
    public void test() {
        assertTrue(10 > 5);
    }
}

// Test3: Test assertFalse
class Test3 {
    @Test
    public void test() {
        assertFalse(5 > 10);
    }
}

// Test4: Test assertNull
class Test4 {
    @Test
    public void test() {
        String value = null;
        assertNull(value);
    }
}

// Test5: Test assertNotNull
class Test5 {
    @Test
    public void test() {
        String value = "Hello";
        assertNotNull(value);
    }
}

// Test6: Test assertEquals with strings
class Test6 {
    @Test
    public void test() {
        String expected = "JUnit";
        String actual = "JUnit";
        assertEquals(expected, actual);
    }
}

// Test7: Test basic math operation
class Test7 {
    @Test
    public void test() {
        int result = 4 * 5;
        assertEquals(20, result);
    }
}

// Test8: Test array equality
class Test8 {
    @Test
    public void test() {
        int[] expected = {1, 2, 3};
        int[] actual = {1, 2, 3};
        assertArrayEquals(expected, actual);
    }
}

// Test9: Test object comparison
class Test9 {
    @Test
    public void test() {
        Integer a = 42;
        Integer b = 42;
        assertEquals(a, b);
    }
}

// Test10: Test multiple assertions
class Test10 {
    @Test
    public void test() {
        String text = "Hello World";
        assertTrue(text.startsWith("Hello"));
        assertFalse(text.isEmpty());
        assertEquals(11, text.length());
        assertNotNull(text.toUpperCase());
    }
}
`,
    translations: {
        ru: {
            title: 'Основы JUnit',
            solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Calculator {
    // Сложить два числа
    public int add(int a, int b) {
        return a + b;
    }

    // Вычесть два числа
    public int subtract(int a, int b) {
        return a - b;
    }

    // Умножить два числа
    public int multiply(int a, int b) {
        return a * b;
    }

    // Разделить два числа
    public int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Нельзя делить на ноль");
        }
        return a / b;
    }
}

class CalculatorTest {
    // Тест сложения с положительными числами
    @Test
    void testAddPositiveNumbers() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }

    // Тест сложения с отрицательными числами
    @Test
    void testAddNegativeNumbers() {
        Calculator calc = new Calculator();
        assertEquals(-5, calc.add(-2, -3));
    }

    // Тест вычитания
    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.subtract(10, 5));
    }

    // Тест умножения
    @Test
    void testMultiply() {
        Calculator calc = new Calculator();
        assertEquals(20, calc.multiply(4, 5));
    }

    // Тест деления
    @Test
    void testDivide() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.divide(20, 5));
    }

    // Тест деления на ноль выбрасывает исключение
    @Test
    void testDivideByZero() {
        Calculator calc = new Calculator();
        assertThrows(ArithmeticException.class, () -> {
            calc.divide(10, 0);
        });
    }

    // Тест с нулевыми значениями
    @Test
    void testMultiplyByZero() {
        Calculator calc = new Calculator();
        assertEquals(0, calc.multiply(5, 0));
    }
}`,
            description: `Напишите свои первые модульные тесты используя **JUnit 5** для тестирования простого класса Calculator.

**Требования:**
1. Создайте класс Calculator со следующими методами:
   1.1. add(int a, int b) - возвращает сумму
   1.2. subtract(int a, int b) - возвращает разность
   1.3. multiply(int a, int b) - возвращает произведение
   1.4. divide(int a, int b) - возвращает частное

2. Создайте класс CalculatorTest с тестовыми методами:
   2.1. Используйте аннотацию @Test для каждого тестового метода
   2.2. Протестируйте каждый метод калькулятора с разными входными данными
   2.3. Используйте базовые утверждения (assertEquals)
   2.4. Включите минимум 5 тестовых случаев

3. Протестируйте граничные случаи:
   3.1. Деление на ноль (должно выбросить ArithmeticException)
   3.2. Отрицательные числа
   3.3. Нулевые значения

**Цели обучения:**
- Понять структуру JUnit теста
- Научиться использовать аннотацию @Test
- Практиковаться в написании базовых утверждений
- Понять соглашения об именовании тестов`,
            hint1: `Начните с реализации класса Calculator с базовыми арифметическими операциями. Каждый метод должен выполнять одну операцию и возвращать результат.`,
            hint2: `В вашем тестовом классе создайте новый экземпляр Calculator в каждом тестовом методе. Используйте assertEquals(ожидаемое, фактическое) для проверки результатов.`,
            whyItMatters: `Модульное тестирование необходимо для обеспечения качества кода и предотвращения ошибок. JUnit - это стандартный в индустрии фреймворк тестирования для Java, используемый в профессиональной разработке для проверки правильности работы отдельных модулей кода. Написание тестов помогает обнаруживать ошибки на ранних этапах и делает рефакторинг более безопасным.

**Продакшен паттерн:**
\`\`\`java
@Test
void testCalculateDiscount() {
    PriceCalculator calculator = new PriceCalculator();
    assertEquals(90.0, calculator.applyDiscount(100.0, 10));
    assertEquals(0.0, calculator.applyDiscount(0.0, 10));
}
\`\`\`

**Практические преимущества:**
- Быстрое обнаружение регрессий при изменении кода
- Документирование ожидаемого поведения системы`
        },
        uz: {
            title: 'JUnit Asoslari',
            solutionCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Calculator {
    // Ikki sonni qo'shish
    public int add(int a, int b) {
        return a + b;
    }

    // Ikki sonni ayirish
    public int subtract(int a, int b) {
        return a - b;
    }

    // Ikki sonni ko'paytirish
    public int multiply(int a, int b) {
        return a * b;
    }

    // Ikki sonni bo'lish
    public int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Nolga bo'lib bo'lmaydi");
        }
        return a / b;
    }
}

class CalculatorTest {
    // Musbat sonlarni qo'shish testi
    @Test
    void testAddPositiveNumbers() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }

    // Manfiy sonlarni qo'shish testi
    @Test
    void testAddNegativeNumbers() {
        Calculator calc = new Calculator();
        assertEquals(-5, calc.add(-2, -3));
    }

    // Ayirish testi
    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.subtract(10, 5));
    }

    // Ko'paytirish testi
    @Test
    void testMultiply() {
        Calculator calc = new Calculator();
        assertEquals(20, calc.multiply(4, 5));
    }

    // Bo'lish testi
    @Test
    void testDivide() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.divide(20, 5));
    }

    // Nolga bo'lish istisno chiqaradi testi
    @Test
    void testDivideByZero() {
        Calculator calc = new Calculator();
        assertThrows(ArithmeticException.class, () -> {
            calc.divide(10, 0);
        });
    }

    // Nol qiymatlari bilan test
    @Test
    void testMultiplyByZero() {
        Calculator calc = new Calculator();
        assertEquals(0, calc.multiply(5, 0));
    }
}`,
            description: `**JUnit 5** yordamida oddiy Calculator sinfini testlash uchun birinchi modul testlaringizni yozing.

**Talablar:**
1. Quyidagi metodlarga ega Calculator sinfini yarating:
   1.1. add(int a, int b) - yig'indini qaytaradi
   1.2. subtract(int a, int b) - ayirmani qaytaradi
   1.3. multiply(int a, int b) - ko'paytmani qaytaradi
   1.4. divide(int a, int b) - bo'linmani qaytaradi

2. Test metodlari bilan CalculatorTest sinfini yarating:
   2.1. Har bir test metodi uchun @Test annotatsiyasidan foydalaning
   2.2. Har bir kalkulyator metodini turli kiritishlar bilan sinab ko'ring
   2.3. Asosiy tasdiqlashlardan foydalaning (assertEquals)
   2.4. Kamida 5 ta test holatini kiriting

3. Chegara holatlarini sinab ko'ring:
   3.1. Nolga bo'lish (ArithmeticException chiqarishi kerak)
   3.2. Manfiy sonlar
   3.3. Nol qiymatlari

**O'rganish maqsadlari:**
- JUnit test tuzilmasini tushunish
- @Test annotatsiyasidan foydalanishni o'rganish
- Asosiy tasdiqlashlarni yozishda amaliyot
- Test nomlash konventsiyalarini tushunish`,
            hint1: `Asosiy arifmetik operatsiyalar bilan Calculator sinfini amalga oshirishdan boshlang. Har bir metod bitta operatsiyani bajarishi va natijani qaytarishi kerak.`,
            hint2: `Test sinfingizda har bir test metodida yangi Calculator nusxasini yarating. Natijalarni tekshirish uchun assertEquals(kutilgan, haqiqiy) dan foydalaning.`,
            whyItMatters: `Modul testlash kod sifatini ta'minlash va xatolarni oldini olish uchun zarur. JUnit - bu Java uchun sanoat standart testlash freymvorki bo'lib, professional ishlab chiqishda kodning alohida qismlarining to'g'ri ishlashini tekshirish uchun ishlatiladi. Testlar yozish xatolarni erta bosqichda aniqlashga yordam beradi va refaktoringni xavfsizroq qiladi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Test
void testCalculateDiscount() {
    PriceCalculator calculator = new PriceCalculator();
    assertEquals(90.0, calculator.applyDiscount(100.0, 10));
    assertEquals(0.0, calculator.applyDiscount(0.0, 10));
}
\`\`\`

**Amaliy foydalari:**
- Kod o'zgartirilganda regressiyalarni tez aniqlash
- Tizimning kutilgan xatti-harakatini hujjatlash`
        }
    }
};

export default task;
