import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-lambda-functional-interfaces',
    title: 'Functional Interfaces for Lambdas',
    difficulty: 'easy',
    tags: ['java', 'lambda', 'functional-interface', 'java8', 'annotation'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Functional Interfaces for Lambdas

A functional interface has exactly one abstract method (SAM - Single Abstract Method). Lambda expressions can only be used with functional interfaces. The @FunctionalInterface annotation ensures compile-time checking of this rule.

## Requirements:
1. Create custom functional interfaces:
   1.1. \`@FunctionalInterface\` annotation
   1.2. Single abstract method
   1.3. Optional default and static methods

2. Define various functional interface types:
   2.1. \`StringProcessor\`: Process string and return string
   2.2. \`Calculator\`: Perform arithmetic operations
   2.3. \`Validator\`: Validate input and return boolean
   2.4. \`Printer\`: Print message (void return)

3. Demonstrate SAM rule:
   3.1. Valid functional interface with one abstract method
   3.2. Multiple default/static methods are allowed
   3.3. Show what makes an interface NOT functional

4. Use lambdas with custom interfaces

## Example Output:
\`\`\`
=== Custom Functional Interfaces ===
StringProcessor: HELLO WORLD
Calculator: 10 + 5 = 15
Validator: Is "test@email.com" valid email? true
Printer: Processing: Important Data

=== Functional Interface Features ===
Default method: Processing string...
Static method: Validation enabled
\`\`\``,
    initialCode: `// TODO: Create StringProcessor functional interface

// TODO: Create Calculator functional interface

// TODO: Create Validator functional interface

// TODO: Create Printer functional interface

public class FunctionalInterfacesDemo {
    public static void main(String[] args) {
        // TODO: Use lambdas with custom functional interfaces
    }
}`,
    solutionCode: `// Custom functional interface for string processing
@FunctionalInterface
interface StringProcessor {
    // Single abstract method (SAM)
    String process(String input);

    // Default methods are allowed
    default String processWithInfo(String input) {
        System.out.println("Processing string...");
        return process(input);
    }

    // Static methods are allowed
    static String uppercase(String s) {
        return s.toUpperCase();
    }
}

// Functional interface for calculations
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);

    // Can have multiple default methods
    default String describe() {
        return "Calculator operation";
    }
}

// Functional interface for validation
@FunctionalInterface
interface Validator {
    boolean validate(String input);

    static Validator emailValidator() {
        return email -> email.contains("@") && email.contains(".");
    }
}

// Functional interface with void return
@FunctionalInterface
interface Printer {
    void print(String message);

    default void printFormatted(String msg) {
        print("Processing: " + msg);
    }
}

public class FunctionalInterfacesDemo {
    public static void main(String[] args) {
        System.out.println("=== Custom Functional Interfaces ===");

        // Lambda for StringProcessor
        StringProcessor toUpperCase = s -> s.toUpperCase();
        System.out.println("StringProcessor: " + toUpperCase.process("hello world"));

        // Lambda for Calculator
        Calculator addition = (a, b) -> a + b;
        System.out.println("Calculator: 10 + 5 = " + addition.calculate(10, 5));

        // Lambda for Validator
        Validator emailValidator = email -> email.contains("@") && email.contains(".");
        System.out.println("Validator: Is \\"test@email.com\\" valid email? " +
            emailValidator.validate("test@email.com"));

        // Lambda for Printer
        Printer printer = msg -> System.out.println("Printer: " + msg);
        printer.printFormatted("Important Data");

        System.out.println("\\n=== Functional Interface Features ===");

        // Using default method
        StringProcessor processor = s -> s.toUpperCase();
        System.out.print("Default method: ");
        processor.processWithInfo("test");

        // Using static method
        System.out.println("Static method: " +
            (Validator.emailValidator().validate("test@test.com") ?
                "Validation enabled" : "Validation failed"));
    }
}`,
    hint1: `@FunctionalInterface annotation is optional but recommended. It makes the compiler check that the interface has exactly one abstract method.`,
    hint2: `Functional interfaces can have multiple default and static methods. Only the abstract method count matters for the SAM rule.`,
    whyItMatters: `Functional interfaces are the foundation of lambda expressions in Java. Understanding how to create and use them allows you to design clean, flexible APIs that work seamlessly with lambda expressions and method references. They're essential for modern Java programming and frameworks.

**Production Pattern:**

\`\`\`java
// Custom functional interface for business logic
@FunctionalInterface
interface PaymentProcessor {
    boolean processPayment(double amount, String currency);

    // Default methods for extended functionality
    default boolean processPaymentWithFee(double amount, String currency) {
        double fee = amount * 0.02; // 2% commission
        return processPayment(amount + fee, currency);
    }

    // Static factory methods
    static PaymentProcessor createStandardProcessor() {
        return (amount, currency) -> {
            System.out.println("Processing " + amount + " " + currency);
            return amount > 0;
        };
    }
}

// Usage in real code
public class PaymentService {
    public void executePayment(PaymentProcessor processor) {
        boolean success = processor.processPayment(100.0, "USD");
    }

    public void run() {
        // Lambda expression
        executePayment((amount, currency) ->
            amount > 0 && currency.equals("USD"));

        // Factory method
        executePayment(PaymentProcessor.createStandardProcessor());
    }
}
\`\`\`

**Practical Benefits:**

1. **Design Flexibility**: Create your own interfaces for specific project needs
2. **Clean API**: Intuitive contracts for users of your code
3. **Extensibility**: Default and static methods add functionality without changing the contract
4. **Type Safety**: Compiler checks signature matching at compile-time`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify StringProcessor with toUpperCase
class Test1 {
    @Test
    public void test() {
        StringProcessor proc = s -> s.toUpperCase();
        assertEquals("HELLO", proc.process("hello"));
    }
}

// Test2: Verify Calculator with addition
class Test2 {
    @Test
    public void test() {
        Calculator calc = (a, b) -> a + b;
        assertEquals(15, calc.calculate(10, 5));
    }
}

// Test3: Verify Validator with email validation
class Test3 {
    @Test
    public void test() {
        Validator validator = email -> email.contains("@") && email.contains(".");
        assertTrue(validator.validate("test@email.com"));
        assertFalse(validator.validate("invalid"));
    }
}

// Test4: Verify Printer with message outputs correctly
class Test4 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Printer printer = msg -> System.out.println(msg);
            printer.print("TestMessage");
            String output = out.toString().trim();
            assertEquals("TestMessage", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test5: Verify StringProcessor default method
class Test5 {
    @Test
    public void test() {
        StringProcessor proc = s -> s.toLowerCase();
        String result = proc.processWithInfo("HELLO");
        assertEquals("hello", result);
    }
}

// Test6: Verify Calculator with multiplication
class Test6 {
    @Test
    public void test() {
        Calculator calc = (a, b) -> a * b;
        assertEquals(20, calc.calculate(4, 5));
    }
}

// Test7: Verify Validator static method
class Test7 {
    @Test
    public void test() {
        Validator validator = Validator.emailValidator();
        assertTrue(validator.validate("user@domain.com"));
    }
}

// Test8: Verify Printer formatted method outputs correctly
class Test8 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Printer printer = msg -> System.out.println("Msg: " + msg);
            printer.printFormatted("Data");
            String output = out.toString().trim();
            assertTrue(output.contains("Data"));
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test9: Verify Calculator subtraction
class Test9 {
    @Test
    public void test() {
        Calculator calc = (a, b) -> a - b;
        assertEquals(3, calc.calculate(10, 7));
    }
}

// Test10: Verify StringProcessor with trim
class Test10 {
    @Test
    public void test() {
        StringProcessor proc = s -> s.trim();
        assertEquals("hello", proc.process("  hello  "));
    }
}`,
    translations: {
        ru: {
            title: 'Функциональные интерфейсы для лямбд',
            solutionCode: `// Пользовательский функциональный интерфейс для обработки строк
@FunctionalInterface
interface StringProcessor {
    // Единственный абстрактный метод (SAM)
    String process(String input);

    // Методы по умолчанию разрешены
    default String processWithInfo(String input) {
        System.out.println("Processing string...");
        return process(input);
    }

    // Статические методы разрешены
    static String uppercase(String s) {
        return s.toUpperCase();
    }
}

// Функциональный интерфейс для вычислений
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);

    // Может иметь несколько методов по умолчанию
    default String describe() {
        return "Calculator operation";
    }
}

// Функциональный интерфейс для валидации
@FunctionalInterface
interface Validator {
    boolean validate(String input);

    static Validator emailValidator() {
        return email -> email.contains("@") && email.contains(".");
    }
}

// Функциональный интерфейс с возвратом void
@FunctionalInterface
interface Printer {
    void print(String message);

    default void printFormatted(String msg) {
        print("Processing: " + msg);
    }
}

public class FunctionalInterfacesDemo {
    public static void main(String[] args) {
        System.out.println("=== Пользовательские функциональные интерфейсы ===");

        // Лямбда для StringProcessor
        StringProcessor toUpperCase = s -> s.toUpperCase();
        System.out.println("StringProcessor: " + toUpperCase.process("hello world"));

        // Лямбда для Calculator
        Calculator addition = (a, b) -> a + b;
        System.out.println("Calculator: 10 + 5 = " + addition.calculate(10, 5));

        // Лямбда для Validator
        Validator emailValidator = email -> email.contains("@") && email.contains(".");
        System.out.println("Validator: Is \\"test@email.com\\" valid email? " +
            emailValidator.validate("test@email.com"));

        // Лямбда для Printer
        Printer printer = msg -> System.out.println("Printer: " + msg);
        printer.printFormatted("Important Data");

        System.out.println("\\n=== Возможности функционального интерфейса ===");

        // Использование метода по умолчанию
        StringProcessor processor = s -> s.toUpperCase();
        System.out.print("Default method: ");
        processor.processWithInfo("test");

        // Использование статического метода
        System.out.println("Static method: " +
            (Validator.emailValidator().validate("test@test.com") ?
                "Validation enabled" : "Validation failed"));
    }
}`,
            description: `# Функциональные интерфейсы для лямбд

Функциональный интерфейс имеет ровно один абстрактный метод (SAM - Single Abstract Method). Лямбда-выражения могут использоваться только с функциональными интерфейсами. Аннотация @FunctionalInterface обеспечивает проверку этого правила на этапе компиляции.

## Требования:
1. Создайте пользовательские функциональные интерфейсы:
   1.1. Аннотация \`@FunctionalInterface\`
   1.2. Единственный абстрактный метод
   1.3. Необязательные методы по умолчанию и статические методы

2. Определите различные типы функциональных интерфейсов:
   2.1. \`StringProcessor\`: Обработать строку и вернуть строку
   2.2. \`Calculator\`: Выполнить арифметические операции
   2.3. \`Validator\`: Проверить ввод и вернуть boolean
   2.4. \`Printer\`: Напечатать сообщение (возврат void)

3. Продемонстрируйте правило SAM:
   3.1. Допустимый функциональный интерфейс с одним абстрактным методом
   3.2. Несколько методов по умолчанию/статических разрешены
   3.3. Покажите, что делает интерфейс НЕ функциональным

4. Используйте лямбды с пользовательскими интерфейсами

## Пример вывода:
\`\`\`
=== Custom Functional Interfaces ===
StringProcessor: HELLO WORLD
Calculator: 10 + 5 = 15
Validator: Is "test@email.com" valid email? true
Printer: Processing: Important Data

=== Functional Interface Features ===
Default method: Processing string...
Static method: Validation enabled
\`\`\``,
            hint1: `Аннотация @FunctionalInterface необязательна, но рекомендуется. Она заставляет компилятор проверять, что интерфейс имеет ровно один абстрактный метод.`,
            hint2: `Функциональные интерфейсы могут иметь несколько методов по умолчанию и статических методов. Для правила SAM важно только количество абстрактных методов.`,
            whyItMatters: `Функциональные интерфейсы - это основа лямбда-выражений в Java. Понимание того, как их создавать и использовать, позволяет вам проектировать чистые, гибкие API, которые беспрепятственно работают с лямбда-выражениями и ссылками на методы. Они необходимы для современного программирования на Java и фреймворков.

**Продакшен паттерн:**

\`\`\`java
// Пользовательский функциональный интерфейс для бизнес-логики
@FunctionalInterface
interface PaymentProcessor {
    boolean processPayment(double amount, String currency);

    // Default методы для расширенной функциональности
    default boolean processPaymentWithFee(double amount, String currency) {
        double fee = amount * 0.02; // 2% комиссия
        return processPayment(amount + fee, currency);
    }

    // Static factory методы
    static PaymentProcessor createStandardProcessor() {
        return (amount, currency) -> {
            System.out.println("Processing " + amount + " " + currency);
            return amount > 0;
        };
    }
}

// Использование в реальном коде
public class PaymentService {
    public void executePayment(PaymentProcessor processor) {
        boolean success = processor.processPayment(100.0, "USD");
    }

    public void run() {
        // Lambda выражение
        executePayment((amount, currency) ->
            amount > 0 && currency.equals("USD"));

        // Factory метод
        executePayment(PaymentProcessor.createStandardProcessor());
    }
}
\`\`\`

**Практические преимущества:**

1. **Гибкость дизайна**: Создавайте свои интерфейсы под конкретные нужды проекта
2. **Чистый API**: Интуитивно понятные контракты для пользователей вашего кода
3. **Расширяемость**: Default и static методы добавляют функциональность без изменения контракта
4. **Type safety**: Компилятор проверяет соответствие сигнатуры в compile-time`
        },
        uz: {
            title: `Lambdalar uchun funksional interfeyslar`,
            solutionCode: `// Satrlarni qayta ishlash uchun maxsus funksional interfeys
@FunctionalInterface
interface StringProcessor {
    // Yagona abstrakt metod (SAM)
    String process(String input);

    // Standart metodlarga ruxsat berilgan
    default String processWithInfo(String input) {
        System.out.println("Processing string...");
        return process(input);
    }

    // Statik metodlarga ruxsat berilgan
    static String uppercase(String s) {
        return s.toUpperCase();
    }
}

// Hisoblashlar uchun funksional interfeys
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);

    // Bir nechta standart metodlarga ega bo'lishi mumkin
    default String describe() {
        return "Calculator operation";
    }
}

// Tekshirish uchun funksional interfeys
@FunctionalInterface
interface Validator {
    boolean validate(String input);

    static Validator emailValidator() {
        return email -> email.contains("@") && email.contains(".");
    }
}

// void qaytarish bilan funksional interfeys
@FunctionalInterface
interface Printer {
    void print(String message);

    default void printFormatted(String msg) {
        print("Processing: " + msg);
    }
}

public class FunctionalInterfacesDemo {
    public static void main(String[] args) {
        System.out.println("=== Maxsus funksional interfeyslar ===");

        // StringProcessor uchun lambda
        StringProcessor toUpperCase = s -> s.toUpperCase();
        System.out.println("StringProcessor: " + toUpperCase.process("hello world"));

        // Calculator uchun lambda
        Calculator addition = (a, b) -> a + b;
        System.out.println("Calculator: 10 + 5 = " + addition.calculate(10, 5));

        // Validator uchun lambda
        Validator emailValidator = email -> email.contains("@") && email.contains(".");
        System.out.println("Validator: Is \\"test@email.com\\" valid email? " +
            emailValidator.validate("test@email.com"));

        // Printer uchun lambda
        Printer printer = msg -> System.out.println("Printer: " + msg);
        printer.printFormatted("Important Data");

        System.out.println("\\n=== Funksional interfeys xususiyatlari ===");

        // Standart metoddan foydalanish
        StringProcessor processor = s -> s.toUpperCase();
        System.out.print("Default method: ");
        processor.processWithInfo("test");

        // Statik metoddan foydalanish
        System.out.println("Static method: " +
            (Validator.emailValidator().validate("test@test.com") ?
                "Validation enabled" : "Validation failed"));
    }
}`,
            description: `# Lambdalar uchun funksional interfeyslar

Funksional interfeys aynan bitta abstrakt metodga ega (SAM - Single Abstract Method). Lambda ifodalari faqat funksional interfeyslar bilan ishlatilishi mumkin. @FunctionalInterface annotatsiyasi bu qoidani kompilyatsiya vaqtida tekshirishni ta'minlaydi.

## Talablar:
1. Maxsus funksional interfeyslarni yarating:
   1.1. \`@FunctionalInterface\` annotatsiyasi
   1.2. Yagona abstrakt metod
   1.3. Ixtiyoriy standart va statik metodlar

2. Turli funksional interfeys turlarini belgilang:
   2.1. \`StringProcessor\`: Satrni qayta ishlash va satrni qaytarish
   2.2. \`Calculator\`: Arifmetik operatsiyalarni bajarish
   2.3. \`Validator\`: Kirishni tekshirish va boolean qaytarish
   2.4. \`Printer\`: Xabarni chiqarish (void qaytarish)

3. SAM qoidasini namoyish eting:
   3.1. Bitta abstrakt metod bilan to'g'ri funksional interfeys
   3.2. Bir nechta standart/statik metodlarga ruxsat berilgan
   3.3. Interfeys funksional bo'lmaganini ko'rsating

4. Maxsus interfeyslar bilan lambdalardan foydalaning

## Chiqish namunasi:
\`\`\`
=== Custom Functional Interfaces ===
StringProcessor: HELLO WORLD
Calculator: 10 + 5 = 15
Validator: Is "test@email.com" valid email? true
Printer: Processing: Important Data

=== Functional Interface Features ===
Default method: Processing string...
Static method: Validation enabled
\`\`\``,
            hint1: `@FunctionalInterface annotatsiyasi ixtiyoriy, lekin tavsiya etiladi. Bu kompilyatorga interfeys aynan bitta abstrakt metodga ega ekanligini tekshirishni majbur qiladi.`,
            hint2: `Funksional interfeyslar bir nechta standart va statik metodlarga ega bo'lishi mumkin. SAM qoidasi uchun faqat abstrakt metodlar soni muhim.`,
            whyItMatters: `Funksional interfeyslar Java-da lambda ifodalarning asosi hisoblanadi. Ularni yaratish va ishlatishni tushunish sizga lambda ifodalari va metod havolalari bilan muammosiz ishlaydigan toza, moslashuvchan API-larni loyihalash imkonini beradi. Ular zamonaviy Java dasturlash va frameworklari uchun zarurdir.

**Ishlab chiqarish patterni:**

\`\`\`java
// Biznes-mantiq uchun maxsus funksional interfeys
@FunctionalInterface
interface PaymentProcessor {
    boolean processPayment(double amount, String currency);

    // Kengaytirilgan funksionallik uchun default metodlar
    default boolean processPaymentWithFee(double amount, String currency) {
        double fee = amount * 0.02; // 2% komissiya
        return processPayment(amount + fee, currency);
    }

    // Static factory metodlar
    static PaymentProcessor createStandardProcessor() {
        return (amount, currency) -> {
            System.out.println("Processing " + amount + " " + currency);
            return amount > 0;
        };
    }
}

// Haqiqiy kodda foydalanish
public class PaymentService {
    public void executePayment(PaymentProcessor processor) {
        boolean success = processor.processPayment(100.0, "USD");
    }

    public void run() {
        // Lambda ifoda
        executePayment((amount, currency) ->
            amount > 0 && currency.equals("USD"));

        // Factory metod
        executePayment(PaymentProcessor.createStandardProcessor());
    }
}
\`\`\`

**Amaliy foydalari:**

1. **Dizayn moslashuvchanligi**: Loyihaning aniq ehtiyojlariga mos interfeyslar yarating
2. **Toza API**: Kodingiz foydalanuvchilari uchun intuitiv shartnomalar
3. **Kengaytiriluvchanlik**: Default va static metodlar shartnomani o'zgartirmasdan funksionallik qo'shadi
4. **Turi xavfsizligi**: Kompilyator compile-time vaqtida signatura mosligini tekshiradi`
        }
    }
};

export default task;
