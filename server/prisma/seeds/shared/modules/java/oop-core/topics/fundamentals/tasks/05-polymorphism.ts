import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-polymorphism',
    title: 'Polymorphism in Action',
    difficulty: 'medium',
    tags: ['java', 'oop', 'polymorphism', 'overloading', 'overriding', 'upcasting', 'downcasting'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a payment processing system that demonstrates both compile-time and runtime polymorphism in Java.

**Requirements:**
1. Create an abstract **Payment** base class with:
   1.1. Protected field: amount (double)
   1.2. Constructor to initialize amount
   1.3. Abstract method: processPayment()
   1.4. Concrete method: displayAmount()

2. Create **CreditCardPayment** class extending Payment:
   2.1. Additional fields: cardNumber, cvv
   2.2. Override processPayment() with credit card specific logic
   2.3. Demonstrate method overloading with multiple process methods

3. Create **PayPalPayment** class extending Payment:
   3.1. Additional field: email
   3.2. Override processPayment() with PayPal specific logic

4. Create **PaymentProcessor** class with:
   4.1. Overloaded methods demonstrating compile-time polymorphism:
        - processPayment(Payment payment)
        - processPayment(Payment[] payments)
        - processPayment(Payment payment, boolean express)

5. In main method:
   5.1. Demonstrate upcasting (parent reference to child object)
   5.2. Demonstrate downcasting with instanceof check
   5.3. Show polymorphic behavior with different payment types
   5.4. Use overloaded methods

**Learning Goals:**
- Understand compile-time polymorphism (method overloading)
- Understand runtime polymorphism (method overriding)
- Practice upcasting and downcasting
- Learn to use instanceof operator safely
- See polymorphism benefits in real-world scenarios`,
    initialCode: `// TODO: Create Payment abstract class

// TODO: Create CreditCardPayment class

// TODO: Create PayPalPayment class

// TODO: Create PaymentProcessor class with overloaded methods

public class PolymorphismDemo {
    public static void main(String[] args) {
        // TODO: Demonstrate polymorphism
    }
}`,
    solutionCode: `// Abstract base class - defines common payment structure
abstract class Payment {
    protected double amount;

    public Payment(double amount) {
        this.amount = amount;
    }

    // Abstract method - must be implemented by subclasses
    public abstract void processPayment();

    // Concrete method - inherited by all subclasses
    public void displayAmount() {
        System.out.println("Payment amount: $" + amount);
    }
}

// CreditCardPayment - demonstrates method overriding and overloading
class CreditCardPayment extends Payment {
    private String cardNumber;
    private String cvv;

    public CreditCardPayment(double amount, String cardNumber, String cvv) {
        super(amount);
        this.cardNumber = cardNumber;
        this.cvv = cvv;
    }

    // Runtime polymorphism - overriding abstract method
    @Override
    public void processPayment() {
        System.out.println("Processing Credit Card Payment...");
        System.out.println("Card: " + maskCardNumber(cardNumber));
        System.out.println("Amount: $" + amount);
        System.out.println("Payment approved!");
    }

    // Compile-time polymorphism - method overloading
    public void processPayment(boolean requireSignature) {
        processPayment();
        if (requireSignature) {
            System.out.println("Signature required.");
        }
    }

    // Compile-time polymorphism - another overloaded method
    public void processPayment(int installments) {
        processPayment();
        System.out.println("Payment split into " + installments + " installments");
    }

    private String maskCardNumber(String cardNumber) {
        return "**** **** **** " + cardNumber.substring(cardNumber.length() - 4);
    }
}

// PayPalPayment - demonstrates method overriding
class PayPalPayment extends Payment {
    private String email;

    public PayPalPayment(double amount, String email) {
        super(amount);
        this.email = email;
    }

    // Runtime polymorphism - overriding abstract method
    @Override
    public void processPayment() {
        System.out.println("Processing PayPal Payment...");
        System.out.println("PayPal Account: " + email);
        System.out.println("Amount: $" + amount);
        System.out.println("Payment sent!");
    }
}

// PaymentProcessor - demonstrates compile-time polymorphism
class PaymentProcessor {
    // Overloaded method #1 - single payment
    public void processPayment(Payment payment) {
        System.out.println("=== Single Payment Processing ===");
        payment.processPayment();
        System.out.println("================================\n");
    }

    // Overloaded method #2 - batch payments
    public void processPayment(Payment[] payments) {
        System.out.println("=== Batch Payment Processing ===");
        for (Payment payment : payments) {
            payment.processPayment();
            System.out.println("---");
        }
        System.out.println("================================\n");
    }

    // Overloaded method #3 - payment with express option
    public void processPayment(Payment payment, boolean express) {
        System.out.println("=== " + (express ? "Express" : "Standard") +
                          " Payment Processing ===");
        payment.processPayment();
        if (express) {
            System.out.println("Express processing - completed in 1 hour");
        }
        System.out.println("================================\n");
    }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        PaymentProcessor processor = new PaymentProcessor();

        // Upcasting - child object referenced by parent type
        Payment payment1 = new CreditCardPayment(150.00, "1234567890123456", "123");
        Payment payment2 = new PayPalPayment(75.50, "user@email.com");

        // Runtime polymorphism - correct method called based on actual object type
        processor.processPayment(payment1);
        processor.processPayment(payment2);

        // Compile-time polymorphism - overloaded method with array
        Payment[] payments = {
            new CreditCardPayment(100.00, "9876543210987654", "456"),
            new PayPalPayment(200.00, "buyer@email.com"),
            new CreditCardPayment(50.00, "1111222233334444", "789")
        };
        processor.processPayment(payments);

        // Compile-time polymorphism - overloaded method with boolean
        Payment expressPayment = new PayPalPayment(500.00, "express@email.com");
        processor.processPayment(expressPayment, true);

        // Downcasting with instanceof check - safe type conversion
        if (payment1 instanceof CreditCardPayment) {
            System.out.println("=== Demonstrating Downcasting ===");
            CreditCardPayment ccPayment = (CreditCardPayment) payment1;
            // Now can access CreditCardPayment-specific overloaded methods
            ccPayment.processPayment(3); // Process with installments
            ccPayment.processPayment(true); // Process with signature
            System.out.println("================================\n");
        }

        // Demonstrating polymorphic array
        System.out.println("=== Polymorphic Behavior ===");
        for (Payment p : payments) {
            p.displayAmount(); // Common method
            p.processPayment(); // Polymorphic behavior
            System.out.println("Type: " + p.getClass().getSimpleName());
            System.out.println("---");
        }
    }
}`,
    hint1: `For compile-time polymorphism, create multiple methods with the same name but different parameters. For runtime polymorphism, override methods in subclasses.`,
    hint2: `Always use instanceof before downcasting to avoid ClassCastException. Upcasting is automatic and safe, but downcasting requires explicit casting and checking.`,
    whyItMatters: `Polymorphism is one of the most powerful features of OOP. Compile-time polymorphism (overloading) provides method flexibility with different parameters. Runtime polymorphism (overriding) allows objects to behave differently based on their actual type, enabling flexible and extensible code. Understanding upcasting and downcasting is essential for working with inheritance hierarchies and designing robust systems that can handle multiple types uniformly.`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.*;

// Test 1: Payment abstract class exists
class Test1 {
    @Test
    void testPaymentClassExists() throws Exception {
        Class<?> cls = Class.forName("Payment");
        assertTrue(Modifier.isAbstract(cls.getModifiers()));
    }
}

// Test 2: CreditCardPayment extends Payment
class Test2 {
    @Test
    void testCreditCardExtendsPayment() throws Exception {
        Class<?> payment = Class.forName("Payment");
        Class<?> cc = Class.forName("CreditCardPayment");
        assertTrue(payment.isAssignableFrom(cc));
    }
}

// Test 3: PayPalPayment extends Payment
class Test3 {
    @Test
    void testPayPalExtendsPayment() throws Exception {
        Class<?> payment = Class.forName("Payment");
        Class<?> paypal = Class.forName("PayPalPayment");
        assertTrue(payment.isAssignableFrom(paypal));
    }
}

// Test 4: Payment has abstract processPayment method
class Test4 {
    @Test
    void testProcessPaymentAbstract() throws Exception {
        Class<?> cls = Class.forName("Payment");
        Method method = cls.getMethod("processPayment");
        assertTrue(Modifier.isAbstract(method.getModifiers()));
    }
}

// Test 5: CreditCardPayment has overloaded processPayment methods
class Test5 {
    @Test
    void testCreditCardOverloadedMethods() throws Exception {
        Class<?> cls = Class.forName("CreditCardPayment");
        Method m1 = cls.getMethod("processPayment");
        Method m2 = cls.getMethod("processPayment", boolean.class);
        Method m3 = cls.getMethod("processPayment", int.class);
        assertNotNull(m1);
        assertNotNull(m2);
        assertNotNull(m3);
    }
}

// Test 6: PaymentProcessor exists with overloaded methods
class Test6 {
    @Test
    void testPaymentProcessorOverloading() throws Exception {
        Class<?> cls = Class.forName("PaymentProcessor");
        Class<?> payment = Class.forName("Payment");
        Method m1 = cls.getMethod("processPayment", payment);
        Method m2 = cls.getMethod("processPayment", payment.arrayType());
        Method m3 = cls.getMethod("processPayment", payment, boolean.class);
        assertNotNull(m1);
        assertNotNull(m2);
        assertNotNull(m3);
    }
}

// Test 7: CreditCardPayment has cardNumber field
class Test7 {
    @Test
    void testCreditCardHasFields() throws Exception {
        Class<?> cls = Class.forName("CreditCardPayment");
        Field cardNumber = cls.getDeclaredField("cardNumber");
        Field cvv = cls.getDeclaredField("cvv");
        assertNotNull(cardNumber);
        assertNotNull(cvv);
    }
}

// Test 8: PayPalPayment has email field
class Test8 {
    @Test
    void testPayPalHasEmail() throws Exception {
        Class<?> cls = Class.forName("PayPalPayment");
        Field email = cls.getDeclaredField("email");
        assertNotNull(email);
    }
}

// Test 9: Payment has displayAmount method
class Test9 {
    @Test
    void testPaymentHasDisplayAmount() throws Exception {
        Class<?> cls = Class.forName("Payment");
        Method method = cls.getMethod("displayAmount");
        assertNotNull(method);
        assertFalse(Modifier.isAbstract(method.getModifiers()));
    }
}

// Test 10: Polymorphic behavior works
class Test10 {
    @Test
    void testPolymorphicBehavior() throws Exception {
        Class<?> ccClass = Class.forName("CreditCardPayment");
        Class<?> ppClass = Class.forName("PayPalPayment");
        Class<?> payment = Class.forName("Payment");
        Constructor<?> ccCon = ccClass.getConstructor(double.class, String.class, String.class);
        Constructor<?> ppCon = ppClass.getConstructor(double.class, String.class);
        Object cc = ccCon.newInstance(100.0, "1234567890123456", "123");
        Object pp = ppCon.newInstance(50.0, "user@email.com");
        assertTrue(payment.isInstance(cc));
        assertTrue(payment.isInstance(pp));
    }
}`,
    translations: {
        ru: {
            title: 'Полиморфизм в Действии',
            solutionCode: `// Абстрактный базовый класс - определяет общую структуру платежа
abstract class Payment {
    protected double amount;

    public Payment(double amount) {
        this.amount = amount;
    }

    // Абстрактный метод - должен быть реализован в подклассах
    public abstract void processPayment();

    // Конкретный метод - наследуется всеми подклассами
    public void displayAmount() {
        System.out.println("Сумма платежа: $" + amount);
    }
}

// CreditCardPayment - демонстрирует переопределение и перегрузку методов
class CreditCardPayment extends Payment {
    private String cardNumber;
    private String cvv;

    public CreditCardPayment(double amount, String cardNumber, String cvv) {
        super(amount);
        this.cardNumber = cardNumber;
        this.cvv = cvv;
    }

    // Полиморфизм времени выполнения - переопределение абстрактного метода
    @Override
    public void processPayment() {
        System.out.println("Обработка платежа по кредитной карте...");
        System.out.println("Карта: " + maskCardNumber(cardNumber));
        System.out.println("Сумма: $" + amount);
        System.out.println("Платеж одобрен!");
    }

    // Полиморфизм времени компиляции - перегрузка метода
    public void processPayment(boolean requireSignature) {
        processPayment();
        if (requireSignature) {
            System.out.println("Требуется подпись.");
        }
    }

    // Полиморфизм времени компиляции - еще один перегруженный метод
    public void processPayment(int installments) {
        processPayment();
        System.out.println("Платеж разделен на " + installments + " взносов");
    }

    private String maskCardNumber(String cardNumber) {
        return "**** **** **** " + cardNumber.substring(cardNumber.length() - 4);
    }
}

// PayPalPayment - демонстрирует переопределение методов
class PayPalPayment extends Payment {
    private String email;

    public PayPalPayment(double amount, String email) {
        super(amount);
        this.email = email;
    }

    // Полиморфизм времени выполнения - переопределение абстрактного метода
    @Override
    public void processPayment() {
        System.out.println("Обработка платежа PayPal...");
        System.out.println("Аккаунт PayPal: " + email);
        System.out.println("Сумма: $" + amount);
        System.out.println("Платеж отправлен!");
    }
}

// PaymentProcessor - демонстрирует полиморфизм времени компиляции
class PaymentProcessor {
    // Перегруженный метод #1 - одиночный платеж
    public void processPayment(Payment payment) {
        System.out.println("=== Обработка одного платежа ===");
        payment.processPayment();
        System.out.println("================================\n");
    }

    // Перегруженный метод #2 - пакетные платежи
    public void processPayment(Payment[] payments) {
        System.out.println("=== Пакетная обработка платежей ===");
        for (Payment payment : payments) {
            payment.processPayment();
            System.out.println("---");
        }
        System.out.println("================================\n");
    }

    // Перегруженный метод #3 - платеж с экспресс-опцией
    public void processPayment(Payment payment, boolean express) {
        System.out.println("=== " + (express ? "Экспресс" : "Стандартная") +
                          " обработка платежа ===");
        payment.processPayment();
        if (express) {
            System.out.println("Экспресс-обработка - завершена за 1 час");
        }
        System.out.println("================================\n");
    }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        PaymentProcessor processor = new PaymentProcessor();

        // Восходящее приведение - дочерний объект по ссылке родительского типа
        Payment payment1 = new CreditCardPayment(150.00, "1234567890123456", "123");
        Payment payment2 = new PayPalPayment(75.50, "user@email.com");

        // Полиморфизм времени выполнения - вызывается правильный метод на основе фактического типа объекта
        processor.processPayment(payment1);
        processor.processPayment(payment2);

        // Полиморфизм времени компиляции - перегруженный метод с массивом
        Payment[] payments = {
            new CreditCardPayment(100.00, "9876543210987654", "456"),
            new PayPalPayment(200.00, "buyer@email.com"),
            new CreditCardPayment(50.00, "1111222233334444", "789")
        };
        processor.processPayment(payments);

        // Полиморфизм времени компиляции - перегруженный метод с boolean
        Payment expressPayment = new PayPalPayment(500.00, "express@email.com");
        processor.processPayment(expressPayment, true);

        // Нисходящее приведение с проверкой instanceof - безопасное преобразование типа
        if (payment1 instanceof CreditCardPayment) {
            System.out.println("=== Демонстрация нисходящего приведения ===");
            CreditCardPayment ccPayment = (CreditCardPayment) payment1;
            // Теперь можно получить доступ к специфичным для CreditCardPayment перегруженным методам
            ccPayment.processPayment(3); // Обработка с рассрочкой
            ccPayment.processPayment(true); // Обработка с подписью
            System.out.println("================================\n");
        }

        // Демонстрация полиморфного массива
        System.out.println("=== Полиморфное поведение ===");
        for (Payment p : payments) {
            p.displayAmount(); // Общий метод
            p.processPayment(); // Полиморфное поведение
            System.out.println("Тип: " + p.getClass().getSimpleName());
            System.out.println("---");
        }
    }
}`,
            description: `Создайте систему обработки платежей, которая демонстрирует полиморфизм времени компиляции и времени выполнения в Java.

**Требования:**
1. Создайте абстрактный базовый класс **Payment** с:
   1.1. Защищенным полем: amount (double)
   1.2. Конструктором для инициализации amount
   1.3. Абстрактным методом: processPayment()
   1.4. Конкретным методом: displayAmount()

2. Создайте класс **CreditCardPayment**, расширяющий Payment:
   2.1. Дополнительные поля: cardNumber, cvv
   2.2. Переопределите processPayment() с логикой кредитной карты
   2.3. Продемонстрируйте перегрузку методов с несколькими методами обработки

3. Создайте класс **PayPalPayment**, расширяющий Payment:
   3.1. Дополнительное поле: email
   3.2. Переопределите processPayment() с логикой PayPal

4. Создайте класс **PaymentProcessor** с:
   4.1. Перегруженными методами, демонстрирующими полиморфизм времени компиляции:
        - processPayment(Payment payment)
        - processPayment(Payment[] payments)
        - processPayment(Payment payment, boolean express)

5. В методе main:
   5.1. Продемонстрируйте восходящее приведение (ссылка родителя на дочерний объект)
   5.2. Продемонстрируйте нисходящее приведение с проверкой instanceof
   5.3. Покажите полиморфное поведение с различными типами платежей
   5.4. Используйте перегруженные методы

**Цели обучения:**
- Понять полиморфизм времени компиляции (перегрузка методов)
- Понять полиморфизм времени выполнения (переопределение методов)
- Практиковать восходящее и нисходящее приведение
- Научиться безопасно использовать оператор instanceof
- Увидеть преимущества полиморфизма в реальных сценариях`,
            hint1: `Для полиморфизма времени компиляции создайте несколько методов с одинаковым именем, но разными параметрами. Для полиморфизма времени выполнения переопределяйте методы в подклассах.`,
            hint2: `Всегда используйте instanceof перед нисходящим приведением, чтобы избежать ClassCastException. Восходящее приведение автоматическое и безопасное, но нисходящее приведение требует явного приведения и проверки.`,
            whyItMatters: `Полиморфизм - это одна из самых мощных возможностей ООП. Полиморфизм времени компиляции (перегрузка) обеспечивает гибкость методов с различными параметрами. Полиморфизм времени выполнения (переопределение) позволяет объектам вести себя по-разному в зависимости от их фактического типа, обеспечивая гибкий и расширяемый код. Понимание восходящего и нисходящего приведения необходимо для работы с иерархиями наследования и проектирования надежных систем, которые могут единообразно обрабатывать несколько типов.`
        },
        uz: {
            title: 'Polimorfizm Amalda',
            solutionCode: `// Abstrakt asosiy sinf - umumiy to'lov strukturasini belgilaydi
abstract class Payment {
    protected double amount;

    public Payment(double amount) {
        this.amount = amount;
    }

    // Abstrakt metod - subsinflar tomonidan amalga oshirilishi kerak
    public abstract void processPayment();

    // Konkret metod - barcha subsinflar tomonidan meros olinadi
    public void displayAmount() {
        System.out.println("To'lov summasi: $" + amount);
    }
}

// CreditCardPayment - metodlarni qayta yozish va ortiqcha yuklashni ko'rsatadi
class CreditCardPayment extends Payment {
    private String cardNumber;
    private String cvv;

    public CreditCardPayment(double amount, String cardNumber, String cvv) {
        super(amount);
        this.cardNumber = cardNumber;
        this.cvv = cvv;
    }

    // Bajarilish vaqti polimorfizmi - abstrakt metodini qayta yozish
    @Override
    public void processPayment() {
        System.out.println("Kredit karta to'lovi qayta ishlanmoqda...");
        System.out.println("Karta: " + maskCardNumber(cardNumber));
        System.out.println("Summa: $" + amount);
        System.out.println("To'lov tasdiqlandi!");
    }

    // Kompilyatsiya vaqti polimorfizmi - metodini ortiqcha yuklash
    public void processPayment(boolean requireSignature) {
        processPayment();
        if (requireSignature) {
            System.out.println("Imzo talab qilinadi.");
        }
    }

    // Kompilyatsiya vaqti polimorfizmi - yana bir ortiqcha yuklangan metod
    public void processPayment(int installments) {
        processPayment();
        System.out.println("To'lov " + installments + " bo'limga bo'lindi");
    }

    private String maskCardNumber(String cardNumber) {
        return "**** **** **** " + cardNumber.substring(cardNumber.length() - 4);
    }
}

// PayPalPayment - metodlarni qayta yozishni ko'rsatadi
class PayPalPayment extends Payment {
    private String email;

    public PayPalPayment(double amount, String email) {
        super(amount);
        this.email = email;
    }

    // Bajarilish vaqti polimorfizmi - abstrakt metodini qayta yozish
    @Override
    public void processPayment() {
        System.out.println("PayPal to'lovi qayta ishlanmoqda...");
        System.out.println("PayPal hisobi: " + email);
        System.out.println("Summa: $" + amount);
        System.out.println("To'lov yuborildi!");
    }
}

// PaymentProcessor - kompilyatsiya vaqti polimorfizmini ko'rsatadi
class PaymentProcessor {
    // Ortiqcha yuklangan metod #1 - bitta to'lov
    public void processPayment(Payment payment) {
        System.out.println("=== Bitta To'lovni Qayta Ishlash ===");
        payment.processPayment();
        System.out.println("================================\n");
    }

    // Ortiqcha yuklangan metod #2 - to'plamli to'lovlar
    public void processPayment(Payment[] payments) {
        System.out.println("=== To'plamli To'lovlarni Qayta Ishlash ===");
        for (Payment payment : payments) {
            payment.processPayment();
            System.out.println("---");
        }
        System.out.println("================================\n");
    }

    // Ortiqcha yuklangan metod #3 - ekspress variantli to'lov
    public void processPayment(Payment payment, boolean express) {
        System.out.println("=== " + (express ? "Ekspress" : "Standart") +
                          " To'lovni Qayta Ishlash ===");
        payment.processPayment();
        if (express) {
            System.out.println("Ekspress qayta ishlash - 1 soatda yakunlandi");
        }
        System.out.println("================================\n");
    }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        PaymentProcessor processor = new PaymentProcessor();

        // Yuqoriga kasting - bola obyekti ota-ona turi bo'yicha havola qilingan
        Payment payment1 = new CreditCardPayment(150.00, "1234567890123456", "123");
        Payment payment2 = new PayPalPayment(75.50, "user@email.com");

        // Bajarilish vaqti polimorfizmi - haqiqiy obyekt turiga asoslanib to'g'ri metod chaqiriladi
        processor.processPayment(payment1);
        processor.processPayment(payment2);

        // Kompilyatsiya vaqti polimorfizmi - massiv bilan ortiqcha yuklangan metod
        Payment[] payments = {
            new CreditCardPayment(100.00, "9876543210987654", "456"),
            new PayPalPayment(200.00, "buyer@email.com"),
            new CreditCardPayment(50.00, "1111222233334444", "789")
        };
        processor.processPayment(payments);

        // Kompilyatsiya vaqti polimorfizmi - boolean bilan ortiqcha yuklangan metod
        Payment expressPayment = new PayPalPayment(500.00, "express@email.com");
        processor.processPayment(expressPayment, true);

        // instanceof tekshiruvi bilan pastga kasting - xavfsiz tur konvertatsiyasi
        if (payment1 instanceof CreditCardPayment) {
            System.out.println("=== Pastga Kastingni Ko'rsatish ===");
            CreditCardPayment ccPayment = (CreditCardPayment) payment1;
            // Endi CreditCardPayment ga xos ortiqcha yuklangan metodlarga kirish mumkin
            ccPayment.processPayment(3); // Bo'lib to'lash bilan qayta ishlash
            ccPayment.processPayment(true); // Imzo bilan qayta ishlash
            System.out.println("================================\n");
        }

        // Polimorf massivni ko'rsatish
        System.out.println("=== Polimorf Xatti-harakat ===");
        for (Payment p : payments) {
            p.displayAmount(); // Umumiy metod
            p.processPayment(); // Polimorf xatti-harakat
            System.out.println("Turi: " + p.getClass().getSimpleName());
            System.out.println("---");
        }
    }
}`,
            description: `Java-da kompilyatsiya vaqti va bajarilish vaqti polimorfizmini ko'rsatadigan to'lovlarni qayta ishlash tizimini yarating.

**Talablar:**
1. Quyidagilar bilan abstrakt **Payment** asosiy sinfini yarating:
   1.1. Himoyalangan maydon: amount (double)
   1.2. amount ni ishga tushirish uchun konstruktor
   1.3. Abstrakt metod: processPayment()
   1.4. Konkret metod: displayAmount()

2. Payment ni kengaytiradigan **CreditCardPayment** sinfini yarating:
   2.1. Qo'shimcha maydonlar: cardNumber, cvv
   2.2. Kredit karta mantiqiga xos processPayment() ni qayta yozing
   2.3. Bir nechta jarayon metodlari bilan metodlarni ortiqcha yuklashni ko'rsating

3. Payment ni kengaytiradigan **PayPalPayment** sinfini yarating:
   3.1. Qo'shimcha maydon: email
   3.2. PayPal mantiqiga xos processPayment() ni qayta yozing

4. Quyidagilar bilan **PaymentProcessor** sinfini yarating:
   4.1. Kompilyatsiya vaqti polimorfizmini ko'rsatadigan ortiqcha yuklangan metodlar:
        - processPayment(Payment payment)
        - processPayment(Payment[] payments)
        - processPayment(Payment payment, boolean express)

5. Main metodida:
   5.1. Yuqoriga kastingni ko'rsating (ota-ona havolasi bola obyektiga)
   5.2. instanceof tekshiruvi bilan pastga kastingni ko'rsating
   5.3. Turli to'lov turlari bilan polimorf xatti-harakatni ko'rsating
   5.4. Ortiqcha yuklangan metodlardan foydalaning

**O'rganish maqsadlari:**
- Kompilyatsiya vaqti polimorfizmini tushunish (metodlarni ortiqcha yuklash)
- Bajarilish vaqti polimorfizmini tushunish (metodlarni qayta yozish)
- Yuqoriga va pastga kastingda amaliyot
- instanceof operatoridan xavfsiz foydalanishni o'rganish
- Haqiqiy dunyodagi stsenariylarda polimorfizmning afzalliklarini ko'rish`,
            hint1: `Kompilyatsiya vaqti polimorfizmi uchun bir xil nomli, lekin turli parametrlarga ega bir nechta metodlar yarating. Bajarilish vaqti polimorfizmi uchun subsinflar ichida metodlarni qayta yozing.`,
            hint2: `ClassCastException dan qochish uchun pastga kastingdan oldin har doim instanceof dan foydalaning. Yuqoriga kasting avtomatik va xavfsiz, lekin pastga kasting aniq kasting va tekshirishni talab qiladi.`,
            whyItMatters: `Polimorfizm OOP ning eng kuchli xususiyatlaridan biridir. Kompilyatsiya vaqti polimorfizmi (ortiqcha yuklash) turli parametrlar bilan metod moslashuvchanligini ta'minlaydi. Bajarilish vaqti polimorfizmi (qayta yozish) obyektlarga ularning haqiqiy turiga qarab turlicha harakat qilish imkonini beradi, bu esa moslashuvchan va kengaytiriladigan kodni ta'minlaydi. Yuqoriga va pastga kastingni tushunish meros ierarxiyalari bilan ishlash va bir nechta turlarni bir xilda qayta ishlashi mumkin bo'lgan mustahkam tizimlarni loyihalash uchun zarurdir.`
        }
    }
};

export default task;
