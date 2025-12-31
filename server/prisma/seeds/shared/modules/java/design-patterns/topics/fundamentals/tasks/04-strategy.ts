import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-strategy-pattern',
    title: 'Strategy Pattern',
    difficulty: 'medium',
    tags: ['java', 'design-patterns', 'behavioral', 'strategy', 'lambda'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Strategy Pattern

The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it. In modern Java, strategies can be implemented using lambda expressions.

## Requirements:
1. Create payment processing system:
   1.1. PaymentStrategy interface with pay() method
   1.2. Concrete strategies: CreditCard, PayPal, Bitcoin

2. Create ShoppingCart that uses strategies:
   2.1. Set payment strategy dynamically
   2.2. Process payment using selected strategy

3. Demonstrate with traditional classes:
   3.1. Create concrete strategy classes
   3.2. Switch between different strategies

4. Demonstrate with lambda expressions:
   4.1. Use lambdas as strategies
   4.2. Show the modern, concise approach

## Example Output:
\`\`\`
=== Traditional Strategy Pattern ===
Shopping Cart Total: $150.00
Processing payment with Credit Card
Paid $150.00 using Credit Card ending in 1234

Changing payment method...
Processing payment with PayPal
Paid $150.00 using PayPal account user@example.com

=== Strategy with Lambdas ===
Bitcoin payment: Transferring $150.00 to wallet 1A2B3C
Discount strategy: Applied 10% discount. Final amount: $135.00
\`\`\``,
    initialCode: `// TODO: Create PaymentStrategy interface

// TODO: Create concrete payment strategies

// TODO: Create ShoppingCart class

public class StrategyPattern {
    public static void main(String[] args) {
        // TODO: Demonstrate traditional strategy pattern

        // TODO: Demonstrate strategy with lambdas
    }
}`,
    solutionCode: `// Strategy interface
interface PaymentStrategy {
    void pay(double amount);
}

// Concrete strategy 1
class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Credit Card");
        System.out.printf("Paid $%.2f using Credit Card ending in %s%n",
            amount, cardNumber.substring(cardNumber.length() - 4));
    }
}

// Concrete strategy 2
class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with PayPal");
        System.out.printf("Paid $%.2f using PayPal account %s%n", amount, email);
    }
}

// Concrete strategy 3
class BitcoinPayment implements PaymentStrategy {
    private String walletAddress;

    public BitcoinPayment(String walletAddress) {
        this.walletAddress = walletAddress;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Bitcoin");
        System.out.printf("Paid $%.2f to Bitcoin wallet %s%n", amount, walletAddress);
    }
}

// Context class that uses strategy
class ShoppingCart {
    private double total;
    private PaymentStrategy paymentStrategy;

    public ShoppingCart(double total) {
        this.total = total;
    }

    // Strategy can be changed at runtime
    public void setPaymentStrategy(PaymentStrategy paymentStrategy) {
        this.paymentStrategy = paymentStrategy;
    }

    public void checkout() {
        if (paymentStrategy == null) {
            System.out.println("Please select a payment method!");
            return;
        }
        System.out.printf("Shopping Cart Total: $%.2f%n", total);
        paymentStrategy.pay(total);
    }

    public double getTotal() {
        return total;
    }
}

public class StrategyPattern {
    public static void main(String[] args) {
        System.out.println("=== Traditional Strategy Pattern ===");

        // Create shopping cart
        ShoppingCart cart = new ShoppingCart(150.00);

        // Use credit card strategy
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678"));
        cart.checkout();

        System.out.println("\\nChanging payment method...");

        // Switch to PayPal strategy
        cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
        cart.checkout();

        System.out.println("\\n=== Strategy with Lambdas ===");

        // Strategy as lambda expression
        PaymentStrategy bitcoinStrategy = (amount) -> {
            System.out.printf("Bitcoin payment: Transferring $%.2f to wallet 1A2B3C%n", amount);
        };

        ShoppingCart cart2 = new ShoppingCart(150.00);
        cart2.setPaymentStrategy(bitcoinStrategy);
        cart2.checkout();

        // Another lambda strategy - discount calculation
        PaymentStrategy discountStrategy = (amount) -> {
            double discount = amount * 0.10;
            double finalAmount = amount - discount;
            System.out.printf("Discount strategy: Applied 10%% discount. Final amount: $%.2f%n",
                finalAmount);
        };

        ShoppingCart cart3 = new ShoppingCart(150.00);
        cart3.setPaymentStrategy(discountStrategy);
        cart3.checkout();
    }
}`,
    hint1: `Create a PaymentStrategy interface with a single method. Concrete strategies implement this interface with their specific payment logic.`,
    hint2: `For lambda-based strategies, since PaymentStrategy is a functional interface (single abstract method), you can use lambdas: PaymentStrategy strategy = (amount) -> { /* implementation */ };`,
    whyItMatters: `Strategy pattern is essential for making algorithms interchangeable and avoiding conditional logic. It promotes Open/Closed Principle - you can add new strategies without modifying existing code. With Java 8+ lambdas, strategies become even more powerful and concise. This pattern is used extensively in frameworks, sorting algorithms, validation, and business rule engines.

**Production Pattern:**
\`\`\`java
// Traditional approach
PaymentStrategy creditCard = new CreditCardPayment("1234-5678");
cart.setPaymentStrategy(creditCard);
cart.checkout();

// Modern approach with lambdas
PaymentStrategy bitcoin = (amount) -> {
    System.out.println("Transferring $" + amount + " to wallet");
};
cart.setPaymentStrategy(bitcoin);

// Dynamic strategy switching
cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
\`\`\`

**Practical Benefits:**
- Avoids large if-else or switch constructs
- New strategies added without changing existing code
- Lambda expressions make strategies concise
- Used in Collections.sort(), Comparator, Stream API`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;

// Test 1: CreditCardPayment processes payment
class Test1 {
    @Test
    void testCreditCardPayment() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        PaymentStrategy payment = new CreditCardPayment("1234567812345678");
        payment.pay(100.0);

        String output = outContent.toString();
        assertTrue(output.contains("Credit Card"));
        assertTrue(output.contains("5678"));
    }
}

// Test 2: PayPalPayment processes payment
class Test2 {
    @Test
    void testPayPalPayment() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        PaymentStrategy payment = new PayPalPayment("user@example.com");
        payment.pay(100.0);

        String output = outContent.toString();
        assertTrue(output.contains("PayPal"));
        assertTrue(output.contains("user@example.com"));
    }
}

// Test 3: BitcoinPayment processes payment
class Test3 {
    @Test
    void testBitcoinPayment() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        PaymentStrategy payment = new BitcoinPayment("1A2B3C");
        payment.pay(100.0);

        String output = outContent.toString();
        assertTrue(output.contains("Bitcoin"));
        assertTrue(output.contains("1A2B3C"));
    }
}

// Test 4: ShoppingCart total is set correctly
class Test4 {
    @Test
    void testShoppingCartTotal() {
        ShoppingCart cart = new ShoppingCart(150.00);
        assertEquals(150.00, cart.getTotal(), 0.01);
    }
}

// Test 5: ShoppingCart checkout without strategy
class Test5 {
    @Test
    void testShoppingCartCheckoutWithoutStrategy() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        ShoppingCart cart = new ShoppingCart(100.0);
        cart.checkout();

        String output = outContent.toString();
        assertTrue(output.contains("select") || output.contains("Please"));
    }
}

// Test 6: ShoppingCart uses set strategy
class Test6 {
    @Test
    void testShoppingCartUsesStrategy() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        ShoppingCart cart = new ShoppingCart(100.0);
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678"));
        cart.checkout();

        String output = outContent.toString();
        assertTrue(output.contains("Credit Card"));
    }
}

// Test 7: Strategy can be changed at runtime
class Test7 {
    @Test
    void testStrategyCanBeChanged() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        ShoppingCart cart = new ShoppingCart(100.0);
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678"));
        cart.setPaymentStrategy(new PayPalPayment("test@test.com"));
        cart.checkout();

        String output = outContent.toString();
        assertTrue(output.contains("PayPal"));
    }
}

// Test 8: Lambda strategy works
class Test8 {
    @Test
    void testLambdaStrategy() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        PaymentStrategy lambda = (amount) -> System.out.println("Lambda paid: " + amount);
        ShoppingCart cart = new ShoppingCart(50.0);
        cart.setPaymentStrategy(lambda);
        cart.checkout();

        String output = outContent.toString();
        assertTrue(output.contains("Lambda paid"));
    }
}

// Test 9: PaymentStrategy is functional interface
class Test9 {
    @Test
    void testPaymentStrategyIsFunctionalInterface() {
        PaymentStrategy strategy = (amount) -> {};
        assertNotNull(strategy);
    }
}

// Test 10: Multiple carts with different strategies
class Test10 {
    @Test
    void testMultipleCartsWithDifferentStrategies() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        ShoppingCart cart1 = new ShoppingCart(100.0);
        ShoppingCart cart2 = new ShoppingCart(200.0);

        cart1.setPaymentStrategy(new CreditCardPayment("1111222233334444"));
        cart2.setPaymentStrategy(new PayPalPayment("user@test.com"));

        cart1.checkout();
        cart2.checkout();

        String output = outContent.toString();
        assertTrue(output.contains("Credit Card"));
        assertTrue(output.contains("PayPal"));
    }
}`,
    translations: {
        ru: {
            title: 'Паттерн Strategy',
            solutionCode: `// Интерфейс стратегии
interface PaymentStrategy {
    void pay(double amount);
}

// Конкретная стратегия 1
class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Credit Card");
        System.out.printf("Paid $%.2f using Credit Card ending in %s%n",
            amount, cardNumber.substring(cardNumber.length() - 4));
    }
}

// Конкретная стратегия 2
class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with PayPal");
        System.out.printf("Paid $%.2f using PayPal account %s%n", amount, email);
    }
}

// Конкретная стратегия 3
class BitcoinPayment implements PaymentStrategy {
    private String walletAddress;

    public BitcoinPayment(String walletAddress) {
        this.walletAddress = walletAddress;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Bitcoin");
        System.out.printf("Paid $%.2f to Bitcoin wallet %s%n", amount, walletAddress);
    }
}

// Класс контекста, использующий стратегию
class ShoppingCart {
    private double total;
    private PaymentStrategy paymentStrategy;

    public ShoppingCart(double total) {
        this.total = total;
    }

    // Стратегия может быть изменена во время выполнения
    public void setPaymentStrategy(PaymentStrategy paymentStrategy) {
        this.paymentStrategy = paymentStrategy;
    }

    public void checkout() {
        if (paymentStrategy == null) {
            System.out.println("Please select a payment method!");
            return;
        }
        System.out.printf("Shopping Cart Total: $%.2f%n", total);
        paymentStrategy.pay(total);
    }

    public double getTotal() {
        return total;
    }
}

public class StrategyPattern {
    public static void main(String[] args) {
        System.out.println("=== Традиционный паттерн Strategy ===");

        // Создание корзины покупок
        ShoppingCart cart = new ShoppingCart(150.00);

        // Использование стратегии кредитной карты
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678"));
        cart.checkout();

        System.out.println("\\nChanging payment method...");

        // Переключение на стратегию PayPal
        cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
        cart.checkout();

        System.out.println("\\n=== Стратегия с лямбдами ===");

        // Стратегия как лямбда-выражение
        PaymentStrategy bitcoinStrategy = (amount) -> {
            System.out.printf("Bitcoin payment: Transferring $%.2f to wallet 1A2B3C%n", amount);
        };

        ShoppingCart cart2 = new ShoppingCart(150.00);
        cart2.setPaymentStrategy(bitcoinStrategy);
        cart2.checkout();

        // Еще одна лямбда стратегия - расчет скидки
        PaymentStrategy discountStrategy = (amount) -> {
            double discount = amount * 0.10;
            double finalAmount = amount - discount;
            System.out.printf("Discount strategy: Applied 10%% discount. Final amount: $%.2f%n",
                finalAmount);
        };

        ShoppingCart cart3 = new ShoppingCart(150.00);
        cart3.setPaymentStrategy(discountStrategy);
        cart3.checkout();
    }
}`,
            description: `# Паттерн Strategy

Паттерн Strategy определяет семейство алгоритмов, инкапсулирует каждый из них и делает их взаимозаменяемыми. Strategy позволяет алгоритму изменяться независимо от клиентов, которые его используют. В современной Java стратегии могут быть реализованы с использованием лямбда-выражений.

## Требования:
1. Создайте систему обработки платежей:
   1.1. Интерфейс PaymentStrategy с методом pay()
   1.2. Конкретные стратегии: CreditCard, PayPal, Bitcoin

2. Создайте ShoppingCart, использующий стратегии:
   2.1. Динамически устанавливайте стратегию платежа
   2.2. Обрабатывайте платеж используя выбранную стратегию

3. Продемонстрируйте с традиционными классами:
   3.1. Создайте конкретные классы стратегий
   3.2. Переключайтесь между различными стратегиями

4. Продемонстрируйте с лямбда-выражениями:
   4.1. Используйте лямбды как стратегии
   4.2. Покажите современный, краткий подход

## Пример вывода:
\`\`\`
=== Traditional Strategy Pattern ===
Shopping Cart Total: $150.00
Processing payment with Credit Card
Paid $150.00 using Credit Card ending in 1234

Changing payment method...
Processing payment with PayPal
Paid $150.00 using PayPal account user@example.com

=== Strategy with Lambdas ===
Bitcoin payment: Transferring $150.00 to wallet 1A2B3C
Discount strategy: Applied 10% discount. Final amount: $135.00
\`\`\``,
            hint1: `Создайте интерфейс PaymentStrategy с единственным методом. Конкретные стратегии реализуют этот интерфейс со своей специфической логикой платежа.`,
            hint2: `Для стратегий на основе лямбд, поскольку PaymentStrategy является функциональным интерфейсом (один абстрактный метод), вы можете использовать лямбды: PaymentStrategy strategy = (amount) -> { /* реализация */ };`,
            whyItMatters: `Паттерн Strategy необходим для того, чтобы сделать алгоритмы взаимозаменяемыми и избежать условной логики. Он продвигает принцип Open/Closed - вы можете добавлять новые стратегии без изменения существующего кода.

**Продакшен паттерн:**
\`\`\`java
// Традиционный подход
PaymentStrategy creditCard = new CreditCardPayment("1234-5678");
cart.setPaymentStrategy(creditCard);
cart.checkout();

// Современный подход с лямбдами
PaymentStrategy bitcoin = (amount) -> {
    System.out.println("Transferring $" + amount + " to wallet");
};
cart.setPaymentStrategy(bitcoin);

// Динамическое переключение стратегий
cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
\`\`\`

**Практические преимущества:**
- Избегает больших if-else или switch конструкций
- Новые стратегии добавляются без изменения существующего кода
- Лямбда-выражения делают стратегии лаконичными
- Используется в Collections.sort(), Comparator, Stream API`
        },
        uz: {
            title: `Strategy namunasi`,
            solutionCode: `// Strategiya interfeysi
interface PaymentStrategy {
    void pay(double amount);
}

// Aniq strategiya 1
class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Credit Card");
        System.out.printf("Paid $%.2f using Credit Card ending in %s%n",
            amount, cardNumber.substring(cardNumber.length() - 4));
    }
}

// Aniq strategiya 2
class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with PayPal");
        System.out.printf("Paid $%.2f using PayPal account %s%n", amount, email);
    }
}

// Aniq strategiya 3
class BitcoinPayment implements PaymentStrategy {
    private String walletAddress;

    public BitcoinPayment(String walletAddress) {
        this.walletAddress = walletAddress;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Processing payment with Bitcoin");
        System.out.printf("Paid $%.2f to Bitcoin wallet %s%n", amount, walletAddress);
    }
}

// Strategiyadan foydalanadigan kontekst klassi
class ShoppingCart {
    private double total;
    private PaymentStrategy paymentStrategy;

    public ShoppingCart(double total) {
        this.total = total;
    }

    // Strategiya ish vaqtida o'zgartirilishi mumkin
    public void setPaymentStrategy(PaymentStrategy paymentStrategy) {
        this.paymentStrategy = paymentStrategy;
    }

    public void checkout() {
        if (paymentStrategy == null) {
            System.out.println("Please select a payment method!");
            return;
        }
        System.out.printf("Shopping Cart Total: $%.2f%n", total);
        paymentStrategy.pay(total);
    }

    public double getTotal() {
        return total;
    }
}

public class StrategyPattern {
    public static void main(String[] args) {
        System.out.println("=== An'anaviy Strategy namunasi ===");

        // Xarid savatchasini yaratish
        ShoppingCart cart = new ShoppingCart(150.00);

        // Kredit karta strategiyasidan foydalanish
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678"));
        cart.checkout();

        System.out.println("\\nChanging payment method...");

        // PayPal strategiyasiga o'tish
        cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
        cart.checkout();

        System.out.println("\\n=== Lambda bilan strategiya ===");

        // Lambda ifoda sifatida strategiya
        PaymentStrategy bitcoinStrategy = (amount) -> {
            System.out.printf("Bitcoin payment: Transferring $%.2f to wallet 1A2B3C%n", amount);
        };

        ShoppingCart cart2 = new ShoppingCart(150.00);
        cart2.setPaymentStrategy(bitcoinStrategy);
        cart2.checkout();

        // Yana bir lambda strategiya - chegirma hisoblash
        PaymentStrategy discountStrategy = (amount) -> {
            double discount = amount * 0.10;
            double finalAmount = amount - discount;
            System.out.printf("Discount strategy: Applied 10%% discount. Final amount: $%.2f%n",
                finalAmount);
        };

        ShoppingCart cart3 = new ShoppingCart(150.00);
        cart3.setPaymentStrategy(discountStrategy);
        cart3.checkout();
    }
}`,
            description: `# Strategy namunasi

Strategy namunasi algoritmlar oilasini belgilaydi, har birini inkapsulyatsiya qiladi va ularni o'zaro almashtiriladigan qiladi. Strategy algoritmning undan foydalanadigan mijozlardan mustaqil ravishda o'zgarishiga imkon beradi. Zamonaviy Java-da strategiyalar lambda ifodalari yordamida amalga oshirilishi mumkin.

## Talablar:
1. To'lov qayta ishlash tizimini yarating:
   1.1. pay() metodi bilan PaymentStrategy interfeysi
   1.2. Aniq strategiyalar: CreditCard, PayPal, Bitcoin

2. Strategiyalardan foydalanadigan ShoppingCart yarating:
   2.1. To'lov strategiyasini dinamik ravishda o'rnating
   2.2. Tanlangan strategiya yordamida to'lovni qayta ishlang

3. An'anaviy klasslar bilan namoyish eting:
   3.1. Aniq strategiya klasslarini yarating
   3.2. Turli strategiyalar o'rtasida almashtiring

4. Lambda ifodalari bilan namoyish eting:
   4.1. Lambdalarni strategiya sifatida ishlating
   4.2. Zamonaviy, qisqa yondashuvni ko'rsating

## Chiqish namunasi:
\`\`\`
=== Traditional Strategy Pattern ===
Shopping Cart Total: $150.00
Processing payment with Credit Card
Paid $150.00 using Credit Card ending in 1234

Changing payment method...
Processing payment with PayPal
Paid $150.00 using PayPal account user@example.com

=== Strategy with Lambdas ===
Bitcoin payment: Transferring $150.00 to wallet 1A2B3C
Discount strategy: Applied 10% discount. Final amount: $135.00
\`\`\``,
            hint1: `Yagona metod bilan PaymentStrategy interfeysini yarating. Aniq strategiyalar o'zlarining maxsus to'lov mantiqi bilan bu interfeysni amalga oshiradi.`,
            hint2: `Lambda-asoslangan strategiyalar uchun, PaymentStrategy funksional interfeys (bitta abstrakt metod) bo'lganligi sababli, lambdalardan foydalanishingiz mumkin: PaymentStrategy strategy = (amount) -> { /* amalga oshirish */ };`,
            whyItMatters: `Strategy namunasi algoritmlarni o'zaro almashtiriladigan qilish va shartli mantiqdan qochish uchun zarurdir. U Open/Closed printsipini ilgari suradi - mavjud kodni o'zgartirmasdan yangi strategiyalarni qo'shishingiz mumkin.

**Ishlab chiqarish patterni:**
\`\`\`java
// An'anaviy yondashuv
PaymentStrategy creditCard = new CreditCardPayment("1234-5678");
cart.setPaymentStrategy(creditCard);
cart.checkout();

// Lambda bilan zamonaviy yondashuv
PaymentStrategy bitcoin = (amount) -> {
    System.out.println("Transferring $" + amount + " to wallet");
};
cart.setPaymentStrategy(bitcoin);

// Strategiyalarni dinamik almashtirish
cart.setPaymentStrategy(new PayPalPayment("user@example.com"));
\`\`\`

**Amaliy foydalari:**
- Katta if-else yoki switch konstruksiyalaridan qochadi
- Yangi strategiyalar mavjud kodni o'zgartirmasdan qo'shiladi
- Lambda ifodalari strategiyalarni qisqa qiladi
- Collections.sort(), Comparator, Stream API da ishlatiladi`
        }
    }
};

export default task;
