import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-strategy',
	title: 'Strategy Pattern',
	difficulty: 'easy',
	tags: ['java', 'design-patterns', 'behavioral', 'strategy'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Strategy Pattern

The **Strategy** pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **Strategy** | Interface declaring the algorithm method |
| **ConcreteStrategy** | Implements the Strategy interface with specific algorithm |
| **Context** | Maintains reference to Strategy; delegates algorithm execution |

---

### Your Task

Implement a sorting system with interchangeable algorithms:

1. **SortStrategy interface** - \`sort(array)\` method returning result string
2. **BubbleSort** - returns "BubbleSort: sorted {length} elements"
3. **QuickSort** - returns "QuickSort: sorted {length} elements"
4. **MergeSort** - returns "MergeSort: sorted {length} elements"
5. **Sorter context** - \`setStrategy()\` and \`performSort()\` methods

---

### Example Usage

\`\`\`java
Sorter sorter = new Sorter();	// create context (no strategy yet)
int[] data = {5, 2, 8, 1, 9};	// sample data to sort

sorter.setStrategy(new BubbleSort());	// set bubble sort strategy
String result1 = sorter.performSort(data);	// delegates to BubbleSort.sort()
// result1 = "BubbleSort: sorted 5 elements"

sorter.setStrategy(new QuickSort());	// change strategy at runtime
String result2 = sorter.performSort(data);	// now uses QuickSort
// result2 = "QuickSort: sorted 5 elements"

sorter.setStrategy(new MergeSort());	// switch to merge sort
String result3 = sorter.performSort(data);	// now uses MergeSort
// result3 = "MergeSort: sorted 5 elements"
\`\`\`

---

### Key Insight

Strategy eliminates conditional statements by replacing algorithm selection code with polymorphism - instead of \`if/else\` or \`switch\` to pick an algorithm, you inject the desired strategy object.`,
	initialCode: `interface SortStrategy {
    String sort(int[] array);
}

class BubbleSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        throw new UnsupportedOperationException("TODO: return 'BubbleSort: sorted {length} elements'");
    }
}

class QuickSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        throw new UnsupportedOperationException("TODO: return 'QuickSort: sorted {length} elements'");
    }
}

class MergeSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        throw new UnsupportedOperationException("TODO: return 'MergeSort: sorted {length} elements'");
    }
}

class Sorter {
    private SortStrategy strategy;

    public void setStrategy(SortStrategy strategy) {
    }

    public String performSort(int[] array) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface SortStrategy {	// Strategy interface - defines algorithm contract
    String sort(int[] array);	// algorithm method signature
}

class BubbleSort implements SortStrategy {	// ConcreteStrategy - bubble sort algorithm
    @Override
    public String sort(int[] array) {	// implement sorting algorithm
        return "BubbleSort: sorted " + array.length + " elements";	// return descriptive result
    }
}

class QuickSort implements SortStrategy {	// ConcreteStrategy - quick sort algorithm
    @Override
    public String sort(int[] array) {	// implement sorting algorithm
        return "QuickSort: sorted " + array.length + " elements";	// return descriptive result
    }
}

class MergeSort implements SortStrategy {	// ConcreteStrategy - merge sort algorithm
    @Override
    public String sort(int[] array) {	// implement sorting algorithm
        return "MergeSort: sorted " + array.length + " elements";	// return descriptive result
    }
}

class Sorter {	// Context - uses strategy to perform sorting
    private SortStrategy strategy;	// reference to current strategy

    public void setStrategy(SortStrategy strategy) {	// inject strategy at runtime
        this.strategy = strategy;	// store strategy reference
    }

    public String performSort(int[] array) {	// delegate to strategy
        if (strategy == null) {	// check if strategy is set
            return "No strategy set";	// return error message if no strategy
        }
        return strategy.sort(array);	// delegate sorting to strategy
    }
}`,
	hint1: `### Understanding Strategy Structure

The Strategy pattern has three main parts:

\`\`\`java
// 1. Strategy interface - defines the algorithm contract
interface SortStrategy {
    String sort(int[] array);	// Method all strategies must implement
}

// 2. ConcreteStrategy - implements the algorithm
class BubbleSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        // Return: "BubbleSort: sorted {length} elements"
        return "BubbleSort: sorted " + array.length + " elements";
    }
}

// 3. Context - uses strategy
class Sorter {
    private SortStrategy strategy;	// Holds reference to strategy

    public void setStrategy(SortStrategy strategy) {
        this.strategy = strategy;	// Set at runtime
    }
}
\`\`\``,
	hint2: `### Implementing performSort with Null Check

The context's performSort method should:
1. Check if a strategy is set
2. Delegate to the strategy if available

\`\`\`java
public String performSort(int[] array) {
    // 1. Check if strategy exists
    if (strategy == null) {
        return "No strategy set";	// Handle missing strategy
    }

    // 2. Delegate to strategy
    return strategy.sort(array);	// Let strategy do the work
}
\`\`\`

Each concrete strategy returns its name and the array length:
- BubbleSort: "BubbleSort: sorted {length} elements"
- QuickSort: "QuickSort: sorted {length} elements"
- MergeSort: "MergeSort: sorted {length} elements"`,
	whyItMatters: `## Why Strategy Pattern Matters

### The Problem and Solution

**Without Strategy:**
\`\`\`java
// Tight coupling with conditionals
class Sorter {
    public String sort(int[] array, String algorithm) {
        if (algorithm.equals("bubble")) {	// conditional logic
            // bubble sort implementation
            return "BubbleSort: sorted " + array.length;
        } else if (algorithm.equals("quick")) {	// more conditions
            // quick sort implementation
            return "QuickSort: sorted " + array.length;
        } else if (algorithm.equals("merge")) {	// even more conditions
            // merge sort implementation
            return "MergeSort: sorted " + array.length;
        }
        // Adding new algorithm requires modifying this class!
        return "Unknown algorithm";
    }
}
\`\`\`

**With Strategy:**
\`\`\`java
// Clean, extensible design
class Sorter {
    private SortStrategy strategy;	// holds any strategy

    public String performSort(int[] array) {
        return strategy.sort(array);	// delegate to strategy
    }
}
// Adding new algorithm = add new class, no changes to Sorter!
\`\`\`

---

### Real-World Applications

| Application | Strategy Interface | Strategies |
|-------------|-------------------|------------|
| **Collections.sort()** | Comparator | Custom comparators |
| **Compression** | CompressionStrategy | ZIP, GZIP, LZ4, Snappy |
| **Payment** | PaymentStrategy | CreditCard, PayPal, Crypto |
| **Validation** | Validator | EmailValidator, PhoneValidator |
| **Routing** | RouteStrategy | ShortestPath, FastestRoute |

---

### Production Pattern: Payment Processing

\`\`\`java
// Strategy interface for payment processing
interface PaymentStrategy {	// payment method contract
    PaymentResult processPayment(Money amount);	// process payment
    boolean supportsRefund();	// check refund capability
    PaymentResult refund(String transactionId);	// process refund
}

class CreditCardPayment implements PaymentStrategy {	// credit card strategy
    private final String cardNumber;	// card details
    private final PaymentGateway gateway;	// external gateway

    public CreditCardPayment(String cardNumber, PaymentGateway gateway) {	// constructor
        this.cardNumber = cardNumber;	// store card
        this.gateway = gateway;	// store gateway
    }

    @Override
    public PaymentResult processPayment(Money amount) {	// process credit card payment
        return gateway.charge(cardNumber, amount);	// delegate to gateway
    }

    @Override
    public boolean supportsRefund() { return true; }	// credit cards support refunds

    @Override
    public PaymentResult refund(String txId) {	// process refund
        return gateway.refund(txId);	// delegate to gateway
    }
}

class CryptoPayment implements PaymentStrategy {	// cryptocurrency strategy
    private final String walletAddress;	// wallet details
    private final BlockchainService blockchain;	// blockchain service

    @Override
    public PaymentResult processPayment(Money amount) {	// process crypto payment
        return blockchain.transfer(walletAddress, amount);	// blockchain transfer
    }

    @Override
    public boolean supportsRefund() { return false; }	// crypto doesn't support refunds

    @Override
    public PaymentResult refund(String txId) {	// refund not supported
        throw new UnsupportedOperationException("Crypto refunds not supported");
    }
}

// Context with strategy selection
class PaymentProcessor {	// payment context
    private PaymentStrategy strategy;	// current payment strategy

    public void setPaymentMethod(PaymentStrategy strategy) {	// set payment method
        this.strategy = strategy;	// store strategy
    }

    public PaymentResult checkout(Order order) {	// process checkout
        Money total = order.calculateTotal();	// get order total
        return strategy.processPayment(total);	// delegate to strategy
    }

    public PaymentResult refundOrder(String transactionId) {	// process refund
        if (!strategy.supportsRefund()) {	// check refund support
            throw new IllegalStateException("Current payment method doesn't support refunds");
        }
        return strategy.refund(transactionId);	// delegate refund
    }
}

// Usage:
PaymentProcessor processor = new PaymentProcessor();	// create processor
processor.setPaymentMethod(new CreditCardPayment(card, gateway));	// set credit card
processor.checkout(order);	// process with credit card

processor.setPaymentMethod(new CryptoPayment(wallet, blockchain));	// switch to crypto
processor.checkout(order);	// process with crypto
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Null strategy** | NullPointerException | Check for null or use NullObject pattern |
| **Strategy state** | Strategies sharing mutable state | Keep strategies stateless or use fresh instances |
| **Too many strategies** | Over-engineering simple cases | Use Strategy only when algorithms truly vary |
| **Leaking context** | Strategy knowing too much about context | Pass only needed data to strategy method |
| **No default** | Requiring strategy before use | Provide sensible default strategy |`,
	order: 1,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: BubbleSort returns correct format
class Test1 {
    @Test
    public void test() {
        SortStrategy bubble = new BubbleSort();
        String result = bubble.sort(new int[]{5, 2, 8});
        assertEquals("BubbleSort: sorted 3 elements", result);
    }
}

// Test2: QuickSort returns correct format
class Test2 {
    @Test
    public void test() {
        SortStrategy quick = new QuickSort();
        String result = quick.sort(new int[]{1, 2, 3, 4, 5});
        assertEquals("QuickSort: sorted 5 elements", result);
    }
}

// Test3: MergeSort returns correct format
class Test3 {
    @Test
    public void test() {
        SortStrategy merge = new MergeSort();
        String result = merge.sort(new int[]{9, 8, 7, 6});
        assertEquals("MergeSort: sorted 4 elements", result);
    }
}

// Test4: Sorter with no strategy
class Test4 {
    @Test
    public void test() {
        Sorter sorter = new Sorter();
        String result = sorter.performSort(new int[]{1, 2, 3});
        assertEquals("No strategy set", result);
    }
}

// Test5: Sorter delegates to BubbleSort
class Test5 {
    @Test
    public void test() {
        Sorter sorter = new Sorter();
        sorter.setStrategy(new BubbleSort());
        String result = sorter.performSort(new int[]{1, 2});
        assertTrue(result.startsWith("BubbleSort"));
    }
}

// Test6: Strategy can be changed at runtime
class Test6 {
    @Test
    public void test() {
        Sorter sorter = new Sorter();
        sorter.setStrategy(new BubbleSort());
        sorter.performSort(new int[]{1});
        sorter.setStrategy(new QuickSort());
        String result = sorter.performSort(new int[]{1, 2});
        assertTrue(result.startsWith("QuickSort"));
    }
}

// Test7: Empty array works
class Test7 {
    @Test
    public void test() {
        SortStrategy bubble = new BubbleSort();
        String result = bubble.sort(new int[]{});
        assertEquals("BubbleSort: sorted 0 elements", result);
    }
}

// Test8: Single element array
class Test8 {
    @Test
    public void test() {
        SortStrategy merge = new MergeSort();
        String result = merge.sort(new int[]{42});
        assertEquals("MergeSort: sorted 1 elements", result);
    }
}

// Test9: Sorter with MergeSort
class Test9 {
    @Test
    public void test() {
        Sorter sorter = new Sorter();
        sorter.setStrategy(new MergeSort());
        String result = sorter.performSort(new int[]{3, 1, 4, 1, 5});
        assertEquals("MergeSort: sorted 5 elements", result);
    }
}

// Test10: Large array
class Test10 {
    @Test
    public void test() {
        SortStrategy quick = new QuickSort();
        int[] arr = new int[100];
        String result = quick.sort(arr);
        assertEquals("QuickSort: sorted 100 elements", result);
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Strategy (Стратегия)',
			description: `## Паттерн Strategy (Стратегия)

Паттерн **Strategy** определяет семейство алгоритмов, инкапсулирует каждый из них и делает их взаимозаменяемыми. Strategy позволяет алгоритму изменяться независимо от клиентов, которые его используют.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Strategy** | Интерфейс, объявляющий метод алгоритма |
| **ConcreteStrategy** | Реализует интерфейс Strategy с конкретным алгоритмом |
| **Context** | Хранит ссылку на Strategy; делегирует выполнение алгоритма |

---

### Ваша задача

Реализуйте систему сортировки с взаимозаменяемыми алгоритмами:

1. **Интерфейс SortStrategy** - метод \`sort(array)\`, возвращающий строку результата
2. **BubbleSort** - возвращает "BubbleSort: sorted {length} elements"
3. **QuickSort** - возвращает "QuickSort: sorted {length} elements"
4. **MergeSort** - возвращает "MergeSort: sorted {length} elements"
5. **Контекст Sorter** - методы \`setStrategy()\` и \`performSort()\`

---

### Пример использования

\`\`\`java
Sorter sorter = new Sorter();	// создаём контекст (пока без стратегии)
int[] data = {5, 2, 8, 1, 9};	// данные для сортировки

sorter.setStrategy(new BubbleSort());	// устанавливаем пузырьковую сортировку
String result1 = sorter.performSort(data);	// делегирует BubbleSort.sort()
// result1 = "BubbleSort: sorted 5 elements"

sorter.setStrategy(new QuickSort());	// меняем стратегию во время выполнения
String result2 = sorter.performSort(data);	// теперь использует QuickSort
// result2 = "QuickSort: sorted 5 elements"

sorter.setStrategy(new MergeSort());	// переключаемся на сортировку слиянием
String result3 = sorter.performSort(data);	// теперь использует MergeSort
// result3 = "MergeSort: sorted 5 elements"
\`\`\`

---

### Ключевая идея

Strategy устраняет условные операторы, заменяя код выбора алгоритма полиморфизмом — вместо \`if/else\` или \`switch\` для выбора алгоритма вы внедряете нужный объект стратегии.`,
			hint1: `### Понимание структуры Strategy

Паттерн Strategy состоит из трёх основных частей:

\`\`\`java
// 1. Интерфейс Strategy - определяет контракт алгоритма
interface SortStrategy {
    String sort(int[] array);	// Метод, который должны реализовать все стратегии
}

// 2. ConcreteStrategy - реализует алгоритм
class BubbleSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        // Возвращаем: "BubbleSort: sorted {length} elements"
        return "BubbleSort: sorted " + array.length + " elements";
    }
}

// 3. Context - использует стратегию
class Sorter {
    private SortStrategy strategy;	// Хранит ссылку на стратегию

    public void setStrategy(SortStrategy strategy) {
        this.strategy = strategy;	// Устанавливается во время выполнения
    }
}
\`\`\``,
			hint2: `### Реализация performSort с проверкой на null

Метод performSort контекста должен:
1. Проверить, установлена ли стратегия
2. Делегировать стратегии, если она доступна

\`\`\`java
public String performSort(int[] array) {
    // 1. Проверяем наличие стратегии
    if (strategy == null) {
        return "No strategy set";	// Обработка отсутствующей стратегии
    }

    // 2. Делегируем стратегии
    return strategy.sort(array);	// Пусть стратегия выполняет работу
}
\`\`\`

Каждая конкретная стратегия возвращает своё имя и длину массива:
- BubbleSort: "BubbleSort: sorted {length} elements"
- QuickSort: "QuickSort: sorted {length} elements"
- MergeSort: "MergeSort: sorted {length} elements"`,
			whyItMatters: `## Почему паттерн Strategy важен

### Проблема и решение

**Без Strategy:**
\`\`\`java
// Тесная связанность с условиями
class Sorter {
    public String sort(int[] array, String algorithm) {
        if (algorithm.equals("bubble")) {	// условная логика
            // реализация пузырьковой сортировки
            return "BubbleSort: sorted " + array.length;
        } else if (algorithm.equals("quick")) {	// ещё условия
            // реализация быстрой сортировки
            return "QuickSort: sorted " + array.length;
        } else if (algorithm.equals("merge")) {	// и ещё условия
            // реализация сортировки слиянием
            return "MergeSort: sorted " + array.length;
        }
        // Добавление нового алгоритма требует изменения этого класса!
        return "Unknown algorithm";
    }
}
\`\`\`

**С Strategy:**
\`\`\`java
// Чистый, расширяемый дизайн
class Sorter {
    private SortStrategy strategy;	// хранит любую стратегию

    public String performSort(int[] array) {
        return strategy.sort(array);	// делегируем стратегии
    }
}
// Добавление нового алгоритма = добавить новый класс, без изменений в Sorter!
\`\`\`

---

### Применение в реальном мире

| Применение | Интерфейс Strategy | Стратегии |
|------------|-------------------|-----------|
| **Collections.sort()** | Comparator | Пользовательские компараторы |
| **Сжатие** | CompressionStrategy | ZIP, GZIP, LZ4, Snappy |
| **Оплата** | PaymentStrategy | CreditCard, PayPal, Crypto |
| **Валидация** | Validator | EmailValidator, PhoneValidator |
| **Маршрутизация** | RouteStrategy | ShortestPath, FastestRoute |

---

### Продакшен паттерн: Обработка платежей

\`\`\`java
// Интерфейс стратегии для обработки платежей
interface PaymentStrategy {	// контракт способа оплаты
    PaymentResult processPayment(Money amount);	// обработать платёж
    boolean supportsRefund();	// проверить возможность возврата
    PaymentResult refund(String transactionId);	// обработать возврат
}

class CreditCardPayment implements PaymentStrategy {	// стратегия кредитной карты
    private final String cardNumber;	// данные карты
    private final PaymentGateway gateway;	// внешний шлюз

    public CreditCardPayment(String cardNumber, PaymentGateway gateway) {	// конструктор
        this.cardNumber = cardNumber;	// сохраняем карту
        this.gateway = gateway;	// сохраняем шлюз
    }

    @Override
    public PaymentResult processPayment(Money amount) {	// обработка платежа картой
        return gateway.charge(cardNumber, amount);	// делегируем шлюзу
    }

    @Override
    public boolean supportsRefund() { return true; }	// карты поддерживают возвраты

    @Override
    public PaymentResult refund(String txId) {	// обработка возврата
        return gateway.refund(txId);	// делегируем шлюзу
    }
}

class CryptoPayment implements PaymentStrategy {	// криптовалютная стратегия
    private final String walletAddress;	// данные кошелька
    private final BlockchainService blockchain;	// сервис блокчейна

    @Override
    public PaymentResult processPayment(Money amount) {	// обработка крипто-платежа
        return blockchain.transfer(walletAddress, amount);	// перевод через блокчейн
    }

    @Override
    public boolean supportsRefund() { return false; }	// крипто не поддерживает возвраты

    @Override
    public PaymentResult refund(String txId) {	// возврат не поддерживается
        throw new UnsupportedOperationException("Crypto refunds not supported");
    }
}

// Контекст с выбором стратегии
class PaymentProcessor {	// контекст платежей
    private PaymentStrategy strategy;	// текущая платёжная стратегия

    public void setPaymentMethod(PaymentStrategy strategy) {	// установить способ оплаты
        this.strategy = strategy;	// сохраняем стратегию
    }

    public PaymentResult checkout(Order order) {	// обработать оплату заказа
        Money total = order.calculateTotal();	// получить сумму заказа
        return strategy.processPayment(total);	// делегировать стратегии
    }

    public PaymentResult refundOrder(String transactionId) {	// обработать возврат
        if (!strategy.supportsRefund()) {	// проверить поддержку возврата
            throw new IllegalStateException("Current payment method doesn't support refunds");
        }
        return strategy.refund(transactionId);	// делегировать возврат
    }
}

// Использование:
PaymentProcessor processor = new PaymentProcessor();	// создаём процессор
processor.setPaymentMethod(new CreditCardPayment(card, gateway));	// устанавливаем карту
processor.checkout(order);	// оплата картой

processor.setPaymentMethod(new CryptoPayment(wallet, blockchain));	// переключаемся на крипто
processor.checkout(order);	// оплата криптой
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Null стратегия** | NullPointerException | Проверяйте на null или используйте NullObject |
| **Состояние стратегии** | Стратегии делят изменяемое состояние | Делайте стратегии без состояния или используйте новые экземпляры |
| **Слишком много стратегий** | Переусложнение простых случаев | Используйте Strategy только когда алгоритмы действительно меняются |
| **Утечка контекста** | Стратегия знает слишком много о контексте | Передавайте только нужные данные в метод стратегии |
| **Нет умолчания** | Требуется стратегия перед использованием | Предоставьте разумную стратегию по умолчанию |`
		},
		uz: {
			title: 'Strategy Pattern',
			description: `## Strategy Pattern

**Strategy** pattern algoritmlar oilasini aniqlaydi, har birini inkapsulyatsiya qiladi va ularni almashtirib turiladigan qiladi. Strategy algoritmni undan foydalanuvchi mijozlardan mustaqil ravishda o'zgartirishga imkon beradi.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **Strategy** | Algoritm metodini e'lon qiluvchi interfeys |
| **ConcreteStrategy** | Strategy interfeysini maxsus algoritm bilan amalga oshiradi |
| **Context** | Strategy ga havolani saqlaydi; algoritm bajarilishini delegatsiya qiladi |

---

### Vazifangiz

Almashtirib turiladigan algoritmlar bilan saralash tizimini amalga oshiring:

1. **SortStrategy interfeysi** - natija satrini qaytaruvchi \`sort(array)\` metodi
2. **BubbleSort** - "BubbleSort: sorted {length} elements" qaytaradi
3. **QuickSort** - "QuickSort: sorted {length} elements" qaytaradi
4. **MergeSort** - "MergeSort: sorted {length} elements" qaytaradi
5. **Sorter konteksti** - \`setStrategy()\` va \`performSort()\` metodlari

---

### Foydalanish Namunasi

\`\`\`java
Sorter sorter = new Sorter();	// kontekst yaratamiz (hali strategiyasiz)
int[] data = {5, 2, 8, 1, 9};	// saralash uchun namuna ma'lumotlar

sorter.setStrategy(new BubbleSort());	// pufakchali saralash strategiyasini o'rnatamiz
String result1 = sorter.performSort(data);	// BubbleSort.sort() ga delegatsiya qiladi
// result1 = "BubbleSort: sorted 5 elements"

sorter.setStrategy(new QuickSort());	// ishlash vaqtida strategiyani almashtiramiz
String result2 = sorter.performSort(data);	// endi QuickSort ishlatadi
// result2 = "QuickSort: sorted 5 elements"

sorter.setStrategy(new MergeSort());	// birlashtirish saralashiga o'tamiz
String result3 = sorter.performSort(data);	// endi MergeSort ishlatadi
// result3 = "MergeSort: sorted 5 elements"
\`\`\`

---

### Asosiy Fikr

Strategy algoritm tanlash kodini polimorfizm bilan almashtirib, shartli operatorlarni yo'q qiladi — algoritmni tanlash uchun \`if/else\` yoki \`switch\` o'rniga kerakli strategiya obyektini kiritasiz.`,
			hint1: `### Strategy Strukturasini Tushunish

Strategy pattern uch asosiy qismdan iborat:

\`\`\`java
// 1. Strategy interfeysi - algoritm shartnomasi aniqlaydi
interface SortStrategy {
    String sort(int[] array);	// Barcha strategiyalar amalga oshirishi kerak bo'lgan metod
}

// 2. ConcreteStrategy - algoritmni amalga oshiradi
class BubbleSort implements SortStrategy {
    @Override
    public String sort(int[] array) {
        // Qaytaradi: "BubbleSort: sorted {length} elements"
        return "BubbleSort: sorted " + array.length + " elements";
    }
}

// 3. Context - strategiyadan foydalanadi
class Sorter {
    private SortStrategy strategy;	// Strategiyaga havolani saqlaydi

    public void setStrategy(SortStrategy strategy) {
        this.strategy = strategy;	// Ishlash vaqtida o'rnatiladi
    }
}
\`\`\``,
			hint2: `### performSort ni Null Tekshirish bilan Amalga Oshirish

Kontekstning performSort metodi:
1. Strategiya o'rnatilganligini tekshirishi kerak
2. Mavjud bo'lsa strategiyaga delegatsiya qilishi kerak

\`\`\`java
public String performSort(int[] array) {
    // 1. Strategiya mavjudligini tekshiramiz
    if (strategy == null) {
        return "No strategy set";	// Yo'q strategiyani boshqaramiz
    }

    // 2. Strategiyaga delegatsiya qilamiz
    return strategy.sort(array);	// Ishni strategiyaga topshiramiz
}
\`\`\`

Har bir aniq strategiya o'z nomini va massiv uzunligini qaytaradi:
- BubbleSort: "BubbleSort: sorted {length} elements"
- QuickSort: "QuickSort: sorted {length} elements"
- MergeSort: "MergeSort: sorted {length} elements"`,
			whyItMatters: `## Nima Uchun Strategy Pattern Muhim

### Muammo va Yechim

**Strategy siz:**
\`\`\`java
// Shartlar bilan qattiq bog'lanish
class Sorter {
    public String sort(int[] array, String algorithm) {
        if (algorithm.equals("bubble")) {	// shartli mantiq
            // pufakchali saralash amalga oshirish
            return "BubbleSort: sorted " + array.length;
        } else if (algorithm.equals("quick")) {	// ko'proq shartlar
            // tez saralash amalga oshirish
            return "QuickSort: sorted " + array.length;
        } else if (algorithm.equals("merge")) {	// yanada ko'proq shartlar
            // birlashtirish saralash amalga oshirish
            return "MergeSort: sorted " + array.length;
        }
        // Yangi algoritm qo'shish ushbu klassni o'zgartirishni talab qiladi!
        return "Unknown algorithm";
    }
}
\`\`\`

**Strategy bilan:**
\`\`\`java
// Toza, kengaytiriladigan dizayn
class Sorter {
    private SortStrategy strategy;	// istalgan strategiyani saqlaydi

    public String performSort(int[] array) {
        return strategy.sort(array);	// strategiyaga delegatsiya
    }
}
// Yangi algoritm qo'shish = yangi klass qo'shish, Sorter da o'zgarishlar yo'q!
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Strategy Interfeysi | Strategiyalar |
|------------|-------------------|---------------|
| **Collections.sort()** | Comparator | Maxsus komparatorlar |
| **Siqish** | CompressionStrategy | ZIP, GZIP, LZ4, Snappy |
| **To'lov** | PaymentStrategy | CreditCard, PayPal, Crypto |
| **Validatsiya** | Validator | EmailValidator, PhoneValidator |
| **Marshrutlash** | RouteStrategy | ShortestPath, FastestRoute |

---

### Prodakshen Pattern: To'lovlarni Qayta Ishlash

\`\`\`java
// To'lovlarni qayta ishlash uchun strategiya interfeysi
interface PaymentStrategy {	// to'lov usuli shartnomasi
    PaymentResult processPayment(Money amount);	// to'lovni qayta ishlash
    boolean supportsRefund();	// qaytarish imkoniyatini tekshirish
    PaymentResult refund(String transactionId);	// qaytarishni qayta ishlash
}

class CreditCardPayment implements PaymentStrategy {	// kredit karta strategiyasi
    private final String cardNumber;	// karta ma'lumotlari
    private final PaymentGateway gateway;	// tashqi shlyuz

    public CreditCardPayment(String cardNumber, PaymentGateway gateway) {	// konstruktor
        this.cardNumber = cardNumber;	// kartani saqlash
        this.gateway = gateway;	// shlyuzni saqlash
    }

    @Override
    public PaymentResult processPayment(Money amount) {	// karta to'lovini qayta ishlash
        return gateway.charge(cardNumber, amount);	// shlyuzga delegatsiya
    }

    @Override
    public boolean supportsRefund() { return true; }	// kartalar qaytarishni qo'llaydi

    @Override
    public PaymentResult refund(String txId) {	// qaytarishni qayta ishlash
        return gateway.refund(txId);	// shlyuzga delegatsiya
    }
}

class CryptoPayment implements PaymentStrategy {	// kriptovalyuta strategiyasi
    private final String walletAddress;	// hamyon ma'lumotlari
    private final BlockchainService blockchain;	// blokcheyn xizmati

    @Override
    public PaymentResult processPayment(Money amount) {	// kripto to'lovini qayta ishlash
        return blockchain.transfer(walletAddress, amount);	// blokcheyn o'tkazma
    }

    @Override
    public boolean supportsRefund() { return false; }	// kripto qaytarishni qo'llamaydi

    @Override
    public PaymentResult refund(String txId) {	// qaytarish qo'llab-quvvatlanmaydi
        throw new UnsupportedOperationException("Crypto refunds not supported");
    }
}

// Strategiya tanlash bilan kontekst
class PaymentProcessor {	// to'lov konteksti
    private PaymentStrategy strategy;	// joriy to'lov strategiyasi

    public void setPaymentMethod(PaymentStrategy strategy) {	// to'lov usulini o'rnatish
        this.strategy = strategy;	// strategiyani saqlash
    }

    public PaymentResult checkout(Order order) {	// buyurtmani to'lash
        Money total = order.calculateTotal();	// buyurtma summasini olish
        return strategy.processPayment(total);	// strategiyaga delegatsiya
    }

    public PaymentResult refundOrder(String transactionId) {	// qaytarishni qayta ishlash
        if (!strategy.supportsRefund()) {	// qaytarish qo'llab-quvvatini tekshirish
            throw new IllegalStateException("Current payment method doesn't support refunds");
        }
        return strategy.refund(transactionId);	// qaytarishni delegatsiya
    }
}

// Foydalanish:
PaymentProcessor processor = new PaymentProcessor();	// protsessor yaratish
processor.setPaymentMethod(new CreditCardPayment(card, gateway));	// karta o'rnatish
processor.checkout(order);	// karta bilan to'lash

processor.setPaymentMethod(new CryptoPayment(wallet, blockchain));	// kripto ga o'tish
processor.checkout(order);	// kripto bilan to'lash
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Null strategiya** | NullPointerException | Null tekshiring yoki NullObject pattern ishlating |
| **Strategiya holati** | Strategiyalar o'zgaruvchan holatni baham ko'radi | Strategiyalarni holatsiz qiling yoki yangi nusxalar ishlating |
| **Juda ko'p strategiyalar** | Oddiy holatlarni murakkablashtirish | Strategy faqat algoritmlar haqiqatan o'zgarganda ishlating |
| **Kontekst oqishi** | Strategiya kontekst haqida juda ko'p biladi | Strategiya metodiga faqat kerakli ma'lumotlarni uzating |
| **Standart yo'q** | Ishlatishdan oldin strategiya talab qilinadi | Oqilona standart strategiya bering |`
		}
	}
};

export default task;
