import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-facade',
	title: 'Facade Pattern',
	difficulty: 'easy',
	tags: ['java', 'design-patterns', 'structural', 'facade'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Facade Pattern** in Java — provide a unified, simplified interface to a complex subsystem.

## Overview

The Facade pattern hides the complexity of a subsystem behind a simple interface. Clients interact with the facade instead of dealing with multiple classes directly.

## Key Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **Facade** | Simple interface | \`ComputerFacade\` class |
| **Subsystem classes** | Complex functionality | \`CPU\`, \`Memory\`, \`HardDrive\` |

## Your Task

Implement a computer startup system:

1. **CPU** - Freeze, jump, execute operations
2. **Memory** - Load data at position
3. **HardDrive** - Read sectors
4. **ComputerFacade** - Simple start() that orchestrates all subsystems

## Example Usage

\`\`\`java
ComputerFacade computer = new ComputerFacade();	// create facade
List<String> bootLog = computer.start();	// one simple call

// Behind the scenes:
// 1. cpu.freeze()	// freeze processor
// 2. hardDrive.read()	// read boot sector
// 3. memory.load()	// load into memory
// 4. cpu.jump()	// jump to address
// 5. cpu.execute()	// execute instructions
\`\`\`

## Key Insight

The facade doesn't add new functionality — it just simplifies access to existing subsystem functionality!`,
	initialCode: `class CPU {
    public String freeze() { return "CPU: Freezing"; }
    public String jump(long address) { return "CPU: Jumping to " + address; }
    public String execute() { return "CPU: Executing"; }
}

class Memory {
    public String load(long position, byte[] data) {
    }
}

class HardDrive {
    public String read(long lba, int size) {
    }
}

class ComputerFacade {
    private CPU cpu;
    private Memory memory;
    private HardDrive hardDrive;

    public ComputerFacade() {
    }

    public java.util.List<String> start() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `class CPU {	// Subsystem class - processor operations
    public String freeze() { return "CPU: Freezing"; }	// freeze processor
    public String jump(long address) { return "CPU: Jumping to " + address; }	// jump to memory address
    public String execute() { return "CPU: Executing"; }	// execute instructions
}

class Memory {	// Subsystem class - memory operations
    public String load(long position, byte[] data) {	// load data into memory
        return "Memory: Loading at " + position;	// return status
    }
}

class HardDrive {	// Subsystem class - disk operations
    public String read(long lba, int size) {	// read sectors from disk
        return "HardDrive: Reading sector " + lba;	// return status
    }
}

class ComputerFacade {	// Facade - simple interface to complex subsystem
    private CPU cpu;	// processor subsystem
    private Memory memory;	// memory subsystem
    private HardDrive hardDrive;	// storage subsystem

    public ComputerFacade() {	// constructor - create all subsystems
        this.cpu = new CPU();	// initialize CPU
        this.memory = new Memory();	// initialize memory
        this.hardDrive = new HardDrive();	// initialize hard drive
    }

    public java.util.List<String> start() {	// simplified interface - one method to boot
        java.util.List<String> results = new java.util.ArrayList<>();	// collect results
        results.add(cpu.freeze());	// step 1: freeze processor
        results.add(hardDrive.read(0, 1024));	// step 2: read boot sector
        results.add(memory.load(0, null));	// step 3: load into memory
        results.add(cpu.jump(0));	// step 4: jump to boot address
        results.add(cpu.execute());	// step 5: start execution
        return results;	// return boot log
    }
}`,
	hint1: `**Start Method Structure**

The start() method orchestrates the boot sequence:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();

    // Add each subsystem call result in order
    results.add(cpu.freeze());        // Step 1
    results.add(hardDrive.read(...)); // Step 2
    // ... continue with other steps

    return results;
}
\`\`\`

The facade hides the complexity of coordinating multiple subsystems.`,
	hint2: `**Complete Boot Sequence**

The complete sequence to implement:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();
    results.add(cpu.freeze());         // 1. Freeze CPU
    results.add(hardDrive.read(0, 1024)); // 2. Read boot sector
    results.add(memory.load(0, null)); // 3. Load to memory
    results.add(cpu.jump(0));          // 4. Jump to address 0
    results.add(cpu.execute());        // 5. Execute
    return results;
}
\`\`\`

Client just calls start() — facade handles all the complexity!`,
	whyItMatters: `## Problem & Solution

**Without Facade:**
\`\`\`java
// Client must know about ALL subsystems
CPU cpu = new CPU();	// create CPU
Memory memory = new Memory();	// create memory
HardDrive hd = new HardDrive();	// create hard drive

cpu.freeze();	// step 1
byte[] data = hd.read(0, 1024);	// step 2
memory.load(0, data);	// step 3
cpu.jump(0);	// step 4
cpu.execute();	// step 5
// Client must know the exact sequence!	// tightly coupled
\`\`\`

**With Facade:**
\`\`\`java
ComputerFacade computer = new ComputerFacade();	// create facade
computer.start();	// one call - done!
// Client doesn't know about CPU, Memory, HardDrive	// loosely coupled
\`\`\`

---

## Real-World Examples

| Domain | Facade | Subsystems Hidden |
|--------|--------|-------------------|
| **SLF4J** | Logger | Logback, Log4j, JUL |
| **JDBC** | DriverManager | Connection, Statement, ResultSet |
| **Spring** | JdbcTemplate | Connection pools, exception handling |
| **Hibernate** | Session | SQL, caching, transactions |
| **JavaMail** | Session | SMTP, POP3, authentication |
| **AWS SDK** | S3Client | HTTP, auth, retry, serialization |

---

## Production Pattern: E-Commerce Order Facade

\`\`\`java
// Subsystem: Inventory Management
class InventoryService {	// inventory subsystem
    public boolean checkStock(String productId, int quantity) {	// check availability
        System.out.println("Checking stock for: " + productId);	// log check
        return true;	// product available
    }

    public void reserveStock(String productId, int quantity) {	// reserve items
        System.out.println("Reserved " + quantity + " of " + productId);	// log reservation
    }
}

// Subsystem: Payment Processing
class PaymentService {	// payment subsystem
    public boolean validateCard(String cardNumber) {	// validate card
        System.out.println("Validating card: ****" + cardNumber.substring(12));	// log validation
        return true;	// card valid
    }

    public String processPayment(double amount) {	// charge card
        System.out.println("Processing payment: $" + amount);	// log payment
        return "TXN" + System.currentTimeMillis();	// return transaction ID
    }
}

// Subsystem: Shipping
class ShippingService {	// shipping subsystem
    public double calculateShipping(String address) {	// calculate cost
        System.out.println("Calculating shipping to: " + address);	// log calculation
        return 9.99;	// shipping cost
    }

    public String createShipment(String orderId, String address) {	// create shipment
        System.out.println("Creating shipment for order: " + orderId);	// log creation
        return "SHIP" + System.currentTimeMillis();	// return tracking number
    }
}

// Subsystem: Notification
class NotificationService {	// notification subsystem
    public void sendOrderConfirmation(String email, String orderId) {	// send email
        System.out.println("Sending confirmation to: " + email);	// log notification
    }
}

// FACADE - Simple interface for placing orders
class OrderFacade {	// facade hides all subsystem complexity
    private InventoryService inventory;	// inventory subsystem
    private PaymentService payment;	// payment subsystem
    private ShippingService shipping;	// shipping subsystem
    private NotificationService notification;	// notification subsystem

    public OrderFacade() {	// constructor creates all subsystems
        this.inventory = new InventoryService();	// init inventory
        this.payment = new PaymentService();	// init payment
        this.shipping = new ShippingService();	// init shipping
        this.notification = new NotificationService();	// init notification
    }

    public String placeOrder(	// ONE method for entire order process
            String productId,
            int quantity,
            String cardNumber,
            String address,
            String email) {

        // Step 1: Check inventory
        if (!inventory.checkStock(productId, quantity)) {	// verify availability
            throw new RuntimeException("Product out of stock");	// fail fast
        }

        // Step 2: Validate payment
        if (!payment.validateCard(cardNumber)) {	// verify card
            throw new RuntimeException("Invalid payment method");	// fail fast
        }

        // Step 3: Calculate total
        double itemPrice = 99.99;	// get from product service
        double shippingCost = shipping.calculateShipping(address);	// calculate shipping
        double total = (itemPrice * quantity) + shippingCost;	// total amount

        // Step 4: Process payment
        String transactionId = payment.processPayment(total);	// charge card

        // Step 5: Reserve inventory
        inventory.reserveStock(productId, quantity);	// reserve items

        // Step 6: Create shipment
        String orderId = "ORD" + System.currentTimeMillis();	// generate order ID
        String trackingNumber = shipping.createShipment(orderId, address);	// create shipment

        // Step 7: Send confirmation
        notification.sendOrderConfirmation(email, orderId);	// notify customer

        return orderId;	// return order ID
    }
}

// Usage - incredibly simple!
OrderFacade orderFacade = new OrderFacade();	// create facade

String orderId = orderFacade.placeOrder(	// one call for entire order
    "LAPTOP-123",	// product
    1,	// quantity
    "1234567890123456",	// card
    "123 Main St, City",	// address
    "customer@email.com"	// email
);
\`\`\`

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **God facade** | Facade doing too much | Split into multiple focused facades |
| **Exposing subsystems** | Returning subsystem objects | Return only facade types or primitives |
| **Tight coupling** | Facade knows too many details | Use interfaces for subsystems |
| **No direct access** | Preventing subsystem access entirely | Allow optional direct access for power users |
| **Business logic in facade** | Facade contains domain logic | Keep facade as coordinator only |`,
	order: 4,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

class Test1 {
    @Test
    void cpuFreeze() {
        CPU cpu = new CPU();
        String result = cpu.freeze();
        assertEquals("CPU: Freezing", result, "CPU freeze should return correct message");
    }
}

class Test2 {
    @Test
    void cpuJump() {
        CPU cpu = new CPU();
        String result = cpu.jump(100);
        assertTrue(result.contains("Jumping to 100"), "CPU jump should include address");
    }
}

class Test3 {
    @Test
    void cpuExecute() {
        CPU cpu = new CPU();
        String result = cpu.execute();
        assertEquals("CPU: Executing", result, "CPU execute should return correct message");
    }
}

class Test4 {
    @Test
    void memoryLoad() {
        Memory memory = new Memory();
        String result = memory.load(0, null);
        assertTrue(result.contains("Loading at 0"), "Memory load should include position");
    }
}

class Test5 {
    @Test
    void hardDriveRead() {
        HardDrive hd = new HardDrive();
        String result = hd.read(0, 1024);
        assertTrue(result.contains("Reading sector 0"), "HardDrive read should include sector");
    }
}

class Test6 {
    @Test
    void facadeStartReturnsAllSteps() {
        ComputerFacade computer = new ComputerFacade();
        List<String> results = computer.start();
        assertEquals(5, results.size(), "Start should return 5 steps");
    }
}

class Test7 {
    @Test
    void facadeStartFirstStepIsCpuFreeze() {
        ComputerFacade computer = new ComputerFacade();
        List<String> results = computer.start();
        assertTrue(results.get(0).contains("Freezing"), "First step should be CPU freeze");
    }
}

class Test8 {
    @Test
    void facadeStartSecondStepIsHardDriveRead() {
        ComputerFacade computer = new ComputerFacade();
        List<String> results = computer.start();
        assertTrue(results.get(1).contains("Reading"), "Second step should be HardDrive read");
    }
}

class Test9 {
    @Test
    void facadeStartIncludesMemoryLoad() {
        ComputerFacade computer = new ComputerFacade();
        List<String> results = computer.start();
        assertTrue(results.get(2).contains("Loading"), "Third step should be Memory load");
    }
}

class Test10 {
    @Test
    void facadeStartLastStepIsExecute() {
        ComputerFacade computer = new ComputerFacade();
        List<String> results = computer.start();
        assertTrue(results.get(4).contains("Executing"), "Last step should be CPU execute");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Facade (Фасад)',
			description: `Реализуйте **паттерн Facade** на Java — предоставьте унифицированный, упрощённый интерфейс к сложной подсистеме.

## Обзор

Паттерн Facade скрывает сложность подсистемы за простым интерфейсом. Клиенты взаимодействуют с фасадом вместо прямой работы с множеством классов.

## Ключевые компоненты

| Компонент | Роль | Реализация |
|-----------|------|------------|
| **Facade** | Простой интерфейс | Класс \`ComputerFacade\` |
| **Subsystem classes** | Сложная функциональность | \`CPU\`, \`Memory\`, \`HardDrive\` |

## Ваша задача

Реализуйте систему запуска компьютера:

1. **CPU** - Операции freeze, jump, execute
2. **Memory** - Загрузка данных по позиции
3. **HardDrive** - Чтение секторов
4. **ComputerFacade** - Простой start(), оркестрирующий все подсистемы

## Пример использования

\`\`\`java
ComputerFacade computer = new ComputerFacade();	// создаём фасад
List<String> bootLog = computer.start();	// один простой вызов

// За кулисами:
// 1. cpu.freeze()	// замораживаем процессор
// 2. hardDrive.read()	// читаем загрузочный сектор
// 3. memory.load()	// загружаем в память
// 4. cpu.jump()	// переходим по адресу
// 5. cpu.execute()	// выполняем инструкции
\`\`\`

## Ключевая идея

Фасад не добавляет новую функциональность — он просто упрощает доступ к существующей функциональности подсистемы!`,
			hint1: `**Структура метода Start**

Метод start() оркестрирует последовательность загрузки:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();

    // Добавляем результат каждого вызова подсистемы по порядку
    results.add(cpu.freeze());        // Шаг 1
    results.add(hardDrive.read(...)); // Шаг 2
    // ... продолжаем с другими шагами

    return results;
}
\`\`\`

Фасад скрывает сложность координации нескольких подсистем.`,
			hint2: `**Полная последовательность загрузки**

Полная последовательность для реализации:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();
    results.add(cpu.freeze());         // 1. Заморозить CPU
    results.add(hardDrive.read(0, 1024)); // 2. Прочитать загрузочный сектор
    results.add(memory.load(0, null)); // 3. Загрузить в память
    results.add(cpu.jump(0));          // 4. Перейти по адресу 0
    results.add(cpu.execute());        // 5. Выполнить
    return results;
}
\`\`\`

Клиент просто вызывает start() — фасад обрабатывает всю сложность!`,
			whyItMatters: `## Проблема и решение

**Без Facade:**
\`\`\`java
// Клиент должен знать о ВСЕХ подсистемах
CPU cpu = new CPU();	// создаём CPU
Memory memory = new Memory();	// создаём память
HardDrive hd = new HardDrive();	// создаём жёсткий диск

cpu.freeze();	// шаг 1
byte[] data = hd.read(0, 1024);	// шаг 2
memory.load(0, data);	// шаг 3
cpu.jump(0);	// шаг 4
cpu.execute();	// шаг 5
// Клиент должен знать точную последовательность!	// сильная связанность
\`\`\`

**С Facade:**
\`\`\`java
ComputerFacade computer = new ComputerFacade();	// создаём фасад
computer.start();	// один вызов - готово!
// Клиент не знает о CPU, Memory, HardDrive	// слабая связанность
\`\`\`

---

## Примеры из реального мира

| Домен | Facade | Скрытые подсистемы |
|-------|--------|-------------------|
| **SLF4J** | Logger | Logback, Log4j, JUL |
| **JDBC** | DriverManager | Connection, Statement, ResultSet |
| **Spring** | JdbcTemplate | Пулы соединений, обработка исключений |
| **Hibernate** | Session | SQL, кэширование, транзакции |
| **JavaMail** | Session | SMTP, POP3, аутентификация |
| **AWS SDK** | S3Client | HTTP, auth, повторы, сериализация |

---

## Production паттерн: Фасад E-Commerce заказов

\`\`\`java
// Подсистема: Управление инвентарём
class InventoryService {	// подсистема инвентаря
    public boolean checkStock(String productId, int quantity) {	// проверка наличия
        System.out.println("Checking stock for: " + productId);	// логируем проверку
        return true;	// товар доступен
    }

    public void reserveStock(String productId, int quantity) {	// резервируем товары
        System.out.println("Reserved " + quantity + " of " + productId);	// логируем резерв
    }
}

// Подсистема: Обработка платежей
class PaymentService {	// подсистема платежей
    public boolean validateCard(String cardNumber) {	// валидация карты
        System.out.println("Validating card: ****" + cardNumber.substring(12));	// логируем валидацию
        return true;	// карта валидна
    }

    public String processPayment(double amount) {	// списание средств
        System.out.println("Processing payment: $" + amount);	// логируем платёж
        return "TXN" + System.currentTimeMillis();	// возвращаем ID транзакции
    }
}

// Подсистема: Доставка
class ShippingService {	// подсистема доставки
    public double calculateShipping(String address) {	// расчёт стоимости
        System.out.println("Calculating shipping to: " + address);	// логируем расчёт
        return 9.99;	// стоимость доставки
    }

    public String createShipment(String orderId, String address) {	// создание отправки
        System.out.println("Creating shipment for order: " + orderId);	// логируем создание
        return "SHIP" + System.currentTimeMillis();	// возвращаем трек-номер
    }
}

// Подсистема: Уведомления
class NotificationService {	// подсистема уведомлений
    public void sendOrderConfirmation(String email, String orderId) {	// отправка email
        System.out.println("Sending confirmation to: " + email);	// логируем уведомление
    }
}

// FACADE - Простой интерфейс для оформления заказов
class OrderFacade {	// фасад скрывает всю сложность подсистем
    private InventoryService inventory;	// подсистема инвентаря
    private PaymentService payment;	// подсистема платежей
    private ShippingService shipping;	// подсистема доставки
    private NotificationService notification;	// подсистема уведомлений

    public OrderFacade() {	// конструктор создаёт все подсистемы
        this.inventory = new InventoryService();	// инициализация инвентаря
        this.payment = new PaymentService();	// инициализация платежей
        this.shipping = new ShippingService();	// инициализация доставки
        this.notification = new NotificationService();	// инициализация уведомлений
    }

    public String placeOrder(	// ОДИН метод для всего процесса заказа
            String productId,
            int quantity,
            String cardNumber,
            String address,
            String email) {

        // Шаг 1: Проверка инвентаря
        if (!inventory.checkStock(productId, quantity)) {	// проверяем наличие
            throw new RuntimeException("Product out of stock");	// быстрый отказ
        }

        // Шаг 2: Валидация платежа
        if (!payment.validateCard(cardNumber)) {	// проверяем карту
            throw new RuntimeException("Invalid payment method");	// быстрый отказ
        }

        // Шаг 3: Расчёт итога
        double itemPrice = 99.99;	// получаем от сервиса товаров
        double shippingCost = shipping.calculateShipping(address);	// рассчитываем доставку
        double total = (itemPrice * quantity) + shippingCost;	// итоговая сумма

        // Шаг 4: Обработка платежа
        String transactionId = payment.processPayment(total);	// списываем средства

        // Шаг 5: Резервирование инвентаря
        inventory.reserveStock(productId, quantity);	// резервируем товары

        // Шаг 6: Создание отправки
        String orderId = "ORD" + System.currentTimeMillis();	// генерируем ID заказа
        String trackingNumber = shipping.createShipment(orderId, address);	// создаём отправку

        // Шаг 7: Отправка подтверждения
        notification.sendOrderConfirmation(email, orderId);	// уведомляем клиента

        return orderId;	// возвращаем ID заказа
    }
}

// Использование - невероятно просто!
OrderFacade orderFacade = new OrderFacade();	// создаём фасад

String orderId = orderFacade.placeOrder(	// один вызов для всего заказа
    "LAPTOP-123",	// товар
    1,	// количество
    "1234567890123456",	// карта
    "123 Main St, City",	// адрес
    "customer@email.com"	// email
);
\`\`\`

---

## Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Бог-фасад** | Фасад делает слишком много | Разделите на несколько сфокусированных фасадов |
| **Экспозиция подсистем** | Возврат объектов подсистем | Возвращайте только типы фасада или примитивы |
| **Сильная связанность** | Фасад знает слишком много деталей | Используйте интерфейсы для подсистем |
| **Нет прямого доступа** | Полный запрет доступа к подсистемам | Разрешите опциональный прямой доступ для продвинутых пользователей |
| **Бизнес-логика в фасаде** | Фасад содержит доменную логику | Фасад должен быть только координатором |`
		},
		uz: {
			title: 'Facade (Fasad) Pattern',
			description: `Java da **Facade patternini** amalga oshiring — murakkab quyi tizimga birlashtirilgan, soddalashtirilgan interfeys taqdim eting.

## Umumiy ko'rinish

Facade patterni quyi tizimning murakkabligini oddiy interfeys ortida yashiradi. Mijozlar bir nechta klasslar bilan to'g'ridan-to'g'ri ishlash o'rniga fasad bilan o'zaro aloqa qiladi.

## Asosiy komponentlar

| Komponent | Rol | Amalga oshirish |
|-----------|-----|-----------------|
| **Facade** | Oddiy interfeys | \`ComputerFacade\` klassi |
| **Subsystem classes** | Murakkab funksionallik | \`CPU\`, \`Memory\`, \`HardDrive\` |

## Vazifangiz

Kompyuter ishga tushirish tizimini amalga oshiring:

1. **CPU** - Freeze, jump, execute operatsiyalari
2. **Memory** - Pozitsiyaga ma'lumot yuklash
3. **HardDrive** - Sektorlarni o'qish
4. **ComputerFacade** - Barcha quyi tizimlarni orkestrlash uchun oddiy start()

## Foydalanish namunasi

\`\`\`java
ComputerFacade computer = new ComputerFacade();	// fasad yaratamiz
List<String> bootLog = computer.start();	// bitta oddiy chaqiruv

// Sahna ortida:
// 1. cpu.freeze()	// protsessorni muzlatish
// 2. hardDrive.read()	// yuklash sektorini o'qish
// 3. memory.load()	// xotiraga yuklash
// 4. cpu.jump()	// manzilga o'tish
// 5. cpu.execute()	// instruksiyalarni bajarish
\`\`\`

## Asosiy tushuncha

Fasad yangi funksionallik qo'shmaydi — u faqat mavjud quyi tizim funksionalligiga kirishni soddalashtiradi!`,
			hint1: `**Start metodi strukturasi**

start() metodi yuklash ketma-ketligini orkestrlaydi:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();

    // Har bir quyi tizim chaqiruvining natijasini tartib bilan qo'shing
    results.add(cpu.freeze());        // Qadam 1
    results.add(hardDrive.read(...)); // Qadam 2
    // ... boshqa qadamlar bilan davom eting

    return results;
}
\`\`\`

Fasad bir nechta quyi tizimlarni muvofiqlashtirish murakkabligini yashiradi.`,
			hint2: `**To'liq yuklash ketma-ketligi**

Amalga oshirish uchun to'liq ketma-ketlik:

\`\`\`java
public java.util.List<String> start() {
    java.util.List<String> results = new java.util.ArrayList<>();
    results.add(cpu.freeze());         // 1. CPU ni muzlatish
    results.add(hardDrive.read(0, 1024)); // 2. Yuklash sektorini o'qish
    results.add(memory.load(0, null)); // 3. Xotiraga yuklash
    results.add(cpu.jump(0));          // 4. 0 manziliga o'tish
    results.add(cpu.execute());        // 5. Bajarish
    return results;
}
\`\`\`

Mijoz shunchaki start() ni chaqiradi — fasad barcha murakkablikni boshqaradi!`,
			whyItMatters: `## Muammo va yechim

**Facade siz:**
\`\`\`java
// Mijoz BARCHA quyi tizimlar haqida bilishi kerak
CPU cpu = new CPU();	// CPU yaratish
Memory memory = new Memory();	// xotira yaratish
HardDrive hd = new HardDrive();	// qattiq disk yaratish

cpu.freeze();	// qadam 1
byte[] data = hd.read(0, 1024);	// qadam 2
memory.load(0, data);	// qadam 3
cpu.jump(0);	// qadam 4
cpu.execute();	// qadam 5
// Mijoz aniq ketma-ketlikni bilishi kerak!	// qattiq bog'lanish
\`\`\`

**Facade bilan:**
\`\`\`java
ComputerFacade computer = new ComputerFacade();	// fasad yaratish
computer.start();	// bitta chaqiruv - tayyor!
// Mijoz CPU, Memory, HardDrive haqida bilmaydi	// yengil bog'lanish
\`\`\`

---

## Haqiqiy dunyo namunalari

| Domen | Facade | Yashirilgan quyi tizimlar |
|-------|--------|---------------------------|
| **SLF4J** | Logger | Logback, Log4j, JUL |
| **JDBC** | DriverManager | Connection, Statement, ResultSet |
| **Spring** | JdbcTemplate | Ulanish pullari, xatolarni boshqarish |
| **Hibernate** | Session | SQL, keshlash, tranzaksiyalar |
| **JavaMail** | Session | SMTP, POP3, autentifikatsiya |
| **AWS SDK** | S3Client | HTTP, auth, qayta urinish, serializatsiya |

---

## Production pattern: E-Commerce buyurtma fasadi

\`\`\`java
// Quyi tizim: Inventar boshqaruvi
class InventoryService {	// inventar quyi tizimi
    public boolean checkStock(String productId, int quantity) {	// mavjudligini tekshirish
        System.out.println("Checking stock for: " + productId);	// tekshirishni log qilish
        return true;	// mahsulot mavjud
    }

    public void reserveStock(String productId, int quantity) {	// tovarlarni zaxiralash
        System.out.println("Reserved " + quantity + " of " + productId);	// zaxirani log qilish
    }
}

// Quyi tizim: To'lovlarni qayta ishlash
class PaymentService {	// to'lov quyi tizimi
    public boolean validateCard(String cardNumber) {	// kartani tekshirish
        System.out.println("Validating card: ****" + cardNumber.substring(12));	// tekshirishni log qilish
        return true;	// karta yaroqli
    }

    public String processPayment(double amount) {	// kartadan yechish
        System.out.println("Processing payment: $" + amount);	// to'lovni log qilish
        return "TXN" + System.currentTimeMillis();	// tranzaksiya ID qaytarish
    }
}

// Quyi tizim: Yetkazib berish
class ShippingService {	// yetkazib berish quyi tizimi
    public double calculateShipping(String address) {	// narxni hisoblash
        System.out.println("Calculating shipping to: " + address);	// hisoblashni log qilish
        return 9.99;	// yetkazib berish narxi
    }

    public String createShipment(String orderId, String address) {	// jo'natmani yaratish
        System.out.println("Creating shipment for order: " + orderId);	// yaratishni log qilish
        return "SHIP" + System.currentTimeMillis();	// trek raqamini qaytarish
    }
}

// Quyi tizim: Bildirishnomalar
class NotificationService {	// bildirishnoma quyi tizimi
    public void sendOrderConfirmation(String email, String orderId) {	// email yuborish
        System.out.println("Sending confirmation to: " + email);	// bildirishnomani log qilish
    }
}

// FACADE - Buyurtma berish uchun oddiy interfeys
class OrderFacade {	// fasad barcha quyi tizim murakkabligini yashiradi
    private InventoryService inventory;	// inventar quyi tizimi
    private PaymentService payment;	// to'lov quyi tizimi
    private ShippingService shipping;	// yetkazib berish quyi tizimi
    private NotificationService notification;	// bildirishnoma quyi tizimi

    public OrderFacade() {	// konstruktor barcha quyi tizimlarni yaratadi
        this.inventory = new InventoryService();	// inventarni ishga tushirish
        this.payment = new PaymentService();	// to'lovni ishga tushirish
        this.shipping = new ShippingService();	// yetkazib berishni ishga tushirish
        this.notification = new NotificationService();	// bildirishnomani ishga tushirish
    }

    public String placeOrder(	// Butun buyurtma jarayoni uchun BITTA metod
            String productId,
            int quantity,
            String cardNumber,
            String address,
            String email) {

        // Qadam 1: Inventarni tekshirish
        if (!inventory.checkStock(productId, quantity)) {	// mavjudligini tekshirish
            throw new RuntimeException("Product out of stock");	// tez rad etish
        }

        // Qadam 2: To'lovni tekshirish
        if (!payment.validateCard(cardNumber)) {	// kartani tekshirish
            throw new RuntimeException("Invalid payment method");	// tez rad etish
        }

        // Qadam 3: Jami hisobni hisoblash
        double itemPrice = 99.99;	// mahsulot xizmatidan olish
        double shippingCost = shipping.calculateShipping(address);	// yetkazib berish narxini hisoblash
        double total = (itemPrice * quantity) + shippingCost;	// jami summa

        // Qadam 4: To'lovni qayta ishlash
        String transactionId = payment.processPayment(total);	// kartadan yechish

        // Qadam 5: Inventarni zaxiralash
        inventory.reserveStock(productId, quantity);	// tovarlarni zaxiralash

        // Qadam 6: Jo'natmani yaratish
        String orderId = "ORD" + System.currentTimeMillis();	// buyurtma ID yaratish
        String trackingNumber = shipping.createShipment(orderId, address);	// jo'natmani yaratish

        // Qadam 7: Tasdiqlash yuborish
        notification.sendOrderConfirmation(email, orderId);	// mijozni xabardor qilish

        return orderId;	// buyurtma ID qaytarish
    }
}

// Foydalanish - nihoyatda oddiy!
OrderFacade orderFacade = new OrderFacade();	// fasad yaratish

String orderId = orderFacade.placeOrder(	// butun buyurtma uchun bitta chaqiruv
    "LAPTOP-123",	// mahsulot
    1,	// miqdor
    "1234567890123456",	// karta
    "123 Main St, City",	// manzil
    "customer@email.com"	// email
);
\`\`\`

---

## Keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Xudo-fasad** | Fasad juda ko'p ish qiladi | Bir nechta yo'naltirilgan fasadlarga bo'ling |
| **Quyi tizimlarni ochish** | Quyi tizim ob'ektlarini qaytarish | Faqat fasad tiplarini yoki primitivlarni qaytaring |
| **Qattiq bog'lanish** | Fasad juda ko'p detallarni biladi | Quyi tizimlar uchun interfeyslardan foydalaning |
| **To'g'ridan-to'g'ri kirish yo'q** | Quyi tizimlarga kirishni to'liq taqiqlash | Tajribali foydalanuvchilar uchun ixtiyoriy to'g'ridan-to'g'ri kirishga ruxsat bering |
| **Fasadda biznes logika** | Fasad domen logikasini o'z ichiga oladi | Fasadni faqat koordinator sifatida saqlang |`
		}
	}
};

export default task;
