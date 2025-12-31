import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-constructors',
    title: 'Constructors and Initialization',
    difficulty: 'easy',
    tags: ['java', 'oop', 'constructors', 'initialization', 'overloading'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a **BankAccount** class that demonstrates constructor overloading and proper object initialization.

**Requirements:**
1. Create a BankAccount class with fields:
   1.1. accountNumber (String)
   1.2. accountHolder (String)
   1.3. balance (double)
   1.4. accountType (String) - "Savings" or "Checking"

2. Implement multiple constructors:
   2.1. Default constructor: sets default values
   2.2. Constructor with accountNumber and accountHolder
   2.3. Constructor with all parameters
   2.4. Use constructor chaining with this()

3. Add an instance initialization block that prints "Initializing account..."

4. Implement methods:
   4.1. deposit(double amount)
   4.2. withdraw(double amount)
   4.3. displayAccountInfo()

5. In main method:
   5.1. Create accounts using different constructors
   5.2. Test deposit and withdrawal
   5.3. Display account information

**Learning Goals:**
- Understand constructor overloading
- Learn constructor chaining with this()
- Practice instance initialization blocks`,
    initialCode: `public class BankAccount {
    // TODO: Add fields

    // TODO: Add instance initialization block

    // TODO: Implement default constructor

    // TODO: Implement constructor with accountNumber and accountHolder

    // TODO: Implement constructor with all parameters

    // TODO: Implement deposit method

    // TODO: Implement withdraw method

    // TODO: Implement displayAccountInfo method

    public static void main(String[] args) {
        // TODO: Create accounts and test the class
    }
}`,
    solutionCode: `public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;
    private String accountType;

    // Instance initialization block - runs before constructor
    {
        System.out.println("Initializing account...");
    }

    // Default constructor
    public BankAccount() {
        this("ACC000000", "Unknown", 0.0, "Savings");
    }

    // Constructor with accountNumber and accountHolder
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0, "Savings");
    }

    // Constructor with all parameters - main constructor
    public BankAccount(String accountNumber, String accountHolder,
                      double balance, String accountType) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
        this.accountType = accountType;
    }

    // Method to deposit money
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Deposited: $" + amount);
        } else {
            System.out.println("Invalid deposit amount");
        }
    }

    // Method to withdraw money
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Withdrawn: $" + amount);
        } else {
            System.out.println("Invalid withdrawal amount or insufficient funds");
        }
    }

    // Display account information
    public void displayAccountInfo() {
        System.out.println("=== Account Information ===");
        System.out.println("Account Number: " + accountNumber);
        System.out.println("Account Holder: " + accountHolder);
        System.out.println("Account Type: " + accountType);
        System.out.println("Balance: $" + balance);
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Using default constructor
        BankAccount account1 = new BankAccount();
        account1.displayAccountInfo();

        // Using constructor with accountNumber and accountHolder
        BankAccount account2 = new BankAccount("ACC123456", "John Doe");
        account2.deposit(1000.0);
        account2.displayAccountInfo();

        // Using constructor with all parameters
        BankAccount account3 = new BankAccount("ACC789012", "Jane Smith",
                                               5000.0, "Checking");
        account3.withdraw(500.0);
        account3.deposit(200.0);
        account3.displayAccountInfo();
    }
}`,
    hint1: `Start with the most complete constructor (all parameters), then use this() to call it from other constructors with default values.`,
    hint2: `Instance initialization blocks are defined using curly braces {} without any keyword. They run before the constructor body executes.`,
    whyItMatters: `Constructor overloading provides flexibility in object creation, allowing users to initialize objects in different ways. Constructor chaining with this() promotes code reuse and reduces redundancy. Understanding initialization order (initialization blocks, then constructors) is essential for proper object setup.

**Production Pattern:**
\`\`\`java
public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;

    // Default constructor
    public BankAccount() {
        this("ACC000000", "Unknown", 0.0);
    }

    // Constructor with parameters
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0); // Constructor chaining
    }

    // Main constructor
    public BankAccount(String accountNumber, String accountHolder, double balance) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
    }
}
\`\`\`

**Practical Benefits:**
- Multiple constructors provide flexibility in object creation
- this() chaining reduces code duplication
- Initialization blocks allow common logic to run before constructors
- Overloading makes the class API more convenient for different use cases`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.*;

// Test 1: BankAccount class exists
class Test1 {
    @Test
    void testBankAccountClassExists() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        assertNotNull(cls);
    }
}

// Test 2: BankAccount has required fields
class Test2 {
    @Test
    void testBankAccountHasFields() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Field accountNumber = cls.getDeclaredField("accountNumber");
        Field accountHolder = cls.getDeclaredField("accountHolder");
        Field balance = cls.getDeclaredField("balance");
        assertNotNull(accountNumber);
        assertNotNull(accountHolder);
        assertNotNull(balance);
    }
}

// Test 3: Default constructor exists
class Test3 {
    @Test
    void testDefaultConstructor() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor();
        Object account = constructor.newInstance();
        assertNotNull(account);
    }
}

// Test 4: Constructor with two parameters exists
class Test4 {
    @Test
    void testTwoParamConstructor() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor(String.class, String.class);
        Object account = constructor.newInstance("ACC123", "John Doe");
        assertNotNull(account);
    }
}

// Test 5: Constructor with all parameters exists
class Test5 {
    @Test
    void testFullConstructor() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor(String.class, String.class, double.class, String.class);
        Object account = constructor.newInstance("ACC123", "John Doe", 1000.0, "Savings");
        assertNotNull(account);
    }
}

// Test 6: Deposit method exists and works
class Test6 {
    @Test
    void testDepositMethod() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor(String.class, String.class, double.class, String.class);
        Object account = constructor.newInstance("ACC123", "John", 100.0, "Savings");
        Method deposit = cls.getMethod("deposit", double.class);
        deposit.invoke(account, 50.0);
        Field balanceField = cls.getDeclaredField("balance");
        balanceField.setAccessible(true);
        assertEquals(150.0, (double) balanceField.get(account), 0.01);
    }
}

// Test 7: Withdraw method exists and works
class Test7 {
    @Test
    void testWithdrawMethod() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor(String.class, String.class, double.class, String.class);
        Object account = constructor.newInstance("ACC123", "John", 100.0, "Savings");
        Method withdraw = cls.getMethod("withdraw", double.class);
        withdraw.invoke(account, 30.0);
        Field balanceField = cls.getDeclaredField("balance");
        balanceField.setAccessible(true);
        assertEquals(70.0, (double) balanceField.get(account), 0.01);
    }
}

// Test 8: displayAccountInfo method exists
class Test8 {
    @Test
    void testDisplayAccountInfoExists() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Method method = cls.getMethod("displayAccountInfo");
        assertNotNull(method);
    }
}

// Test 9: Constructor chaining works correctly
class Test9 {
    @Test
    void testConstructorChaining() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> twoParam = cls.getConstructor(String.class, String.class);
        Object account = twoParam.newInstance("ACC456", "Jane");
        Field balance = cls.getDeclaredField("balance");
        balance.setAccessible(true);
        assertEquals(0.0, (double) balance.get(account), 0.01);
    }
}

// Test 10: Multiple accounts are independent
class Test10 {
    @Test
    void testMultipleAccountsIndependent() throws Exception {
        Class<?> cls = Class.forName("BankAccount");
        Constructor<?> constructor = cls.getConstructor(String.class, String.class, double.class, String.class);
        Object account1 = constructor.newInstance("ACC1", "Alice", 100.0, "Savings");
        Object account2 = constructor.newInstance("ACC2", "Bob", 200.0, "Checking");
        Field balance = cls.getDeclaredField("balance");
        balance.setAccessible(true);
        assertEquals(100.0, (double) balance.get(account1), 0.01);
        assertEquals(200.0, (double) balance.get(account2), 0.01);
    }
}`,
    translations: {
        ru: {
            title: 'Конструкторы и Инициализация',
            solutionCode: `public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;
    private String accountType;

    // Блок инициализации экземпляра - выполняется перед конструктором
    {
        System.out.println("Инициализация счета...");
    }

    // Конструктор по умолчанию
    public BankAccount() {
        this("ACC000000", "Неизвестный", 0.0, "Сберегательный");
    }

    // Конструктор с accountNumber и accountHolder
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0, "Сберегательный");
    }

    // Конструктор со всеми параметрами - основной конструктор
    public BankAccount(String accountNumber, String accountHolder,
                      double balance, String accountType) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
        this.accountType = accountType;
    }

    // Метод для внесения денег
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Внесено: $" + amount);
        } else {
            System.out.println("Неверная сумма для внесения");
        }
    }

    // Метод для снятия денег
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Снято: $" + amount);
        } else {
            System.out.println("Неверная сумма для снятия или недостаточно средств");
        }
    }

    // Отображение информации о счете
    public void displayAccountInfo() {
        System.out.println("=== Информация о счете ===");
        System.out.println("Номер счета: " + accountNumber);
        System.out.println("Владелец счета: " + accountHolder);
        System.out.println("Тип счета: " + accountType);
        System.out.println("Баланс: $" + balance);
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Использование конструктора по умолчанию
        BankAccount account1 = new BankAccount();
        account1.displayAccountInfo();

        // Использование конструктора с accountNumber и accountHolder
        BankAccount account2 = new BankAccount("ACC123456", "Джон Доу");
        account2.deposit(1000.0);
        account2.displayAccountInfo();

        // Использование конструктора со всеми параметрами
        BankAccount account3 = new BankAccount("ACC789012", "Джейн Смит",
                                               5000.0, "Текущий");
        account3.withdraw(500.0);
        account3.deposit(200.0);
        account3.displayAccountInfo();
    }
}`,
            description: `Создайте класс **BankAccount**, который демонстрирует перегрузку конструкторов и правильную инициализацию объектов.

**Требования:**
1. Создайте класс BankAccount с полями:
   1.1. accountNumber (String)
   1.2. accountHolder (String)
   1.3. balance (double)
   1.4. accountType (String) - "Сберегательный" или "Текущий"

2. Реализуйте несколько конструкторов:
   2.1. Конструктор по умолчанию: устанавливает значения по умолчанию
   2.2. Конструктор с accountNumber и accountHolder
   2.3. Конструктор со всеми параметрами
   2.4. Используйте цепочку конструкторов с this()

3. Добавьте блок инициализации экземпляра, который печатает "Инициализация счета..."

4. Реализуйте методы:
   4.1. deposit(double amount)
   4.2. withdraw(double amount)
   4.3. displayAccountInfo()

5. В методе main:
   5.1. Создайте счета, используя разные конструкторы
   5.2. Протестируйте внесение и снятие денег
   5.3. Отобразите информацию о счете

**Цели обучения:**
- Понять перегрузку конструкторов
- Изучить цепочку конструкторов с this()
- Практиковать блоки инициализации экземпляра`,
            hint1: `Начните с самого полного конструктора (со всеми параметрами), затем используйте this() для вызова его из других конструкторов со значениями по умолчанию.`,
            hint2: `Блоки инициализации экземпляра определяются с помощью фигурных скобок {} без каких-либо ключевых слов. Они выполняются до выполнения тела конструктора.`,
            whyItMatters: `Перегрузка конструкторов обеспечивает гибкость при создании объектов, позволяя пользователям инициализировать объекты различными способами. Цепочка конструкторов с this() способствует повторному использованию кода и уменьшает избыточность. Понимание порядка инициализации (блоки инициализации, затем конструкторы) необходимо для правильной настройки объекта.

**Продакшен паттерн:**
\`\`\`java
public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;

    // Конструктор по умолчанию
    public BankAccount() {
        this("ACC000000", "Неизвестный", 0.0);
    }

    // Конструктор с параметрами
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0); // Цепочка вызовов
    }

    // Главный конструктор
    public BankAccount(String accountNumber, String accountHolder, double balance) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
    }
}
\`\`\`

**Практические преимущества:**
- Несколько конструкторов обеспечивают гибкость при создании объектов
- Цепочка this() уменьшает дублирование кода
- Блоки инициализации позволяют выполнять общую логику до конструктора
- Перегрузка делает API класса более удобным для разных сценариев использования`
        },
        uz: {
            title: 'Konstruktorlar va Ishga Tushirish',
            solutionCode: `public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;
    private String accountType;

    // Nusxa ishga tushirish bloki - konstruktordan oldin ishga tushadi
    {
        System.out.println("Hisob ishga tushirilmoqda...");
    }

    // Standart konstruktor
    public BankAccount() {
        this("ACC000000", "Noma'lum", 0.0, "Jamg'arma");
    }

    // accountNumber va accountHolder bilan konstruktor
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0, "Jamg'arma");
    }

    // Barcha parametrlar bilan konstruktor - asosiy konstruktor
    public BankAccount(String accountNumber, String accountHolder,
                      double balance, String accountType) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
        this.accountType = accountType;
    }

    // Pul qo'shish metodi
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Qo'shildi: $" + amount);
        } else {
            System.out.println("Noto'g'ri qo'shish summasi");
        }
    }

    // Pul yechish metodi
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Yechildi: $" + amount);
        } else {
            System.out.println("Noto'g'ri yechish summasi yoki mablag' yetarli emas");
        }
    }

    // Hisob ma'lumotlarini ko'rsatish
    public void displayAccountInfo() {
        System.out.println("=== Hisob Ma'lumotlari ===");
        System.out.println("Hisob raqami: " + accountNumber);
        System.out.println("Hisob egasi: " + accountHolder);
        System.out.println("Hisob turi: " + accountType);
        System.out.println("Balans: $" + balance);
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Standart konstruktordan foydalanish
        BankAccount account1 = new BankAccount();
        account1.displayAccountInfo();

        // accountNumber va accountHolder bilan konstruktordan foydalanish
        BankAccount account2 = new BankAccount("ACC123456", "Jon Dou");
        account2.deposit(1000.0);
        account2.displayAccountInfo();

        // Barcha parametrlar bilan konstruktordan foydalanish
        BankAccount account3 = new BankAccount("ACC789012", "Jeyn Smit",
                                               5000.0, "Joriy");
        account3.withdraw(500.0);
        account3.deposit(200.0);
        account3.displayAccountInfo();
    }
}`,
            description: `Konstruktorlarning ortiqcha yuklanishini va obyektni to'g'ri ishga tushirishni ko'rsatadigan **BankAccount** sinfini yarating.

**Talablar:**
1. Quyidagi maydonlarga ega BankAccount sinfini yarating:
   1.1. accountNumber (String)
   1.2. accountHolder (String)
   1.3. balance (double)
   1.4. accountType (String) - "Jamg'arma" yoki "Joriy"

2. Bir nechta konstruktorlarni amalga oshiring:
   2.1. Standart konstruktor: standart qiymatlarni o'rnatadi
   2.2. accountNumber va accountHolder bilan konstruktor
   2.3. Barcha parametrlar bilan konstruktor
   2.4. this() bilan konstruktorlar zanjiridan foydalaning

3. "Hisob ishga tushirilmoqda..." deb chop etadigan nusxa ishga tushirish blokini qo'shing

4. Metodlarni amalga oshiring:
   4.1. deposit(double amount)
   4.2. withdraw(double amount)
   4.3. displayAccountInfo()

5. Main metodida:
   5.1. Turli konstruktorlardan foydalanib hisoblar yarating
   5.2. Pul qo'shish va yechishni sinab ko'ring
   5.3. Hisob ma'lumotlarini ko'rsating

**O'rganish maqsadlari:**
- Konstruktorlarning ortiqcha yuklanishini tushunish
- this() bilan konstruktorlar zanjirini o'rganish
- Nusxa ishga tushirish bloklarida amaliyot`,
            hint1: `Eng to'liq konstruktordan (barcha parametrlar bilan) boshlang, keyin this() dan foydalanib uni boshqa konstruktorlardan standart qiymatlar bilan chaqiring.`,
            hint2: `Nusxa ishga tushirish bloklari hech qanday kalit so'zsiz jingalak qavslar {} yordamida aniqlanadi. Ular konstruktor tanasi bajarilishidan oldin ishga tushadi.`,
            whyItMatters: `Konstruktorlarning ortiqcha yuklanishi obyektlarni yaratishda moslashuvchanlikni ta'minlaydi va foydalanuvchilarga obyektlarni turli yo'llar bilan ishga tushirish imkonini beradi. this() bilan konstruktorlar zanjiri kodni qayta ishlatishni rag'batlantiradi va ortiqchalikni kamaytiradi. Ishga tushirish tartibini tushunish (ishga tushirish bloklari, keyin konstruktorlar) obyektni to'g'ri sozlash uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
public class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;

    // Standart konstruktor
    public BankAccount() {
        this("ACC000000", "Noma'lum", 0.0);
    }

    // Parametrlar bilan konstruktor
    public BankAccount(String accountNumber, String accountHolder) {
        this(accountNumber, accountHolder, 0.0); // Zanjir chaqiruv
    }

    // Asosiy konstruktor
    public BankAccount(String accountNumber, String accountHolder, double balance) {
        this.accountNumber = accountNumber;
        this.accountHolder = accountHolder;
        this.balance = balance;
    }
}
\`\`\`

**Amaliy foydalari:**
- Bir nechta konstruktorlar obyektlarni yaratishda moslashuvchanlikni ta'minlaydi
- this() zanjiri kod takrorlanishini kamaytiradi
- Ishga tushirish bloklari konstruktordan oldin umumiy mantiqni bajarish imkonini beradi
- Ortiqcha yuklash sinf API sini turli foydalanish stsenariylari uchun qulayroq qiladi`
        }
    }
};

export default task;
