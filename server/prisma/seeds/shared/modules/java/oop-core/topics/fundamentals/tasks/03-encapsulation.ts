import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-encapsulation',
    title: 'Encapsulation and Access Modifiers',
    difficulty: 'easy',
    tags: ['java', 'oop', 'encapsulation', 'getters', 'setters', 'access-modifiers'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a **Product** class that demonstrates encapsulation principles and proper use of access modifiers.

**Requirements:**
1. Create a Product class with private fields:
   1.1. id (int)
   1.2. name (String)
   1.3. price (double)
   1.4. quantity (int)

2. Implement proper getters for all fields

3. Implement setters with validation:
   3.1. setPrice: only accept positive values
   3.2. setQuantity: only accept non-negative values
   3.3. name and id should only be set via constructor (no setters)

4. Add methods demonstrating encapsulation:
   4.1. calculateTotalValue(): returns price * quantity
   4.2. applyDiscount(double percentage): validates and applies discount
   4.3. isInStock(): returns true if quantity > 0

5. In main method:
   5.1. Create products
   5.2. Try to set invalid values (should be rejected)
   5.3. Use methods to interact with the object
   5.4. Demonstrate data hiding

**Learning Goals:**
- Understand the principle of encapsulation
- Learn proper use of access modifiers (private, public)
- Practice data validation in setters
- Understand why direct field access should be avoided`,
    initialCode: `public class Product {
    // TODO: Add private fields

    // TODO: Implement constructor

    // TODO: Implement getters

    // TODO: Implement setters with validation

    // TODO: Implement calculateTotalValue method

    // TODO: Implement applyDiscount method

    // TODO: Implement isInStock method

    public static void main(String[] args) {
        // TODO: Create products and test encapsulation
    }
}`,
    solutionCode: `public class Product {
    // Private fields - data is hidden from outside access
    private int id;
    private String name;
    private double price;
    private int quantity;

    // Constructor - only way to set id and name
    public Product(int id, String name, double price, int quantity) {
        this.id = id;
        this.name = name;
        setPrice(price);	// Using setter for validation
        setQuantity(quantity);	// Using setter for validation
    }

    // Getters - provide read access to private fields
    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    public int getQuantity() {
        return quantity;
    }

    // Setter with validation - only positive prices allowed
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Price must be positive. Price not updated.");
        }
    }

    // Setter with validation - only non-negative quantities allowed
    public void setQuantity(int quantity) {
        if (quantity >= 0) {
            this.quantity = quantity;
        } else {
            System.out.println("Quantity cannot be negative. Quantity not updated.");
        }
    }

    // Calculate total inventory value
    public double calculateTotalValue() {
        return price * quantity;
    }

    // Apply discount with validation
    public void applyDiscount(double percentage) {
        if (percentage > 0 && percentage <= 100) {
            double discount = price * (percentage / 100);
            price -= discount;
            System.out.println("Discount of " + percentage + "% applied.");
        } else {
            System.out.println("Invalid discount percentage.");
        }
    }

    // Check if product is in stock
    public boolean isInStock() {
        return quantity > 0;
    }

    // Display product information
    public void displayInfo() {
        System.out.println("=== Product Information ===");
        System.out.println("ID: " + id);
        System.out.println("Name: " + name);
        System.out.println("Price: $" + price);
        System.out.println("Quantity: " + quantity);
        System.out.println("Total Value: $" + calculateTotalValue());
        System.out.println("In Stock: " + (isInStock() ? "Yes" : "No"));
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Create a product
        Product laptop = new Product(101, "Gaming Laptop", 1500.0, 10);
        laptop.displayInfo();

        // Try to set invalid price (will be rejected)
        laptop.setPrice(-500.0);

        // Set valid price
        laptop.setPrice(1400.0);
        System.out.println("New price: $" + laptop.getPrice());

        // Apply discount
        laptop.applyDiscount(10);
        laptop.displayInfo();

        // Try invalid discount
        laptop.applyDiscount(150);

        // Update quantity
        laptop.setQuantity(5);

        // Try to set negative quantity (will be rejected)
        laptop.setQuantity(-3);

        laptop.displayInfo();

        // Check stock status
        Product outOfStock = new Product(102, "Mouse", 25.0, 0);
        System.out.println(outOfStock.getName() + " in stock: " +
                          outOfStock.isInStock());
    }
}`,
    hint1: `Make all fields private. This is the foundation of encapsulation - hiding implementation details from outside access.`,
    hint2: `In your setters, always validate the input before updating the field. If validation fails, print an error message and don't update the field.`,
    whyItMatters: `Encapsulation is a fundamental OOP principle that protects object integrity by controlling how data is accessed and modified. By making fields private and providing controlled access through getters/setters, you can validate data, maintain invariants, and change implementation without breaking external code. This leads to more maintainable and robust applications.

**Production Pattern:**
\`\`\`java
public class Product {
    private double price;
    private int quantity;

    // Setter with validation
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Price must be positive");
        }
    }

    // Getter for safe reading
    public double getPrice() {
        return price;
    }

    // Method with business logic
    public double calculateTotalValue() {
        return price * quantity; // Encapsulated calculations
    }
}
\`\`\`

**Practical Benefits:**
- Data validation in setters prevents invalid states
- Private fields protect against accidental modification
- Can change internal implementation without changing API
- Business logic is centralized in class methods`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.*;

// Test 1: Product class exists with private fields
class Test1 {
    @Test
    void testProductClassExists() throws Exception {
        Class<?> cls = Class.forName("Product");
        Field id = cls.getDeclaredField("id");
        Field name = cls.getDeclaredField("name");
        assertTrue(Modifier.isPrivate(id.getModifiers()));
        assertTrue(Modifier.isPrivate(name.getModifiers()));
    }
}

// Test 2: All fields are private
class Test2 {
    @Test
    void testAllFieldsPrivate() throws Exception {
        Class<?> cls = Class.forName("Product");
        Field price = cls.getDeclaredField("price");
        Field quantity = cls.getDeclaredField("quantity");
        assertTrue(Modifier.isPrivate(price.getModifiers()));
        assertTrue(Modifier.isPrivate(quantity.getModifiers()));
    }
}

// Test 3: Getters exist for all fields
class Test3 {
    @Test
    void testGettersExist() throws Exception {
        Class<?> cls = Class.forName("Product");
        assertNotNull(cls.getMethod("getId"));
        assertNotNull(cls.getMethod("getName"));
        assertNotNull(cls.getMethod("getPrice"));
        assertNotNull(cls.getMethod("getQuantity"));
    }
}

// Test 4: setPrice validates positive values
class Test4 {
    @Test
    void testSetPriceValidation() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 100.0, 10);
        Method setPrice = cls.getMethod("setPrice", double.class);
        setPrice.invoke(product, -50.0);
        Method getPrice = cls.getMethod("getPrice");
        assertEquals(100.0, (double) getPrice.invoke(product), 0.01);
    }
}

// Test 5: setQuantity validates non-negative values
class Test5 {
    @Test
    void testSetQuantityValidation() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 100.0, 10);
        Method setQuantity = cls.getMethod("setQuantity", int.class);
        setQuantity.invoke(product, -5);
        Method getQuantity = cls.getMethod("getQuantity");
        assertEquals(10, (int) getQuantity.invoke(product));
    }
}

// Test 6: calculateTotalValue works correctly
class Test6 {
    @Test
    void testCalculateTotalValue() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 25.0, 4);
        Method calcTotal = cls.getMethod("calculateTotalValue");
        assertEquals(100.0, (double) calcTotal.invoke(product), 0.01);
    }
}

// Test 7: applyDiscount works correctly
class Test7 {
    @Test
    void testApplyDiscount() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 100.0, 1);
        Method applyDiscount = cls.getMethod("applyDiscount", double.class);
        applyDiscount.invoke(product, 10.0);
        Method getPrice = cls.getMethod("getPrice");
        assertEquals(90.0, (double) getPrice.invoke(product), 0.01);
    }
}

// Test 8: isInStock returns correct values
class Test8 {
    @Test
    void testIsInStock() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object inStock = constructor.newInstance(1, "Test", 10.0, 5);
        Object outOfStock = constructor.newInstance(2, "Test2", 10.0, 0);
        Method isInStock = cls.getMethod("isInStock");
        assertTrue((boolean) isInStock.invoke(inStock));
        assertFalse((boolean) isInStock.invoke(outOfStock));
    }
}

// Test 9: applyDiscount rejects invalid percentages
class Test9 {
    @Test
    void testApplyDiscountValidation() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 100.0, 1);
        Method applyDiscount = cls.getMethod("applyDiscount", double.class);
        applyDiscount.invoke(product, 150.0);
        Method getPrice = cls.getMethod("getPrice");
        assertEquals(100.0, (double) getPrice.invoke(product), 0.01);
    }
}

// Test 10: Valid price update works
class Test10 {
    @Test
    void testValidPriceUpdate() throws Exception {
        Class<?> cls = Class.forName("Product");
        Constructor<?> constructor = cls.getConstructor(int.class, String.class, double.class, int.class);
        Object product = constructor.newInstance(1, "Test", 100.0, 10);
        Method setPrice = cls.getMethod("setPrice", double.class);
        setPrice.invoke(product, 150.0);
        Method getPrice = cls.getMethod("getPrice");
        assertEquals(150.0, (double) getPrice.invoke(product), 0.01);
    }
}`,
    translations: {
        ru: {
            title: 'Инкапсуляция и Модификаторы Доступа',
            solutionCode: `public class Product {
    // Приватные поля - данные скрыты от внешнего доступа
    private int id;
    private String name;
    private double price;
    private int quantity;

    // Конструктор - единственный способ установить id и name
    public Product(int id, String name, double price, int quantity) {
        this.id = id;
        this.name = name;
        setPrice(price);	// Использование setter для валидации
        setQuantity(quantity);	// Использование setter для валидации
    }

    // Геттеры - предоставляют доступ на чтение к приватным полям
    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    public int getQuantity() {
        return quantity;
    }

    // Сеттер с валидацией - разрешены только положительные цены
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Цена должна быть положительной. Цена не обновлена.");
        }
    }

    // Сеттер с валидацией - разрешены только неотрицательные количества
    public void setQuantity(int quantity) {
        if (quantity >= 0) {
            this.quantity = quantity;
        } else {
            System.out.println("Количество не может быть отрицательным. Количество не обновлено.");
        }
    }

    // Расчет общей стоимости товара
    public double calculateTotalValue() {
        return price * quantity;
    }

    // Применение скидки с валидацией
    public void applyDiscount(double percentage) {
        if (percentage > 0 && percentage <= 100) {
            double discount = price * (percentage / 100);
            price -= discount;
            System.out.println("Применена скидка " + percentage + "%.");
        } else {
            System.out.println("Недопустимый процент скидки.");
        }
    }

    // Проверка наличия товара на складе
    public boolean isInStock() {
        return quantity > 0;
    }

    // Отображение информации о товаре
    public void displayInfo() {
        System.out.println("=== Информация о товаре ===");
        System.out.println("ID: " + id);
        System.out.println("Название: " + name);
        System.out.println("Цена: $" + price);
        System.out.println("Количество: " + quantity);
        System.out.println("Общая стоимость: $" + calculateTotalValue());
        System.out.println("На складе: " + (isInStock() ? "Да" : "Нет"));
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Создание товара
        Product laptop = new Product(101, "Игровой ноутбук", 1500.0, 10);
        laptop.displayInfo();

        // Попытка установить недопустимую цену (будет отклонена)
        laptop.setPrice(-500.0);

        // Установка допустимой цены
        laptop.setPrice(1400.0);
        System.out.println("Новая цена: $" + laptop.getPrice());

        // Применение скидки
        laptop.applyDiscount(10);
        laptop.displayInfo();

        // Попытка применить недопустимую скидку
        laptop.applyDiscount(150);

        // Обновление количества
        laptop.setQuantity(5);

        // Попытка установить отрицательное количество (будет отклонена)
        laptop.setQuantity(-3);

        laptop.displayInfo();

        // Проверка статуса наличия
        Product outOfStock = new Product(102, "Мышь", 25.0, 0);
        System.out.println(outOfStock.getName() + " на складе: " +
                          outOfStock.isInStock());
    }
}`,
            description: `Создайте класс **Product**, который демонстрирует принципы инкапсуляции и правильное использование модификаторов доступа.

**Требования:**
1. Создайте класс Product с приватными полями:
   1.1. id (int)
   1.2. name (String)
   1.3. price (double)
   1.4. quantity (int)

2. Реализуйте геттеры для всех полей

3. Реализуйте сеттеры с валидацией:
   3.1. setPrice: принимать только положительные значения
   3.2. setQuantity: принимать только неотрицательные значения
   3.3. name и id должны устанавливаться только через конструктор (без сеттеров)

4. Добавьте методы, демонстрирующие инкапсуляцию:
   4.1. calculateTotalValue(): возвращает price * quantity
   4.2. applyDiscount(double percentage): проверяет и применяет скидку
   4.3. isInStock(): возвращает true, если quantity > 0

5. В методе main:
   5.1. Создайте товары
   5.2. Попытайтесь установить недопустимые значения (должны быть отклонены)
   5.3. Используйте методы для взаимодействия с объектом
   5.4. Продемонстрируйте сокрытие данных

**Цели обучения:**
- Понять принцип инкапсуляции
- Изучить правильное использование модификаторов доступа (private, public)
- Практиковать валидацию данных в сеттерах
- Понять, почему следует избегать прямого доступа к полям`,
            hint1: `Сделайте все поля приватными. Это основа инкапсуляции - сокрытие деталей реализации от внешнего доступа.`,
            hint2: `В ваших сеттерах всегда проверяйте входные данные перед обновлением поля. Если валидация не пройдена, выведите сообщение об ошибке и не обновляйте поле.`,
            whyItMatters: `Инкапсуляция - это фундаментальный принцип ООП, который защищает целостность объекта, контролируя доступ к данным и их изменение. Делая поля приватными и предоставляя контролируемый доступ через геттеры/сеттеры, вы можете проверять данные, поддерживать инварианты и изменять реализацию без нарушения внешнего кода. Это приводит к более поддерживаемым и надежным приложениям.

**Продакшен паттерн:**
\`\`\`java
public class Product {
    private double price;
    private int quantity;

    // Сеттер с валидацией
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Цена должна быть положительной");
        }
    }

    // Геттер для безопасного чтения
    public double getPrice() {
        return price;
    }

    // Метод с бизнес-логикой
    public double calculateTotalValue() {
        return price * quantity; // Инкапсулированные вычисления
    }
}
\`\`\`

**Практические преимущества:**
- Валидация данных в сеттерах предотвращает недопустимые состояния
- Приватные поля защищают от случайного изменения
- Можно изменить внутреннюю реализацию без изменения API
- Бизнес-логика централизована в методах класса`
        },
        uz: {
            title: 'Inkapsulyatsiya va Kirish Modifikatorlari',
            solutionCode: `public class Product {
    // Xususiy maydonlar - ma'lumotlar tashqi kirishdan yashirilgan
    private int id;
    private String name;
    private double price;
    private int quantity;

    // Konstruktor - id va name ni o'rnatishning yagona yo'li
    public Product(int id, String name, double price, int quantity) {
        this.id = id;
        this.name = name;
        setPrice(price);	// Tekshirish uchun setter dan foydalanish
        setQuantity(quantity);	// Tekshirish uchun setter dan foydalanish
    }

    // Getterlar - xususiy maydonlarga o'qish kirishi beradi
    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    public int getQuantity() {
        return quantity;
    }

    // Tekshiruvli setter - faqat musbat narxlar ruxsat etiladi
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Narx musbat bo'lishi kerak. Narx yangilanmadi.");
        }
    }

    // Tekshiruvli setter - faqat manfiy bo'lmagan miqdorlar ruxsat etiladi
    public void setQuantity(int quantity) {
        if (quantity >= 0) {
            this.quantity = quantity;
        } else {
            System.out.println("Miqdor manfiy bo'lishi mumkin emas. Miqdor yangilanmadi.");
        }
    }

    // Umumiy inventar qiymatini hisoblash
    public double calculateTotalValue() {
        return price * quantity;
    }

    // Tekshiruv bilan chegirma qo'llash
    public void applyDiscount(double percentage) {
        if (percentage > 0 && percentage <= 100) {
            double discount = price * (percentage / 100);
            price -= discount;
            System.out.println(percentage + "% chegirma qo'llandi.");
        } else {
            System.out.println("Noto'g'ri chegirma foizi.");
        }
    }

    // Mahsulot omborda borligini tekshirish
    public boolean isInStock() {
        return quantity > 0;
    }

    // Mahsulot ma'lumotlarini ko'rsatish
    public void displayInfo() {
        System.out.println("=== Mahsulot Ma'lumotlari ===");
        System.out.println("ID: " + id);
        System.out.println("Nomi: " + name);
        System.out.println("Narxi: $" + price);
        System.out.println("Miqdori: " + quantity);
        System.out.println("Umumiy qiymati: $" + calculateTotalValue());
        System.out.println("Omborda: " + (isInStock() ? "Ha" : "Yo'q"));
        System.out.println("========================");
    }

    public static void main(String[] args) {
        // Mahsulot yaratish
        Product laptop = new Product(101, "O'yin noutbuki", 1500.0, 10);
        laptop.displayInfo();

        // Noto'g'ri narx o'rnatishga harakat (rad etiladi)
        laptop.setPrice(-500.0);

        // To'g'ri narx o'rnatish
        laptop.setPrice(1400.0);
        System.out.println("Yangi narx: $" + laptop.getPrice());

        // Chegirma qo'llash
        laptop.applyDiscount(10);
        laptop.displayInfo();

        // Noto'g'ri chegirma qo'llashga harakat
        laptop.applyDiscount(150);

        // Miqdorni yangilash
        laptop.setQuantity(5);

        // Manfiy miqdor o'rnatishga harakat (rad etiladi)
        laptop.setQuantity(-3);

        laptop.displayInfo();

        // Ombor holatini tekshirish
        Product outOfStock = new Product(102, "Sichqoncha", 25.0, 0);
        System.out.println(outOfStock.getName() + " omborda: " +
                          outOfStock.isInStock());
    }
}`,
            description: `Inkapsulyatsiya tamoyillarini va kirish modifikatorlaridan to'g'ri foydalanishni ko'rsatadigan **Product** sinfini yarating.

**Talablar:**
1. Xususiy maydonlarga ega Product sinfini yarating:
   1.1. id (int)
   1.2. name (String)
   1.3. price (double)
   1.4. quantity (int)

2. Barcha maydonlar uchun getterlarni amalga oshiring

3. Tekshiruv bilan setterlarni amalga oshiring:
   3.1. setPrice: faqat musbat qiymatlarni qabul qilish
   3.2. setQuantity: faqat manfiy bo'lmagan qiymatlarni qabul qilish
   3.3. name va id faqat konstruktor orqali o'rnatilishi kerak (setterlar yo'q)

4. Inkapsulyatsiyani ko'rsatadigan metodlar qo'shing:
   4.1. calculateTotalValue(): price * quantity ni qaytaradi
   4.2. applyDiscount(double percentage): tekshiradi va chegirma qo'llaydi
   4.3. isInStock(): agar quantity > 0 bo'lsa true qaytaradi

5. Main metodida:
   5.1. Mahsulotlar yarating
   5.2. Noto'g'ri qiymatlarni o'rnatishga harakat qiling (rad etilishi kerak)
   5.3. Obyekt bilan muloqot qilish uchun metodlardan foydalaning
   5.4. Ma'lumotlarni yashirishni namoyish eting

**O'rganish maqsadlari:**
- Inkapsulyatsiya tamoyilini tushunish
- Kirish modifikatorlaridan to'g'ri foydalanishni o'rganish (private, public)
- Setterlarda ma'lumotlarni tekshirishda amaliyot
- Maydonlarga to'g'ridan-to'g'ri kirishdan nima uchun qochish kerakligini tushunish`,
            hint1: `Barcha maydonlarni xususiy qiling. Bu inkapsulyatsiyaning asosi - amalga oshirish tafsilotlarini tashqi kirishdan yashirish.`,
            hint2: `Setterlaringizda maydonni yangilashdan oldin har doim kirishni tekshiring. Agar tekshiruv muvaffaqiyatsiz bo'lsa, xato xabarini chop eting va maydonni yangilamang.`,
            whyItMatters: `Inkapsulyatsiya - bu ma'lumotlarga qanday kirilishi va o'zgartirilishini nazorat qilish orqali obyekt yaxlitligini himoya qiladigan asosiy OOP tamoyili. Maydonlarni xususiy qilib va getterlar/setterlar orqali nazorat qilinadigan kirish berish orqali siz ma'lumotlarni tekshirishingiz, invariantlarni saqlashingiz va tashqi kodni buzmasdan amalga oshirishni o'zgartirishingiz mumkin. Bu yanada barqaror va ishonchli ilovalarga olib keladi.

**Ishlab chiqarish patterni:**
\`\`\`java
public class Product {
    private double price;
    private int quantity;

    // Tekshiruv bilan setter
    public void setPrice(double price) {
        if (price > 0) {
            this.price = price;
        } else {
            System.out.println("Narx musbat bo'lishi kerak");
        }
    }

    // Xavfsiz o'qish uchun getter
    public double getPrice() {
        return price;
    }

    // Biznes mantiq bilan metod
    public double calculateTotalValue() {
        return price * quantity; // Inkapsulyatsiyalangan hisoblashlar
    }
}
\`\`\`

**Amaliy foydalari:**
- Setterlarda ma'lumotlarni tekshirish noto'g'ri holatlarning oldini oladi
- Xususiy maydonlar tasodifiy o'zgarishlardan himoya qiladi
- API ni o'zgartirmasdan ichki amalga oshirishni o'zgartirish mumkin
- Biznes mantiq sinf metodlarida markazlashtirilgan`
        }
    }
};

export default task;
