import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-inheritance',
    title: 'Inheritance and super Keyword',
    difficulty: 'medium',
    tags: ['java', 'oop', 'inheritance', 'super', 'extends', 'method-overriding'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create an inheritance hierarchy for a vehicle system that demonstrates proper use of inheritance and the super keyword.

**Requirements:**
1. Create a base class **Vehicle** with:
   1.1. Protected fields: brand, model, year
   1.2. Constructor to initialize all fields
   1.3. Methods: startEngine(), stopEngine(), displayInfo()

2. Create a **Car** class that extends Vehicle:
   2.1. Additional field: numberOfDoors (int)
   2.2. Constructor that calls super() and initializes numberOfDoors
   2.3. Override displayInfo() to include car-specific information
   2.4. Add a car-specific method: openTrunk()

3. Create a **Motorcycle** class that extends Vehicle:
   3.1. Additional field: hasStorage (boolean)
   3.2. Constructor that calls super() and initializes hasStorage
   3.3. Override displayInfo() to include motorcycle-specific information
   3.4. Add a motorcycle-specific method: wheelie()

4. In main method:
   4.1. Create instances of Car and Motorcycle
   4.2. Call inherited methods
   4.3. Call overridden methods
   4.4. Call child-specific methods
   4.5. Demonstrate the inheritance relationship

**Learning Goals:**
- Understand inheritance and the "is-a" relationship
- Learn to use extends keyword
- Practice using super() to call parent constructor
- Learn method overriding and @Override annotation
- Understand protected access modifier`,
    initialCode: `// TODO: Create Vehicle base class

// TODO: Create Car class extending Vehicle

// TODO: Create Motorcycle class extending Vehicle

public class InheritanceDemo {
    public static void main(String[] args) {
        // TODO: Create and test Vehicle hierarchy
    }
}`,
    solutionCode: `// Base class - represents common vehicle properties and behaviors
class Vehicle {
    // Protected fields - accessible to subclasses
    protected String brand;
    protected String model;
    protected int year;

    // Constructor
    public Vehicle(String brand, String model, int year) {
        this.brand = brand;
        this.model = model;
        this.year = year;
    }

    // Method to start the engine
    public void startEngine() {
        System.out.println("Engine started for " + brand + " " + model);
    }

    // Method to stop the engine
    public void stopEngine() {
        System.out.println("Engine stopped for " + brand + " " + model);
    }

    // Method to display vehicle information
    public void displayInfo() {
        System.out.println("=== Vehicle Information ===");
        System.out.println("Brand: " + brand);
        System.out.println("Model: " + model);
        System.out.println("Year: " + year);
    }
}

// Car class - inherits from Vehicle
class Car extends Vehicle {
    private int numberOfDoors;

    // Constructor - calls parent constructor using super()
    public Car(String brand, String model, int year, int numberOfDoors) {
        super(brand, model, year); // Call parent constructor
        this.numberOfDoors = numberOfDoors;
    }

    // Override parent method to add car-specific information
    @Override
    public void displayInfo() {
        super.displayInfo(); // Call parent version first
        System.out.println("Type: Car");
        System.out.println("Number of Doors: " + numberOfDoors);
        System.out.println("========================");
    }

    // Car-specific method
    public void openTrunk() {
        System.out.println("Opening trunk of " + brand + " " + model);
    }
}

// Motorcycle class - inherits from Vehicle
class Motorcycle extends Vehicle {
    private boolean hasStorage;

    // Constructor - calls parent constructor using super()
    public Motorcycle(String brand, String model, int year, boolean hasStorage) {
        super(brand, model, year); // Call parent constructor
        this.hasStorage = hasStorage;
    }

    // Override parent method to add motorcycle-specific information
    @Override
    public void displayInfo() {
        super.displayInfo(); // Call parent version first
        System.out.println("Type: Motorcycle");
        System.out.println("Has Storage: " + (hasStorage ? "Yes" : "No"));
        System.out.println("========================");
    }

    // Motorcycle-specific method
    public void wheelie() {
        System.out.println("Performing wheelie on " + brand + " " + model + "!");
    }
}

public class InheritanceDemo {
    public static void main(String[] args) {
        // Create a Car object
        Car myCar = new Car("Toyota", "Camry", 2023, 4);

        // Call inherited methods
        myCar.startEngine();

        // Call overridden method
        myCar.displayInfo();

        // Call car-specific method
        myCar.openTrunk();
        myCar.stopEngine();

        System.out.println();

        // Create a Motorcycle object
        Motorcycle myBike = new Motorcycle("Harley-Davidson", "Street 750", 2022, true);

        // Call inherited methods
        myBike.startEngine();

        // Call overridden method
        myBike.displayInfo();

        // Call motorcycle-specific method
        myBike.wheelie();
        myBike.stopEngine();

        System.out.println();

        // Demonstrate that Car and Motorcycle are both Vehicles
        System.out.println("Car is a Vehicle: " + (myCar instanceof Vehicle));
        System.out.println("Motorcycle is a Vehicle: " + (myBike instanceof Vehicle));
    }
}`,
    hint1: `Use the extends keyword to create a subclass. In the child constructor, the first line should be super() to call the parent constructor.`,
    hint2: `When overriding methods, use @Override annotation. You can call the parent version of the method using super.methodName().`,
    whyItMatters: `Inheritance is a cornerstone of OOP that promotes code reuse and establishes "is-a" relationships between classes. It allows you to create specialized versions of classes while inheriting common functionality. Understanding super() is crucial for proper initialization of inherited objects, and method overriding enables polymorphic behavior. This creates more maintainable and extensible code hierarchies.`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.*;

// Test 1: Vehicle class exists with protected fields
class Test1 {
    @Test
    void testVehicleClassExists() throws Exception {
        Class<?> cls = Class.forName("Vehicle");
        Field brand = cls.getDeclaredField("brand");
        Field model = cls.getDeclaredField("model");
        assertTrue(Modifier.isProtected(brand.getModifiers()));
        assertTrue(Modifier.isProtected(model.getModifiers()));
    }
}

// Test 2: Car extends Vehicle
class Test2 {
    @Test
    void testCarExtendsVehicle() throws Exception {
        Class<?> vehicle = Class.forName("Vehicle");
        Class<?> car = Class.forName("Car");
        assertTrue(vehicle.isAssignableFrom(car));
    }
}

// Test 3: Motorcycle extends Vehicle
class Test3 {
    @Test
    void testMotorcycleExtendsVehicle() throws Exception {
        Class<?> vehicle = Class.forName("Vehicle");
        Class<?> motorcycle = Class.forName("Motorcycle");
        assertTrue(vehicle.isAssignableFrom(motorcycle));
    }
}

// Test 4: Car has numberOfDoors field
class Test4 {
    @Test
    void testCarHasNumberOfDoors() throws Exception {
        Class<?> cls = Class.forName("Car");
        Field field = cls.getDeclaredField("numberOfDoors");
        assertNotNull(field);
    }
}

// Test 5: Motorcycle has hasStorage field
class Test5 {
    @Test
    void testMotorcycleHasStorage() throws Exception {
        Class<?> cls = Class.forName("Motorcycle");
        Field field = cls.getDeclaredField("hasStorage");
        assertNotNull(field);
    }
}

// Test 6: Vehicle startEngine method exists
class Test6 {
    @Test
    void testStartEngineMethod() throws Exception {
        Class<?> cls = Class.forName("Vehicle");
        Method method = cls.getMethod("startEngine");
        assertNotNull(method);
    }
}

// Test 7: Car overrides displayInfo
class Test7 {
    @Test
    void testCarOverridesDisplayInfo() throws Exception {
        Class<?> car = Class.forName("Car");
        Method method = car.getMethod("displayInfo");
        assertEquals(car, method.getDeclaringClass());
    }
}

// Test 8: Car has openTrunk method
class Test8 {
    @Test
    void testCarHasOpenTrunk() throws Exception {
        Class<?> cls = Class.forName("Car");
        Method method = cls.getMethod("openTrunk");
        assertNotNull(method);
    }
}

// Test 9: Motorcycle has wheelie method
class Test9 {
    @Test
    void testMotorcycleHasWheelie() throws Exception {
        Class<?> cls = Class.forName("Motorcycle");
        Method method = cls.getMethod("wheelie");
        assertNotNull(method);
    }
}

// Test 10: Car is instance of Vehicle
class Test10 {
    @Test
    void testCarIsInstanceOfVehicle() throws Exception {
        Class<?> carClass = Class.forName("Car");
        Constructor<?> constructor = carClass.getConstructor(String.class, String.class, int.class, int.class);
        Object car = constructor.newInstance("Toyota", "Camry", 2023, 4);
        Class<?> vehicle = Class.forName("Vehicle");
        assertTrue(vehicle.isInstance(car));
    }
}`,
    translations: {
        ru: {
            title: 'Наследование и Ключевое Слово super',
            solutionCode: `// Базовый класс - представляет общие свойства и поведение транспортных средств
class Vehicle {
    // Защищенные поля - доступны для подклассов
    protected String brand;
    protected String model;
    protected int year;

    // Конструктор
    public Vehicle(String brand, String model, int year) {
        this.brand = brand;
        this.model = model;
        this.year = year;
    }

    // Метод для запуска двигателя
    public void startEngine() {
        System.out.println("Двигатель запущен для " + brand + " " + model);
    }

    // Метод для остановки двигателя
    public void stopEngine() {
        System.out.println("Двигатель остановлен для " + brand + " " + model);
    }

    // Метод для отображения информации о транспортном средстве
    public void displayInfo() {
        System.out.println("=== Информация о транспортном средстве ===");
        System.out.println("Марка: " + brand);
        System.out.println("Модель: " + model);
        System.out.println("Год: " + year);
    }
}

// Класс Car - наследуется от Vehicle
class Car extends Vehicle {
    private int numberOfDoors;

    // Конструктор - вызывает конструктор родителя с помощью super()
    public Car(String brand, String model, int year, int numberOfDoors) {
        super(brand, model, year); // Вызов конструктора родителя
        this.numberOfDoors = numberOfDoors;
    }

    // Переопределение метода родителя для добавления информации о машине
    @Override
    public void displayInfo() {
        super.displayInfo(); // Сначала вызываем версию родителя
        System.out.println("Тип: Автомобиль");
        System.out.println("Количество дверей: " + numberOfDoors);
        System.out.println("========================");
    }

    // Метод специфичный для автомобиля
    public void openTrunk() {
        System.out.println("Открытие багажника " + brand + " " + model);
    }
}

// Класс Motorcycle - наследуется от Vehicle
class Motorcycle extends Vehicle {
    private boolean hasStorage;

    // Конструктор - вызывает конструктор родителя с помощью super()
    public Motorcycle(String brand, String model, int year, boolean hasStorage) {
        super(brand, model, year); // Вызов конструктора родителя
        this.hasStorage = hasStorage;
    }

    // Переопределение метода родителя для добавления информации о мотоцикле
    @Override
    public void displayInfo() {
        super.displayInfo(); // Сначала вызываем версию родителя
        System.out.println("Тип: Мотоцикл");
        System.out.println("Есть хранилище: " + (hasStorage ? "Да" : "Нет"));
        System.out.println("========================");
    }

    // Метод специфичный для мотоцикла
    public void wheelie() {
        System.out.println("Выполнение вилли на " + brand + " " + model + "!");
    }
}

public class InheritanceDemo {
    public static void main(String[] args) {
        // Создание объекта Car
        Car myCar = new Car("Toyota", "Camry", 2023, 4);

        // Вызов унаследованных методов
        myCar.startEngine();

        // Вызов переопределенного метода
        myCar.displayInfo();

        // Вызов метода специфичного для автомобиля
        myCar.openTrunk();
        myCar.stopEngine();

        System.out.println();

        // Создание объекта Motorcycle
        Motorcycle myBike = new Motorcycle("Harley-Davidson", "Street 750", 2022, true);

        // Вызов унаследованных методов
        myBike.startEngine();

        // Вызов переопределенного метода
        myBike.displayInfo();

        // Вызов метода специфичного для мотоцикла
        myBike.wheelie();
        myBike.stopEngine();

        System.out.println();

        // Демонстрация того, что Car и Motorcycle являются Vehicle
        System.out.println("Car является Vehicle: " + (myCar instanceof Vehicle));
        System.out.println("Motorcycle является Vehicle: " + (myBike instanceof Vehicle));
    }
}`,
            description: `Создайте иерархию наследования для системы транспортных средств, которая демонстрирует правильное использование наследования и ключевого слова super.

**Требования:**
1. Создайте базовый класс **Vehicle** с:
   1.1. Защищенными полями: brand, model, year
   1.2. Конструктором для инициализации всех полей
   1.3. Методами: startEngine(), stopEngine(), displayInfo()

2. Создайте класс **Car**, расширяющий Vehicle:
   2.1. Дополнительное поле: numberOfDoors (int)
   2.2. Конструктор, вызывающий super() и инициализирующий numberOfDoors
   2.3. Переопределите displayInfo() для включения информации о машине
   2.4. Добавьте метод специфичный для машины: openTrunk()

3. Создайте класс **Motorcycle**, расширяющий Vehicle:
   3.1. Дополнительное поле: hasStorage (boolean)
   3.2. Конструктор, вызывающий super() и инициализирующий hasStorage
   3.3. Переопределите displayInfo() для включения информации о мотоцикле
   3.4. Добавьте метод специфичный для мотоцикла: wheelie()

4. В методе main:
   4.1. Создайте экземпляры Car и Motorcycle
   4.2. Вызовите унаследованные методы
   4.3. Вызовите переопределенные методы
   4.4. Вызовите методы специфичные для потомков
   4.5. Продемонстрируйте отношение наследования

**Цели обучения:**
- Понять наследование и отношение "является"
- Научиться использовать ключевое слово extends
- Практиковаться в использовании super() для вызова конструктора родителя
- Изучить переопределение методов и аннотацию @Override
- Понять модификатор доступа protected`,
            hint1: `Используйте ключевое слово extends для создания подкласса. В конструкторе потомка первой строкой должен быть super() для вызова конструктора родителя.`,
            hint2: `При переопределении методов используйте аннотацию @Override. Вы можете вызвать родительскую версию метода, используя super.methodName().`,
            whyItMatters: `Наследование - это краеугольный камень ООП, который способствует повторному использованию кода и устанавливает отношения "является" между классами. Оно позволяет создавать специализированные версии классов, наследуя общую функциональность. Понимание super() имеет решающее значение для правильной инициализации унаследованных объектов, а переопределение методов обеспечивает полиморфное поведение. Это создает более поддерживаемые и расширяемые иерархии кода.`
        },
        uz: {
            title: 'Meros va super Kalit So\'zi',
            solutionCode: `// Asosiy sinf - umumiy transport vositalarining xususiyatlari va xatti-harakatlarini ifodalaydi
class Vehicle {
    // Himoyalangan maydonlar - subsinflarga ochiq
    protected String brand;
    protected String model;
    protected int year;

    // Konstruktor
    public Vehicle(String brand, String model, int year) {
        this.brand = brand;
        this.model = model;
        this.year = year;
    }

    // Dvigatelni ishga tushirish metodi
    public void startEngine() {
        System.out.println(brand + " " + model + " uchun dvigatel ishga tushdi");
    }

    // Dvigatelni to'xtatish metodi
    public void stopEngine() {
        System.out.println(brand + " " + model + " uchun dvigatel to'xtatildi");
    }

    // Transport vositasi ma'lumotlarini ko'rsatish metodi
    public void displayInfo() {
        System.out.println("=== Transport Vositasi Ma'lumotlari ===");
        System.out.println("Brend: " + brand);
        System.out.println("Model: " + model);
        System.out.println("Yil: " + year);
    }
}

// Car sinfi - Vehicle dan meros oladi
class Car extends Vehicle {
    private int numberOfDoors;

    // Konstruktor - super() yordamida ota-ona konstruktorini chaqiradi
    public Car(String brand, String model, int year, int numberOfDoors) {
        super(brand, model, year); // Ota-ona konstruktorini chaqirish
        this.numberOfDoors = numberOfDoors;
    }

    // Avtomobil ma'lumotlarini qo'shish uchun ota-ona metodini qayta yozish
    @Override
    public void displayInfo() {
        super.displayInfo(); // Avval ota-ona versiyasini chaqirish
        System.out.println("Turi: Avtomobil");
        System.out.println("Eshiklar soni: " + numberOfDoors);
        System.out.println("========================");
    }

    // Avtomobilga xos metod
    public void openTrunk() {
        System.out.println(brand + " " + model + " ning yukxonasini ochish");
    }
}

// Motorcycle sinfi - Vehicle dan meros oladi
class Motorcycle extends Vehicle {
    private boolean hasStorage;

    // Konstruktor - super() yordamida ota-ona konstruktorini chaqiradi
    public Motorcycle(String brand, String model, int year, boolean hasStorage) {
        super(brand, model, year); // Ota-ona konstruktorini chaqirish
        this.hasStorage = hasStorage;
    }

    // Mototsikl ma'lumotlarini qo'shish uchun ota-ona metodini qayta yozish
    @Override
    public void displayInfo() {
        super.displayInfo(); // Avval ota-ona versiyasini chaqirish
        System.out.println("Turi: Mototsikl");
        System.out.println("Saqlash joyi bor: " + (hasStorage ? "Ha" : "Yo'q"));
        System.out.println("========================");
    }

    // Mototsiklga xos metod
    public void wheelie() {
        System.out.println(brand + " " + model + " da wheelie bajarish!");
    }
}

public class InheritanceDemo {
    public static void main(String[] args) {
        // Car obyektini yaratish
        Car myCar = new Car("Toyota", "Camry", 2023, 4);

        // Meros qilib olingan metodlarni chaqirish
        myCar.startEngine();

        // Qayta yozilgan metodini chaqirish
        myCar.displayInfo();

        // Avtomobilga xos metodini chaqirish
        myCar.openTrunk();
        myCar.stopEngine();

        System.out.println();

        // Motorcycle obyektini yaratish
        Motorcycle myBike = new Motorcycle("Harley-Davidson", "Street 750", 2022, true);

        // Meros qilib olingan metodlarni chaqirish
        myBike.startEngine();

        // Qayta yozilgan metodini chaqirish
        myBike.displayInfo();

        // Mototsiklga xos metodini chaqirish
        myBike.wheelie();
        myBike.stopEngine();

        System.out.println();

        // Car va Motorcycle ikkalasi ham Vehicle ekanligini namoyish etish
        System.out.println("Car Vehicle hisoblanadi: " + (myCar instanceof Vehicle));
        System.out.println("Motorcycle Vehicle hisoblanadi: " + (myBike instanceof Vehicle));
    }
}`,
            description: `Meros olish va super kalit so'zidan to'g'ri foydalanishni ko'rsatadigan transport vositalari tizimi uchun meros ierarxiyasini yarating.

**Talablar:**
1. **Vehicle** asosiy sinfini quyidagilar bilan yarating:
   1.1. Himoyalangan maydonlar: brand, model, year
   1.2. Barcha maydonlarni ishga tushirish uchun konstruktor
   1.3. Metodlar: startEngine(), stopEngine(), displayInfo()

2. Vehicle ni kengaytiradigan **Car** sinfini yarating:
   2.1. Qo'shimcha maydon: numberOfDoors (int)
   2.2. super() ni chaqiruvchi va numberOfDoors ni ishga tushiruvchi konstruktor
   2.3. Avtomobilga xos ma'lumotlarni kiritish uchun displayInfo() ni qayta yozing
   2.4. Avtomobilga xos metod qo'shing: openTrunk()

3. Vehicle ni kengaytiradigan **Motorcycle** sinfini yarating:
   3.1. Qo'shimcha maydon: hasStorage (boolean)
   3.2. super() ni chaqiruvchi va hasStorage ni ishga tushiruvchi konstruktor
   3.3. Mototsiklga xos ma'lumotlarni kiritish uchun displayInfo() ni qayta yozing
   3.4. Mototsiklga xos metod qo'shing: wheelie()

4. Main metodida:
   4.1. Car va Motorcycle nusxalarini yarating
   4.2. Meros qilib olingan metodlarni chaqiring
   4.3. Qayta yozilgan metodlarni chaqiring
   4.4. Bolalarga xos metodlarni chaqiring
   4.5. Meros munosabatini namoyish eting

**O'rganish maqsadlari:**
- Meros va "hisoblanadi" munosabatini tushunish
- extends kalit so'zidan foydalanishni o'rganish
- Ota-ona konstruktorini chaqirish uchun super() dan foydalanishda amaliyot
- Metodlarni qayta yozish va @Override annotatsiyasini o'rganish
- protected kirish modifikatorini tushunish`,
            hint1: `Subsinf yaratish uchun extends kalit so'zidan foydalaning. Bola konstruktorida birinchi qator ota-ona konstruktorini chaqirish uchun super() bo'lishi kerak.`,
            hint2: `Metodlarni qayta yozishda @Override annotatsiyasidan foydalaning. super.methodName() dan foydalanib metodning ota-ona versiyasini chaqirishingiz mumkin.`,
            whyItMatters: `Meros OOP ning asosiy tosh bo'lib, kodni qayta ishlatishni rag'batlantiradi va sinflar o'rtasida "hisoblanadi" munosabatlarini o'rnatadi. U umumiy funksionallikni meros qilib olgan holda sinflarning maxsus versiyalarini yaratishga imkon beradi. super() ni tushunish meros qilib olingan obyektlarni to'g'ri ishga tushirish uchun juda muhimdir va metodlarni qayta yozish polimorf xatti-harakatni ta'minlaydi. Bu yanada barqaror va kengaytiriladigan kod ierarxiyalarini yaratadi.`
        }
    }
};

export default task;
