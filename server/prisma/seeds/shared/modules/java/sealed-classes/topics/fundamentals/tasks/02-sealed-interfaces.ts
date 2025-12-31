import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-sealed-interfaces',
    title: 'Sealed Interfaces',
    difficulty: 'medium',
    tags: ['java', 'sealed', 'interfaces', 'permits', 'implementation'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to use sealed interfaces for controlled implementations.

**Requirements:**
1. Create a sealed interface Vehicle that permits Car, Motorcycle, and Truck
2. Create a final record Car that implements Vehicle with brand and model fields
3. Create a final record Motorcycle that implements Vehicle with brand and engineSize fields
4. Create a non-sealed class Truck that implements Vehicle with brand and capacity fields
5. Add methods to the interface: String getBrand(), String getType()
6. Create a method that takes a Vehicle and prints information using pattern matching (instanceof)

Sealed interfaces restrict which classes can implement them, providing the same benefits as sealed classes for interface hierarchies.`,
    initialCode: `// Create sealed Vehicle interface that permits Car, Motorcycle, Truck
// Add String getBrand() method
// Add String getType() method

// Create final record Car
// - implements Vehicle
// - String brand, String model
// - implement interface methods

// Create final record Motorcycle
// - implements Vehicle
// - String brand, int engineSize
// - implement interface methods

// Create non-sealed class Truck
// - implements Vehicle
// - String brand, double capacity
// - constructor
// - implement interface methods

public class SealedInterfaces {
    // Create static method printVehicleInfo(Vehicle vehicle)
    // Use instanceof pattern matching to print specific info

    public static void main(String[] args) {
        // Create instances of all vehicle types

        // Print info for each vehicle
    }
}`,
    solutionCode: `// Sealed interface restricts which types can implement it
sealed interface Vehicle permits Car, Motorcycle, Truck {
    String getBrand();
    String getType();
}

// Final record implementing sealed interface
final record Car(String brand, String model) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Car";
    }
}

// Final record implementing sealed interface
final record Motorcycle(String brand, int engineSize) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Motorcycle";
    }
}

// Non-sealed class implementing sealed interface
non-sealed class Truck implements Vehicle {
    private final String brand;
    private final double capacity;

    public Truck(String brand, double capacity) {
        this.brand = brand;
        this.capacity = capacity;
    }

    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Truck";
    }

    public double getCapacity() {
        return capacity;
    }
}

public class SealedInterfaces {
    // Pattern matching with instanceof for sealed types
    static void printVehicleInfo(Vehicle vehicle) {
        System.out.println("Vehicle Type: " + vehicle.getType());
        System.out.println("Brand: " + vehicle.getBrand());

        // Pattern matching allows safe casting
        if (vehicle instanceof Car car) {
            System.out.println("Model: " + car.model());
        } else if (vehicle instanceof Motorcycle motorcycle) {
            System.out.println("Engine Size: " + motorcycle.engineSize() + "cc");
        } else if (vehicle instanceof Truck truck) {
            System.out.println("Capacity: " + truck.getCapacity() + " tons");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        // Create instances of permitted implementations
        Vehicle car = new Car("Toyota", "Camry");
        Vehicle motorcycle = new Motorcycle("Harley-Davidson", 1200);
        Vehicle truck = new Truck("Volvo", 18.5);

        // Print information for each vehicle
        printVehicleInfo(car);
        printVehicleInfo(motorcycle);
        printVehicleInfo(truck);

        // This would cause compile-time error:
        // class Bicycle implements Vehicle { } // Error: not permitted
    }
}`,
    hint1: `Declare a sealed interface the same way as a sealed class: sealed interface Vehicle permits Car, Motorcycle, Truck { ... }`,
    hint2: `Records can implement interfaces and are perfect for sealed interfaces because they're implicitly final and immutable.`,
    whyItMatters: `Sealed interfaces are crucial for defining closed sets of implementations. They're perfect for algebraic data types, state machines, and domain models where you need exhaustive handling of all possible implementations.

**Production Pattern:**
\`\`\`java
sealed interface ApiResponse<T> permits SuccessResponse, ErrorResponse {
    int getStatusCode();
}

record SuccessResponse<T>(T data, int statusCode) implements ApiResponse<T> {
    public SuccessResponse(T data) { this(data, 200); }
}

record ErrorResponse(String message, int statusCode) implements ApiResponse {
    public ErrorResponse(String message) { this(message, 500); }
}
\`\`\`

**Practical Benefits:**
- Type-safe API response handling
- Eliminates forgotten error handling cases`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Define sealed interface and permitted implementations
sealed interface Vehicle permits Car, Motorcycle, Truck {
    String getBrand();
    int getWheels();
}

final class Car implements Vehicle {
    private final String brand;
    Car(String brand) { this.brand = brand; }
    public String getBrand() { return brand; }
    public int getWheels() { return 4; }
}

final class Motorcycle implements Vehicle {
    private final String brand;
    Motorcycle(String brand) { this.brand = brand; }
    public String getBrand() { return brand; }
    public int getWheels() { return 2; }
}

final class Truck implements Vehicle {
    private final String brand;
    Truck(String brand) { this.brand = brand; }
    public String getBrand() { return brand; }
    public int getWheels() { return 6; }
}

// Test1: Test Car implementation
class Test1 {
    @Test
    public void test() {
        Vehicle car = new Car("Toyota");
        assertEquals("Toyota", car.getBrand());
        assertEquals(4, car.getWheels());
    }
}

// Test2: Test Motorcycle implementation
class Test2 {
    @Test
    public void test() {
        Vehicle motorcycle = new Motorcycle("Harley");
        assertEquals("Harley", motorcycle.getBrand());
        assertEquals(2, motorcycle.getWheels());
    }
}

// Test3: Test Truck implementation
class Test3 {
    @Test
    public void test() {
        Vehicle truck = new Truck("Volvo");
        assertEquals("Volvo", truck.getBrand());
        assertEquals(6, truck.getWheels());
    }
}

// Test4: Test instanceof with Car
class Test4 {
    @Test
    public void test() {
        Vehicle vehicle = new Car("Honda");
        assertTrue(vehicle instanceof Car);
        assertFalse(vehicle instanceof Motorcycle);
    }
}

// Test5: Test instanceof with Motorcycle
class Test5 {
    @Test
    public void test() {
        Vehicle vehicle = new Motorcycle("Yamaha");
        assertTrue(vehicle instanceof Motorcycle);
        assertFalse(vehicle instanceof Truck);
    }
}

// Test6: Test instanceof with Truck
class Test6 {
    @Test
    public void test() {
        Vehicle vehicle = new Truck("Mercedes");
        assertTrue(vehicle instanceof Truck);
        assertFalse(vehicle instanceof Car);
    }
}

// Test7: Test different car brands
class Test7 {
    @Test
    public void test() {
        Vehicle car1 = new Car("BMW");
        Vehicle car2 = new Car("Audi");
        assertEquals("BMW", car1.getBrand());
        assertEquals("Audi", car2.getBrand());
    }
}

// Test8: Test wheel counts
class Test8 {
    @Test
    public void test() {
        assertEquals(4, new Car("Ford").getWheels());
        assertEquals(2, new Motorcycle("Suzuki").getWheels());
        assertEquals(6, new Truck("Scania").getWheels());
    }
}

// Test9: Test polymorphism
class Test9 {
    @Test
    public void test() {
        Vehicle[] vehicles = {
            new Car("Tesla"),
            new Motorcycle("Ducati"),
            new Truck("MAN")
        };
        assertEquals(3, vehicles.length);
        assertTrue(vehicles[0] instanceof Vehicle);
        assertTrue(vehicles[1] instanceof Vehicle);
        assertTrue(vehicles[2] instanceof Vehicle);
    }
}

// Test10: Test brand equality
class Test10 {
    @Test
    public void test() {
        Vehicle car = new Car("Mazda");
        Vehicle motorcycle = new Motorcycle("Mazda");
        assertEquals(car.getBrand(), motorcycle.getBrand());
        assertNotEquals(car.getWheels(), motorcycle.getWheels());
    }
}
`,
    translations: {
        ru: {
            title: 'Запечатанные интерфейсы',
            solutionCode: `// Запечатанный интерфейс ограничивает, какие типы могут его реализовать
sealed interface Vehicle permits Car, Motorcycle, Truck {
    String getBrand();
    String getType();
}

// Финальная запись, реализующая запечатанный интерфейс
final record Car(String brand, String model) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Car";
    }
}

// Финальная запись, реализующая запечатанный интерфейс
final record Motorcycle(String brand, int engineSize) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Motorcycle";
    }
}

// Незапечатанный класс, реализующий запечатанный интерфейс
non-sealed class Truck implements Vehicle {
    private final String brand;
    private final double capacity;

    public Truck(String brand, double capacity) {
        this.brand = brand;
        this.capacity = capacity;
    }

    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Truck";
    }

    public double getCapacity() {
        return capacity;
    }
}

public class SealedInterfaces {
    // Сопоставление с образцом с instanceof для запечатанных типов
    static void printVehicleInfo(Vehicle vehicle) {
        System.out.println("Тип транспорта: " + vehicle.getType());
        System.out.println("Бренд: " + vehicle.getBrand());

        // Сопоставление с образцом позволяет безопасное приведение типов
        if (vehicle instanceof Car car) {
            System.out.println("Модель: " + car.model());
        } else if (vehicle instanceof Motorcycle motorcycle) {
            System.out.println("Объем двигателя: " + motorcycle.engineSize() + "cc");
        } else if (vehicle instanceof Truck truck) {
            System.out.println("Грузоподъемность: " + truck.getCapacity() + " тонн");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        // Создаем экземпляры разрешенных реализаций
        Vehicle car = new Car("Toyota", "Camry");
        Vehicle motorcycle = new Motorcycle("Harley-Davidson", 1200);
        Vehicle truck = new Truck("Volvo", 18.5);

        // Выводим информацию о каждом транспортном средстве
        printVehicleInfo(car);
        printVehicleInfo(motorcycle);
        printVehicleInfo(truck);

        // Это вызовет ошибку компиляции:
        // class Bicycle implements Vehicle { } // Ошибка: не разрешено
    }
}`,
            description: `Изучите использование запечатанных интерфейсов для контролируемых реализаций.

**Требования:**
1. Создайте запечатанный интерфейс Vehicle, который разрешает Car, Motorcycle и Truck
2. Создайте финальную запись Car, реализующую Vehicle с полями brand и model
3. Создайте финальную запись Motorcycle, реализующую Vehicle с полями brand и engineSize
4. Создайте незапечатанный класс Truck, реализующий Vehicle с полями brand и capacity
5. Добавьте методы в интерфейс: String getBrand(), String getType()
6. Создайте метод, принимающий Vehicle и выводящий информацию с использованием сопоставления с образцом (instanceof)

Запечатанные интерфейсы ограничивают, какие классы могут их реализовывать, предоставляя те же преимущества, что и запечатанные классы для иерархий интерфейсов.`,
            hint1: `Объявите запечатанный интерфейс так же, как запечатанный класс: sealed interface Vehicle permits Car, Motorcycle, Truck { ... }`,
            hint2: `Записи могут реализовывать интерфейсы и идеально подходят для запечатанных интерфейсов, потому что они неявно финальные и неизменяемые.`,
            whyItMatters: `Запечатанные интерфейсы критически важны для определения закрытых наборов реализаций. Они идеальны для алгебраических типов данных, конечных автоматов и доменных моделей, где требуется исчерпывающая обработка всех возможных реализаций.

**Продакшен паттерн:**
\`\`\`java
sealed interface ApiResponse<T> permits SuccessResponse, ErrorResponse {
    int getStatusCode();
}

record SuccessResponse<T>(T data, int statusCode) implements ApiResponse<T> {
    public SuccessResponse(T data) { this(data, 200); }
}

record ErrorResponse(String message, int statusCode) implements ApiResponse {
    public ErrorResponse(String message) { this(message, 500); }
}
\`\`\`

**Практические преимущества:**
- Типобезопасная обработка API ответов
- Исключение забытых случаев обработки ошибок`
        },
        uz: {
            title: 'Muhrlangan interfeyslar',
            solutionCode: `// Muhrlangan interfeys qaysi turlar uni amalga oshirishi mumkinligini cheklaydi
sealed interface Vehicle permits Car, Motorcycle, Truck {
    String getBrand();
    String getType();
}

// Muhrlangan interfeysni amalga oshiruvchi final record
final record Car(String brand, String model) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Car";
    }
}

// Muhrlangan interfeysni amalga oshiruvchi final record
final record Motorcycle(String brand, int engineSize) implements Vehicle {
    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Motorcycle";
    }
}

// Muhrlangan interfeysni amalga oshiruvchi non-sealed klass
non-sealed class Truck implements Vehicle {
    private final String brand;
    private final double capacity;

    public Truck(String brand, double capacity) {
        this.brand = brand;
        this.capacity = capacity;
    }

    @Override
    public String getBrand() {
        return brand;
    }

    @Override
    public String getType() {
        return "Truck";
    }

    public double getCapacity() {
        return capacity;
    }
}

public class SealedInterfaces {
    // Muhrlangan turlar uchun instanceof bilan pattern matching
    static void printVehicleInfo(Vehicle vehicle) {
        System.out.println("Transport turi: " + vehicle.getType());
        System.out.println("Brend: " + vehicle.getBrand());

        // Pattern matching xavfsiz castingni ta'minlaydi
        if (vehicle instanceof Car car) {
            System.out.println("Model: " + car.model());
        } else if (vehicle instanceof Motorcycle motorcycle) {
            System.out.println("Dvigatel hajmi: " + motorcycle.engineSize() + "cc");
        } else if (vehicle instanceof Truck truck) {
            System.out.println("Sig'im: " + truck.getCapacity() + " tonna");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        // Ruxsat etilgan amalga oshirishlarning misollarini yaratamiz
        Vehicle car = new Car("Toyota", "Camry");
        Vehicle motorcycle = new Motorcycle("Harley-Davidson", 1200);
        Vehicle truck = new Truck("Volvo", 18.5);

        // Har bir transport vositasi haqida ma'lumot chiqaramiz
        printVehicleInfo(car);
        printVehicleInfo(motorcycle);
        printVehicleInfo(truck);

        // Bu kompilyatsiya xatosiga olib keladi:
        // class Bicycle implements Vehicle { } // Xato: ruxsat etilmagan
    }
}`,
            description: `Nazorat ostidagi amalga oshirishlar uchun muhrlangan interfeyslardan foydalanishni o'rganing.

**Talablar:**
1. Car, Motorcycle va Truck ga ruxsat beruvchi muhrlangan Vehicle interfeysini yarating
2. Vehicle ni amalga oshiruvchi brand va model maydonli final Car recordini yarating
3. Vehicle ni amalga oshiruvchi brand va engineSize maydonli final Motorcycle recordini yarating
4. Vehicle ni amalga oshiruvchi brand va capacity maydonli non-sealed Truck klassini yarating
5. Interfeysga metodlar qo'shing: String getBrand(), String getType()
6. Vehicle qabul qilib, pattern matching (instanceof) yordamida ma'lumot chiqaruvchi metod yarating

Muhrlangan interfeyslar qaysi klasslar ularni amalga oshirishi mumkinligini cheklaydi va interfeys iyerarxiyalari uchun muhrlangan klasslar bilan bir xil afzalliklarni taqdim etadi.`,
            hint1: `Muhrlangan interfeysni muhrlangan klass kabi e'lon qiling: sealed interface Vehicle permits Car, Motorcycle, Truck { ... }`,
            hint2: `Recordlar interfeyslarni amalga oshirishi mumkin va muhrlangan interfeyslar uchun juda mos keladi, chunki ular yashirin tarzda final va o'zgarmasdir.`,
            whyItMatters: `Muhrlangan interfeyslar yopiq amalga oshirishlar to'plamini aniqlash uchun juda muhimdir. Ular algebraik ma'lumotlar turlari, holat mashinalari va barcha mumkin bo'lgan amalga oshirishlarni to'liq qayta ishlash talab qilinadigan domen modellari uchun idealdir.

**Ishlab chiqarish patterni:**
\`\`\`java
sealed interface ApiResponse<T> permits SuccessResponse, ErrorResponse {
    int getStatusCode();
}

record SuccessResponse<T>(T data, int statusCode) implements ApiResponse<T> {
    public SuccessResponse(T data) { this(data, 200); }
}

record ErrorResponse(String message, int statusCode) implements ApiResponse {
    public ErrorResponse(String message) { this(message, 500); }
}
\`\`\`

**Amaliy foydalari:**
- API javoblarini tip-xavfsiz qayta ishlash
- Unutilgan xatoliklarni qayta ishlash holatlarini istisno qilish`
        }
    }
};

export default task;
