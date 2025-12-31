import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-switch-patterns',
    title: 'Pattern Matching in Switch',
    difficulty: 'medium',
    tags: ['java', 'pattern-matching', 'switch', 'java-21'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master pattern matching in switch expressions and statements (Java 21+).

**Requirements:**
1. Create a Vehicle hierarchy with Car, Truck, and Motorcycle classes
2. Use switch expressions with type patterns to classify vehicles
3. Implement a pricing calculator using switch with patterns
4. Demonstrate pattern matching with null handling
5. Use switch expressions to return different values based on type
6. Compare traditional switch with pattern matching switch

Pattern matching in switch combines the power of switch expressions with type patterns, making complex conditional logic cleaner and more maintainable.`,
    initialCode: `public class SwitchPatterns {
    sealed interface Vehicle permits Car, Truck, Motorcycle {}

    record Car(String brand, int passengers) implements Vehicle {}
    record Truck(String brand, double loadCapacity) implements Vehicle {}
    record Motorcycle(String brand, int cc) implements Vehicle {}

    // Traditional approach
    public static String describeVehicleOld(Vehicle vehicle) {
        // TODO: Use traditional if-else with instanceof
        return "";
    }

    // Pattern matching in switch
    public static String describeVehicle(Vehicle vehicle) {
        // TODO: Use switch with type patterns
        return "";
    }

    // Calculate toll based on vehicle type
    public static double calculateToll(Vehicle vehicle) {
        // TODO: Use switch expression with patterns
        return 0.0;
    }

    public static void main(String[] args) {
        // Create vehicles and test methods
    }
}`,
    solutionCode: `public class SwitchPatterns {
    sealed interface Vehicle permits Car, Truck, Motorcycle {}

    record Car(String brand, int passengers) implements Vehicle {}
    record Truck(String brand, double loadCapacity) implements Vehicle {}
    record Motorcycle(String brand, int cc) implements Vehicle {}

    // Traditional approach - verbose and repetitive
    public static String describeVehicleOld(Vehicle vehicle) {
        if (vehicle instanceof Car car) {
            return "Car: " + car.brand() + " (" + car.passengers() + " passengers)";
        } else if (vehicle instanceof Truck truck) {
            return "Truck: " + truck.brand() + " (capacity: " + truck.loadCapacity() + "t)";
        } else if (vehicle instanceof Motorcycle motorcycle) {
            return "Motorcycle: " + motorcycle.brand() + " (" + motorcycle.cc() + "cc)";
        }
        return "Unknown vehicle";
    }

    // Pattern matching in switch - clean and expressive
    public static String describeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            // Type pattern with automatic destructuring
            case Car c -> "Car: " + c.brand() + " (" + c.passengers() + " passengers)";
            case Truck t -> "Truck: " + t.brand() + " (capacity: " + t.loadCapacity() + "t)";
            case Motorcycle m -> "Motorcycle: " + m.brand() + " (" + m.cc() + "cc)";
            // No default needed - sealed interface is exhaustive
        };
    }

    // Calculate toll based on vehicle type
    public static double calculateToll(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c -> c.passengers() > 4 ? 5.0 : 3.0;
            case Truck t -> 10.0 + (t.loadCapacity() * 2.0);
            case Motorcycle m -> 2.0;
        };
    }

    // Complex pattern matching with conditions
    public static String analyzeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c when c.passengers() > 5 ->
                "Large car: " + c.brand() + " (minivan or SUV)";
            case Car c ->
                "Regular car: " + c.brand();
            case Truck t when t.loadCapacity() > 10 ->
                "Heavy truck: " + t.brand() + " (commercial)";
            case Truck t ->
                "Light truck: " + t.brand();
            case Motorcycle m when m.cc() > 1000 ->
                "Sport bike: " + m.brand();
            case Motorcycle m ->
                "Standard motorcycle: " + m.brand();
        };
    }

    // Handling null in switch patterns
    public static String safeDescribe(Vehicle vehicle) {
        return switch (vehicle) {
            case null -> "No vehicle";
            case Car c -> "Car: " + c.brand();
            case Truck t -> "Truck: " + t.brand();
            case Motorcycle m -> "Motorcycle: " + m.brand();
        };
    }

    public static void main(String[] args) {
        Vehicle car = new Car("Toyota", 5);
        Vehicle truck = new Truck("Volvo", 12.5);
        Vehicle motorcycle = new Motorcycle("Harley", 1200);

        System.out.println("=== Traditional approach ===");
        System.out.println(describeVehicleOld(car));
        System.out.println(describeVehicleOld(truck));

        System.out.println("\\n=== Pattern matching switch ===");
        System.out.println(describeVehicle(car));
        System.out.println(describeVehicle(truck));
        System.out.println(describeVehicle(motorcycle));

        System.out.println("\\n=== Toll calculation ===");
        System.out.println("Car toll: $" + calculateToll(car));
        System.out.println("Truck toll: $" + calculateToll(truck));
        System.out.println("Motorcycle toll: $" + calculateToll(motorcycle));

        System.out.println("\\n=== Vehicle analysis ===");
        System.out.println(analyzeVehicle(new Car("Honda", 7)));
        System.out.println(analyzeVehicle(truck));
        System.out.println(analyzeVehicle(motorcycle));

        System.out.println("\\n=== Null handling ===");
        System.out.println(safeDescribe(null));
        System.out.println(safeDescribe(car));
    }
}`,
    hint1: `Switch with patterns: switch(obj) { case Type t -> ... }. Each case can have a type pattern that extracts the value.`,
    hint2: `Use 'when' clauses for conditions: case Car c when c.passengers() > 5 -> ... This is called a guarded pattern.`,
    whyItMatters: `Pattern matching in switch makes complex type-based logic much cleaner and safer. It's exhaustive checking with sealed types prevents bugs, and it's more maintainable than chains of if-else statements.

**Production Pattern:**
\`\`\`java
public ResponseEntity<?> handleRequest(ApiRequest request) {
    return switch (request) {
        case GetRequest get -> fetchData(get.getResourceId());
        case PostRequest post when post.isValid() -> createResource(post.getData());
        case DeleteRequest delete -> removeResource(delete.getId());
        case null -> ResponseEntity.badRequest().build();
    };
}
\`\`\`

**Practical Benefits:**
- Compile-time guarantee for handling all request types
- 50% reduction in request handling code`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify Car creation and basic fields
class Test1 {
    @Test
    public void test() {
        SwitchPatterns.Car car = new SwitchPatterns.Car("Toyota", 5);
        assertEquals("Toyota", car.brand());
        assertEquals(5, car.passengers());
    }
}

// Test2: Verify Truck creation and basic fields
class Test2 {
    @Test
    public void test() {
        SwitchPatterns.Truck truck = new SwitchPatterns.Truck("Volvo", 12.5);
        assertEquals("Volvo", truck.brand());
        assertEquals(12.5, truck.loadCapacity(), 0.0001);
    }
}

// Test3: Verify Motorcycle creation and basic fields
class Test3 {
    @Test
    public void test() {
        SwitchPatterns.Motorcycle motorcycle = new SwitchPatterns.Motorcycle("Harley", 1200);
        assertEquals("Harley", motorcycle.brand());
        assertEquals(1200, motorcycle.cc());
    }
}

// Test4: Test describeVehicleOld with all vehicle types
class Test4 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle car = new SwitchPatterns.Car("Honda", 4);
        String result = SwitchPatterns.describeVehicleOld(car);
        assertTrue(result.contains("Car") && result.contains("Honda") && result.contains("4"));
    }
}

// Test5: Test describeVehicle with pattern matching for Car
class Test5 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle car = new SwitchPatterns.Car("Toyota", 5);
        String result = SwitchPatterns.describeVehicle(car);
        assertTrue(result.contains("Car") && result.contains("Toyota") && result.contains("5"));
    }
}

// Test6: Test describeVehicle with pattern matching for Truck
class Test6 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle truck = new SwitchPatterns.Truck("Volvo", 12.5);
        String result = SwitchPatterns.describeVehicle(truck);
        assertTrue(result.contains("Truck") && result.contains("Volvo"));
    }
}

// Test7: Test calculateToll for different vehicle types
class Test7 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle car = new SwitchPatterns.Car("Toyota", 3);
        SwitchPatterns.Vehicle largeCar = new SwitchPatterns.Car("Honda", 6);
        SwitchPatterns.Vehicle motorcycle = new SwitchPatterns.Motorcycle("Yamaha", 600);

        assertEquals(3.0, SwitchPatterns.calculateToll(car), 0.0001);
        assertEquals(5.0, SwitchPatterns.calculateToll(largeCar), 0.0001);
        assertEquals(2.0, SwitchPatterns.calculateToll(motorcycle), 0.0001);
    }
}

// Test8: Test calculateToll for Truck based on load capacity
class Test8 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle truck = new SwitchPatterns.Truck("Ford", 5.0);
        double expectedToll = 10.0 + (5.0 * 2.0);
        assertEquals(expectedToll, SwitchPatterns.calculateToll(truck), 0.0001);
    }
}

// Test9: Test analyzeVehicle with guarded patterns
class Test9 {
    @Test
    public void test() {
        SwitchPatterns.Vehicle largeCar = new SwitchPatterns.Car("Honda", 7);
        SwitchPatterns.Vehicle heavyTruck = new SwitchPatterns.Truck("Scania", 15.0);
        SwitchPatterns.Vehicle sportBike = new SwitchPatterns.Motorcycle("Ducati", 1500);

        String carResult = SwitchPatterns.analyzeVehicle(largeCar);
        assertTrue(carResult.contains("Large car") || carResult.contains("minivan") || carResult.contains("SUV"));

        String truckResult = SwitchPatterns.analyzeVehicle(heavyTruck);
        assertTrue(truckResult.contains("Heavy truck") || truckResult.contains("commercial"));

        String bikeResult = SwitchPatterns.analyzeVehicle(sportBike);
        assertTrue(bikeResult.contains("Sport bike"));
    }
}

// Test10: Test safeDescribe with null handling
class Test10 {
    @Test
    public void test() {
        String nullResult = SwitchPatterns.safeDescribe(null);
        assertEquals("No vehicle", nullResult);

        SwitchPatterns.Vehicle car = new SwitchPatterns.Car("BMW", 4);
        String carResult = SwitchPatterns.safeDescribe(car);
        assertTrue(carResult.contains("Car") && carResult.contains("BMW"));
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Сопоставление с образцом в Switch',
            solutionCode: `public class SwitchPatterns {
    sealed interface Vehicle permits Car, Truck, Motorcycle {}

    record Car(String brand, int passengers) implements Vehicle {}
    record Truck(String brand, double loadCapacity) implements Vehicle {}
    record Motorcycle(String brand, int cc) implements Vehicle {}

    // Традиционный подход - многословный и повторяющийся
    public static String describeVehicleOld(Vehicle vehicle) {
        if (vehicle instanceof Car car) {
            return "Автомобиль: " + car.brand() + " (" + car.passengers() + " пассажиров)";
        } else if (vehicle instanceof Truck truck) {
            return "Грузовик: " + truck.brand() + " (грузоподъемность: " + truck.loadCapacity() + "т)";
        } else if (vehicle instanceof Motorcycle motorcycle) {
            return "Мотоцикл: " + motorcycle.brand() + " (" + motorcycle.cc() + "cc)";
        }
        return "Неизвестное транспортное средство";
    }

    // Сопоставление с образцом в switch - чистый и выразительный
    public static String describeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            // Паттерн типа с автоматической деструктуризацией
            case Car c -> "Автомобиль: " + c.brand() + " (" + c.passengers() + " пассажиров)";
            case Truck t -> "Грузовик: " + t.brand() + " (грузоподъемность: " + t.loadCapacity() + "т)";
            case Motorcycle m -> "Мотоцикл: " + m.brand() + " (" + m.cc() + "cc)";
            // default не нужен - sealed интерфейс исчерпывающий
        };
    }

    // Расчет дорожного сбора на основе типа транспортного средства
    public static double calculateToll(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c -> c.passengers() > 4 ? 5.0 : 3.0;
            case Truck t -> 10.0 + (t.loadCapacity() * 2.0);
            case Motorcycle m -> 2.0;
        };
    }

    // Сложное сопоставление с образцом с условиями
    public static String analyzeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c when c.passengers() > 5 ->
                "Большой автомобиль: " + c.brand() + " (минивэн или внедорожник)";
            case Car c ->
                "Обычный автомобиль: " + c.brand();
            case Truck t when t.loadCapacity() > 10 ->
                "Тяжелый грузовик: " + t.brand() + " (коммерческий)";
            case Truck t ->
                "Легкий грузовик: " + t.brand();
            case Motorcycle m when m.cc() > 1000 ->
                "Спортивный мотоцикл: " + m.brand();
            case Motorcycle m ->
                "Стандартный мотоцикл: " + m.brand();
        };
    }

    // Обработка null в паттернах switch
    public static String safeDescribe(Vehicle vehicle) {
        return switch (vehicle) {
            case null -> "Нет транспортного средства";
            case Car c -> "Автомобиль: " + c.brand();
            case Truck t -> "Грузовик: " + t.brand();
            case Motorcycle m -> "Мотоцикл: " + m.brand();
        };
    }

    public static void main(String[] args) {
        Vehicle car = new Car("Toyota", 5);
        Vehicle truck = new Truck("Volvo", 12.5);
        Vehicle motorcycle = new Motorcycle("Harley", 1200);

        System.out.println("=== Традиционный подход ===");
        System.out.println(describeVehicleOld(car));
        System.out.println(describeVehicleOld(truck));

        System.out.println("\\n=== Сопоставление с образцом в switch ===");
        System.out.println(describeVehicle(car));
        System.out.println(describeVehicle(truck));
        System.out.println(describeVehicle(motorcycle));

        System.out.println("\\n=== Расчет дорожного сбора ===");
        System.out.println("Сбор за автомобиль: $" + calculateToll(car));
        System.out.println("Сбор за грузовик: $" + calculateToll(truck));
        System.out.println("Сбор за мотоцикл: $" + calculateToll(motorcycle));

        System.out.println("\\n=== Анализ транспортного средства ===");
        System.out.println(analyzeVehicle(new Car("Honda", 7)));
        System.out.println(analyzeVehicle(truck));
        System.out.println(analyzeVehicle(motorcycle));

        System.out.println("\\n=== Обработка null ===");
        System.out.println(safeDescribe(null));
        System.out.println(safeDescribe(car));
    }
}`,
            description: `Освойте сопоставление с образцом в выражениях и операторах switch (Java 21+).

**Требования:**
1. Создайте иерархию Vehicle с классами Car, Truck и Motorcycle
2. Используйте выражения switch с паттернами типов для классификации транспортных средств
3. Реализуйте калькулятор цен, используя switch с паттернами
4. Продемонстрируйте сопоставление с образцом с обработкой null
5. Используйте выражения switch для возврата разных значений на основе типа
6. Сравните традиционный switch с switch сопоставлением с образцом

Сопоставление с образцом в switch объединяет мощь выражений switch с паттернами типов, делая сложную условную логику чище и более поддерживаемой.`,
            hint1: `Switch с паттернами: switch(obj) { case Type t -> ... }. Каждый case может иметь паттерн типа, который извлекает значение.`,
            hint2: `Используйте предложения 'when' для условий: case Car c when c.passengers() > 5 -> ... Это называется охраняемым паттерном.`,
            whyItMatters: `Сопоставление с образцом в switch делает сложную логику на основе типов намного чище и безопаснее. Исчерпывающая проверка с sealed типами предотвращает ошибки, и это более поддерживаемо, чем цепочки if-else.

**Продакшен паттерн:**
\`\`\`java
public ResponseEntity<?> handleRequest(ApiRequest request) {
    return switch (request) {
        case GetRequest get -> fetchData(get.getResourceId());
        case PostRequest post when post.isValid() -> createResource(post.getData());
        case DeleteRequest delete -> removeResource(delete.getId());
        case null -> ResponseEntity.badRequest().build();
    };
}
\`\`\`

**Практические преимущества:**
- Гарантия обработки всех типов запросов на уровне компиляции
- Сокращение кода обработки запросов на 50%`
        },
        uz: {
            title: 'Switch da namuna moslash',
            solutionCode: `public class SwitchPatterns {
    sealed interface Vehicle permits Car, Truck, Motorcycle {}

    record Car(String brand, int passengers) implements Vehicle {}
    record Truck(String brand, double loadCapacity) implements Vehicle {}
    record Motorcycle(String brand, int cc) implements Vehicle {}

    // An'anaviy yondashuv - ko'p so'zli va takrorlanuvchi
    public static String describeVehicleOld(Vehicle vehicle) {
        if (vehicle instanceof Car car) {
            return "Avtomobil: " + car.brand() + " (" + car.passengers() + " yo'lovchi)";
        } else if (vehicle instanceof Truck truck) {
            return "Yuk mashinasi: " + truck.brand() + " (yuklik: " + truck.loadCapacity() + "t)";
        } else if (vehicle instanceof Motorcycle motorcycle) {
            return "Mototsikl: " + motorcycle.brand() + " (" + motorcycle.cc() + "cc)";
        }
        return "Noma'lum transport vositasi";
    }

    // Switch da namuna moslash - toza va ifodali
    public static String describeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            // Avtomatik dekonstruksiya bilan tur namunasi
            case Car c -> "Avtomobil: " + c.brand() + " (" + c.passengers() + " yo'lovchi)";
            case Truck t -> "Yuk mashinasi: " + t.brand() + " (yuklik: " + t.loadCapacity() + "t)";
            case Motorcycle m -> "Mototsikl: " + m.brand() + " (" + m.cc() + "cc)";
            // default kerak emas - sealed interfeys to'liq
        };
    }

    // Transport vositasi turiga qarab yo'l to'lovini hisoblash
    public static double calculateToll(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c -> c.passengers() > 4 ? 5.0 : 3.0;
            case Truck t -> 10.0 + (t.loadCapacity() * 2.0);
            case Motorcycle m -> 2.0;
        };
    }

    // Shartlar bilan murakkab namuna moslash
    public static String analyzeVehicle(Vehicle vehicle) {
        return switch (vehicle) {
            case Car c when c.passengers() > 5 ->
                "Katta avtomobil: " + c.brand() + " (miniven yoki SUV)";
            case Car c ->
                "Oddiy avtomobil: " + c.brand();
            case Truck t when t.loadCapacity() > 10 ->
                "Og'ir yuk mashinasi: " + t.brand() + " (tijorat)";
            case Truck t ->
                "Yengil yuk mashinasi: " + t.brand();
            case Motorcycle m when m.cc() > 1000 ->
                "Sport mototsikl: " + m.brand();
            case Motorcycle m ->
                "Standart mototsikl: " + m.brand();
        };
    }

    // Switch namunalarida null ni boshqarish
    public static String safeDescribe(Vehicle vehicle) {
        return switch (vehicle) {
            case null -> "Transport vositasi yo'q";
            case Car c -> "Avtomobil: " + c.brand();
            case Truck t -> "Yuk mashinasi: " + t.brand();
            case Motorcycle m -> "Mototsikl: " + m.brand();
        };
    }

    public static void main(String[] args) {
        Vehicle car = new Car("Toyota", 5);
        Vehicle truck = new Truck("Volvo", 12.5);
        Vehicle motorcycle = new Motorcycle("Harley", 1200);

        System.out.println("=== An'anaviy yondashuv ===");
        System.out.println(describeVehicleOld(car));
        System.out.println(describeVehicleOld(truck));

        System.out.println("\\n=== Switch da namuna moslash ===");
        System.out.println(describeVehicle(car));
        System.out.println(describeVehicle(truck));
        System.out.println(describeVehicle(motorcycle));

        System.out.println("\\n=== Yo'l to'lovi hisoblash ===");
        System.out.println("Avtomobil to'lovi: $" + calculateToll(car));
        System.out.println("Yuk mashinasi to'lovi: $" + calculateToll(truck));
        System.out.println("Mototsikl to'lovi: $" + calculateToll(motorcycle));

        System.out.println("\\n=== Transport vositasi tahlili ===");
        System.out.println(analyzeVehicle(new Car("Honda", 7)));
        System.out.println(analyzeVehicle(truck));
        System.out.println(analyzeVehicle(motorcycle));

        System.out.println("\\n=== Null ni boshqarish ===");
        System.out.println(safeDescribe(null));
        System.out.println(safeDescribe(car));
    }
}`,
            description: `Switch ifodalari va operatorlarida namuna moslashni o'zlashtiiring (Java 21+).

**Talablar:**
1. Car, Truck va Motorcycle klasslari bilan Vehicle ierarxiyasini yarating
2. Transport vositalarini tasniflash uchun tur namunalari bilan switch ifodalardidan foydalaning
3. Switch dan namunalar bilan foydalanib, narx kalkulyatorini amalga oshiring
4. null ni boshqarish bilan namuna moslashni ko'rsating
5. Turga qarab turli qiymatlarni qaytarish uchun switch ifodalardidan foydalaning
6. An'anaviy switch ni namuna moslash switch bilan taqqoslang

Switch da namuna moslash switch ifodalari kuchini tur namunalari bilan birlashtiradi, bu murakkab shartli mantiqni toza va yanada boshqariladigan qiladi.`,
            hint1: `Namunalar bilan switch: switch(obj) { case Type t -> ... }. Har bir case qiymatni oluvchi tur namunasiga ega bo'lishi mumkin.`,
            hint2: `Shartlar uchun 'when' bandlaridan foydalaning: case Car c when c.passengers() > 5 -> ... Bu muhofazalangan namuna deb ataladi.`,
            whyItMatters: `Switch da namuna moslash turlarga asoslangan murakkab mantiqni yanada tozaroq va xavfsizroq qiladi. Sealed turlar bilan to'liq tekshirish xatolarni oldini oladi va bu if-else zanjirlariga qaraganda yanada boshqariladigan.

**Ishlab chiqarish patterni:**
\`\`\`java
public ResponseEntity<?> handleRequest(ApiRequest request) {
    return switch (request) {
        case GetRequest get -> fetchData(get.getResourceId());
        case PostRequest post when post.isValid() -> createResource(post.getData());
        case DeleteRequest delete -> removeResource(delete.getId());
        case null -> ResponseEntity.badRequest().build();
    };
}
\`\`\`

**Amaliy foydalari:**
- Kompilyatsiya darajasida barcha so'rov turlarini qayta ishlashni kafolatlaydi
- So'rovlarni qayta ishlash kodini 50% qisqartiradi`
        }
    }
};

export default task;
