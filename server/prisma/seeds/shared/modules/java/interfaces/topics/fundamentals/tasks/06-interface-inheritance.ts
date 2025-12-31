import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-interface-inheritance',
    title: 'Interface Inheritance and Composition',
    difficulty: 'medium',
    tags: ['java', 'interfaces', 'inheritance', 'composition', 'design-patterns'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Interface Inheritance and Composition

Interfaces can extend other interfaces, creating hierarchies. This enables building complex contracts from simpler ones. Java also supports marker interfaces (empty interfaces used for tagging).

## Requirements:
1. Create a base \`Vehicle\` interface:
   1.1. \`void start()\`
   1.2. \`void stop()\`
   1.3. \`String getType()\`

2. Create \`Electric\` interface extending \`Vehicle\`:
   2.1. \`int getBatteryLevel()\`
   2.2. \`void charge(int percentage)\`

3. Create \`Autonomous\` interface extending \`Vehicle\`:
   3.1. \`void enableAutoPilot()\`
   3.2. \`void disableAutoPilot()\`
   3.3. \`boolean isAutoPilotActive()\`

4. Create \`Premium\` marker interface (empty - for premium features)

5. Create \`TeslaModelS\` class:
   5.1. Implements \`Electric\`, \`Autonomous\`, and \`Premium\`
   5.2. Provides full implementation of all methods
   5.3. Tracks state (battery level, autopilot status)

6. Demonstrate polymorphism using different interface references

## Example Output:
\`\`\`
=== Tesla Model S ===
Starting Tesla Model S
Battery Level: 85%
Charging to 100%
Battery Level: 100%
Enabling AutoPilot
AutoPilot Active: true
Is Premium? true
Stopping Tesla Model S

=== Polymorphism Demonstration ===
As Vehicle: Tesla Model S
As Electric: Battery at 100%
As Autonomous: AutoPilot is ON
\`\`\``,
    initialCode: `// TODO: Create Vehicle interface

// TODO: Create Electric interface extending Vehicle

// TODO: Create Autonomous interface extending Vehicle

// TODO: Create Premium marker interface

// TODO: Create TeslaModelS implementing multiple interfaces

public class InterfaceInheritance {
    public static void main(String[] args) {
        // TODO: Create TeslaModelS instance

        // TODO: Demonstrate using all interface methods

        // TODO: Demonstrate polymorphism with different interface types
    }
}`,
    solutionCode: `// Base interface for all vehicles
interface Vehicle {
    void start();
    void stop();
    String getType();
}

// Electric interface extends Vehicle, adding electric-specific methods
interface Electric extends Vehicle {
    int getBatteryLevel();
    void charge(int percentage);
}

// Autonomous interface extends Vehicle, adding autonomous features
interface Autonomous extends Vehicle {
    void enableAutoPilot();
    void disableAutoPilot();
    boolean isAutoPilotActive();
}

// Marker interface - empty interface for tagging premium features
interface Premium {
    // No methods - just a marker
}

// TeslaModelS implements multiple interface hierarchies
class TeslaModelS implements Electric, Autonomous, Premium {
    private int batteryLevel;
    private boolean autoPilotActive;

    public TeslaModelS(int initialBattery) {
        this.batteryLevel = initialBattery;
        this.autoPilotActive = false;
    }

    // Implementing Vehicle methods (inherited through Electric and Autonomous)
    @Override
    public void start() {
        System.out.println("Starting Tesla Model S");
    }

    @Override
    public void stop() {
        System.out.println("Stopping Tesla Model S");
    }

    @Override
    public String getType() {
        return "Tesla Model S";
    }

    // Implementing Electric methods
    @Override
    public int getBatteryLevel() {
        return batteryLevel;
    }

    @Override
    public void charge(int percentage) {
        this.batteryLevel = Math.min(100, batteryLevel + percentage);
        System.out.println("Charging to " + this.batteryLevel + "%");
    }

    // Implementing Autonomous methods
    @Override
    public void enableAutoPilot() {
        this.autoPilotActive = true;
        System.out.println("Enabling AutoPilot");
    }

    @Override
    public void disableAutoPilot() {
        this.autoPilotActive = false;
        System.out.println("Disabling AutoPilot");
    }

    @Override
    public boolean isAutoPilotActive() {
        return autoPilotActive;
    }
}

public class InterfaceInheritance {
    public static void main(String[] args) {
        System.out.println("=== Tesla Model S ===");
        TeslaModelS tesla = new TeslaModelS(85);

        // Using methods from all interfaces
        tesla.start();
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.charge(15);
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.enableAutoPilot();
        System.out.println("AutoPilot Active: " + tesla.isAutoPilotActive());

        // Check if premium (marker interface)
        System.out.println("Is Premium? " + (tesla instanceof Premium));
        tesla.stop();

        System.out.println("\\n=== Polymorphism Demonstration ===");

        // Reference as different interface types
        Vehicle asVehicle = tesla;
        System.out.println("As Vehicle: " + asVehicle.getType());

        Electric asElectric = tesla;
        System.out.println("As Electric: Battery at " + asElectric.getBatteryLevel() + "%");

        Autonomous asAutonomous = tesla;
        String autopilotStatus = asAutonomous.isAutoPilotActive() ? "ON" : "OFF";
        System.out.println("As Autonomous: AutoPilot is " + autopilotStatus);
    }
}`,
    hint1: `An interface can extend another interface using the 'extends' keyword: interface Electric extends Vehicle { }`,
    hint2: `When implementing a child interface, you must implement ALL methods from both the child and parent interfaces.`,
    whyItMatters: `Interface inheritance enables building complex type hierarchies while maintaining flexibility. It's fundamental to Java's Collections Framework (List extends Collection, Set extends Collection) and is used extensively in enterprise applications. Marker interfaces like Serializable and Cloneable tag classes with special capabilities without adding methods. This pattern promotes the Interface Segregation Principle - clients shouldn't depend on interfaces they don't use.`,
    order: 6,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test 1: TeslaModelS implements Vehicle
class Test1 {
    @Test
    void testImplementsVehicle() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertTrue(tesla instanceof Vehicle);
    }
}

// Test 2: TeslaModelS implements Electric
class Test2 {
    @Test
    void testImplementsElectric() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertTrue(tesla instanceof Electric);
    }
}

// Test 3: TeslaModelS implements Autonomous
class Test3 {
    @Test
    void testImplementsAutonomous() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertTrue(tesla instanceof Autonomous);
    }
}

// Test 4: TeslaModelS implements Premium marker interface
class Test4 {
    @Test
    void testImplementsPremium() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertTrue(tesla instanceof Premium);
    }
}

// Test 5: getBatteryLevel returns correct value
class Test5 {
    @Test
    void testGetBatteryLevel() {
        TeslaModelS tesla = new TeslaModelS(85);
        assertEquals(85, tesla.getBatteryLevel());
    }
}

// Test 6: charge increases battery level
class Test6 {
    @Test
    void testCharge() {
        TeslaModelS tesla = new TeslaModelS(80);
        tesla.charge(15);
        assertEquals(95, tesla.getBatteryLevel());
    }
}

// Test 7: charge does not exceed 100
class Test7 {
    @Test
    void testChargeMaxLimit() {
        TeslaModelS tesla = new TeslaModelS(90);
        tesla.charge(20);
        assertEquals(100, tesla.getBatteryLevel());
    }
}

// Test 8: Autopilot toggles correctly
class Test8 {
    @Test
    void testAutopilotToggle() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertFalse(tesla.isAutoPilotActive());
        tesla.enableAutoPilot();
        assertTrue(tesla.isAutoPilotActive());
        tesla.disableAutoPilot();
        assertFalse(tesla.isAutoPilotActive());
    }
}

// Test 9: getType returns correct type
class Test9 {
    @Test
    void testGetType() {
        TeslaModelS tesla = new TeslaModelS(80);
        assertEquals("Tesla Model S", tesla.getType());
    }
}

// Test 10: Polymorphic reference works
class Test10 {
    @Test
    void testPolymorphicReference() {
        TeslaModelS tesla = new TeslaModelS(75);

        Vehicle asVehicle = tesla;
        assertEquals("Tesla Model S", asVehicle.getType());

        Electric asElectric = tesla;
        assertEquals(75, asElectric.getBatteryLevel());

        Autonomous asAutonomous = tesla;
        assertFalse(asAutonomous.isAutoPilotActive());
    }
}`,
    translations: {
        ru: {
            title: 'Наследование и композиция интерфейсов',
            solutionCode: `// Базовый интерфейс для всех транспортных средств
interface Vehicle {
    void start();
    void stop();
    String getType();
}

// Интерфейс Electric расширяет Vehicle, добавляя специфические для электромобилей методы
interface Electric extends Vehicle {
    int getBatteryLevel();
    void charge(int percentage);
}

// Интерфейс Autonomous расширяет Vehicle, добавляя автономные функции
interface Autonomous extends Vehicle {
    void enableAutoPilot();
    void disableAutoPilot();
    boolean isAutoPilotActive();
}

// Маркерный интерфейс - пустой интерфейс для пометки премиум-функций
interface Premium {
    // Нет методов - просто маркер
}

// TeslaModelS реализует несколько иерархий интерфейсов
class TeslaModelS implements Electric, Autonomous, Premium {
    private int batteryLevel;
    private boolean autoPilotActive;

    public TeslaModelS(int initialBattery) {
        this.batteryLevel = initialBattery;
        this.autoPilotActive = false;
    }

    // Реализация методов Vehicle (унаследованы через Electric и Autonomous)
    @Override
    public void start() {
        System.out.println("Starting Tesla Model S");
    }

    @Override
    public void stop() {
        System.out.println("Stopping Tesla Model S");
    }

    @Override
    public String getType() {
        return "Tesla Model S";
    }

    // Реализация методов Electric
    @Override
    public int getBatteryLevel() {
        return batteryLevel;
    }

    @Override
    public void charge(int percentage) {
        this.batteryLevel = Math.min(100, batteryLevel + percentage);
        System.out.println("Charging to " + this.batteryLevel + "%");
    }

    // Реализация методов Autonomous
    @Override
    public void enableAutoPilot() {
        this.autoPilotActive = true;
        System.out.println("Enabling AutoPilot");
    }

    @Override
    public void disableAutoPilot() {
        this.autoPilotActive = false;
        System.out.println("Disabling AutoPilot");
    }

    @Override
    public boolean isAutoPilotActive() {
        return autoPilotActive;
    }
}

public class InterfaceInheritance {
    public static void main(String[] args) {
        System.out.println("=== Tesla Model S ===");
        TeslaModelS tesla = new TeslaModelS(85);

        // Используем методы из всех интерфейсов
        tesla.start();
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.charge(15);
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.enableAutoPilot();
        System.out.println("AutoPilot Active: " + tesla.isAutoPilotActive());

        // Проверяем премиум (маркерный интерфейс)
        System.out.println("Is Premium? " + (tesla instanceof Premium));
        tesla.stop();

        System.out.println("\\n=== Демонстрация полиморфизма ===");

        // Ссылка как разные типы интерфейсов
        Vehicle asVehicle = tesla;
        System.out.println("As Vehicle: " + asVehicle.getType());

        Electric asElectric = tesla;
        System.out.println("As Electric: Battery at " + asElectric.getBatteryLevel() + "%");

        Autonomous asAutonomous = tesla;
        String autopilotStatus = asAutonomous.isAutoPilotActive() ? "ON" : "OFF";
        System.out.println("As Autonomous: AutoPilot is " + autopilotStatus);
    }
}`,
            description: `# Наследование и композиция интерфейсов

Интерфейсы могут расширять другие интерфейсы, создавая иерархии. Это позволяет строить сложные контракты из более простых. Java также поддерживает маркерные интерфейсы (пустые интерфейсы, используемые для пометки).

## Требования:
1. Создайте базовый интерфейс \`Vehicle\`:
   1.1. \`void start()\`
   1.2. \`void stop()\`
   1.3. \`String getType()\`

2. Создайте интерфейс \`Electric\`, расширяющий \`Vehicle\`:
   2.1. \`int getBatteryLevel()\`
   2.2. \`void charge(int percentage)\`

3. Создайте интерфейс \`Autonomous\`, расширяющий \`Vehicle\`:
   3.1. \`void enableAutoPilot()\`
   3.2. \`void disableAutoPilot()\`
   3.3. \`boolean isAutoPilotActive()\`

4. Создайте маркерный интерфейс \`Premium\` (пустой - для премиум-функций)

5. Создайте класс \`TeslaModelS\`:
   5.1. Реализует \`Electric\`, \`Autonomous\` и \`Premium\`
   5.2. Предоставляет полную реализацию всех методов
   5.3. Отслеживает состояние (уровень батареи, статус автопилота)

6. Продемонстрируйте полиморфизм используя разные ссылки на интерфейсы

## Пример вывода:
\`\`\`
=== Tesla Model S ===
Starting Tesla Model S
Battery Level: 85%
Charging to 100%
Battery Level: 100%
Enabling AutoPilot
AutoPilot Active: true
Is Premium? true
Stopping Tesla Model S

=== Polymorphism Demonstration ===
As Vehicle: Tesla Model S
As Electric: Battery at 100%
As Autonomous: AutoPilot is ON
\`\`\``,
            hint1: `Интерфейс может расширять другой интерфейс с помощью ключевого слова 'extends': interface Electric extends Vehicle { }`,
            hint2: `При реализации дочернего интерфейса вы должны реализовать ВСЕ методы как дочернего, так и родительского интерфейсов.`,
            whyItMatters: `Наследование интерфейсов позволяет строить сложные типовые иерархии, сохраняя гибкость. Это фундаментально для Collections Framework в Java (List extends Collection, Set extends Collection) и широко используется в корпоративных приложениях. Маркерные интерфейсы, такие как Serializable и Cloneable, помечают классы специальными возможностями без добавления методов. Этот паттерн продвигает Принцип разделения интерфейса - клиенты не должны зависеть от интерфейсов, которые они не используют.`
        },
        uz: {
            title: `Interfeys merosi va kompozitsiyasi`,
            solutionCode: `// Barcha transport vositalari uchun asosiy interfeys
interface Vehicle {
    void start();
    void stop();
    String getType();
}

// Electric interfeysi Vehicle ni kengaytiradi, elektromobillarga xos metodlarni qo'shadi
interface Electric extends Vehicle {
    int getBatteryLevel();
    void charge(int percentage);
}

// Autonomous interfeysi Vehicle ni kengaytiradi, avtonom xususiyatlarni qo'shadi
interface Autonomous extends Vehicle {
    void enableAutoPilot();
    void disableAutoPilot();
    boolean isAutoPilotActive();
}

// Marker interfeysi - premium xususiyatlar uchun bo'sh interfeys
interface Premium {
    // Metodlar yo'q - faqat marker
}

// TeslaModelS bir nechta interfeys ierarxiyalarini amalga oshiradi
class TeslaModelS implements Electric, Autonomous, Premium {
    private int batteryLevel;
    private boolean autoPilotActive;

    public TeslaModelS(int initialBattery) {
        this.batteryLevel = initialBattery;
        this.autoPilotActive = false;
    }

    // Vehicle metodlarini amalga oshirish (Electric va Autonomous orqali meros olingan)
    @Override
    public void start() {
        System.out.println("Starting Tesla Model S");
    }

    @Override
    public void stop() {
        System.out.println("Stopping Tesla Model S");
    }

    @Override
    public String getType() {
        return "Tesla Model S";
    }

    // Electric metodlarini amalga oshirish
    @Override
    public int getBatteryLevel() {
        return batteryLevel;
    }

    @Override
    public void charge(int percentage) {
        this.batteryLevel = Math.min(100, batteryLevel + percentage);
        System.out.println("Charging to " + this.batteryLevel + "%");
    }

    // Autonomous metodlarini amalga oshirish
    @Override
    public void enableAutoPilot() {
        this.autoPilotActive = true;
        System.out.println("Enabling AutoPilot");
    }

    @Override
    public void disableAutoPilot() {
        this.autoPilotActive = false;
        System.out.println("Disabling AutoPilot");
    }

    @Override
    public boolean isAutoPilotActive() {
        return autoPilotActive;
    }
}

public class InterfaceInheritance {
    public static void main(String[] args) {
        System.out.println("=== Tesla Model S ===");
        TeslaModelS tesla = new TeslaModelS(85);

        // Barcha interfeyslardan metodlardan foydalanamiz
        tesla.start();
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.charge(15);
        System.out.println("Battery Level: " + tesla.getBatteryLevel() + "%");
        tesla.enableAutoPilot();
        System.out.println("AutoPilot Active: " + tesla.isAutoPilotActive());

        // Premium ekanligini tekshiramiz (marker interfeysi)
        System.out.println("Is Premium? " + (tesla instanceof Premium));
        tesla.stop();

        System.out.println("\\n=== Polimorfizm namoyishi ===");

        // Turli interfeys turlari sifatida havola
        Vehicle asVehicle = tesla;
        System.out.println("As Vehicle: " + asVehicle.getType());

        Electric asElectric = tesla;
        System.out.println("As Electric: Battery at " + asElectric.getBatteryLevel() + "%");

        Autonomous asAutonomous = tesla;
        String autopilotStatus = asAutonomous.isAutoPilotActive() ? "ON" : "OFF";
        System.out.println("As Autonomous: AutoPilot is " + autopilotStatus);
    }
}`,
            description: `# Interfeys merosi va kompozitsiyasi

Interfeyslar boshqa interfeyslarni kengaytirib, ierarxiyalar yaratishi mumkin. Bu oddiyroq interfeyslardan murakkab shartnomalarni qurishga imkon beradi. Java shuningdek marker interfeyslarni (belgilash uchun ishlatiadigan bo'sh interfeyslar) qo'llab-quvvatlaydi.

## Talablar:
1. Asosiy \`Vehicle\` interfeysini yarating:
   1.1. \`void start()\`
   1.2. \`void stop()\`
   1.3. \`String getType()\`

2. \`Vehicle\` ni kengaytiradigan \`Electric\` interfeysini yarating:
   2.1. \`int getBatteryLevel()\`
   2.2. \`void charge(int percentage)\`

3. \`Vehicle\` ni kengaytiradigan \`Autonomous\` interfeysini yarating:
   3.1. \`void enableAutoPilot()\`
   3.2. \`void disableAutoPilot()\`
   3.3. \`boolean isAutoPilotActive()\`

4. \`Premium\` marker interfeysini yarating (bo'sh - premium xususiyatlar uchun)

5. \`TeslaModelS\` klassini yarating:
   5.1. \`Electric\`, \`Autonomous\` va \`Premium\` ni amalga oshiradi
   5.2. Barcha metodlarning to'liq implementatsiyasini taqdim etadi
   5.3. Holatni kuzatadi (batareya darajasi, avtopilot holati)

6. Turli interfeys havolalaridan foydalanib polimorfizmni namoyish eting

## Chiqish namunasi:
\`\`\`
=== Tesla Model S ===
Starting Tesla Model S
Battery Level: 85%
Charging to 100%
Battery Level: 100%
Enabling AutoPilot
AutoPilot Active: true
Is Premium? true
Stopping Tesla Model S

=== Polymorphism Demonstration ===
As Vehicle: Tesla Model S
As Electric: Battery at 100%
As Autonomous: AutoPilot is ON
\`\`\``,
            hint1: `Interfeys 'extends' kalit so'zi yordamida boshqa interfeysni kengaytirishi mumkin: interface Electric extends Vehicle { }`,
            hint2: `Bolа interfeysini amalga oshirishda siz ham bola, ham ota-ona interfeyslaridan BARCHA metodlarni amalga oshirishingiz kerak.`,
            whyItMatters: `Interfeys merosi moslashuvchanlikni saqlab qolgan holda murakkab turdagi ierarxiyalarni qurishga imkon beradi. Bu Java-ning Collections Framework uchun asosiy hisoblanadi (List extends Collection, Set extends Collection) va korporativ ilovalarda keng qo'llaniladi. Serializable va Cloneable kabi marker interfeyslar metodlar qo'shmasdan klasslarni maxsus imkoniyatlar bilan belgilaydi. Bu namuna Interfeys ajratish printsipini ilgari suradi - mijozlar ular ishlatmaydigan interfeyslaгa bog'liq bo'lmasligi kerak.`
        }
    }
};

export default task;
