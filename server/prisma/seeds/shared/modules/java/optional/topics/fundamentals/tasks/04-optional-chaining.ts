import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-optional-chaining',
    title: 'Optional Chaining',
    difficulty: 'medium',
    tags: ['java', 'optional', 'java8', 'chaining', 'nested-objects'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Optional Chaining

Optional chaining allows you to safely navigate through nested objects without null pointer exceptions. By combining map(), flatMap(), and filter(), you can create elegant chains of operations that handle missing values gracefully. This is especially useful when working with complex object graphs.

## Requirements:
1. Navigate nested objects safely:
   1.1. Access nested properties without null checks
   1.2. Use flatMap() for methods returning Optional
   1.3. Chain multiple property accesses

2. Handle nested Optionals:
   2.1. Avoid Optional<Optional<T>> with flatMap()
   2.2. Combine multiple Optional-returning methods
   2.3. Extract deeply nested values

3. Combine chaining with transformations:
   3.1. Apply transformations during navigation
   3.2. Filter values at any chain step
   3.3. Provide defaults at the end of chain

4. Create complex navigation patterns:
   4.1. Multi-level object traversal
   4.2. Conditional navigation based on values
   4.3. Safe extraction from collections

## Example Output:
\`\`\`
=== Basic Chaining ===
User name: John Doe
User email: john@example.com
User city: New York

=== Nested Optional Handling ===
With flatMap: NEW YORK
Without flatMap would be: Optional[Optional[NEW YORK]]

=== Complex Chaining ===
User: John Doe
Email domain: EXAMPLE.COM
Address: NEW YORK, NY
ZIP code: 10001

=== Conditional Navigation ===
Premium user discount: 20%
Regular user discount: Not eligible
\`\`\``,
    initialCode: `// TODO: Import Optional

public class OptionalChaining {
    public static void main(String[] args) {
        // TODO: Create test objects with nested properties

        // TODO: Demonstrate basic chaining

        // TODO: Handle nested Optionals with flatMap

        // TODO: Create complex navigation chains

        // TODO: Show conditional navigation
    }

    // TODO: Create helper classes (User, Address, etc.)
}`,
    solutionCode: `import java.util.Optional;

public class OptionalChaining {
    public static void main(String[] args) {
        System.out.println("=== Basic Chaining ===");

        User user = new User("John Doe", "john@example.com",
            new Address("New York", "NY", "10001"));

        // Chain through nested objects safely
        String name = Optional.of(user)
            .map(User::getName)
            .orElse("Unknown");
        System.out.println("User name: " + name);

        String email = Optional.of(user)
            .map(User::getEmail)
            .orElse("No email");
        System.out.println("User email: " + email);

        // Navigate to nested object
        String city = Optional.of(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .orElse("Unknown city");
        System.out.println("User city: " + city);

        System.out.println("\\n=== Nested Optional Handling ===");

        // User with Optional address
        User user2 = new User("Jane Smith", "jane@example.com", null);

        // flatMap prevents Optional<Optional<String>>
        String cityWithFlatMap = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getCity)
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("With flatMap: " + cityWithFlatMap);

        // Without flatMap would create nested Optional
        Optional<Optional<String>> nested = Optional.of(user)
            .map(User::getOptionalAddress)
            .map(opt -> opt.map(Address::getCity).map(String::toUpperCase));
        System.out.println("Without flatMap would be: " + nested);

        System.out.println("\\n=== Complex Chaining ===");

        // Multiple transformations in chain
        String userInfo = Optional.of(user)
            .filter(u -> u.getEmail() != null)
            .map(u -> "User: " + u.getName())
            .orElse("Invalid user");
        System.out.println(userInfo);

        // Extract and transform nested property
        String emailDomain = Optional.of(user)
            .map(User::getEmail)
            .filter(e -> e.contains("@"))
            .map(e -> e.substring(e.indexOf("@") + 1))
            .map(String::toUpperCase)
            .orElse("N/A");
        System.out.println("Email domain: " + emailDomain);

        // Deep navigation with multiple flatMaps
        String fullAddress = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(addr -> addr.getCity() + ", " + addr.getState())
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("Address: " + fullAddress);

        // Chain with type conversion
        Integer zipCode = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getZipCode)
            .flatMap(OptionalChaining::parseZipCode)
            .orElse(0);
        System.out.println("ZIP code: " + zipCode);

        System.out.println("\\n=== Conditional Navigation ===");

        // Navigate based on conditions
        User premiumUser = new User("Alice", "alice@example.com",
            new Address("Boston", "MA", "02101"));
        premiumUser.setPremium(true);

        String discount = Optional.of(premiumUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Premium user discount: " + discount);

        User regularUser = new User("Bob", "bob@example.com", null);
        regularUser.setPremium(false);

        String regularDiscount = Optional.of(regularUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Regular user discount: " + regularDiscount);

        System.out.println("\\n=== Safe Collection Navigation ===");

        // Navigate through collection safely
        User userWithOrders = new User("Charlie", "charlie@example.com", null);
        userWithOrders.addOrder(new Order(100.50));
        userWithOrders.addOrder(new Order(250.75));

        Double firstOrderAmount = Optional.of(userWithOrders)
            .flatMap(User::getFirstOrder)
            .map(Order::getAmount)
            .orElse(0.0);
        System.out.println("First order amount: $" + firstOrderAmount);

        // Complex chain with multiple conditions
        String orderInfo = Optional.of(userWithOrders)
            .filter(u -> u.getEmail() != null)
            .flatMap(User::getFirstOrder)
            .filter(order -> order.getAmount() > 50)
            .map(order -> "Large order: $" + order.getAmount())
            .orElse("No large orders");
        System.out.println(orderInfo);
    }

    private static Optional<Integer> parseZipCode(String zip) {
        try {
            return Optional.of(Integer.parseInt(zip));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class User {
        private String name;
        private String email;
        private Address address;
        private boolean premium;
        private java.util.List<Order> orders = new java.util.ArrayList<>();

        public User(String name, String email, Address address) {
            this.name = name;
            this.email = email;
            this.address = address;
        }

        public String getName() {
            return name;
        }

        public String getEmail() {
            return email;
        }

        public Address getAddress() {
            return address;
        }

        public Optional<Address> getOptionalAddress() {
            return Optional.ofNullable(address);
        }

        public boolean isPremium() {
            return premium;
        }

        public void setPremium(boolean premium) {
            this.premium = premium;
        }

        public Optional<Integer> getDiscount() {
            return premium ? Optional.of(20) : Optional.empty();
        }

        public void addOrder(Order order) {
            orders.add(order);
        }

        public Optional<Order> getFirstOrder() {
            return orders.isEmpty() ? Optional.empty() : Optional.of(orders.get(0));
        }
    }

    static class Address {
        private String city;
        private String state;
        private String zipCode;

        public Address(String city, String state, String zipCode) {
            this.city = city;
            this.state = state;
            this.zipCode = zipCode;
        }

        public String getCity() {
            return city;
        }

        public String getState() {
            return state;
        }

        public String getZipCode() {
            return zipCode;
        }
    }

    static class Order {
        private double amount;

        public Order(double amount) {
            this.amount = amount;
        }

        public double getAmount() {
            return amount;
        }
    }
}`,
    hint1: `Use flatMap() whenever a method returns Optional to avoid creating Optional<Optional<T>>. Use map() for methods that return regular values.`,
    hint2: `You can insert filter() at any point in the chain to conditionally continue processing. If filter returns false, the rest of the chain returns empty Optional.`,
    whyItMatters: `Optional chaining eliminates the need for nested null checks and makes code much more readable. It's especially powerful when working with complex domain models where properties may be absent. This pattern is fundamental to functional programming in Java and prevents many runtime errors.

**Production Pattern:**
\`\`\`java
// Safe navigation through nested DTOs
String shippingCity = order
    .flatMap(Order::getShippingAddress)
    .map(Address::getCity)
    .orElse("City not specified");

// Complex business logic chain
BigDecimal finalPrice = order
    .filter(Order::isValid)
    .flatMap(Order::getCustomer)
    .filter(Customer::isPremium)
    .flatMap(Customer::getDiscount)
    .map(discount -> order.getPrice().multiply(discount))
    .orElse(order.getPrice());

// Repository query chain
User activeUser = userRepository.findById(userId)
    .filter(User::isActive)
    .filter(User::isVerified)
    .orElseThrow(() -> new UserNotFoundException());
\`\`\`

**Practical Benefits:**
- Replaces deep null check pyramids with flat chains
- Type-safe navigation prevents ClassCastException
- Self-documenting - chain shows possible absence points`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Optional;

// Test 1: Basic chaining with map works
class Test1 {
    @Test
    void testBasicChainingWithMap() {
        Optional<String> opt = Optional.of("John Doe");
        String result = opt
            .map(String::toUpperCase)
            .orElse("Unknown");
        assertEquals("JOHN DOE", result);
    }
}

// Test 2: Chaining through nested objects
class Test2 {
    @Test
    void testChainingNestedObjects() {
        User user = new User("John", "john@example.com");
        String email = Optional.of(user)
            .map(User::getEmail)
            .orElse("No email");
        assertEquals("john@example.com", email);
    }
    static class User {
        private String name, email;
        User(String n, String e) { name = n; email = e; }
        String getName() { return name; }
        String getEmail() { return email; }
    }
}

// Test 3: flatMap avoids nested Optional
class Test3 {
    @Test
    void testFlatMapAvoidsNesting() {
        Optional<String> opt = Optional.of("test");
        Optional<String> result = opt.flatMap(s -> Optional.of(s.toUpperCase()));
        assertEquals("TEST", result.get());
    }
}

// Test 4: Chain with filter condition
class Test4 {
    @Test
    void testChainWithFilter() {
        Optional<String> opt = Optional.of("john@example.com");
        String domain = opt
            .filter(e -> e.contains("@"))
            .map(e -> e.substring(e.indexOf("@") + 1))
            .orElse("N/A");
        assertEquals("example.com", domain);
    }
}

// Test 5: Chain returns empty when filter fails
class Test5 {
    @Test
    void testChainReturnsEmptyOnFilterFail() {
        Optional<String> opt = Optional.of("noemail");
        Optional<String> result = opt
            .filter(e -> e.contains("@"))
            .map(String::toUpperCase);
        assertTrue(result.isEmpty());
    }
}

// Test 6: Complex chain with multiple operations
class Test6 {
    @Test
    void testComplexChain() {
        Optional<String> opt = Optional.of("  hello world  ");
        String result = opt
            .map(String::trim)
            .filter(s -> s.length() > 5)
            .map(String::toUpperCase)
            .orElse("too short");
        assertEquals("HELLO WORLD", result);
    }
}

// Test 7: Chain short-circuits on empty
class Test7 {
    @Test
    void testChainShortCircuitsOnEmpty() {
        Optional<String> opt = Optional.empty();
        String result = opt
            .map(String::toUpperCase)
            .map(s -> s + "!")
            .orElse("default");
        assertEquals("default", result);
    }
}

// Test 8: Chain with type transformation
class Test8 {
    @Test
    void testChainTypeTransformation() {
        Optional<String> opt = Optional.of("123");
        Integer result = opt
            .map(Integer::parseInt)
            .filter(n -> n > 100)
            .map(n -> n * 2)
            .orElse(0);
        assertEquals(246, result);
    }
}

// Test 9: Chain handles null in middle
class Test9 {
    @Test
    void testChainHandlesNullInMiddle() {
        Optional<String> opt = Optional.ofNullable("test");
        String result = opt
            .map(s -> (String) null)
            .map(String::toUpperCase)
            .orElse("was null");
        assertEquals("was null", result);
    }
}

// Test 10: Multiple flatMap in chain
class Test10 {
    @Test
    void testMultipleFlatMapChain() {
        Optional<String> opt = Optional.of("42");
        Integer result = opt
            .flatMap(s -> {
                try { return Optional.of(Integer.parseInt(s)); }
                catch (NumberFormatException e) { return Optional.empty(); }
            })
            .flatMap(n -> n > 0 ? Optional.of(n * 2) : Optional.empty())
            .orElse(0);
        assertEquals(84, result);
    }
}`,
    translations: {
        ru: {
            title: 'Цепочки Optional',
            solutionCode: `import java.util.Optional;

public class OptionalChaining {
    public static void main(String[] args) {
        System.out.println("=== Базовые цепочки ===");

        User user = new User("John Doe", "john@example.com",
            new Address("New York", "NY", "10001"));

        // Безопасная цепочка через вложенные объекты
        String name = Optional.of(user)
            .map(User::getName)
            .orElse("Unknown");
        System.out.println("User name: " + name);

        String email = Optional.of(user)
            .map(User::getEmail)
            .orElse("No email");
        System.out.println("User email: " + email);

        // Навигация к вложенному объекту
        String city = Optional.of(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .orElse("Unknown city");
        System.out.println("User city: " + city);

        System.out.println("\\n=== Обработка вложенных Optional ===");

        // Пользователь с Optional адресом
        User user2 = new User("Jane Smith", "jane@example.com", null);

        // flatMap предотвращает Optional<Optional<String>>
        String cityWithFlatMap = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getCity)
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("With flatMap: " + cityWithFlatMap);

        // Без flatMap создался бы вложенный Optional
        Optional<Optional<String>> nested = Optional.of(user)
            .map(User::getOptionalAddress)
            .map(opt -> opt.map(Address::getCity).map(String::toUpperCase));
        System.out.println("Without flatMap would be: " + nested);

        System.out.println("\\n=== Сложные цепочки ===");

        // Множественные трансформации в цепочке
        String userInfo = Optional.of(user)
            .filter(u -> u.getEmail() != null)
            .map(u -> "User: " + u.getName())
            .orElse("Invalid user");
        System.out.println(userInfo);

        // Извлечение и трансформация вложенного свойства
        String emailDomain = Optional.of(user)
            .map(User::getEmail)
            .filter(e -> e.contains("@"))
            .map(e -> e.substring(e.indexOf("@") + 1))
            .map(String::toUpperCase)
            .orElse("N/A");
        System.out.println("Email domain: " + emailDomain);

        // Глубокая навигация с несколькими flatMap
        String fullAddress = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(addr -> addr.getCity() + ", " + addr.getState())
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("Address: " + fullAddress);

        // Цепочка с преобразованием типа
        Integer zipCode = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getZipCode)
            .flatMap(OptionalChaining::parseZipCode)
            .orElse(0);
        System.out.println("ZIP code: " + zipCode);

        System.out.println("\\n=== Условная навигация ===");

        // Навигация на основе условий
        User premiumUser = new User("Alice", "alice@example.com",
            new Address("Boston", "MA", "02101"));
        premiumUser.setPremium(true);

        String discount = Optional.of(premiumUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Premium user discount: " + discount);

        User regularUser = new User("Bob", "bob@example.com", null);
        regularUser.setPremium(false);

        String regularDiscount = Optional.of(regularUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Regular user discount: " + regularDiscount);

        System.out.println("\\n=== Безопасная навигация по коллекции ===");

        // Безопасная навигация по коллекции
        User userWithOrders = new User("Charlie", "charlie@example.com", null);
        userWithOrders.addOrder(new Order(100.50));
        userWithOrders.addOrder(new Order(250.75));

        Double firstOrderAmount = Optional.of(userWithOrders)
            .flatMap(User::getFirstOrder)
            .map(Order::getAmount)
            .orElse(0.0);
        System.out.println("First order amount: $" + firstOrderAmount);

        // Сложная цепочка с несколькими условиями
        String orderInfo = Optional.of(userWithOrders)
            .filter(u -> u.getEmail() != null)
            .flatMap(User::getFirstOrder)
            .filter(order -> order.getAmount() > 50)
            .map(order -> "Large order: $" + order.getAmount())
            .orElse("No large orders");
        System.out.println(orderInfo);
    }

    private static Optional<Integer> parseZipCode(String zip) {
        try {
            return Optional.of(Integer.parseInt(zip));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class User {
        private String name;
        private String email;
        private Address address;
        private boolean premium;
        private java.util.List<Order> orders = new java.util.ArrayList<>();

        public User(String name, String email, Address address) {
            this.name = name;
            this.email = email;
            this.address = address;
        }

        public String getName() {
            return name;
        }

        public String getEmail() {
            return email;
        }

        public Address getAddress() {
            return address;
        }

        public Optional<Address> getOptionalAddress() {
            return Optional.ofNullable(address);
        }

        public boolean isPremium() {
            return premium;
        }

        public void setPremium(boolean premium) {
            this.premium = premium;
        }

        public Optional<Integer> getDiscount() {
            return premium ? Optional.of(20) : Optional.empty();
        }

        public void addOrder(Order order) {
            orders.add(order);
        }

        public Optional<Order> getFirstOrder() {
            return orders.isEmpty() ? Optional.empty() : Optional.of(orders.get(0));
        }
    }

    static class Address {
        private String city;
        private String state;
        private String zipCode;

        public Address(String city, String state, String zipCode) {
            this.city = city;
            this.state = state;
            this.zipCode = zipCode;
        }

        public String getCity() {
            return city;
        }

        public String getState() {
            return state;
        }

        public String getZipCode() {
            return zipCode;
        }
    }

    static class Order {
        private double amount;

        public Order(double amount) {
            this.amount = amount;
        }

        public double getAmount() {
            return amount;
        }
    }
}`,
            description: `# Цепочки Optional

Цепочки Optional позволяют безопасно перемещаться по вложенным объектам без исключений null pointer. Комбинируя map(), flatMap() и filter(), вы можете создавать элегантные цепочки операций, которые грациозно обрабатывают отсутствующие значения. Это особенно полезно при работе со сложными графами объектов.

## Требования:
1. Безопасная навигация по вложенным объектам:
   1.1. Доступ к вложенным свойствам без проверок на null
   1.2. Использование flatMap() для методов, возвращающих Optional
   1.3. Объединение нескольких доступов к свойствам

2. Обработка вложенных Optionals:
   2.1. Избегайте Optional<Optional<T>> с помощью flatMap()
   2.2. Комбинируйте несколько методов, возвращающих Optional
   2.3. Извлекайте глубоко вложенные значения

3. Комбинируйте цепочки с трансформациями:
   3.1. Применяйте трансформации во время навигации
   3.2. Фильтруйте значения на любом шаге цепочки
   3.3. Предоставляйте значения по умолчанию в конце цепочки

4. Создавайте сложные паттерны навигации:
   4.1. Многоуровневый обход объектов
   4.2. Условная навигация на основе значений
   4.3. Безопасное извлечение из коллекций

## Пример вывода:
\`\`\`
=== Basic Chaining ===
User name: John Doe
User email: john@example.com
User city: New York

=== Nested Optional Handling ===
With flatMap: NEW YORK
Without flatMap would be: Optional[Optional[NEW YORK]]

=== Complex Chaining ===
User: John Doe
Email domain: EXAMPLE.COM
Address: NEW YORK, NY
ZIP code: 10001

=== Conditional Navigation ===
Premium user discount: 20%
Regular user discount: Not eligible
\`\`\``,
            hint1: `Используйте flatMap() когда метод возвращает Optional, чтобы избежать создания Optional<Optional<T>>. Используйте map() для методов, возвращающих обычные значения.`,
            hint2: `Вы можете вставить filter() в любой точке цепочки для условного продолжения обработки. Если filter возвращает false, остальная часть цепочки возвращает пустой Optional.`,
            whyItMatters: `Цепочки Optional устраняют необходимость в вложенных проверках на null и делают код гораздо более читаемым. Это особенно мощно при работе со сложными доменными моделями, где свойства могут отсутствовать. Этот паттерн фундаментален для функционального программирования в Java и предотвращает множество runtime ошибок.

**Продакшен паттерн:**
\`\`\`java
// Безопасная навигация через вложенные DTO
String shippingCity = order
    .flatMap(Order::getShippingAddress)
    .map(Address::getCity)
    .orElse("City not specified");

// Цепочка сложной бизнес-логики
BigDecimal finalPrice = order
    .filter(Order::isValid)
    .flatMap(Order::getCustomer)
    .filter(Customer::isPremium)
    .flatMap(Customer::getDiscount)
    .map(discount -> order.getPrice().multiply(discount))
    .orElse(order.getPrice());

// Цепочка запросов к репозиторию
User activeUser = userRepository.findById(userId)
    .filter(User::isActive)
    .filter(User::isVerified)
    .orElseThrow(() -> new UserNotFoundException());
\`\`\`

**Практические преимущества:**
- Заменяет глубокие пирамиды null-проверок плоскими цепочками
- Типобезопасная навигация предотвращает ClassCastException
- Самодокументирующийся код - цепочка показывает точки возможного отсутствия`
        },
        uz: {
            title: `Optional zanjirlari`,
            solutionCode: `import java.util.Optional;

public class OptionalChaining {
    public static void main(String[] args) {
        System.out.println("=== Asosiy zanjirlar ===");

        User user = new User("John Doe", "john@example.com",
            new Address("New York", "NY", "10001"));

        // Ichki obyektlar orqali xavfsiz zanjir
        String name = Optional.of(user)
            .map(User::getName)
            .orElse("Unknown");
        System.out.println("User name: " + name);

        String email = Optional.of(user)
            .map(User::getEmail)
            .orElse("No email");
        System.out.println("User email: " + email);

        // Ichki obyektga navigatsiya
        String city = Optional.of(user)
            .map(User::getAddress)
            .map(Address::getCity)
            .orElse("Unknown city");
        System.out.println("User city: " + city);

        System.out.println("\\n=== Ichki Optional larni boshqarish ===");

        // Optional manzil bilan foydalanuvchi
        User user2 = new User("Jane Smith", "jane@example.com", null);

        // flatMap Optional<Optional<String>> ni oldini oladi
        String cityWithFlatMap = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getCity)
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("With flatMap: " + cityWithFlatMap);

        // flatMap siz ichki Optional yaratilgan bo'lardi
        Optional<Optional<String>> nested = Optional.of(user)
            .map(User::getOptionalAddress)
            .map(opt -> opt.map(Address::getCity).map(String::toUpperCase));
        System.out.println("Without flatMap would be: " + nested);

        System.out.println("\\n=== Murakkab zanjirlar ===");

        // Zanjirda bir nechta transformatsiya
        String userInfo = Optional.of(user)
            .filter(u -> u.getEmail() != null)
            .map(u -> "User: " + u.getName())
            .orElse("Invalid user");
        System.out.println(userInfo);

        // Ichki xususiyatni ajratib olish va transformatsiya qilish
        String emailDomain = Optional.of(user)
            .map(User::getEmail)
            .filter(e -> e.contains("@"))
            .map(e -> e.substring(e.indexOf("@") + 1))
            .map(String::toUpperCase)
            .orElse("N/A");
        System.out.println("Email domain: " + emailDomain);

        // Bir nechta flatMap bilan chuqur navigatsiya
        String fullAddress = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(addr -> addr.getCity() + ", " + addr.getState())
            .map(String::toUpperCase)
            .orElse("No address");
        System.out.println("Address: " + fullAddress);

        // Tur o'zgartirish bilan zanjir
        Integer zipCode = Optional.of(user)
            .flatMap(User::getOptionalAddress)
            .map(Address::getZipCode)
            .flatMap(OptionalChaining::parseZipCode)
            .orElse(0);
        System.out.println("ZIP code: " + zipCode);

        System.out.println("\\n=== Shartli navigatsiya ===");

        // Shartlar asosida navigatsiya
        User premiumUser = new User("Alice", "alice@example.com",
            new Address("Boston", "MA", "02101"));
        premiumUser.setPremium(true);

        String discount = Optional.of(premiumUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Premium user discount: " + discount);

        User regularUser = new User("Bob", "bob@example.com", null);
        regularUser.setPremium(false);

        String regularDiscount = Optional.of(regularUser)
            .filter(User::isPremium)
            .flatMap(User::getDiscount)
            .map(d -> d + "%")
            .orElse("Not eligible");
        System.out.println("Regular user discount: " + regularDiscount);

        System.out.println("\\n=== Xavfsiz kolleksiya navigatsiyasi ===");

        // Kolleksiya bo'ylab xavfsiz navigatsiya
        User userWithOrders = new User("Charlie", "charlie@example.com", null);
        userWithOrders.addOrder(new Order(100.50));
        userWithOrders.addOrder(new Order(250.75));

        Double firstOrderAmount = Optional.of(userWithOrders)
            .flatMap(User::getFirstOrder)
            .map(Order::getAmount)
            .orElse(0.0);
        System.out.println("First order amount: $" + firstOrderAmount);

        // Bir nechta shart bilan murakkab zanjir
        String orderInfo = Optional.of(userWithOrders)
            .filter(u -> u.getEmail() != null)
            .flatMap(User::getFirstOrder)
            .filter(order -> order.getAmount() > 50)
            .map(order -> "Large order: $" + order.getAmount())
            .orElse("No large orders");
        System.out.println(orderInfo);
    }

    private static Optional<Integer> parseZipCode(String zip) {
        try {
            return Optional.of(Integer.parseInt(zip));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class User {
        private String name;
        private String email;
        private Address address;
        private boolean premium;
        private java.util.List<Order> orders = new java.util.ArrayList<>();

        public User(String name, String email, Address address) {
            this.name = name;
            this.email = email;
            this.address = address;
        }

        public String getName() {
            return name;
        }

        public String getEmail() {
            return email;
        }

        public Address getAddress() {
            return address;
        }

        public Optional<Address> getOptionalAddress() {
            return Optional.ofNullable(address);
        }

        public boolean isPremium() {
            return premium;
        }

        public void setPremium(boolean premium) {
            this.premium = premium;
        }

        public Optional<Integer> getDiscount() {
            return premium ? Optional.of(20) : Optional.empty();
        }

        public void addOrder(Order order) {
            orders.add(order);
        }

        public Optional<Order> getFirstOrder() {
            return orders.isEmpty() ? Optional.empty() : Optional.of(orders.get(0));
        }
    }

    static class Address {
        private String city;
        private String state;
        private String zipCode;

        public Address(String city, String state, String zipCode) {
            this.city = city;
            this.state = state;
            this.zipCode = zipCode;
        }

        public String getCity() {
            return city;
        }

        public String getState() {
            return state;
        }

        public String getZipCode() {
            return zipCode;
        }
    }

    static class Order {
        private double amount;

        public Order(double amount) {
            this.amount = amount;
        }

        public double getAmount() {
            return amount;
        }
    }
}`,
            description: `# Optional zanjirlari

Optional zanjirlari null pointer istisnosiz ichki obyektlar orqali xavfsiz navigatsiya qilish imkonini beradi. map(), flatMap() va filter() ni birlashtirib, yo'qolgan qiymatlarni nazokat bilan boshqaradigan operatsiyalarning nafis zanjirlarini yaratishingiz mumkin. Bu murakkab obyekt grafikalari bilan ishlashda ayniqsa foydali.

## Talablar:
1. Ichki obyektlar bo'ylab xavfsiz navigatsiya:
   1.1. Null tekshiruvisiz ichki xususiyatlarga kirish
   1.2. Optional qaytaradigan metodlar uchun flatMap() dan foydalanish
   1.3. Bir nechta xususiyat kirishlarini zanjir qilish

2. Ichki Optionallarni boshqarish:
   2.1. flatMap() yordamida Optional<Optional<T>> dan qoching
   2.2. Bir nechta Optional qaytaradigan metodlarni birlashtiring
   2.3. Chuqur joylashgan qiymatlarni ajratib oling

3. Transformatsiyalar bilan zanjirlarni birlashtiring:
   3.1. Navigatsiya paytida transformatsiyalarni qo'llang
   3.2. Har qanday zanjir qadamida qiymatlarni filtrlang
   3.3. Zanjir oxirida standart qiymatlarni taqdim eting

4. Murakkab navigatsiya naqshlarini yarating:
   4.1. Ko'p darajali obyekt aylanishi
   4.2. Qiymatlarga asoslangan shartli navigatsiya
   4.3. Kolleksiyalardan xavfsiz ajratib olish

## Chiqish namunasi:
\`\`\`
=== Basic Chaining ===
User name: John Doe
User email: john@example.com
User city: New York

=== Nested Optional Handling ===
With flatMap: NEW YORK
Without flatMap would be: Optional[Optional[NEW YORK]]

=== Complex Chaining ===
User: John Doe
Email domain: EXAMPLE.COM
Address: NEW YORK, NY
ZIP code: 10001

=== Conditional Navigation ===
Premium user discount: 20%
Regular user discount: Not eligible
\`\`\``,
            hint1: `Metod Optional qaytarganda Optional<Optional<T>> yaratilishini oldini olish uchun flatMap() dan foydalaning. Oddiy qiymat qaytaradigan metodlar uchun map() dan foydalaning.`,
            hint2: `Zanjirning istalgan nuqtasiga shartli davom ettirish uchun filter() ni kiritishingiz mumkin. Agar filter false qaytarsa, zanjirning qolgan qismi bo'sh Optional qaytaradi.`,
            whyItMatters: `Optional zanjirlari ichki null tekshiruvlari zaruriyatini yo'q qiladi va kodni ancha o'qilishi osonroq qiladi. Bu xususiyatlar yo'q bo'lishi mumkin bo'lgan murakkab domen modellari bilan ishlashda ayniqsa kuchli. Bu naqsh Java da funksional dasturlash uchun asosiy hisoblanadi va ko'plab runtime xatolarining oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ichki DTO lar orqali xavfsiz navigatsiya
String shippingCity = order
    .flatMap(Order::getShippingAddress)
    .map(Address::getCity)
    .orElse("City not specified");

// Murakkab biznes-mantiq zanjiri
BigDecimal finalPrice = order
    .filter(Order::isValid)
    .flatMap(Order::getCustomer)
    .filter(Customer::isPremium)
    .flatMap(Customer::getDiscount)
    .map(discount -> order.getPrice().multiply(discount))
    .orElse(order.getPrice());

// Repository so'rovlar zanjiri
User activeUser = userRepository.findById(userId)
    .filter(User::isActive)
    .filter(User::isVerified)
    .orElseThrow(() -> new UserNotFoundException());
\`\`\`

**Amaliy foydalari:**
- Chuqur null-tekshiruv piramidalarini tekis zanjirlar bilan almashtiradi
- Tur-xavfsiz navigatsiya ClassCastException oldini oladi
- O'z-o'zini hujjatlashtiradigan kod - zanjir yo'qolish mumkin bo'lgan nuqtalarni ko'rsatadi`
        }
    }
};

export default task;
