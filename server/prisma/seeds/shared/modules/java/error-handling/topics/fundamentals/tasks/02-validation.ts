import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-input-validation',
    title: 'Input Validation',
    difficulty: 'medium',
    tags: ['java', 'validation', 'bean-validation', 'constraints'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Implement comprehensive input validation using manual checks and Bean Validation concepts.

Requirements:
1. Create a Product class with validation rules
2. Implement manual validation with detailed error messages
3. Create ValidationResult class to collect multiple errors
4. Add field-level and cross-field validation
5. Demonstrate validation in a product registration system

Example:
\`\`\`java
Product product = new Product("", -10.0, 0);
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(System.out::println);
}
\`\`\``,
    initialCode: `import java.util.*;

// TODO: Create ValidationError class

// TODO: Create ValidationResult class to collect errors

class Product {
    private String name;
    private double price;
    private int quantity;

    public Product(String name, double price, int quantity) {
        this.name = name;
        this.price = price;
        this.quantity = quantity;
    }

    // Getters
    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getQuantity() { return quantity; }
}

class ProductValidator {
    // TODO: Implement validate method
    public ValidationResult validate(Product product) {
        return null;
    }
}

public class ValidationDemo {
    public static void main(String[] args) {
        ProductValidator validator = new ProductValidator();

        // TODO: Test validation with valid and invalid products
    }
}`,
    solutionCode: `import java.util.*;

// Represents a single validation error
class ValidationError {
    private final String field;
    private final String message;
    private final Object rejectedValue;

    public ValidationError(String field, String message, Object rejectedValue) {
        this.field = field;
        this.message = message;
        this.rejectedValue = rejectedValue;
    }

    public String getField() { return field; }
    public String getMessage() { return message; }
    public Object getRejectedValue() { return rejectedValue; }

    @Override
    public String toString() {
        return String.format("Field '%s': %s (rejected value: %s)",
            field, message, rejectedValue);
    }
}

// Collects validation errors
class ValidationResult {
    private final List<ValidationError> errors;

    public ValidationResult() {
        this.errors = new ArrayList<>();
    }

    // Add a validation error
    public void addError(String field, String message, Object rejectedValue) {
        errors.add(new ValidationError(field, message, rejectedValue));
    }

    // Check if validation passed
    public boolean isValid() {
        return errors.isEmpty();
    }

    // Get all errors
    public List<ValidationError> getErrors() {
        return Collections.unmodifiableList(errors);
    }

    // Get errors for specific field
    public List<ValidationError> getFieldErrors(String field) {
        List<ValidationError> fieldErrors = new ArrayList<>();
        for (ValidationError error : errors) {
            if (error.getField().equals(field)) {
                fieldErrors.add(error);
            }
        }
        return fieldErrors;
    }

    // Get error count
    public int getErrorCount() {
        return errors.size();
    }

    @Override
    public String toString() {
        if (isValid()) {
            return "Validation passed";
        }
        StringBuilder sb = new StringBuilder("Validation failed:\\n");
        for (ValidationError error : errors) {
            sb.append("  - ").append(error).append("\\n");
        }
        return sb.toString();
    }
}

class Product {
    private String name;
    private double price;
    private int quantity;
    private String category;

    public Product(String name, double price, int quantity) {
        this(name, price, quantity, "General");
    }

    public Product(String name, double price, int quantity, String category) {
        this.name = name;
        this.price = price;
        this.quantity = quantity;
        this.category = category;
    }

    // Getters
    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getQuantity() { return quantity; }
    public String getCategory() { return category; }

    @Override
    public String toString() {
        return String.format("Product{name='%s', price=%.2f, quantity=%d, category='%s'}",
            name, price, quantity, category);
    }
}

class ProductValidator {
    // Validate product with comprehensive checks
    public ValidationResult validate(Product product) {
        ValidationResult result = new ValidationResult();

        // Validate name
        validateName(product.getName(), result);

        // Validate price
        validatePrice(product.getPrice(), result);

        // Validate quantity
        validateQuantity(product.getQuantity(), result);

        // Validate category
        validateCategory(product.getCategory(), result);

        // Cross-field validation
        validatePriceQuantityConsistency(product, result);

        return result;
    }

    // Validate product name
    private void validateName(String name, ValidationResult result) {
        if (name == null || name.trim().isEmpty()) {
            result.addError("name", "Product name is required", name);
            return;
        }

        if (name.length() < 3) {
            result.addError("name",
                "Product name must be at least 3 characters long", name);
        }

        if (name.length() > 100) {
            result.addError("name",
                "Product name must not exceed 100 characters", name);
        }

        // Check for invalid characters
        if (!name.matches("[a-zA-Z0-9\\\\s\\\\-]+")) {
            result.addError("name",
                "Product name contains invalid characters", name);
        }
    }

    // Validate price
    private void validatePrice(double price, ValidationResult result) {
        if (price < 0) {
            result.addError("price", "Price cannot be negative", price);
        }

        if (price == 0) {
            result.addError("price", "Price must be greater than zero", price);
        }

        if (price > 1_000_000) {
            result.addError("price",
                "Price exceeds maximum allowed value", price);
        }
    }

    // Validate quantity
    private void validateQuantity(int quantity, ValidationResult result) {
        if (quantity < 0) {
            result.addError("quantity", "Quantity cannot be negative", quantity);
        }

        if (quantity > 10_000) {
            result.addError("quantity",
                "Quantity exceeds maximum stock limit", quantity);
        }
    }

    // Validate category
    private void validateCategory(String category, ValidationResult result) {
        if (category == null || category.trim().isEmpty()) {
            result.addError("category", "Category is required", category);
            return;
        }

        List<String> validCategories = Arrays.asList(
            "Electronics", "Clothing", "Food", "Books", "General"
        );

        if (!validCategories.contains(category)) {
            result.addError("category",
                "Invalid category. Must be one of: " + validCategories, category);
        }
    }

    // Cross-field validation
    private void validatePriceQuantityConsistency(Product product,
                                                   ValidationResult result) {
        // Business rule: expensive items should have limited quantity
        if (product.getPrice() > 10000 && product.getQuantity() > 100) {
            result.addError("quantity",
                "High-value items (price > $10,000) cannot have quantity > 100",
                product.getQuantity());
        }
    }
}

public class ValidationDemo {
    public static void main(String[] args) {
        ProductValidator validator = new ProductValidator();

        // Example 1: Valid product
        System.out.println("=== Example 1: Valid Product ===");
        Product validProduct = new Product("Laptop Computer", 999.99, 50, "Electronics");
        ValidationResult result1 = validator.validate(validProduct);
        System.out.println(result1);
        System.out.println("Product: " + validProduct);

        // Example 2: Multiple validation errors
        System.out.println("\\n=== Example 2: Invalid Product (Multiple Errors) ===");
        Product invalidProduct = new Product("", -10.0, -5);
        ValidationResult result2 = validator.validate(invalidProduct);
        System.out.println("Is valid: " + result2.isValid());
        System.out.println("Error count: " + result2.getErrorCount());
        System.out.println("Errors:");
        result2.getErrors().forEach(error ->
            System.out.println("  - " + error));

        // Example 3: Name validation
        System.out.println("\\n=== Example 3: Name Validation ===");
        Product shortName = new Product("AB", 100.0, 10, "Books");
        ValidationResult result3 = validator.validate(shortName);
        if (!result3.isValid()) {
            List<ValidationError> nameErrors = result3.getFieldErrors("name");
            System.out.println("Name validation errors:");
            nameErrors.forEach(error -> System.out.println("  - " + error.getMessage()));
        }

        // Example 4: Price validation
        System.out.println("\\n=== Example 4: Price Validation ===");
        Product expensiveProduct = new Product("Luxury Car", 2_000_000, 5, "General");
        ValidationResult result4 = validator.validate(expensiveProduct);
        if (!result4.isValid()) {
            System.out.println("Validation errors:");
            result4.getErrors().forEach(System.out::println);
        }

        // Example 5: Category validation
        System.out.println("\\n=== Example 5: Category Validation ===");
        Product invalidCategory = new Product("Widget", 50.0, 100, "InvalidCategory");
        ValidationResult result5 = validator.validate(invalidCategory);
        if (!result5.isValid()) {
            List<ValidationError> categoryErrors = result5.getFieldErrors("category");
            categoryErrors.forEach(error ->
                System.out.println("Category error: " + error.getMessage()));
        }

        // Example 6: Cross-field validation
        System.out.println("\\n=== Example 6: Cross-field Validation ===");
        Product expensiveWithHighQty = new Product("Diamond Ring", 50000, 500, "General");
        ValidationResult result6 = validator.validate(expensiveWithHighQty);
        if (!result6.isValid()) {
            System.out.println("Business rule violations:");
            result6.getErrors().forEach(System.out::println);
        }

        // Example 7: Batch validation
        System.out.println("\\n=== Example 7: Batch Validation ===");
        List<Product> products = Arrays.asList(
            new Product("Book", 29.99, 100, "Books"),
            new Product("", 0, -1, ""),
            new Product("Phone", 799.99, 200, "Electronics")
        );

        for (int i = 0; i < products.size(); i++) {
            ValidationResult result = validator.validate(products.get(i));
            System.out.println("Product " + (i + 1) + ": " +
                (result.isValid() ? "VALID" : "INVALID (" +
                result.getErrorCount() + " errors)"));
        }
    }
}`,
    hint1: `Create a ValidationResult class that stores a list of ValidationError objects. Each error should contain the field name, message, and rejected value for detailed reporting.`,
    hint2: `In the validator, perform multiple checks for each field and add errors to the result. Check for null values first, then validate constraints like length, range, and format. Don't return early - collect all errors.`,
    whyItMatters: `Input validation is crucial for data integrity and security. Collecting all validation errors at once provides better user experience than failing on the first error. This pattern is the foundation for Bean Validation (JSR 303/380) and is essential for building robust applications.

**Production Pattern:**
\`\`\`java
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(error ->
        System.out.println("Field '" + error.getField() +
                         "': " + error.getMessage()));
}
\`\`\`

**Practical Benefits:**
- Collects all errors in one pass for better UX
- Foundation for Bean Validation (JSR 303/380)
- Critical for data integrity and security`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.List;

// Test1: Verify ValidationResult starts empty
class Test1 {
    @Test
    public void test() {
        ValidationResult result = new ValidationResult();
        assertTrue(result.isValid());
        assertEquals(0, result.getErrorCount());
    }
}

// Test2: Verify adding errors makes result invalid
class Test2 {
    @Test
    public void test() {
        ValidationResult result = new ValidationResult();
        result.addError("field", "message", "value");
        assertFalse(result.isValid());
        assertEquals(1, result.getErrorCount());
    }
}

// Test3: Verify ValidationError contains correct data
class Test3 {
    @Test
    public void test() {
        ValidationError error = new ValidationError("name", "Name is required", null);
        assertEquals("name", error.getField());
        assertEquals("Name is required", error.getMessage());
        assertNull(error.getRejectedValue());
    }
}

// Test4: Verify valid Product passes validation
class Test4 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("Laptop", 999.99, 50, "Electronics");
        ValidationResult result = validator.validate(product);
        assertTrue(result.isValid());
    }
}

// Test5: Verify empty name fails validation
class Test5 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("", 100.0, 10, "General");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        List<ValidationError> nameErrors = result.getFieldErrors("name");
        assertTrue(nameErrors.size() > 0);
    }
}

// Test6: Verify negative price fails validation
class Test6 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("Test Product", -10.0, 10, "General");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        List<ValidationError> priceErrors = result.getFieldErrors("price");
        assertTrue(priceErrors.size() > 0);
    }
}

// Test7: Verify negative quantity fails validation
class Test7 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("Test Product", 100.0, -5, "General");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        List<ValidationError> qtyErrors = result.getFieldErrors("quantity");
        assertTrue(qtyErrors.size() > 0);
    }
}

// Test8: Verify invalid category fails validation
class Test8 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("Test", 100.0, 10, "InvalidCategory");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        List<ValidationError> catErrors = result.getFieldErrors("category");
        assertTrue(catErrors.size() > 0);
    }
}

// Test9: Verify multiple errors are collected
class Test9 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        Product product = new Product("", -10.0, -5, "");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        assertTrue(result.getErrorCount() >= 3);
    }
}

// Test10: Verify cross-field validation
class Test10 {
    @Test
    public void test() {
        ProductValidator validator = new ProductValidator();
        // High price with high quantity should trigger cross-field validation
        Product product = new Product("Expensive Item", 50000.0, 500, "General");
        ValidationResult result = validator.validate(product);
        assertFalse(result.isValid());
        boolean hasQuantityError = false;
        for (ValidationError error : result.getErrors()) {
            if (error.getField().equals("quantity") &&
                error.getMessage().contains("High-value")) {
                hasQuantityError = true;
                break;
            }
        }
        assertTrue(hasQuantityError);
    }
}
`,
    order: 2,
    translations: {
        ru: {
            title: 'Валидация Входных Данных',
            solutionCode: `import java.util.*;

// Представляет одну ошибку валидации
class ValidationError {
    private final String field;
    private final String message;
    private final Object rejectedValue;

    public ValidationError(String field, String message, Object rejectedValue) {
        this.field = field;
        this.message = message;
        this.rejectedValue = rejectedValue;
    }

    public String getField() { return field; }
    public String getMessage() { return message; }
    public Object getRejectedValue() { return rejectedValue; }

    @Override
    public String toString() {
        return String.format("Поле '%s': %s (отклоненное значение: %s)",
            field, message, rejectedValue);
    }
}

// Собирает ошибки валидации
class ValidationResult {
    private final List<ValidationError> errors;

    public ValidationResult() {
        this.errors = new ArrayList<>();
    }

    // Добавить ошибку валидации
    public void addError(String field, String message, Object rejectedValue) {
        errors.add(new ValidationError(field, message, rejectedValue));
    }

    // Проверить успешность валидации
    public boolean isValid() {
        return errors.isEmpty();
    }

    // Получить все ошибки
    public List<ValidationError> getErrors() {
        return Collections.unmodifiableList(errors);
    }

    // Получить ошибки для конкретного поля
    public List<ValidationError> getFieldErrors(String field) {
        List<ValidationError> fieldErrors = new ArrayList<>();
        for (ValidationError error : errors) {
            if (error.getField().equals(field)) {
                fieldErrors.add(error);
            }
        }
        return fieldErrors;
    }

    // Получить количество ошибок
    public int getErrorCount() {
        return errors.size();
    }

    @Override
    public String toString() {
        if (isValid()) {
            return "Валидация пройдена";
        }
        StringBuilder sb = new StringBuilder("Валидация не пройдена:\\n");
        for (ValidationError error : errors) {
            sb.append("  - ").append(error).append("\\n");
        }
        return sb.toString();
    }
}

class Product {
    private String name;
    private double price;
    private int quantity;
    private String category;

    public Product(String name, double price, int quantity) {
        this(name, price, quantity, "Общее");
    }

    public Product(String name, double price, int quantity, String category) {
        this.name = name;
        this.price = price;
        this.quantity = quantity;
        this.category = category;
    }

    // Геттеры
    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getQuantity() { return quantity; }
    public String getCategory() { return category; }

    @Override
    public String toString() {
        return String.format("Product{name='%s', price=%.2f, quantity=%d, category='%s'}",
            name, price, quantity, category);
    }
}

class ProductValidator {
    // Валидация продукта с комплексными проверками
    public ValidationResult validate(Product product) {
        ValidationResult result = new ValidationResult();

        // Валидация имени
        validateName(product.getName(), result);

        // Валидация цены
        validatePrice(product.getPrice(), result);

        // Валидация количества
        validateQuantity(product.getQuantity(), result);

        // Валидация категории
        validateCategory(product.getCategory(), result);

        // Кросс-полевая валидация
        validatePriceQuantityConsistency(product, result);

        return result;
    }

    // Валидация названия продукта
    private void validateName(String name, ValidationResult result) {
        if (name == null || name.trim().isEmpty()) {
            result.addError("name", "Название продукта обязательно", name);
            return;
        }

        if (name.length() < 3) {
            result.addError("name",
                "Название продукта должно содержать минимум 3 символа", name);
        }

        if (name.length() > 100) {
            result.addError("name",
                "Название продукта не должно превышать 100 символов", name);
        }

        // Проверка недопустимых символов
        if (!name.matches("[a-zA-Zа-яА-Я0-9\\\\s\\\\-]+")) {
            result.addError("name",
                "Название продукта содержит недопустимые символы", name);
        }
    }

    // Валидация цены
    private void validatePrice(double price, ValidationResult result) {
        if (price < 0) {
            result.addError("price", "Цена не может быть отрицательной", price);
        }

        if (price == 0) {
            result.addError("price", "Цена должна быть больше нуля", price);
        }

        if (price > 1_000_000) {
            result.addError("price",
                "Цена превышает максимально допустимое значение", price);
        }
    }

    // Валидация количества
    private void validateQuantity(int quantity, ValidationResult result) {
        if (quantity < 0) {
            result.addError("quantity", "Количество не может быть отрицательным", quantity);
        }

        if (quantity > 10_000) {
            result.addError("quantity",
                "Количество превышает максимальный лимит запасов", quantity);
        }
    }

    // Валидация категории
    private void validateCategory(String category, ValidationResult result) {
        if (category == null || category.trim().isEmpty()) {
            result.addError("category", "Категория обязательна", category);
            return;
        }

        List<String> validCategories = Arrays.asList(
            "Электроника", "Одежда", "Продукты", "Книги", "Общее"
        );

        if (!validCategories.contains(category)) {
            result.addError("category",
                "Недопустимая категория. Должна быть одна из: " + validCategories, category);
        }
    }

    // Кросс-полевая валидация
    private void validatePriceQuantityConsistency(Product product,
                                                   ValidationResult result) {
        // Бизнес-правило: дорогие товары должны иметь ограниченное количество
        if (product.getPrice() > 10000 && product.getQuantity() > 100) {
            result.addError("quantity",
                "Дорогие товары (цена > 10000₽) не могут иметь количество > 100",
                product.getQuantity());
        }
    }
}

public class ValidationDemo {
    public static void main(String[] args) {
        ProductValidator validator = new ProductValidator();

        // Пример 1: Валидный продукт
        System.out.println("=== Пример 1: Валидный Продукт ===");
        Product validProduct = new Product("Ноутбук", 999.99, 50, "Электроника");
        ValidationResult result1 = validator.validate(validProduct);
        System.out.println(result1);
        System.out.println("Продукт: " + validProduct);

        // Пример 2: Множественные ошибки валидации
        System.out.println("\\n=== Пример 2: Невалидный Продукт (Множественные Ошибки) ===");
        Product invalidProduct = new Product("", -10.0, -5);
        ValidationResult result2 = validator.validate(invalidProduct);
        System.out.println("Валиден: " + result2.isValid());
        System.out.println("Количество ошибок: " + result2.getErrorCount());
        System.out.println("Ошибки:");
        result2.getErrors().forEach(error ->
            System.out.println("  - " + error));

        // Пример 3: Валидация имени
        System.out.println("\\n=== Пример 3: Валидация Имени ===");
        Product shortName = new Product("AB", 100.0, 10, "Книги");
        ValidationResult result3 = validator.validate(shortName);
        if (!result3.isValid()) {
            List<ValidationError> nameErrors = result3.getFieldErrors("name");
            System.out.println("Ошибки валидации имени:");
            nameErrors.forEach(error -> System.out.println("  - " + error.getMessage()));
        }

        // Пример 4: Валидация цены
        System.out.println("\\n=== Пример 4: Валидация Цены ===");
        Product expensiveProduct = new Product("Роскошный Автомобиль", 2_000_000, 5, "Общее");
        ValidationResult result4 = validator.validate(expensiveProduct);
        if (!result4.isValid()) {
            System.out.println("Ошибки валидации:");
            result4.getErrors().forEach(System.out::println);
        }

        // Пример 5: Валидация категории
        System.out.println("\\n=== Пример 5: Валидация Категории ===");
        Product invalidCategory = new Product("Виджет", 50.0, 100, "НедопустимаяКатегория");
        ValidationResult result5 = validator.validate(invalidCategory);
        if (!result5.isValid()) {
            List<ValidationError> categoryErrors = result5.getFieldErrors("category");
            categoryErrors.forEach(error ->
                System.out.println("Ошибка категории: " + error.getMessage()));
        }

        // Пример 6: Кросс-полевая валидация
        System.out.println("\\n=== Пример 6: Кросс-полевая Валидация ===");
        Product expensiveWithHighQty = new Product("Бриллиантовое Кольцо", 50000, 500, "Общее");
        ValidationResult result6 = validator.validate(expensiveWithHighQty);
        if (!result6.isValid()) {
            System.out.println("Нарушения бизнес-правил:");
            result6.getErrors().forEach(System.out::println);
        }

        // Пример 7: Пакетная валидация
        System.out.println("\\n=== Пример 7: Пакетная Валидация ===");
        List<Product> products = Arrays.asList(
            new Product("Книга", 29.99, 100, "Книги"),
            new Product("", 0, -1, ""),
            new Product("Телефон", 799.99, 200, "Электроника")
        );

        for (int i = 0; i < products.size(); i++) {
            ValidationResult result = validator.validate(products.get(i));
            System.out.println("Продукт " + (i + 1) + ": " +
                (result.isValid() ? "ВАЛИДЕН" : "НЕВАЛИДЕН (" +
                result.getErrorCount() + " ошибок)"));
        }
    }
}`,
            description: `Реализуйте комплексную валидацию входных данных с использованием ручных проверок и концепций Bean Validation.

Требования:
1. Создайте класс Product с правилами валидации
2. Реализуйте ручную валидацию с детальными сообщениями об ошибках
3. Создайте класс ValidationResult для сбора множественных ошибок
4. Добавьте валидацию на уровне полей и кросс-полевую валидацию
5. Продемонстрируйте валидацию в системе регистрации продуктов

Пример:
\`\`\`java
Product product = new Product("", -10.0, 0);
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(System.out::println);
}
\`\`\``,
            hint1: `Создайте класс ValidationResult, который хранит список объектов ValidationError. Каждая ошибка должна содержать имя поля, сообщение и отклоненное значение для детального отчета.`,
            hint2: `В валидаторе выполняйте множественные проверки для каждого поля и добавляйте ошибки в результат. Сначала проверяйте значения null, затем проверяйте ограничения, такие как длина, диапазон и формат. Не возвращайтесь раньше - собирайте все ошибки.`,
            whyItMatters: `Валидация входных данных критична для целостности данных и безопасности. Сбор всех ошибок валидации сразу обеспечивает лучший пользовательский опыт, чем отказ на первой ошибке.

**Продакшен паттерн:**
\`\`\`java
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(error ->
        System.out.println("Поле '" + error.getField() +
                         "': " + error.getMessage()));
}
\`\`\`

**Практические преимущества:**
- Сбор всех ошибок за один проход для лучшего UX
- Основа для Bean Validation (JSR 303/380)
- Критично для целостности данных и безопасности`
        },
        uz: {
            title: `Kiruvchi Ma'lumotlarni Validatsiya Qilish`,
            solutionCode: `import java.util.*;

// Bitta validatsiya xatosini ifodalaydi
class ValidationError {
    private final String field;
    private final String message;
    private final Object rejectedValue;

    public ValidationError(String field, String message, Object rejectedValue) {
        this.field = field;
        this.message = message;
        this.rejectedValue = rejectedValue;
    }

    public String getField() { return field; }
    public String getMessage() { return message; }
    public Object getRejectedValue() { return rejectedValue; }

    @Override
    public String toString() {
        return String.format("Maydon '%s': %s (rad etilgan qiymat: %s)",
            field, message, rejectedValue);
    }
}

// Validatsiya xatolarini to'playdi
class ValidationResult {
    private final List<ValidationError> errors;

    public ValidationResult() {
        this.errors = new ArrayList<>();
    }

    // Validatsiya xatosini qo'shish
    public void addError(String field, String message, Object rejectedValue) {
        errors.add(new ValidationError(field, message, rejectedValue));
    }

    // Validatsiya muvaffaqiyatligini tekshirish
    public boolean isValid() {
        return errors.isEmpty();
    }

    // Barcha xatolarni olish
    public List<ValidationError> getErrors() {
        return Collections.unmodifiableList(errors);
    }

    // Muayyan maydon uchun xatolarni olish
    public List<ValidationError> getFieldErrors(String field) {
        List<ValidationError> fieldErrors = new ArrayList<>();
        for (ValidationError error : errors) {
            if (error.getField().equals(field)) {
                fieldErrors.add(error);
            }
        }
        return fieldErrors;
    }

    // Xatolar sonini olish
    public int getErrorCount() {
        return errors.size();
    }

    @Override
    public String toString() {
        if (isValid()) {
            return "Validatsiya o'tdi";
        }
        StringBuilder sb = new StringBuilder("Validatsiya muvaffaqiyatsiz:\\n");
        for (ValidationError error : errors) {
            sb.append("  - ").append(error).append("\\n");
        }
        return sb.toString();
    }
}

class Product {
    private String name;
    private double price;
    private int quantity;
    private String category;

    public Product(String name, double price, int quantity) {
        this(name, price, quantity, "Umumiy");
    }

    public Product(String name, double price, int quantity, String category) {
        this.name = name;
        this.price = price;
        this.quantity = quantity;
        this.category = category;
    }

    // Getterlar
    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getQuantity() { return quantity; }
    public String getCategory() { return category; }

    @Override
    public String toString() {
        return String.format("Product{name='%s', price=%.2f, quantity=%d, category='%s'}",
            name, price, quantity, category);
    }
}

class ProductValidator {
    // Mahsulotni kompleks tekshiruvlar bilan validatsiya qilish
    public ValidationResult validate(Product product) {
        ValidationResult result = new ValidationResult();

        // Nomni validatsiya qilish
        validateName(product.getName(), result);

        // Narxni validatsiya qilish
        validatePrice(product.getPrice(), result);

        // Sonni validatsiya qilish
        validateQuantity(product.getQuantity(), result);

        // Kategoriyani validatsiya qilish
        validateCategory(product.getCategory(), result);

        // Kross-maydon validatsiyasi
        validatePriceQuantityConsistency(product, result);

        return result;
    }

    // Mahsulot nomini validatsiya qilish
    private void validateName(String name, ValidationResult result) {
        if (name == null || name.trim().isEmpty()) {
            result.addError("name", "Mahsulot nomi majburiy", name);
            return;
        }

        if (name.length() < 3) {
            result.addError("name",
                "Mahsulot nomi kamida 3 ta belgidan iborat bo'lishi kerak", name);
        }

        if (name.length() > 100) {
            result.addError("name",
                "Mahsulot nomi 100 ta belgidan oshmasligi kerak", name);
        }

        // Noto'g'ri belgilarni tekshirish
        if (!name.matches("[a-zA-Z0-9\\\\s\\\\-]+")) {
            result.addError("name",
                "Mahsulot nomi noto'g'ri belgilarni o'z ichiga oladi", name);
        }
    }

    // Narxni validatsiya qilish
    private void validatePrice(double price, ValidationResult result) {
        if (price < 0) {
            result.addError("price", "Narx manfiy bo'lishi mumkin emas", price);
        }

        if (price == 0) {
            result.addError("price", "Narx noldan katta bo'lishi kerak", price);
        }

        if (price > 1_000_000) {
            result.addError("price",
                "Narx maksimal ruxsat etilgan qiymatdan oshadi", price);
        }
    }

    // Sonni validatsiya qilish
    private void validateQuantity(int quantity, ValidationResult result) {
        if (quantity < 0) {
            result.addError("quantity", "Son manfiy bo'lishi mumkin emas", quantity);
        }

        if (quantity > 10_000) {
            result.addError("quantity",
                "Son maksimal zaxira limitidan oshadi", quantity);
        }
    }

    // Kategoriyani validatsiya qilish
    private void validateCategory(String category, ValidationResult result) {
        if (category == null || category.trim().isEmpty()) {
            result.addError("category", "Kategoriya majburiy", category);
            return;
        }

        List<String> validCategories = Arrays.asList(
            "Elektronika", "Kiyim", "Oziq-ovqat", "Kitoblar", "Umumiy"
        );

        if (!validCategories.contains(category)) {
            result.addError("category",
                "Noto'g'ri kategoriya. Quyidagilardan biri bo'lishi kerak: " + validCategories, category);
        }
    }

    // Kross-maydon validatsiyasi
    private void validatePriceQuantityConsistency(Product product,
                                                   ValidationResult result) {
        // Biznes qoidasi: qimmat mahsulotlar cheklangan songa ega bo'lishi kerak
        if (product.getPrice() > 10000 && product.getQuantity() > 100) {
            result.addError("quantity",
                "Yuqori qiymatli mahsulotlar (narx > 10,000) 100 dan ortiq songa ega bo'lolmaydi",
                product.getQuantity());
        }
    }
}

public class ValidationDemo {
    public static void main(String[] args) {
        ProductValidator validator = new ProductValidator();

        // Misol 1: To'g'ri mahsulot
        System.out.println("=== Misol 1: To'g'ri Mahsulot ===");
        Product validProduct = new Product("Noutbuk Kompyuter", 999.99, 50, "Elektronika");
        ValidationResult result1 = validator.validate(validProduct);
        System.out.println(result1);
        System.out.println("Mahsulot: " + validProduct);

        // Misol 2: Ko'plab validatsiya xatolari
        System.out.println("\\n=== Misol 2: Noto'g'ri Mahsulot (Ko'plab Xatolar) ===");
        Product invalidProduct = new Product("", -10.0, -5);
        ValidationResult result2 = validator.validate(invalidProduct);
        System.out.println("To'g'rimi: " + result2.isValid());
        System.out.println("Xatolar soni: " + result2.getErrorCount());
        System.out.println("Xatolar:");
        result2.getErrors().forEach(error ->
            System.out.println("  - " + error));

        // Misol 3: Nom validatsiyasi
        System.out.println("\\n=== Misol 3: Nom Validatsiyasi ===");
        Product shortName = new Product("AB", 100.0, 10, "Kitoblar");
        ValidationResult result3 = validator.validate(shortName);
        if (!result3.isValid()) {
            List<ValidationError> nameErrors = result3.getFieldErrors("name");
            System.out.println("Nom validatsiya xatolari:");
            nameErrors.forEach(error -> System.out.println("  - " + error.getMessage()));
        }

        // Misol 4: Narx validatsiyasi
        System.out.println("\\n=== Misol 4: Narx Validatsiyasi ===");
        Product expensiveProduct = new Product("Hashamatli Avtomobil", 2_000_000, 5, "Umumiy");
        ValidationResult result4 = validator.validate(expensiveProduct);
        if (!result4.isValid()) {
            System.out.println("Validatsiya xatolari:");
            result4.getErrors().forEach(System.out::println);
        }

        // Misol 5: Kategoriya validatsiyasi
        System.out.println("\\n=== Misol 5: Kategoriya Validatsiyasi ===");
        Product invalidCategory = new Product("Vidjet", 50.0, 100, "Noto'g'riKategoriya");
        ValidationResult result5 = validator.validate(invalidCategory);
        if (!result5.isValid()) {
            List<ValidationError> categoryErrors = result5.getFieldErrors("category");
            categoryErrors.forEach(error ->
                System.out.println("Kategoriya xatosi: " + error.getMessage()));
        }

        // Misol 6: Kross-maydon validatsiyasi
        System.out.println("\\n=== Misol 6: Kross-maydon Validatsiyasi ===");
        Product expensiveWithHighQty = new Product("Olmos Uzuk", 50000, 500, "Umumiy");
        ValidationResult result6 = validator.validate(expensiveWithHighQty);
        if (!result6.isValid()) {
            System.out.println("Biznes qoidalari buzilishlari:");
            result6.getErrors().forEach(System.out::println);
        }

        // Misol 7: Paketli validatsiya
        System.out.println("\\n=== Misol 7: Paketli Validatsiya ===");
        List<Product> products = Arrays.asList(
            new Product("Kitob", 29.99, 100, "Kitoblar"),
            new Product("", 0, -1, ""),
            new Product("Telefon", 799.99, 200, "Elektronika")
        );

        for (int i = 0; i < products.size(); i++) {
            ValidationResult result = validator.validate(products.get(i));
            System.out.println("Mahsulot " + (i + 1) + ": " +
                (result.isValid() ? "TO'G'RI" : "NOTO'G'RI (" +
                result.getErrorCount() + " xatolar)"));
        }
    }
}`,
            description: `Qo'lda tekshiruvlar va Bean Validation kontseptsiyalaridan foydalanib, kompleks kiruvchi ma'lumotlarni validatsiya qilishni amalga oshiring.

Talablar:
1. Validatsiya qoidalari bilan Product klassini yarating
2. Batafsil xato xabarlari bilan qo'lda validatsiya qilishni amalga oshiring
3. Ko'plab xatolarni to'plash uchun ValidationResult klassini yarating
4. Maydon darajasidagi va kross-maydon validatsiyasini qo'shing
5. Mahsulotlarni ro'yxatga olish tizimida validatsiyani namoyish eting

Misol:
\`\`\`java
Product product = new Product("", -10.0, 0);
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(System.out::println);
}
\`\`\``,
            hint1: `ValidationError obyektlari ro'yxatini saqlaydigan ValidationResult klassini yarating. Har bir xato batafsil hisobot uchun maydon nomi, xabar va rad etilgan qiymatni o'z ichiga olishi kerak.`,
            hint2: `Validatorda har bir maydon uchun ko'plab tekshiruvlarni bajaring va xatolarni natijaga qo'shing. Avval null qiymatlarini tekshiring, so'ngra uzunlik, diapazon va format kabi cheklovlarni tekshiring. Erta qaytmang - barcha xatolarni to'plang.`,
            whyItMatters: `Kiruvchi ma'lumotlarni validatsiya qilish ma'lumotlar yaxlitligi va xavfsizligi uchun juda muhim. Barcha validatsiya xatolarini bir vaqtning o'zida to'plash birinchi xatoda ishlamay qolishdan ko'ra yaxshiroq foydalanuvchi tajribasini ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
ValidationResult result = validator.validate(product);
if (!result.isValid()) {
    result.getErrors().forEach(error ->
        System.out.println("Maydon '" + error.getField() +
                         "': " + error.getMessage()));
}
\`\`\`

**Amaliy foydalari:**
- Yaxshiroq UX uchun barcha xatolarni bir o'tishda to'plash
- Bean Validation (JSR 303/380) uchun asos
- Ma'lumotlar yaxlitligi va xavfsizligi uchun muhim`
        }
    }
};

export default task;
