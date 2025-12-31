import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-template-method',
	title: 'Template Method Pattern',
	difficulty: 'easy',
	tags: ['java', 'design-patterns', 'behavioral', 'template-method'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Template Method Pattern

The **Template Method Pattern** defines the skeleton of an algorithm in a base class, letting subclasses override specific steps without changing the algorithm's structure. It's "the Hollywood Principle" - don't call us, we'll call you.

---

### Key Components

| Component | Role |
|-----------|------|
| **AbstractClass** | Defines template method and abstract steps |
| **ConcreteClass** | Implements the abstract steps |
| **Template Method** | Final method that defines the algorithm sequence |
| **Primitive Operations** | Abstract methods customized by subclasses |

---

### Your Task

Implement a **Data Processing Pipeline** using the Template Method pattern:

1. **DataProcessor** (AbstractClass): Template with read → process → write steps
2. **CSVProcessor** (ConcreteClass): CSV-specific implementation
3. **JSONProcessor** (ConcreteClass): JSON-specific implementation

---

### Example Usage

\`\`\`java
DataProcessor csvProcessor = new CSVProcessor();	// create CSV processor
List<String> csvResults = csvProcessor.process();	// execute template method

// Results: ["Reading CSV data", "Processing CSV rows", "Writing CSV output"]
System.out.println(csvResults);	// template method calls all steps in order

DataProcessor jsonProcessor = new JSONProcessor();	// create JSON processor
List<String> jsonResults = jsonProcessor.process();	// same template, different steps

// Results: ["Reading JSON data", "Processing JSON objects", "Writing JSON output"]
System.out.println(jsonResults);	// different implementation, same algorithm structure
\`\`\`

---

### Key Insight

> The template method is \`final\` to prevent subclasses from changing the algorithm structure. Subclasses can only customize the individual steps (primitive operations), not the order in which they're called.`,
	initialCode: `import java.util.*;

abstract class DataProcessor {
    public final List<String> process() {
    }

    protected abstract String readData();
    protected abstract String processData();
    protected abstract String writeData();
}

class CSVProcessor extends DataProcessor {
    @Override
    protected String readData() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    protected String processData() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    protected String writeData() {
        throw new UnsupportedOperationException("TODO");
    }
}

class JSONProcessor extends DataProcessor {
    @Override
    protected String readData() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    protected String processData() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    protected String writeData() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `import java.util.*;	// import ArrayList, List for collecting results

abstract class DataProcessor {	// AbstractClass - defines algorithm skeleton
    // Template method (final to prevent override)
    public final List<String> process() {	// FINAL - subclasses cannot change algorithm structure
        List<String> results = new ArrayList<>();	// collect results from each step
        results.add(readData());	// Step 1: read data (abstract)
        results.add(processData());	// Step 2: process data (abstract)
        results.add(writeData());	// Step 3: write data (abstract)
        return results;	// return all results
    }

    protected abstract String readData();	// primitive operation - subclass must implement
    protected abstract String processData();	// primitive operation - subclass must implement
    protected abstract String writeData();	// primitive operation - subclass must implement
}

class CSVProcessor extends DataProcessor {	// ConcreteClass - CSV implementation
    @Override
    protected String readData() {	// CSV-specific read implementation
        return "Reading CSV data";	// read from CSV file
    }

    @Override
    protected String processData() {	// CSV-specific process implementation
        return "Processing CSV rows";	// parse and process CSV rows
    }

    @Override
    protected String writeData() {	// CSV-specific write implementation
        return "Writing CSV output";	// write processed CSV data
    }
}

class JSONProcessor extends DataProcessor {	// ConcreteClass - JSON implementation
    @Override
    protected String readData() {	// JSON-specific read implementation
        return "Reading JSON data";	// read from JSON file
    }

    @Override
    protected String processData() {	// JSON-specific process implementation
        return "Processing JSON objects";	// parse and process JSON objects
    }

    @Override
    protected String writeData() {	// JSON-specific write implementation
        return "Writing JSON output";	// write processed JSON data
    }
}`,
	hint1: `## Hint 1: Understanding Template Method Structure

The template method defines the algorithm skeleton, while subclasses provide specific implementations:

\`\`\`java
// Base class - DO NOT MODIFY
abstract class DataProcessor {
    public final List<String> process() {	// template method - FINAL!
        List<String> results = new ArrayList<>();	// create result collector
        results.add(readData());	// call abstract step 1
        results.add(processData());	// call abstract step 2
        results.add(writeData());	// call abstract step 3
        return results;	// return collected results
    }
    // Subclasses implement these abstract methods
    protected abstract String readData();
    protected abstract String processData();
    protected abstract String writeData();
}
\`\`\`

Each subclass provides its own implementation of the abstract methods without changing the order.`,
	hint2: `## Hint 2: Implementing Concrete Classes

Each concrete class implements all abstract methods with format-specific logic:

\`\`\`java
class CSVProcessor extends DataProcessor {	// CSV implementation
    @Override
    protected String readData() {	// step 1: read
        return "Reading CSV data";	// CSV-specific message
    }

    @Override
    protected String processData() {	// step 2: process
        return "Processing CSV rows";	// CSV-specific message
    }

    @Override
    protected String writeData() {	// step 3: write
        return "Writing CSV output";	// CSV-specific message
    }
}

class JSONProcessor extends DataProcessor {	// JSON implementation
    @Override
    protected String readData() {	// step 1: read
        return "Reading JSON data";	// JSON-specific message
    }
    // ... same pattern for other methods
}
\`\`\`

Return descriptive strings indicating the format being processed at each step.`,
	whyItMatters: `## Why Template Method Pattern Matters

### The Problem: Code Duplication in Similar Algorithms

Without Template Method, similar algorithms lead to duplicated control flow:

\`\`\`java
// ❌ Without Template Method - duplicated algorithm structure
class CSVProcessor {	// CSV processing - duplicated structure
    public List<String> process() {	// method with same structure
        List<String> results = new ArrayList<>();	// same collection logic
        // Step 1
        results.add("Reading CSV data");	// CSV-specific
        // Step 2
        results.add("Processing CSV rows");	// CSV-specific
        // Step 3
        results.add("Writing CSV output");	// CSV-specific
        return results;	// same return
    }
}

class JSONProcessor {	// JSON processing - duplicated structure
    public List<String> process() {	// SAME method structure!
        List<String> results = new ArrayList<>();	// SAME collection logic!
        // Step 1
        results.add("Reading JSON data");	// JSON-specific
        // Step 2
        results.add("Processing JSON objects");	// JSON-specific
        // Step 3
        results.add("Writing JSON output");	// JSON-specific
        return results;	// SAME return!
    }
}

class XMLProcessor {	// XML processing - more duplication
    public List<String> process() {	// SAME structure again!
        List<String> results = new ArrayList<>();	// SAME!
        results.add("Reading XML data");	// XML-specific
        results.add("Processing XML nodes");	// XML-specific
        results.add("Writing XML output");	// XML-specific
        return results;	// SAME!
    }
}

// Problems:
// 1. Algorithm structure duplicated in every class
// 2. If algorithm changes, must update ALL classes
// 3. Easy to introduce bugs in one class
// 4. No enforcement that all processors follow same structure
\`\`\`

\`\`\`java
// ✅ With Template Method - algorithm defined once
abstract class DataProcessor {	// define algorithm skeleton once
    public final List<String> process() {	// template method - FINAL
        List<String> results = new ArrayList<>();	// algorithm structure defined once
        results.add(readData());	// step 1 - delegated to subclass
        results.add(processData());	// step 2 - delegated to subclass
        results.add(writeData());	// step 3 - delegated to subclass
        return results;	// consistent return
    }

    protected abstract String readData();	// subclass must implement
    protected abstract String processData();	// subclass must implement
    protected abstract String writeData();	// subclass must implement
}

class CSVProcessor extends DataProcessor {	// only implements steps
    @Override
    protected String readData() { return "Reading CSV data"; }	// CSV-specific
    @Override
    protected String processData() { return "Processing CSV rows"; }	// CSV-specific
    @Override
    protected String writeData() { return "Writing CSV output"; }	// CSV-specific
}

class JSONProcessor extends DataProcessor {	// only implements steps
    @Override
    protected String readData() { return "Reading JSON data"; }	// JSON-specific
    @Override
    protected String processData() { return "Processing JSON objects"; }	// JSON-specific
    @Override
    protected String writeData() { return "Writing JSON output"; }	// JSON-specific
}

// Benefits:
// 1. Algorithm defined once - DRY principle
// 2. Changes to algorithm in one place
// 3. Consistent behavior guaranteed
// 4. Subclasses only customize what varies
\`\`\`

---

### Real-World Applications

| Application | Abstract Class | Template Method | Steps |
|-------------|----------------|-----------------|-------|
| **HttpServlet** | HttpServlet | service() | doGet(), doPost(), doPut() |
| **JUnit** | TestCase | runTest() | setUp(), test*, tearDown() |
| **Spring JdbcTemplate** | JdbcTemplate | execute() | open, query, map, close |
| **Build Systems** | BuildTask | build() | compile, test, package |
| **Document Export** | Exporter | export() | header, body, footer |

---

### Production Pattern: Test Framework with Lifecycle Hooks

\`\`\`java
import java.util.*;	// import utilities for collections

// AbstractClass - Test Framework base
abstract class TestCase {	// defines test lifecycle template
    private List<String> log = new ArrayList<>();	// execution log
    private boolean passed = true;	// test result
    private String failureMessage = null;	// failure details

    // Template method - FINAL, cannot be overridden
    public final TestResult run() {	// template method defines execution order
        log.clear();	// clear previous log
        passed = true;	// reset state
        failureMessage = null;	// reset failure

        try {
            log.add("Setting up test: " + getTestName());	// log setup
            setUp();	// Step 1: setup (hook)

            log.add("Running test: " + getTestName());	// log execution
            runTest();	// Step 2: run test (abstract)

            log.add("Verifying results");	// log verification
            verify();	// Step 3: verify (hook with default)

        } catch (AssertionError e) {	// catch test failures
            passed = false;	// mark as failed
            failureMessage = e.getMessage();	// store failure message
            log.add("FAILED: " + failureMessage);	// log failure
        } catch (Exception e) {	// catch unexpected errors
            passed = false;	// mark as failed
            failureMessage = "Exception: " + e.getMessage();	// store error
            log.add("ERROR: " + failureMessage);	// log error
        } finally {
            log.add("Tearing down test");	// log teardown
            tearDown();	// Step 4: cleanup (hook)
        }

        return new TestResult(getTestName(), passed, failureMessage, log);	// return result
    }

    // Abstract method - subclasses MUST implement
    protected abstract void runTest() throws Exception;	// the actual test logic

    // Abstract method - test name
    protected abstract String getTestName();	// identify the test

    // Hook methods - default implementations, can be overridden
    protected void setUp() { }	// optional setup, default does nothing

    protected void tearDown() { }	// optional cleanup, default does nothing

    protected void verify() { }	// optional verification, default does nothing

    // Assertion helpers
    protected void assertEquals(Object expected, Object actual) {	// equality assertion
        if (!Objects.equals(expected, actual)) {	// compare objects
            throw new AssertionError("Expected " + expected + " but got " + actual);	// fail if different
        }
    }

    protected void assertTrue(boolean condition, String message) {	// boolean assertion
        if (!condition) {	// check condition
            throw new AssertionError(message);	// fail if false
        }
    }

    protected void assertNotNull(Object obj, String message) {	// null check assertion
        if (obj == null) {	// check for null
            throw new AssertionError(message);	// fail if null
        }
    }
}

// Value object for test results
class TestResult {	// immutable test result
    private final String testName;	// name of test
    private final boolean passed;	// pass/fail status
    private final String failureMessage;	// failure details (null if passed)
    private final List<String> executionLog;	// execution history

    public TestResult(String testName, boolean passed,
                      String failureMessage, List<String> log) {	// constructor
        this.testName = testName;	// store test name
        this.passed = passed;	// store result
        this.failureMessage = failureMessage;	// store failure (may be null)
        this.executionLog = new ArrayList<>(log);	// defensive copy of log
    }

    public String getTestName() { return testName; }	// get test name
    public boolean isPassed() { return passed; }	// get pass status
    public String getFailureMessage() { return failureMessage; }	// get failure message
    public List<String> getExecutionLog() { return new ArrayList<>(executionLog); }	// get log copy

    @Override
    public String toString() {	// string representation
        return testName + ": " + (passed ? "PASSED" : "FAILED - " + failureMessage);	// summary
    }
}

// ConcreteClass - User service test
class UserServiceTest extends TestCase {	// specific test implementation
    private UserService userService;	// service under test
    private User testUser;	// test data

    @Override
    protected String getTestName() {	// identify this test
        return "UserServiceTest.testCreateUser";	// test name
    }

    @Override
    protected void setUp() {	// prepare test environment
        userService = new UserService();	// create service
        testUser = new User("john@example.com", "John Doe");	// create test data
    }

    @Override
    protected void runTest() throws Exception {	// the actual test
        User created = userService.create(testUser);	// call method under test
        assertNotNull(created, "Created user should not be null");	// verify not null
        assertEquals(testUser.getEmail(), created.getEmail());	// verify email matches
    }

    @Override
    protected void verify() {	// additional verification
        assertTrue(userService.exists(testUser.getEmail()),
            "User should exist after creation");	// verify persistence
    }

    @Override
    protected void tearDown() {	// cleanup
        if (userService != null && testUser != null) {	// if resources exist
            userService.delete(testUser.getEmail());	// clean up test data
        }
    }
}

// ConcreteClass - Database connection test
class DatabaseConnectionTest extends TestCase {	// another test implementation
    private Database db;	// database connection

    @Override
    protected String getTestName() {	// identify this test
        return "DatabaseConnectionTest.testConnection";	// test name
    }

    @Override
    protected void setUp() {	// prepare database
        db = Database.getInstance();	// get database instance
        db.connect();	// establish connection
    }

    @Override
    protected void runTest() throws Exception {	// test database operations
        assertTrue(db.isConnected(), "Database should be connected");	// verify connection
        db.execute("SELECT 1");	// execute simple query
    }

    @Override
    protected void tearDown() {	// cleanup connection
        if (db != null) {	// if database exists
            db.disconnect();	// close connection
        }
    }
}

// Test Runner - executes multiple tests
class TestRunner {	// orchestrates test execution
    private List<TestResult> results = new ArrayList<>();	// collected results

    public void runAll(List<TestCase> tests) {	// run all tests
        results.clear();	// clear previous results
        for (TestCase test : tests) {	// iterate tests
            TestResult result = test.run();	// execute template method
            results.add(result);	// collect result
            System.out.println(result);	// print result
        }
    }

    public int getPassedCount() {	// count passed tests
        return (int) results.stream().filter(TestResult::isPassed).count();	// filter and count
    }

    public int getFailedCount() {	// count failed tests
        return results.size() - getPassedCount();	// total minus passed
    }

    public void printSummary() {	// print test summary
        System.out.println("\\n=== Test Summary ===");	// header
        System.out.println("Total: " + results.size());	// total count
        System.out.println("Passed: " + getPassedCount());	// passed count
        System.out.println("Failed: " + getFailedCount());	// failed count
    }
}

// Usage:
TestRunner runner = new TestRunner();	// create test runner

List<TestCase> tests = Arrays.asList(	// create test suite
    new UserServiceTest(),	// user service test
    new DatabaseConnectionTest()	// database connection test
);

runner.runAll(tests);	// execute all tests using template method
runner.printSummary();	// print results

// Output:
// Setting up test: UserServiceTest.testCreateUser
// Running test: UserServiceTest.testCreateUser
// Verifying results
// Tearing down test
// UserServiceTest.testCreateUser: PASSED
// ...
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Template method not final | Subclasses can break algorithm | Mark template method as \`final\` |
| Too many abstract methods | Hard to extend | Use hooks with default implementations |
| Calling abstract methods in constructor | Subclass not initialized | Use template method, not constructor |
| No hooks | Rigid, not flexible | Add optional hooks with default behavior |
| Steps too fine-grained | Too many methods to implement | Balance abstraction vs. practicality |`,
	order: 8,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;
import java.util.List;

// Test1: CSVProcessor process returns 3 steps
class Test1 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        List<String> results = csv.process();
        assertEquals(3, results.size());
    }
}

// Test2: CSVProcessor readData step
class Test2 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        List<String> results = csv.process();
        assertEquals("Reading CSV data", results.get(0));
    }
}

// Test3: CSVProcessor processData step
class Test3 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        List<String> results = csv.process();
        assertEquals("Processing CSV rows", results.get(1));
    }
}

// Test4: CSVProcessor writeData step
class Test4 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        List<String> results = csv.process();
        assertEquals("Writing CSV output", results.get(2));
    }
}

// Test5: JSONProcessor process returns 3 steps
class Test5 {
    @Test
    public void test() {
        DataProcessor json = new JSONProcessor();
        List<String> results = json.process();
        assertEquals(3, results.size());
    }
}

// Test6: JSONProcessor readData step
class Test6 {
    @Test
    public void test() {
        DataProcessor json = new JSONProcessor();
        List<String> results = json.process();
        assertEquals("Reading JSON data", results.get(0));
    }
}

// Test7: JSONProcessor processData step
class Test7 {
    @Test
    public void test() {
        DataProcessor json = new JSONProcessor();
        List<String> results = json.process();
        assertEquals("Processing JSON objects", results.get(1));
    }
}

// Test8: JSONProcessor writeData step
class Test8 {
    @Test
    public void test() {
        DataProcessor json = new JSONProcessor();
        List<String> results = json.process();
        assertEquals("Writing JSON output", results.get(2));
    }
}

// Test9: Steps execute in correct order
class Test9 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        List<String> results = csv.process();
        assertTrue(results.get(0).contains("Reading"));
        assertTrue(results.get(1).contains("Processing"));
        assertTrue(results.get(2).contains("Writing"));
    }
}

// Test10: Different processors return different results
class Test10 {
    @Test
    public void test() {
        DataProcessor csv = new CSVProcessor();
        DataProcessor json = new JSONProcessor();
        List<String> csvResults = csv.process();
        List<String> jsonResults = json.process();
        assertNotEquals(csvResults.get(0), jsonResults.get(0));
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Template Method',
			description: `## Паттерн Template Method

Паттерн **Template Method** определяет скелет алгоритма в базовом классе, позволяя подклассам переопределять конкретные шаги без изменения структуры алгоритма. Это "Принцип Голливуда" - не звоните нам, мы сами вам позвоним.

---

### Ключевые компоненты

| Компонент | Роль |
|-----------|------|
| **AbstractClass** | Определяет шаблонный метод и абстрактные шаги |
| **ConcreteClass** | Реализует абстрактные шаги |
| **Template Method** | Final метод, определяющий последовательность алгоритма |
| **Primitive Operations** | Абстрактные методы, настраиваемые подклассами |

---

### Ваша задача

Реализуйте **Конвейер обработки данных** используя паттерн Template Method:

1. **DataProcessor** (AbstractClass): Шаблон с шагами чтение → обработка → запись
2. **CSVProcessor** (ConcreteClass): CSV-специфичная реализация
3. **JSONProcessor** (ConcreteClass): JSON-специфичная реализация

---

### Пример использования

\`\`\`java
DataProcessor csvProcessor = new CSVProcessor();	// создаём CSV процессор
List<String> csvResults = csvProcessor.process();	// выполняем шаблонный метод

// Результаты: ["Reading CSV data", "Processing CSV rows", "Writing CSV output"]
System.out.println(csvResults);	// шаблонный метод вызывает все шаги по порядку

DataProcessor jsonProcessor = new JSONProcessor();	// создаём JSON процессор
List<String> jsonResults = jsonProcessor.process();	// тот же шаблон, другие шаги

// Результаты: ["Reading JSON data", "Processing JSON objects", "Writing JSON output"]
System.out.println(jsonResults);	// разная реализация, та же структура алгоритма
\`\`\`

---

### Ключевая идея

> Шаблонный метод объявлен как \`final\`, чтобы подклассы не могли изменить структуру алгоритма. Подклассы могут только настраивать отдельные шаги (примитивные операции), но не порядок их вызова.`,
			hint1: `## Подсказка 1: Понимание структуры Template Method

Шаблонный метод определяет скелет алгоритма, а подклассы предоставляют конкретные реализации:

\`\`\`java
// Базовый класс - НЕ ИЗМЕНЯТЬ
abstract class DataProcessor {
    public final List<String> process() {	// шаблонный метод - FINAL!
        List<String> results = new ArrayList<>();	// создать сборщик результатов
        results.add(readData());	// вызвать абстрактный шаг 1
        results.add(processData());	// вызвать абстрактный шаг 2
        results.add(writeData());	// вызвать абстрактный шаг 3
        return results;	// вернуть собранные результаты
    }
    // Подклассы реализуют эти абстрактные методы
    protected abstract String readData();
    protected abstract String processData();
    protected abstract String writeData();
}
\`\`\`

Каждый подкласс предоставляет свою реализацию абстрактных методов без изменения порядка.`,
			hint2: `## Подсказка 2: Реализация конкретных классов

Каждый конкретный класс реализует все абстрактные методы с логикой, специфичной для формата:

\`\`\`java
class CSVProcessor extends DataProcessor {	// CSV реализация
    @Override
    protected String readData() {	// шаг 1: чтение
        return "Reading CSV data";	// CSV-специфичное сообщение
    }

    @Override
    protected String processData() {	// шаг 2: обработка
        return "Processing CSV rows";	// CSV-специфичное сообщение
    }

    @Override
    protected String writeData() {	// шаг 3: запись
        return "Writing CSV output";	// CSV-специфичное сообщение
    }
}

class JSONProcessor extends DataProcessor {	// JSON реализация
    @Override
    protected String readData() {	// шаг 1: чтение
        return "Reading JSON data";	// JSON-специфичное сообщение
    }
    // ... тот же паттерн для других методов
}
\`\`\`

Возвращайте описательные строки, указывающие обрабатываемый формат на каждом шаге.`,
			whyItMatters: `## Почему паттерн Template Method важен

### Проблема: Дублирование кода в похожих алгоритмах

Без Template Method похожие алгоритмы ведут к дублированию потока управления:

\`\`\`java
// ❌ Без Template Method - дублированная структура алгоритма
class CSVProcessor {	// обработка CSV - дублированная структура
    public List<String> process() {	// метод с той же структурой
        List<String> results = new ArrayList<>();	// та же логика сбора
        // Шаг 1
        results.add("Reading CSV data");	// CSV-специфично
        // Шаг 2
        results.add("Processing CSV rows");	// CSV-специфично
        // Шаг 3
        results.add("Writing CSV output");	// CSV-специфично
        return results;	// тот же return
    }
}

class JSONProcessor {	// обработка JSON - дублированная структура
    public List<String> process() {	// ТА ЖЕ структура метода!
        List<String> results = new ArrayList<>();	// ТА ЖЕ логика сбора!
        // Шаг 1
        results.add("Reading JSON data");	// JSON-специфично
        // Шаг 2
        results.add("Processing JSON objects");	// JSON-специфично
        // Шаг 3
        results.add("Writing JSON output");	// JSON-специфично
        return results;	// ТОТ ЖЕ return!
    }
}

class XMLProcessor {	// обработка XML - ещё больше дублирования
    public List<String> process() {	// ТА ЖЕ структура снова!
        List<String> results = new ArrayList<>();	// ТО ЖЕ САМОЕ!
        results.add("Reading XML data");	// XML-специфично
        results.add("Processing XML nodes");	// XML-специфично
        results.add("Writing XML output");	// XML-специфично
        return results;	// ТО ЖЕ САМОЕ!
    }
}

// Проблемы:
// 1. Структура алгоритма дублируется в каждом классе
// 2. При изменении алгоритма нужно обновить ВСЕ классы
// 3. Легко внести ошибку в один из классов
// 4. Нет гарантии что все процессоры следуют одной структуре
\`\`\`

\`\`\`java
// ✅ С Template Method - алгоритм определён один раз
abstract class DataProcessor {	// определить скелет алгоритма один раз
    public final List<String> process() {	// шаблонный метод - FINAL
        List<String> results = new ArrayList<>();	// структура алгоритма определена один раз
        results.add(readData());	// шаг 1 - делегирован подклассу
        results.add(processData());	// шаг 2 - делегирован подклассу
        results.add(writeData());	// шаг 3 - делегирован подклассу
        return results;	// согласованный return
    }

    protected abstract String readData();	// подкласс должен реализовать
    protected abstract String processData();	// подкласс должен реализовать
    protected abstract String writeData();	// подкласс должен реализовать
}

class CSVProcessor extends DataProcessor {	// реализует только шаги
    @Override
    protected String readData() { return "Reading CSV data"; }	// CSV-специфично
    @Override
    protected String processData() { return "Processing CSV rows"; }	// CSV-специфично
    @Override
    protected String writeData() { return "Writing CSV output"; }	// CSV-специфично
}

class JSONProcessor extends DataProcessor {	// реализует только шаги
    @Override
    protected String readData() { return "Reading JSON data"; }	// JSON-специфично
    @Override
    protected String processData() { return "Processing JSON objects"; }	// JSON-специфично
    @Override
    protected String writeData() { return "Writing JSON output"; }	// JSON-специфично
}

// Преимущества:
// 1. Алгоритм определён один раз - принцип DRY
// 2. Изменения алгоритма в одном месте
// 3. Согласованное поведение гарантировано
// 4. Подклассы настраивают только то, что различается
\`\`\`

---

### Применение в реальном мире

| Применение | Абстрактный класс | Шаблонный метод | Шаги |
|------------|-------------------|-----------------|------|
| **HttpServlet** | HttpServlet | service() | doGet(), doPost(), doPut() |
| **JUnit** | TestCase | runTest() | setUp(), test*, tearDown() |
| **Spring JdbcTemplate** | JdbcTemplate | execute() | open, query, map, close |
| **Системы сборки** | BuildTask | build() | compile, test, package |
| **Экспорт документов** | Exporter | export() | header, body, footer |

---

### Продакшн паттерн: Тестовый фреймворк с хуками жизненного цикла

\`\`\`java
import java.util.*;	// импорт утилит для коллекций

// AbstractClass - базовый класс тестового фреймворка
abstract class TestCase {	// определяет шаблон жизненного цикла теста
    private List<String> log = new ArrayList<>();	// журнал выполнения
    private boolean passed = true;	// результат теста
    private String failureMessage = null;	// детали ошибки

    // Шаблонный метод - FINAL, не может быть переопределён
    public final TestResult run() {	// шаблонный метод определяет порядок выполнения
        log.clear();	// очистить предыдущий журнал
        passed = true;	// сбросить состояние
        failureMessage = null;	// сбросить ошибку

        try {
            log.add("Setting up test: " + getTestName());	// логировать настройку
            setUp();	// Шаг 1: настройка (хук)

            log.add("Running test: " + getTestName());	// логировать выполнение
            runTest();	// Шаг 2: запуск теста (абстрактный)

            log.add("Verifying results");	// логировать проверку
            verify();	// Шаг 3: проверка (хук с умолчанием)

        } catch (AssertionError e) {	// перехват ошибок теста
            passed = false;	// пометить как провал
            failureMessage = e.getMessage();	// сохранить сообщение об ошибке
            log.add("FAILED: " + failureMessage);	// логировать провал
        } catch (Exception e) {	// перехват неожиданных ошибок
            passed = false;	// пометить как провал
            failureMessage = "Exception: " + e.getMessage();	// сохранить ошибку
            log.add("ERROR: " + failureMessage);	// логировать ошибку
        } finally {
            log.add("Tearing down test");	// логировать очистку
            tearDown();	// Шаг 4: очистка (хук)
        }

        return new TestResult(getTestName(), passed, failureMessage, log);	// вернуть результат
    }

    // Абстрактный метод - подклассы ДОЛЖНЫ реализовать
    protected abstract void runTest() throws Exception;	// собственно логика теста

    // Абстрактный метод - имя теста
    protected abstract String getTestName();	// идентифицировать тест

    // Хук-методы - реализации по умолчанию, могут быть переопределены
    protected void setUp() { }	// опциональная настройка, по умолчанию ничего

    protected void tearDown() { }	// опциональная очистка, по умолчанию ничего

    protected void verify() { }	// опциональная проверка, по умолчанию ничего

    // Вспомогательные методы утверждений
    protected void assertEquals(Object expected, Object actual) {	// утверждение равенства
        if (!Objects.equals(expected, actual)) {	// сравнить объекты
            throw new AssertionError("Expected " + expected + " but got " + actual);	// провал если разные
        }
    }

    protected void assertTrue(boolean condition, String message) {	// булево утверждение
        if (!condition) {	// проверить условие
            throw new AssertionError(message);	// провал если false
        }
    }

    protected void assertNotNull(Object obj, String message) {	// утверждение не-null
        if (obj == null) {	// проверить на null
            throw new AssertionError(message);	// провал если null
        }
    }
}

// Value-объект для результатов теста
class TestResult {	// неизменяемый результат теста
    private final String testName;	// имя теста
    private final boolean passed;	// статус прохождения
    private final String failureMessage;	// детали ошибки (null если прошёл)
    private final List<String> executionLog;	// история выполнения

    public TestResult(String testName, boolean passed,
                      String failureMessage, List<String> log) {	// конструктор
        this.testName = testName;	// сохранить имя теста
        this.passed = passed;	// сохранить результат
        this.failureMessage = failureMessage;	// сохранить ошибку (может быть null)
        this.executionLog = new ArrayList<>(log);	// защитная копия журнала
    }

    public String getTestName() { return testName; }	// получить имя теста
    public boolean isPassed() { return passed; }	// получить статус прохождения
    public String getFailureMessage() { return failureMessage; }	// получить сообщение об ошибке
    public List<String> getExecutionLog() { return new ArrayList<>(executionLog); }	// получить копию журнала

    @Override
    public String toString() {	// строковое представление
        return testName + ": " + (passed ? "PASSED" : "FAILED - " + failureMessage);	// резюме
    }
}

// ConcreteClass - тест сервиса пользователей
class UserServiceTest extends TestCase {	// конкретная реализация теста
    private UserService userService;	// тестируемый сервис
    private User testUser;	// тестовые данные

    @Override
    protected String getTestName() {	// идентифицировать этот тест
        return "UserServiceTest.testCreateUser";	// имя теста
    }

    @Override
    protected void setUp() {	// подготовить тестовое окружение
        userService = new UserService();	// создать сервис
        testUser = new User("john@example.com", "John Doe");	// создать тестовые данные
    }

    @Override
    protected void runTest() throws Exception {	// собственно тест
        User created = userService.create(testUser);	// вызвать тестируемый метод
        assertNotNull(created, "Created user should not be null");	// проверить не null
        assertEquals(testUser.getEmail(), created.getEmail());	// проверить совпадение email
    }

    @Override
    protected void verify() {	// дополнительная проверка
        assertTrue(userService.exists(testUser.getEmail()),
            "User should exist after creation");	// проверить сохранение
    }

    @Override
    protected void tearDown() {	// очистка
        if (userService != null && testUser != null) {	// если ресурсы существуют
            userService.delete(testUser.getEmail());	// очистить тестовые данные
        }
    }
}

// ConcreteClass - тест подключения к базе данных
class DatabaseConnectionTest extends TestCase {	// другая реализация теста
    private Database db;	// подключение к БД

    @Override
    protected String getTestName() {	// идентифицировать этот тест
        return "DatabaseConnectionTest.testConnection";	// имя теста
    }

    @Override
    protected void setUp() {	// подготовить базу данных
        db = Database.getInstance();	// получить экземпляр БД
        db.connect();	// установить соединение
    }

    @Override
    protected void runTest() throws Exception {	// тестировать операции с БД
        assertTrue(db.isConnected(), "Database should be connected");	// проверить соединение
        db.execute("SELECT 1");	// выполнить простой запрос
    }

    @Override
    protected void tearDown() {	// очистить соединение
        if (db != null) {	// если БД существует
            db.disconnect();	// закрыть соединение
        }
    }
}

// Test Runner - выполняет множество тестов
class TestRunner {	// оркестрирует выполнение тестов
    private List<TestResult> results = new ArrayList<>();	// собранные результаты

    public void runAll(List<TestCase> tests) {	// запустить все тесты
        results.clear();	// очистить предыдущие результаты
        for (TestCase test : tests) {	// перебрать тесты
            TestResult result = test.run();	// выполнить шаблонный метод
            results.add(result);	// собрать результат
            System.out.println(result);	// вывести результат
        }
    }

    public int getPassedCount() {	// подсчитать пройденные тесты
        return (int) results.stream().filter(TestResult::isPassed).count();	// фильтр и подсчёт
    }

    public int getFailedCount() {	// подсчитать проваленные тесты
        return results.size() - getPassedCount();	// всего минус пройденные
    }

    public void printSummary() {	// вывести сводку тестов
        System.out.println("\\n=== Test Summary ===");	// заголовок
        System.out.println("Total: " + results.size());	// общее количество
        System.out.println("Passed: " + getPassedCount());	// количество пройденных
        System.out.println("Failed: " + getFailedCount());	// количество проваленных
    }
}

// Использование:
TestRunner runner = new TestRunner();	// создать запускатель тестов

List<TestCase> tests = Arrays.asList(	// создать набор тестов
    new UserServiceTest(),	// тест сервиса пользователей
    new DatabaseConnectionTest()	// тест подключения к БД
);

runner.runAll(tests);	// выполнить все тесты используя шаблонный метод
runner.printSummary();	// вывести результаты

// Вывод:
// Setting up test: UserServiceTest.testCreateUser
// Running test: UserServiceTest.testCreateUser
// Verifying results
// Tearing down test
// UserServiceTest.testCreateUser: PASSED
// ...
\`\`\`

---

### Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| Шаблонный метод не final | Подклассы могут сломать алгоритм | Пометить шаблонный метод как \`final\` |
| Слишком много абстрактных методов | Трудно расширять | Использовать хуки с реализациями по умолчанию |
| Вызов абстрактных методов в конструкторе | Подкласс не инициализирован | Использовать шаблонный метод, не конструктор |
| Нет хуков | Жёстко, не гибко | Добавить опциональные хуки с поведением по умолчанию |
| Слишком мелкие шаги | Слишком много методов для реализации | Баланс между абстракцией и практичностью |`
		},
		uz: {
			title: 'Template Method Pattern',
			description: `## Template Method Pattern

**Template Method Pattern** bazaviy klassda algoritmning skeletini belgilaydi, subklasslarga algoritm strukturasini o'zgartirmasdan aniq qadamlarni qayta belgilash imkonini beradi. Bu "Gollivud printsipi" - bizga qo'ng'iroq qilmang, biz sizga qo'ng'iroq qilamiz.

---

### Asosiy komponentlar

| Komponent | Vazifa |
|-----------|--------|
| **AbstractClass** | Shablon metodi va abstrakt qadamlarni belgilaydi |
| **ConcreteClass** | Abstrakt qadamlarni amalga oshiradi |
| **Template Method** | Algoritm ketma-ketligini belgilovchi final metod |
| **Primitive Operations** | Subklasslar tomonidan sozlanadigan abstrakt metodlar |

---

### Vazifangiz

Template Method patternidan foydalanib **Ma'lumotlarni qayta ishlash konveyeri** amalga oshiring:

1. **DataProcessor** (AbstractClass): O'qish → qayta ishlash → yozish qadamli shablon
2. **CSVProcessor** (ConcreteClass): CSV-maxsus realizatsiya
3. **JSONProcessor** (ConcreteClass): JSON-maxsus realizatsiya

---

### Foydalanish namunasi

\`\`\`java
DataProcessor csvProcessor = new CSVProcessor();	// CSV protsessor yaratish
List<String> csvResults = csvProcessor.process();	// shablon metodini bajarish

// Natijalar: ["Reading CSV data", "Processing CSV rows", "Writing CSV output"]
System.out.println(csvResults);	// shablon metodi barcha qadamlarni tartibda chaqiradi

DataProcessor jsonProcessor = new JSONProcessor();	// JSON protsessor yaratish
List<String> jsonResults = jsonProcessor.process();	// bir xil shablon, turli qadamlar

// Natijalar: ["Reading JSON data", "Processing JSON objects", "Writing JSON output"]
System.out.println(jsonResults);	// turli realizatsiya, bir xil algoritm strukturasi
\`\`\`

---

### Asosiy tushuncha

> Shablon metodi \`final\` deb e'lon qilinadi, shunda subklasslar algoritm strukturasini o'zgartira olmaydi. Subklasslar faqat alohida qadamlarni (primitiv operatsiyalarni) sozlashi mumkin, ularning chaqirilish tartibini emas.`,
			hint1: `## Maslahat 1: Template Method strukturasini tushunish

Shablon metodi algoritm skeletini belgilaydi, subklasslar esa aniq realizatsiyalarni ta'minlaydi:

\`\`\`java
// Bazaviy klass - O'ZGARTIRMANG
abstract class DataProcessor {
    public final List<String> process() {	// shablon metodi - FINAL!
        List<String> results = new ArrayList<>();	// natija yig'uvchi yaratish
        results.add(readData());	// abstrakt qadam 1 ni chaqirish
        results.add(processData());	// abstrakt qadam 2 ni chaqirish
        results.add(writeData());	// abstrakt qadam 3 ni chaqirish
        return results;	// yig'ilgan natijalarni qaytarish
    }
    // Subklasslar bu abstrakt metodlarni amalga oshiradi
    protected abstract String readData();
    protected abstract String processData();
    protected abstract String writeData();
}
\`\`\`

Har bir subklass tartibni o'zgartirmasdan abstrakt metodlarning o'z realizatsiyasini taqdim etadi.`,
			hint2: `## Maslahat 2: Aniq klasslarni amalga oshirish

Har bir aniq klass barcha abstrakt metodlarni format-maxsus mantiq bilan amalga oshiradi:

\`\`\`java
class CSVProcessor extends DataProcessor {	// CSV realizatsiyasi
    @Override
    protected String readData() {	// qadam 1: o'qish
        return "Reading CSV data";	// CSV-maxsus xabar
    }

    @Override
    protected String processData() {	// qadam 2: qayta ishlash
        return "Processing CSV rows";	// CSV-maxsus xabar
    }

    @Override
    protected String writeData() {	// qadam 3: yozish
        return "Writing CSV output";	// CSV-maxsus xabar
    }
}

class JSONProcessor extends DataProcessor {	// JSON realizatsiyasi
    @Override
    protected String readData() {	// qadam 1: o'qish
        return "Reading JSON data";	// JSON-maxsus xabar
    }
    // ... boshqa metodlar uchun bir xil pattern
}
\`\`\`

Har bir qadamda qayta ishlanayotgan formatni ko'rsatuvchi tavsiflovchi satrlarni qaytaring.`,
			whyItMatters: `## Nima uchun Template Method Pattern muhim

### Muammo: O'xshash algoritmlarda kod takrorlanishi

Template Method siz o'xshash algoritmlar boshqaruv oqimining takrorlanishiga olib keladi:

\`\`\`java
// ❌ Template Method siz - takrorlangan algoritm strukturasi
class CSVProcessor {	// CSV qayta ishlash - takrorlangan struktura
    public List<String> process() {	// bir xil strukturadagi metod
        List<String> results = new ArrayList<>();	// bir xil yig'ish mantiqi
        // Qadam 1
        results.add("Reading CSV data");	// CSV-maxsus
        // Qadam 2
        results.add("Processing CSV rows");	// CSV-maxsus
        // Qadam 3
        results.add("Writing CSV output");	// CSV-maxsus
        return results;	// bir xil return
    }
}

class JSONProcessor {	// JSON qayta ishlash - takrorlangan struktura
    public List<String> process() {	// BIR XIL metod strukturasi!
        List<String> results = new ArrayList<>();	// BIR XIL yig'ish mantiqi!
        // Qadam 1
        results.add("Reading JSON data");	// JSON-maxsus
        // Qadam 2
        results.add("Processing JSON objects");	// JSON-maxsus
        // Qadam 3
        results.add("Writing JSON output");	// JSON-maxsus
        return results;	// BIR XIL return!
    }
}

class XMLProcessor {	// XML qayta ishlash - yana takrorlanish
    public List<String> process() {	// BIR XIL struktura yana!
        List<String> results = new ArrayList<>();	// BIR XIL!
        results.add("Reading XML data");	// XML-maxsus
        results.add("Processing XML nodes");	// XML-maxsus
        results.add("Writing XML output");	// XML-maxsus
        return results;	// BIR XIL!
    }
}

// Muammolar:
// 1. Algoritm strukturasi har bir klassda takrorlanadi
// 2. Algoritm o'zgarsa, BARCHA klasslarni yangilash kerak
// 3. Bir klassda xato kiritish oson
// 4. Barcha protsessorlar bir xil strukturaga amal qilishiga kafolat yo'q
\`\`\`

\`\`\`java
// ✅ Template Method bilan - algoritm bir marta belgilangan
abstract class DataProcessor {	// algoritm skeletini bir marta belgilash
    public final List<String> process() {	// shablon metodi - FINAL
        List<String> results = new ArrayList<>();	// algoritm strukturasi bir marta belgilangan
        results.add(readData());	// qadam 1 - subklassga delegatsiya
        results.add(processData());	// qadam 2 - subklassga delegatsiya
        results.add(writeData());	// qadam 3 - subklassga delegatsiya
        return results;	// izchil return
    }

    protected abstract String readData();	// subklass amalga oshirishi kerak
    protected abstract String processData();	// subklass amalga oshirishi kerak
    protected abstract String writeData();	// subklass amalga oshirishi kerak
}

class CSVProcessor extends DataProcessor {	// faqat qadamlarni amalga oshiradi
    @Override
    protected String readData() { return "Reading CSV data"; }	// CSV-maxsus
    @Override
    protected String processData() { return "Processing CSV rows"; }	// CSV-maxsus
    @Override
    protected String writeData() { return "Writing CSV output"; }	// CSV-maxsus
}

class JSONProcessor extends DataProcessor {	// faqat qadamlarni amalga oshiradi
    @Override
    protected String readData() { return "Reading JSON data"; }	// JSON-maxsus
    @Override
    protected String processData() { return "Processing JSON objects"; }	// JSON-maxsus
    @Override
    protected String writeData() { return "Writing JSON output"; }	// JSON-maxsus
}

// Afzalliklar:
// 1. Algoritm bir marta belgilangan - DRY printsipi
// 2. Algoritmga o'zgartirishlar bir joyda
// 3. Izchil xatti-harakat kafolatlangan
// 4. Subklasslar faqat farq qiladigan narsani sozlaydi
\`\`\`

---

### Haqiqiy dunyo qo'llanilishi

| Qo'llanilish | Abstrakt klass | Shablon metodi | Qadamlar |
|--------------|----------------|----------------|----------|
| **HttpServlet** | HttpServlet | service() | doGet(), doPost(), doPut() |
| **JUnit** | TestCase | runTest() | setUp(), test*, tearDown() |
| **Spring JdbcTemplate** | JdbcTemplate | execute() | open, query, map, close |
| **Build tizimlari** | BuildTask | build() | compile, test, package |
| **Hujjat eksporti** | Exporter | export() | header, body, footer |

---

### Production Pattern: Hayot sikli hooklari bilan test freymvorki

\`\`\`java
import java.util.*;	// to'plamlar uchun utilitalarni import qilish

// AbstractClass - Test freymvorki bazasi
abstract class TestCase {	// test hayot sikli shablonini belgilaydi
    private List<String> log = new ArrayList<>();	// bajarish jurnali
    private boolean passed = true;	// test natijasi
    private String failureMessage = null;	// xato tafsilotlari

    // Shablon metodi - FINAL, qayta belgilab bo'lmaydi
    public final TestResult run() {	// shablon metodi bajarish tartibini belgilaydi
        log.clear();	// oldingi jurnalni tozalash
        passed = true;	// holatni qayta o'rnatish
        failureMessage = null;	// xatoni qayta o'rnatish

        try {
            log.add("Setting up test: " + getTestName());	// sozlashni jurnalga yozish
            setUp();	// Qadam 1: sozlash (hook)

            log.add("Running test: " + getTestName());	// bajarishni jurnalga yozish
            runTest();	// Qadam 2: testni bajarish (abstrakt)

            log.add("Verifying results");	// tekshirishni jurnalga yozish
            verify();	// Qadam 3: tekshirish (standart hook)

        } catch (AssertionError e) {	// test xatolarini ushlash
            passed = false;	// muvaffaqiyatsiz deb belgilash
            failureMessage = e.getMessage();	// xato xabarini saqlash
            log.add("FAILED: " + failureMessage);	// muvaffaqiyatsizlikni jurnalga yozish
        } catch (Exception e) {	// kutilmagan xatolarni ushlash
            passed = false;	// muvaffaqiyatsiz deb belgilash
            failureMessage = "Exception: " + e.getMessage();	// xatoni saqlash
            log.add("ERROR: " + failureMessage);	// xatoni jurnalga yozish
        } finally {
            log.add("Tearing down test");	// tozalashni jurnalga yozish
            tearDown();	// Qadam 4: tozalash (hook)
        }

        return new TestResult(getTestName(), passed, failureMessage, log);	// natijani qaytarish
    }

    // Abstrakt metod - subklasslar AMALGA OSHIRISHI KERAK
    protected abstract void runTest() throws Exception;	// haqiqiy test mantiqi

    // Abstrakt metod - test nomi
    protected abstract String getTestName();	// testni aniqlash

    // Hook metodlar - standart realizatsiyalar, qayta belgilanishi mumkin
    protected void setUp() { }	// ixtiyoriy sozlash, standart hech narsa qilmaydi

    protected void tearDown() { }	// ixtiyoriy tozalash, standart hech narsa qilmaydi

    protected void verify() { }	// ixtiyoriy tekshirish, standart hech narsa qilmaydi

    // Assertion yordamchilari
    protected void assertEquals(Object expected, Object actual) {	// tenglik assertion
        if (!Objects.equals(expected, actual)) {	// ob'ektlarni solishtirish
            throw new AssertionError("Expected " + expected + " but got " + actual);	// farq qilsa muvaffaqiyatsiz
        }
    }

    protected void assertTrue(boolean condition, String message) {	// boolean assertion
        if (!condition) {	// shartni tekshirish
            throw new AssertionError(message);	// false bo'lsa muvaffaqiyatsiz
        }
    }

    protected void assertNotNull(Object obj, String message) {	// null tekshiruv assertion
        if (obj == null) {	// null ga tekshirish
            throw new AssertionError(message);	// null bo'lsa muvaffaqiyatsiz
        }
    }
}

// Test natijalari uchun value ob'ekti
class TestResult {	// o'zgarmas test natijasi
    private final String testName;	// test nomi
    private final boolean passed;	// o'tish/muvaffaqiyatsizlik holati
    private final String failureMessage;	// xato tafsilotlari (o'tsa null)
    private final List<String> executionLog;	// bajarish tarixi

    public TestResult(String testName, boolean passed,
                      String failureMessage, List<String> log) {	// konstruktor
        this.testName = testName;	// test nomini saqlash
        this.passed = passed;	// natijani saqlash
        this.failureMessage = failureMessage;	// xatoni saqlash (null bo'lishi mumkin)
        this.executionLog = new ArrayList<>(log);	// jurnal himoyaviy nusxasi
    }

    public String getTestName() { return testName; }	// test nomini olish
    public boolean isPassed() { return passed; }	// o'tish holatini olish
    public String getFailureMessage() { return failureMessage; }	// xato xabarini olish
    public List<String> getExecutionLog() { return new ArrayList<>(executionLog); }	// jurnal nusxasini olish

    @Override
    public String toString() {	// satr ko'rinishi
        return testName + ": " + (passed ? "PASSED" : "FAILED - " + failureMessage);	// qisqacha
    }
}

// ConcreteClass - Foydalanuvchi xizmati testi
class UserServiceTest extends TestCase {	// aniq test realizatsiyasi
    private UserService userService;	// test qilinadigan xizmat
    private User testUser;	// test ma'lumotlari

    @Override
    protected String getTestName() {	// bu testni aniqlash
        return "UserServiceTest.testCreateUser";	// test nomi
    }

    @Override
    protected void setUp() {	// test muhitini tayyorlash
        userService = new UserService();	// xizmat yaratish
        testUser = new User("john@example.com", "John Doe");	// test ma'lumotlarini yaratish
    }

    @Override
    protected void runTest() throws Exception {	// haqiqiy test
        User created = userService.create(testUser);	// test qilinayotgan metodini chaqirish
        assertNotNull(created, "Created user should not be null");	// null emasligini tekshirish
        assertEquals(testUser.getEmail(), created.getEmail());	// email mosligini tekshirish
    }

    @Override
    protected void verify() {	// qo'shimcha tekshirish
        assertTrue(userService.exists(testUser.getEmail()),
            "User should exist after creation");	// saqlanganligini tekshirish
    }

    @Override
    protected void tearDown() {	// tozalash
        if (userService != null && testUser != null) {	// resurslar mavjud bo'lsa
            userService.delete(testUser.getEmail());	// test ma'lumotlarini tozalash
        }
    }
}

// ConcreteClass - Ma'lumotlar bazasi ulanish testi
class DatabaseConnectionTest extends TestCase {	// boshqa test realizatsiyasi
    private Database db;	// ma'lumotlar bazasi ulanishi

    @Override
    protected String getTestName() {	// bu testni aniqlash
        return "DatabaseConnectionTest.testConnection";	// test nomi
    }

    @Override
    protected void setUp() {	// ma'lumotlar bazasini tayyorlash
        db = Database.getInstance();	// ma'lumotlar bazasi instansini olish
        db.connect();	// ulanish o'rnatish
    }

    @Override
    protected void runTest() throws Exception {	// ma'lumotlar bazasi operatsiyalarini test qilish
        assertTrue(db.isConnected(), "Database should be connected");	// ulanishni tekshirish
        db.execute("SELECT 1");	// oddiy so'rov bajarish
    }

    @Override
    protected void tearDown() {	// ulanishni tozalash
        if (db != null) {	// ma'lumotlar bazasi mavjud bo'lsa
            db.disconnect();	// ulanishni yopish
        }
    }
}

// Test Runner - bir nechta testlarni bajaradi
class TestRunner {	// test bajarishni boshqaradi
    private List<TestResult> results = new ArrayList<>();	// yig'ilgan natijalar

    public void runAll(List<TestCase> tests) {	// barcha testlarni bajarish
        results.clear();	// oldingi natijalarni tozalash
        for (TestCase test : tests) {	// testlarni takrorlash
            TestResult result = test.run();	// shablon metodini bajarish
            results.add(result);	// natijani yig'ish
            System.out.println(result);	// natijani chop etish
        }
    }

    public int getPassedCount() {	// o'tgan testlarni sanash
        return (int) results.stream().filter(TestResult::isPassed).count();	// filtrlash va sanash
    }

    public int getFailedCount() {	// muvaffaqiyatsiz testlarni sanash
        return results.size() - getPassedCount();	// jami minus o'tganlar
    }

    public void printSummary() {	// test xulosasini chop etish
        System.out.println("\\n=== Test Summary ===");	// sarlavha
        System.out.println("Total: " + results.size());	// jami son
        System.out.println("Passed: " + getPassedCount());	// o'tganlar soni
        System.out.println("Failed: " + getFailedCount());	// muvaffaqiyatsizlar soni
    }
}

// Foydalanish:
TestRunner runner = new TestRunner();	// test bajaruvchi yaratish

List<TestCase> tests = Arrays.asList(	// test to'plamini yaratish
    new UserServiceTest(),	// foydalanuvchi xizmati testi
    new DatabaseConnectionTest()	// ma'lumotlar bazasi ulanish testi
);

runner.runAll(tests);	// barcha testlarni shablon metodi yordamida bajarish
runner.printSummary();	// natijalarni chop etish

// Chiqish:
// Setting up test: UserServiceTest.testCreateUser
// Running test: UserServiceTest.testCreateUser
// Verifying results
// Tearing down test
// UserServiceTest.testCreateUser: PASSED
// ...
\`\`\`

---

### Oldini olish kerak bo'lgan keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| Shablon metodi final emas | Subklasslar algoritmni buzishi mumkin | Shablon metodini \`final\` deb belgilash |
| Juda ko'p abstrakt metodlar | Kengaytirish qiyin | Standart realizatsiyali hooklardan foydalanish |
| Konstruktorda abstrakt metodlarni chaqirish | Subklass initsializatsiya qilinmagan | Shablon metodidan foydalaning, konstruktordan emas |
| Hooklar yo'q | Qattiq, moslashuvchan emas | Standart xatti-harakatli ixtiyoriy hooklar qo'shish |
| Qadamlar juda mayda | Amalga oshirish uchun juda ko'p metodlar | Abstraksiya va amaliylik o'rtasida muvozanat |`
		}
	}
};

export default task;
