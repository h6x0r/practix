import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-abstract-factory',
	title: 'Abstract Factory Pattern',
	difficulty: 'hard',
	tags: ['java', 'design-patterns', 'creational', 'abstract-factory'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Abstract Factory pattern in Java - provide an interface for creating families of related objects.

**You will implement:**

1. **Button, Checkbox interfaces** - Product interfaces
2. **WindowsButton, MacButton** - Concrete products
3. **GUIFactory interface** - Abstract factory
4. **WindowsFactory, MacFactory** - Concrete factories

**Example Usage:**

\`\`\`java
GUIFactory factory = GUIFactory.getFactory("windows");	// get platform-specific factory
Button btn = factory.createButton();	// create platform-specific button
Checkbox chk = factory.createCheckbox();	// create platform-specific checkbox
String rendered = btn.render();	// "Rendering Windows button"
String checked = chk.check();	// "Windows checkbox checked"

// Switch platform
GUIFactory macFactory = GUIFactory.getFactory("mac");	// get Mac factory
Button macBtn = macFactory.createButton();	// all products are Mac-compatible
macBtn.render();	// "Rendering Mac button"
\`\`\``,
	initialCode: `interface Button {
    String render();
}

interface Checkbox {
    String check();
}

class WindowsButton implements Button {
    @Override
    public String render() {
        throw new UnsupportedOperationException("TODO");
    }
}

class MacButton implements Button {
    @Override
    public String render() {
        throw new UnsupportedOperationException("TODO");
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public String check() {
        throw new UnsupportedOperationException("TODO");
    }
}

class MacCheckbox implements Checkbox {
    @Override
    public String check() {
        throw new UnsupportedOperationException("TODO");
    }
}

interface GUIFactory {

    static GUIFactory getFactory(String osType) {
        throw new UnsupportedOperationException("TODO");
    }
}

class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public Checkbox createCheckbox() {
        throw new UnsupportedOperationException("TODO");
    }
}

class MacFactory implements GUIFactory {
    @Override
    public Button createButton() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public Checkbox createCheckbox() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface Button {	// Abstract Product A - button interface
    String render();	// all buttons must be renderable
}

interface Checkbox {	// Abstract Product B - checkbox interface
    String check();	// all checkboxes must be checkable
}

class WindowsButton implements Button {	// Concrete Product A1 - Windows button
    @Override
    public String render() {	// Windows-specific rendering
        return "Rendering Windows button";	// uses Windows native look
    }
}

class MacButton implements Button {	// Concrete Product A2 - Mac button
    @Override
    public String render() {	// Mac-specific rendering
        return "Rendering Mac button";	// uses macOS native look
    }
}

class WindowsCheckbox implements Checkbox {	// Concrete Product B1 - Windows checkbox
    @Override
    public String check() {	// Windows-specific behavior
        return "Windows checkbox checked";	// Windows style check
    }
}

class MacCheckbox implements Checkbox {	// Concrete Product B2 - Mac checkbox
    @Override
    public String check() {	// Mac-specific behavior
        return "Mac checkbox checked";	// macOS style check
    }
}

interface GUIFactory {	// Abstract Factory - creates family of products
    Button createButton();	// factory method for buttons
    Checkbox createCheckbox();	// factory method for checkboxes

    static GUIFactory getFactory(String osType) {	// static method to get appropriate factory
        if ("mac".equalsIgnoreCase(osType)) {	// case-insensitive comparison
            return new MacFactory();	// return Mac factory for Mac OS
        }
        return new WindowsFactory();	// default to Windows factory
    }
}

class WindowsFactory implements GUIFactory {	// Concrete Factory - Windows family
    @Override
    public Button createButton() {	// creates Windows button
        return new WindowsButton();	// returns Windows-specific button
    }

    @Override
    public Checkbox createCheckbox() {	// creates Windows checkbox
        return new WindowsCheckbox();	// returns Windows-specific checkbox
    }
}

class MacFactory implements GUIFactory {	// Concrete Factory - Mac family
    @Override
    public Button createButton() {	// creates Mac button
        return new MacButton();	// returns Mac-specific button
    }

    @Override
    public Checkbox createCheckbox() {	// creates Mac checkbox
        return new MacCheckbox();	// returns Mac-specific checkbox
    }
}`,
	hint1: `**Product Interfaces and Implementations:**

The product interfaces define what all products in a family can do:

\`\`\`java
interface Button {
    String render();	// render operation for buttons
}

interface Checkbox {
    String check();	// check operation for checkboxes
}
\`\`\`

Each platform implements both products with platform-specific behavior:

\`\`\`java
class WindowsButton implements Button {
    @Override
    public String render() {
        return "Rendering Windows button";	// Windows look and feel
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public String check() {
        return "Windows checkbox checked";	// Windows behavior
    }
}
\`\`\`

Mac products follow the same pattern with Mac-specific messages.`,
	hint2: `**Abstract Factory and Concrete Factories:**

The GUIFactory interface declares creation methods for each product:

\`\`\`java
interface GUIFactory {
    Button createButton();	// abstract creation method
    Checkbox createCheckbox();	// abstract creation method

    static GUIFactory getFactory(String osType) {	// convenience factory selector
        if ("mac".equalsIgnoreCase(osType)) {	// case-insensitive check
            return new MacFactory();	// return Mac factory
        }
        return new WindowsFactory();	// default: Windows factory
    }
}
\`\`\`

Each concrete factory creates products of one family:

\`\`\`java
class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();	// Windows family product
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();	// Windows family product
    }
}
\`\`\``,
	whyItMatters: `## Why Abstract Factory Exists

Abstract Factory ensures that products from the same family work together. It prevents mixing incompatible products (like Windows button with Mac checkbox).

**Problem - Incompatible Product Combinations:**

\`\`\`java
// ❌ Bad: Manually creating products leads to mismatches
class Application {
    public void createUI() {
        Button btn = new WindowsButton();	// Windows button
        Checkbox chk = new MacCheckbox();	// Mac checkbox - INCOMPATIBLE!
        // Mixing platforms causes inconsistent UI
    }
}
\`\`\`

**Solution - Abstract Factory Guarantees Compatibility:**

\`\`\`java
// ✅ Good: Factory ensures all products match
class Application {
    private final GUIFactory factory;	// abstract factory

    public Application(GUIFactory factory) {
        this.factory = factory;	// injected factory
    }

    public void createUI() {
        Button btn = factory.createButton();	// from same factory
        Checkbox chk = factory.createCheckbox();	// guaranteed compatible
        // All products belong to same platform family
    }
}
\`\`\`

---

## Real-World Examples

1. **Swing Look and Feel** - UIManager.getLookAndFeel() returns factory for specific L&F
2. **JDBC Database Drivers** - Connection creates Statement, ResultSet from same driver
3. **Java XML Parsers** - DocumentBuilderFactory creates compatible DOM objects
4. **Spring Application Context** - Creates related beans that work together
5. **JPA EntityManagerFactory** - Creates EntityManager, Query from same persistence provider

---

## Production Pattern: Cloud Provider Factory

\`\`\`java
// Abstract Products
interface StorageService {	// storage operations
    String uploadFile(String name, byte[] data);	// upload file
    byte[] downloadFile(String name);	// download file
    void deleteFile(String name);	// delete file
}

interface ComputeService {	// compute operations
    String createInstance(String type);	// create VM instance
    void terminateInstance(String instanceId);	// terminate instance
    String getInstanceStatus(String instanceId);	// check status
}

interface DatabaseService {	// database operations
    Connection createConnection(String dbName);	// get DB connection
    void createDatabase(String name);	// create new database
    void deleteDatabase(String name);	// delete database
}

// AWS Concrete Products
class AWSStorage implements StorageService {	// S3 implementation
    private final String bucketName;	// S3 bucket

    public AWSStorage(String bucketName) {	// constructor with bucket
        this.bucketName = bucketName;	// store bucket name
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// upload to S3
        return "s3://" + bucketName + "/" + name;	// return S3 URI
    }

    @Override
    public byte[] downloadFile(String name) {	// download from S3
        return new byte[0];	// simplified implementation
    }

    @Override
    public void deleteFile(String name) {	// delete from S3
        // S3 delete implementation
    }
}

class AWSCompute implements ComputeService {	// EC2 implementation
    @Override
    public String createInstance(String type) {	// launch EC2 instance
        return "i-" + UUID.randomUUID().toString().substring(0, 8);	// EC2 instance ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// terminate EC2
        // EC2 termination implementation
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// EC2 status
        return "running";	// simplified status
    }
}

class AWSDatabase implements DatabaseService {	// RDS implementation
    @Override
    public Connection createConnection(String dbName) {	// RDS connection
        return null;	// simplified - would return JDBC connection
    }

    @Override
    public void createDatabase(String name) {	// create RDS instance
        // RDS creation implementation
    }

    @Override
    public void deleteDatabase(String name) {	// delete RDS instance
        // RDS deletion implementation
    }
}

// GCP Concrete Products
class GCPStorage implements StorageService {	// Cloud Storage implementation
    private final String bucketName;	// GCS bucket

    public GCPStorage(String bucketName) {	// constructor with bucket
        this.bucketName = bucketName;	// store bucket name
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// upload to GCS
        return "gs://" + bucketName + "/" + name;	// return GCS URI
    }

    @Override
    public byte[] downloadFile(String name) {	// download from GCS
        return new byte[0];	// simplified implementation
    }

    @Override
    public void deleteFile(String name) {	// delete from GCS
        // GCS delete implementation
    }
}

class GCPCompute implements ComputeService {	// Compute Engine implementation
    @Override
    public String createInstance(String type) {	// launch GCE instance
        return "gce-" + UUID.randomUUID().toString().substring(0, 8);	// GCE instance ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// terminate GCE
        // GCE termination implementation
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// GCE status
        return "RUNNING";	// GCP status format
    }
}

class GCPDatabase implements DatabaseService {	// Cloud SQL implementation
    @Override
    public Connection createConnection(String dbName) {	// Cloud SQL connection
        return null;	// simplified - would return JDBC connection
    }

    @Override
    public void createDatabase(String name) {	// create Cloud SQL instance
        // Cloud SQL creation implementation
    }

    @Override
    public void deleteDatabase(String name) {	// delete Cloud SQL instance
        // Cloud SQL deletion implementation
    }
}

// Abstract Factory
interface CloudProviderFactory {	// creates family of cloud services
    StorageService createStorage(String bucketName);	// create storage service
    ComputeService createCompute();	// create compute service
    DatabaseService createDatabase();	// create database service

    static CloudProviderFactory getFactory(String provider) {	// factory selector
        switch (provider.toLowerCase()) {	// case-insensitive match
            case "gcp":
                return new GCPFactory();	// Google Cloud Platform
            case "aws":
            default:
                return new AWSFactory();	// Amazon Web Services (default)
        }
    }
}

// Concrete Factories
class AWSFactory implements CloudProviderFactory {	// AWS service family
    @Override
    public StorageService createStorage(String bucketName) {	// create S3
        return new AWSStorage(bucketName);	// AWS storage product
    }

    @Override
    public ComputeService createCompute() {	// create EC2
        return new AWSCompute();	// AWS compute product
    }

    @Override
    public DatabaseService createDatabase() {	// create RDS
        return new AWSDatabase();	// AWS database product
    }
}

class GCPFactory implements CloudProviderFactory {	// GCP service family
    @Override
    public StorageService createStorage(String bucketName) {	// create GCS
        return new GCPStorage(bucketName);	// GCP storage product
    }

    @Override
    public ComputeService createCompute() {	// create GCE
        return new GCPCompute();	// GCP compute product
    }

    @Override
    public DatabaseService createDatabase() {	// create Cloud SQL
        return new GCPDatabase();	// GCP database product
    }
}

// Client code
class CloudInfrastructure {	// uses abstract factory
    private final StorageService storage;	// storage component
    private final ComputeService compute;	// compute component
    private final DatabaseService database;	// database component

    public CloudInfrastructure(CloudProviderFactory factory) {	// inject factory
        this.storage = factory.createStorage("my-app-bucket");	// create storage
        this.compute = factory.createCompute();	// create compute
        this.database = factory.createDatabase();	// create database
    }

    public void deployApplication() {	// deploy using cloud services
        String instanceId = compute.createInstance("t2.medium");	// start VM
        database.createDatabase("app_db");	// create database
        storage.uploadFile("config.json", "{}".getBytes());	// upload config
    }
}
\`\`\`

---

## Common Mistakes to Avoid

1. **Mixing products from different factories** - Defeats the purpose of ensuring compatibility
2. **Too many product types** - If factory has many methods, split into multiple factories
3. **Not using interface for factory** - Use abstract class/interface, not concrete class
4. **Forgetting static factory method** - Provide convenient way to get correct factory instance`,
	order: 2,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void windowsButtonRendersWindows() {
        Button btn = new WindowsButton();
        String result = btn.render();
        assertTrue(result.contains("Windows"), "WindowsButton should mention Windows");
    }
}

class Test2 {
    @Test
    void macButtonRendersMac() {
        Button btn = new MacButton();
        String result = btn.render();
        assertTrue(result.contains("Mac"), "MacButton should mention Mac");
    }
}

class Test3 {
    @Test
    void windowsCheckboxChecksWindows() {
        Checkbox chk = new WindowsCheckbox();
        String result = chk.check();
        assertTrue(result.contains("Windows"), "WindowsCheckbox should mention Windows");
    }
}

class Test4 {
    @Test
    void macCheckboxChecksMac() {
        Checkbox chk = new MacCheckbox();
        String result = chk.check();
        assertTrue(result.contains("Mac"), "MacCheckbox should mention Mac");
    }
}

class Test5 {
    @Test
    void windowsFactoryCreatesWindowsButton() {
        GUIFactory factory = new WindowsFactory();
        Button btn = factory.createButton();
        assertTrue(btn instanceof WindowsButton, "WindowsFactory should create WindowsButton");
    }
}

class Test6 {
    @Test
    void macFactoryCreatesMacCheckbox() {
        GUIFactory factory = new MacFactory();
        Checkbox chk = factory.createCheckbox();
        assertTrue(chk instanceof MacCheckbox, "MacFactory should create MacCheckbox");
    }
}

class Test7 {
    @Test
    void getFactoryReturnsWindowsForDefault() {
        GUIFactory factory = GUIFactory.getFactory("windows");
        assertTrue(factory instanceof WindowsFactory, "Should return WindowsFactory for windows");
    }
}

class Test8 {
    @Test
    void getFactoryReturnsMacForMac() {
        GUIFactory factory = GUIFactory.getFactory("mac");
        assertTrue(factory instanceof MacFactory, "Should return MacFactory for mac");
    }
}

class Test9 {
    @Test
    void factoriesCreateCompatibleProducts() {
        GUIFactory factory = new WindowsFactory();
        Button btn = factory.createButton();
        Checkbox chk = factory.createCheckbox();
        assertTrue(btn.render().contains("Windows") && chk.check().contains("Windows"));
    }
}

class Test10 {
    @Test
    void getFactoryCaseInsensitive() {
        GUIFactory factory = GUIFactory.getFactory("MAC");
        assertTrue(factory instanceof MacFactory, "getFactory should be case insensitive");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Abstract Factory (Абстрактная Фабрика)',
			description: `Реализуйте паттерн Abstract Factory на Java — предоставьте интерфейс для создания семейств связанных объектов.

**Вы реализуете:**

1. **Интерфейсы Button, Checkbox** - Интерфейсы продуктов
2. **WindowsButton, MacButton** - Конкретные продукты
3. **Интерфейс GUIFactory** - Абстрактная фабрика
4. **WindowsFactory, MacFactory** - Конкретные фабрики

**Пример использования:**

\`\`\`java
GUIFactory factory = GUIFactory.getFactory("windows");	// получаем фабрику для платформы
Button btn = factory.createButton();	// создаём кнопку для платформы
Checkbox chk = factory.createCheckbox();	// создаём чекбокс для платформы
String rendered = btn.render();	// "Rendering Windows button"
String checked = chk.check();	// "Windows checkbox checked"

// Смена платформы
GUIFactory macFactory = GUIFactory.getFactory("mac");	// получаем Mac фабрику
Button macBtn = macFactory.createButton();	// все продукты совместимы с Mac
macBtn.render();	// "Rendering Mac button"
\`\`\``,
			hint1: `**Интерфейсы продуктов и реализации:**

Интерфейсы продуктов определяют, что могут делать все продукты семейства:

\`\`\`java
interface Button {
    String render();	// операция рендеринга для кнопок
}

interface Checkbox {
    String check();	// операция проверки для чекбоксов
}
\`\`\`

Каждая платформа реализует оба продукта с платформо-специфичным поведением:

\`\`\`java
class WindowsButton implements Button {
    @Override
    public String render() {
        return "Rendering Windows button";	// Windows look and feel
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public String check() {
        return "Windows checkbox checked";	// Windows поведение
    }
}
\`\`\`

Mac продукты следуют той же схеме с Mac-специфичными сообщениями.`,
			hint2: `**Абстрактная фабрика и конкретные фабрики:**

Интерфейс GUIFactory объявляет методы создания для каждого продукта:

\`\`\`java
interface GUIFactory {
    Button createButton();	// абстрактный метод создания
    Checkbox createCheckbox();	// абстрактный метод создания

    static GUIFactory getFactory(String osType) {	// удобный селектор фабрики
        if ("mac".equalsIgnoreCase(osType)) {	// проверка без учёта регистра
            return new MacFactory();	// возвращаем Mac фабрику
        }
        return new WindowsFactory();	// по умолчанию: Windows фабрика
    }
}
\`\`\`

Каждая конкретная фабрика создаёт продукты одного семейства:

\`\`\`java
class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();	// продукт семейства Windows
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();	// продукт семейства Windows
    }
}
\`\`\``,
			whyItMatters: `## Зачем нужен Abstract Factory

Abstract Factory гарантирует, что продукты одного семейства работают вместе. Он предотвращает смешивание несовместимых продуктов (например, Windows кнопка с Mac чекбоксом).

**Проблема - Несовместимые комбинации продуктов:**

\`\`\`java
// ❌ Плохо: Ручное создание продуктов ведёт к несоответствиям
class Application {
    public void createUI() {
        Button btn = new WindowsButton();	// Windows кнопка
        Checkbox chk = new MacCheckbox();	// Mac чекбокс - НЕСОВМЕСТИМО!
        // Смешивание платформ вызывает несогласованный UI
    }
}
\`\`\`

**Решение - Abstract Factory гарантирует совместимость:**

\`\`\`java
// ✅ Хорошо: Фабрика гарантирует совместимость всех продуктов
class Application {
    private final GUIFactory factory;	// абстрактная фабрика

    public Application(GUIFactory factory) {
        this.factory = factory;	// внедрённая фабрика
    }

    public void createUI() {
        Button btn = factory.createButton();	// от той же фабрики
        Checkbox chk = factory.createCheckbox();	// гарантированно совместим
        // Все продукты принадлежат одному семейству платформы
    }
}
\`\`\`

---

## Примеры из реального мира

1. **Swing Look and Feel** - UIManager.getLookAndFeel() возвращает фабрику для конкретного L&F
2. **JDBC Database Drivers** - Connection создаёт Statement, ResultSet от того же драйвера
3. **Java XML Parsers** - DocumentBuilderFactory создаёт совместимые DOM объекты
4. **Spring Application Context** - Создаёт связанные бины, которые работают вместе
5. **JPA EntityManagerFactory** - Создаёт EntityManager, Query от того же провайдера

---

## Production паттерн: Фабрика облачного провайдера

\`\`\`java
// Абстрактные продукты
interface StorageService {	// операции хранения
    String uploadFile(String name, byte[] data);	// загрузить файл
    byte[] downloadFile(String name);	// скачать файл
    void deleteFile(String name);	// удалить файл
}

interface ComputeService {	// вычислительные операции
    String createInstance(String type);	// создать VM инстанс
    void terminateInstance(String instanceId);	// завершить инстанс
    String getInstanceStatus(String instanceId);	// проверить статус
}

interface DatabaseService {	// операции с базой данных
    Connection createConnection(String dbName);	// получить подключение к БД
    void createDatabase(String name);	// создать новую базу данных
    void deleteDatabase(String name);	// удалить базу данных
}

// AWS Конкретные продукты
class AWSStorage implements StorageService {	// реализация S3
    private final String bucketName;	// S3 bucket

    public AWSStorage(String bucketName) {	// конструктор с bucket
        this.bucketName = bucketName;	// сохраняем имя bucket
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// загрузка в S3
        return "s3://" + bucketName + "/" + name;	// возвращаем S3 URI
    }

    @Override
    public byte[] downloadFile(String name) {	// скачивание из S3
        return new byte[0];	// упрощённая реализация
    }

    @Override
    public void deleteFile(String name) {	// удаление из S3
        // Реализация удаления из S3
    }
}

class AWSCompute implements ComputeService {	// реализация EC2
    @Override
    public String createInstance(String type) {	// запуск EC2 инстанса
        return "i-" + UUID.randomUUID().toString().substring(0, 8);	// EC2 instance ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// завершение EC2
        // Реализация завершения EC2
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// статус EC2
        return "running";	// упрощённый статус
    }
}

class AWSDatabase implements DatabaseService {	// реализация RDS
    @Override
    public Connection createConnection(String dbName) {	// подключение к RDS
        return null;	// упрощённо - вернул бы JDBC connection
    }

    @Override
    public void createDatabase(String name) {	// создание RDS инстанса
        // Реализация создания RDS
    }

    @Override
    public void deleteDatabase(String name) {	// удаление RDS инстанса
        // Реализация удаления RDS
    }
}

// GCP Конкретные продукты
class GCPStorage implements StorageService {	// реализация Cloud Storage
    private final String bucketName;	// GCS bucket

    public GCPStorage(String bucketName) {	// конструктор с bucket
        this.bucketName = bucketName;	// сохраняем имя bucket
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// загрузка в GCS
        return "gs://" + bucketName + "/" + name;	// возвращаем GCS URI
    }

    @Override
    public byte[] downloadFile(String name) {	// скачивание из GCS
        return new byte[0];	// упрощённая реализация
    }

    @Override
    public void deleteFile(String name) {	// удаление из GCS
        // Реализация удаления из GCS
    }
}

class GCPCompute implements ComputeService {	// реализация Compute Engine
    @Override
    public String createInstance(String type) {	// запуск GCE инстанса
        return "gce-" + UUID.randomUUID().toString().substring(0, 8);	// GCE instance ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// завершение GCE
        // Реализация завершения GCE
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// статус GCE
        return "RUNNING";	// формат статуса GCP
    }
}

class GCPDatabase implements DatabaseService {	// реализация Cloud SQL
    @Override
    public Connection createConnection(String dbName) {	// подключение к Cloud SQL
        return null;	// упрощённо - вернул бы JDBC connection
    }

    @Override
    public void createDatabase(String name) {	// создание Cloud SQL инстанса
        // Реализация создания Cloud SQL
    }

    @Override
    public void deleteDatabase(String name) {	// удаление Cloud SQL инстанса
        // Реализация удаления Cloud SQL
    }
}

// Абстрактная фабрика
interface CloudProviderFactory {	// создаёт семейство облачных сервисов
    StorageService createStorage(String bucketName);	// создать сервис хранения
    ComputeService createCompute();	// создать вычислительный сервис
    DatabaseService createDatabase();	// создать сервис базы данных

    static CloudProviderFactory getFactory(String provider) {	// селектор фабрики
        switch (provider.toLowerCase()) {	// сравнение без учёта регистра
            case "gcp":
                return new GCPFactory();	// Google Cloud Platform
            case "aws":
            default:
                return new AWSFactory();	// Amazon Web Services (по умолчанию)
        }
    }
}

// Конкретные фабрики
class AWSFactory implements CloudProviderFactory {	// семейство сервисов AWS
    @Override
    public StorageService createStorage(String bucketName) {	// создать S3
        return new AWSStorage(bucketName);	// AWS продукт хранения
    }

    @Override
    public ComputeService createCompute() {	// создать EC2
        return new AWSCompute();	// AWS вычислительный продукт
    }

    @Override
    public DatabaseService createDatabase() {	// создать RDS
        return new AWSDatabase();	// AWS продукт базы данных
    }
}

class GCPFactory implements CloudProviderFactory {	// семейство сервисов GCP
    @Override
    public StorageService createStorage(String bucketName) {	// создать GCS
        return new GCPStorage(bucketName);	// GCP продукт хранения
    }

    @Override
    public ComputeService createCompute() {	// создать GCE
        return new GCPCompute();	// GCP вычислительный продукт
    }

    @Override
    public DatabaseService createDatabase() {	// создать Cloud SQL
        return new GCPDatabase();	// GCP продукт базы данных
    }
}

// Клиентский код
class CloudInfrastructure {	// использует абстрактную фабрику
    private final StorageService storage;	// компонент хранения
    private final ComputeService compute;	// вычислительный компонент
    private final DatabaseService database;	// компонент базы данных

    public CloudInfrastructure(CloudProviderFactory factory) {	// внедряем фабрику
        this.storage = factory.createStorage("my-app-bucket");	// создаём хранилище
        this.compute = factory.createCompute();	// создаём вычисления
        this.database = factory.createDatabase();	// создаём базу данных
    }

    public void deployApplication() {	// развёртывание с использованием облачных сервисов
        String instanceId = compute.createInstance("t2.medium");	// запускаем VM
        database.createDatabase("app_db");	// создаём базу данных
        storage.uploadFile("config.json", "{}".getBytes());	// загружаем конфиг
    }
}
\`\`\`

---

## Частые ошибки, которых следует избегать

1. **Смешивание продуктов от разных фабрик** - Нарушает цель обеспечения совместимости
2. **Слишком много типов продуктов** - Если у фабрики много методов, разделите на несколько фабрик
3. **Не использовать интерфейс для фабрики** - Используйте абстрактный класс/интерфейс, а не конкретный класс
4. **Забывать о статическом методе фабрики** - Предоставьте удобный способ получить правильный экземпляр фабрики`
		},
		uz: {
			title: 'Abstract Factory Pattern',
			description: `Java da Abstract Factory patternini amalga oshiring — bog'liq ob'ektlar oilalarini yaratish uchun interfeys taqdim eting.

**Siz amalga oshirasiz:**

1. **Button, Checkbox interfeyslari** - Mahsulot interfeyslari
2. **WindowsButton, MacButton** - Konkret mahsulotlar
3. **GUIFactory interfeysi** - Abstrakt fabrika
4. **WindowsFactory, MacFactory** - Konkret fabrikalar

**Foydalanish misoli:**

\`\`\`java
GUIFactory factory = GUIFactory.getFactory("windows");	// platforma uchun fabrika olish
Button btn = factory.createButton();	// platforma uchun tugma yaratish
Checkbox chk = factory.createCheckbox();	// platforma uchun checkbox yaratish
String rendered = btn.render();	// "Rendering Windows button"
String checked = chk.check();	// "Windows checkbox checked"

// Platformani almashtirish
GUIFactory macFactory = GUIFactory.getFactory("mac");	// Mac fabrikasini olish
Button macBtn = macFactory.createButton();	// barcha mahsulotlar Mac bilan mos
macBtn.render();	// "Rendering Mac button"
\`\`\``,
			hint1: `**Mahsulot interfeyslari va amalga oshirishlar:**

Mahsulot interfeyslari oiladagi barcha mahsulotlar nima qila olishini belgilaydi:

\`\`\`java
interface Button {
    String render();	// tugmalar uchun render operatsiyasi
}

interface Checkbox {
    String check();	// checkboxlar uchun tekshirish operatsiyasi
}
\`\`\`

Har bir platforma ikkala mahsulotni platformaga xos xatti-harakat bilan amalga oshiradi:

\`\`\`java
class WindowsButton implements Button {
    @Override
    public String render() {
        return "Rendering Windows button";	// Windows ko'rinishi
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public String check() {
        return "Windows checkbox checked";	// Windows xatti-harakati
    }
}
\`\`\`

Mac mahsulotlari xuddi shunday sxemaga amal qiladi, Mac ga xos xabarlar bilan.`,
			hint2: `**Abstrakt fabrika va konkret fabrikalar:**

GUIFactory interfeysi har bir mahsulot uchun yaratish metodlarini e'lon qiladi:

\`\`\`java
interface GUIFactory {
    Button createButton();	// abstrakt yaratish metodi
    Checkbox createCheckbox();	// abstrakt yaratish metodi

    static GUIFactory getFactory(String osType) {	// qulay fabrika tanlash
        if ("mac".equalsIgnoreCase(osType)) {	// katta-kichik harflarni hisobga olmasdan tekshirish
            return new MacFactory();	// Mac fabrikasini qaytarish
        }
        return new WindowsFactory();	// standart: Windows fabrikasi
    }
}
\`\`\`

Har bir konkret fabrika bitta oilaning mahsulotlarini yaratadi:

\`\`\`java
class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();	// Windows oilasi mahsuloti
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();	// Windows oilasi mahsuloti
    }
}
\`\`\``,
			whyItMatters: `## Abstract Factory nima uchun kerak

Abstract Factory bir oiladagi mahsulotlarning birga ishlashini ta'minlaydi. U mos kelmaydigan mahsulotlarni aralashtirish (masalan, Windows tugmasi Mac checkbox bilan) oldini oladi.

**Muammo - Mos kelmaydigan mahsulot kombinatsiyalari:**

\`\`\`java
// ❌ Yomon: Mahsulotlarni qo'lda yaratish nomuvofiqlikka olib keladi
class Application {
    public void createUI() {
        Button btn = new WindowsButton();	// Windows tugmasi
        Checkbox chk = new MacCheckbox();	// Mac checkbox - MOS KELMAYDI!
        // Platformalarni aralashtirish nomuvofiq UI ga olib keladi
    }
}
\`\`\`

**Yechim - Abstract Factory muvofiqlikni kafolatlaydi:**

\`\`\`java
// ✅ Yaxshi: Fabrika barcha mahsulotlarning mosligini ta'minlaydi
class Application {
    private final GUIFactory factory;	// abstrakt fabrika

    public Application(GUIFactory factory) {
        this.factory = factory;	// kiritilgan fabrika
    }

    public void createUI() {
        Button btn = factory.createButton();	// bir xil fabrikadan
        Checkbox chk = factory.createCheckbox();	// kafolatlangan mos
        // Barcha mahsulotlar bitta platforma oilasiga tegishli
    }
}
\`\`\`

---

## Haqiqiy dunyo misollari

1. **Swing Look and Feel** - UIManager.getLookAndFeel() muayyan L&F uchun fabrika qaytaradi
2. **JDBC Database Drivers** - Connection bir xil drayverdan Statement, ResultSet yaratadi
3. **Java XML Parsers** - DocumentBuilderFactory mos DOM ob'ektlarni yaratadi
4. **Spring Application Context** - Birga ishlaydigan bog'liq beanlarni yaratadi
5. **JPA EntityManagerFactory** - Bir xil provayderdan EntityManager, Query yaratadi

---

## Production pattern: Bulut provayder fabrikasi

\`\`\`java
// Abstrakt mahsulotlar
interface StorageService {	// saqlash operatsiyalari
    String uploadFile(String name, byte[] data);	// faylni yuklash
    byte[] downloadFile(String name);	// faylni yuklab olish
    void deleteFile(String name);	// faylni o'chirish
}

interface ComputeService {	// hisoblash operatsiyalari
    String createInstance(String type);	// VM instansiya yaratish
    void terminateInstance(String instanceId);	// instansiyani tugatish
    String getInstanceStatus(String instanceId);	// holatni tekshirish
}

interface DatabaseService {	// ma'lumotlar bazasi operatsiyalari
    Connection createConnection(String dbName);	// DB ulanishini olish
    void createDatabase(String name);	// yangi ma'lumotlar bazasi yaratish
    void deleteDatabase(String name);	// ma'lumotlar bazasini o'chirish
}

// AWS Konkret mahsulotlar
class AWSStorage implements StorageService {	// S3 amalga oshirish
    private final String bucketName;	// S3 bucket

    public AWSStorage(String bucketName) {	// bucket bilan konstruktor
        this.bucketName = bucketName;	// bucket nomini saqlash
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// S3 ga yuklash
        return "s3://" + bucketName + "/" + name;	// S3 URI qaytarish
    }

    @Override
    public byte[] downloadFile(String name) {	// S3 dan yuklab olish
        return new byte[0];	// soddalashtirilgan amalga oshirish
    }

    @Override
    public void deleteFile(String name) {	// S3 dan o'chirish
        // S3 o'chirish amalga oshirish
    }
}

class AWSCompute implements ComputeService {	// EC2 amalga oshirish
    @Override
    public String createInstance(String type) {	// EC2 instansiyasini ishga tushirish
        return "i-" + UUID.randomUUID().toString().substring(0, 8);	// EC2 instansiya ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// EC2 ni tugatish
        // EC2 tugatish amalga oshirish
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// EC2 holati
        return "running";	// soddalashtirilgan holat
    }
}

class AWSDatabase implements DatabaseService {	// RDS amalga oshirish
    @Override
    public Connection createConnection(String dbName) {	// RDS ulanishi
        return null;	// soddalashtirilgan - JDBC connection qaytaradi
    }

    @Override
    public void createDatabase(String name) {	// RDS instansiyasini yaratish
        // RDS yaratish amalga oshirish
    }

    @Override
    public void deleteDatabase(String name) {	// RDS instansiyasini o'chirish
        // RDS o'chirish amalga oshirish
    }
}

// GCP Konkret mahsulotlar
class GCPStorage implements StorageService {	// Cloud Storage amalga oshirish
    private final String bucketName;	// GCS bucket

    public GCPStorage(String bucketName) {	// bucket bilan konstruktor
        this.bucketName = bucketName;	// bucket nomini saqlash
    }

    @Override
    public String uploadFile(String name, byte[] data) {	// GCS ga yuklash
        return "gs://" + bucketName + "/" + name;	// GCS URI qaytarish
    }

    @Override
    public byte[] downloadFile(String name) {	// GCS dan yuklab olish
        return new byte[0];	// soddalashtirilgan amalga oshirish
    }

    @Override
    public void deleteFile(String name) {	// GCS dan o'chirish
        // GCS o'chirish amalga oshirish
    }
}

class GCPCompute implements ComputeService {	// Compute Engine amalga oshirish
    @Override
    public String createInstance(String type) {	// GCE instansiyasini ishga tushirish
        return "gce-" + UUID.randomUUID().toString().substring(0, 8);	// GCE instansiya ID
    }

    @Override
    public void terminateInstance(String instanceId) {	// GCE ni tugatish
        // GCE tugatish amalga oshirish
    }

    @Override
    public String getInstanceStatus(String instanceId) {	// GCE holati
        return "RUNNING";	// GCP holat formati
    }
}

class GCPDatabase implements DatabaseService {	// Cloud SQL amalga oshirish
    @Override
    public Connection createConnection(String dbName) {	// Cloud SQL ulanishi
        return null;	// soddalashtirilgan - JDBC connection qaytaradi
    }

    @Override
    public void createDatabase(String name) {	// Cloud SQL instansiyasini yaratish
        // Cloud SQL yaratish amalga oshirish
    }

    @Override
    public void deleteDatabase(String name) {	// Cloud SQL instansiyasini o'chirish
        // Cloud SQL o'chirish amalga oshirish
    }
}

// Abstrakt fabrika
interface CloudProviderFactory {	// bulut xizmatlari oilasini yaratadi
    StorageService createStorage(String bucketName);	// saqlash xizmatini yaratish
    ComputeService createCompute();	// hisoblash xizmatini yaratish
    DatabaseService createDatabase();	// ma'lumotlar bazasi xizmatini yaratish

    static CloudProviderFactory getFactory(String provider) {	// fabrika tanlash
        switch (provider.toLowerCase()) {	// katta-kichik harflarni hisobga olmasdan solishtirish
            case "gcp":
                return new GCPFactory();	// Google Cloud Platform
            case "aws":
            default:
                return new AWSFactory();	// Amazon Web Services (standart)
        }
    }
}

// Konkret fabrikalar
class AWSFactory implements CloudProviderFactory {	// AWS xizmatlari oilasi
    @Override
    public StorageService createStorage(String bucketName) {	// S3 yaratish
        return new AWSStorage(bucketName);	// AWS saqlash mahsuloti
    }

    @Override
    public ComputeService createCompute() {	// EC2 yaratish
        return new AWSCompute();	// AWS hisoblash mahsuloti
    }

    @Override
    public DatabaseService createDatabase() {	// RDS yaratish
        return new AWSDatabase();	// AWS ma'lumotlar bazasi mahsuloti
    }
}

class GCPFactory implements CloudProviderFactory {	// GCP xizmatlari oilasi
    @Override
    public StorageService createStorage(String bucketName) {	// GCS yaratish
        return new GCPStorage(bucketName);	// GCP saqlash mahsuloti
    }

    @Override
    public ComputeService createCompute() {	// GCE yaratish
        return new GCPCompute();	// GCP hisoblash mahsuloti
    }

    @Override
    public DatabaseService createDatabase() {	// Cloud SQL yaratish
        return new GCPDatabase();	// GCP ma'lumotlar bazasi mahsuloti
    }
}

// Klient kodi
class CloudInfrastructure {	// abstrakt fabrikadan foydalanadi
    private final StorageService storage;	// saqlash komponenti
    private final ComputeService compute;	// hisoblash komponenti
    private final DatabaseService database;	// ma'lumotlar bazasi komponenti

    public CloudInfrastructure(CloudProviderFactory factory) {	// fabrikani kiritish
        this.storage = factory.createStorage("my-app-bucket");	// saqlash yaratish
        this.compute = factory.createCompute();	// hisoblash yaratish
        this.database = factory.createDatabase();	// ma'lumotlar bazasi yaratish
    }

    public void deployApplication() {	// bulut xizmatlari yordamida deploy
        String instanceId = compute.createInstance("t2.medium");	// VM ishga tushirish
        database.createDatabase("app_db");	// ma'lumotlar bazasi yaratish
        storage.uploadFile("config.json", "{}".getBytes());	// konfigni yuklash
    }
}
\`\`\`

---

## Qochish kerak bo'lgan keng tarqalgan xatolar

1. **Turli fabrikalardan mahsulotlarni aralashtirish** - Muvofiqlikni ta'minlash maqsadini buzadi
2. **Juda ko'p mahsulot turlari** - Fabrikada ko'p metodlar bo'lsa, bir nechta fabrikalarga bo'ling
3. **Fabrika uchun interfeys ishlatmaslik** - Konkret klass emas, abstrakt klass/interfeys ishlating
4. **Statik fabrika metodini unutish** - To'g'ri fabrika instansiyasini olishning qulay usulini taqdim eting`
		}
	}
};

export default task;
