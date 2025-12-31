import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-composite',
	title: 'Composite Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'structural', 'composite'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Composite Pattern** in Java — compose objects into tree structures to represent part-whole hierarchies.

## Overview

The Composite pattern lets you treat individual objects and compositions of objects uniformly. It creates tree structures where both leaves and containers implement the same interface.

## Key Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **Component** | Common interface | \`Employee\` interface |
| **Leaf** | Individual object | \`Developer\` class |
| **Composite** | Container with children | \`Manager\` with subordinates |

## Your Task

Implement a company hierarchy system:

1. **Employee** interface - Component with getName(), getSalary(), showDetails()
2. **Developer** class - Leaf employee (no subordinates)
3. **Manager** class - Composite that manages other employees

## Example Usage

\`\`\`java
Developer dev1 = new Developer("Alice", 5000);	// create leaf employee
Developer dev2 = new Developer("Bob", 4500);	// create another leaf
Manager manager = new Manager("Charlie", 7000);	// create composite (manager)

manager.addSubordinate(dev1);	// add developer to team
manager.addSubordinate(dev2);	// add another developer

System.out.println(manager.getSalary());	// Output: 16500 (sum of all)
System.out.println(manager.showDetails());	// shows tree structure
\`\`\`

## Key Insight

The Manager's getSalary() recursively calculates total salary of entire team — this is the power of Composite pattern!`,
	initialCode: `import java.util.ArrayList;
import java.util.List;

interface Employee {
    String getName();
    String showDetails();
}

class Developer implements Employee {
    private String name;
    private double salary;

    public Developer(String name, double salary) {
    }

    @Override
    public String getName() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public double getSalary() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public String showDetails() { throw new UnsupportedOperationException("TODO"); }
}

class Manager implements Employee {
    private String name;
    private double salary;
    private List<Employee> subordinates = new ArrayList<>();

    public Manager(String name, double salary) {
    }

    public void addSubordinate(Employee e) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public String getName() { throw new UnsupportedOperationException("TODO"); }

    @Override
    public double getSalary() { throw new UnsupportedOperationException("TODO"); }

    @Override
    public String showDetails() { throw new UnsupportedOperationException("TODO"); }
}`,
	solutionCode: `import java.util.ArrayList;	// for ArrayList collection
import java.util.List;	// for List interface

interface Employee {	// Component - common interface for all employees
    String getName();	// get employee name
    double getSalary();	// get salary (recursive for composites)
    String showDetails();	// display employee info
}

class Developer implements Employee {	// Leaf - individual employee without subordinates
    private String name;	// developer's name
    private double salary;	// developer's salary

    public Developer(String name, double salary) {	// constructor
        this.name = name;	// store name
        this.salary = salary;	// store salary
    }

    @Override
    public String getName() { return name; }	// simple getter - return name
    @Override
    public double getSalary() { return salary; }	// simple getter - return salary
    @Override
    public String showDetails() {	// format developer info
        return "Developer: " + name + ", Salary: " + salary;	// return formatted string
    }
}

class Manager implements Employee {	// Composite - employee with subordinates
    private String name;	// manager's name
    private double salary;	// manager's own salary
    private List<Employee> subordinates = new ArrayList<>();	// list of team members

    public Manager(String name, double salary) {	// constructor
        this.name = name;	// store name
        this.salary = salary;	// store own salary
    }

    public void addSubordinate(Employee e) {	// add team member
        subordinates.add(e);	// add to subordinates list
    }

    @Override
    public String getName() { return name; }	// simple getter - return name

    @Override
    public double getSalary() {	// recursive salary calculation
        double total = salary;	// start with own salary
        for (Employee e : subordinates) {	// iterate all subordinates
            total += e.getSalary();	// add subordinate's salary (recursive!)
        }
        return total;	// return total team salary
    }

    @Override
    public String showDetails() {	// build tree view of organization
        StringBuilder sb = new StringBuilder();	// for building output
        sb.append("Manager: ").append(name).append(", Team Salary: ").append(getSalary());	// manager info
        for (Employee e : subordinates) {	// iterate subordinates
            sb.append("\\n  - ").append(e.showDetails());	// indent and append subordinate details
        }
        return sb.toString();	// return complete tree string
    }
}`,
	hint1: `**Developer Implementation (Leaf)**

The Developer class is a leaf node — it has no children:

\`\`\`java
class Developer implements Employee {
    private String name;
    private double salary;

    @Override
    public String getName() {
        return name;  // Simple getter
    }

    @Override
    public double getSalary() {
        return salary;  // Just return own salary
    }

    @Override
    public String showDetails() {
        return "Developer: " + name + ", Salary: " + salary;
    }
}
\`\`\`

Key points:
- Leaf nodes have no children
- All methods are straightforward implementations
- No delegation or recursion needed`,
	hint2: `**Manager Implementation (Composite)**

The Manager is a composite — it contains other Employees:

\`\`\`java
class Manager implements Employee {
    private List<Employee> subordinates = new ArrayList<>();

    public void addSubordinate(Employee e) {
        subordinates.add(e);
    }

    @Override
    public double getSalary() {
        double total = salary;  // Start with own salary
        for (Employee e : subordinates) {
            total += e.getSalary();  // Add each subordinate's salary
        }
        return total;
    }

    @Override
    public String showDetails() {
        StringBuilder sb = new StringBuilder();
        sb.append("Manager: ").append(name)
          .append(", Team Salary: ").append(getSalary());
        for (Employee e : subordinates) {
            sb.append("\\n  - ").append(e.showDetails());
        }
        return sb.toString();
    }
}
\`\`\`

The recursive call e.getSalary() works because subordinates can be Developers OR other Managers!`,
	whyItMatters: `## Problem & Solution

**Without Composite:**
\`\`\`java
// Must handle leaves and containers differently
if (node instanceof File) {
    return ((File) node).getSize();	// handle file
} else if (node instanceof Folder) {
    long size = 0;	// calculate folder size
    for (Object child : ((Folder) node).getChildren()) {
        // Repeat the same instanceof checks...	// messy recursion
    }
}
\`\`\`

**With Composite:**
\`\`\`java
interface FileSystemNode {
    long getSize();	// uniform interface
}
// Both File and Folder implement getSize()
node.getSize();	// works uniformly for both!
\`\`\`

---

## Real-World Examples

| Domain | Leaf | Composite | Operation |
|--------|------|-----------|-----------|
| **File System** | File | Directory | getSize() |
| **UI Framework** | Button, Label | Panel, Window | render() |
| **Organization** | Employee | Department | getSalary() |
| **Graphics** | Shape | Group | draw() |
| **Menu System** | MenuItem | Menu | execute() |
| **XML/HTML** | Text Node | Element | getText() |

---

## Production Pattern: File System

\`\`\`java
interface FileSystemNode {	// Component interface
    String getName();	// get node name
    long getSize();	// get size (recursive for directories)
    void print(String indent);	// display tree structure
}

class File implements FileSystemNode {	// Leaf - actual file
    private String name;	// file name
    private long size;	// file size in bytes

    public File(String name, long size) {	// constructor
        this.name = name;	// store name
        this.size = size;	// store size
    }

    @Override
    public String getName() { return name; }	// return file name

    @Override
    public long getSize() { return size; }	// return file size

    @Override
    public void print(String indent) {	// print file info
        System.out.println(indent + "📄 " + name + " (" + size + " bytes)");	// display with icon
    }
}

class Directory implements FileSystemNode {	// Composite - folder with children
    private String name;	// directory name
    private List<FileSystemNode> children = new ArrayList<>();	// child nodes

    public Directory(String name) {	// constructor
        this.name = name;	// store name
    }

    public void add(FileSystemNode node) {	// add child
        children.add(node);	// add to children list
    }

    public void remove(FileSystemNode node) {	// remove child
        children.remove(node);	// remove from list
    }

    @Override
    public String getName() { return name; }	// return directory name

    @Override
    public long getSize() {	// calculate total size
        long total = 0;	// initialize counter
        for (FileSystemNode child : children) {	// iterate children
            total += child.getSize();	// add child size (recursive!)
        }
        return total;	// return total size
    }

    @Override
    public void print(String indent) {	// print directory tree
        System.out.println(indent + "📁 " + name + "/");	// print folder
        for (FileSystemNode child : children) {	// iterate children
            child.print(indent + "  ");	// print with increased indent
        }
    }
}

// Usage:
Directory root = new Directory("project");	// create root directory
Directory src = new Directory("src");	// create src folder
Directory docs = new Directory("docs");	// create docs folder

src.add(new File("Main.java", 2048));	// add source file
src.add(new File("Utils.java", 1024));	// add utils file
docs.add(new File("README.md", 512));	// add readme

root.add(src);	// add src to root
root.add(docs);	// add docs to root
root.add(new File("pom.xml", 256));	// add config file

root.print("");	// print entire tree
System.out.println("Total size: " + root.getSize());	// 3840 bytes
\`\`\`

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Type checking** | Using instanceof to differentiate | Use polymorphism through Component interface |
| **Missing child operations** | Forgetting add/remove in composite | Include child management in composites |
| **Unsafe operations on leaves** | Calling add() on leaf throws | Either ignore or throw UnsupportedOperationException |
| **Circular references** | Adding parent as child | Check for cycles when adding |
| **Forgetting recursion** | Only calculating direct children | Ensure operations traverse full tree |`,
	order: 2,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void developerReturnsName() {
        Developer dev = new Developer("Alice", 5000);
        assertEquals("Alice", dev.getName(), "Developer should return correct name");
    }
}

class Test2 {
    @Test
    void developerReturnsSalary() {
        Developer dev = new Developer("Alice", 5000);
        assertEquals(5000, dev.getSalary(), "Developer should return own salary");
    }
}

class Test3 {
    @Test
    void developerShowDetails() {
        Developer dev = new Developer("Bob", 4500);
        String details = dev.showDetails();
        assertTrue(details.contains("Developer"), "Details should contain Developer");
        assertTrue(details.contains("Bob"), "Details should contain name");
        assertTrue(details.contains("4500"), "Details should contain salary");
    }
}

class Test4 {
    @Test
    void managerReturnsName() {
        Manager mgr = new Manager("Charlie", 7000);
        assertEquals("Charlie", mgr.getName(), "Manager should return correct name");
    }
}

class Test5 {
    @Test
    void managerSalaryWithoutSubordinates() {
        Manager mgr = new Manager("Charlie", 7000);
        assertEquals(7000, mgr.getSalary(), "Manager without subordinates should return own salary");
    }
}

class Test6 {
    @Test
    void managerSalaryWithSubordinates() {
        Developer dev1 = new Developer("Alice", 5000);
        Developer dev2 = new Developer("Bob", 4500);
        Manager mgr = new Manager("Charlie", 7000);
        mgr.addSubordinate(dev1);
        mgr.addSubordinate(dev2);
        assertEquals(16500, mgr.getSalary(), "Manager salary should include all subordinates");
    }
}

class Test7 {
    @Test
    void nestedManagersSalary() {
        Developer dev = new Developer("Alice", 5000);
        Manager subMgr = new Manager("Bob", 6000);
        subMgr.addSubordinate(dev);
        Manager topMgr = new Manager("Charlie", 8000);
        topMgr.addSubordinate(subMgr);
        assertEquals(19000, topMgr.getSalary(), "Nested manager salary should be calculated recursively");
    }
}

class Test8 {
    @Test
    void managerShowDetailsIncludesSubordinates() {
        Developer dev = new Developer("Alice", 5000);
        Manager mgr = new Manager("Charlie", 7000);
        mgr.addSubordinate(dev);
        String details = mgr.showDetails();
        assertTrue(details.contains("Manager"), "Details should contain Manager");
        assertTrue(details.contains("Alice"), "Details should contain subordinate name");
    }
}

class Test9 {
    @Test
    void employeeInterfaceWorks() {
        Employee dev = new Developer("Test", 1000);
        Employee mgr = new Manager("Test2", 2000);
        assertNotNull(dev.getName(), "Developer as Employee should work");
        assertNotNull(mgr.getName(), "Manager as Employee should work");
    }
}

class Test10 {
    @Test
    void emptyManagerHasZeroSubordinates() {
        Manager mgr = new Manager("Empty", 5000);
        assertEquals(5000, mgr.getSalary(), "Empty manager should only return own salary");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Composite (Компоновщик)',
			description: `Реализуйте **паттерн Composite** на Java — компонуйте объекты в древовидные структуры для представления иерархий часть-целое.

## Обзор

Паттерн Composite позволяет обрабатывать отдельные объекты и композиции объектов единообразно. Он создаёт древовидные структуры, где и листья, и контейнеры реализуют один интерфейс.

## Ключевые компоненты

| Компонент | Роль | Реализация |
|-----------|------|------------|
| **Component** | Общий интерфейс | Интерфейс \`Employee\` |
| **Leaf** | Отдельный объект | Класс \`Developer\` |
| **Composite** | Контейнер с детьми | \`Manager\` с подчинёнными |

## Ваша задача

Реализуйте систему иерархии компании:

1. **Employee** интерфейс - Component с getName(), getSalary(), showDetails()
2. **Developer** класс - Leaf сотрудник (без подчинённых)
3. **Manager** класс - Composite, управляющий другими сотрудниками

## Пример использования

\`\`\`java
Developer dev1 = new Developer("Alice", 5000);	// создаём leaf сотрудника
Developer dev2 = new Developer("Bob", 4500);	// создаём ещё одного leaf
Manager manager = new Manager("Charlie", 7000);	// создаём composite (менеджер)

manager.addSubordinate(dev1);	// добавляем разработчика в команду
manager.addSubordinate(dev2);	// добавляем другого разработчика

System.out.println(manager.getSalary());	// Вывод: 16500 (сумма всех)
System.out.println(manager.showDetails());	// показывает структуру дерева
\`\`\`

## Ключевая идея

getSalary() менеджера рекурсивно вычисляет общую зарплату всей команды — в этом сила паттерна Composite!`,
			hint1: `**Реализация Developer (Leaf)**

Класс Developer — это листовой узел — у него нет детей:

\`\`\`java
class Developer implements Employee {
    private String name;
    private double salary;

    @Override
    public String getName() {
        return name;  // Простой геттер
    }

    @Override
    public double getSalary() {
        return salary;  // Просто возвращаем свою зарплату
    }

    @Override
    public String showDetails() {
        return "Developer: " + name + ", Salary: " + salary;
    }
}
\`\`\`

Ключевые моменты:
- Листовые узлы не имеют детей
- Все методы — простые реализации
- Делегирование или рекурсия не нужны`,
			hint2: `**Реализация Manager (Composite)**

Manager — это композит — он содержит других Employee:

\`\`\`java
class Manager implements Employee {
    private List<Employee> subordinates = new ArrayList<>();

    public void addSubordinate(Employee e) {
        subordinates.add(e);
    }

    @Override
    public double getSalary() {
        double total = salary;  // Начинаем со своей зарплаты
        for (Employee e : subordinates) {
            total += e.getSalary();  // Добавляем зарплату каждого подчинённого
        }
        return total;
    }

    @Override
    public String showDetails() {
        StringBuilder sb = new StringBuilder();
        sb.append("Manager: ").append(name)
          .append(", Team Salary: ").append(getSalary());
        for (Employee e : subordinates) {
            sb.append("\\n  - ").append(e.showDetails());
        }
        return sb.toString();
    }
}
\`\`\`

Рекурсивный вызов e.getSalary() работает, потому что подчинёнными могут быть как Developer, так и другие Manager!`,
			whyItMatters: `## Проблема и решение

**Без Composite:**
\`\`\`java
// Нужно по-разному обрабатывать листья и контейнеры
if (node instanceof File) {
    return ((File) node).getSize();	// обрабатываем файл
} else if (node instanceof Folder) {
    long size = 0;	// вычисляем размер папки
    for (Object child : ((Folder) node).getChildren()) {
        // Повторяем те же проверки instanceof...	// запутанная рекурсия
    }
}
\`\`\`

**С Composite:**
\`\`\`java
interface FileSystemNode {
    long getSize();	// единый интерфейс
}
// И File, и Folder реализуют getSize()
node.getSize();	// работает единообразно для обоих!
\`\`\`

---

## Примеры из реального мира

| Домен | Leaf | Composite | Операция |
|-------|------|-----------|----------|
| **Файловая система** | File | Directory | getSize() |
| **UI Framework** | Button, Label | Panel, Window | render() |
| **Организация** | Employee | Department | getSalary() |
| **Графика** | Shape | Group | draw() |
| **Система меню** | MenuItem | Menu | execute() |
| **XML/HTML** | Text Node | Element | getText() |

---

## Production паттерн: Файловая система

\`\`\`java
interface FileSystemNode {	// Интерфейс Component
    String getName();	// получить имя узла
    long getSize();	// получить размер (рекурсивно для директорий)
    void print(String indent);	// отобразить структуру дерева
}

class File implements FileSystemNode {	// Leaf - реальный файл
    private String name;	// имя файла
    private long size;	// размер в байтах

    public File(String name, long size) {	// конструктор
        this.name = name;	// сохраняем имя
        this.size = size;	// сохраняем размер
    }

    @Override
    public String getName() { return name; }	// возвращаем имя файла

    @Override
    public long getSize() { return size; }	// возвращаем размер файла

    @Override
    public void print(String indent) {	// печатаем информацию о файле
        System.out.println(indent + "📄 " + name + " (" + size + " bytes)");	// отображаем с иконкой
    }
}

class Directory implements FileSystemNode {	// Composite - папка с детьми
    private String name;	// имя директории
    private List<FileSystemNode> children = new ArrayList<>();	// дочерние узлы

    public Directory(String name) {	// конструктор
        this.name = name;	// сохраняем имя
    }

    public void add(FileSystemNode node) {	// добавить дочерний элемент
        children.add(node);	// добавляем в список детей
    }

    public void remove(FileSystemNode node) {	// удалить дочерний элемент
        children.remove(node);	// удаляем из списка
    }

    @Override
    public String getName() { return name; }	// возвращаем имя директории

    @Override
    public long getSize() {	// вычисляем общий размер
        long total = 0;	// инициализируем счётчик
        for (FileSystemNode child : children) {	// итерируем детей
            total += child.getSize();	// добавляем размер ребёнка (рекурсия!)
        }
        return total;	// возвращаем общий размер
    }

    @Override
    public void print(String indent) {	// печатаем дерево директории
        System.out.println(indent + "📁 " + name + "/");	// печатаем папку
        for (FileSystemNode child : children) {	// итерируем детей
            child.print(indent + "  ");	// печатаем с увеличенным отступом
        }
    }
}

// Использование:
Directory root = new Directory("project");	// создаём корневую директорию
Directory src = new Directory("src");	// создаём папку src
Directory docs = new Directory("docs");	// создаём папку docs

src.add(new File("Main.java", 2048));	// добавляем исходный файл
src.add(new File("Utils.java", 1024));	// добавляем utils файл
docs.add(new File("README.md", 512));	// добавляем readme

root.add(src);	// добавляем src в корень
root.add(docs);	// добавляем docs в корень
root.add(new File("pom.xml", 256));	// добавляем конфиг файл

root.print("");	// печатаем всё дерево
System.out.println("Total size: " + root.getSize());	// 3840 байт
\`\`\`

---

## Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Проверка типов** | Использование instanceof для различения | Используйте полиморфизм через интерфейс Component |
| **Отсутствие операций с детьми** | Забывают add/remove в composite | Включайте управление детьми в композиты |
| **Небезопасные операции на листьях** | Вызов add() на leaf выбрасывает исключение | Игнорируйте или выбрасывайте UnsupportedOperationException |
| **Циклические ссылки** | Добавление родителя как ребёнка | Проверяйте на циклы при добавлении |
| **Забывают рекурсию** | Вычисляют только прямых детей | Убедитесь, что операции обходят всё дерево |`
		},
		uz: {
			title: 'Composite (Kompozit) Pattern',
			description: `Java da **Composite patternini** amalga oshiring — ob'ektlarni qism-butun ierarxiyalarini ifodalash uchun daraxt strukturalarida jamlang.

## Umumiy ko'rinish

Composite patterni alohida ob'ektlar va ob'ektlar kompozitsiyalarini bir xil tarzda boshqarish imkonini beradi. U daraxt strukturalarini yaratadi, bu yerda barglar ham, konteynerlar ham bir xil interfeysni amalga oshiradi.

## Asosiy komponentlar

| Komponent | Rol | Amalga oshirish |
|-----------|-----|-----------------|
| **Component** | Umumiy interfeys | \`Employee\` interfeysi |
| **Leaf** | Alohida ob'ekt | \`Developer\` klassi |
| **Composite** | Bolalari bor konteyner | Bo'ysunuvchilari bilan \`Manager\` |

## Vazifangiz

Kompaniya ierarxiyasi tizimini amalga oshiring:

1. **Employee** interfeysi - getName(), getSalary(), showDetails() bilan Component
2. **Developer** klassi - Leaf xodim (bo'ysunuvchilarsiz)
3. **Manager** klassi - Boshqa xodimlarni boshqaradigan Composite

## Foydalanish namunasi

\`\`\`java
Developer dev1 = new Developer("Alice", 5000);	// leaf xodim yaratamiz
Developer dev2 = new Developer("Bob", 4500);	// yana bir leaf yaratamiz
Manager manager = new Manager("Charlie", 7000);	// composite (menejer) yaratamiz

manager.addSubordinate(dev1);	// dasturchini jamoaga qo'shamiz
manager.addSubordinate(dev2);	// boshqa dasturchini qo'shamiz

System.out.println(manager.getSalary());	// Chiqish: 16500 (hammasi yig'indisi)
System.out.println(manager.showDetails());	// daraxt strukturasini ko'rsatadi
\`\`\`

## Asosiy tushuncha

Menejerning getSalary() butun jamoaning umumiy maoshini rekursiv hisoblaydi — bu Composite patternining kuchi!`,
			hint1: `**Developer amalga oshirish (Leaf)**

Developer klassi barg tugunidir — uning bolalari yo'q:

\`\`\`java
class Developer implements Employee {
    private String name;
    private double salary;

    @Override
    public String getName() {
        return name;  // Oddiy getter
    }

    @Override
    public double getSalary() {
        return salary;  // Shunchaki o'z maoshini qaytaradi
    }

    @Override
    public String showDetails() {
        return "Developer: " + name + ", Salary: " + salary;
    }
}
\`\`\`

Asosiy nuqtalar:
- Barg tugunlarining bolalari yo'q
- Barcha metodlar oddiy amalga oshirishlar
- Delegatsiya yoki rekursiya kerak emas`,
			hint2: `**Manager amalga oshirish (Composite)**

Manager kompozitdir — u boshqa Employee larni o'z ichiga oladi:

\`\`\`java
class Manager implements Employee {
    private List<Employee> subordinates = new ArrayList<>();

    public void addSubordinate(Employee e) {
        subordinates.add(e);
    }

    @Override
    public double getSalary() {
        double total = salary;  // O'z maoshidan boshlaymiz
        for (Employee e : subordinates) {
            total += e.getSalary();  // Har bir bo'ysunuvchi maoshini qo'shamiz
        }
        return total;
    }

    @Override
    public String showDetails() {
        StringBuilder sb = new StringBuilder();
        sb.append("Manager: ").append(name)
          .append(", Team Salary: ").append(getSalary());
        for (Employee e : subordinates) {
            sb.append("\\n  - ").append(e.showDetails());
        }
        return sb.toString();
    }
}
\`\`\`

e.getSalary() rekursiv chaqiruvi ishlaydi, chunki bo'ysunuvchilar Developer YOKI boshqa Manager bo'lishi mumkin!`,
			whyItMatters: `## Muammo va yechim

**Composite siz:**
\`\`\`java
// Barglar va konteynerlarni turlicha boshqarish kerak
if (node instanceof File) {
    return ((File) node).getSize();	// faylni boshqarish
} else if (node instanceof Folder) {
    long size = 0;	// papka hajmini hisoblash
    for (Object child : ((Folder) node).getChildren()) {
        // O'sha instanceof tekshiruvlarini takrorlash...	// chalkash rekursiya
    }
}
\`\`\`

**Composite bilan:**
\`\`\`java
interface FileSystemNode {
    long getSize();	// yagona interfeys
}
// File ham, Folder ham getSize() ni amalga oshiradi
node.getSize();	// ikkalasi uchun ham bir xil ishlaydi!
\`\`\`

---

## Haqiqiy dunyo namunalari

| Domen | Leaf | Composite | Operatsiya |
|-------|------|-----------|------------|
| **Fayl tizimi** | File | Directory | getSize() |
| **UI Framework** | Button, Label | Panel, Window | render() |
| **Tashkilot** | Employee | Department | getSalary() |
| **Grafika** | Shape | Group | draw() |
| **Menyu tizimi** | MenuItem | Menu | execute() |
| **XML/HTML** | Text Node | Element | getText() |

---

## Production pattern: Fayl tizimi

\`\`\`java
interface FileSystemNode {	// Component interfeysi
    String getName();	// tugun nomini olish
    long getSize();	// hajmni olish (direktoriyalar uchun rekursiv)
    void print(String indent);	// daraxt strukturasini ko'rsatish
}

class File implements FileSystemNode {	// Leaf - haqiqiy fayl
    private String name;	// fayl nomi
    private long size;	// hajm baytlarda

    public File(String name, long size) {	// konstruktor
        this.name = name;	// nomni saqlash
        this.size = size;	// hajmni saqlash
    }

    @Override
    public String getName() { return name; }	// fayl nomini qaytarish

    @Override
    public long getSize() { return size; }	// fayl hajmini qaytarish

    @Override
    public void print(String indent) {	// fayl ma'lumotini chop etish
        System.out.println(indent + "📄 " + name + " (" + size + " bytes)");	// ikonka bilan ko'rsatish
    }
}

class Directory implements FileSystemNode {	// Composite - bolalari bor papka
    private String name;	// direktoriya nomi
    private List<FileSystemNode> children = new ArrayList<>();	// bola tugunlar

    public Directory(String name) {	// konstruktor
        this.name = name;	// nomni saqlash
    }

    public void add(FileSystemNode node) {	// bolani qo'shish
        children.add(node);	// bolalar ro'yxatiga qo'shish
    }

    public void remove(FileSystemNode node) {	// bolani o'chirish
        children.remove(node);	// ro'yxatdan o'chirish
    }

    @Override
    public String getName() { return name; }	// direktoriya nomini qaytarish

    @Override
    public long getSize() {	// umumiy hajmni hisoblash
        long total = 0;	// hisoblagichni ishga tushirish
        for (FileSystemNode child : children) {	// bolalarni takrorlash
            total += child.getSize();	// bola hajmini qo'shish (rekursiya!)
        }
        return total;	// umumiy hajmni qaytarish
    }

    @Override
    public void print(String indent) {	// direktoriya daraxtini chop etish
        System.out.println(indent + "📁 " + name + "/");	// papkani chop etish
        for (FileSystemNode child : children) {	// bolalarni takrorlash
            child.print(indent + "  ");	// kattaroq chekinish bilan chop etish
        }
    }
}

// Foydalanish:
Directory root = new Directory("project");	// ildiz direktoriyasini yaratish
Directory src = new Directory("src");	// src papkasini yaratish
Directory docs = new Directory("docs");	// docs papkasini yaratish

src.add(new File("Main.java", 2048));	// manba faylini qo'shish
src.add(new File("Utils.java", 1024));	// utils faylini qo'shish
docs.add(new File("README.md", 512));	// readme qo'shish

root.add(src);	// src ni ildizga qo'shish
root.add(docs);	// docs ni ildizga qo'shish
root.add(new File("pom.xml", 256));	// konfig faylini qo'shish

root.print("");	// butun daraxtni chop etish
System.out.println("Total size: " + root.getSize());	// 3840 bayt
\`\`\`

---

## Keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Tip tekshirish** | Farqlash uchun instanceof ishlatish | Component interfeysi orqali polimorfizmdan foydalaning |
| **Bola operatsiyalari yo'q** | Composite da add/remove ni unutish | Kompozitlarga bolalarni boshqarishni kiriting |
| **Barglarda xavfli operatsiyalar** | Leaf da add() chaqirish xato beradi | E'tiborsiz qoldiring yoki UnsupportedOperationException tashling |
| **Siklik havolalar** | Ota-onani bola sifatida qo'shish | Qo'shishda sikllarni tekshiring |
| **Rekursiyani unutish** | Faqat bevosita bolalarni hisoblash | Operatsiyalar butun daraxtni aylanib o'tishini ta'minlang |`
		}
	}
};

export default task;
