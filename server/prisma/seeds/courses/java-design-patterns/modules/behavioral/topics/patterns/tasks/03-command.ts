import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-command',
	title: 'Command Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'command'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Command Pattern

The **Command** pattern encapsulates a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

---

### Key Components

| Component | Description |
|-----------|-------------|
| **Command** | Interface declaring \`execute()\` and \`undo()\` methods |
| **ConcreteCommand** | Implements Command; binds Receiver with an action |
| **Receiver** | Knows how to perform the actual work |
| **Invoker** | Asks the command to execute; stores command history |

---

### Your Task

Implement a text editor with undo functionality:

1. **Command interface** - \`execute()\` and \`undo()\` methods
2. **TextEditor (Receiver)** - has \`type()\` and \`delete()\` methods
3. **TypeCommand** - executes \`type()\`, undo executes \`delete()\`
4. **EditorInvoker** - executes commands and maintains history for undo

---

### Example Usage

\`\`\`java
TextEditor editor = new TextEditor();	// create the receiver (actual text editor)
EditorInvoker invoker = new EditorInvoker();	// create the invoker (command manager)

Command typeHello = new TypeCommand(editor, "Hello ");	// create command to type "Hello "
invoker.executeCommand(typeHello);	// execute command, adds to history
// editor.getText() = "Hello "

Command typeWorld = new TypeCommand(editor, "World");	// create another command
invoker.executeCommand(typeWorld);	// execute, adds to history
// editor.getText() = "Hello World"

invoker.undo();	// undo last command (TypeCommand for "World")
// editor.getText() = "Hello "

invoker.undo();	// undo previous command (TypeCommand for "Hello ")
// editor.getText() = ""

invoker.undo();	// returns "Nothing to undo" - history is empty
\`\`\`

---

### Key Insight

Command decouples the object that invokes the operation from the object that knows how to perform it. This enables features like **undo/redo**, **macro recording**, **transaction logging**, and **delayed execution**.`,
	initialCode: `import java.util.Stack;

interface Command {
    String execute();
    String undo();
}

class TextEditor {
    private StringBuilder text = new StringBuilder();

    public String type(String words) {
    }

    public String delete(int count) {
        int len = text.length();
        int deleteFrom = Math.max(0, len - count);
        String deleted = text.substring(deleteFrom);
    }

    public String getText() { return text.toString(); }
}

class TypeCommand implements Command {
    private TextEditor editor;
    private String text;

    public TypeCommand(TextEditor editor, String text) {
    }

    @Override
    public String execute() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public String undo() { throw new UnsupportedOperationException("TODO"); }
}

class EditorInvoker {
    private Stack<Command> history = new Stack<>();

    public String executeCommand(Command cmd) {
        throw new UnsupportedOperationException("TODO");
    }

    public String undo() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `import java.util.Stack;	// import Stack for command history

interface Command {	// Command interface - defines execute and undo contract
    String execute();	// perform the command action
    String undo();	// reverse the command action
}

class TextEditor {	// Receiver - knows how to perform actual operations
    private StringBuilder text = new StringBuilder();	// text content storage

    public String type(String words) {	// add text to the editor
        text.append(words);	// append the words
        return "Typed: " + words;	// return confirmation
    }

    public String delete(int count) {	// delete characters from end
        int len = text.length();	// current text length
        int deleteFrom = Math.max(0, len - count);	// calculate start position
        String deleted = text.substring(deleteFrom);	// get text being deleted
        text.delete(deleteFrom, len);	// remove the text
        return "Deleted: " + deleted;	// return what was deleted
    }

    public String getText() { return text.toString(); }	// get current text
}

class TypeCommand implements Command {	// ConcreteCommand - encapsulates type action
    private TextEditor editor;	// reference to receiver
    private String text;	// text to type (needed for undo)

    public TypeCommand(TextEditor editor, String text) {	// constructor binds receiver and data
        this.editor = editor;	// store receiver reference
        this.text = text;	// store text for this command
    }

    @Override
    public String execute() {	// execute the type operation
        return editor.type(text);	// delegate to receiver
    }

    @Override
    public String undo() {	// reverse the type operation
        return editor.delete(text.length());	// delete same number of chars that were typed
    }
}

class EditorInvoker {	// Invoker - executes commands and manages history
    private Stack<Command> history = new Stack<>();	// command history stack for undo

    public String executeCommand(Command cmd) {	// execute a command
        String result = cmd.execute();	// perform the command
        history.push(cmd);	// add to history for undo
        return result;	// return execution result
    }

    public String undo() {	// undo last command
        if (history.isEmpty()) {	// check if history exists
            return "Nothing to undo";	// no commands to undo
        }
        Command cmd = history.pop();	// get last command
        return cmd.undo();	// execute its undo method
    }
}`,
	hint1: `### Understanding Command Structure

The Command pattern has four main parts:

\`\`\`java
// 1. Command interface - contract for all commands
interface Command {
    String execute();	// Do the action
    String undo();	// Reverse the action
}

// 2. Receiver - knows how to do the actual work
class TextEditor {
    public String type(String words) { ... }
    public String delete(int count) { ... }
}

// 3. ConcreteCommand - binds receiver with action
class TypeCommand implements Command {
    private TextEditor editor;	// Receiver reference
    private String text;	// Data for this command

    @Override
    public String execute() {
        return editor.type(text);	// Delegate to receiver
    }

    @Override
    public String undo() {
        return editor.delete(text.length());	// Reverse by deleting
    }
}
\`\`\``,
	hint2: `### Implementing EditorInvoker with History

The invoker executes commands and maintains history for undo:

\`\`\`java
class EditorInvoker {
    private Stack<Command> history = new Stack<>();	// Store executed commands

    public String executeCommand(Command cmd) {
        // 1. Execute the command
        String result = cmd.execute();

        // 2. Add to history (for undo)
        history.push(cmd);

        // 3. Return result
        return result;
    }

    public String undo() {
        // 1. Check if there's anything to undo
        if (history.isEmpty()) {
            return "Nothing to undo";
        }

        // 2. Get last command from history
        Command cmd = history.pop();

        // 3. Call its undo method
        return cmd.undo();
    }
}
\`\`\`

The key is that each command stores enough information to reverse itself.`,
	whyItMatters: `## Why Command Pattern Matters

### The Problem and Solution

**Without Command:**
\`\`\`java
// Tight coupling, no undo capability
class EditorUI {
    private TextEditor editor;

    public void onTypeButton(String text) {
        editor.type(text);	// direct call, no history
    }

    public void onDeleteButton(int count) {
        editor.delete(count);	// direct call, no history
    }

    // How to implement undo? Need to track every action manually!
}
\`\`\`

**With Command:**
\`\`\`java
// Decoupled, with undo support
class EditorUI {
    private EditorInvoker invoker;
    private TextEditor editor;

    public void onTypeButton(String text) {
        invoker.executeCommand(new TypeCommand(editor, text));	// command object
    }

    public void onDeleteButton(int count) {
        invoker.executeCommand(new DeleteCommand(editor, count));	// command object
    }

    public void onUndoButton() {
        invoker.undo();	// automatic undo from history!
    }
}
\`\`\`

---

### Real-World Applications

| Application | Command | Receiver |
|-------------|---------|----------|
| **Runnable** | Runnable.run() | Thread execution |
| **Swing Actions** | Action.actionPerformed() | UI components |
| **Database** | Transaction commands | Database connection |
| **Game** | Move, Attack commands | Game entities |
| **Smart Home** | TurnOn, TurnOff | Devices |

---

### Production Pattern: Transaction System

\`\`\`java
// Command interface with transaction support
interface TransactionCommand {	// command with rollback capability
    void execute() throws Exception;	// perform transaction step
    void rollback();	// reverse on failure
    String getDescription();	// for logging
}

class TransferMoneyCommand implements TransactionCommand {	// concrete transaction command
    private final Account from;	// source account
    private final Account to;	// destination account
    private final Money amount;	// transfer amount
    private boolean executed = false;	// track execution state

    public TransferMoneyCommand(Account from, Account to, Money amount) {	// constructor
        this.from = from;	// store source
        this.to = to;	// store destination
        this.amount = amount;	// store amount
    }

    @Override
    public void execute() throws Exception {	// perform transfer
        from.withdraw(amount);	// withdraw from source
        to.deposit(amount);	// deposit to destination
        executed = true;	// mark as executed
    }

    @Override
    public void rollback() {	// reverse the transfer
        if (executed) {	// only rollback if executed
            to.withdraw(amount);	// withdraw from destination
            from.deposit(amount);	// deposit back to source
            executed = false;	// mark as rolled back
        }
    }

    @Override
    public String getDescription() {	// for logging
        return "Transfer " + amount + " from " + from.getId() + " to " + to.getId();
    }
}

class TransactionManager {	// invoker with rollback support
    private final List<TransactionCommand> executedCommands = new ArrayList<>();	// executed commands
    private final Logger logger;	// for audit logging

    public void executeTransaction(List<TransactionCommand> commands) throws TransactionException {
        executedCommands.clear();	// start fresh

        for (TransactionCommand cmd : commands) {	// execute each command
            try {
                logger.info("Executing: " + cmd.getDescription());	// log action
                cmd.execute();	// execute command
                executedCommands.add(cmd);	// track for rollback
            } catch (Exception e) {	// if any command fails
                logger.error("Failed: " + cmd.getDescription());	// log failure
                rollbackAll();	// rollback all executed commands
                throw new TransactionException("Transaction failed", e);	// propagate error
            }
        }
        logger.info("Transaction completed successfully");	// log success
    }

    private void rollbackAll() {	// rollback all executed commands in reverse order
        for (int i = executedCommands.size() - 1; i >= 0; i--) {	// reverse iteration
            TransactionCommand cmd = executedCommands.get(i);	// get command
            logger.info("Rolling back: " + cmd.getDescription());	// log rollback
            cmd.rollback();	// execute rollback
        }
    }
}

// Usage:
TransactionManager manager = new TransactionManager();	// create manager
List<TransactionCommand> commands = Arrays.asList(	// create command list
    new TransferMoneyCommand(accountA, accountB, new Money(100)),	// first transfer
    new TransferMoneyCommand(accountB, accountC, new Money(50)),	// second transfer
    new DeductFeeCommand(accountA, new Money(5))	// fee deduction
);
manager.executeTransaction(commands);	// execute all or rollback all on failure
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Not storing undo data** | Can't undo properly | Store all data needed to reverse action |
| **Mutable command state** | Incorrect undo after multiple executions | Make commands immutable or track execution count |
| **No null checks** | NullPointerException on empty history | Check before pop/undo |
| **Memory leaks** | Unbounded history growth | Limit history size or implement clear() |
| **Partial execution** | Inconsistent state on failure | Implement proper rollback mechanism |`,
	order: 2,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: TypeCommand execute types text
class Test1 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        Command cmd = new TypeCommand(editor, "Hello");
        cmd.execute();
        assertEquals("Hello", editor.getText());
    }
}

// Test2: TypeCommand undo removes text
class Test2 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        Command cmd = new TypeCommand(editor, "World");
        cmd.execute();
        cmd.undo();
        assertEquals("", editor.getText());
    }
}

// Test3: EditorInvoker executeCommand works
class Test3 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        EditorInvoker invoker = new EditorInvoker();
        invoker.executeCommand(new TypeCommand(editor, "Test"));
        assertEquals("Test", editor.getText());
    }
}

// Test4: EditorInvoker undo reverses command
class Test4 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        EditorInvoker invoker = new EditorInvoker();
        invoker.executeCommand(new TypeCommand(editor, "ABC"));
        invoker.undo();
        assertEquals("", editor.getText());
    }
}

// Test5: Multiple commands and undo
class Test5 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        EditorInvoker invoker = new EditorInvoker();
        invoker.executeCommand(new TypeCommand(editor, "Hello "));
        invoker.executeCommand(new TypeCommand(editor, "World"));
        assertEquals("Hello World", editor.getText());
        invoker.undo();
        assertEquals("Hello ", editor.getText());
    }
}

// Test6: Undo on empty history
class Test6 {
    @Test
    public void test() {
        EditorInvoker invoker = new EditorInvoker();
        String result = invoker.undo();
        assertEquals("Nothing to undo", result);
    }
}

// Test7: Execute returns typed message
class Test7 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        Command cmd = new TypeCommand(editor, "Java");
        String result = cmd.execute();
        assertEquals("Typed: Java", result);
    }
}

// Test8: Undo returns deleted message
class Test8 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        Command cmd = new TypeCommand(editor, "Code");
        cmd.execute();
        String result = cmd.undo();
        assertEquals("Deleted: Code", result);
    }
}

// Test9: Multiple undos clear all text
class Test9 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        EditorInvoker invoker = new EditorInvoker();
        invoker.executeCommand(new TypeCommand(editor, "A"));
        invoker.executeCommand(new TypeCommand(editor, "B"));
        invoker.executeCommand(new TypeCommand(editor, "C"));
        invoker.undo();
        invoker.undo();
        invoker.undo();
        assertEquals("", editor.getText());
    }
}

// Test10: TextEditor delete removes from end
class Test10 {
    @Test
    public void test() {
        TextEditor editor = new TextEditor();
        editor.type("Hello World");
        editor.delete(5);
        assertEquals("Hello ", editor.getText());
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Command (Команда)',
			description: `## Паттерн Command (Команда)

Паттерн **Command** инкапсулирует запрос как объект, позволяя параметризовать клиенты различными запросами, ставить запросы в очередь или логировать их, а также поддерживать отменяемые операции.

---

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **Command** | Интерфейс, объявляющий методы \`execute()\` и \`undo()\` |
| **ConcreteCommand** | Реализует Command; связывает Receiver с действием |
| **Receiver** | Знает, как выполнить фактическую работу |
| **Invoker** | Просит команду выполниться; хранит историю команд |

---

### Ваша задача

Реализуйте текстовый редактор с функцией отмены:

1. **Интерфейс Command** - методы \`execute()\` и \`undo()\`
2. **TextEditor (Receiver)** - имеет методы \`type()\` и \`delete()\`
3. **TypeCommand** - execute вызывает \`type()\`, undo вызывает \`delete()\`
4. **EditorInvoker** - выполняет команды и хранит историю для отмены

---

### Пример использования

\`\`\`java
TextEditor editor = new TextEditor();	// создаём получателя (текстовый редактор)
EditorInvoker invoker = new EditorInvoker();	// создаём инвокер (менеджер команд)

Command typeHello = new TypeCommand(editor, "Hello ");	// создаём команду для ввода "Hello "
invoker.executeCommand(typeHello);	// выполняем команду, добавляем в историю
// editor.getText() = "Hello "

Command typeWorld = new TypeCommand(editor, "World");	// создаём другую команду
invoker.executeCommand(typeWorld);	// выполняем, добавляем в историю
// editor.getText() = "Hello World"

invoker.undo();	// отменяем последнюю команду (TypeCommand для "World")
// editor.getText() = "Hello "

invoker.undo();	// отменяем предыдущую команду (TypeCommand для "Hello ")
// editor.getText() = ""

invoker.undo();	// возвращает "Nothing to undo" - история пуста
\`\`\`

---

### Ключевая идея

Command отделяет объект, вызывающий операцию, от объекта, который знает, как её выполнить. Это позволяет реализовать **отмену/повтор**, **запись макросов**, **логирование транзакций** и **отложенное выполнение**.`,
			hint1: `### Понимание структуры Command

Паттерн Command состоит из четырёх основных частей:

\`\`\`java
// 1. Интерфейс Command - контракт для всех команд
interface Command {
    String execute();	// Выполнить действие
    String undo();	// Отменить действие
}

// 2. Receiver - знает, как выполнить работу
class TextEditor {
    public String type(String words) { ... }
    public String delete(int count) { ... }
}

// 3. ConcreteCommand - связывает получателя с действием
class TypeCommand implements Command {
    private TextEditor editor;	// Ссылка на получателя
    private String text;	// Данные для этой команды

    @Override
    public String execute() {
        return editor.type(text);	// Делегируем получателю
    }

    @Override
    public String undo() {
        return editor.delete(text.length());	// Отменяем удалением
    }
}
\`\`\``,
			hint2: `### Реализация EditorInvoker с историей

Инвокер выполняет команды и хранит историю для отмены:

\`\`\`java
class EditorInvoker {
    private Stack<Command> history = new Stack<>();	// Хранение выполненных команд

    public String executeCommand(Command cmd) {
        // 1. Выполняем команду
        String result = cmd.execute();

        // 2. Добавляем в историю (для отмены)
        history.push(cmd);

        // 3. Возвращаем результат
        return result;
    }

    public String undo() {
        // 1. Проверяем, есть ли что отменять
        if (history.isEmpty()) {
            return "Nothing to undo";
        }

        // 2. Получаем последнюю команду из истории
        Command cmd = history.pop();

        // 3. Вызываем её метод undo
        return cmd.undo();
    }
}
\`\`\`

Ключевой момент - каждая команда хранит достаточно информации для отмены.`,
			whyItMatters: `## Почему паттерн Command важен

### Проблема и решение

**Без Command:**
\`\`\`java
// Тесная связанность, нет возможности отмены
class EditorUI {
    private TextEditor editor;

    public void onTypeButton(String text) {
        editor.type(text);	// прямой вызов, нет истории
    }

    public void onDeleteButton(int count) {
        editor.delete(count);	// прямой вызов, нет истории
    }

    // Как реализовать отмену? Нужно вручную отслеживать каждое действие!
}
\`\`\`

**С Command:**
\`\`\`java
// Развязка, с поддержкой отмены
class EditorUI {
    private EditorInvoker invoker;
    private TextEditor editor;

    public void onTypeButton(String text) {
        invoker.executeCommand(new TypeCommand(editor, text));	// объект команды
    }

    public void onDeleteButton(int count) {
        invoker.executeCommand(new DeleteCommand(editor, count));	// объект команды
    }

    public void onUndoButton() {
        invoker.undo();	// автоматическая отмена из истории!
    }
}
\`\`\`

---

### Применение в реальном мире

| Применение | Command | Receiver |
|------------|---------|----------|
| **Runnable** | Runnable.run() | Выполнение потока |
| **Swing Actions** | Action.actionPerformed() | UI компоненты |
| **База данных** | Команды транзакций | Соединение с БД |
| **Игры** | Move, Attack команды | Игровые сущности |
| **Умный дом** | TurnOn, TurnOff | Устройства |

---

### Продакшен паттерн: Система транзакций

\`\`\`java
// Интерфейс команды с поддержкой транзакций
interface TransactionCommand {	// команда с возможностью отката
    void execute() throws Exception;	// выполнить шаг транзакции
    void rollback();	// откатить при ошибке
    String getDescription();	// для логирования
}

class TransferMoneyCommand implements TransactionCommand {	// конкретная команда транзакции
    private final Account from;	// счёт-источник
    private final Account to;	// счёт-получатель
    private final Money amount;	// сумма перевода
    private boolean executed = false;	// отслеживание состояния выполнения

    public TransferMoneyCommand(Account from, Account to, Money amount) {	// конструктор
        this.from = from;	// сохраняем источник
        this.to = to;	// сохраняем получателя
        this.amount = amount;	// сохраняем сумму
    }

    @Override
    public void execute() throws Exception {	// выполнить перевод
        from.withdraw(amount);	// снять с источника
        to.deposit(amount);	// зачислить получателю
        executed = true;	// отметить как выполненную
    }

    @Override
    public void rollback() {	// откатить перевод
        if (executed) {	// откатываем только если выполнена
            to.withdraw(amount);	// снять с получателя
            from.deposit(amount);	// вернуть на источник
            executed = false;	// отметить как откаченную
        }
    }

    @Override
    public String getDescription() {	// для логирования
        return "Transfer " + amount + " from " + from.getId() + " to " + to.getId();
    }
}

class TransactionManager {	// инвокер с поддержкой отката
    private final List<TransactionCommand> executedCommands = new ArrayList<>();	// выполненные команды
    private final Logger logger;	// для аудит-логирования

    public void executeTransaction(List<TransactionCommand> commands) throws TransactionException {
        executedCommands.clear();	// начинаем с чистого листа

        for (TransactionCommand cmd : commands) {	// выполняем каждую команду
            try {
                logger.info("Executing: " + cmd.getDescription());	// логируем действие
                cmd.execute();	// выполняем команду
                executedCommands.add(cmd);	// отслеживаем для отката
            } catch (Exception e) {	// если команда не удалась
                logger.error("Failed: " + cmd.getDescription());	// логируем ошибку
                rollbackAll();	// откатываем все выполненные команды
                throw new TransactionException("Transaction failed", e);	// пробрасываем ошибку
            }
        }
        logger.info("Transaction completed successfully");	// логируем успех
    }

    private void rollbackAll() {	// откат всех команд в обратном порядке
        for (int i = executedCommands.size() - 1; i >= 0; i--) {	// обратная итерация
            TransactionCommand cmd = executedCommands.get(i);	// получаем команду
            logger.info("Rolling back: " + cmd.getDescription());	// логируем откат
            cmd.rollback();	// выполняем откат
        }
    }
}

// Использование:
TransactionManager manager = new TransactionManager();	// создаём менеджер
List<TransactionCommand> commands = Arrays.asList(	// создаём список команд
    new TransferMoneyCommand(accountA, accountB, new Money(100)),	// первый перевод
    new TransferMoneyCommand(accountB, accountC, new Money(50)),	// второй перевод
    new DeductFeeCommand(accountA, new Money(5))	// снятие комиссии
);
manager.executeTransaction(commands);	// выполнить все или откатить все при ошибке
\`\`\`

---

### Частые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Не сохраняются данные для отмены** | Невозможно правильно отменить | Храните все данные для отмены действия |
| **Изменяемое состояние команды** | Неправильная отмена после нескольких выполнений | Делайте команды неизменяемыми или отслеживайте число выполнений |
| **Нет проверки на null** | NullPointerException при пустой истории | Проверяйте перед pop/undo |
| **Утечки памяти** | Неограниченный рост истории | Ограничьте размер истории или реализуйте clear() |
| **Частичное выполнение** | Несогласованное состояние при ошибке | Реализуйте правильный механизм отката |`
		},
		uz: {
			title: 'Command Pattern',
			description: `## Command Pattern

**Command** pattern so'rovni obyekt sifatida inkapsulyatsiya qiladi, bu orqali mijozlarni turli so'rovlar bilan parametrlash, so'rovlarni navbatga qo'yish yoki loglash va bekor qilinadigan operatsiyalarni qo'llab-quvvatlash mumkin.

---

### Asosiy Komponentlar

| Komponent | Tavsif |
|-----------|--------|
| **Command** | \`execute()\` va \`undo()\` metodlarini e'lon qiluvchi interfeys |
| **ConcreteCommand** | Command ni amalga oshiradi; Receiver ni harakat bilan bog'laydi |
| **Receiver** | Haqiqiy ishni qanday bajarishni biladi |
| **Invoker** | Buyruqdan bajarishni so'raydi; buyruqlar tarixini saqlaydi |

---

### Vazifangiz

Bekor qilish funksiyasi bilan matn muharririni amalga oshiring:

1. **Command interfeysi** - \`execute()\` va \`undo()\` metodlari
2. **TextEditor (Receiver)** - \`type()\` va \`delete()\` metodlariga ega
3. **TypeCommand** - execute \`type()\` ni, undo \`delete()\` ni chaqiradi
4. **EditorInvoker** - buyruqlarni bajaradi va bekor qilish uchun tarixni saqlaydi

---

### Foydalanish Namunasi

\`\`\`java
TextEditor editor = new TextEditor();	// qabul qiluvchini (matn muharriri) yaratamiz
EditorInvoker invoker = new EditorInvoker();	// invokerni (buyruq menejeri) yaratamiz

Command typeHello = new TypeCommand(editor, "Hello ");	// "Hello " yozish uchun buyruq yaratamiz
invoker.executeCommand(typeHello);	// buyruqni bajaramiz, tarixga qo'shamiz
// editor.getText() = "Hello "

Command typeWorld = new TypeCommand(editor, "World");	// boshqa buyruq yaratamiz
invoker.executeCommand(typeWorld);	// bajaramiz, tarixga qo'shamiz
// editor.getText() = "Hello World"

invoker.undo();	// oxirgi buyruqni bekor qilamiz ("World" uchun TypeCommand)
// editor.getText() = "Hello "

invoker.undo();	// oldingi buyruqni bekor qilamiz ("Hello " uchun TypeCommand)
// editor.getText() = ""

invoker.undo();	// "Nothing to undo" qaytaradi - tarix bo'sh
\`\`\`

---

### Asosiy Fikr

Command operatsiyani chaqiruvchi obyektni uni bajarishni biladigan obyektdan ajratadi. Bu **bekor qilish/qayta bajarish**, **makro yozib olish**, **tranzaksiya loglash** va **kechiktirilgan bajarish** kabi funksiyalarni yoqadi.`,
			hint1: `### Command Strukturasini Tushunish

Command pattern to'rt asosiy qismdan iborat:

\`\`\`java
// 1. Command interfeysi - barcha buyruqlar uchun shartnoma
interface Command {
    String execute();	// Harakatni bajaring
    String undo();	// Harakatni bekor qiling
}

// 2. Receiver - haqiqiy ishni qanday bajarishni biladi
class TextEditor {
    public String type(String words) { ... }
    public String delete(int count) { ... }
}

// 3. ConcreteCommand - qabul qiluvchini harakat bilan bog'laydi
class TypeCommand implements Command {
    private TextEditor editor;	// Qabul qiluvchiga havola
    private String text;	// Ushbu buyruq uchun ma'lumotlar

    @Override
    public String execute() {
        return editor.type(text);	// Qabul qiluvchiga delegatsiya
    }

    @Override
    public String undo() {
        return editor.delete(text.length());	// O'chirish orqali bekor qilish
    }
}
\`\`\``,
			hint2: `### EditorInvoker ni Tarix bilan Amalga Oshirish

Invoker buyruqlarni bajaradi va bekor qilish uchun tarixni saqlaydi:

\`\`\`java
class EditorInvoker {
    private Stack<Command> history = new Stack<>();	// Bajarilgan buyruqlarni saqlash

    public String executeCommand(Command cmd) {
        // 1. Buyruqni bajaramiz
        String result = cmd.execute();

        // 2. Tarixga qo'shamiz (bekor qilish uchun)
        history.push(cmd);

        // 3. Natijani qaytaramiz
        return result;
    }

    public String undo() {
        // 1. Bekor qilish uchun biror narsa borligini tekshiramiz
        if (history.isEmpty()) {
            return "Nothing to undo";
        }

        // 2. Tarixdan oxirgi buyruqni olamiz
        Command cmd = history.pop();

        // 3. Uning undo metodini chaqiramiz
        return cmd.undo();
    }
}
\`\`\`

Asosiy narsa - har bir buyruq o'zini bekor qilish uchun yetarli ma'lumotlarni saqlaydi.`,
			whyItMatters: `## Nima Uchun Command Pattern Muhim

### Muammo va Yechim

**Command siz:**
\`\`\`java
// Qattiq bog'lanish, bekor qilish imkoniyati yo'q
class EditorUI {
    private TextEditor editor;

    public void onTypeButton(String text) {
        editor.type(text);	// to'g'ridan-to'g'ri chaqiriq, tarix yo'q
    }

    public void onDeleteButton(int count) {
        editor.delete(count);	// to'g'ridan-to'g'ri chaqiriq, tarix yo'q
    }

    // Bekor qilishni qanday amalga oshirish? Har bir harakatni qo'lda kuzatish kerak!
}
\`\`\`

**Command bilan:**
\`\`\`java
// Ajratilgan, bekor qilish qo'llab-quvvatlanadi
class EditorUI {
    private EditorInvoker invoker;
    private TextEditor editor;

    public void onTypeButton(String text) {
        invoker.executeCommand(new TypeCommand(editor, text));	// buyruq obyekti
    }

    public void onDeleteButton(int count) {
        invoker.executeCommand(new DeleteCommand(editor, count));	// buyruq obyekti
    }

    public void onUndoButton() {
        invoker.undo();	// tarixdan avtomatik bekor qilish!
    }
}
\`\`\`

---

### Haqiqiy Dunyo Qo'llanilishi

| Qo'llanish | Command | Receiver |
|------------|---------|----------|
| **Runnable** | Runnable.run() | Thread bajarish |
| **Swing Actions** | Action.actionPerformed() | UI komponentlari |
| **Ma'lumotlar bazasi** | Tranzaksiya buyruqlari | DB ulanishi |
| **O'yin** | Move, Attack buyruqlari | O'yin obyektlari |
| **Aqlli uy** | TurnOn, TurnOff | Qurilmalar |

---

### Prodakshen Pattern: Tranzaksiya Tizimi

\`\`\`java
// Tranzaksiya qo'llab-quvvati bilan buyruq interfeysi
interface TransactionCommand {	// orqaga qaytarish imkoniyati bilan buyruq
    void execute() throws Exception;	// tranzaksiya qadamini bajarish
    void rollback();	// muvaffaqiyatsizlikda orqaga qaytarish
    String getDescription();	// loglash uchun
}

class TransferMoneyCommand implements TransactionCommand {	// aniq tranzaksiya buyrug'i
    private final Account from;	// manba hisob
    private final Account to;	// maqsad hisob
    private final Money amount;	// o'tkazma summasi
    private boolean executed = false;	// bajarish holatini kuzatish

    public TransferMoneyCommand(Account from, Account to, Money amount) {	// konstruktor
        this.from = from;	// manbani saqlash
        this.to = to;	// maqsadni saqlash
        this.amount = amount;	// summani saqlash
    }

    @Override
    public void execute() throws Exception {	// o'tkazmani bajarish
        from.withdraw(amount);	// manbadan yechish
        to.deposit(amount);	// maqsadga qo'yish
        executed = true;	// bajarilgan deb belgilash
    }

    @Override
    public void rollback() {	// o'tkazmani orqaga qaytarish
        if (executed) {	// faqat bajarilgan bo'lsa orqaga qaytarish
            to.withdraw(amount);	// maqsaddan yechish
            from.deposit(amount);	// manbaga qaytarish
            executed = false;	// orqaga qaytarilgan deb belgilash
        }
    }

    @Override
    public String getDescription() {	// loglash uchun
        return "Transfer " + amount + " from " + from.getId() + " to " + to.getId();
    }
}

class TransactionManager {	// orqaga qaytarish qo'llab-quvvati bilan invoker
    private final List<TransactionCommand> executedCommands = new ArrayList<>();	// bajarilgan buyruqlar
    private final Logger logger;	// audit loglash uchun

    public void executeTransaction(List<TransactionCommand> commands) throws TransactionException {
        executedCommands.clear();	// yangi boshlash

        for (TransactionCommand cmd : commands) {	// har bir buyruqni bajarish
            try {
                logger.info("Executing: " + cmd.getDescription());	// harakatni loglash
                cmd.execute();	// buyruqni bajarish
                executedCommands.add(cmd);	// orqaga qaytarish uchun kuzatish
            } catch (Exception e) {	// agar biror buyruq muvaffaqiyatsiz bo'lsa
                logger.error("Failed: " + cmd.getDescription());	// xatoni loglash
                rollbackAll();	// barcha bajarilgan buyruqlarni orqaga qaytarish
                throw new TransactionException("Transaction failed", e);	// xatoni tarqatish
            }
        }
        logger.info("Transaction completed successfully");	// muvaffaqiyatni loglash
    }

    private void rollbackAll() {	// barcha buyruqlarni teskari tartibda orqaga qaytarish
        for (int i = executedCommands.size() - 1; i >= 0; i--) {	// teskari iteratsiya
            TransactionCommand cmd = executedCommands.get(i);	// buyruqni olish
            logger.info("Rolling back: " + cmd.getDescription());	// orqaga qaytarishni loglash
            cmd.rollback();	// orqaga qaytarishni bajarish
        }
    }
}

// Foydalanish:
TransactionManager manager = new TransactionManager();	// menejerni yaratish
List<TransactionCommand> commands = Arrays.asList(	// buyruqlar ro'yxatini yaratish
    new TransferMoneyCommand(accountA, accountB, new Money(100)),	// birinchi o'tkazma
    new TransferMoneyCommand(accountB, accountC, new Money(50)),	// ikkinchi o'tkazma
    new DeductFeeCommand(accountA, new Money(5))	// komissiya yechish
);
manager.executeTransaction(commands);	// barchasini bajarish yoki muvaffaqiyatsizlikda barchasini orqaga qaytarish
\`\`\`

---

### Oldini Olish Kerak Bo'lgan Xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **Bekor qilish ma'lumotlari saqlanmaydi** | To'g'ri bekor qilib bo'lmaydi | Harakatni bekor qilish uchun barcha ma'lumotlarni saqlang |
| **O'zgaruvchan buyruq holati** | Bir nechta bajarishdan keyin noto'g'ri bekor qilish | Buyruqlarni o'zgarmas qiling yoki bajarish sonini kuzating |
| **Null tekshiruvlari yo'q** | Bo'sh tarixda NullPointerException | pop/undo dan oldin tekshiring |
| **Xotira oqishi** | Chegarasiz tarix o'sishi | Tarix hajmini cheklang yoki clear() amalga oshiring |
| **Qisman bajarish** | Muvaffaqiyatsizlikda nomuvofiq holat | To'g'ri orqaga qaytarish mexanizmini amalga oshiring |`
		}
	}
};

export default task;
