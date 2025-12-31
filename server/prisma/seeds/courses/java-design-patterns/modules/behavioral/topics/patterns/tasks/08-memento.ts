import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-memento',
	title: 'Memento Pattern',
	difficulty: 'medium',
	tags: ['java', 'design-patterns', 'behavioral', 'memento'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Memento Pattern

The **Memento Pattern** captures and externalizes an object's internal state without violating encapsulation, allowing the object to be restored to this state later. It's the foundation of undo/redo functionality.

---

### Key Components

| Component | Role |
|-----------|------|
| **Originator** | Object whose state needs to be saved/restored |
| **Memento** | Stores the internal state of the Originator |
| **Caretaker** | Manages memento history (save/retrieve) |

---

### Your Task

Implement a **Text Editor with Undo** using the Memento pattern:

1. **EditorMemento** (Memento): Immutable snapshot of editor content
2. **Editor** (Originator): Creates and restores from mementos
3. **History** (Caretaker): Manages stack of mementos

---

### Example Usage

\`\`\`java
Editor editor = new Editor();	// create the originator (text editor)
History history = new History();	// create the caretaker (history manager)

editor.setContent("Hello");	// set initial content
history.push(editor.save());	// save state: "Hello"

editor.setContent("Hello World");	// modify content
history.push(editor.save());	// save state: "Hello World"

editor.setContent("Hello World!");	// modify again
System.out.println(editor.getContent());	// "Hello World!"

// Undo: restore previous state
editor.restore(history.pop());	// restore to "Hello World"
System.out.println(editor.getContent());	// "Hello World"

editor.restore(history.pop());	// restore to "Hello"
System.out.println(editor.getContent());	// "Hello"
\`\`\`

---

### Key Insight

> The Memento is opaque to the Caretaker - it cannot access or modify the stored state. Only the Originator can create mementos and restore from them, preserving encapsulation.`,
	initialCode: `import java.util.Stack;

class EditorMemento {
    private final String content;

    public EditorMemento(String content) {
    }

    public String getContent() { return content; }
}

class Editor {
    private String content = "";

    public void setContent(String content) {
    }

    public String getContent() { return content; }

    public EditorMemento save() {
        throw new UnsupportedOperationException("TODO");
    }

    public void restore(EditorMemento memento) {
        throw new UnsupportedOperationException("TODO");
    }
}

class History {
    private Stack<EditorMemento> history = new Stack<>();

    public void push(EditorMemento memento) {
        throw new UnsupportedOperationException("TODO");
    }

    public EditorMemento pop() {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `import java.util.Stack;	// import Stack for history management

// Memento - immutable snapshot of Originator's state
class EditorMemento {	// stores editor state without exposing it
    private final String content;	// the saved state (immutable)

    public EditorMemento(String content) {	// constructor - captures state
        this.content = content;	// store the content snapshot
    }

    public String getContent() { return content; }	// getter for restoration
}

// Originator - object whose state needs saving
class Editor {	// text editor that creates/restores mementos
    private String content = "";	// internal state to be saved

    public void setContent(String content) {	// modify the state
        this.content = content;	// update internal state
    }

    public String getContent() { return content; }	// get current state

    public EditorMemento save() {	// create memento from current state
        return new EditorMemento(content);	// capture current content in memento
    }

    public void restore(EditorMemento memento) {	// restore state from memento
        if (memento != null) {	// check for valid memento
            this.content = memento.getContent();	// restore content from memento
        }
    }
}

// Caretaker - manages memento history
class History {	// stores and retrieves mementos without examining them
    private Stack<EditorMemento> history = new Stack<>();	// LIFO stack for undo

    public void push(EditorMemento memento) {	// save a memento
        history.push(memento);	// add to top of stack
    }

    public EditorMemento pop() {	// retrieve most recent memento
        if (history.isEmpty()) {	// check if history exists
            return null;	// nothing to undo
        }
        return history.pop();	// remove and return top memento
    }
}`,
	hint1: `## Hint 1: Originator (Editor) Methods

The Originator creates mementos and restores from them:

\`\`\`java
public EditorMemento save() {	// capture current state
    return new EditorMemento(content);	// create memento with current content
}

public void restore(EditorMemento memento) {	// restore previous state
    if (memento != null) {	// null check for safety
        this.content = memento.getContent();	// restore from memento
    }
}
\`\`\`

**save()** creates a snapshot of current state. **restore()** sets state from a memento. Always null-check the memento before restoring.`,
	hint2: `## Hint 2: Caretaker (History) Methods

The Caretaker manages mementos but never examines their contents:

\`\`\`java
public void push(EditorMemento memento) {	// store memento
    history.push(memento);	// simply add to stack
}

public EditorMemento pop() {	// retrieve memento
    if (history.isEmpty()) {	// check if anything to undo
        return null;	// no history available
    }
    return history.pop();	// return most recent memento
}
\`\`\`

The Caretaker treats mementos as opaque tokens - it stores and retrieves them but doesn't look inside. This preserves encapsulation.`,
	whyItMatters: `## Why Memento Pattern Matters

### The Problem: Exposing Internal State Breaks Encapsulation

Without Memento, implementing undo requires exposing internal state:

\`\`\`java
// ❌ Without Memento - violates encapsulation
class Editor {	// editor with exposed internals
    public String content = "";	// PUBLIC - anyone can access/modify
    public int cursorPosition = 0;	// PUBLIC - breaks encapsulation
    public Map<String, Object> formatting = new HashMap<>();	// PUBLIC - dangerous
}

class History {	// history tightly coupled to Editor internals
    private List<String> contentHistory = new ArrayList<>();	// must know about content
    private List<Integer> cursorHistory = new ArrayList<>();	// must know about cursor
    private List<Map<String, Object>> formatHistory = new ArrayList<>();	// must know about formatting

    public void save(Editor editor) {	// coupled to Editor's structure
        contentHistory.add(editor.content);	// directly access internal state
        cursorHistory.add(editor.cursorPosition);	// if Editor changes, History breaks
        formatHistory.add(new HashMap<>(editor.formatting));	// maintenance nightmare
    }

    public void restore(Editor editor, int index) {	// restore is complex
        editor.content = contentHistory.get(index);	// directly modify internal state
        editor.cursorPosition = cursorHistory.get(index);	// tightly coupled
        editor.formatting = new HashMap<>(formatHistory.get(index));	// hard to maintain
    }
}

// Problems:
// 1. Editor internals are PUBLIC - no encapsulation
// 2. History must know all fields - tightly coupled
// 3. Adding new fields requires updating History
// 4. History can corrupt Editor state
\`\`\`

\`\`\`java
// ✅ With Memento - encapsulation preserved
class EditorMemento {	// opaque snapshot - hides internal structure
    private final String content;	// PRIVATE - only accessible by Editor
    private final int cursorPosition;	// PRIVATE - encapsulated
    private final Map<String, Object> formatting;	// PRIVATE - safe

    EditorMemento(String content, int cursor, Map<String, Object> fmt) {	// package-private constructor
        this.content = content;	// store snapshot
        this.cursorPosition = cursor;	// only Editor can create
        this.formatting = new HashMap<>(fmt);	// defensive copy
    }
    // Getters are package-private - only Originator can access
}

class Editor {	// Originator - creates and uses mementos
    private String content = "";	// PRIVATE - encapsulated
    private int cursorPosition = 0;	// PRIVATE - hidden from outside
    private Map<String, Object> formatting = new HashMap<>();	// PRIVATE - safe

    public EditorMemento save() {	// only Editor creates mementos
        return new EditorMemento(content, cursorPosition, formatting);	// capture all state
    }

    public void restore(EditorMemento m) {	// only Editor restores from mementos
        this.content = m.content;	// access allowed - same package
        this.cursorPosition = m.cursorPosition;	// restore all state
        this.formatting = new HashMap<>(m.formatting);	// defensive copy
    }
}

class History {	// Caretaker - knows nothing about memento contents
    private Stack<EditorMemento> history = new Stack<>();	// just stores mementos

    public void push(EditorMemento m) { history.push(m); }	// store opaque token
    public EditorMemento pop() { return history.isEmpty() ? null : history.pop(); }	// retrieve token
    // History doesn't know what's inside EditorMemento!
}
\`\`\`

---

### Real-World Applications

| Application | Originator | Memento | Caretaker |
|-------------|------------|---------|-----------|
| **Text Editor** | Document | DocumentSnapshot | UndoManager |
| **Game Save** | GameState | SaveFile | SaveManager |
| **Database** | Connection | Transaction | TransactionLog |
| **Drawing App** | Canvas | CanvasState | HistoryStack |
| **Form Wizard** | FormData | FormSnapshot | WizardNavigator |

---

### Production Pattern: Document Editor with Full Undo/Redo

\`\`\`java
import java.util.*;	// import utilities for collections

// Memento - immutable snapshot of document state
class DocumentMemento {	// stores complete document state
    private final String content;	// text content
    private final int cursorPosition;	// cursor location
    private final int selectionStart;	// selection start position
    private final int selectionEnd;	// selection end position
    private final Map<String, String> metadata;	// document metadata
    private final long timestamp;	// when snapshot was created

    DocumentMemento(String content, int cursor, int selStart, int selEnd,
                    Map<String, String> metadata) {	// package-private constructor
        this.content = content;	// store content
        this.cursorPosition = cursor;	// store cursor position
        this.selectionStart = selStart;	// store selection bounds
        this.selectionEnd = selEnd;	// store selection bounds
        this.metadata = new HashMap<>(metadata);	// defensive copy
        this.timestamp = System.currentTimeMillis();	// record creation time
    }

    // Package-private getters - only Originator can access
    String getContent() { return content; }	// get saved content
    int getCursorPosition() { return cursorPosition; }	// get saved cursor
    int getSelectionStart() { return selectionStart; }	// get selection start
    int getSelectionEnd() { return selectionEnd; }	// get selection end
    Map<String, String> getMetadata() { return new HashMap<>(metadata); }	// defensive copy
    long getTimestamp() { return timestamp; }	// get creation time
}

// Originator - the document being edited
class Document {	// creates and restores from mementos
    private String content = "";	// document text
    private int cursorPosition = 0;	// current cursor location
    private int selectionStart = 0;	// selection start
    private int selectionEnd = 0;	// selection end
    private Map<String, String> metadata = new HashMap<>();	// document properties

    public void type(String text) {	// type text at cursor
        String before = content.substring(0, cursorPosition);	// text before cursor
        String after = content.substring(cursorPosition);	// text after cursor
        content = before + text + after;	// insert text
        cursorPosition += text.length();	// move cursor after typed text
        clearSelection();	// clear any selection
    }

    public void delete(int count) {	// delete characters after cursor
        if (cursorPosition + count <= content.length()) {	// bounds check
            String before = content.substring(0, cursorPosition);	// keep before
            String after = content.substring(cursorPosition + count);	// skip deleted
            content = before + after;	// join remaining
        }
    }

    public void moveCursor(int position) {	// move cursor to position
        this.cursorPosition = Math.max(0, Math.min(position, content.length()));	// clamp to valid range
    }

    public void select(int start, int end) {	// select text range
        this.selectionStart = Math.max(0, start);	// clamp start
        this.selectionEnd = Math.min(end, content.length());	// clamp end
    }

    private void clearSelection() {	// clear selection
        selectionStart = selectionEnd = cursorPosition;	// selection is empty at cursor
    }

    public void setMetadata(String key, String value) {	// set document property
        metadata.put(key, value);	// store in metadata map
    }

    public String getContent() { return content; }	// get current content
    public int getCursorPosition() { return cursorPosition; }	// get cursor position

    public DocumentMemento save() {	// create memento from current state
        return new DocumentMemento(content, cursorPosition,
            selectionStart, selectionEnd, metadata);	// capture all state
    }

    public void restore(DocumentMemento memento) {	// restore from memento
        if (memento != null) {	// null check
            this.content = memento.getContent();	// restore content
            this.cursorPosition = memento.getCursorPosition();	// restore cursor
            this.selectionStart = memento.getSelectionStart();	// restore selection
            this.selectionEnd = memento.getSelectionEnd();	// restore selection
            this.metadata = memento.getMetadata();	// restore metadata
        }
    }
}

// Caretaker - manages undo/redo stacks
class UndoManager {	// handles history without knowing memento contents
    private Stack<DocumentMemento> undoStack = new Stack<>();	// for undo operations
    private Stack<DocumentMemento> redoStack = new Stack<>();	// for redo operations
    private static final int MAX_HISTORY = 100;	// limit history size

    public void saveState(DocumentMemento memento) {	// save current state before change
        undoStack.push(memento);	// add to undo stack
        redoStack.clear();	// clear redo stack on new action
        if (undoStack.size() > MAX_HISTORY) {	// enforce size limit
            undoStack.remove(0);	// remove oldest entry
        }
    }

    public DocumentMemento undo(DocumentMemento current) {	// undo last action
        if (undoStack.isEmpty()) {	// nothing to undo
            return null;	// return null to indicate no undo available
        }
        redoStack.push(current);	// save current state for redo
        return undoStack.pop();	// return previous state
    }

    public DocumentMemento redo(DocumentMemento current) {	// redo undone action
        if (redoStack.isEmpty()) {	// nothing to redo
            return null;	// return null to indicate no redo available
        }
        undoStack.push(current);	// save current state for undo
        return redoStack.pop();	// return next state
    }

    public boolean canUndo() { return !undoStack.isEmpty(); }	// check if undo available
    public boolean canRedo() { return !redoStack.isEmpty(); }	// check if redo available
    public int getUndoCount() { return undoStack.size(); }	// number of undo steps
    public int getRedoCount() { return redoStack.size(); }	// number of redo steps
}

// Usage:
Document doc = new Document();	// create document (originator)
UndoManager undoManager = new UndoManager();	// create undo manager (caretaker)

// Type some text
undoManager.saveState(doc.save());	// save state before change
doc.type("Hello");	// type "Hello"

undoManager.saveState(doc.save());	// save state before change
doc.type(" World");	// type " World"

System.out.println(doc.getContent());	// "Hello World"
System.out.println(undoManager.canUndo());	// true

// Undo
DocumentMemento previous = undoManager.undo(doc.save());	// get previous state
doc.restore(previous);	// restore it
System.out.println(doc.getContent());	// "Hello"

// Redo
DocumentMemento next = undoManager.redo(doc.save());	// get next state
doc.restore(next);	// restore it
System.out.println(doc.getContent());	// "Hello World"
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Mutable mementos | State can be corrupted after saving | Make memento fields final, use defensive copies |
| Exposing memento internals | Breaks encapsulation | Make getters package-private or use inner class |
| No history limit | Memory leak with many operations | Implement max history size, remove oldest |
| Storing in Originator | Wrong responsibility | Caretaker stores mementos, not Originator |
| Saving after every keystroke | Too many snapshots, memory waste | Batch changes, save on meaningful actions |`,
	order: 7,
	testCode: `import org.junit.Test;
import static org.junit.Assert.*;

// Test1: Editor save creates memento
class Test1 {
    @Test
    public void test() {
        Editor editor = new Editor();
        editor.setContent("Hello");
        EditorMemento memento = editor.save();
        assertNotNull(memento);
    }
}

// Test2: Memento stores content correctly
class Test2 {
    @Test
    public void test() {
        Editor editor = new Editor();
        editor.setContent("Test");
        EditorMemento memento = editor.save();
        assertEquals("Test", memento.getContent());
    }
}

// Test3: Editor restore restores content
class Test3 {
    @Test
    public void test() {
        Editor editor = new Editor();
        editor.setContent("Original");
        EditorMemento memento = editor.save();
        editor.setContent("Changed");
        editor.restore(memento);
        assertEquals("Original", editor.getContent());
    }
}

// Test4: History push stores memento
class Test4 {
    @Test
    public void test() {
        History history = new History();
        EditorMemento memento = new EditorMemento("Test");
        history.push(memento);
        EditorMemento popped = history.pop();
        assertNotNull(popped);
    }
}

// Test5: History pop returns last memento
class Test5 {
    @Test
    public void test() {
        History history = new History();
        history.push(new EditorMemento("First"));
        history.push(new EditorMemento("Second"));
        EditorMemento popped = history.pop();
        assertEquals("Second", popped.getContent());
    }
}

// Test6: History pop returns null when empty
class Test6 {
    @Test
    public void test() {
        History history = new History();
        EditorMemento popped = history.pop();
        assertNull(popped);
    }
}

// Test7: Restore with null memento is safe
class Test7 {
    @Test
    public void test() {
        Editor editor = new Editor();
        editor.setContent("Original");
        editor.restore(null);
        assertEquals("Original", editor.getContent());
    }
}

// Test8: Multiple save/restore cycle
class Test8 {
    @Test
    public void test() {
        Editor editor = new Editor();
        History history = new History();

        editor.setContent("V1");
        history.push(editor.save());

        editor.setContent("V2");
        history.push(editor.save());

        editor.setContent("V3");

        editor.restore(history.pop());
        assertEquals("V2", editor.getContent());

        editor.restore(history.pop());
        assertEquals("V1", editor.getContent());
    }
}

// Test9: Editor getContent returns current content
class Test9 {
    @Test
    public void test() {
        Editor editor = new Editor();
        assertEquals("", editor.getContent());
        editor.setContent("Hello");
        assertEquals("Hello", editor.getContent());
    }
}

// Test10: EditorMemento is immutable
class Test10 {
    @Test
    public void test() {
        EditorMemento memento = new EditorMemento("Fixed");
        assertEquals("Fixed", memento.getContent());
        assertEquals("Fixed", memento.getContent());
    }
}`,
	translations: {
		ru: {
			title: 'Паттерн Memento (Снимок)',
			description: `## Паттерн Memento (Снимок)

Паттерн **Memento** захватывает и выносит наружу внутреннее состояние объекта, не нарушая инкапсуляцию, позволяя позже восстановить объект в это состояние. Это основа функциональности отмены/повтора.

---

### Ключевые компоненты

| Компонент | Роль |
|-----------|------|
| **Originator** | Объект, состояние которого нужно сохранять/восстанавливать |
| **Memento** | Хранит внутреннее состояние Originator |
| **Caretaker** | Управляет историей memento (сохранение/получение) |

---

### Ваша задача

Реализуйте **Текстовый редактор с отменой** используя паттерн Memento:

1. **EditorMemento** (Memento): Неизменяемый снимок содержимого редактора
2. **Editor** (Originator): Создаёт и восстанавливается из memento
3. **History** (Caretaker): Управляет стеком memento

---

### Пример использования

\`\`\`java
Editor editor = new Editor();	// создаём originator (текстовый редактор)
History history = new History();	// создаём caretaker (менеджер истории)

editor.setContent("Hello");	// устанавливаем начальное содержимое
history.push(editor.save());	// сохраняем состояние: "Hello"

editor.setContent("Hello World");	// изменяем содержимое
history.push(editor.save());	// сохраняем состояние: "Hello World"

editor.setContent("Hello World!");	// изменяем снова
System.out.println(editor.getContent());	// "Hello World!"

// Отмена: восстанавливаем предыдущее состояние
editor.restore(history.pop());	// восстанавливаем к "Hello World"
System.out.println(editor.getContent());	// "Hello World"

editor.restore(history.pop());	// восстанавливаем к "Hello"
System.out.println(editor.getContent());	// "Hello"
\`\`\`

---

### Ключевая идея

> Memento непрозрачен для Caretaker - он не может получить доступ или изменить сохранённое состояние. Только Originator может создавать memento и восстанавливаться из них, сохраняя инкапсуляцию.`,
			hint1: `## Подсказка 1: Методы Originator (Editor)

Originator создаёт memento и восстанавливается из них:

\`\`\`java
public EditorMemento save() {	// захватить текущее состояние
    return new EditorMemento(content);	// создать memento с текущим содержимым
}

public void restore(EditorMemento memento) {	// восстановить предыдущее состояние
    if (memento != null) {	// проверка на null для безопасности
        this.content = memento.getContent();	// восстановить из memento
    }
}
\`\`\`

**save()** создаёт снимок текущего состояния. **restore()** устанавливает состояние из memento. Всегда проверяйте memento на null перед восстановлением.`,
			hint2: `## Подсказка 2: Методы Caretaker (History)

Caretaker управляет memento, но никогда не изучает их содержимое:

\`\`\`java
public void push(EditorMemento memento) {	// сохранить memento
    history.push(memento);	// просто добавить в стек
}

public EditorMemento pop() {	// получить memento
    if (history.isEmpty()) {	// проверить есть ли что отменять
        return null;	// история недоступна
    }
    return history.pop();	// вернуть самый последний memento
}
\`\`\`

Caretaker обращается с memento как с непрозрачными токенами - он сохраняет и извлекает их, но не заглядывает внутрь. Это сохраняет инкапсуляцию.`,
			whyItMatters: `## Почему паттерн Memento важен

### Проблема: Раскрытие внутреннего состояния нарушает инкапсуляцию

Без Memento реализация отмены требует раскрытия внутреннего состояния:

\`\`\`java
// ❌ Без Memento - нарушает инкапсуляцию
class Editor {	// редактор с открытыми внутренностями
    public String content = "";	// PUBLIC - любой может получить доступ/изменить
    public int cursorPosition = 0;	// PUBLIC - нарушает инкапсуляцию
    public Map<String, Object> formatting = new HashMap<>();	// PUBLIC - опасно
}

class History {	// история тесно связана с внутренностями Editor
    private List<String> contentHistory = new ArrayList<>();	// должна знать о content
    private List<Integer> cursorHistory = new ArrayList<>();	// должна знать о cursor
    private List<Map<String, Object>> formatHistory = new ArrayList<>();	// должна знать о formatting

    public void save(Editor editor) {	// связана со структурой Editor
        contentHistory.add(editor.content);	// прямой доступ к внутреннему состоянию
        cursorHistory.add(editor.cursorPosition);	// если Editor изменится, History сломается
        formatHistory.add(new HashMap<>(editor.formatting));	// кошмар поддержки
    }

    public void restore(Editor editor, int index) {	// восстановление сложное
        editor.content = contentHistory.get(index);	// прямое изменение внутреннего состояния
        editor.cursorPosition = cursorHistory.get(index);	// тесно связано
        editor.formatting = new HashMap<>(formatHistory.get(index));	// трудно поддерживать
    }
}

// Проблемы:
// 1. Внутренности Editor PUBLIC - нет инкапсуляции
// 2. History должна знать все поля - тесная связанность
// 3. Добавление новых полей требует обновления History
// 4. History может испортить состояние Editor
\`\`\`

\`\`\`java
// ✅ С Memento - инкапсуляция сохранена
class EditorMemento {	// непрозрачный снимок - скрывает внутреннюю структуру
    private final String content;	// PRIVATE - доступно только для Editor
    private final int cursorPosition;	// PRIVATE - инкапсулировано
    private final Map<String, Object> formatting;	// PRIVATE - безопасно

    EditorMemento(String content, int cursor, Map<String, Object> fmt) {	// пакетный конструктор
        this.content = content;	// сохранить снимок
        this.cursorPosition = cursor;	// только Editor может создать
        this.formatting = new HashMap<>(fmt);	// защитная копия
    }
    // Геттеры пакетно-приватные - только Originator может получить доступ
}

class Editor {	// Originator - создаёт и использует memento
    private String content = "";	// PRIVATE - инкапсулировано
    private int cursorPosition = 0;	// PRIVATE - скрыто от внешнего мира
    private Map<String, Object> formatting = new HashMap<>();	// PRIVATE - безопасно

    public EditorMemento save() {	// только Editor создаёт memento
        return new EditorMemento(content, cursorPosition, formatting);	// захватить всё состояние
    }

    public void restore(EditorMemento m) {	// только Editor восстанавливается из memento
        this.content = m.content;	// доступ разрешён - тот же пакет
        this.cursorPosition = m.cursorPosition;	// восстановить всё состояние
        this.formatting = new HashMap<>(m.formatting);	// защитная копия
    }
}

class History {	// Caretaker - ничего не знает о содержимом memento
    private Stack<EditorMemento> history = new Stack<>();	// просто хранит memento

    public void push(EditorMemento m) { history.push(m); }	// сохранить непрозрачный токен
    public EditorMemento pop() { return history.isEmpty() ? null : history.pop(); }	// получить токен
    // History не знает что внутри EditorMemento!
}
\`\`\`

---

### Применение в реальном мире

| Применение | Originator | Memento | Caretaker |
|------------|------------|---------|-----------|
| **Текстовый редактор** | Document | DocumentSnapshot | UndoManager |
| **Сохранение игры** | GameState | SaveFile | SaveManager |
| **База данных** | Connection | Transaction | TransactionLog |
| **Графический редактор** | Canvas | CanvasState | HistoryStack |
| **Мастер форм** | FormData | FormSnapshot | WizardNavigator |

---

### Продакшн паттерн: Редактор документов с полной отменой/повтором

\`\`\`java
import java.util.*;	// импорт утилит для коллекций

// Memento - неизменяемый снимок состояния документа
class DocumentMemento {	// хранит полное состояние документа
    private final String content;	// текстовое содержимое
    private final int cursorPosition;	// позиция курсора
    private final int selectionStart;	// начало выделения
    private final int selectionEnd;	// конец выделения
    private final Map<String, String> metadata;	// метаданные документа
    private final long timestamp;	// когда создан снимок

    DocumentMemento(String content, int cursor, int selStart, int selEnd,
                    Map<String, String> metadata) {	// пакетно-приватный конструктор
        this.content = content;	// сохранить содержимое
        this.cursorPosition = cursor;	// сохранить позицию курсора
        this.selectionStart = selStart;	// сохранить границы выделения
        this.selectionEnd = selEnd;	// сохранить границы выделения
        this.metadata = new HashMap<>(metadata);	// защитная копия
        this.timestamp = System.currentTimeMillis();	// записать время создания
    }

    // Пакетно-приватные геттеры - только Originator может получить доступ
    String getContent() { return content; }	// получить сохранённое содержимое
    int getCursorPosition() { return cursorPosition; }	// получить сохранённый курсор
    int getSelectionStart() { return selectionStart; }	// получить начало выделения
    int getSelectionEnd() { return selectionEnd; }	// получить конец выделения
    Map<String, String> getMetadata() { return new HashMap<>(metadata); }	// защитная копия
    long getTimestamp() { return timestamp; }	// получить время создания
}

// Originator - редактируемый документ
class Document {	// создаёт и восстанавливается из memento
    private String content = "";	// текст документа
    private int cursorPosition = 0;	// текущая позиция курсора
    private int selectionStart = 0;	// начало выделения
    private int selectionEnd = 0;	// конец выделения
    private Map<String, String> metadata = new HashMap<>();	// свойства документа

    public void type(String text) {	// набрать текст в позиции курсора
        String before = content.substring(0, cursorPosition);	// текст до курсора
        String after = content.substring(cursorPosition);	// текст после курсора
        content = before + text + after;	// вставить текст
        cursorPosition += text.length();	// переместить курсор после набранного текста
        clearSelection();	// очистить выделение
    }

    public void delete(int count) {	// удалить символы после курсора
        if (cursorPosition + count <= content.length()) {	// проверка границ
            String before = content.substring(0, cursorPosition);	// оставить до
            String after = content.substring(cursorPosition + count);	// пропустить удалённое
            content = before + after;	// соединить оставшееся
        }
    }

    public void moveCursor(int position) {	// переместить курсор в позицию
        this.cursorPosition = Math.max(0, Math.min(position, content.length()));	// ограничить допустимым диапазоном
    }

    public void select(int start, int end) {	// выделить диапазон текста
        this.selectionStart = Math.max(0, start);	// ограничить начало
        this.selectionEnd = Math.min(end, content.length());	// ограничить конец
    }

    private void clearSelection() {	// очистить выделение
        selectionStart = selectionEnd = cursorPosition;	// выделение пусто в позиции курсора
    }

    public void setMetadata(String key, String value) {	// установить свойство документа
        metadata.put(key, value);	// сохранить в карте метаданных
    }

    public String getContent() { return content; }	// получить текущее содержимое
    public int getCursorPosition() { return cursorPosition; }	// получить позицию курсора

    public DocumentMemento save() {	// создать memento из текущего состояния
        return new DocumentMemento(content, cursorPosition,
            selectionStart, selectionEnd, metadata);	// захватить всё состояние
    }

    public void restore(DocumentMemento memento) {	// восстановить из memento
        if (memento != null) {	// проверка на null
            this.content = memento.getContent();	// восстановить содержимое
            this.cursorPosition = memento.getCursorPosition();	// восстановить курсор
            this.selectionStart = memento.getSelectionStart();	// восстановить выделение
            this.selectionEnd = memento.getSelectionEnd();	// восстановить выделение
            this.metadata = memento.getMetadata();	// восстановить метаданные
        }
    }
}

// Caretaker - управляет стеками отмены/повтора
class UndoManager {	// работает с историей не зная содержимого memento
    private Stack<DocumentMemento> undoStack = new Stack<>();	// для операций отмены
    private Stack<DocumentMemento> redoStack = new Stack<>();	// для операций повтора
    private static final int MAX_HISTORY = 100;	// ограничение размера истории

    public void saveState(DocumentMemento memento) {	// сохранить текущее состояние перед изменением
        undoStack.push(memento);	// добавить в стек отмены
        redoStack.clear();	// очистить стек повтора при новом действии
        if (undoStack.size() > MAX_HISTORY) {	// применить ограничение размера
            undoStack.remove(0);	// удалить самую старую запись
        }
    }

    public DocumentMemento undo(DocumentMemento current) {	// отменить последнее действие
        if (undoStack.isEmpty()) {	// нечего отменять
            return null;	// вернуть null чтобы показать что отмена недоступна
        }
        redoStack.push(current);	// сохранить текущее состояние для повтора
        return undoStack.pop();	// вернуть предыдущее состояние
    }

    public DocumentMemento redo(DocumentMemento current) {	// повторить отменённое действие
        if (redoStack.isEmpty()) {	// нечего повторять
            return null;	// вернуть null чтобы показать что повтор недоступен
        }
        undoStack.push(current);	// сохранить текущее состояние для отмены
        return redoStack.pop();	// вернуть следующее состояние
    }

    public boolean canUndo() { return !undoStack.isEmpty(); }	// проверить доступна ли отмена
    public boolean canRedo() { return !redoStack.isEmpty(); }	// проверить доступен ли повтор
    public int getUndoCount() { return undoStack.size(); }	// количество шагов отмены
    public int getRedoCount() { return redoStack.size(); }	// количество шагов повтора
}

// Использование:
Document doc = new Document();	// создать документ (originator)
UndoManager undoManager = new UndoManager();	// создать менеджер отмены (caretaker)

// Набираем текст
undoManager.saveState(doc.save());	// сохранить состояние перед изменением
doc.type("Hello");	// набрать "Hello"

undoManager.saveState(doc.save());	// сохранить состояние перед изменением
doc.type(" World");	// набрать " World"

System.out.println(doc.getContent());	// "Hello World"
System.out.println(undoManager.canUndo());	// true

// Отмена
DocumentMemento previous = undoManager.undo(doc.save());	// получить предыдущее состояние
doc.restore(previous);	// восстановить его
System.out.println(doc.getContent());	// "Hello"

// Повтор
DocumentMemento next = undoManager.redo(doc.save());	// получить следующее состояние
doc.restore(next);	// восстановить его
System.out.println(doc.getContent());	// "Hello World"
\`\`\`

---

### Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| Изменяемые memento | Состояние может быть повреждено после сохранения | Сделать поля memento final, использовать защитные копии |
| Раскрытие внутренностей memento | Нарушает инкапсуляцию | Сделать геттеры пакетно-приватными или использовать внутренний класс |
| Нет ограничения истории | Утечка памяти при многих операциях | Реализовать максимальный размер истории, удалять старейшие |
| Хранение в Originator | Неправильная ответственность | Caretaker хранит memento, не Originator |
| Сохранение после каждого нажатия | Слишком много снимков, трата памяти | Группировать изменения, сохранять на значимых действиях |`
		},
		uz: {
			title: 'Memento Pattern',
			description: `## Memento Pattern

**Memento Pattern** ob'ektning ichki holatini inkapsulyatsiyani buzmasdan tashqariga chiqaradi va saqlaydi, bu ob'ektni keyinchalik ushbu holatga qaytarish imkonini beradi. Bu bekor qilish/qaytarish funksiyasining asosi.

---

### Asosiy komponentlar

| Komponent | Vazifa |
|-----------|--------|
| **Originator** | Holati saqlanishi/tiklanishi kerak bo'lgan ob'ekt |
| **Memento** | Originator ning ichki holatini saqlaydi |
| **Caretaker** | Memento tarixini boshqaradi (saqlash/olish) |

---

### Vazifangiz

Memento patternidan foydalanib **Bekor qilishli matn muharriri** amalga oshiring:

1. **EditorMemento** (Memento): Muharrir kontentining o'zgarmas surati
2. **Editor** (Originator): Memento larni yaratadi va ulardan tiklanadi
3. **History** (Caretaker): Memento lar stekini boshqaradi

---

### Foydalanish namunasi

\`\`\`java
Editor editor = new Editor();	// originator (matn muharriri) yaratish
History history = new History();	// caretaker (tarix menejeri) yaratish

editor.setContent("Hello");	// boshlang'ich kontentni o'rnatish
history.push(editor.save());	// holatni saqlash: "Hello"

editor.setContent("Hello World");	// kontentni o'zgartirish
history.push(editor.save());	// holatni saqlash: "Hello World"

editor.setContent("Hello World!");	// yana o'zgartirish
System.out.println(editor.getContent());	// "Hello World!"

// Bekor qilish: oldingi holatni tiklash
editor.restore(history.pop());	// "Hello World" ga tiklash
System.out.println(editor.getContent());	// "Hello World"

editor.restore(history.pop());	// "Hello" ga tiklash
System.out.println(editor.getContent());	// "Hello"
\`\`\`

---

### Asosiy tushuncha

> Memento Caretaker uchun noaniq - u saqlangan holatga kira olmaydi yoki o'zgartira olmaydi. Faqat Originator memento lar yaratishi va ulardan tiklanishi mumkin, inkapsulyatsiyani saqlab.`,
			hint1: `## Maslahat 1: Originator (Editor) metodlari

Originator memento lar yaratadi va ulardan tiklanadi:

\`\`\`java
public EditorMemento save() {	// joriy holatni saqlash
    return new EditorMemento(content);	// joriy kontent bilan memento yaratish
}

public void restore(EditorMemento memento) {	// oldingi holatni tiklash
    if (memento != null) {	// xavfsizlik uchun null tekshiruv
        this.content = memento.getContent();	// mementodan tiklash
    }
}
\`\`\`

**save()** joriy holatning suratini yaratadi. **restore()** mementodan holatni o'rnatadi. Tiklashdan oldin doimo mementoni null ga tekshiring.`,
			hint2: `## Maslahat 2: Caretaker (History) metodlari

Caretaker memento larni boshqaradi lekin hech qachon ularning ichidagini ko'rmaydi:

\`\`\`java
public void push(EditorMemento memento) {	// mementoni saqlash
    history.push(memento);	// shunchaki stekga qo'shish
}

public EditorMemento pop() {	// mementoni olish
    if (history.isEmpty()) {	// bekor qiladigan narsa bormi tekshirish
        return null;	// tarix mavjud emas
    }
    return history.pop();	// eng so'nggi mementoni qaytarish
}
\`\`\`

Caretaker memento larga noaniq tokenlar sifatida munosabatda bo'ladi - ularni saqlaydi va oladi lekin ichiga qaramaydi. Bu inkapsulyatsiyani saqlaydi.`,
			whyItMatters: `## Nima uchun Memento Pattern muhim

### Muammo: Ichki holatni ochish inkapsulyatsiyani buzadi

Mementosiz bekor qilishni amalga oshirish ichki holatni ochishni talab qiladi:

\`\`\`java
// ❌ Mementosiz - inkapsulyatsiyani buzadi
class Editor {	// ochiq ichki qismlarga ega muharrir
    public String content = "";	// PUBLIC - har kim kira oladi/o'zgartira oladi
    public int cursorPosition = 0;	// PUBLIC - inkapsulyatsiyani buzadi
    public Map<String, Object> formatting = new HashMap<>();	// PUBLIC - xavfli
}

class History {	// tarix Editor ichki qismlariga qattiq bog'langan
    private List<String> contentHistory = new ArrayList<>();	// content haqida bilishi kerak
    private List<Integer> cursorHistory = new ArrayList<>();	// cursor haqida bilishi kerak
    private List<Map<String, Object>> formatHistory = new ArrayList<>();	// formatting haqida bilishi kerak

    public void save(Editor editor) {	// Editor strukturasiga bog'langan
        contentHistory.add(editor.content);	// ichki holatga to'g'ridan-to'g'ri kirish
        cursorHistory.add(editor.cursorPosition);	// Editor o'zgarsa, History buziladi
        formatHistory.add(new HashMap<>(editor.formatting));	// qo'llab-quvvatlash dahshati
    }

    public void restore(Editor editor, int index) {	// tiklash murakkab
        editor.content = contentHistory.get(index);	// ichki holatni to'g'ridan-to'g'ri o'zgartirish
        editor.cursorPosition = cursorHistory.get(index);	// qattiq bog'langan
        editor.formatting = new HashMap<>(formatHistory.get(index));	// qo'llab-quvvatlash qiyin
    }
}

// Muammolar:
// 1. Editor ichki qismlari PUBLIC - inkapsulyatsiya yo'q
// 2. History barcha maydonlarni bilishi kerak - qattiq bog'lanish
// 3. Yangi maydonlar qo'shish History ni yangilashni talab qiladi
// 4. History Editor holatini buzishi mumkin
\`\`\`

\`\`\`java
// ✅ Memento bilan - inkapsulyatsiya saqlanadi
class EditorMemento {	// noaniq surat - ichki strukturani yashiradi
    private final String content;	// PRIVATE - faqat Editor uchun mavjud
    private final int cursorPosition;	// PRIVATE - inkapsulyatsiyalangan
    private final Map<String, Object> formatting;	// PRIVATE - xavfsiz

    EditorMemento(String content, int cursor, Map<String, Object> fmt) {	// paket-private konstruktor
        this.content = content;	// suratni saqlash
        this.cursorPosition = cursor;	// faqat Editor yarata oladi
        this.formatting = new HashMap<>(fmt);	// himoyaviy nusxa
    }
    // Getterlar paket-private - faqat Originator kira oladi
}

class Editor {	// Originator - memento lar yaratadi va ishlatadi
    private String content = "";	// PRIVATE - inkapsulyatsiyalangan
    private int cursorPosition = 0;	// PRIVATE - tashqaridan yashirilgan
    private Map<String, Object> formatting = new HashMap<>();	// PRIVATE - xavfsiz

    public EditorMemento save() {	// faqat Editor memento lar yaratadi
        return new EditorMemento(content, cursorPosition, formatting);	// barcha holatni saqlash
    }

    public void restore(EditorMemento m) {	// faqat Editor memento lardan tiklanadi
        this.content = m.content;	// kirish ruxsat etilgan - bir xil paket
        this.cursorPosition = m.cursorPosition;	// barcha holatni tiklash
        this.formatting = new HashMap<>(m.formatting);	// himoyaviy nusxa
    }
}

class History {	// Caretaker - memento ichidagilari haqida hech narsa bilmaydi
    private Stack<EditorMemento> history = new Stack<>();	// shunchaki memento larni saqlaydi

    public void push(EditorMemento m) { history.push(m); }	// noaniq tokenni saqlash
    public EditorMemento pop() { return history.isEmpty() ? null : history.pop(); }	// tokenni olish
    // History EditorMemento ichida nima borligini bilmaydi!
}
\`\`\`

---

### Haqiqiy dunyo qo'llanilishi

| Qo'llanilish | Originator | Memento | Caretaker |
|--------------|------------|---------|-----------|
| **Matn muharriri** | Document | DocumentSnapshot | UndoManager |
| **O'yin saqlash** | GameState | SaveFile | SaveManager |
| **Ma'lumotlar bazasi** | Connection | Transaction | TransactionLog |
| **Rasm muharriri** | Canvas | CanvasState | HistoryStack |
| **Forma ustasi** | FormData | FormSnapshot | WizardNavigator |

---

### Production Pattern: To'liq bekor qilish/qaytarishli hujjat muharriri

\`\`\`java
import java.util.*;	// to'plamlar uchun utilitalarni import qilish

// Memento - hujjat holatining o'zgarmas surati
class DocumentMemento {	// to'liq hujjat holatini saqlaydi
    private final String content;	// matn kontenti
    private final int cursorPosition;	// kursor joyi
    private final int selectionStart;	// tanlash boshi pozitsiyasi
    private final int selectionEnd;	// tanlash oxiri pozitsiyasi
    private final Map<String, String> metadata;	// hujjat metama'lumotlari
    private final long timestamp;	// surat qachon yaratilgani

    DocumentMemento(String content, int cursor, int selStart, int selEnd,
                    Map<String, String> metadata) {	// paket-private konstruktor
        this.content = content;	// kontentni saqlash
        this.cursorPosition = cursor;	// kursor pozitsiyasini saqlash
        this.selectionStart = selStart;	// tanlash chegaralarini saqlash
        this.selectionEnd = selEnd;	// tanlash chegaralarini saqlash
        this.metadata = new HashMap<>(metadata);	// himoyaviy nusxa
        this.timestamp = System.currentTimeMillis();	// yaratilish vaqtini yozib olish
    }

    // Paket-private getterlar - faqat Originator kira oladi
    String getContent() { return content; }	// saqlangan kontentni olish
    int getCursorPosition() { return cursorPosition; }	// saqlangan kursorni olish
    int getSelectionStart() { return selectionStart; }	// tanlash boshini olish
    int getSelectionEnd() { return selectionEnd; }	// tanlash oxirini olish
    Map<String, String> getMetadata() { return new HashMap<>(metadata); }	// himoyaviy nusxa
    long getTimestamp() { return timestamp; }	// yaratilish vaqtini olish
}

// Originator - tahrirlanayotgan hujjat
class Document {	// memento larni yaratadi va ulardan tiklanadi
    private String content = "";	// hujjat matni
    private int cursorPosition = 0;	// joriy kursor joyi
    private int selectionStart = 0;	// tanlash boshi
    private int selectionEnd = 0;	// tanlash oxiri
    private Map<String, String> metadata = new HashMap<>();	// hujjat xususiyatlari

    public void type(String text) {	// kursor joyida matn yozish
        String before = content.substring(0, cursorPosition);	// kursordan oldingi matn
        String after = content.substring(cursorPosition);	// kursordan keyingi matn
        content = before + text + after;	// matnni kiritish
        cursorPosition += text.length();	// kursorni yozilgan matndan keyin ko'chirish
        clearSelection();	// tanlashni tozalash
    }

    public void delete(int count) {	// kursordan keyingi belgilarni o'chirish
        if (cursorPosition + count <= content.length()) {	// chegaralarni tekshirish
            String before = content.substring(0, cursorPosition);	// oldinni saqlash
            String after = content.substring(cursorPosition + count);	// o'chirilganni o'tkazib yuborish
            content = before + after;	// qolganini birlashtirish
        }
    }

    public void moveCursor(int position) {	// kursorni pozitsiyaga ko'chirish
        this.cursorPosition = Math.max(0, Math.min(position, content.length()));	// to'g'ri diapazon ga cheklash
    }

    public void select(int start, int end) {	// matn diapazonini tanlash
        this.selectionStart = Math.max(0, start);	// boshini cheklash
        this.selectionEnd = Math.min(end, content.length());	// oxirini cheklash
    }

    private void clearSelection() {	// tanlashni tozalash
        selectionStart = selectionEnd = cursorPosition;	// tanlash kursor joyida bo'sh
    }

    public void setMetadata(String key, String value) {	// hujjat xususiyatini o'rnatish
        metadata.put(key, value);	// metadata xaritasida saqlash
    }

    public String getContent() { return content; }	// joriy kontentni olish
    public int getCursorPosition() { return cursorPosition; }	// kursor pozitsiyasini olish

    public DocumentMemento save() {	// joriy holatdan memento yaratish
        return new DocumentMemento(content, cursorPosition,
            selectionStart, selectionEnd, metadata);	// barcha holatni saqlash
    }

    public void restore(DocumentMemento memento) {	// mementodan tiklash
        if (memento != null) {	// null tekshiruv
            this.content = memento.getContent();	// kontentni tiklash
            this.cursorPosition = memento.getCursorPosition();	// kursorni tiklash
            this.selectionStart = memento.getSelectionStart();	// tanlashni tiklash
            this.selectionEnd = memento.getSelectionEnd();	// tanlashni tiklash
            this.metadata = memento.getMetadata();	// metama'lumotlarni tiklash
        }
    }
}

// Caretaker - bekor qilish/qaytarish steklarini boshqaradi
class UndoManager {	// memento ichidagilarini bilmasdan tarix bilan ishlaydi
    private Stack<DocumentMemento> undoStack = new Stack<>();	// bekor qilish operatsiyalari uchun
    private Stack<DocumentMemento> redoStack = new Stack<>();	// qaytarish operatsiyalari uchun
    private static final int MAX_HISTORY = 100;	// tarix hajmini cheklash

    public void saveState(DocumentMemento memento) {	// o'zgartirishdan oldin joriy holatni saqlash
        undoStack.push(memento);	// bekor qilish stekiga qo'shish
        redoStack.clear();	// yangi amalda qaytarish stekini tozalash
        if (undoStack.size() > MAX_HISTORY) {	// hajm chegarasini qo'llash
            undoStack.remove(0);	// eng eski yozuvni olib tashlash
        }
    }

    public DocumentMemento undo(DocumentMemento current) {	// oxirgi amalni bekor qilish
        if (undoStack.isEmpty()) {	// bekor qiladigan narsa yo'q
            return null;	// bekor qilish mavjud emasligini ko'rsatish uchun null qaytarish
        }
        redoStack.push(current);	// qaytarish uchun joriy holatni saqlash
        return undoStack.pop();	// oldingi holatni qaytarish
    }

    public DocumentMemento redo(DocumentMemento current) {	// bekor qilingan amalni qaytarish
        if (redoStack.isEmpty()) {	// qaytaradigan narsa yo'q
            return null;	// qaytarish mavjud emasligini ko'rsatish uchun null qaytarish
        }
        undoStack.push(current);	// bekor qilish uchun joriy holatni saqlash
        return redoStack.pop();	// keyingi holatni qaytarish
    }

    public boolean canUndo() { return !undoStack.isEmpty(); }	// bekor qilish mavjudligini tekshirish
    public boolean canRedo() { return !redoStack.isEmpty(); }	// qaytarish mavjudligini tekshirish
    public int getUndoCount() { return undoStack.size(); }	// bekor qilish qadamlari soni
    public int getRedoCount() { return redoStack.size(); }	// qaytarish qadamlari soni
}

// Foydalanish:
Document doc = new Document();	// hujjat yaratish (originator)
UndoManager undoManager = new UndoManager();	// bekor qilish menejeri yaratish (caretaker)

// Matn yozish
undoManager.saveState(doc.save());	// o'zgartirishdan oldin holatni saqlash
doc.type("Hello");	// "Hello" yozish

undoManager.saveState(doc.save());	// o'zgartirishdan oldin holatni saqlash
doc.type(" World");	// " World" yozish

System.out.println(doc.getContent());	// "Hello World"
System.out.println(undoManager.canUndo());	// true

// Bekor qilish
DocumentMemento previous = undoManager.undo(doc.save());	// oldingi holatni olish
doc.restore(previous);	// uni tiklash
System.out.println(doc.getContent());	// "Hello"

// Qaytarish
DocumentMemento next = undoManager.redo(doc.save());	// keyingi holatni olish
doc.restore(next);	// uni tiklash
System.out.println(doc.getContent());	// "Hello World"
\`\`\`

---

### Oldini olish kerak bo'lgan keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| O'zgaruvchan memento lar | Holat saqlangandan keyin buzilishi mumkin | Memento maydonlarini final qilish, himoyaviy nusxalar ishlatish |
| Memento ichki qismlarini ochish | Inkapsulyatsiyani buzadi | Getterlarni paket-private qilish yoki ichki klass ishlatish |
| Tarix chegarasi yo'q | Ko'p operatsiyalarda xotira oqishi | Maksimal tarix hajmini amalga oshirish, eng eskisini o'chirish |
| Originator da saqlash | Noto'g'ri mas'uliyat | Caretaker memento larni saqlaydi, Originator emas |
| Har bir tugma bosishdan keyin saqlash | Juda ko'p suratlar, xotira isrofi | O'zgarishlarni guruhlash, ma'noli amallarda saqlash |`
		}
	}
};

export default task;
