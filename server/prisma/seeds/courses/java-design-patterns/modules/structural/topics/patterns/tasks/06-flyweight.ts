import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-flyweight',
	title: 'Flyweight Pattern',
	difficulty: 'hard',
	tags: ['java', 'design-patterns', 'structural', 'flyweight'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Flyweight Pattern** in Java — use sharing to support large numbers of fine-grained objects efficiently.

## Overview

The Flyweight pattern minimizes memory usage by sharing common state between multiple objects. It separates intrinsic (shared) state from extrinsic (unique) state.

## Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Intrinsic State** | Shared, immutable data | TreeType (name, color, texture) |
| **Extrinsic State** | Unique, context-specific | Tree position (x, y) |
| **Flyweight Factory** | Creates/caches shared objects | TreeFactory |

## Your Task

Implement a forest rendering system:

1. **TreeType** (Flyweight) - Shared intrinsic state (name, color, texture)
2. **Tree** (Context) - Contains extrinsic state (x, y) and reference to TreeType
3. **TreeFactory** - Creates and caches TreeType instances
4. **Forest** - Manages trees, uses factory

## Example Usage

\`\`\`java
Forest forest = new Forest();	// create forest manager

// Plant 1 million trees - but only few TreeTypes!
forest.plantTree(10, 20, "Oak", "Green", "Rough");	// creates new TreeType
forest.plantTree(30, 40, "Oak", "Green", "Rough");	// reuses existing TreeType
forest.plantTree(50, 60, "Pine", "DarkGreen", "Smooth");	// creates new TreeType

System.out.println(forest.getTreeCount());	// 3 trees
System.out.println(TreeFactory.getTypeCount());	// only 2 TreeTypes!
\`\`\`

## Key Insight

1 million trees can share just 10-20 TreeType objects, saving massive amounts of memory!`,
	initialCode: `import java.util.HashMap;
import java.util.Map;

class TreeType {
    private String name;
    private String color;
    private String texture;

    public TreeType(String name, String color, String texture) {
    }

    public String draw(int x, int y) {
    }
}

class Tree {
    private int x;
    private int y;
    private TreeType type;

    public Tree(int x, int y, TreeType type) {
    }

    public String draw() {
    }
}

class TreeFactory {
    private static Map<String, TreeType> treeTypes = new HashMap<>();

    public static TreeType getTreeType(String name, String color, String texture) {
        throw new UnsupportedOperationException("TODO");
    }

    public static int getTypeCount() {
    }
}

class Forest {
    private java.util.List<Tree> trees = new java.util.ArrayList<>();

    public void plantTree(int x, int y, String name, String color, String texture) {
        throw new UnsupportedOperationException("TODO");
    }

    public int getTreeCount() { return trees.size(); }
}`,
	solutionCode: `import java.util.HashMap;	// for caching flyweights
import java.util.Map;	// map interface

// Flyweight (intrinsic state) - shared between many trees
class TreeType {	// stores only shared, immutable data
    private String name;	// tree species name
    private String color;	// leaf color
    private String texture;	// bark texture

    public TreeType(String name, String color, String texture) {	// constructor
        this.name = name;	// store name
        this.color = color;	// store color
        this.texture = texture;	// store texture
    }

    public String draw(int x, int y) {	// accepts extrinsic state as parameter
        return String.format("Drawing %s tree at (%d,%d)", name, x, y);	// combine intrinsic + extrinsic
    }
}

// Context (extrinsic state) - unique for each tree
class Tree {	// lightweight object - only stores unique data
    private int x;	// unique x position
    private int y;	// unique y position
    private TreeType type;	// reference to shared flyweight

    public Tree(int x, int y, TreeType type) {	// constructor
        this.x = x;	// store position
        this.y = y;	// store position
        this.type = type;	// store reference to shared type
    }

    public String draw() {	// delegate to flyweight
        return type.draw(x, y);	// pass extrinsic state to flyweight
    }
}

// Flyweight Factory - creates and caches flyweights
class TreeFactory {	// ensures flyweight reuse
    private static Map<String, TreeType> treeTypes = new HashMap<>();	// flyweight cache

    public static TreeType getTreeType(String name, String color, String texture) {	// get or create flyweight
        String key = name + "_" + color + "_" + texture;	// create unique key from intrinsic state
        if (!treeTypes.containsKey(key)) {	// check if flyweight exists
            treeTypes.put(key, new TreeType(name, color, texture));	// create and cache new flyweight
        }
        return treeTypes.get(key);	// return cached flyweight
    }

    public static int getTypeCount() {	// get number of unique flyweights
        return treeTypes.size();	// return cache size
    }
}

// Client that uses flyweights
class Forest {	// manages many trees efficiently
    private java.util.List<Tree> trees = new java.util.ArrayList<>();	// list of all trees

    public void plantTree(int x, int y, String name, String color, String texture) {	// plant a tree
        TreeType type = TreeFactory.getTreeType(name, color, texture);	// get shared flyweight
        trees.add(new Tree(x, y, type));	// create tree with unique position + shared type
    }

    public int getTreeCount() { return trees.size(); }	// return total tree count
}`,
	hint1: `**TreeFactory Implementation**

The factory creates a unique key and caches TreeTypes:

\`\`\`java
class TreeFactory {
    private static Map<String, TreeType> treeTypes = new HashMap<>();

    public static TreeType getTreeType(String name, String color, String texture) {
        // Create unique key from all intrinsic state
        String key = name + "_" + color + "_" + texture;

        // Check if already cached
        if (!treeTypes.containsKey(key)) {
            // Create and cache new flyweight
            treeTypes.put(key, new TreeType(name, color, texture));
        }

        return treeTypes.get(key);  // Return cached instance
    }
}
\`\`\`

Key points:
- Key must include ALL intrinsic state
- Only create new instance if not in cache
- Always return from cache`,
	hint2: `**Forest.plantTree Implementation**

plantTree uses the factory to get shared flyweights:

\`\`\`java
public void plantTree(int x, int y, String name, String color, String texture) {
    // Get shared flyweight from factory
    TreeType type = TreeFactory.getTreeType(name, color, texture);

    // Create tree with unique position + shared type
    trees.add(new Tree(x, y, type));
}
\`\`\`

Memory savings visualization:
- 1,000,000 trees with 10 tree types
- Without Flyweight: 1,000,000 × (name + color + texture + x + y)
- With Flyweight: 10 × (name + color + texture) + 1,000,000 × (x + y + reference)`,
	whyItMatters: `## Problem & Solution

**Without Flyweight:**
\`\`\`java
// Each tree stores ALL data - massive memory waste!
class Tree {
    int x, y;	// 8 bytes
    String name;	// ~40 bytes
    String color;	// ~30 bytes
    String texture;	// ~50 bytes
    byte[] textureData;	// ~10KB
}
// 1 million trees = 10+ GB of memory!	// out of memory error
\`\`\`

**With Flyweight:**
\`\`\`java
class TreeType {	// shared between many trees
    String name, color, texture;	// ~120 bytes
    byte[] textureData;	// ~10KB
}

class Tree {	// lightweight - only unique data
    int x, y;	// 8 bytes
    TreeType type;	// 8 bytes (reference)
}
// 1 million trees + 10 types = ~16 MB!	// 99.8% memory saved
\`\`\`

---

## Real-World Examples

| Domain | Intrinsic (Shared) | Extrinsic (Unique) |
|--------|-------------------|-------------------|
| **String.intern()** | Character sequence | Reference |
| **Integer cache** | Values -128 to 127 | Variable binding |
| **Font rendering** | Glyph shapes | Position, size |
| **Game particles** | Sprite, color | Position, velocity |
| **Word processor** | Character style | Document position |
| **Browser DOM** | CSS styles | Element |

---

## Production Pattern: Text Editor Characters

\`\`\`java
// Flyweight - shared character formatting
class CharacterStyle {	// intrinsic state - shared by many characters
    private final String fontFamily;	// font name
    private final int fontSize;	// font size
    private final String color;	// text color
    private final boolean bold;	// bold flag
    private final boolean italic;	// italic flag

    public CharacterStyle(String fontFamily, int fontSize, String color,	// constructor
                          boolean bold, boolean italic) {
        this.fontFamily = fontFamily;	// store font
        this.fontSize = fontSize;	// store size
        this.color = color;	// store color
        this.bold = bold;	// store bold
        this.italic = italic;	// store italic
    }

    public void render(char c, int x, int y) {	// render with extrinsic data
        System.out.printf("Rendering '%c' at (%d,%d) with %s %dpt %s%s%s%n",	// output
            c, x, y, fontFamily, fontSize, color,	// basic info
            bold ? " bold" : "", italic ? " italic" : "");	// formatting
    }

    @Override
    public boolean equals(Object o) {	// for proper caching
        if (this == o) return true;	// same reference
        if (!(o instanceof CharacterStyle)) return false;	// type check
        CharacterStyle that = (CharacterStyle) o;	// cast
        return fontSize == that.fontSize && bold == that.bold &&	// compare primitives
               italic == that.italic && fontFamily.equals(that.fontFamily) &&	// compare font
               color.equals(that.color);	// compare color
    }

    @Override
    public int hashCode() {	// for HashMap
        return java.util.Objects.hash(fontFamily, fontSize, color, bold, italic);	// compute hash
    }
}

// Flyweight Factory
class StyleFactory {	// manages style flyweights
    private static final Map<CharacterStyle, CharacterStyle> styles = new HashMap<>();	// cache

    public static CharacterStyle getStyle(String fontFamily, int fontSize,	// get or create style
                                          String color, boolean bold, boolean italic) {
        CharacterStyle key = new CharacterStyle(fontFamily, fontSize, color, bold, italic);	// create key
        return styles.computeIfAbsent(key, k -> k);	// return cached or add new
    }

    public static int getStyleCount() { return styles.size(); }	// cache size
}

// Context - represents a character in document
class Character {	// extrinsic state + flyweight reference
    private final char character;	// the actual character
    private final int x;	// x position in document
    private final int y;	// y position in document
    private final CharacterStyle style;	// reference to shared style

    public Character(char character, int x, int y, CharacterStyle style) {	// constructor
        this.character = character;	// store char
        this.x = x;	// store position
        this.y = y;	// store position
        this.style = style;	// store style reference
    }

    public void render() {	// render character
        style.render(character, x, y);	// delegate to flyweight
    }
}

// Document using flyweights
class Document {	// text editor document
    private final List<Character> characters = new ArrayList<>();	// all characters

    public void addCharacter(char c, int x, int y,	// add character with styling
                            String fontFamily, int fontSize, String color,
                            boolean bold, boolean italic) {
        CharacterStyle style = StyleFactory.getStyle(	// get shared style
            fontFamily, fontSize, color, bold, italic);
        characters.add(new Character(c, x, y, style));	// add character with style
    }

    public void render() {	// render all characters
        for (Character c : characters) {	// iterate characters
            c.render();	// render each
        }
    }

    public int getCharacterCount() { return characters.size(); }	// total chars
}

// Usage
Document doc = new Document();	// create document

// Add many characters - styles are shared!
doc.addCharacter('H', 0, 0, "Arial", 12, "black", false, false);	// creates new style
doc.addCharacter('e', 10, 0, "Arial", 12, "black", false, false);	// reuses style!
doc.addCharacter('l', 20, 0, "Arial", 12, "black", false, false);	// reuses style!
doc.addCharacter('l', 30, 0, "Arial", 12, "black", false, false);	// reuses style!
doc.addCharacter('o', 40, 0, "Arial", 12, "black", false, false);	// reuses style!

doc.addCharacter('!', 50, 0, "Arial", 14, "red", true, false);	// creates new style

System.out.println("Characters: " + doc.getCharacterCount());	// 6
System.out.println("Styles: " + StyleFactory.getStyleCount());	// only 2!
\`\`\`

---

## Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| **Mutable intrinsic state** | Shared state changes affect all objects | Make flyweight immutable |
| **Wrong state separation** | Putting unique data in flyweight | Carefully analyze what's truly shared |
| **Missing factory** | Creating flyweights directly | Always use factory for caching |
| **Identity comparison** | Using == instead of equals | Implement proper equals/hashCode |
| **Over-optimization** | Using flyweight for few objects | Only use when memory is actually a problem |`,
	order: 5,
	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void treeTypeDrawsCorrectly() {
        TreeType type = new TreeType("Oak", "Green", "Rough");
        String result = type.draw(10, 20);
        assertTrue(result.contains("Oak"), "Draw should contain tree name");
        assertTrue(result.contains("10") && result.contains("20"), "Draw should contain coordinates");
    }
}

class Test2 {
    @Test
    void treeDrawsDelegates() {
        TreeType type = new TreeType("Pine", "DarkGreen", "Smooth");
        Tree tree = new Tree(50, 60, type);
        String result = tree.draw();
        assertTrue(result.contains("Pine"), "Tree draw should delegate to TreeType");
    }
}

class Test3 {
    @Test
    void treeFactoryCreatesNewType() {
        TreeType type = TreeFactory.getTreeType("Birch", "White", "Smooth");
        assertNotNull(type, "TreeFactory should create TreeType");
    }
}

class Test4 {
    @Test
    void treeFactoryReusesExistingType() {
        TreeType type1 = TreeFactory.getTreeType("Oak", "Green", "Rough");
        TreeType type2 = TreeFactory.getTreeType("Oak", "Green", "Rough");
        assertSame(type1, type2, "TreeFactory should reuse existing TreeType");
    }
}

class Test5 {
    @Test
    void treeFactoryCreatesDifferentTypesForDifferentParams() {
        TreeType type1 = TreeFactory.getTreeType("Oak", "Green", "Rough");
        TreeType type2 = TreeFactory.getTreeType("Pine", "DarkGreen", "Smooth");
        assertNotSame(type1, type2, "Different params should create different types");
    }
}

class Test6 {
    @Test
    void forestPlantTreeAddsTree() {
        Forest forest = new Forest();
        forest.plantTree(10, 20, "Oak", "Green", "Rough");
        assertEquals(1, forest.getTreeCount(), "Forest should have 1 tree");
    }
}

class Test7 {
    @Test
    void forestPlantMultipleTrees() {
        Forest forest = new Forest();
        forest.plantTree(10, 20, "Oak", "Green", "Rough");
        forest.plantTree(30, 40, "Oak", "Green", "Rough");
        forest.plantTree(50, 60, "Pine", "DarkGreen", "Smooth");
        assertEquals(3, forest.getTreeCount(), "Forest should have 3 trees");
    }
}

class Test8 {
    @Test
    void factoryTypesCountIsCorrect() {
        // Clear any previous state by using unique types
        String unique = String.valueOf(System.nanoTime());
        TreeFactory.getTreeType("Test1" + unique, "Color1", "Texture1");
        TreeFactory.getTreeType("Test2" + unique, "Color2", "Texture2");
        assertTrue(TreeFactory.getTypeCount() >= 2, "Factory should track type count");
    }
}

class Test9 {
    @Test
    void forestUsesFactory() {
        Forest forest = new Forest();
        int initialCount = TreeFactory.getTypeCount();
        forest.plantTree(100, 200, "Unique" + System.nanoTime(), "Red", "Bumpy");
        assertTrue(TreeFactory.getTypeCount() > initialCount, "Forest should use factory to create types");
    }
}

class Test10 {
    @Test
    void treeContainsCorrectPosition() {
        TreeType type = new TreeType("Maple", "Orange", "Rough");
        Tree tree = new Tree(123, 456, type);
        String result = tree.draw();
        assertTrue(result.contains("123") && result.contains("456"), "Tree should draw at correct position");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Flyweight (Приспособленец)',
			description: `Реализуйте **паттерн Flyweight** на Java — используйте разделение для эффективной поддержки большого количества мелкогранулированных объектов.

## Обзор

Паттерн Flyweight минимизирует использование памяти, разделяя общее состояние между множеством объектов. Он отделяет внутреннее (разделяемое) состояние от внешнего (уникального).

## Ключевые концепции

| Концепция | Описание | Пример |
|-----------|----------|--------|
| **Intrinsic State** | Разделяемые, неизменяемые данные | TreeType (name, color, texture) |
| **Extrinsic State** | Уникальные, контекстно-зависимые | Позиция дерева (x, y) |
| **Flyweight Factory** | Создаёт/кэширует разделяемые объекты | TreeFactory |

## Ваша задача

Реализуйте систему отрисовки леса:

1. **TreeType** (Flyweight) - Разделяемое внутреннее состояние (name, color, texture)
2. **Tree** (Context) - Содержит внешнее состояние (x, y) и ссылку на TreeType
3. **TreeFactory** - Создаёт и кэширует экземпляры TreeType
4. **Forest** - Управляет деревьями, использует фабрику

## Пример использования

\`\`\`java
Forest forest = new Forest();	// создаём менеджер леса

// Сажаем 1 миллион деревьев - но только несколько TreeTypes!
forest.plantTree(10, 20, "Oak", "Green", "Rough");	// создаёт новый TreeType
forest.plantTree(30, 40, "Oak", "Green", "Rough");	// переиспользует существующий TreeType
forest.plantTree(50, 60, "Pine", "DarkGreen", "Smooth");	// создаёт новый TreeType

System.out.println(forest.getTreeCount());	// 3 дерева
System.out.println(TreeFactory.getTypeCount());	// только 2 TreeTypes!
\`\`\`

## Ключевая идея

1 миллион деревьев могут разделять всего 10-20 объектов TreeType, экономя огромное количество памяти!`,
			hint1: `**Реализация TreeFactory**

Фабрика создаёт уникальный ключ и кэширует TreeTypes:

\`\`\`java
class TreeFactory {
    private static Map<String, TreeType> treeTypes = new HashMap<>();

    public static TreeType getTreeType(String name, String color, String texture) {
        // Создаём уникальный ключ из всего внутреннего состояния
        String key = name + "_" + color + "_" + texture;

        // Проверяем, есть ли уже в кэше
        if (!treeTypes.containsKey(key)) {
            // Создаём и кэшируем новый flyweight
            treeTypes.put(key, new TreeType(name, color, texture));
        }

        return treeTypes.get(key);  // Возвращаем кэшированный экземпляр
    }
}
\`\`\`

Ключевые моменты:
- Ключ должен включать ВСЁ внутреннее состояние
- Создавайте новый экземпляр только если его нет в кэше
- Всегда возвращайте из кэша`,
			hint2: `**Реализация Forest.plantTree**

plantTree использует фабрику для получения разделяемых flyweights:

\`\`\`java
public void plantTree(int x, int y, String name, String color, String texture) {
    // Получаем разделяемый flyweight из фабрики
    TreeType type = TreeFactory.getTreeType(name, color, texture);

    // Создаём дерево с уникальной позицией + разделяемым типом
    trees.add(new Tree(x, y, type));
}
\`\`\`

Визуализация экономии памяти:
- 1,000,000 деревьев с 10 типами деревьев
- Без Flyweight: 1,000,000 × (name + color + texture + x + y)
- С Flyweight: 10 × (name + color + texture) + 1,000,000 × (x + y + ссылка)`,
			whyItMatters: `## Проблема и решение

**Без Flyweight:**
\`\`\`java
// Каждое дерево хранит ВСЕ данные - огромная трата памяти!
class Tree {
    int x, y;	// 8 байт
    String name;	// ~40 байт
    String color;	// ~30 байт
    String texture;	// ~50 байт
    byte[] textureData;	// ~10КБ
}
// 1 миллион деревьев = 10+ ГБ памяти!	// ошибка нехватки памяти
\`\`\`

**С Flyweight:**
\`\`\`java
class TreeType {	// разделяется между многими деревьями
    String name, color, texture;	// ~120 байт
    byte[] textureData;	// ~10КБ
}

class Tree {	// легковесный - только уникальные данные
    int x, y;	// 8 байт
    TreeType type;	// 8 байт (ссылка)
}
// 1 миллион деревьев + 10 типов = ~16 МБ!	// экономия 99.8% памяти
\`\`\`

---

## Примеры из реального мира

| Домен | Intrinsic (Разделяемое) | Extrinsic (Уникальное) |
|-------|------------------------|----------------------|
| **String.intern()** | Последовательность символов | Ссылка |
| **Integer cache** | Значения -128 до 127 | Привязка переменной |
| **Рендеринг шрифтов** | Формы глифов | Позиция, размер |
| **Игровые частицы** | Спрайт, цвет | Позиция, скорость |
| **Текстовый процессор** | Стиль символа | Позиция в документе |
| **DOM браузера** | CSS стили | Элемент |

---

## Production паттерн: Символы текстового редактора

\`\`\`java
// Flyweight - разделяемое форматирование символов
class CharacterStyle {	// внутреннее состояние - разделяется многими символами
    private final String fontFamily;	// имя шрифта
    private final int fontSize;	// размер шрифта
    private final String color;	// цвет текста
    private final boolean bold;	// флаг жирный
    private final boolean italic;	// флаг курсив

    public CharacterStyle(String fontFamily, int fontSize, String color,	// конструктор
                          boolean bold, boolean italic) {
        this.fontFamily = fontFamily;	// сохраняем шрифт
        this.fontSize = fontSize;	// сохраняем размер
        this.color = color;	// сохраняем цвет
        this.bold = bold;	// сохраняем жирный
        this.italic = italic;	// сохраняем курсив
    }

    public void render(char c, int x, int y) {	// отрисовка с внешними данными
        System.out.printf("Rendering '%c' at (%d,%d) with %s %dpt %s%s%s%n",	// вывод
            c, x, y, fontFamily, fontSize, color,	// базовая информация
            bold ? " bold" : "", italic ? " italic" : "");	// форматирование
    }

    @Override
    public boolean equals(Object o) {	// для правильного кэширования
        if (this == o) return true;	// та же ссылка
        if (!(o instanceof CharacterStyle)) return false;	// проверка типа
        CharacterStyle that = (CharacterStyle) o;	// приведение типа
        return fontSize == that.fontSize && bold == that.bold &&	// сравнение примитивов
               italic == that.italic && fontFamily.equals(that.fontFamily) &&	// сравнение шрифта
               color.equals(that.color);	// сравнение цвета
    }

    @Override
    public int hashCode() {	// для HashMap
        return java.util.Objects.hash(fontFamily, fontSize, color, bold, italic);	// вычисление хэша
    }
}

// Flyweight Factory
class StyleFactory {	// управляет flyweights стилей
    private static final Map<CharacterStyle, CharacterStyle> styles = new HashMap<>();	// кэш

    public static CharacterStyle getStyle(String fontFamily, int fontSize,	// получить или создать стиль
                                          String color, boolean bold, boolean italic) {
        CharacterStyle key = new CharacterStyle(fontFamily, fontSize, color, bold, italic);	// создаём ключ
        return styles.computeIfAbsent(key, k -> k);	// возвращаем кэшированный или добавляем новый
    }

    public static int getStyleCount() { return styles.size(); }	// размер кэша
}

// Context - представляет символ в документе
class Character {	// внешнее состояние + ссылка на flyweight
    private final char character;	// сам символ
    private final int x;	// x позиция в документе
    private final int y;	// y позиция в документе
    private final CharacterStyle style;	// ссылка на разделяемый стиль

    public Character(char character, int x, int y, CharacterStyle style) {	// конструктор
        this.character = character;	// сохраняем символ
        this.x = x;	// сохраняем позицию
        this.y = y;	// сохраняем позицию
        this.style = style;	// сохраняем ссылку на стиль
    }

    public void render() {	// отрисовка символа
        style.render(character, x, y);	// делегируем flyweight
    }
}

// Документ, использующий flyweights
class Document {	// документ текстового редактора
    private final List<Character> characters = new ArrayList<>();	// все символы

    public void addCharacter(char c, int x, int y,	// добавить символ со стилем
                            String fontFamily, int fontSize, String color,
                            boolean bold, boolean italic) {
        CharacterStyle style = StyleFactory.getStyle(	// получаем разделяемый стиль
            fontFamily, fontSize, color, bold, italic);
        characters.add(new Character(c, x, y, style));	// добавляем символ со стилем
    }

    public void render() {	// отрисовка всех символов
        for (Character c : characters) {	// итерируем символы
            c.render();	// отрисовываем каждый
        }
    }

    public int getCharacterCount() { return characters.size(); }	// всего символов
}

// Использование
Document doc = new Document();	// создаём документ

// Добавляем много символов - стили разделяются!
doc.addCharacter('H', 0, 0, "Arial", 12, "black", false, false);	// создаёт новый стиль
doc.addCharacter('e', 10, 0, "Arial", 12, "black", false, false);	// переиспользует стиль!
doc.addCharacter('l', 20, 0, "Arial", 12, "black", false, false);	// переиспользует стиль!
doc.addCharacter('l', 30, 0, "Arial", 12, "black", false, false);	// переиспользует стиль!
doc.addCharacter('o', 40, 0, "Arial", 12, "black", false, false);	// переиспользует стиль!

doc.addCharacter('!', 50, 0, "Arial", 14, "red", true, false);	// создаёт новый стиль

System.out.println("Characters: " + doc.getCharacterCount());	// 6
System.out.println("Styles: " + StyleFactory.getStyleCount());	// только 2!
\`\`\`

---

## Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| **Изменяемое внутреннее состояние** | Изменения разделяемого состояния влияют на все объекты | Сделайте flyweight неизменяемым |
| **Неправильное разделение состояния** | Помещение уникальных данных в flyweight | Тщательно анализируйте, что действительно разделяется |
| **Отсутствие фабрики** | Создание flyweights напрямую | Всегда используйте фабрику для кэширования |
| **Сравнение идентичности** | Использование == вместо equals | Реализуйте правильные equals/hashCode |
| **Чрезмерная оптимизация** | Использование flyweight для малого числа объектов | Используйте только когда память действительно проблема |`
		},
		uz: {
			title: 'Flyweight Pattern',
			description: `Java da **Flyweight patternini** amalga oshiring — ko'p sonli mayda ob'ektlarni samarali qo'llab-quvvatlash uchun ulashishdan foydalaning.

## Umumiy ko'rinish

Flyweight patterni ko'p ob'ektlar o'rtasida umumiy holatni ulashish orqali xotira sarfini kamaytiradi. U ichki (ulashiladigan) holatni tashqi (noyob) holatdan ajratadi.

## Asosiy tushunchalar

| Tushuncha | Tavsif | Namuna |
|-----------|--------|--------|
| **Intrinsic State** | Ulashiladigan, o'zgarmas ma'lumotlar | TreeType (name, color, texture) |
| **Extrinsic State** | Noyob, kontekstga xos | Daraxt pozitsiyasi (x, y) |
| **Flyweight Factory** | Ulashiladigan ob'ektlarni yaratadi/keshlaydi | TreeFactory |

## Vazifangiz

O'rmon ko'rsatish tizimini amalga oshiring:

1. **TreeType** (Flyweight) - Ulashiladigan ichki holat (name, color, texture)
2. **Tree** (Context) - Tashqi holatni (x, y) va TreeType ga havolani o'z ichiga oladi
3. **TreeFactory** - TreeType namunalarini yaratadi va keshlaydi
4. **Forest** - Daraxtlarni boshqaradi, fabrikadan foydalanadi

## Foydalanish namunasi

\`\`\`java
Forest forest = new Forest();	// o'rmon menejeri yaratamiz

// 1 million daraxt ekamiz - lekin faqat bir nechta TreeTypes!
forest.plantTree(10, 20, "Oak", "Green", "Rough");	// yangi TreeType yaratadi
forest.plantTree(30, 40, "Oak", "Green", "Rough");	// mavjud TreeType ni qayta ishlatadi
forest.plantTree(50, 60, "Pine", "DarkGreen", "Smooth");	// yangi TreeType yaratadi

System.out.println(forest.getTreeCount());	// 3 ta daraxt
System.out.println(TreeFactory.getTypeCount());	// faqat 2 ta TreeTypes!
\`\`\`

## Asosiy tushuncha

1 million daraxt faqat 10-20 ta TreeType ob'ektini ulashishi mumkin, bu juda ko'p xotirani tejaydi!`,
			hint1: `**TreeFactory amalga oshirish**

Fabrika noyob kalit yaratadi va TreeTypes ni keshlaydi:

\`\`\`java
class TreeFactory {
    private static Map<String, TreeType> treeTypes = new HashMap<>();

    public static TreeType getTreeType(String name, String color, String texture) {
        // Barcha ichki holatdan noyob kalit yaratamiz
        String key = name + "_" + color + "_" + texture;

        // Keshda borligini tekshiramiz
        if (!treeTypes.containsKey(key)) {
            // Yangi flyweight yaratamiz va keshlaymiz
            treeTypes.put(key, new TreeType(name, color, texture));
        }

        return treeTypes.get(key);  // Keshlangan namunani qaytaramiz
    }
}
\`\`\`

Asosiy nuqtalar:
- Kalit BARCHA ichki holatni o'z ichiga olishi kerak
- Keshda bo'lmaganda faqat yangi namuna yarating
- Doimo keshdan qaytaring`,
			hint2: `**Forest.plantTree amalga oshirish**

plantTree ulashiladigan flyweights ni olish uchun fabrikadan foydalanadi:

\`\`\`java
public void plantTree(int x, int y, String name, String color, String texture) {
    // Fabrikadan ulashiladigan flyweight olamiz
    TreeType type = TreeFactory.getTreeType(name, color, texture);

    // Noyob pozitsiya + ulashiladigan tur bilan daraxt yaratamiz
    trees.add(new Tree(x, y, type));
}
\`\`\`

Xotira tejash vizualizatsiyasi:
- 1,000,000 daraxt 10 ta daraxt turi bilan
- Flyweight siz: 1,000,000 × (name + color + texture + x + y)
- Flyweight bilan: 10 × (name + color + texture) + 1,000,000 × (x + y + havola)`,
			whyItMatters: `## Muammo va yechim

**Flyweight siz:**
\`\`\`java
// Har bir daraxt BARCHA ma'lumotlarni saqlaydi - katta xotira isrofi!
class Tree {
    int x, y;	// 8 bayt
    String name;	// ~40 bayt
    String color;	// ~30 bayt
    String texture;	// ~50 bayt
    byte[] textureData;	// ~10KB
}
// 1 million daraxt = 10+ GB xotira!	// xotira yetishmovchiligi xatosi
\`\`\`

**Flyweight bilan:**
\`\`\`java
class TreeType {	// ko'p daraxtlar o'rtasida ulashiladi
    String name, color, texture;	// ~120 bayt
    byte[] textureData;	// ~10KB
}

class Tree {	// yengil - faqat noyob ma'lumotlar
    int x, y;	// 8 bayt
    TreeType type;	// 8 bayt (havola)
}
// 1 million daraxt + 10 tur = ~16 MB!	// 99.8% xotira tejalgan
\`\`\`

---

## Haqiqiy dunyo namunalari

| Domen | Intrinsic (Ulashiladigan) | Extrinsic (Noyob) |
|-------|--------------------------|-------------------|
| **String.intern()** | Belgilar ketma-ketligi | Havola |
| **Integer cache** | -128 dan 127 gacha qiymatlar | O'zgaruvchi bog'lanishi |
| **Shrift ko'rsatish** | Glif shakllari | Pozitsiya, o'lcham |
| **O'yin zarralari** | Sprite, rang | Pozitsiya, tezlik |
| **Matn protsessori** | Belgi uslubi | Hujjatdagi pozitsiya |
| **Brauzer DOM** | CSS uslublari | Element |

---

## Production pattern: Matn muharriri belgilari

\`\`\`java
// Flyweight - ulashiladigan belgi formatlash
class CharacterStyle {	// ichki holat - ko'p belgilar tomonidan ulashiladi
    private final String fontFamily;	// shrift nomi
    private final int fontSize;	// shrift o'lchami
    private final String color;	// matn rangi
    private final boolean bold;	// qalin bayrog'i
    private final boolean italic;	// qiyshiq bayrog'i

    public CharacterStyle(String fontFamily, int fontSize, String color,	// konstruktor
                          boolean bold, boolean italic) {
        this.fontFamily = fontFamily;	// shriftni saqlash
        this.fontSize = fontSize;	// o'lchamni saqlash
        this.color = color;	// rangni saqlash
        this.bold = bold;	// qalinni saqlash
        this.italic = italic;	// qiyshiqni saqlash
    }

    public void render(char c, int x, int y) {	// tashqi ma'lumotlar bilan ko'rsatish
        System.out.printf("Rendering '%c' at (%d,%d) with %s %dpt %s%s%s%n",	// chiqish
            c, x, y, fontFamily, fontSize, color,	// asosiy ma'lumot
            bold ? " bold" : "", italic ? " italic" : "");	// formatlash
    }

    @Override
    public boolean equals(Object o) {	// to'g'ri keshlash uchun
        if (this == o) return true;	// xuddi shu havola
        if (!(o instanceof CharacterStyle)) return false;	// tip tekshirish
        CharacterStyle that = (CharacterStyle) o;	// tip o'zgartirish
        return fontSize == that.fontSize && bold == that.bold &&	// primitivlarni solishtirish
               italic == that.italic && fontFamily.equals(that.fontFamily) &&	// shriftni solishtirish
               color.equals(that.color);	// rangni solishtirish
    }

    @Override
    public int hashCode() {	// HashMap uchun
        return java.util.Objects.hash(fontFamily, fontSize, color, bold, italic);	// xesh hisoblash
    }
}

// Flyweight Factory
class StyleFactory {	// uslub flyweights ni boshqaradi
    private static final Map<CharacterStyle, CharacterStyle> styles = new HashMap<>();	// kesh

    public static CharacterStyle getStyle(String fontFamily, int fontSize,	// uslub olish yoki yaratish
                                          String color, boolean bold, boolean italic) {
        CharacterStyle key = new CharacterStyle(fontFamily, fontSize, color, bold, italic);	// kalit yaratish
        return styles.computeIfAbsent(key, k -> k);	// keshlanganni qaytarish yoki yangisini qo'shish
    }

    public static int getStyleCount() { return styles.size(); }	// kesh o'lchami
}

// Context - hujjatdagi belgini ifodalaydi
class Character {	// tashqi holat + flyweight havolasi
    private final char character;	// haqiqiy belgi
    private final int x;	// hujjatdagi x pozitsiya
    private final int y;	// hujjatdagi y pozitsiya
    private final CharacterStyle style;	// ulashiladigan uslubga havola

    public Character(char character, int x, int y, CharacterStyle style) {	// konstruktor
        this.character = character;	// belgini saqlash
        this.x = x;	// pozitsiyani saqlash
        this.y = y;	// pozitsiyani saqlash
        this.style = style;	// uslub havolasini saqlash
    }

    public void render() {	// belgini ko'rsatish
        style.render(character, x, y);	// flyweight ga delegatsiya
    }
}

// Flyweights dan foydalanadigan hujjat
class Document {	// matn muharriri hujjati
    private final List<Character> characters = new ArrayList<>();	// barcha belgilar

    public void addCharacter(char c, int x, int y,	// uslub bilan belgi qo'shish
                            String fontFamily, int fontSize, String color,
                            boolean bold, boolean italic) {
        CharacterStyle style = StyleFactory.getStyle(	// ulashiladigan uslubni olish
            fontFamily, fontSize, color, bold, italic);
        characters.add(new Character(c, x, y, style));	// uslub bilan belgi qo'shish
    }

    public void render() {	// barcha belgilarni ko'rsatish
        for (Character c : characters) {	// belgilarni takrorlash
            c.render();	// har birini ko'rsatish
        }
    }

    public int getCharacterCount() { return characters.size(); }	// jami belgilar
}

// Foydalanish
Document doc = new Document();	// hujjat yaratish

// Ko'p belgilar qo'shamiz - uslublar ulashiladi!
doc.addCharacter('H', 0, 0, "Arial", 12, "black", false, false);	// yangi uslub yaratadi
doc.addCharacter('e', 10, 0, "Arial", 12, "black", false, false);	// uslubni qayta ishlatadi!
doc.addCharacter('l', 20, 0, "Arial", 12, "black", false, false);	// uslubni qayta ishlatadi!
doc.addCharacter('l', 30, 0, "Arial", 12, "black", false, false);	// uslubni qayta ishlatadi!
doc.addCharacter('o', 40, 0, "Arial", 12, "black", false, false);	// uslubni qayta ishlatadi!

doc.addCharacter('!', 50, 0, "Arial", 14, "red", true, false);	// yangi uslub yaratadi

System.out.println("Characters: " + doc.getCharacterCount());	// 6
System.out.println("Styles: " + StyleFactory.getStyleCount());	// faqat 2!
\`\`\`

---

## Keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| **O'zgaruvchan ichki holat** | Ulashiladigan holat o'zgarishi barcha ob'ektlarga ta'sir qiladi | Flyweight ni o'zgarmas qiling |
| **Noto'g'ri holat ajratish** | Noyob ma'lumotlarni flyweight ga qo'yish | Nima haqiqatan ulashilishini sinchiklab tahlil qiling |
| **Fabrika yo'q** | Flyweights ni to'g'ridan-to'g'ri yaratish | Keshlash uchun doimo fabrikadan foydalaning |
| **Identifikator solishtirish** | equals o'rniga == ishlatish | To'g'ri equals/hashCode ni amalga oshiring |
| **Haddan tashqari optimallashtirish** | Kam ob'ektlar uchun flyweight ishlatish | Faqat xotira haqiqatan muammo bo'lganda foydalaning |`
		}
	}
};

export default task;
