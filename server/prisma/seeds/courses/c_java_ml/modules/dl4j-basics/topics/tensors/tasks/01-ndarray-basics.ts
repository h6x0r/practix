import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-ndarray-basics',
	title: 'NDArray Basics',
	difficulty: 'easy',
	tags: ['nd4j', 'tensor', 'java'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# NDArray Basics

Learn to create and manipulate NDArrays in ND4J, the numerical computing library for Java.

## Task

Implement utility methods for working with NDArrays:
- Create arrays from various sources
- Basic array operations
- Shape manipulation

## Example

\`\`\`java
INDArray array = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
// [[1, 2], [3, 4]]

INDArray zeros = Nd4j.zeros(3, 3);
INDArray ones = Nd4j.ones(2, 4);
\`\`\``,

	initialCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NDArrayUtils {

    /**
     */
    public static INDArray create2D(double[][] data) {
        return null;
    }

    /**
     */
    public static INDArray createFilled(int[] shape, double value) {
        return null;
    }

    /**
     */
    public static INDArray reshape(INDArray array, int... newShape) {
        return null;
    }

    /**
     */
    public static INDArray getRow(INDArray array, int rowIndex) {
        return null;
    }

    /**
     */
    public static INDArray add(INDArray a, INDArray b) {
        return null;
    }
}`,

	solutionCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NDArrayUtils {

    /**
     * Create a 2D array from nested array.
     */
    public static INDArray create2D(double[][] data) {
        return Nd4j.create(data);
    }

    /**
     * Create an array filled with a specific value.
     */
    public static INDArray createFilled(int[] shape, double value) {
        INDArray array = Nd4j.zeros(shape);
        return array.addi(value);
    }

    /**
     * Reshape an array to new dimensions.
     */
    public static INDArray reshape(INDArray array, int... newShape) {
        return array.reshape(newShape);
    }

    /**
     * Get a slice of the array.
     */
    public static INDArray getRow(INDArray array, int rowIndex) {
        return array.getRow(rowIndex);
    }

    /**
     * Calculate element-wise sum of two arrays.
     */
    public static INDArray add(INDArray a, INDArray b) {
        return a.add(b);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class NDArrayUtilsTest {

    @Test
    void testCreate2D() {
        double[][] data = {{1, 2}, {3, 4}};
        INDArray result = NDArrayUtils.create2D(data);
        assertEquals(2, result.rows());
        assertEquals(2, result.columns());
        assertEquals(1.0, result.getDouble(0, 0), 0.001);
    }

    @Test
    void testCreateFilled() {
        INDArray result = NDArrayUtils.createFilled(new int[]{2, 3}, 5.0);
        assertEquals(2, result.rows());
        assertEquals(3, result.columns());
        assertEquals(5.0, result.getDouble(0, 0), 0.001);
    }

    @Test
    void testReshape() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6});
        INDArray result = NDArrayUtils.reshape(array, 2, 3);
        assertArrayEquals(new long[]{2, 3}, result.shape());
    }

    @Test
    void testGetRow() {
        INDArray array = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray row = NDArrayUtils.getRow(array, 1);
        assertEquals(3.0, row.getDouble(0), 0.001);
    }

    @Test
    void testAdd() {
        INDArray a = Nd4j.ones(2, 2);
        INDArray b = Nd4j.ones(2, 2);
        INDArray result = NDArrayUtils.add(a, b);
        assertEquals(2.0, result.getDouble(0, 0), 0.001);
    }

    @Test
    void testCreate2DShape() {
        double[][] data = {{1, 2, 3}, {4, 5, 6}};
        INDArray result = NDArrayUtils.create2D(data);
        assertArrayEquals(new long[]{2, 3}, result.shape());
    }

    @Test
    void testCreateFilledAllValues() {
        INDArray result = NDArrayUtils.createFilled(new int[]{2, 2}, 7.0);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(7.0, result.getDouble(i, j), 0.001);
            }
        }
    }

    @Test
    void testReshapePreservesElements() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray result = NDArrayUtils.reshape(array, 2, 2);
        assertEquals(4, result.length());
    }

    @Test
    void testGetRowLength() {
        INDArray array = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}});
        INDArray row = NDArrayUtils.getRow(array, 0);
        assertEquals(3, row.length());
    }

    @Test
    void testAddDoesNotModifyOriginal() {
        INDArray a = Nd4j.ones(2, 2);
        INDArray b = Nd4j.ones(2, 2);
        NDArrayUtils.add(a, b);
        assertEquals(1.0, a.getDouble(0, 0), 0.001);
    }
}`,

	hint1: 'Use Nd4j.create() factory methods for array creation',
	hint2: 'addi() modifies in place, add() returns new array',

	whyItMatters: `ND4J is the foundation of DL4J:

- **NumPy equivalent**: Familiar tensor operations for Java
- **GPU support**: Seamless CUDA acceleration
- **Performance**: Native operations via JavaCPP
- **Integration**: Works with DL4J neural networks

Essential for any Java ML project.`,

	translations: {
		ru: {
			title: 'Основы NDArray',
			description: `# Основы NDArray

Научитесь создавать и манипулировать NDArray в ND4J, библиотеке для численных вычислений на Java.

## Задача

Реализуйте утилитные методы для работы с NDArray:
- Создание массивов из различных источников
- Базовые операции с массивами
- Манипуляции с формой

## Пример

\`\`\`java
INDArray array = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
// [[1, 2], [3, 4]]

INDArray zeros = Nd4j.zeros(3, 3);
INDArray ones = Nd4j.ones(2, 4);
\`\`\``,
			hint1: 'Используйте фабричные методы Nd4j.create() для создания массивов',
			hint2: 'addi() модифицирует на месте, add() возвращает новый массив',
			whyItMatters: `ND4J - основа DL4J:

- **Эквивалент NumPy**: Знакомые тензорные операции для Java
- **Поддержка GPU**: Бесшовное ускорение CUDA
- **Производительность**: Нативные операции через JavaCPP
- **Интеграция**: Работает с нейросетями DL4J`,
		},
		uz: {
			title: 'NDArray asoslari',
			description: `# NDArray asoslari

Java uchun raqamli hisoblash kutubxonasi ND4J da NDArray yaratish va boshqarishni o'rganing.

## Topshiriq

NDArray bilan ishlash uchun yordamchi metodlarni amalga oshiring:
- Turli manbalardan massivlar yaratish
- Asosiy massiv operatsiyalari
- Shakl manipulyatsiyalari

## Misol

\`\`\`java
INDArray array = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
// [[1, 2], [3, 4]]

INDArray zeros = Nd4j.zeros(3, 3);
INDArray ones = Nd4j.ones(2, 4);
\`\`\``,
			hint1: "Massivlar yaratish uchun Nd4j.create() fabrika metodlaridan foydalaning",
			hint2: "addi() joyida o'zgartiradi, add() yangi massiv qaytaradi",
			whyItMatters: `ND4J DL4J ning asosi:

- **NumPy ekvivalenti**: Java uchun tanish tensor operatsiyalari
- **GPU qo'llab-quvvatlash**: Uzluksiz CUDA tezlashtirish
- **Samaradorlik**: JavaCPP orqali mahalliy operatsiyalar
- **Integratsiya**: DL4J neyron tarmoqlari bilan ishlaydi`,
		},
	},
};

export default task;
