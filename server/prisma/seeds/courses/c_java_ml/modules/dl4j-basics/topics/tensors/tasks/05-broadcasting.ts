import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-broadcasting',
	title: 'Broadcasting Operations',
	difficulty: 'medium',
	tags: ['nd4j', 'broadcasting', 'operations'],
	estimatedTime: '15m',
	isPremium: true,
	order: 5,
	description: `# Broadcasting Operations

Understand and apply broadcasting for efficient array operations.

## Task

Implement broadcasting:
- Add scalar to array
- Add vector to matrix rows/columns
- Element-wise operations with broadcasting

## Example

\`\`\`java
INDArray matrix = Nd4j.ones(3, 4);
INDArray rowVector = Nd4j.create(new double[]{1, 2, 3, 4});

// Add vector to each row
INDArray result = matrix.addRowVector(rowVector);
\`\`\``,

	initialCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BroadcastOps {

    /**
     */
    public static INDArray addScalar(INDArray array, double scalar) {
        return null;
    }

    /**
     */
    public static INDArray addRowVector(INDArray matrix, INDArray rowVector) {
        return null;
    }

    /**
     */
    public static INDArray addColumnVector(INDArray matrix, INDArray colVector) {
        return null;
    }

    /**
     */
    public static INDArray mulRowVector(INDArray matrix, INDArray rowVector) {
        return null;
    }
}`,

	solutionCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BroadcastOps {

    /**
     * Add scalar to all elements.
     */
    public static INDArray addScalar(INDArray array, double scalar) {
        return array.add(scalar);
    }

    /**
     * Add row vector to each row of matrix.
     */
    public static INDArray addRowVector(INDArray matrix, INDArray rowVector) {
        return matrix.addRowVector(rowVector);
    }

    /**
     * Add column vector to each column.
     */
    public static INDArray addColumnVector(INDArray matrix, INDArray colVector) {
        return matrix.addColumnVector(colVector);
    }

    /**
     * Multiply each row by vector.
     */
    public static INDArray mulRowVector(INDArray matrix, INDArray rowVector) {
        return matrix.mulRowVector(rowVector);
    }

    /**
     * Divide each row by vector.
     */
    public static INDArray divRowVector(INDArray matrix, INDArray rowVector) {
        return matrix.divRowVector(rowVector);
    }

    /**
     * Subtract column vector from each column.
     */
    public static INDArray subColumnVector(INDArray matrix, INDArray colVector) {
        return matrix.subColumnVector(colVector);
    }

    /**
     * Broadcast and add two arrays.
     */
    public static INDArray broadcastAdd(INDArray a, INDArray b) {
        // ND4J handles broadcasting automatically in many cases
        return a.add(b);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class BroadcastOpsTest {

    @Test
    void testAddScalar() {
        INDArray a = Nd4j.ones(2, 3);
        INDArray result = BroadcastOps.addScalar(a, 5);

        assertEquals(6.0, result.getDouble(0, 0), 0.001);
    }

    @Test
    void testAddRowVector() {
        INDArray matrix = Nd4j.zeros(3, 4);
        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4});

        INDArray result = BroadcastOps.addRowVector(matrix, row);

        assertEquals(1.0, result.getDouble(0, 0), 0.001);
        assertEquals(2.0, result.getDouble(0, 1), 0.001);
        assertEquals(1.0, result.getDouble(2, 0), 0.001);
    }

    @Test
    void testMulRowVector() {
        INDArray matrix = Nd4j.ones(2, 3).mul(2);
        INDArray row = Nd4j.create(new double[]{1, 2, 3});

        INDArray result = BroadcastOps.mulRowVector(matrix, row);

        assertEquals(2.0, result.getDouble(0, 0), 0.001);
        assertEquals(4.0, result.getDouble(0, 1), 0.001);
        assertEquals(6.0, result.getDouble(0, 2), 0.001);
    }

    @Test
    void testAddColumnVector() {
        INDArray matrix = Nd4j.zeros(3, 2);
        INDArray col = Nd4j.create(new double[][]{{1}, {2}, {3}});
        INDArray result = BroadcastOps.addColumnVector(matrix, col);
        assertEquals(1.0, result.getDouble(0, 0), 0.001);
        assertEquals(2.0, result.getDouble(1, 0), 0.001);
    }

    @Test
    void testAddScalarShape() {
        INDArray a = Nd4j.ones(3, 4);
        INDArray result = BroadcastOps.addScalar(a, 10);
        assertArrayEquals(new long[]{3, 4}, result.shape());
    }

    @Test
    void testAddRowVectorShape() {
        INDArray matrix = Nd4j.zeros(5, 3);
        INDArray row = Nd4j.create(new double[]{1, 2, 3});
        INDArray result = BroadcastOps.addRowVector(matrix, row);
        assertEquals(5, result.rows());
        assertEquals(3, result.columns());
    }

    @Test
    void testDivRowVector() {
        INDArray matrix = Nd4j.create(new double[][]{{2, 4, 6}, {8, 10, 12}});
        INDArray row = Nd4j.create(new double[]{2, 2, 2});
        INDArray result = BroadcastOps.divRowVector(matrix, row);
        assertEquals(1.0, result.getDouble(0, 0), 0.001);
        assertEquals(3.0, result.getDouble(0, 2), 0.001);
    }

    @Test
    void testSubColumnVector() {
        INDArray matrix = Nd4j.ones(2, 3).mul(5);
        INDArray col = Nd4j.create(new double[][]{{1}, {2}});
        INDArray result = BroadcastOps.subColumnVector(matrix, col);
        assertEquals(4.0, result.getDouble(0, 0), 0.001);
        assertEquals(3.0, result.getDouble(1, 0), 0.001);
    }

    @Test
    void testAddScalarAllElements() {
        INDArray a = Nd4j.zeros(2, 2);
        INDArray result = BroadcastOps.addScalar(a, 3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(3.0, result.getDouble(i, j), 0.001);
            }
        }
    }

    @Test
    void testMulRowVectorSecondRow() {
        INDArray matrix = Nd4j.ones(3, 2);
        INDArray row = Nd4j.create(new double[]{3, 4});
        INDArray result = BroadcastOps.mulRowVector(matrix, row);
        assertEquals(3.0, result.getDouble(1, 0), 0.001);
        assertEquals(4.0, result.getDouble(1, 1), 0.001);
    }
}`,

	hint1: 'addRowVector() adds vector to each row of matrix',
	hint2: 'addColumnVector() adds vector to each column',

	whyItMatters: `Broadcasting makes array programming efficient:

- **No loops**: Operate on entire arrays at once
- **Memory efficient**: Avoids creating intermediate arrays
- **NumPy-like**: Same concept as Python broadcasting
- **GPU friendly**: Enables parallel execution

Broadcasting is fundamental for vectorized ML code.`,

	translations: {
		ru: {
			title: 'Операции broadcasting',
			description: `# Операции broadcasting

Поймите и применяйте broadcasting для эффективных операций с массивами.

## Задача

Реализуйте broadcasting:
- Добавление скаляра к массиву
- Добавление вектора к строкам/столбцам матрицы
- Поэлементные операции с broadcasting

## Пример

\`\`\`java
INDArray matrix = Nd4j.ones(3, 4);
INDArray rowVector = Nd4j.create(new double[]{1, 2, 3, 4});

// Add vector to each row
INDArray result = matrix.addRowVector(rowVector);
\`\`\``,
			hint1: 'addRowVector() добавляет вектор к каждой строке матрицы',
			hint2: 'addColumnVector() добавляет вектор к каждому столбцу',
			whyItMatters: `Broadcasting делает программирование массивов эффективным:

- **Без циклов**: Операции над целыми массивами сразу
- **Эффективность памяти**: Избегает создания промежуточных массивов
- **Как в NumPy**: Та же концепция что в Python
- **Дружелюбно к GPU**: Позволяет параллельное выполнение`,
		},
		uz: {
			title: 'Broadcasting operatsiyalari',
			description: `# Broadcasting operatsiyalari

Samarali massiv operatsiyalari uchun broadcastingni tushuning va qo'llang.

## Topshiriq

Broadcastingni amalga oshiring:
- Massivga skalar qo'shish
- Matritsa qatorlari/ustunlariga vektor qo'shish
- Broadcasting bilan element bo'yicha operatsiyalar

## Misol

\`\`\`java
INDArray matrix = Nd4j.ones(3, 4);
INDArray rowVector = Nd4j.create(new double[]{1, 2, 3, 4});

// Add vector to each row
INDArray result = matrix.addRowVector(rowVector);
\`\`\``,
			hint1: "addRowVector() vektorni matritsa har bir qatoriga qo'shadi",
			hint2: "addColumnVector() vektorni har bir ustunga qo'shadi",
			whyItMatters: `Broadcasting massiv dasturlashni samarali qiladi:

- **Tsiklsiz**: Butun massivlarda bir vaqtda operatsiyalar
- **Xotira samaradorligi**: Oraliq massivlarni yaratishdan qochadi
- **NumPy-ga o'xshash**: Python broadcastingi bilan bir xil kontseptsiya
- **GPU uchun qulay**: Parallel bajarishni yoqadi`,
		},
	},
};

export default task;
