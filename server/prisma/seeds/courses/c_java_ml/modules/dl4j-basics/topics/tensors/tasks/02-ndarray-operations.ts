import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-ndarray-operations',
	title: 'NDArray Operations',
	difficulty: 'medium',
	tags: ['nd4j', 'operations', 'java'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# NDArray Operations

Implement common mathematical and linear algebra operations with ND4J.

## Task

Implement methods for:
- Matrix multiplication
- Broadcasting operations
- Aggregation functions
- Element-wise operations

## Example

\`\`\`java
INDArray a = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray b = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray product = a.mmul(b);  // Matrix multiplication
double sum = a.sumNumber().doubleValue();  // Sum all elements
\`\`\``,

	initialCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NDArrayOperations {

    /**
     */
    public static INDArray matmul(INDArray a, INDArray b) {
        return null;
    }

    /**
     */
    public static INDArray elementwiseMul(INDArray a, INDArray b) {
        return null;
    }

    /**
     */
    public static INDArray softmax(INDArray array) {
        return null;
    }

    /**
     */
    public static INDArray meanAlongAxis(INDArray array, int axis) {
        return null;
    }

    /**
     */
    public static INDArray normalize(INDArray array) {
        return null;
    }
}`,

	solutionCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class NDArrayOperations {

    /**
     * Matrix multiplication of two 2D arrays.
     */
    public static INDArray matmul(INDArray a, INDArray b) {
        return a.mmul(b);
    }

    /**
     * Element-wise multiplication (Hadamard product).
     */
    public static INDArray elementwiseMul(INDArray a, INDArray b) {
        return a.mul(b);
    }

    /**
     * Apply softmax to array.
     */
    public static INDArray softmax(INDArray array) {
        return Transforms.softmax(array);
    }

    /**
     * Calculate mean along specified axis.
     */
    public static INDArray meanAlongAxis(INDArray array, int axis) {
        return array.mean(axis);
    }

    /**
     * Normalize array to have zero mean and unit variance.
     */
    public static INDArray normalize(INDArray array) {
        double mean = array.meanNumber().doubleValue();
        double std = array.stdNumber().doubleValue();

        if (std == 0) {
            return array.sub(mean);
        }

        return array.sub(mean).div(std);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class NDArrayOperationsTest {

    @Test
    void testMatmul() {
        INDArray a = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray b = Nd4j.create(new double[][]{{5, 6}, {7, 8}});
        INDArray result = NDArrayOperations.matmul(a, b);
        assertEquals(19.0, result.getDouble(0, 0), 0.001);
        assertEquals(22.0, result.getDouble(0, 1), 0.001);
    }

    @Test
    void testElementwiseMul() {
        INDArray a = Nd4j.create(new double[]{2, 3, 4});
        INDArray b = Nd4j.create(new double[]{1, 2, 3});
        INDArray result = NDArrayOperations.elementwiseMul(a, b);
        assertEquals(2.0, result.getDouble(0), 0.001);
        assertEquals(6.0, result.getDouble(1), 0.001);
    }

    @Test
    void testSoftmax() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3});
        INDArray result = NDArrayOperations.softmax(array);
        assertEquals(1.0, result.sumNumber().doubleValue(), 0.001);
    }

    @Test
    void testMeanAlongAxis() {
        INDArray array = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}});
        INDArray result = NDArrayOperations.meanAlongAxis(array, 0);
        assertEquals(2.5, result.getDouble(0), 0.001);
    }

    @Test
    void testNormalize() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray result = NDArrayOperations.normalize(array);
        assertEquals(0.0, result.meanNumber().doubleValue(), 0.001);
    }

    @Test
    void testMatmulShape() {
        INDArray a = Nd4j.create(2, 3);
        INDArray b = Nd4j.create(3, 4);
        INDArray result = NDArrayOperations.matmul(a, b);
        assertArrayEquals(new long[]{2, 4}, result.shape());
    }

    @Test
    void testElementwiseMulShape() {
        INDArray a = Nd4j.ones(3, 3);
        INDArray b = Nd4j.ones(3, 3);
        INDArray result = NDArrayOperations.elementwiseMul(a, b);
        assertArrayEquals(new long[]{3, 3}, result.shape());
    }

    @Test
    void testSoftmaxValues() {
        INDArray array = Nd4j.create(new double[]{0, 0});
        INDArray result = NDArrayOperations.softmax(array);
        assertEquals(0.5, result.getDouble(0), 0.001);
    }

    @Test
    void testMeanAlongAxisShape() {
        INDArray array = Nd4j.create(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        INDArray result = NDArrayOperations.meanAlongAxis(array, 0);
        assertEquals(2, result.length());
    }

    @Test
    void testNormalizeStd() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray result = NDArrayOperations.normalize(array);
        assertEquals(1.0, result.stdNumber().doubleValue(), 0.1);
    }
}`,

	hint1: 'Use mmul() for matrix multiplication, mul() for element-wise',
	hint2: 'Transforms class has many useful functions like softmax, sigmoid',

	whyItMatters: `Array operations are core to ML:

- **Linear algebra**: Matrix operations power neural networks
- **Normalization**: Essential for stable training
- **Softmax**: Used in classification outputs
- **Efficient**: Vectorized operations are much faster

These operations are used in every ML model.`,

	translations: {
		ru: {
			title: 'Операции NDArray',
			description: `# Операции NDArray

Реализуйте распространенные математические операции и операции линейной алгебры с ND4J.

## Задача

Реализуйте методы для:
- Матричного умножения
- Операций с broadcasting
- Функций агрегации
- Поэлементных операций

## Пример

\`\`\`java
INDArray a = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray b = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray product = a.mmul(b);  // Matrix multiplication
double sum = a.sumNumber().doubleValue();  // Sum all elements
\`\`\``,
			hint1: 'Используйте mmul() для матричного умножения, mul() для поэлементного',
			hint2: 'Класс Transforms имеет много полезных функций: softmax, sigmoid',
			whyItMatters: `Операции с массивами - ядро ML:

- **Линейная алгебра**: Матричные операции питают нейросети
- **Нормализация**: Необходима для стабильного обучения
- **Softmax**: Используется в выходах классификации
- **Эффективность**: Векторизованные операции намного быстрее`,
		},
		uz: {
			title: 'NDArray operatsiyalari',
			description: `# NDArray operatsiyalari

ND4J bilan keng tarqalgan matematik va chiziqli algebra operatsiyalarini amalga oshiring.

## Topshiriq

Metodlarni amalga oshiring:
- Matritsa ko'paytirish
- Broadcasting operatsiyalar
- Agregatsiya funksiyalari
- Element bo'yicha operatsiyalar

## Misol

\`\`\`java
INDArray a = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray b = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray product = a.mmul(b);  // Matrix multiplication
double sum = a.sumNumber().doubleValue();  // Sum all elements
\`\`\``,
			hint1: "Matritsa ko'paytirish uchun mmul(), element bo'yicha uchun mul() dan foydalaning",
			hint2: "Transforms sinfida softmax, sigmoid kabi ko'p foydali funksiyalar bor",
			whyItMatters: `Massiv operatsiyalari ML ning yadrosi:

- **Chiziqli algebra**: Matritsa operatsiyalari neyron tarmoqlarni quvvatlaydi
- **Normalizatsiya**: Barqaror o'qitish uchun zarur
- **Softmax**: Klassifikatsiya chiqishlarida ishlatiladi
- **Samarali**: Vektorlashtirilgan operatsiyalar ancha tezroq`,
		},
	},
};

export default task;
