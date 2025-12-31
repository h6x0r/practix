import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-matrix-operations',
	title: 'Matrix Operations',
	difficulty: 'medium',
	tags: ['nd4j', 'matrix', 'linear-algebra'],
	estimatedTime: '20m',
	isPremium: false,
	order: 4,
	description: `# Matrix Operations

Perform essential linear algebra operations for ML.

## Task

Implement matrix operations:
- Matrix multiplication
- Transpose and inverse
- Eigenvalue decomposition

## Example

\`\`\`java
INDArray A = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray B = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray C = A.mmul(B);  // Matrix multiplication
INDArray T = A.transpose();  // Transpose
\`\`\``,

	initialCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;

public class MatrixOps {

    /**
     */
    public static INDArray multiply(INDArray A, INDArray B) {
        return null;
    }

    /**
     */
    public static INDArray transpose(INDArray A) {
        return null;
    }

    /**
     */
    public static INDArray inverse(INDArray A) {
        return null;
    }

    /**
     */
    public static double dotProduct(INDArray a, INDArray b) {
        return 0.0;
    }
}`,

	solutionCode: `import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MatrixOps {

    /**
     * Multiply two matrices.
     */
    public static INDArray multiply(INDArray A, INDArray B) {
        return A.mmul(B);
    }

    /**
     * Compute transpose.
     */
    public static INDArray transpose(INDArray A) {
        return A.transpose();
    }

    /**
     * Compute inverse (for square matrices).
     */
    public static INDArray inverse(INDArray A) {
        return InvertMatrix.invert(A, false);
    }

    /**
     * Compute dot product of two vectors.
     */
    public static double dotProduct(INDArray a, INDArray b) {
        return Nd4j.getBlasWrapper().dot(a, b);
    }

    /**
     * Compute matrix determinant.
     */
    public static double determinant(INDArray A) {
        return Nd4j.getBlasWrapper().lapack().dgetrf(A).getFirst();
    }

    /**
     * Compute L2 norm of vector.
     */
    public static double norm(INDArray v) {
        return v.norm2Number().doubleValue();
    }

    /**
     * Solve linear system Ax = b.
     */
    public static INDArray solve(INDArray A, INDArray b) {
        INDArray Ainv = inverse(A);
        return Ainv.mmul(b);
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixOpsTest {

    @Test
    void testMultiply() {
        INDArray A = Nd4j.eye(2);
        INDArray B = Nd4j.create(new double[][]{{1, 2}, {3, 4}});

        INDArray C = MatrixOps.multiply(A, B);

        assertEquals(B.getDouble(0, 0), C.getDouble(0, 0), 0.001);
    }

    @Test
    void testTranspose() {
        INDArray A = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}});
        INDArray T = MatrixOps.transpose(A);

        assertEquals(3, T.rows());
        assertEquals(2, T.columns());
    }

    @Test
    void testDotProduct() {
        INDArray a = Nd4j.create(new double[]{1, 2, 3});
        INDArray b = Nd4j.create(new double[]{4, 5, 6});

        double dot = MatrixOps.dotProduct(a, b);

        assertEquals(32.0, dot, 0.001);  // 1*4 + 2*5 + 3*6 = 32
    }

    @Test
    void testMultiplyShape() {
        INDArray A = Nd4j.create(2, 3);
        INDArray B = Nd4j.create(3, 4);
        INDArray C = MatrixOps.multiply(A, B);
        assertArrayEquals(new long[]{2, 4}, C.shape());
    }

    @Test
    void testTransposeSquare() {
        INDArray A = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray T = MatrixOps.transpose(A);
        assertEquals(1.0, T.getDouble(0, 0), 0.001);
        assertEquals(3.0, T.getDouble(0, 1), 0.001);
    }

    @Test
    void testInverse() {
        INDArray A = Nd4j.eye(2);
        INDArray inv = MatrixOps.inverse(A);
        assertEquals(1.0, inv.getDouble(0, 0), 0.001);
    }

    @Test
    void testDotProductZero() {
        INDArray a = Nd4j.create(new double[]{1, 0, 0});
        INDArray b = Nd4j.create(new double[]{0, 1, 0});
        double dot = MatrixOps.dotProduct(a, b);
        assertEquals(0.0, dot, 0.001);
    }

    @Test
    void testNorm() {
        INDArray v = Nd4j.create(new double[]{3, 4});
        double norm = MatrixOps.norm(v);
        assertEquals(5.0, norm, 0.001);
    }

    @Test
    void testTransposePreservesElements() {
        INDArray A = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray T = MatrixOps.transpose(A);
        assertEquals(4, T.length());
    }

    @Test
    void testMultiplyIdentity() {
        INDArray A = Nd4j.create(new double[][]{{5, 6}, {7, 8}});
        INDArray I = Nd4j.eye(2);
        INDArray C = MatrixOps.multiply(A, I);
        assertEquals(5.0, C.getDouble(0, 0), 0.001);
    }
}`,

	hint1: 'Use mmul() for matrix multiplication, not mul()',
	hint2: 'InvertMatrix.invert() computes matrix inverse',

	whyItMatters: `Linear algebra is the foundation of ML:

- **Neural networks**: Weight matrices and activations
- **Optimization**: Gradient computations
- **Dimensionality reduction**: PCA, SVD
- **Efficiency**: Vectorized operations are fast

Understanding matrix ops is essential for ML.`,

	translations: {
		ru: {
			title: 'Матричные операции',
			description: `# Матричные операции

Выполняйте основные операции линейной алгебры для ML.

## Задача

Реализуйте матричные операции:
- Умножение матриц
- Транспонирование и обратная матрица
- Разложение по собственным значениям

## Пример

\`\`\`java
INDArray A = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray B = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray C = A.mmul(B);  // Matrix multiplication
INDArray T = A.transpose();  // Transpose
\`\`\``,
			hint1: 'Используйте mmul() для умножения матриц, не mul()',
			hint2: 'InvertMatrix.invert() вычисляет обратную матрицу',
			whyItMatters: `Линейная алгебра - основа ML:

- **Нейронные сети**: Матрицы весов и активации
- **Оптимизация**: Вычисления градиентов
- **Снижение размерности**: PCA, SVD
- **Эффективность**: Векторизованные операции быстры`,
		},
		uz: {
			title: 'Matritsa operatsiyalari',
			description: `# Matritsa operatsiyalari

ML uchun asosiy chiziqli algebra operatsiyalarini bajaring.

## Topshiriq

Matritsa operatsiyalarini amalga oshiring:
- Matritsa ko'paytirish
- Transpozitsiya va teskari matritsa
- Eigen qiymat dekompozitsiyasi

## Misol

\`\`\`java
INDArray A = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
INDArray B = Nd4j.create(new double[][]{{5, 6}, {7, 8}});

INDArray C = A.mmul(B);  // Matrix multiplication
INDArray T = A.transpose();  // Transpose
\`\`\``,
			hint1: "Matritsa ko'paytirish uchun mul() emas, mmul() dan foydalaning",
			hint2: "InvertMatrix.invert() teskari matritsani hisoblaydi",
			whyItMatters: `Chiziqli algebra ML ning asosi:

- **Neyron tarmoqlar**: Vazn matritsalari va aktivatsiyalar
- **Optimallashtirish**: Gradient hisoblashlari
- **O'lchamni kamaytirish**: PCA, SVD
- **Samaradorlik**: Vektorlashtirilgan operatsiyalar tez`,
		},
	},
};

export default task;
