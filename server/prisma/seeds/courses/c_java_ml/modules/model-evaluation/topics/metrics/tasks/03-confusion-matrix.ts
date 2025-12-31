import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-confusion-matrix',
	title: 'Confusion Matrix',
	difficulty: 'medium',
	tags: ['metrics', 'confusion-matrix', 'classification'],
	estimatedTime: '20m',
	isPremium: false,
	order: 3,
	description: `# Confusion Matrix

Create and analyze confusion matrices for classification evaluation.

## Task

Implement confusion matrix operations:
- Build confusion matrix from predictions
- Extract TP, TN, FP, FN
- Calculate derived metrics

## Example

\`\`\`java
Evaluation eval = new Evaluation(3);
eval.eval(labels, predictions);
String matrix = eval.confusionMatrix();
\`\`\``,

	initialCode: `import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ConfusionMatrixAnalyzer {

    /**
     */
    public static int[][] getConfusionMatrix(INDArray actual, INDArray predicted, int numClasses) {
        return null;
    }

    /**
     */
    public static int getTruePositives(int[][] matrix, int classIndex) {
        return 0;
    }

    /**
     */
    public static int getFalsePositives(int[][] matrix, int classIndex) {
        return 0;
    }

    /**
     */
    public static int getFalseNegatives(int[][] matrix, int classIndex) {
        return 0;
    }
}`,

	solutionCode: `import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ConfusionMatrixAnalyzer {

    /**
     * Get confusion matrix as 2D array.
     */
    public static int[][] getConfusionMatrix(INDArray actual, INDArray predicted, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];
        int n = (int) actual.rows();

        for (int i = 0; i < n; i++) {
            int actualClass = (int) actual.getRow(i).argMax().getDouble(0);
            int predictedClass = (int) predicted.getRow(i).argMax().getDouble(0);
            matrix[actualClass][predictedClass]++;
        }

        return matrix;
    }

    /**
     * Calculate True Positives for a class.
     */
    public static int getTruePositives(int[][] matrix, int classIndex) {
        return matrix[classIndex][classIndex];
    }

    /**
     * Calculate False Positives for a class.
     */
    public static int getFalsePositives(int[][] matrix, int classIndex) {
        int fp = 0;
        for (int i = 0; i < matrix.length; i++) {
            if (i != classIndex) {
                fp += matrix[i][classIndex];
            }
        }
        return fp;
    }

    /**
     * Calculate False Negatives for a class.
     */
    public static int getFalseNegatives(int[][] matrix, int classIndex) {
        int fn = 0;
        for (int j = 0; j < matrix[classIndex].length; j++) {
            if (j != classIndex) {
                fn += matrix[classIndex][j];
            }
        }
        return fn;
    }

    /**
     * Calculate True Negatives for a class.
     */
    public static int getTrueNegatives(int[][] matrix, int classIndex) {
        int tn = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (i != classIndex && j != classIndex) {
                    tn += matrix[i][j];
                }
            }
        }
        return tn;
    }

    /**
     * Format confusion matrix as string.
     */
    public static String formatMatrix(int[][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (int[] row : matrix) {
            for (int val : row) {
                sb.append(String.format("%5d ", val));
            }
            sb.append("\\n");
        }
        return sb.toString();
    }

    /**
     * Calculate precision from confusion matrix.
     */
    public static double precisionFromMatrix(int[][] matrix, int classIndex) {
        int tp = getTruePositives(matrix, classIndex);
        int fp = getFalsePositives(matrix, classIndex);
        return tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ConfusionMatrixAnalyzerTest {

    @Test
    void testGetTruePositives() {
        int[][] matrix = {{10, 2}, {3, 15}};
        assertEquals(10, ConfusionMatrixAnalyzer.getTruePositives(matrix, 0));
        assertEquals(15, ConfusionMatrixAnalyzer.getTruePositives(matrix, 1));
    }

    @Test
    void testGetFalsePositives() {
        int[][] matrix = {{10, 2}, {3, 15}};
        assertEquals(3, ConfusionMatrixAnalyzer.getFalsePositives(matrix, 0));
        assertEquals(2, ConfusionMatrixAnalyzer.getFalsePositives(matrix, 1));
    }

    @Test
    void testGetFalseNegatives() {
        int[][] matrix = {{10, 2}, {3, 15}};
        assertEquals(2, ConfusionMatrixAnalyzer.getFalseNegatives(matrix, 0));
        assertEquals(3, ConfusionMatrixAnalyzer.getFalseNegatives(matrix, 1));
    }

    @Test
    void testPrecisionFromMatrix() {
        int[][] matrix = {{10, 2}, {3, 15}};
        double precision = ConfusionMatrixAnalyzer.precisionFromMatrix(matrix, 0);
        assertEquals(10.0 / 13.0, precision, 0.001);
    }

    @Test
    void testGetTrueNegatives() {
        int[][] matrix = {{10, 2}, {3, 15}};
        assertEquals(15, ConfusionMatrixAnalyzer.getTrueNegatives(matrix, 0));
        assertEquals(10, ConfusionMatrixAnalyzer.getTrueNegatives(matrix, 1));
    }

    @Test
    void testFormatMatrix() {
        int[][] matrix = {{10, 2}, {3, 15}};
        String formatted = ConfusionMatrixAnalyzer.formatMatrix(matrix);
        assertNotNull(formatted);
        assertTrue(formatted.contains("10"));
    }

    @Test
    void testPrecisionForSecondClass() {
        int[][] matrix = {{10, 2}, {3, 15}};
        double precision = ConfusionMatrixAnalyzer.precisionFromMatrix(matrix, 1);
        assertEquals(15.0 / 17.0, precision, 0.001);
    }

    @Test
    void testThreeClassMatrix() {
        int[][] matrix = {{10, 1, 1}, {2, 15, 1}, {1, 1, 20}};
        assertEquals(10, ConfusionMatrixAnalyzer.getTruePositives(matrix, 0));
        assertEquals(15, ConfusionMatrixAnalyzer.getTruePositives(matrix, 1));
        assertEquals(20, ConfusionMatrixAnalyzer.getTruePositives(matrix, 2));
    }

    @Test
    void testFalsePositivesThreeClass() {
        int[][] matrix = {{10, 1, 1}, {2, 15, 1}, {1, 1, 20}};
        assertEquals(3, ConfusionMatrixAnalyzer.getFalsePositives(matrix, 0));
    }

    @Test
    void testFalseNegativesThreeClass() {
        int[][] matrix = {{10, 1, 1}, {2, 15, 1}, {1, 1, 20}};
        assertEquals(2, ConfusionMatrixAnalyzer.getFalseNegatives(matrix, 0));
    }
}`,

	hint1: 'TP is the diagonal element matrix[i][i]',
	hint2: 'FP for class i is sum of column i excluding diagonal',

	whyItMatters: `Confusion matrices reveal classification details:

- **Error analysis**: See which classes are confused with each other
- **Imbalance detection**: Identify classes with poor performance
- **Metric derivation**: All classification metrics derive from the matrix
- **Debugging**: Essential for understanding model behavior

Understanding confusion matrices is fundamental to improving classifiers.`,

	translations: {
		ru: {
			title: 'Матрица ошибок',
			description: `# Матрица ошибок

Создавайте и анализируйте матрицы ошибок для оценки классификации.

## Задача

Реализуйте операции с матрицей ошибок:
- Построение матрицы ошибок из предсказаний
- Извлечение TP, TN, FP, FN
- Вычисление производных метрик

## Пример

\`\`\`java
Evaluation eval = new Evaluation(3);
eval.eval(labels, predictions);
String matrix = eval.confusionMatrix();
\`\`\``,
			hint1: 'TP это диагональный элемент matrix[i][i]',
			hint2: 'FP для класса i это сумма столбца i исключая диагональ',
			whyItMatters: `Матрицы ошибок раскрывают детали классификации:

- **Анализ ошибок**: Видно какие классы путаются друг с другом
- **Обнаружение дисбаланса**: Определение классов с плохой производительностью
- **Вывод метрик**: Все метрики классификации выводятся из матрицы
- **Отладка**: Необходимо для понимания поведения модели`,
		},
		uz: {
			title: 'Chalkashlik matritsasi',
			description: `# Chalkashlik matritsasi

Klassifikatsiya baholash uchun chalkashlik matritsalarini yarating va tahlil qiling.

## Topshiriq

Chalkashlik matritsasi operatsiyalarini amalga oshiring:
- Bashoratlardan chalkashlik matritsasini qurish
- TP, TN, FP, FN ni ajratib olish
- Hosila metrikalarni hisoblash

## Misol

\`\`\`java
Evaluation eval = new Evaluation(3);
eval.eval(labels, predictions);
String matrix = eval.confusionMatrix();
\`\`\``,
			hint1: "TP diagonal element matrix[i][i]",
			hint2: "i sinfi uchun FP diagonal dan tashqari i ustun yig'indisi",
			whyItMatters: `Chalkashlik matritsalari klassifikatsiya tafsilotlarini ochadi:

- **Xato tahlili**: Qaysi sinflar bir-biri bilan chalkashtirilayotganini ko'rish
- **Muvozanatsizlikni aniqlash**: Yomon samaradorlikka ega sinflarni aniqlash
- **Metrika chiqarish**: Barcha klassifikatsiya metrikalari matritsadan chiqariladi
- **Debugging**: Model xatti-harakatini tushunish uchun zarur`,
		},
	},
};

export default task;
