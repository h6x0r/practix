import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-regression-metrics',
	title: 'Regression Metrics',
	difficulty: 'easy',
	tags: ['metrics', 'mse', 'mae', 'r2', 'rmse'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Regression Metrics

Calculate MSE, MAE, RMSE, and R² for regression models.

## Task

Implement regression metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (Coefficient of Determination)

## Example

\`\`\`java
RegressionEvaluation eval = new RegressionEvaluation();
eval.eval(labels, predictions);
double mse = eval.meanSquaredError(0);
\`\`\``,

	initialCode: `import org.deeplearning4j.eval.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RegressionMetrics {

    /**
     */
    public static double calculateMSE(INDArray actual, INDArray predicted) {
        return 0.0;
    }

    /**
     */
    public static double calculateMAE(INDArray actual, INDArray predicted) {
        return 0.0;
    }

    /**
     */
    public static double calculateRMSE(INDArray actual, INDArray predicted) {
        return 0.0;
    }

    /**
     */
    public static double calculateR2(INDArray actual, INDArray predicted) {
        return 0.0;
    }
}`,

	solutionCode: `import org.deeplearning4j.eval.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class RegressionMetrics {

    /**
     * Calculate Mean Squared Error.
     */
    public static double calculateMSE(INDArray actual, INDArray predicted) {
        INDArray diff = actual.sub(predicted);
        INDArray squared = diff.mul(diff);
        return squared.meanNumber().doubleValue();
    }

    /**
     * Calculate Mean Absolute Error.
     */
    public static double calculateMAE(INDArray actual, INDArray predicted) {
        INDArray diff = actual.sub(predicted);
        INDArray abs = Transforms.abs(diff);
        return abs.meanNumber().doubleValue();
    }

    /**
     * Calculate Root Mean Squared Error.
     */
    public static double calculateRMSE(INDArray actual, INDArray predicted) {
        return Math.sqrt(calculateMSE(actual, predicted));
    }

    /**
     * Calculate R² score.
     */
    public static double calculateR2(INDArray actual, INDArray predicted) {
        double mean = actual.meanNumber().doubleValue();
        INDArray ssRes = actual.sub(predicted);
        ssRes = ssRes.mul(ssRes);
        double ssResSum = ssRes.sumNumber().doubleValue();

        INDArray ssTot = actual.sub(mean);
        ssTot = ssTot.mul(ssTot);
        double ssTotSum = ssTot.sumNumber().doubleValue();

        return 1.0 - (ssResSum / ssTotSum);
    }

    /**
     * Use DL4J RegressionEvaluation for metrics.
     */
    public static RegressionEvaluation evaluate(INDArray labels, INDArray predictions) {
        RegressionEvaluation eval = new RegressionEvaluation(1);
        eval.eval(labels, predictions);
        return eval;
    }

    /**
     * Calculate all metrics at once.
     */
    public static double[] calculateAllMetrics(INDArray actual, INDArray predicted) {
        return new double[] {
            calculateMSE(actual, predicted),
            calculateMAE(actual, predicted),
            calculateRMSE(actual, predicted),
            calculateR2(actual, predicted)
        };
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class RegressionMetricsTest {

    @Test
    void testCalculateMSE() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        assertEquals(0.0, RegressionMetrics.calculateMSE(actual, predicted), 0.001);
    }

    @Test
    void testCalculateMAE() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{2.0, 3.0, 4.0});
        assertEquals(1.0, RegressionMetrics.calculateMAE(actual, predicted), 0.001);
    }

    @Test
    void testCalculateRMSE() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        assertEquals(0.0, RegressionMetrics.calculateRMSE(actual, predicted), 0.001);
    }

    @Test
    void testCalculateR2Perfect() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        assertEquals(1.0, RegressionMetrics.calculateR2(actual, predicted), 0.001);
    }

    @Test
    void testMSEWithError() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{2.0, 3.0, 4.0});
        double mse = RegressionMetrics.calculateMSE(actual, predicted);
        assertEquals(1.0, mse, 0.001);
    }

    @Test
    void testRMSEWithError() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        INDArray predicted = Nd4j.create(new double[]{2.0, 3.0, 4.0, 5.0});
        double rmse = RegressionMetrics.calculateRMSE(actual, predicted);
        assertEquals(1.0, rmse, 0.001);
    }

    @Test
    void testCalculateAllMetrics() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray predicted = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        double[] metrics = RegressionMetrics.calculateAllMetrics(actual, predicted);
        assertEquals(4, metrics.length);
    }

    @Test
    void testEvaluate() {
        INDArray labels = Nd4j.create(new double[][]{{1.0}, {2.0}, {3.0}});
        INDArray predictions = Nd4j.create(new double[][]{{1.1}, {2.1}, {3.1}});
        var eval = RegressionMetrics.evaluate(labels, predictions);
        assertNotNull(eval);
    }

    @Test
    void testR2Range() {
        INDArray actual = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        INDArray predicted = Nd4j.create(new double[]{1.2, 2.1, 2.9, 4.2, 4.8});
        double r2 = RegressionMetrics.calculateR2(actual, predicted);
        assertTrue(r2 >= 0 && r2 <= 1);
    }

    @Test
    void testMAEWithError() {
        INDArray actual = Nd4j.create(new double[]{10.0, 20.0, 30.0});
        INDArray predicted = Nd4j.create(new double[]{12.0, 18.0, 33.0});
        double mae = RegressionMetrics.calculateMAE(actual, predicted);
        assertEquals(2.0, mae, 0.001);
    }
}`,

	hint1: 'MSE = mean((actual - predicted)²)',
	hint2: 'R² = 1 - (SS_residual / SS_total)',

	whyItMatters: `Regression metrics quantify prediction error:

- **MSE**: Penalizes large errors more heavily
- **MAE**: More robust to outliers than MSE
- **RMSE**: Same units as target variable
- **R²**: Proportion of variance explained (0-1 scale)

Different metrics reveal different aspects of model performance.`,

	translations: {
		ru: {
			title: 'Метрики регрессии',
			description: `# Метрики регрессии

Вычисляйте MSE, MAE, RMSE и R² для моделей регрессии.

## Задача

Реализуйте метрики регрессии:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (коэффициент детерминации)

## Пример

\`\`\`java
RegressionEvaluation eval = new RegressionEvaluation();
eval.eval(labels, predictions);
double mse = eval.meanSquaredError(0);
\`\`\``,
			hint1: 'MSE = mean((actual - predicted)²)',
			hint2: 'R² = 1 - (SS_residual / SS_total)',
			whyItMatters: `Метрики регрессии количественно оценивают ошибку предсказания:

- **MSE**: Сильнее штрафует большие ошибки
- **MAE**: Более устойчив к выбросам чем MSE
- **RMSE**: Те же единицы что и целевая переменная
- **R²**: Доля объясненной дисперсии (шкала 0-1)`,
		},
		uz: {
			title: 'Regressiya metrikalari',
			description: `# Regressiya metrikalari

Regressiya modellari uchun MSE, MAE, RMSE va R² ni hisoblang.

## Topshiriq

Regressiya metrikalarini amalga oshiring:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (determinatsiya koeffitsiyenti)

## Misol

\`\`\`java
RegressionEvaluation eval = new RegressionEvaluation();
eval.eval(labels, predictions);
double mse = eval.meanSquaredError(0);
\`\`\``,
			hint1: 'MSE = mean((actual - predicted)²)',
			hint2: 'R² = 1 - (SS_residual / SS_total)',
			whyItMatters: `Regressiya metrikalari bashorat xatosini miqdoriy baholaydi:

- **MSE**: Katta xatolarni ko'proq jarimaydi
- **MAE**: MSE dan ko'ra outlierlarga barqarorroq
- **RMSE**: Maqsad o'zgaruvchisi bilan bir xil birliklar
- **R²**: Tushuntirilgan dispersiya ulushi (0-1 shkalasi)`,
		},
	},
};

export default task;
