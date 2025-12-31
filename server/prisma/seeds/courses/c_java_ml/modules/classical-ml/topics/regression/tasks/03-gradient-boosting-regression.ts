import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-gradient-boosting-regression',
	title: 'Gradient Boosting Regression',
	difficulty: 'medium',
	tags: ['tribuo', 'xgboost', 'regression'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# Gradient Boosting Regression

Use XGBoost for powerful regression predictions.

## Task

Implement gradient boosting regression:
- Configure XGBoost for regression
- Tune hyperparameters
- Handle overfitting

## Example

\`\`\`java
XGBoostRegressionTrainer trainer = new XGBoostRegressionTrainer(
    100,   // trees
    0.1,   // learning rate
    6      // max depth
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;

public class GradientBoostingRegressor {

    /**
     * Create XGBoost regression trainer.
     */
    public static XGBoostRegressionTrainer createTrainer(
            int numTrees, double learningRate, int maxDepth) {
        return null;
    }

    /**
     * Create trainer with regularization.
     */
    public static XGBoostRegressionTrainer createRegularizedTrainer(
            int numTrees, double learningRate, int maxDepth,
            double l2Reg, double subsample) {
        return null;
    }

    /**
     * Train and return model.
     */
    public static Model<Regressor> train(
            XGBoostRegressionTrainer trainer, Dataset<Regressor> data) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;

public class GradientBoostingRegressor {

    /**
     * Create XGBoost regression trainer.
     */
    public static XGBoostRegressionTrainer createTrainer(
            int numTrees, double learningRate, int maxDepth) {
        return new XGBoostRegressionTrainer(
            numTrees,
            learningRate,
            0.0,     // gamma
            maxDepth,
            1.0,     // min child weight
            1.0,     // subsample
            1.0,     // colsample
            0.0,     // alpha (L1)
            1.0,     // lambda (L2)
            Trainer.DEFAULT_SEED,
            Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Create trainer with regularization.
     */
    public static XGBoostRegressionTrainer createRegularizedTrainer(
            int numTrees, double learningRate, int maxDepth,
            double l2Reg, double subsample) {
        return new XGBoostRegressionTrainer(
            numTrees,
            learningRate,
            0.1,     // gamma for pruning
            maxDepth,
            3.0,     // min child weight
            subsample,
            0.8,     // colsample
            0.0,     // alpha (L1)
            l2Reg,   // lambda (L2)
            Trainer.DEFAULT_SEED,
            Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Train and return model.
     */
    public static Model<Regressor> train(
            XGBoostRegressionTrainer trainer, Dataset<Regressor> data) {
        return trainer.train(data);
    }

    /**
     * Evaluate model performance.
     */
    public static RegressionEvaluation evaluate(
            Model<Regressor> model, Dataset<Regressor> testData) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        return evaluator.evaluate(model, testData);
    }

    /**
     * Get all metrics.
     */
    public static String getMetricsSummary(RegressionEvaluation eval) {
        return String.format(
            "RMSE: %.4f, MAE: %.4f, R2: %.4f",
            eval.rmse(), eval.mae(), eval.r2()
        );
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class GradientBoostingRegressorTest {

    @Test
    void testCreateTrainer() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createTrainer(
            100, 0.1, 6);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRegularizedTrainer() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createRegularizedTrainer(
            50, 0.05, 4, 1.0, 0.8);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createTrainer(
            10, 0.3, 3);
        MutableDataset<Regressor> data = createTestDataset();

        Model<Regressor> model = GradientBoostingRegressor.train(trainer, data);
        assertNotNull(model);
    }

    private MutableDataset<Regressor> createTestDataset() {
        RegressionFactory factory = new RegressionFactory();
        MutableDataset<Regressor> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 100; i++) {
            double x1 = Math.random() * 10;
            double x2 = Math.random() * 5;
            double y = Math.sin(x1) + x2 * 0.5 + Math.random() * 0.1;
            dataset.add(new ArrayExample<>(new Regressor("target", y),
                new String[]{"x1", "x2"}, new double[]{x1, x2}));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerLowLearningRate() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createTrainer(200, 0.01, 4);
        assertNotNull(trainer);
    }

    @Test
    void testCreateTrainerHighLearningRate() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createTrainer(20, 0.5, 3);
        assertNotNull(trainer);
    }

    @Test
    void testRegularizedTrainerDifferentParams() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createRegularizedTrainer(
            100, 0.1, 6, 2.0, 0.7);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetSize() {
        MutableDataset<Regressor> data = createTestDataset();
        assertEquals(100, data.size());
    }

    @Test
    void testTrainedModelNotNull() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createTrainer(5, 0.3, 3);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = GradientBoostingRegressor.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testRegularizedLowSubsample() {
        XGBoostRegressionTrainer trainer = GradientBoostingRegressor.createRegularizedTrainer(
            50, 0.05, 4, 0.5, 0.5);
        assertNotNull(trainer);
    }

    @Test
    void testMultipleTrainersCanBeCreated() {
        XGBoostRegressionTrainer t1 = GradientBoostingRegressor.createTrainer(10, 0.1, 4);
        XGBoostRegressionTrainer t2 = GradientBoostingRegressor.createTrainer(20, 0.2, 6);
        assertNotNull(t1);
        assertNotNull(t2);
    }
}`,

	hint1: 'XGBoostRegressionTrainer has same parameters as classification version',
	hint2: 'Lower learning rate with more trees often gives better results',

	whyItMatters: `XGBoost regression is extremely powerful:

- **Accuracy**: State-of-the-art for tabular data
- **Handles complexity**: Non-linear relationships and interactions
- **Feature selection**: Automatic importance ranking
- **Production-ready**: Optimized for real-world deployment

XGBoost is the default choice for regression competitions.`,

	translations: {
		ru: {
			title: 'Градиентный бустинг регрессия',
			description: `# Градиентный бустинг регрессия

Используйте XGBoost для мощных регрессионных предсказаний.

## Задача

Реализуйте градиентный бустинг регрессию:
- Настройте XGBoost для регрессии
- Подберите гиперпараметры
- Обработайте переобучение

## Пример

\`\`\`java
XGBoostRegressionTrainer trainer = new XGBoostRegressionTrainer(
    100,   // trees
    0.1,   // learning rate
    6      // max depth
);
\`\`\``,
			hint1: 'XGBoostRegressionTrainer имеет те же параметры что и версия для классификации',
			hint2: 'Низкий learning rate с большим числом деревьев часто дает лучшие результаты',
			whyItMatters: `XGBoost регрессия чрезвычайно мощна:

- **Точность**: State-of-the-art для табличных данных
- **Обработка сложности**: Нелинейные отношения и взаимодействия
- **Отбор признаков**: Автоматическое ранжирование важности
- **Production-ready**: Оптимизирован для реального развертывания`,
		},
		uz: {
			title: 'Gradient boosting regressiya',
			description: `# Gradient boosting regressiya

Kuchli regressiya bashoratlari uchun XGBoost dan foydalaning.

## Topshiriq

Gradient boosting regressiyasini amalga oshiring:
- Regressiya uchun XGBoostni sozlang
- Giperparametrlarni sozlang
- Overfittingni boshqaring

## Misol

\`\`\`java
XGBoostRegressionTrainer trainer = new XGBoostRegressionTrainer(
    100,   // trees
    0.1,   // learning rate
    6      // max depth
);
\`\`\``,
			hint1: "XGBoostRegressionTrainer klassifikatsiya versiyasi bilan bir xil parametrlarga ega",
			hint2: "Ko'proq daraxtlar bilan past learning rate ko'pincha yaxshiroq natija beradi",
			whyItMatters: `XGBoost regressiya juda kuchli:

- **Aniqlik**: Jadval ma'lumotlari uchun state-of-the-art
- **Murakkablikni boshqarish**: Nolinear munosabatlar va o'zaro ta'sirlar
- **Xususiyat tanlash**: Avtomatik muhimlik reytingi
- **Production-ready**: Haqiqiy deployment uchun optimallashtirilgan`,
		},
	},
};

export default task;
