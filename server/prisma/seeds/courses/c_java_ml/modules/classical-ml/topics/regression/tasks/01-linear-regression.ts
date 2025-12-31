import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-linear-regression',
	title: 'Linear Regression',
	difficulty: 'easy',
	tags: ['tribuo', 'regression', 'linear'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Linear Regression

Implement linear regression for continuous predictions using Tribuo.

## Task

Build linear regression models:
- Configure SGD-based trainer
- Train on regression data
- Evaluate with MSE and R-squared

## Example

\`\`\`java
LinearSGDTrainer trainer = new LinearSGDTrainer(
    new AdaGrad(0.1),
    100,  // epochs
    Trainer.DEFAULT_SEED
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.math.optimisers.*;

public class LinearRegressionModel {

    /**
     * @param learningRate Learning rate for SGD
     * @param epochs Number of training epochs
     */
    public static LinearSGDTrainer createTrainer(double learningRate, int epochs) {
        return null;
    }

    /**
     */
    public static Model<Regressor> train(
            LinearSGDTrainer trainer, Dataset<Regressor> data) {
        return null;
    }

    /**
     */
    public static double evaluateRMSE(
            Model<Regressor> model, Dataset<Regressor> testData) {
        return 0.0;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.math.optimisers.*;

public class LinearRegressionModel {

    /**
     * Create linear regression trainer.
     * @param learningRate Learning rate for SGD
     * @param epochs Number of training epochs
     */
    public static LinearSGDTrainer createTrainer(double learningRate, int epochs) {
        return new LinearSGDTrainer(
            new AdaGrad(learningRate),
            epochs,
            Trainer.DEFAULT_SEED
        );
    }

    /**
     * Train linear regression model.
     */
    public static Model<Regressor> train(
            LinearSGDTrainer trainer, Dataset<Regressor> data) {
        return trainer.train(data);
    }

    /**
     * Evaluate model and get RMSE.
     */
    public static double evaluateRMSE(
            Model<Regressor> model, Dataset<Regressor> testData) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        RegressionEvaluation eval = evaluator.evaluate(model, testData);
        return eval.rmse();
    }

    /**
     * Get R-squared score.
     */
    public static double evaluateR2(
            Model<Regressor> model, Dataset<Regressor> testData) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        RegressionEvaluation eval = evaluator.evaluate(model, testData);
        return eval.r2();
    }

    /**
     * Make single prediction.
     */
    public static double predict(Model<Regressor> model, Example<Regressor> example) {
        Prediction<Regressor> prediction = model.predict(example);
        return prediction.getOutput().getValues()[0];
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class LinearRegressionModelTest {

    @Test
    void testCreateTrainer() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.01, 100);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.1, 50);
        MutableDataset<Regressor> data = createTestDataset();

        Model<Regressor> model = LinearRegressionModel.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testEvaluateRMSE() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.1, 100);
        MutableDataset<Regressor> data = createTestDataset();

        Model<Regressor> model = LinearRegressionModel.train(trainer, data);
        double rmse = LinearRegressionModel.evaluateRMSE(model, data);

        assertTrue(rmse >= 0);
    }

    private MutableDataset<Regressor> createTestDataset() {
        RegressionFactory factory = new RegressionFactory();
        MutableDataset<Regressor> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 50; i++) {
            double x = Math.random() * 10;
            double y = 2 * x + 1 + Math.random() * 0.1; // y = 2x + 1 + noise
            dataset.add(new ArrayExample<>(new Regressor("target", y),
                new String[]{"x"}, new double[]{x}));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerReturnsType() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.05, 50);
        assertInstanceOf(LinearSGDTrainer.class, trainer);
    }

    @Test
    void testTrainReturnsModel() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.1, 10);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = LinearRegressionModel.train(trainer, data);
        assertInstanceOf(Model.class, model);
    }

    @Test
    void testRMSEIsPositive() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.1, 50);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = LinearRegressionModel.train(trainer, data);
        double rmse = LinearRegressionModel.evaluateRMSE(model, data);
        assertTrue(rmse > 0 || rmse == 0);
    }

    @Test
    void testDifferentLearningRates() {
        LinearSGDTrainer trainer1 = LinearRegressionModel.createTrainer(0.01, 50);
        LinearSGDTrainer trainer2 = LinearRegressionModel.createTrainer(0.1, 50);
        assertNotNull(trainer1);
        assertNotNull(trainer2);
    }

    @Test
    void testDifferentEpochs() {
        LinearSGDTrainer trainer1 = LinearRegressionModel.createTrainer(0.1, 10);
        LinearSGDTrainer trainer2 = LinearRegressionModel.createTrainer(0.1, 100);
        assertNotNull(trainer1);
        assertNotNull(trainer2);
    }

    @Test
    void testModelNotNull() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.1, 20);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = LinearRegressionModel.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testTrainerWithHighEpochs() {
        LinearSGDTrainer trainer = LinearRegressionModel.createTrainer(0.01, 200);
        assertNotNull(trainer);
    }
}`,

	hint1: 'Use LinearSGDTrainer with an optimizer like AdaGrad',
	hint2: 'RegressionEvaluator.evaluate() returns RegressionEvaluation with rmse() and r2()',

	whyItMatters: `Linear regression is the foundation of predictive modeling:

- **Interpretable**: Coefficients show feature effects
- **Fast**: Trains quickly even on large datasets
- **Baseline**: Compare complex models against it
- **Statistical foundation**: Basis for many advanced methods

Understanding linear regression is essential for any ML practitioner.`,

	translations: {
		ru: {
			title: 'Линейная регрессия',
			description: `# Линейная регрессия

Реализуйте линейную регрессию для непрерывных предсказаний с Tribuo.

## Задача

Создайте модели линейной регрессии:
- Настройте тренер на основе SGD
- Обучите на данных регрессии
- Оцените с MSE и R-squared

## Пример

\`\`\`java
LinearSGDTrainer trainer = new LinearSGDTrainer(
    new AdaGrad(0.1),
    100,  // epochs
    Trainer.DEFAULT_SEED
);
\`\`\``,
			hint1: 'Используйте LinearSGDTrainer с оптимизатором типа AdaGrad',
			hint2: 'RegressionEvaluator.evaluate() возвращает RegressionEvaluation с rmse() и r2()',
			whyItMatters: `Линейная регрессия - основа предиктивного моделирования:

- **Интерпретируемость**: Коэффициенты показывают эффекты признаков
- **Скорость**: Быстро обучается даже на больших датасетах
- **Базовая модель**: Сравнивайте сложные модели с ней
- **Статистическая основа**: База для многих продвинутых методов`,
		},
		uz: {
			title: 'Lineer regressiya',
			description: `# Lineer regressiya

Tribuo bilan uzluksiz bashoratlar uchun lineer regressiyani amalga oshiring.

## Topshiriq

Lineer regressiya modellarini yarating:
- SGD asosidagi trenerni sozlang
- Regressiya datalarida o'rgating
- MSE va R-squared bilan baholang

## Misol

\`\`\`java
LinearSGDTrainer trainer = new LinearSGDTrainer(
    new AdaGrad(0.1),
    100,  // epochs
    Trainer.DEFAULT_SEED
);
\`\`\``,
			hint1: "AdaGrad kabi optimallashtiruvchi bilan LinearSGDTrainer dan foydalaning",
			hint2: "RegressionEvaluator.evaluate() rmse() va r2() bilan RegressionEvaluation qaytaradi",
			whyItMatters: `Lineer regressiya prediktiv modellashtirishning asosi:

- **Interpretatsiya qilinadigan**: Koeffitsientlar xususiyat ta'sirlarini ko'rsatadi
- **Tez**: Katta datasetlarda ham tez o'qitiladi
- **Boshlang'ich model**: Murakkab modellarni u bilan solishtiring
- **Statistik asos**: Ko'p ilg'or metodlar uchun asos`,
		},
	},
};

export default task;
