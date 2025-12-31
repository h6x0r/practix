import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-logistic-regression',
	title: 'Logistic Regression',
	difficulty: 'easy',
	tags: ['tribuo', 'classification', 'logistic-regression'],
	estimatedTime: '20m',
	isPremium: false,
	order: 1,
	description: `# Logistic Regression

Implement binary and multiclass logistic regression using Tribuo.

## Task

Build logistic regression classifiers:
- Load and prepare dataset
- Train logistic regression model
- Evaluate accuracy and confusion matrix

## Example

\`\`\`java
var trainer = new LogisticRegressionTrainer();
Model<Label> model = trainer.train(trainData);
LabelEvaluation eval = evaluator.evaluate(model, testData);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.evaluation.TrainTestSplitter;

public class LogisticRegressionClassifier {

    /**
     * @param dataset Training dataset
     */
    public static Model<Label> trainModel(Dataset<Label> dataset) {
        return null;
    }

    /**
     * @param model Trained model
     * @param testData Test dataset
     */
    public static LabelEvaluation evaluateModel(
            Model<Label> model, Dataset<Label> testData) {
        return null;
    }

    /**
     */
    public static Prediction<Label> predict(
            Model<Label> model, Example<Label> example) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.evaluation.TrainTestSplitter;

public class LogisticRegressionClassifier {

    /**
     * Train a logistic regression model.
     * @param dataset Training dataset
     */
    public static Model<Label> trainModel(Dataset<Label> dataset) {
        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        return trainer.train(dataset);
    }

    /**
     * Evaluate the model on test data.
     * @param model Trained model
     * @param testData Test dataset
     */
    public static LabelEvaluation evaluateModel(
            Model<Label> model, Dataset<Label> testData) {
        LabelEvaluator evaluator = new LabelEvaluator();
        return evaluator.evaluate(model, testData);
    }

    /**
     * Make prediction for a single example.
     */
    public static Prediction<Label> predict(
            Model<Label> model, Example<Label> example) {
        return model.predict(example);
    }

    /**
     * Get accuracy from evaluation.
     */
    public static double getAccuracy(LabelEvaluation eval) {
        return eval.accuracy();
    }

    /**
     * Print confusion matrix.
     */
    public static String getConfusionMatrix(LabelEvaluation eval) {
        return eval.getConfusionMatrix().toString();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class LogisticRegressionClassifierTest {

    @Test
    void testTrainModel() {
        // Create simple test data
        MutableDataset<Label> dataset = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(dataset);
        assertNotNull(model);
    }

    @Test
    void testEvaluateModel() {
        MutableDataset<Label> trainData = createTestDataset();
        MutableDataset<Label> testData = createTestDataset();

        Model<Label> model = LogisticRegressionClassifier.trainModel(trainData);
        LabelEvaluation eval = LogisticRegressionClassifier.evaluateModel(model, testData);

        assertNotNull(eval);
        assertTrue(eval.accuracy() >= 0 && eval.accuracy() <= 1);
    }

    private MutableDataset<Label> createTestDataset() {
        LabelFactory factory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        // Add sample data points
        dataset.add(new ArrayExample<>(new Label("A"),
            new String[]{"f1", "f2"}, new double[]{1.0, 0.0}));
        dataset.add(new ArrayExample<>(new Label("B"),
            new String[]{"f1", "f2"}, new double[]{0.0, 1.0}));

        return dataset;
    }

    @Test
    void testTrainModelReturnsType() {
        MutableDataset<Label> dataset = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(dataset);
        assertInstanceOf(Model.class, model);
    }

    @Test
    void testEvaluateModelReturnsType() {
        MutableDataset<Label> trainData = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(trainData);
        LabelEvaluation eval = LogisticRegressionClassifier.evaluateModel(model, trainData);
        assertInstanceOf(LabelEvaluation.class, eval);
    }

    @Test
    void testAccuracyRange() {
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(data);
        LabelEvaluation eval = LogisticRegressionClassifier.evaluateModel(model, data);
        assertTrue(eval.accuracy() >= 0 && eval.accuracy() <= 1);
    }

    @Test
    void testPredict() {
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(data);
        Example<Label> example = new ArrayExample<>(new Label("A"),
            new String[]{"f1", "f2"}, new double[]{1.0, 0.0});
        Prediction<Label> pred = LogisticRegressionClassifier.predict(model, example);
        assertNotNull(pred);
    }

    @Test
    void testPredictNotNull() {
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(data);
        Example<Label> example = new ArrayExample<>(new Label("B"),
            new String[]{"f1", "f2"}, new double[]{0.0, 1.0});
        Prediction<Label> pred = LogisticRegressionClassifier.predict(model, example);
        assertNotNull(pred.getOutput());
    }

    @Test
    void testMultipleExamples() {
        MutableDataset<Label> dataset = createTestDataset();
        dataset.add(new ArrayExample<>(new Label("A"),
            new String[]{"f1", "f2"}, new double[]{0.9, 0.1}));
        dataset.add(new ArrayExample<>(new Label("B"),
            new String[]{"f1", "f2"}, new double[]{0.1, 0.9}));
        Model<Label> model = LogisticRegressionClassifier.trainModel(dataset);
        assertNotNull(model);
    }

    @Test
    void testModelNotNull() {
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(data);
        assertNotNull(model);
    }

    @Test
    void testEvaluationNotNull() {
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = LogisticRegressionClassifier.trainModel(data);
        LabelEvaluation eval = LogisticRegressionClassifier.evaluateModel(model, data);
        assertNotNull(eval);
    }
}`,

	hint1: 'Use LogisticRegressionTrainer from Tribuo SGD package',
	hint2: 'LabelEvaluator.evaluate() returns LabelEvaluation with metrics',

	whyItMatters: `Logistic regression is a foundational classification algorithm:

- **Interpretable**: Coefficients show feature importance
- **Probabilistic**: Outputs class probabilities
- **Fast training**: Scales to large datasets
- **Baseline model**: Compare more complex models against it

Understanding logistic regression helps with all classification tasks.`,

	translations: {
		ru: {
			title: 'Логистическая регрессия',
			description: `# Логистическая регрессия

Реализуйте бинарную и многоклассовую логистическую регрессию с Tribuo.

## Задача

Создайте классификаторы логистической регрессии:
- Загрузите и подготовьте датасет
- Обучите модель логистической регрессии
- Оцените accuracy и матрицу ошибок

## Пример

\`\`\`java
var trainer = new LogisticRegressionTrainer();
Model<Label> model = trainer.train(trainData);
LabelEvaluation eval = evaluator.evaluate(model, testData);
\`\`\``,
			hint1: 'Используйте LogisticRegressionTrainer из пакета Tribuo SGD',
			hint2: 'LabelEvaluator.evaluate() возвращает LabelEvaluation с метриками',
			whyItMatters: `Логистическая регрессия - фундаментальный алгоритм классификации:

- **Интерпретируемость**: Коэффициенты показывают важность признаков
- **Вероятностный**: Выводит вероятности классов
- **Быстрое обучение**: Масштабируется на большие датасеты
- **Базовая модель**: Сравнивайте более сложные модели с ней`,
		},
		uz: {
			title: 'Logistik regressiya',
			description: `# Logistik regressiya

Tribuo bilan binary va ko'p sinfli logistik regressiyani amalga oshiring.

## Topshiriq

Logistik regressiya klassifikatorlarini yarating:
- Datasetni yuklang va tayyorlang
- Logistik regressiya modelini o'rgating
- Accuracy va confusion matrixni baholang

## Misol

\`\`\`java
var trainer = new LogisticRegressionTrainer();
Model<Label> model = trainer.train(trainData);
LabelEvaluation eval = evaluator.evaluate(model, testData);
\`\`\``,
			hint1: "Tribuo SGD paketidan LogisticRegressionTrainer dan foydalaning",
			hint2: "LabelEvaluator.evaluate() metrikalar bilan LabelEvaluation qaytaradi",
			whyItMatters: `Logistik regressiya fundamental klassifikatsiya algoritmi:

- **Interpretatsiya qilinadigan**: Koeffitsientlar xususiyat muhimligini ko'rsatadi
- **Ehtimollik**: Sinf ehtimolliklarini chiqaradi
- **Tez o'qitish**: Katta datasetlarga masshtablanadi
- **Boshlang'ich model**: Murakkabroq modellarni u bilan solishtiring`,
		},
	},
};

export default task;
