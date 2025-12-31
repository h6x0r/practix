import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-xgboost',
	title: 'XGBoost Classifier',
	difficulty: 'medium',
	tags: ['tribuo', 'xgboost', 'gradient-boosting'],
	estimatedTime: '25m',
	isPremium: true,
	order: 4,
	description: `# XGBoost Classifier

Implement gradient boosting with XGBoost through Tribuo.

## Task

Build XGBoost classifiers:
- Configure boosting parameters
- Set learning rate and tree depth
- Handle regularization parameters

## Example

\`\`\`java
XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(
    100,   // number of trees
    0.1,   // learning rate
    6,     // max depth
    1.0,   // subsample
    1.0,   // colsample
    1      // num threads
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;

public class XGBoostClassifier {

    /**
     * @param numTrees Number of boosting rounds
     * @param learningRate Learning rate (eta)
     */
    public static XGBoostClassificationTrainer createTrainer(
            int numTrees, double learningRate, int maxDepth) {
        return null;
    }

    /**
     */
    public static XGBoostClassificationTrainer createRegularizedTrainer(
            int numTrees, double learningRate, int maxDepth,
            double l1Reg, double l2Reg) {
        return null;
    }

    /**
     */
    public static Model<Label> train(
            XGBoostClassificationTrainer trainer, Dataset<Label> data) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.common.xgboost.XGBoostTrainer;

public class XGBoostClassifier {

    /**
     * Create basic XGBoost trainer.
     * @param numTrees Number of boosting rounds
     * @param learningRate Learning rate (eta)
     * @param maxDepth Maximum tree depth
     */
    public static XGBoostClassificationTrainer createTrainer(
            int numTrees, double learningRate, int maxDepth) {
        return new XGBoostClassificationTrainer(
            numTrees,
            learningRate,
            0.0,     // min loss reduction (gamma)
            maxDepth,
            1.0,     // min child weight
            1.0,     // subsample ratio
            1.0,     // colsample by tree
            0.0,     // L1 regularization (alpha)
            1.0,     // L2 regularization (lambda)
            Trainer.DEFAULT_SEED,
            Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Create XGBoost with regularization.
     */
    public static XGBoostClassificationTrainer createRegularizedTrainer(
            int numTrees, double learningRate, int maxDepth,
            double l1Reg, double l2Reg) {
        return new XGBoostClassificationTrainer(
            numTrees,
            learningRate,
            0.0,     // min loss reduction
            maxDepth,
            1.0,     // min child weight
            0.8,     // subsample ratio (for regularization)
            0.8,     // colsample by tree (for regularization)
            l1Reg,   // L1 regularization (alpha)
            l2Reg,   // L2 regularization (lambda)
            Trainer.DEFAULT_SEED,
            Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Train XGBoost model.
     */
    public static Model<Label> train(
            XGBoostClassificationTrainer trainer, Dataset<Label> data) {
        return trainer.train(data);
    }

    /**
     * Create trainer for early stopping.
     */
    public static XGBoostClassificationTrainer createEarlyStoppingTrainer(
            int numTrees, double learningRate, int maxDepth) {
        // Configure with subsample for better generalization
        return new XGBoostClassificationTrainer(
            numTrees,
            learningRate,
            0.1,     // some gamma for pruning
            maxDepth,
            3.0,     // higher min child weight
            0.8,     // subsample
            0.8,     // colsample
            0.0,
            1.0,
            Trainer.DEFAULT_SEED,
            Runtime.getRuntime().availableProcessors()
        );
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class XGBoostClassifierTest {

    @Test
    void testCreateTrainer() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createTrainer(
            100, 0.1, 6);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRegularizedTrainer() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createRegularizedTrainer(
            100, 0.05, 4, 0.1, 1.0);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createTrainer(
            10, 0.3, 3);
        MutableDataset<Label> data = createTestDataset();

        Model<Label> model = XGBoostClassifier.train(trainer, data);
        assertNotNull(model);
    }

    private MutableDataset<Label> createTestDataset() {
        LabelFactory factory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 50; i++) {
            double[] features = {Math.random(), Math.random(), Math.random()};
            String label = (features[0] + features[1]) > 1.0 ? "A" : "B";
            dataset.add(new ArrayExample<>(new Label(label),
                new String[]{"f1", "f2", "f3"}, features));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerWithDifferentParams() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createTrainer(50, 0.2, 4);
        assertNotNull(trainer);
    }

    @Test
    void testCreateTrainerLowLearningRate() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createTrainer(200, 0.01, 8);
        assertNotNull(trainer);
    }

    @Test
    void testRegularizedTrainerL1Only() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createRegularizedTrainer(
            50, 0.1, 5, 0.5, 0.0);
        assertNotNull(trainer);
    }

    @Test
    void testRegularizedTrainerL2Only() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createRegularizedTrainer(
            50, 0.1, 5, 0.0, 2.0);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetNotEmpty() {
        MutableDataset<Label> data = createTestDataset();
        assertTrue(data.size() > 0);
    }

    @Test
    void testDatasetHasFiftyExamples() {
        MutableDataset<Label> data = createTestDataset();
        assertEquals(50, data.size());
    }

    @Test
    void testTrainModelNotNull() {
        XGBoostClassificationTrainer trainer = XGBoostClassifier.createTrainer(5, 0.5, 2);
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = XGBoostClassifier.train(trainer, data);
        assertNotNull(model);
    }
}`,

	hint1: 'XGBoostClassificationTrainer constructor takes many hyperparameters',
	hint2: 'Use subsample < 1.0 and colsample < 1.0 for regularization',

	whyItMatters: `XGBoost is a competition-winning algorithm:

- **State-of-the-art**: Wins many Kaggle competitions
- **Efficient**: Optimized for speed and memory
- **Regularized**: Built-in L1 and L2 regularization
- **Flexible**: Handles missing values and custom objectives

XGBoost is the go-to algorithm for tabular data problems.`,

	translations: {
		ru: {
			title: 'Классификатор XGBoost',
			description: `# Классификатор XGBoost

Реализуйте градиентный бустинг с XGBoost через Tribuo.

## Задача

Создайте классификаторы XGBoost:
- Настройте параметры бустинга
- Установите learning rate и глубину дерева
- Настройте параметры регуляризации

## Пример

\`\`\`java
XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(
    100,   // number of trees
    0.1,   // learning rate
    6,     // max depth
    1.0,   // subsample
    1.0,   // colsample
    1      // num threads
);
\`\`\``,
			hint1: 'Конструктор XGBoostClassificationTrainer принимает много гиперпараметров',
			hint2: 'Используйте subsample < 1.0 и colsample < 1.0 для регуляризации',
			whyItMatters: `XGBoost - алгоритм-победитель соревнований:

- **State-of-the-art**: Побеждает во многих соревнованиях Kaggle
- **Эффективность**: Оптимизирован для скорости и памяти
- **Регуляризация**: Встроенная L1 и L2 регуляризация
- **Гибкость**: Обрабатывает пропущенные значения и кастомные цели`,
		},
		uz: {
			title: 'XGBoost klassifikatori',
			description: `# XGBoost klassifikatori

Tribuo orqali XGBoost bilan gradient boosting ni amalga oshiring.

## Topshiriq

XGBoost klassifikatorlarini yarating:
- Boosting parametrlarini sozlang
- Learning rate va daraxt chuqurligini o'rnating
- Regularizatsiya parametrlarini boshqaring

## Misol

\`\`\`java
XGBoostClassificationTrainer trainer = new XGBoostClassificationTrainer(
    100,   // number of trees
    0.1,   // learning rate
    6,     // max depth
    1.0,   // subsample
    1.0,   // colsample
    1      // num threads
);
\`\`\``,
			hint1: "XGBoostClassificationTrainer konstruktori ko'p giperparametrlarni qabul qiladi",
			hint2: "Regularizatsiya uchun subsample < 1.0 va colsample < 1.0 dan foydalaning",
			whyItMatters: `XGBoost musobaqa g'olib algoritmi:

- **State-of-the-art**: Ko'p Kaggle musobaqalarida g'alaba qozonadi
- **Samarali**: Tezlik va xotira uchun optimallashtirilgan
- **Regularizatsiyalangan**: O'rnatilgan L1 va L2 regularizatsiya
- **Moslashuvchan**: Yo'qolgan qiymatlar va maxsus maqsadlarni boshqaradi`,
		},
	},
};

export default task;
