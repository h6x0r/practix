import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-regression-tree',
	title: 'Regression Tree',
	difficulty: 'easy',
	tags: ['tribuo', 'regression', 'decision-tree'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Regression Tree

Build decision trees for regression tasks with Tribuo.

## Task

Implement regression trees:
- Configure tree parameters
- Train on continuous target
- Evaluate predictions

## Example

\`\`\`java
CARTRegressionTrainer trainer = new CARTRegressionTrainer(
    6,    // max depth
    5,    // min samples in leaf
    0.0f  // min impurity decrease
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;

public class RegressionTreeModel {

    /**
     * Create regression tree trainer.
     * @param maxDepth Maximum tree depth
     */
    public static CARTRegressionTrainer createTrainer(int maxDepth) {
        return null;
    }

    /**
     * Create trainer with custom parameters.
     */
    public static CARTRegressionTrainer createCustomTrainer(
            int maxDepth, int minSamplesLeaf) {
        return null;
    }

    /**
     * Train regression tree model.
     */
    public static Model<Regressor> train(
            CARTRegressionTrainer trainer, Dataset<Regressor> data) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.evaluation.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.common.tree.TreeModel;

public class RegressionTreeModel {

    /**
     * Create regression tree trainer.
     * @param maxDepth Maximum tree depth
     */
    public static CARTRegressionTrainer createTrainer(int maxDepth) {
        return new CARTRegressionTrainer(maxDepth);
    }

    /**
     * Create trainer with custom parameters.
     */
    public static CARTRegressionTrainer createCustomTrainer(
            int maxDepth, int minSamplesLeaf) {
        return new CARTRegressionTrainer(
            maxDepth,
            minSamplesLeaf,
            0.0f,  // min impurity decrease
            1.0f,  // fraction of features
            true,  // use random split
            Trainer.DEFAULT_SEED
        );
    }

    /**
     * Train regression tree model.
     */
    public static Model<Regressor> train(
            CARTRegressionTrainer trainer, Dataset<Regressor> data) {
        return trainer.train(data);
    }

    /**
     * Get feature importances.
     */
    public static java.util.Map<String, Double> getFeatureImportances(
            TreeModel<Regressor> model) {
        return model.getFeatureImportances();
    }

    /**
     * Evaluate with MAE.
     */
    public static double evaluateMAE(
            Model<Regressor> model, Dataset<Regressor> testData) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        RegressionEvaluation eval = evaluator.evaluate(model, testData);
        return eval.mae();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class RegressionTreeModelTest {

    @Test
    void testCreateTrainer() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createTrainer(5);
        assertNotNull(trainer);
    }

    @Test
    void testCreateCustomTrainer() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createCustomTrainer(8, 3);
        assertNotNull(trainer);
    }

    @Test
    void testTrain() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createTrainer(4);
        MutableDataset<Regressor> data = createTestDataset();

        Model<Regressor> model = RegressionTreeModel.train(trainer, data);
        assertNotNull(model);
    }

    private MutableDataset<Regressor> createTestDataset() {
        RegressionFactory factory = new RegressionFactory();
        MutableDataset<Regressor> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 30; i++) {
            double x = Math.random() * 10;
            double y = x * x + Math.random(); // y = x^2 + noise
            dataset.add(new ArrayExample<>(new Regressor("target", y),
                new String[]{"x"}, new double[]{x}));
        }

        return dataset;
    }

    @Test
    void testCreateTrainerWithDifferentDepth() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createTrainer(10);
        assertNotNull(trainer);
    }

    @Test
    void testCreateTrainerShallowTree() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createTrainer(2);
        assertNotNull(trainer);
    }

    @Test
    void testCustomTrainerDifferentParams() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createCustomTrainer(6, 10);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetSize() {
        MutableDataset<Regressor> data = createTestDataset();
        assertEquals(30, data.size());
    }

    @Test
    void testTrainReturnsModel() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createTrainer(3);
        MutableDataset<Regressor> data = createTestDataset();
        Model<Regressor> model = RegressionTreeModel.train(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testMultipleTrainersCanBeCreated() {
        CARTRegressionTrainer t1 = RegressionTreeModel.createTrainer(3);
        CARTRegressionTrainer t2 = RegressionTreeModel.createTrainer(5);
        assertNotNull(t1);
        assertNotNull(t2);
    }

    @Test
    void testCustomTrainerMinSamplesOne() {
        CARTRegressionTrainer trainer = RegressionTreeModel.createCustomTrainer(5, 1);
        assertNotNull(trainer);
    }
}`,

	hint1: 'CARTRegressionTrainer works like CARTClassificationTrainer but for regression',
	hint2: 'Trees can capture non-linear relationships without feature engineering',

	whyItMatters: `Regression trees handle non-linear patterns:

- **Non-linear**: Captures complex relationships
- **Interpretable**: Easy to visualize decision rules
- **No scaling needed**: Works with raw features
- **Robust**: Handles outliers well

Trees are building blocks for gradient boosting methods.`,

	translations: {
		ru: {
			title: 'Дерево регрессии',
			description: `# Дерево регрессии

Создайте деревья решений для задач регрессии с Tribuo.

## Задача

Реализуйте деревья регрессии:
- Настройте параметры дерева
- Обучите на непрерывной цели
- Оцените предсказания

## Пример

\`\`\`java
CARTRegressionTrainer trainer = new CARTRegressionTrainer(
    6,    // max depth
    5,    // min samples in leaf
    0.0f  // min impurity decrease
);
\`\`\``,
			hint1: 'CARTRegressionTrainer работает как CARTClassificationTrainer но для регрессии',
			hint2: 'Деревья могут захватывать нелинейные отношения без feature engineering',
			whyItMatters: `Деревья регрессии обрабатывают нелинейные паттерны:

- **Нелинейность**: Захватывает сложные отношения
- **Интерпретируемость**: Легко визуализировать правила решений
- **Без масштабирования**: Работает с сырыми признаками
- **Устойчивость**: Хорошо обрабатывает выбросы`,
		},
		uz: {
			title: 'Regressiya daraxti',
			description: `# Regressiya daraxti

Tribuo bilan regressiya topshiriqlari uchun qaror daraxtlarini yarating.

## Topshiriq

Regressiya daraxtlarini amalga oshiring:
- Daraxt parametrlarini sozlang
- Uzluksiz maqsadda o'rgating
- Bashoratlarni baholang

## Misol

\`\`\`java
CARTRegressionTrainer trainer = new CARTRegressionTrainer(
    6,    // max depth
    5,    // min samples in leaf
    0.0f  // min impurity decrease
);
\`\`\``,
			hint1: "CARTRegressionTrainer CARTClassificationTrainer kabi ishlaydi lekin regressiya uchun",
			hint2: "Daraxtlar feature engineering siz nolinear munosabatlarni ushlaydi",
			whyItMatters: `Regressiya daraxtlari nolinear patternlarni boshqaradi:

- **Nolinear**: Murakkab munosabatlarni ushlaydi
- **Interpretatsiya qilinadigan**: Qaror qoidalarini vizualizatsiya qilish oson
- **Masshtablash kerak emas**: Xom xususiyatlar bilan ishlaydi
- **Barqaror**: Chiqindilarni yaxshi boshqaradi`,
		},
	},
};

export default task;
