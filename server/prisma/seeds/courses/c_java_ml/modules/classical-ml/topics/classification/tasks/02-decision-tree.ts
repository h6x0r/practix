import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-decision-tree',
	title: 'Decision Tree Classifier',
	difficulty: 'easy',
	tags: ['tribuo', 'decision-tree', 'classification'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Decision Tree Classifier

Build interpretable decision tree classifiers with Tribuo.

## Task

Implement decision trees:
- Configure tree depth and split criteria
- Train on classification data
- Visualize tree structure

## Example

\`\`\`java
CARTClassificationTrainer trainer = new CARTClassificationTrainer(
    6,  // max depth
    2,  // min samples leaf
    0.0f,  // min impurity decrease
    1.0f,  // fraction of features
    true,  // use random split
    new SplitDataSourceProvenance()
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;

public class DecisionTreeClassifier {

    /**
     * Create a decision tree trainer with specified depth.
     * @param maxDepth Maximum tree depth
     */
    public static CARTClassificationTrainer createTrainer(int maxDepth) {
        return null;
    }

    /**
     * Train decision tree model.
     */
    public static Model<Label> trainTree(
            CARTClassificationTrainer trainer, Dataset<Label> data) {
        return null;
    }

    /**
     * Create trainer with custom parameters.
     */
    public static CARTClassificationTrainer createCustomTrainer(
            int maxDepth, int minSamplesLeaf, float minImpurityDecrease) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.common.tree.TreeModel;

public class DecisionTreeClassifier {

    /**
     * Create a decision tree trainer with specified depth.
     * @param maxDepth Maximum tree depth
     */
    public static CARTClassificationTrainer createTrainer(int maxDepth) {
        return new CARTClassificationTrainer(maxDepth);
    }

    /**
     * Train decision tree model.
     */
    public static Model<Label> trainTree(
            CARTClassificationTrainer trainer, Dataset<Label> data) {
        return trainer.train(data);
    }

    /**
     * Create trainer with custom parameters.
     */
    public static CARTClassificationTrainer createCustomTrainer(
            int maxDepth, int minSamplesLeaf, float minImpurityDecrease) {
        return new CARTClassificationTrainer(
            maxDepth,
            minSamplesLeaf,
            minImpurityDecrease,
            1.0f,  // fraction of features to consider
            true,  // use random split points
            Trainer.DEFAULT_SEED
        );
    }

    /**
     * Get feature importances from tree model.
     */
    public static java.util.Map<String, Double> getFeatureImportances(
            TreeModel<Label> model) {
        return model.getFeatureImportances();
    }

    /**
     * Print tree structure.
     */
    public static String getTreeDescription(TreeModel<Label> model) {
        return model.toString();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class DecisionTreeClassifierTest {

    @Test
    void testCreateTrainer() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createTrainer(5);
        assertNotNull(trainer);
    }

    @Test
    void testTrainTree() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createTrainer(3);
        MutableDataset<Label> data = createTestDataset();

        Model<Label> model = DecisionTreeClassifier.trainTree(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testCreateCustomTrainer() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createCustomTrainer(
            10, 5, 0.01f);
        assertNotNull(trainer);
    }

    private MutableDataset<Label> createTestDataset() {
        LabelFactory factory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        dataset.add(new ArrayExample<>(new Label("A"),
            new String[]{"f1", "f2"}, new double[]{1.0, 0.0}));
        dataset.add(new ArrayExample<>(new Label("B"),
            new String[]{"f1", "f2"}, new double[]{0.0, 1.0}));
        dataset.add(new ArrayExample<>(new Label("A"),
            new String[]{"f1", "f2"}, new double[]{0.8, 0.2}));

        return dataset;
    }

    @Test
    void testCreateTrainerWithDepth10() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createTrainer(10);
        assertNotNull(trainer);
    }

    @Test
    void testCreateTrainerWithDepth1() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createTrainer(1);
        assertNotNull(trainer);
    }

    @Test
    void testCustomTrainerMinSamples() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createCustomTrainer(5, 10, 0.05f);
        assertNotNull(trainer);
    }

    @Test
    void testCustomTrainerZeroImpurity() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createCustomTrainer(8, 1, 0.0f);
        assertNotNull(trainer);
    }

    @Test
    void testTrainedModelNotNull() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createTrainer(4);
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = DecisionTreeClassifier.trainTree(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testDatasetSize() {
        MutableDataset<Label> data = createTestDataset();
        assertEquals(3, data.size());
    }

    @Test
    void testCustomTrainerDeepTree() {
        CARTClassificationTrainer trainer = DecisionTreeClassifier.createCustomTrainer(20, 2, 0.001f);
        assertNotNull(trainer);
    }
}`,

	hint1: 'CARTClassificationTrainer takes maxDepth as constructor parameter',
	hint2: 'TreeModel provides getFeatureImportances() for interpretability',

	whyItMatters: `Decision trees offer unique advantages:

- **Interpretability**: Easy to visualize and explain
- **No feature scaling**: Works with raw feature values
- **Feature importance**: Built-in importance scores
- **Non-linear**: Captures complex decision boundaries

Trees are the basis for powerful ensemble methods like Random Forest.`,

	translations: {
		ru: {
			title: 'Классификатор дерева решений',
			description: `# Классификатор дерева решений

Создайте интерпретируемые деревья решений с Tribuo.

## Задача

Реализуйте деревья решений:
- Настройте глубину дерева и критерии разбиения
- Обучите на данных классификации
- Визуализируйте структуру дерева

## Пример

\`\`\`java
CARTClassificationTrainer trainer = new CARTClassificationTrainer(
    6,  // max depth
    2,  // min samples leaf
    0.0f,  // min impurity decrease
    1.0f,  // fraction of features
    true,  // use random split
    new SplitDataSourceProvenance()
);
\`\`\``,
			hint1: 'CARTClassificationTrainer принимает maxDepth как параметр конструктора',
			hint2: 'TreeModel предоставляет getFeatureImportances() для интерпретируемости',
			whyItMatters: `Деревья решений имеют уникальные преимущества:

- **Интерпретируемость**: Легко визуализировать и объяснить
- **Без масштабирования признаков**: Работает с сырыми значениями
- **Важность признаков**: Встроенные оценки важности
- **Нелинейность**: Захватывает сложные границы решений`,
		},
		uz: {
			title: "Qaror daraxti klassifikatori",
			description: `# Qaror daraxti klassifikatori

Tribuo bilan interpretatsiya qilinadigan qaror daraxtlarini yarating.

## Topshiriq

Qaror daraxtlarini amalga oshiring:
- Daraxt chuqurligi va bo'linish mezonlarini sozlang
- Klassifikatsiya datalarida o'rgating
- Daraxt strukturasini vizualizatsiya qiling

## Misol

\`\`\`java
CARTClassificationTrainer trainer = new CARTClassificationTrainer(
    6,  // max depth
    2,  // min samples leaf
    0.0f,  // min impurity decrease
    1.0f,  // fraction of features
    true,  // use random split
    new SplitDataSourceProvenance()
);
\`\`\``,
			hint1: "CARTClassificationTrainer konstruktor parametri sifatida maxDepth ni qabul qiladi",
			hint2: "TreeModel interpretatsiya uchun getFeatureImportances() beradi",
			whyItMatters: `Qaror daraxtlari noyob afzalliklarga ega:

- **Interpretatsiya qilinadigan**: Vizualizatsiya va tushuntirish oson
- **Xususiyat masshtablashsiz**: Xom xususiyat qiymatlari bilan ishlaydi
- **Xususiyat muhimligi**: O'rnatilgan muhimlik ballari
- **Nolinear**: Murakkab qaror chegaralarini ushlaydi`,
		},
	},
};

export default task;
