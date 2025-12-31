import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-random-forest',
	title: 'Random Forest Classifier',
	difficulty: 'medium',
	tags: ['tribuo', 'random-forest', 'ensemble'],
	estimatedTime: '25m',
	isPremium: false,
	order: 3,
	description: `# Random Forest Classifier

Build powerful ensemble classifiers with Random Forest.

## Task

Implement Random Forest:
- Configure number of trees
- Set feature sampling parameters
- Aggregate predictions from ensemble

## Example

\`\`\`java
RandomForestTrainer<Label> trainer = new RandomForestTrainer<>(
    new CARTClassificationTrainer(8),  // base tree trainer
    new VotingCombiner(),  // combining strategy
    100  // number of trees
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.ensemble.BaggingTrainer;

public class RandomForestClassifier {

    /**
     * @param numTrees Number of trees in forest
     * @param maxDepth Maximum depth of each tree
     */
    public static BaggingTrainer<Label> createRandomForest(
            int numTrees, int maxDepth) {
        return null;
    }

    /**
     */
    public static Model<Label> trainForest(
            BaggingTrainer<Label> trainer, Dataset<Label> data) {
        return null;
    }

    /**
     */
    public static BaggingTrainer<Label> createCustomRandomForest(
            int numTrees, int maxDepth, int minSamplesLeaf, float subsampleRatio) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.ensemble.BaggingTrainer;

public class RandomForestClassifier {

    /**
     * Create Random Forest trainer.
     * @param numTrees Number of trees in forest
     * @param maxDepth Maximum depth of each tree
     */
    public static BaggingTrainer<Label> createRandomForest(
            int numTrees, int maxDepth) {
        CARTClassificationTrainer baseTrainer = new CARTClassificationTrainer(maxDepth);
        VotingCombiner combiner = new VotingCombiner();

        return new BaggingTrainer<>(
            baseTrainer,
            combiner,
            numTrees
        );
    }

    /**
     * Train Random Forest model.
     */
    public static Model<Label> trainForest(
            BaggingTrainer<Label> trainer, Dataset<Label> data) {
        return trainer.train(data);
    }

    /**
     * Create Random Forest with custom parameters.
     */
    public static BaggingTrainer<Label> createCustomRandomForest(
            int numTrees, int maxDepth, int minSamplesLeaf, float subsampleRatio) {
        CARTClassificationTrainer baseTrainer = new CARTClassificationTrainer(
            maxDepth,
            minSamplesLeaf,
            0.0f,  // min impurity decrease
            0.7f,  // fraction of features (for randomness)
            true,  // use random split
            Trainer.DEFAULT_SEED
        );

        VotingCombiner combiner = new VotingCombiner();

        return new BaggingTrainer<>(
            baseTrainer,
            combiner,
            numTrees,
            subsampleRatio  // subsample ratio for bagging
        );
    }

    /**
     * Get out-of-bag error estimate.
     */
    public static double getOOBError(BaggingTrainer<Label> trainer,
                                      Dataset<Label> data) {
        // Train with OOB estimation enabled
        Model<Label> model = trainer.train(data);
        // OOB error would be computed during training
        return 0.0; // Placeholder - actual implementation depends on Tribuo version
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.impl.ArrayExample;
import static org.junit.jupiter.api.Assertions.*;

public class RandomForestClassifierTest {

    @Test
    void testCreateRandomForest() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(10, 5);
        assertNotNull(trainer);
    }

    @Test
    void testTrainForest() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(5, 3);
        MutableDataset<Label> data = createTestDataset();

        Model<Label> model = RandomForestClassifier.trainForest(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testCreateCustomRandomForest() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createCustomRandomForest(
            50, 10, 2, 0.8f);
        assertNotNull(trainer);
    }

    private MutableDataset<Label> createTestDataset() {
        LabelFactory factory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 20; i++) {
            double[] features = {Math.random(), Math.random()};
            String label = features[0] > 0.5 ? "A" : "B";
            dataset.add(new ArrayExample<>(new Label(label),
                new String[]{"f1", "f2"}, features));
        }

        return dataset;
    }

    @Test
    void testCreateRandomForestWithMoreTrees() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(100, 8);
        assertNotNull(trainer);
    }

    @Test
    void testCreateRandomForestWithShallowTrees() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(20, 2);
        assertNotNull(trainer);
    }

    @Test
    void testCustomForestSubsampleRatio() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createCustomRandomForest(30, 6, 5, 0.5f);
        assertNotNull(trainer);
    }

    @Test
    void testCustomForestHighSubsample() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createCustomRandomForest(25, 8, 3, 1.0f);
        assertNotNull(trainer);
    }

    @Test
    void testDatasetNotEmpty() {
        MutableDataset<Label> data = createTestDataset();
        assertTrue(data.size() > 0);
    }

    @Test
    void testTrainForestWithLargeDataset() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(10, 5);
        MutableDataset<Label> data = createTestDataset();
        Model<Label> model = RandomForestClassifier.trainForest(trainer, data);
        assertNotNull(model);
    }

    @Test
    void testCreateRandomForestSingleTree() {
        BaggingTrainer<Label> trainer = RandomForestClassifier.createRandomForest(1, 3);
        assertNotNull(trainer);
    }
}`,

	hint1: 'Use BaggingTrainer with CARTClassificationTrainer as base',
	hint2: 'VotingCombiner aggregates predictions by majority vote',

	whyItMatters: `Random Forest is one of the most effective ML algorithms:

- **Robust**: Resistant to overfitting through averaging
- **Feature importance**: Aggregate importance from all trees
- **Parallelizable**: Trees can be trained independently
- **Versatile**: Works well on many problem types

Random Forest is often the first algorithm to try on new problems.`,

	translations: {
		ru: {
			title: 'Классификатор Random Forest',
			description: `# Классификатор Random Forest

Создайте мощные ансамблевые классификаторы с Random Forest.

## Задача

Реализуйте Random Forest:
- Настройте количество деревьев
- Установите параметры сэмплирования признаков
- Агрегируйте предсказания ансамбля

## Пример

\`\`\`java
RandomForestTrainer<Label> trainer = new RandomForestTrainer<>(
    new CARTClassificationTrainer(8),  // base tree trainer
    new VotingCombiner(),  // combining strategy
    100  // number of trees
);
\`\`\``,
			hint1: 'Используйте BaggingTrainer с CARTClassificationTrainer как базовый',
			hint2: 'VotingCombiner агрегирует предсказания большинством голосов',
			whyItMatters: `Random Forest - один из самых эффективных ML алгоритмов:

- **Устойчивость**: Защита от переобучения через усреднение
- **Важность признаков**: Агрегированная важность от всех деревьев
- **Параллелизация**: Деревья могут обучаться независимо
- **Универсальность**: Хорошо работает на многих типах задач`,
		},
		uz: {
			title: 'Random Forest klassifikatori',
			description: `# Random Forest klassifikatori

Random Forest bilan kuchli ansambl klassifikatorlarni yarating.

## Topshiriq

Random Forest ni amalga oshiring:
- Daraxtlar sonini sozlang
- Xususiyat namunalash parametrlarini o'rnating
- Ansambldan bashoratlarni yig'ing

## Misol

\`\`\`java
RandomForestTrainer<Label> trainer = new RandomForestTrainer<>(
    new CARTClassificationTrainer(8),  // base tree trainer
    new VotingCombiner(),  // combining strategy
    100  // number of trees
);
\`\`\``,
			hint1: "Bazaviy sifatida CARTClassificationTrainer bilan BaggingTrainer dan foydalaning",
			hint2: "VotingCombiner bashoratlarni ko'pchilik ovozi bilan birlashtiradi",
			whyItMatters: `Random Forest eng samarali ML algoritmlaridan biri:

- **Barqaror**: O'rtachalash orqali overfittingga chidamli
- **Xususiyat muhimligi**: Barcha daraxtlardan agregatlangan muhimlik
- **Parallellashtiriladigan**: Daraxtlar mustaqil o'qitilishi mumkin
- **Ko'p qirrali**: Ko'p muammo turlarida yaxshi ishlaydi`,
		},
	},
};

export default task;
