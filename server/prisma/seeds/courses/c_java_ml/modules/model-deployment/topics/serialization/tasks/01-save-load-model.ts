import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-save-load-model',
	title: 'Save and Load Models',
	difficulty: 'easy',
	tags: ['dl4j', 'serialization', 'deployment'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Save and Load Models

Learn to persist trained models for later use.

## Task

Implement model persistence:
- Save trained DL4J model to file
- Load model for inference
- Handle model versioning

## Example

\`\`\`java
// Save model
ModelSerializer.writeModel(model, new File("model.zip"), true);

// Load model
MultiLayerNetwork loaded = ModelSerializer.restoreMultiLayerNetwork(
    new File("model.zip")
);
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;
import java.io.IOException;

public class ModelPersistence {

    /**
     * @param model Trained model
     * @param path File path to save
     */
    public static void saveModel(MultiLayerNetwork model, String path,
                                  boolean saveUpdater) throws IOException {
    }

    /**
     * @param path File path to load from
     */
    public static MultiLayerNetwork loadModel(String path) throws IOException {
        return null;
    }

    /**
     */
    public static void saveModelWithMetadata(MultiLayerNetwork model,
                                              String path, String version) throws IOException {
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelPersistence {

    /**
     * Save model to file.
     * @param model Trained model
     * @param path File path to save
     * @param saveUpdater Whether to save optimizer state
     */
    public static void saveModel(MultiLayerNetwork model, String path,
                                  boolean saveUpdater) throws IOException {
        File file = new File(path);
        // Ensure parent directory exists
        file.getParentFile().mkdirs();
        ModelSerializer.writeModel(model, file, saveUpdater);
    }

    /**
     * Load model from file.
     * @param path File path to load from
     */
    public static MultiLayerNetwork loadModel(String path) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(path));
    }

    /**
     * Save model with metadata.
     */
    public static void saveModelWithMetadata(MultiLayerNetwork model,
                                              String path, String version) throws IOException {
        // Save model
        saveModel(model, path, true);

        // Save metadata alongside
        String metadataPath = path.replace(".zip", "_metadata.json");
        String metadata = String.format(
            "{\"version\":\"%s\",\"timestamp\":%d,\"numParams\":%d}",
            version, System.currentTimeMillis(), model.numParams()
        );
        Files.writeString(Paths.get(metadataPath), metadata);
    }

    /**
     * Check if model file exists.
     */
    public static boolean modelExists(String path) {
        return new File(path).exists();
    }

    /**
     * Get model file size in bytes.
     */
    public static long getModelSize(String path) {
        return new File(path).length();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

public class ModelPersistenceTest {

    @TempDir
    Path tempDir;

    @Test
    void testSaveAndLoadModel() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("model.zip").toString();

        ModelPersistence.saveModel(model, path, true);
        assertTrue(ModelPersistence.modelExists(path));

        MultiLayerNetwork loaded = ModelPersistence.loadModel(path);
        assertNotNull(loaded);
        assertEquals(model.numParams(), loaded.numParams());
    }

    @Test
    void testSaveModelWithMetadata() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("model_v1.zip").toString();

        ModelPersistence.saveModelWithMetadata(model, path, "1.0.0");

        assertTrue(new File(path).exists());
        assertTrue(new File(path.replace(".zip", "_metadata.json")).exists());
    }

    @Test
    void testGetModelSize() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("model.zip").toString();

        ModelPersistence.saveModel(model, path, false);
        long size = ModelPersistence.getModelSize(path);

        assertTrue(size > 0);
    }

    private MultiLayerNetwork createTestModel() {
        var conf = new NeuralNetConfiguration.Builder()
            .list()
            .layer(new DenseLayer.Builder().nIn(10).nOut(5).build())
            .layer(new OutputLayer.Builder().nIn(5).nOut(2)
                .lossFunction(LossFunctions.LossFunction.MSE).build())
            .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    @Test
    void testModelExists() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("test_model.zip").toString();
        ModelPersistence.saveModel(model, path, false);
        assertTrue(ModelPersistence.modelExists(path));
    }

    @Test
    void testModelNotExists() {
        assertFalse(ModelPersistence.modelExists("/nonexistent/path/model.zip"));
    }

    @Test
    void testLoadedModelNotNull() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("loaded_model.zip").toString();
        ModelPersistence.saveModel(model, path, true);
        MultiLayerNetwork loaded = ModelPersistence.loadModel(path);
        assertNotNull(loaded);
    }

    @Test
    void testSaveWithoutUpdater() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("no_updater.zip").toString();
        ModelPersistence.saveModel(model, path, false);
        assertTrue(new File(path).exists());
    }

    @Test
    void testModelSizePositive() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("size_test.zip").toString();
        ModelPersistence.saveModel(model, path, true);
        assertTrue(ModelPersistence.getModelSize(path) > 0);
    }

    @Test
    void testLoadedModelParams() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("params_test.zip").toString();
        ModelPersistence.saveModel(model, path, true);
        MultiLayerNetwork loaded = ModelPersistence.loadModel(path);
        assertEquals(model.numParams(), loaded.numParams());
    }

    @Test
    void testMetadataFileCreated() throws Exception {
        MultiLayerNetwork model = createTestModel();
        String path = tempDir.resolve("meta_model.zip").toString();
        ModelPersistence.saveModelWithMetadata(model, path, "2.0.0");
        String metaPath = path.replace(".zip", "_metadata.json");
        assertTrue(new File(metaPath).exists());
    }
}`,

	hint1: 'ModelSerializer.writeModel() saves model to file',
	hint2: 'Set saveUpdater to true if you plan to continue training',

	whyItMatters: `Model persistence is essential for production:

- **Reproducibility**: Load exact model for predictions
- **Deployment**: Move models between environments
- **Versioning**: Track model iterations
- **Efficiency**: Train once, deploy many times

Proper serialization enables reliable ML systems.`,

	translations: {
		ru: {
			title: 'Сохранение и загрузка моделей',
			description: `# Сохранение и загрузка моделей

Научитесь сохранять обученные модели для последующего использования.

## Задача

Реализуйте персистентность моделей:
- Сохраните обученную DL4J модель в файл
- Загрузите модель для инференса
- Обработайте версионирование моделей

## Пример

\`\`\`java
// Save model
ModelSerializer.writeModel(model, new File("model.zip"), true);

// Load model
MultiLayerNetwork loaded = ModelSerializer.restoreMultiLayerNetwork(
    new File("model.zip")
);
\`\`\``,
			hint1: 'ModelSerializer.writeModel() сохраняет модель в файл',
			hint2: 'Установите saveUpdater в true если планируете продолжить обучение',
			whyItMatters: `Персистентность моделей критична для production:

- **Воспроизводимость**: Загрузка точной модели для предсказаний
- **Развертывание**: Перемещение моделей между окружениями
- **Версионирование**: Отслеживание итераций модели
- **Эффективность**: Обучить один раз, развернуть много раз`,
		},
		uz: {
			title: 'Modellarni saqlash va yuklash',
			description: `# Modellarni saqlash va yuklash

O'qitilgan modellarni keyingi foydalanish uchun saqlashni o'rganing.

## Topshiriq

Model persistence ni amalga oshiring:
- O'qitilgan DL4J modelni faylga saqlang
- Inference uchun modelni yuklang
- Model versiyalashni boshqaring

## Misol

\`\`\`java
// Save model
ModelSerializer.writeModel(model, new File("model.zip"), true);

// Load model
MultiLayerNetwork loaded = ModelSerializer.restoreMultiLayerNetwork(
    new File("model.zip")
);
\`\`\``,
			hint1: "ModelSerializer.writeModel() modelni faylga saqlaydi",
			hint2: "O'qitishni davom ettirishni rejalashtirmoqchi bo'lsangiz saveUpdater ni true qiling",
			whyItMatters: `Model persistence production uchun zarur:

- **Takroriylik**: Bashoratlar uchun aniq modelni yuklash
- **Joylashtirish**: Modellarni muhitlar orasida ko'chirish
- **Versiyalash**: Model iteratsiyalarini kuzatish
- **Samaradorlik**: Bir marta o'rgating, ko'p marta joylashtiring`,
		},
	},
};

export default task;
