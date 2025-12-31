import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-tribuo-serialization',
	title: 'Tribuo Model Serialization',
	difficulty: 'easy',
	tags: ['tribuo', 'serialization', 'protobuf'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Tribuo Model Serialization

Save and load Tribuo models using native serialization.

## Task

Implement Tribuo persistence:
- Save models to protobuf format
- Load models for production use
- Handle model provenance

## Example

\`\`\`java
// Save model
model.serializeToFile(Paths.get("model.tribuo"));

// Load model
Model<Label> loaded = Model.deserializeFromFile(
    Paths.get("model.tribuo")
);
\`\`\``,

	initialCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TribuoSerializer {

    /**
     * @param model Model to save
     * @param path File path
     */
    public static void saveModel(Model<?> model, String path) throws IOException {
    }

    /**
     * @param path File path
     */
    public static Model<Label> loadClassificationModel(String path) throws IOException {
        return null;
    }

    /**
     */
    public static String getProvenance(Model<?> model) {
        return null;
    }
}`,

	solutionCode: `import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.provenance.*;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TribuoSerializer {

    /**
     * Save Tribuo model to file.
     * @param model Model to save
     * @param path File path
     */
    public static void saveModel(Model<?> model, String path) throws IOException {
        model.serializeToFile(Paths.get(path));
    }

    /**
     * Load Tribuo classification model.
     * @param path File path
     */
    @SuppressWarnings("unchecked")
    public static Model<Label> loadClassificationModel(String path) throws IOException {
        return (Model<Label>) Model.deserializeFromFile(Paths.get(path));
    }

    /**
     * Get model provenance information.
     */
    public static String getProvenance(Model<?> model) {
        ModelProvenance provenance = model.getProvenance();
        StringBuilder sb = new StringBuilder();

        sb.append("Trainer: ").append(provenance.getTrainerProvenance().toString()).append("\\n");
        sb.append("Training time: ").append(provenance.getTrainingTime()).append("\\n");
        sb.append("Dataset: ").append(provenance.getDatasetProvenance().toString()).append("\\n");

        return sb.toString();
    }

    /**
     * Save model using Java serialization (alternative).
     */
    public static void saveModelJava(Model<?> model, String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(path))) {
            oos.writeObject(model);
        }
    }

    /**
     * Load model using Java serialization.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Output<T>> Model<T> loadModelJava(String path)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(path))) {
            return (Model<T>) ois.readObject();
        }
    }

    /**
     * Get training dataset size from provenance.
     */
    public static int getTrainingDataSize(Model<?> model) {
        return model.getProvenance().getDatasetProvenance().getNumExamples();
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.impl.ArrayExample;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

public class TribuoSerializerTest {

    @TempDir
    Path tempDir;

    @Test
    void testSaveAndLoadModel() throws Exception {
        Model<Label> model = createTestModel();
        String path = tempDir.resolve("model.tribuo").toString();

        TribuoSerializer.saveModel(model, path);
        Model<Label> loaded = TribuoSerializer.loadClassificationModel(path);

        assertNotNull(loaded);
    }

    @Test
    void testGetProvenance() {
        Model<Label> model = createTestModel();
        String provenance = TribuoSerializer.getProvenance(model);

        assertNotNull(provenance);
        assertTrue(provenance.contains("Trainer"));
    }

    @Test
    void testJavaSerialization() throws Exception {
        Model<Label> model = createTestModel();
        String path = tempDir.resolve("model.ser").toString();

        TribuoSerializer.saveModelJava(model, path);
        Model<Label> loaded = TribuoSerializer.loadModelJava(path);

        assertNotNull(loaded);
    }

    private Model<Label> createTestModel() {
        LabelFactory factory = new LabelFactory();
        MutableDataset<Label> dataset = new MutableDataset<>(
            new SimpleDataSourceProvenance("test", factory), factory);

        for (int i = 0; i < 20; i++) {
            dataset.add(new ArrayExample<>(new Label(i % 2 == 0 ? "A" : "B"),
                new String[]{"f1", "f2"}, new double[]{Math.random(), Math.random()}));
        }

        LogisticRegressionTrainer trainer = new LogisticRegressionTrainer();
        return trainer.train(dataset);
    }

    @Test
    void testModelNotNull() {
        Model<Label> model = createTestModel();
        assertNotNull(model);
    }

    @Test
    void testProvenanceContainsDataset() {
        Model<Label> model = createTestModel();
        String provenance = TribuoSerializer.getProvenance(model);
        assertTrue(provenance.contains("Dataset"));
    }

    @Test
    void testTrainingDataSize() {
        Model<Label> model = createTestModel();
        int size = TribuoSerializer.getTrainingDataSize(model);
        assertEquals(20, size);
    }

    @Test
    void testSaveCreatesFile() throws Exception {
        Model<Label> model = createTestModel();
        String path = tempDir.resolve("test_model.tribuo").toString();
        TribuoSerializer.saveModel(model, path);
        assertTrue(new java.io.File(path).exists());
    }

    @Test
    void testLoadedModelHasProvenance() throws Exception {
        Model<Label> model = createTestModel();
        String path = tempDir.resolve("prov_model.tribuo").toString();
        TribuoSerializer.saveModel(model, path);
        Model<Label> loaded = TribuoSerializer.loadClassificationModel(path);
        assertNotNull(loaded.getProvenance());
    }

    @Test
    void testJavaSerializationCreatesFile() throws Exception {
        Model<Label> model = createTestModel();
        String path = tempDir.resolve("java_model.ser").toString();
        TribuoSerializer.saveModelJava(model, path);
        assertTrue(new java.io.File(path).exists());
    }

    @Test
    void testProvenanceContainsTime() {
        Model<Label> model = createTestModel();
        String provenance = TribuoSerializer.getProvenance(model);
        assertTrue(provenance.contains("Training time"));
    }
}`,

	hint1: 'model.serializeToFile() saves to protobuf format',
	hint2: 'Model.deserializeFromFile() loads the model back',

	whyItMatters: `Tribuo serialization provides reproducibility:

- **Provenance tracking**: Full training history preserved
- **Format options**: Protobuf or Java serialization
- **Type safety**: Maintains output type information
- **Auditability**: Know exactly how model was trained

Proper serialization is key to ML reproducibility.`,

	translations: {
		ru: {
			title: 'Сериализация моделей Tribuo',
			description: `# Сериализация моделей Tribuo

Сохраняйте и загружайте модели Tribuo с использованием нативной сериализации.

## Задача

Реализуйте персистентность Tribuo:
- Сохраняйте модели в формате protobuf
- Загружайте модели для production использования
- Обрабатывайте provenance моделей

## Пример

\`\`\`java
// Save model
model.serializeToFile(Paths.get("model.tribuo"));

// Load model
Model<Label> loaded = Model.deserializeFromFile(
    Paths.get("model.tribuo")
);
\`\`\``,
			hint1: 'model.serializeToFile() сохраняет в формате protobuf',
			hint2: 'Model.deserializeFromFile() загружает модель обратно',
			whyItMatters: `Сериализация Tribuo обеспечивает воспроизводимость:

- **Отслеживание provenance**: Полная история обучения сохранена
- **Варианты форматов**: Protobuf или Java сериализация
- **Типобезопасность**: Сохраняет информацию о типе выхода
- **Аудируемость**: Точно знаете как модель была обучена`,
		},
		uz: {
			title: 'Tribuo modellarini serializatsiya qilish',
			description: `# Tribuo modellarini serializatsiya qilish

Nativ serializatsiya yordamida Tribuo modellarini saqlang va yuklang.

## Topshiriq

Tribuo persistence ni amalga oshiring:
- Modellarni protobuf formatida saqlang
- Production foydalanish uchun modellarni yuklang
- Model provenance ni boshqaring

## Misol

\`\`\`java
// Save model
model.serializeToFile(Paths.get("model.tribuo"));

// Load model
Model<Label> loaded = Model.deserializeFromFile(
    Paths.get("model.tribuo")
);
\`\`\``,
			hint1: "model.serializeToFile() protobuf formatiga saqlaydi",
			hint2: "Model.deserializeFromFile() modelni qayta yuklaydi",
			whyItMatters: `Tribuo serializatsiyasi takroriylikni ta'minlaydi:

- **Provenance kuzatuvi**: To'liq o'qitish tarixi saqlanadi
- **Format variantlari**: Protobuf yoki Java serializatsiya
- **Tip xavfsizligi**: Chiqish turi ma'lumotlarini saqlaydi
- **Auditlanish**: Model qanday o'qitilganini aniq bilasiz`,
		},
	},
};

export default task;
