import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'jml-model-versioning',
	title: 'Model Versioning',
	difficulty: 'medium',
	tags: ['versioning', 'deployment', 'mlops'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# Model Versioning

Implement model versioning for A/B testing and rollbacks.

## Task

Build versioning system:
- Load multiple model versions
- Route requests to specific versions
- Support canary deployments

## Example

\`\`\`java
@GetMapping("/v1/predict")
public Response predictV1(@RequestBody Request req) {
    return modelRegistry.getModel("v1").predict(req);
}

@GetMapping("/v2/predict")
public Response predictV2(@RequestBody Request req) {
    return modelRegistry.getModel("v2").predict(req);
}
\`\`\``,

	initialCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;

public class ModelRegistry {

    private Map<String, MultiLayerNetwork> models;
    private String defaultVersion;

    public ModelRegistry() {
        this.models = new ConcurrentHashMap<>();
    }

    /**
     */
    public void registerModel(String version, MultiLayerNetwork model) {
    }

    /**
     */
    public MultiLayerNetwork getModel(String version) {
        return null;
    }

    /**
     */
    public void setDefaultVersion(String version) {
    }

    /**
     * @param canaryPercentage Percentage of traffic to canary version
     */
    public MultiLayerNetwork getModelWithCanary(String stableVersion,
                                                  String canaryVersion,
                                                  double canaryPercentage) {
        return null;
    }
}`,

	solutionCode: `import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Random;
import java.io.File;

public class ModelRegistry {

    private Map<String, MultiLayerNetwork> models;
    private Map<String, Long> loadTimes;
    private String defaultVersion;
    private Random random;

    public ModelRegistry() {
        this.models = new ConcurrentHashMap<>();
        this.loadTimes = new ConcurrentHashMap<>();
        this.random = new Random();
    }

    /**
     * Register a model version.
     */
    public void registerModel(String version, MultiLayerNetwork model) {
        models.put(version, model);
        loadTimes.put(version, System.currentTimeMillis());

        // Set as default if first model
        if (defaultVersion == null) {
            defaultVersion = version;
        }
    }

    /**
     * Load and register model from file.
     */
    public void loadModel(String version, String path) throws Exception {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(
            new File(path)
        );
        registerModel(version, model);
    }

    /**
     * Get model by version.
     */
    public MultiLayerNetwork getModel(String version) {
        MultiLayerNetwork model = models.get(version);
        if (model == null) {
            throw new IllegalArgumentException("Model version not found: " + version);
        }
        return model;
    }

    /**
     * Get default model.
     */
    public MultiLayerNetwork getDefaultModel() {
        return getModel(defaultVersion);
    }

    /**
     * Set default model version.
     */
    public void setDefaultVersion(String version) {
        if (!models.containsKey(version)) {
            throw new IllegalArgumentException("Cannot set default: version not found");
        }
        this.defaultVersion = version;
    }

    /**
     * Get model with canary routing.
     * @param canaryPercentage Percentage of traffic to canary version (0-100)
     */
    public MultiLayerNetwork getModelWithCanary(String stableVersion,
                                                  String canaryVersion,
                                                  double canaryPercentage) {
        double roll = random.nextDouble() * 100;
        if (roll < canaryPercentage) {
            return getModel(canaryVersion);
        }
        return getModel(stableVersion);
    }

    /**
     * Get all registered versions.
     */
    public Set<String> getVersions() {
        return models.keySet();
    }

    /**
     * Unload a model version.
     */
    public void unloadModel(String version) {
        if (version.equals(defaultVersion)) {
            throw new IllegalStateException("Cannot unload default version");
        }
        models.remove(version);
        loadTimes.remove(version);
    }

    /**
     * Get model info.
     */
    public String getModelInfo(String version) {
        MultiLayerNetwork model = getModel(version);
        return String.format(
            "Version: %s, Params: %d, Loaded: %d",
            version, model.numParams(), loadTimes.get(version)
        );
    }
}`,

	testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import static org.junit.jupiter.api.Assertions.*;

public class ModelRegistryTest {

    private ModelRegistry registry;

    @BeforeEach
    void setUp() {
        registry = new ModelRegistry();
    }

    @Test
    void testRegisterAndGetModel() {
        MultiLayerNetwork model = createTestModel();
        registry.registerModel("v1.0", model);

        MultiLayerNetwork retrieved = registry.getModel("v1.0");
        assertNotNull(retrieved);
        assertEquals(model.numParams(), retrieved.numParams());
    }

    @Test
    void testDefaultVersion() {
        MultiLayerNetwork model1 = createTestModel();
        registry.registerModel("v1.0", model1);

        // First registered should be default
        assertEquals(model1.numParams(), registry.getDefaultModel().numParams());
    }

    @Test
    void testCanaryRouting() {
        MultiLayerNetwork stable = createTestModel();
        MultiLayerNetwork canary = createTestModel();

        registry.registerModel("stable", stable);
        registry.registerModel("canary", canary);

        // With 50% canary, should get mix of both
        int canaryCount = 0;
        for (int i = 0; i < 100; i++) {
            MultiLayerNetwork model = registry.getModelWithCanary("stable", "canary", 50);
            if (model == canary) canaryCount++;
        }

        // Should be roughly 50% (allowing variance)
        assertTrue(canaryCount > 20 && canaryCount < 80);
    }

    @Test
    void testUnknownVersion() {
        assertThrows(IllegalArgumentException.class, () -> {
            registry.getModel("unknown");
        });
    }

    private MultiLayerNetwork createTestModel() {
        var conf = new NeuralNetConfiguration.Builder()
            .list()
            .layer(new DenseLayer.Builder().nIn(10).nOut(5).build())
            .layer(new OutputLayer.Builder().nIn(5).nOut(2).build())
            .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    @Test
    void testSetDefaultVersion() {
        MultiLayerNetwork model1 = createTestModel();
        MultiLayerNetwork model2 = createTestModel();
        registry.registerModel("v1.0", model1);
        registry.registerModel("v2.0", model2);

        registry.setDefaultVersion("v2.0");
        assertEquals(model2.numParams(), registry.getDefaultModel().numParams());
    }

    @Test
    void testGetVersions() {
        registry.registerModel("v1.0", createTestModel());
        registry.registerModel("v2.0", createTestModel());

        assertEquals(2, registry.getVersions().size());
    }

    @Test
    void testUnloadModel() {
        registry.registerModel("v1.0", createTestModel());
        registry.registerModel("v2.0", createTestModel());

        registry.unloadModel("v2.0");
        assertEquals(1, registry.getVersions().size());
    }

    @Test
    void testUnloadDefaultThrows() {
        registry.registerModel("v1.0", createTestModel());

        assertThrows(IllegalStateException.class, () -> {
            registry.unloadModel("v1.0");
        });
    }

    @Test
    void testGetModelInfo() {
        registry.registerModel("v1.0", createTestModel());
        String info = registry.getModelInfo("v1.0");
        assertTrue(info.contains("v1.0"));
    }

    @Test
    void testSetInvalidDefaultThrows() {
        assertThrows(IllegalArgumentException.class, () -> {
            registry.setDefaultVersion("nonexistent");
        });
    }
}`,

	hint1: 'Use ConcurrentHashMap for thread-safe model storage',
	hint2: 'Random percentage check enables canary deployment',

	whyItMatters: `Model versioning enables safe deployments:

- **A/B testing**: Compare model performance in production
- **Rollback**: Quickly revert to previous version
- **Canary releases**: Gradually roll out new models
- **Audit trail**: Track which version served which predictions

Proper versioning is essential for production ML ops.`,

	translations: {
		ru: {
			title: 'Версионирование моделей',
			description: `# Версионирование моделей

Реализуйте версионирование моделей для A/B тестирования и откатов.

## Задача

Создайте систему версионирования:
- Загружайте несколько версий модели
- Направляйте запросы к конкретным версиям
- Поддержите canary развертывания

## Пример

\`\`\`java
@GetMapping("/v1/predict")
public Response predictV1(@RequestBody Request req) {
    return modelRegistry.getModel("v1").predict(req);
}

@GetMapping("/v2/predict")
public Response predictV2(@RequestBody Request req) {
    return modelRegistry.getModel("v2").predict(req);
}
\`\`\``,
			hint1: 'Используйте ConcurrentHashMap для потокобезопасного хранения моделей',
			hint2: 'Случайная проверка процента включает canary deployment',
			whyItMatters: `Версионирование моделей обеспечивает безопасные развертывания:

- **A/B тестирование**: Сравнение производительности моделей в production
- **Откат**: Быстрый возврат к предыдущей версии
- **Canary релизы**: Постепенный выпуск новых моделей
- **Аудит**: Отслеживание какая версия обслужила какие предсказания`,
		},
		uz: {
			title: 'Model versiyalash',
			description: `# Model versiyalash

A/B test va rollback uchun model versiyalashni amalga oshiring.

## Topshiriq

Versiyalash tizimini yarating:
- Bir nechta model versiyalarini yuklang
- So'rovlarni ma'lum versiyalarga yo'naltiring
- Canary joylashtirishlarni qo'llab-quvvatlang

## Misol

\`\`\`java
@GetMapping("/v1/predict")
public Response predictV1(@RequestBody Request req) {
    return modelRegistry.getModel("v1").predict(req);
}

@GetMapping("/v2/predict")
public Response predictV2(@RequestBody Request req) {
    return modelRegistry.getModel("v2").predict(req);
}
\`\`\``,
			hint1: "Thread-safe model saqlash uchun ConcurrentHashMap dan foydalaning",
			hint2: "Tasodifiy foiz tekshiruvi canary deploymentni yoqadi",
			whyItMatters: `Model versiyalash xavfsiz joylashtirishlarni ta'minlaydi:

- **A/B test**: Production da model samaradorligini solishtirish
- **Rollback**: Oldingi versiyaga tez qaytish
- **Canary relizlar**: Yangi modellarni asta-sekin chiqarish
- **Audit izi**: Qaysi versiya qaysi bashoratlarni xizmat qilganini kuzatish`,
		},
	},
};

export default task;
