import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-hot-reload',
	title: 'Hot Model Reload',
	difficulty: 'hard',
	tags: ['go', 'ml', 'deployment', 'hot-reload'],
	estimatedTime: '35m',
	isPremium: true,
	order: 2,
	description: `# Hot Model Reload

Implement hot model reloading without downtime.

## Task

Build a hot reload system that:
- Loads new model version without stopping
- Atomically swaps models
- Validates new model before swap
- Supports rollback on failure

## Example

\`\`\`go
manager := NewModelManager(modelPath)
manager.EnableHotReload(checkInterval)
// Automatically reloads when model file changes
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// Model interface
type Model interface {
	Predict(input []float32) []float32
	Version() string
}

// ModelManager manages model lifecycle
type ModelManager struct {
	// Your fields here
}

// NewModelManager creates a model manager
func NewModelManager(modelPath string, loader ModelLoader) *ModelManager {
	// Your code here
	return nil
}

// Reload triggers model reload
func (m *ModelManager) Reload() error {
	// Your code here
	return nil
}

// GetModel returns current model
func (m *ModelManager) GetModel() Model {
	// Your code here
	return nil
}

// ModelLoader loads models from path
type ModelLoader interface {
	Load(path string) (Model, error)
}

func main() {
	fmt.Println("Hot Model Reload")
}`,

	solutionCode: `package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Model interface
type Model interface {
	Predict(input []float32) []float32
	Version() string
	Validate() error
}

// MockModel implements Model for testing
type MockModel struct {
	version string
}

func NewMockModel(version string) *MockModel {
	return &MockModel{version: version}
}

func (m *MockModel) Predict(input []float32) []float32 {
	return []float32{0.8, 0.2}
}

func (m *MockModel) Version() string {
	return m.version
}

func (m *MockModel) Validate() error {
	return nil
}

// ModelLoader loads models from path
type ModelLoader interface {
	Load(path string) (Model, error)
}

// MockLoader implements ModelLoader
type MockLoader struct {
	loadCount int
}

func (l *MockLoader) Load(path string) (Model, error) {
	l.loadCount++
	return NewMockModel(fmt.Sprintf("v%d", l.loadCount)), nil
}

// ModelState holds model and metadata
type ModelState struct {
	Model     Model
	Version   string
	LoadedAt  time.Time
	FileHash  string
}

// ModelManager manages model lifecycle
type ModelManager struct {
	modelPath    string
	loader       ModelLoader
	current      atomic.Value // *ModelState
	previous     atomic.Value // *ModelState (for rollback)

	checkInterval time.Duration
	stopChan      chan struct{}
	running       bool
	mu            sync.Mutex

	onReload      []func(old, new Model)
	onError       []func(error)
}

// NewModelManager creates a model manager
func NewModelManager(modelPath string, loader ModelLoader) *ModelManager {
	mm := &ModelManager{
		modelPath:     modelPath,
		loader:        loader,
		checkInterval: 5 * time.Second,
		stopChan:      make(chan struct{}),
	}

	return mm
}

// Load performs initial model load
func (m *ModelManager) Load() error {
	model, err := m.loader.Load(m.modelPath)
	if err != nil {
		return fmt.Errorf("initial load failed: %w", err)
	}

	if err := model.Validate(); err != nil {
		return fmt.Errorf("model validation failed: %w", err)
	}

	state := &ModelState{
		Model:    model,
		Version:  model.Version(),
		LoadedAt: time.Now(),
		FileHash: m.getFileHash(),
	}

	m.current.Store(state)
	log.Printf("Model loaded: version=%s", state.Version)

	return nil
}

// Reload triggers model reload
func (m *ModelManager) Reload() error {
	newModel, err := m.loader.Load(m.modelPath)
	if err != nil {
		m.notifyError(err)
		return fmt.Errorf("reload failed: %w", err)
	}

	// Validate before swap
	if err := newModel.Validate(); err != nil {
		m.notifyError(err)
		return fmt.Errorf("validation failed: %w", err)
	}

	// Store previous for rollback
	current := m.current.Load()
	if current != nil {
		m.previous.Store(current)
	}

	newState := &ModelState{
		Model:    newModel,
		Version:  newModel.Version(),
		LoadedAt: time.Now(),
		FileHash: m.getFileHash(),
	}

	// Atomic swap
	m.current.Store(newState)

	// Notify callbacks
	var oldModel Model
	if current != nil {
		oldModel = current.(*ModelState).Model
	}
	m.notifyReload(oldModel, newModel)

	log.Printf("Model reloaded: version=%s", newState.Version)
	return nil
}

// Rollback reverts to previous model
func (m *ModelManager) Rollback() error {
	prev := m.previous.Load()
	if prev == nil {
		return fmt.Errorf("no previous model to rollback to")
	}

	current := m.current.Load()
	m.current.Store(prev)

	if current != nil {
		m.previous.Store(current)
	}

	state := prev.(*ModelState)
	log.Printf("Model rolled back to: version=%s", state.Version)
	return nil
}

// GetModel returns current model
func (m *ModelManager) GetModel() Model {
	state := m.current.Load()
	if state == nil {
		return nil
	}
	return state.(*ModelState).Model
}

// GetState returns current model state
func (m *ModelManager) GetState() *ModelState {
	state := m.current.Load()
	if state == nil {
		return nil
	}
	return state.(*ModelState)
}

// EnableHotReload enables automatic reload on file change
func (m *ModelManager) EnableHotReload(interval time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.running {
		return
	}

	m.checkInterval = interval
	m.running = true

	go m.watchLoop()
}

// DisableHotReload stops automatic reload
func (m *ModelManager) DisableHotReload() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.running {
		return
	}

	close(m.stopChan)
	m.running = false
	m.stopChan = make(chan struct{})
}

// watchLoop watches for file changes
func (m *ModelManager) watchLoop() {
	ticker := time.NewTicker(m.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if m.hasFileChanged() {
				log.Println("Model file changed, reloading...")
				if err := m.Reload(); err != nil {
					log.Printf("Hot reload failed: %v", err)
				}
			}
		case <-m.stopChan:
			return
		}
	}
}

// hasFileChanged checks if model file has changed
func (m *ModelManager) hasFileChanged() bool {
	current := m.current.Load()
	if current == nil {
		return true
	}

	currentHash := current.(*ModelState).FileHash
	newHash := m.getFileHash()

	return currentHash != newHash
}

// getFileHash returns SHA256 hash of model file
func (m *ModelManager) getFileHash() string {
	file, err := os.Open(m.modelPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return ""
	}

	return fmt.Sprintf("%x", hash.Sum(nil))
}

// OnReload registers a reload callback
func (m *ModelManager) OnReload(fn func(old, new Model)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onReload = append(m.onReload, fn)
}

// OnError registers an error callback
func (m *ModelManager) OnError(fn func(error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onError = append(m.onError, fn)
}

func (m *ModelManager) notifyReload(old, new Model) {
	m.mu.Lock()
	callbacks := m.onReload
	m.mu.Unlock()

	for _, fn := range callbacks {
		fn(old, new)
	}
}

func (m *ModelManager) notifyError(err error) {
	m.mu.Lock()
	callbacks := m.onError
	m.mu.Unlock()

	for _, fn := range callbacks {
		fn(err)
	}
}

func main() {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	// Register callbacks
	manager.OnReload(func(old, new Model) {
		oldVer := "none"
		if old != nil {
			oldVer = old.Version()
		}
		log.Printf("Model updated: %s -> %s", oldVer, new.Version())
	})

	manager.OnError(func(err error) {
		log.Printf("Model error: %v", err)
	})

	// Initial load
	if err := manager.Load(); err != nil {
		log.Fatal(err)
	}

	model := manager.GetModel()
	fmt.Printf("Current model: %s\\n", model.Version())

	// Simulate reload
	manager.Reload()
	model = manager.GetModel()
	fmt.Printf("After reload: %s\\n", model.Version())

	// Rollback
	manager.Rollback()
	model = manager.GetModel()
	fmt.Printf("After rollback: %s\\n", model.Version())
}`,

	testCode: `package main

import (
	"errors"
	"testing"
	"time"
)

func TestModelManager(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	err := manager.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	model := manager.GetModel()
	if model == nil {
		t.Fatal("Model should not be nil")
	}
}

func TestModelReload(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	manager.Load()
	v1 := manager.GetModel().Version()

	manager.Reload()
	v2 := manager.GetModel().Version()

	if v1 == v2 {
		t.Error("Version should change after reload")
	}
}

func TestModelRollback(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	manager.Load()
	v1 := manager.GetModel().Version()

	manager.Reload()
	v2 := manager.GetModel().Version()

	manager.Rollback()
	v3 := manager.GetModel().Version()

	if v2 == v3 {
		t.Error("Should rollback to previous version")
	}
	if v1 != v3 {
		t.Error("Should rollback to original version")
	}
}

func TestRollbackNoHistory(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	err := manager.Rollback()
	if err == nil {
		t.Error("Should error when no previous model")
	}
}

type FailingLoader struct{}

func (l *FailingLoader) Load(path string) (Model, error) {
	return nil, errors.New("load failed")
}

func TestReloadError(t *testing.T) {
	// First load succeeds
	mockLoader := &MockLoader{}
	manager := NewModelManager("/path/to/model", mockLoader)
	manager.Load()

	// Replace with failing loader
	manager.loader = &FailingLoader{}

	var errorReceived error
	manager.OnError(func(err error) {
		errorReceived = err
	})

	err := manager.Reload()
	if err == nil {
		t.Error("Reload should fail")
	}
	if errorReceived == nil {
		t.Error("Error callback should be called")
	}
}

func TestOnReloadCallback(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	var newVersion string
	manager.OnReload(func(old, new Model) {
		newVersion = new.Version()
	})

	manager.Load()
	manager.Reload()

	if newVersion == "" {
		t.Error("OnReload callback should be called")
	}
}

func TestGetState(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	state := manager.GetState()
	if state != nil {
		t.Error("State should be nil before load")
	}

	manager.Load()
	state = manager.GetState()

	if state == nil {
		t.Fatal("State should not be nil after load")
	}
	if state.Version == "" {
		t.Error("Version should be set")
	}
	if state.LoadedAt.IsZero() {
		t.Error("LoadedAt should be set")
	}
}

func TestEnableDisableHotReload(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	manager.EnableHotReload(100 * time.Millisecond)
	if !manager.running {
		t.Error("Should be running after enable")
	}

	manager.DisableHotReload()
	if manager.running {
		t.Error("Should not be running after disable")
	}
}

func TestNewModelManager(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/test/path", loader)

	if manager == nil {
		t.Fatal("Manager should not be nil")
	}
}

func TestOnErrorCallback(t *testing.T) {
	loader := &MockLoader{}
	manager := NewModelManager("/path/to/model", loader)

	var errorReceived bool
	manager.OnError(func(err error) {
		errorReceived = true
	})
	// Callback registered, should not panic
	if errorReceived {
		t.Error("Error should not be received yet")
	}
}`,

	hint1: 'Use atomic.Value for lock-free model swapping',
	hint2: 'Keep previous model for quick rollback',

	whyItMatters: `Hot reload enables zero-downtime model updates:

- **Continuous availability**: Update models without restarts
- **Fast iteration**: Deploy new models instantly
- **Safe updates**: Validate before swap, rollback on failure
- **A/B testing**: Switch between model versions

Hot reload is essential for ML systems that need frequent updates.`,

	translations: {
		ru: {
			title: 'Горячая перезагрузка моделей',
			description: `# Горячая перезагрузка моделей

Реализуйте горячую перезагрузку моделей без простоя.

## Задача

Создайте систему горячей перезагрузки:
- Загрузка новой версии модели без остановки
- Атомарная замена моделей
- Валидация новой модели перед заменой
- Поддержка отката при сбое

## Пример

\`\`\`go
manager := NewModelManager(modelPath)
manager.EnableHotReload(checkInterval)
// Automatically reloads when model file changes
\`\`\``,
			hint1: 'Используйте atomic.Value для безблокировочной замены моделей',
			hint2: 'Храните предыдущую модель для быстрого отката',
			whyItMatters: `Горячая перезагрузка обеспечивает обновления без простоя:

- **Непрерывная доступность**: Обновление моделей без рестартов
- **Быстрая итерация**: Мгновенный деплой новых моделей
- **Безопасные обновления**: Валидация перед заменой, откат при сбое
- **A/B тестирование**: Переключение между версиями моделей`,
		},
		uz: {
			title: 'Issiq model qayta yuklash',
			description: `# Issiq model qayta yuklash

Downtime siz issiq model qayta yuklashni amalga oshiring.

## Topshiriq

Issiq qayta yuklash tizimini yarating:
- To'xtamasdan yangi model versiyasini yuklash
- Modellarni atomik ravishda almashtirish
- Almashtirishdan oldin yangi modelni tekshirish
- Nosozlikda rollback ni qo'llab-quvvatlash

## Misol

\`\`\`go
manager := NewModelManager(modelPath)
manager.EnableHotReload(checkInterval)
// Automatically reloads when model file changes
\`\`\``,
			hint1: "Bloklashsiz model almashtirish uchun atomic.Value dan foydalaning",
			hint2: "Tez rollback uchun oldingi modelni saqlang",
			whyItMatters: `Issiq qayta yuklash downtime siz yangilanishlarni ta'minlaydi:

- **Doimiy mavjudlik**: Qayta ishga tushirmasdan modellarni yangilash
- **Tez iteratsiya**: Yangi modellarni bir zumda deploy qilish
- **Xavfsiz yangilanishlar**: Almashtirishdan oldin tekshirish, nosozlikda rollback
- **A/B testing**: Model versiyalari orasida almashish`,
		},
	},
};

export default task;
