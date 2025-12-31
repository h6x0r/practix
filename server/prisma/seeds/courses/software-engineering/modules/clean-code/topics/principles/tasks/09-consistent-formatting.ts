import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-consistent-formatting',
	title: 'Consistent Code Formatting',
	difficulty: 'easy',
	tags: ['go', 'clean-code', 'formatting', 'style'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply consistent formatting rules: proper indentation, spacing, line breaks, and Go conventions.

**You will fix:**

1. Inconsistent indentation
2. Missing blank lines between logical sections
3. Improper spacing around operators
4. Inconsistent brace placement

**Key Concepts:**
- **gofmt**: Go's official formatter
- **Vertical Spacing**: Separate logical sections
- **Horizontal Spacing**: Consistent operator spacing
- **Indentation**: Use tabs, not spaces (Go convention)

**Constraints:**
- Follow gofmt rules
- Add blank lines between logical sections
- Consistent spacing around operators`,
	initialCode: `package principles

import(
"fmt"
"strings"
)

type Config struct{
Host string
Port int
Timeout int}

func ProcessConfig(c *Config)error{
if c==nil{return fmt.Errorf("nil config")}
if c.Host==""{c.Host="localhost"}
if c.Port<=0{c.Port=8080}
if c.Timeout<=0{c.Timeout=30}
url:=fmt.Sprintf("http://%s:%d",c.Host,c.Port)
fmt.Printf("Connecting to: %s\n",url)
return nil}

func ValidateConfig(c *Config)bool{
if c==nil{return false}
if strings.TrimSpace(c.Host)==""{return false}
if c.Port<1||c.Port>65535{return false}
if c.Timeout<1||c.Timeout>300{return false}
return true}`,
	solutionCode: `package principles

import (
	"fmt"
	"strings"
)

type Config struct {
	Host    string
	Port    int
	Timeout int
}

func ProcessConfig(c *Config) error {
	if c == nil {
		return fmt.Errorf("nil config")
	}

	// Apply default values
	if c.Host == "" {
		c.Host = "localhost"
	}
	if c.Port <= 0 {
		c.Port = 8080
	}
	if c.Timeout <= 0 {
		c.Timeout = 30
	}

	// Build and display connection URL
	url := fmt.Sprintf("http://%s:%d", c.Host, c.Port)
	fmt.Printf("Connecting to: %s\n", url)

	return nil
}

func ValidateConfig(c *Config) bool {
	if c == nil {
		return false
	}

	if strings.TrimSpace(c.Host) == "" {
		return false
	}

	if c.Port < 1 || c.Port > 65535 {
		return false
	}

	if c.Timeout < 1 || c.Timeout > 300 {
		return false
	}

	return true
}`,
	hint1: `Add spaces around operators (==, <=, :=). Put opening braces on same line as declaration. Add blank lines between logical sections.`,
	hint2: `Format import statement with parentheses and proper spacing. Add newlines in struct definition. Separate function sections with blank lines.`,
	whyItMatters: `Consistent formatting improves readability and reduces cognitive load.

**Why Formatting Matters:**

**gofmt is your friend:**
- Eliminates formatting debates
- Ensures consistency across team
- Run automatically with editor

**Vertical Spacing:**
\`\`\`go
// BAD: No separation
func Process() {
    doValidation()
    doTransformation()
    doSaving()
}

// GOOD: Logical sections
func Process() {
    // Validation
    doValidation()

    // Transformation
    doTransformation()

    // Persistence
    doSaving()
}
\`\`\`

**Go Conventions:**
- Opening brace on same line
- Tabs for indentation
- Spaces around operators
- One blank line between functions`,
	order: 8,
	testCode: `package principles

import (
	"testing"
)

// Test1: ProcessConfig with valid config
func Test1(t *testing.T) {
	config := &Config{Host: "api.example.com", Port: 3000, Timeout: 60}
	err := ProcessConfig(config)
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
}

// Test2: ProcessConfig with nil config
func Test2(t *testing.T) {
	err := ProcessConfig(nil)
	if err == nil {
		t.Error("expected error for nil config")
	}
}

// Test3: ProcessConfig applies default host
func Test3(t *testing.T) {
	config := &Config{Port: 3000, Timeout: 60}
	ProcessConfig(config)
	if config.Host != "localhost" {
		t.Errorf("expected default host 'localhost', got: %s", config.Host)
	}
}

// Test4: ProcessConfig applies default port
func Test4(t *testing.T) {
	config := &Config{Host: "example.com", Timeout: 60}
	ProcessConfig(config)
	if config.Port != 8080 {
		t.Errorf("expected default port 8080, got: %d", config.Port)
	}
}

// Test5: ProcessConfig applies default timeout
func Test5(t *testing.T) {
	config := &Config{Host: "example.com", Port: 3000}
	ProcessConfig(config)
	if config.Timeout != 30 {
		t.Errorf("expected default timeout 30, got: %d", config.Timeout)
	}
}

// Test6: ValidateConfig with valid config
func Test6(t *testing.T) {
	config := &Config{Host: "example.com", Port: 8080, Timeout: 60}
	if !ValidateConfig(config) {
		t.Error("expected valid config")
	}
}

// Test7: ValidateConfig with nil config
func Test7(t *testing.T) {
	if ValidateConfig(nil) {
		t.Error("expected false for nil config")
	}
}

// Test8: ValidateConfig with invalid port (0)
func Test8(t *testing.T) {
	config := &Config{Host: "example.com", Port: 0, Timeout: 60}
	if ValidateConfig(config) {
		t.Error("expected false for port 0")
	}
}

// Test9: ValidateConfig with port out of range (>65535)
func Test9(t *testing.T) {
	config := &Config{Host: "example.com", Port: 70000, Timeout: 60}
	if ValidateConfig(config) {
		t.Error("expected false for port > 65535")
	}
}

// Test10: ValidateConfig with invalid timeout (>300)
func Test10(t *testing.T) {
	config := &Config{Host: "example.com", Port: 8080, Timeout: 500}
	if ValidateConfig(config) {
		t.Error("expected false for timeout > 300")
	}
}
`,
	translations: {
		ru: {
			title: 'Последовательное форматирование кода',
			description: `Применяйте последовательные правила форматирования: правильные отступы, пробелы, переносы строк и конвенции Go.`,
			hint1: `Добавьте пробелы вокруг операторов. Ставьте открывающие скобки на той же строке. Добавьте пустые строки между логическими секциями.`,
			hint2: `Форматируйте import с круглыми скобками. Добавьте переносы в определении структуры. Разделяйте секции функций пустыми строками.`,
			whyItMatters: `Последовательное форматирование улучшает читаемость и снижает когнитивную нагрузку.`,
			solutionCode: `package principles

import (
	"fmt"
	"strings"
)

type Config struct {
	Host    string
	Port    int
	Timeout int
}

func ProcessConfig(c *Config) error {
	if c == nil {
		return fmt.Errorf("nil config")
	}

	// Применяем значения по умолчанию
	if c.Host == "" {
		c.Host = "localhost"
	}
	if c.Port <= 0 {
		c.Port = 8080
	}
	if c.Timeout <= 0 {
		c.Timeout = 30
	}

	// Строим и отображаем URL подключения
	url := fmt.Sprintf("http://%s:%d", c.Host, c.Port)
	fmt.Printf("Connecting to: %s\n", url)

	return nil
}

func ValidateConfig(c *Config) bool {
	if c == nil {
		return false
	}

	if strings.TrimSpace(c.Host) == "" {
		return false
	}

	if c.Port < 1 || c.Port > 65535 {
		return false
	}

	if c.Timeout < 1 || c.Timeout > 300 {
		return false
	}

	return true
}`
		},
		uz: {
			title: 'Izchil kod formatlash',
			description: `Izchil formatlash qoidalarini qo'llang: to'g'ri chekinishlar, bo'shliqlar, qator uzilishlari va Go konventsiyalari.`,
			hint1: `Operatorlar atrofida bo'shliqlar qo'shing. Ochuvchi qavslarni bir xil qatorda qo'ying. Mantiqiy bo'limlar orasiga bo'sh qatorlar qo'shing.`,
			hint2: `Import ni qavslar bilan formatlang. Struct ta'rifiga yangi qatorlar qo'shing. Funksiya bo'limlarini bo'sh qatorlar bilan ajrating.`,
			whyItMatters: `Izchil formatlash o'qilishni yaxshilaydi va kognitiv yukni kamaytiradi.`,
			solutionCode: `package principles

import (
	"fmt"
	"strings"
)

type Config struct {
	Host    string
	Port    int
	Timeout int
}

func ProcessConfig(c *Config) error {
	if c == nil {
		return fmt.Errorf("nil config")
	}

	// Standart qiymatlarni qo'llash
	if c.Host == "" {
		c.Host = "localhost"
	}
	if c.Port <= 0 {
		c.Port = 8080
	}
	if c.Timeout <= 0 {
		c.Timeout = 30
	}

	// Ulanish URL sini qurish va ko'rsatish
	url := fmt.Sprintf("http://%s:%d", c.Host, c.Port)
	fmt.Printf("Connecting to: %s\n", url)

	return nil
}

func ValidateConfig(c *Config) bool {
	if c == nil {
		return false
	}

	if strings.TrimSpace(c.Host) == "" {
		return false
	}

	if c.Port < 1 || c.Port > 65535 {
		return false
	}

	if c.Timeout < 1 || c.Timeout > 300 {
		return false
	}

	return true
}`
		}
	}
};

export default task;
