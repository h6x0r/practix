import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-command',
	title: 'Command Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'behavioral', 'command'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Command pattern in Go - encapsulate a request as an object, enabling parameterization, queuing, logging, and undo operations.

The Command pattern turns a request into a stand-alone object containing all information about the request. This lets you pass requests as method arguments, delay or queue execution, and support undoable operations. The pattern separates the object that invokes the operation from the one that knows how to perform it.

**You will implement:**

1. **Command interface** - Common interface with Execute() and Undo() methods
2. **Light struct** - Receiver that performs the actual operations
3. **LightOnCommand/LightOffCommand** - Concrete commands that encapsulate operations
4. **RemoteControl** - Invoker that stores and executes commands with history

**Example Usage:**

\`\`\`go
light := &Light{}	// create receiver
remote := NewRemoteControl()	// create invoker

onCmd := &LightOnCommand{light: light}	// create command with receiver
remote.SetCommand(onCmd)	// set command on invoker
result1 := remote.PressButton()	// execute command
// result1: "Light is ON"

remote.SetCommand(&LightOffCommand{light: light})	// set different command
result2 := remote.PressButton()	// execute new command
// result2: "Light is OFF"

result3 := remote.PressUndo()	// undo last command
// result3: "Light is ON"

result4 := remote.PressUndo()	// undo previous command
// result4: "Light is OFF"
\`\`\``,
	initialCode: `package patterns

type Command interface {
}

type Light struct {
	isOn bool
}

func (l *Light) On() string {
}

func (l *Light) Off() string {
}

type LightOnCommand struct {
	light *Light
}

func (c *LightOnCommand) Execute() string {
}

func (c *LightOnCommand) Undo() string {
}

type LightOffCommand struct {
	light *Light
}

func (c *LightOffCommand) Execute() string {
}

func (c *LightOffCommand) Undo() string {
}

type RemoteControl struct {
	command Command
}

func NewRemoteControl() *RemoteControl {
}

func (r *RemoteControl) SetCommand(cmd Command) {
}

func (r *RemoteControl) PressButton() string {
}

func (r *RemoteControl) PressUndo() string {
}`,
	solutionCode: `package patterns

// Command interface defines operations all commands must implement
type Command interface {	// all commands implement this interface
	Execute() string	// perform the command action
	Undo() string	// reverse the command action
}

// Light is the receiver that performs actual operations
type Light struct {	// receiver class
	isOn bool	// internal state
}

// On turns the light on
func (l *Light) On() string {	// receiver method
	l.isOn = true	// update state
	return "Light is ON"	// return confirmation
}

// Off turns the light off
func (l *Light) Off() string {	// receiver method
	l.isOn = false	// update state
	return "Light is OFF"	// return confirmation
}

// LightOnCommand encapsulates the "turn on" action
type LightOnCommand struct {	// concrete command
	light *Light	// reference to receiver
}

// Execute turns the light on
func (c *LightOnCommand) Execute() string {	// implements Command interface
	return c.light.On()	// delegate to receiver
}

// Undo reverses the action (turns light off)
func (c *LightOnCommand) Undo() string {	// implements Command interface
	return c.light.Off()	// reverse action
}

// LightOffCommand encapsulates the "turn off" action
type LightOffCommand struct {	// concrete command
	light *Light	// reference to receiver
}

// Execute turns the light off
func (c *LightOffCommand) Execute() string {	// implements Command interface
	return c.light.Off()	// delegate to receiver
}

// Undo reverses the action (turns light on)
func (c *LightOffCommand) Undo() string {	// implements Command interface
	return c.light.On()	// reverse action
}

// RemoteControl is the invoker that triggers commands
type RemoteControl struct {	// invoker class
	command Command	// current command to execute
	history []Command	// stack of executed commands for undo
}

// NewRemoteControl creates a new remote control
func NewRemoteControl() *RemoteControl {	// factory function
	return &RemoteControl{history: make([]Command, 0)}	// initialize empty history
}

// SetCommand sets the current command
func (r *RemoteControl) SetCommand(cmd Command) {	// configure invoker
	r.command = cmd	// store command reference
}

// PressButton executes the current command
func (r *RemoteControl) PressButton() string {	// trigger execution
	if r.command == nil {	// check if command is set
		return "No command set"	// error message if no command
	}
	result := r.command.Execute()	// execute the command
	r.history = append(r.history, r.command)	// add to history for undo
	return result	// return result
}

// PressUndo undoes the last executed command
func (r *RemoteControl) PressUndo() string {	// undo functionality
	if len(r.history) == 0 {	// check if history has commands
		return "No commands to undo"	// error if nothing to undo
	}
	lastCmd := r.history[len(r.history)-1]	// get last executed command
	r.history = r.history[:len(r.history)-1]	// remove from history
	return lastCmd.Undo()	// call undo on the command
}`,
	hint1: `Light is the receiver with On() and Off() methods that set isOn state and return status strings. LightOnCommand.Execute calls light.On(), and its Undo calls light.Off() (opposite action). LightOffCommand is the reverse.`,
	hint2: `RemoteControl is the invoker. SetCommand stores the command reference. PressButton checks for nil, executes the command, adds it to history, and returns the result. PressUndo checks if history is empty, pops the last command from history, and calls its Undo method.`,
	whyItMatters: `**Why the Command Pattern Exists**

Without Command, the invoker directly calls methods on receivers:

\`\`\`go
// Problem: No way to undo, queue, or log operations
type RemoteControl struct {
    light *Light
}

func (r *RemoteControl) TurnOn() string {
    return r.light.On()  // directly coupled to Light
    // How do we undo? How do we queue multiple operations?
    // How do we support different devices?
}
\`\`\`

With Command, operations become first-class objects:

\`\`\`go
// Solution: Operations are encapsulated as objects
type Command interface {
    Execute() string
    Undo() string
}

func (r *RemoteControl) PressButton() string {
    result := r.command.Execute()  // works with any command
    r.history = append(r.history, r.command)  // can track history
    return result
}
// Commands can be queued, logged, undone, serialized
\`\`\`

**Real-World Command Examples in Go**

1. **Text Editor with Undo/Redo Stack**:
\`\`\`go
type TextCommand interface {
    Execute() string
    Undo() string
}

type InsertTextCommand struct {
    document *Document
    position int
    text     string
}

func (c *InsertTextCommand) Execute() string {
    c.document.InsertAt(c.position, c.text)
    return fmt.Sprintf("Inserted '%s' at position %d", c.text, c.position)
}

func (c *InsertTextCommand) Undo() string {
    c.document.DeleteAt(c.position, len(c.text))
    return fmt.Sprintf("Removed '%s' from position %d", c.text, c.position)
}

type TextEditor struct {
    undoStack []TextCommand
    redoStack []TextCommand
}

func (e *TextEditor) Execute(cmd TextCommand) {
    cmd.Execute()
    e.undoStack = append(e.undoStack, cmd)
    e.redoStack = nil  // clear redo stack on new command
}

func (e *TextEditor) Undo() {
    if len(e.undoStack) == 0 { return }
    cmd := e.undoStack[len(e.undoStack)-1]
    e.undoStack = e.undoStack[:len(e.undoStack)-1]
    cmd.Undo()
    e.redoStack = append(e.redoStack, cmd)
}
\`\`\`

2. **Database Transaction with Rollback**:
\`\`\`go
type DBCommand interface {
    Execute() error
    Rollback() error
}

type InsertCommand struct {
    db    *Database
    table string
    data  map[string]interface{}
    id    int  // stored after execute for rollback
}

func (c *InsertCommand) Execute() error {
    id, err := c.db.Insert(c.table, c.data)
    c.id = id
    return err
}

func (c *InsertCommand) Rollback() error {
    return c.db.Delete(c.table, c.id)
}

type Transaction struct {
    commands []DBCommand
}

func (t *Transaction) Commit() error {
    for i, cmd := range t.commands {
        if err := cmd.Execute(); err != nil {
            // Rollback all executed commands
            for j := i - 1; j >= 0; j-- {
                t.commands[j].Rollback()
            }
            return err
        }
    }
    return nil
}
\`\`\`

**Production Pattern: Job Queue System**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Job represents a command with metadata
type Job interface {
    Execute() error
    GetID() string
    GetPriority() int
}

// EmailJob sends an email
type EmailJob struct {
    ID        string
    Priority  int
    To        string
    Subject   string
    Body      string
}

func (j *EmailJob) Execute() error {
    fmt.Printf("Sending email to %s: %s\n", j.To, j.Subject)
    time.Sleep(100 * time.Millisecond)  // simulate sending
    return nil
}

func (j *EmailJob) GetID() string      { return j.ID }
func (j *EmailJob) GetPriority() int   { return j.Priority }

// ReportJob generates a report
type ReportJob struct {
    ID       string
    Priority int
    Type     string
    Period   string
}

func (j *ReportJob) Execute() error {
    fmt.Printf("Generating %s report for %s\n", j.Type, j.Period)
    time.Sleep(200 * time.Millisecond)  // simulate generation
    return nil
}

func (j *ReportJob) GetID() string      { return j.ID }
func (j *ReportJob) GetPriority() int   { return j.Priority }

// JobQueue manages and executes jobs
type JobQueue struct {
    jobs      []Job
    history   []Job
    running   bool
    mu        sync.Mutex
    workers   int
    jobChan   chan Job
}

func NewJobQueue(workers int) *JobQueue {
    return &JobQueue{
        jobs:    make([]Job, 0),
        history: make([]Job, 0),
        workers: workers,
        jobChan: make(chan Job, 100),
    }
}

func (q *JobQueue) Enqueue(job Job) {
    q.mu.Lock()
    defer q.mu.Unlock()
    q.jobs = append(q.jobs, job)
    // Sort by priority (higher first)
    for i := len(q.jobs) - 1; i > 0; i-- {
        if q.jobs[i].GetPriority() > q.jobs[i-1].GetPriority() {
            q.jobs[i], q.jobs[i-1] = q.jobs[i-1], q.jobs[i]
        }
    }
}

func (q *JobQueue) Start() {
    q.running = true
    // Start worker goroutines
    for i := 0; i < q.workers; i++ {
        go q.worker(i)
    }
    // Job dispatcher
    go func() {
        for q.running {
            q.mu.Lock()
            if len(q.jobs) > 0 {
                job := q.jobs[0]
                q.jobs = q.jobs[1:]
                q.mu.Unlock()
                q.jobChan <- job
            } else {
                q.mu.Unlock()
                time.Sleep(100 * time.Millisecond)
            }
        }
    }()
}

func (q *JobQueue) worker(id int) {
    for job := range q.jobChan {
        fmt.Printf("Worker %d processing job %s\n", id, job.GetID())
        if err := job.Execute(); err != nil {
            fmt.Printf("Job %s failed: %v\n", job.GetID(), err)
        } else {
            q.mu.Lock()
            q.history = append(q.history, job)
            q.mu.Unlock()
        }
    }
}

func (q *JobQueue) Stop() {
    q.running = false
    close(q.jobChan)
}

func (q *JobQueue) GetHistory() []Job {
    q.mu.Lock()
    defer q.mu.Unlock()
    return append([]Job{}, q.history...)
}
\`\`\`

**Common Mistakes to Avoid**

1. **Commands with too much logic** - Commands should delegate to receivers, not implement business logic
2. **Forgetting to store state for undo** - Commands need to capture enough state to reverse the operation
3. **Not handling nil command** - Always check if command is set before executing
4. **Mutable command state** - Commands should be immutable after creation for safe undo/redo
5. **Unbounded history** - Consider limiting history size or implementing history cleanup`,
	order: 2,
	testCode: `package patterns

import (
	"testing"
)

// Test1: Light.On returns correct message
func Test1(t *testing.T) {
	l := &Light{}
	result := l.On()
	if result != "Light is ON" {
		t.Errorf("Expected 'Light is ON', got '%s'", result)
	}
}

// Test2: Light.Off returns correct message
func Test2(t *testing.T) {
	l := &Light{}
	result := l.Off()
	if result != "Light is OFF" {
		t.Errorf("Expected 'Light is OFF', got '%s'", result)
	}
}

// Test3: LightOnCommand.Execute calls light.On
func Test3(t *testing.T) {
	l := &Light{}
	cmd := &LightOnCommand{light: l}
	result := cmd.Execute()
	if result != "Light is ON" {
		t.Error("Execute should turn light on")
	}
}

// Test4: LightOnCommand.Undo calls light.Off
func Test4(t *testing.T) {
	l := &Light{}
	cmd := &LightOnCommand{light: l}
	result := cmd.Undo()
	if result != "Light is OFF" {
		t.Error("Undo should turn light off")
	}
}

// Test5: LightOffCommand.Execute calls light.Off
func Test5(t *testing.T) {
	l := &Light{}
	cmd := &LightOffCommand{light: l}
	result := cmd.Execute()
	if result != "Light is OFF" {
		t.Error("Execute should turn light off")
	}
}

// Test6: LightOffCommand.Undo calls light.On
func Test6(t *testing.T) {
	l := &Light{}
	cmd := &LightOffCommand{light: l}
	result := cmd.Undo()
	if result != "Light is ON" {
		t.Error("Undo should turn light on")
	}
}

// Test7: RemoteControl.PressButton executes command
func Test7(t *testing.T) {
	l := &Light{}
	r := NewRemoteControl()
	r.SetCommand(&LightOnCommand{light: l})
	result := r.PressButton()
	if result != "Light is ON" {
		t.Error("PressButton should execute command")
	}
}

// Test8: RemoteControl.PressUndo undoes last command
func Test8(t *testing.T) {
	l := &Light{}
	r := NewRemoteControl()
	r.SetCommand(&LightOnCommand{light: l})
	r.PressButton()
	result := r.PressUndo()
	if result != "Light is OFF" {
		t.Error("PressUndo should undo last command")
	}
}

// Test9: PressButton without command returns error
func Test9(t *testing.T) {
	r := NewRemoteControl()
	result := r.PressButton()
	if result != "No command set" {
		t.Error("Should return error when no command set")
	}
}

// Test10: PressUndo with empty history returns error
func Test10(t *testing.T) {
	r := NewRemoteControl()
	result := r.PressUndo()
	if result != "No commands to undo" {
		t.Error("Should return error when history empty")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Command (Команда)',
			description: `Реализуйте паттерн Command на Go — инкапсулируйте запрос как объект, позволяя параметризацию, очередь, логирование и операции отмены.

Паттерн Command превращает запрос в самостоятельный объект, содержащий всю информацию о запросе. Это позволяет передавать запросы как аргументы методов, откладывать или ставить в очередь выполнение, и поддерживать отменяемые операции. Паттерн отделяет объект, вызывающий операцию, от объекта, который знает как её выполнить.

**Вы реализуете:**

1. **Интерфейс Command** — Общий интерфейс с методами Execute() и Undo()
2. **Структура Light** — Получатель, выполняющий реальные операции
3. **LightOnCommand/LightOffCommand** — Конкретные команды, инкапсулирующие операции
4. **RemoteControl** — Инициатор, хранящий и выполняющий команды с историей

**Пример использования:**

\`\`\`go
light := &Light{}	// создаём получателя
remote := NewRemoteControl()	// создаём инициатора

onCmd := &LightOnCommand{light: light}	// создаём команду с получателем
remote.SetCommand(onCmd)	// устанавливаем команду на инициатор
result1 := remote.PressButton()	// выполняем команду
// result1: "Light is ON"

remote.SetCommand(&LightOffCommand{light: light})	// устанавливаем другую команду
result2 := remote.PressButton()	// выполняем новую команду
// result2: "Light is OFF"

result3 := remote.PressUndo()	// отменяем последнюю команду
// result3: "Light is ON"

result4 := remote.PressUndo()	// отменяем предыдущую команду
// result4: "Light is OFF"
\`\`\``,
			hint1: `Light — получатель с методами On() и Off(), которые устанавливают состояние isOn и возвращают строки статуса. LightOnCommand.Execute вызывает light.On(), а его Undo вызывает light.Off() (противоположное действие). LightOffCommand — наоборот.`,
			hint2: `RemoteControl — инициатор. SetCommand сохраняет ссылку на команду. PressButton проверяет на nil, выполняет команду, добавляет в историю и возвращает результат. PressUndo проверяет пустоту истории, извлекает последнюю команду из истории и вызывает её метод Undo.`,
			whyItMatters: `**Зачем нужен паттерн Command**

Без Command инициатор напрямую вызывает методы получателей:

\`\`\`go
// Проблема: Нет способа отменить, поставить в очередь или залогировать операции
type RemoteControl struct {
    light *Light
}

func (r *RemoteControl) TurnOn() string {
    return r.light.On()  // напрямую связан с Light
    // Как отменить? Как поставить несколько операций в очередь?
    // Как поддержать разные устройства?
}
\`\`\`

С Command операции становятся объектами первого класса:

\`\`\`go
// Решение: Операции инкапсулированы как объекты
type Command interface {
    Execute() string
    Undo() string
}

func (r *RemoteControl) PressButton() string {
    result := r.command.Execute()  // работает с любой командой
    r.history = append(r.history, r.command)  // можно отслеживать историю
    return result
}
// Команды можно ставить в очередь, логировать, отменять, сериализовать
\`\`\`

**Реальные примеры Command в Go**

1. **Текстовый редактор с Undo/Redo стеком**:
\`\`\`go
type TextCommand interface {
    Execute() string
    Undo() string
}

type InsertTextCommand struct {
    document *Document
    position int
    text     string
}

func (c *InsertTextCommand) Execute() string {
    c.document.InsertAt(c.position, c.text)
    return fmt.Sprintf("Inserted '%s' at position %d", c.text, c.position)
}

func (c *InsertTextCommand) Undo() string {
    c.document.DeleteAt(c.position, len(c.text))
    return fmt.Sprintf("Removed '%s' from position %d", c.text, c.position)
}

type TextEditor struct {
    undoStack []TextCommand
    redoStack []TextCommand
}

func (e *TextEditor) Execute(cmd TextCommand) {
    cmd.Execute()
    e.undoStack = append(e.undoStack, cmd)
    e.redoStack = nil  // очищаем redo стек при новой команде
}

func (e *TextEditor) Undo() {
    if len(e.undoStack) == 0 { return }
    cmd := e.undoStack[len(e.undoStack)-1]
    e.undoStack = e.undoStack[:len(e.undoStack)-1]
    cmd.Undo()
    e.redoStack = append(e.redoStack, cmd)
}
\`\`\`

2. **Транзакция базы данных с откатом**:
\`\`\`go
type DBCommand interface {
    Execute() error
    Rollback() error
}

type InsertCommand struct {
    db    *Database
    table string
    data  map[string]interface{}
    id    int  // сохраняется после execute для отката
}

func (c *InsertCommand) Execute() error {
    id, err := c.db.Insert(c.table, c.data)
    c.id = id
    return err
}

func (c *InsertCommand) Rollback() error {
    return c.db.Delete(c.table, c.id)
}

type Transaction struct {
    commands []DBCommand
}

func (t *Transaction) Commit() error {
    for i, cmd := range t.commands {
        if err := cmd.Execute(); err != nil {
            // Откатываем все выполненные команды
            for j := i - 1; j >= 0; j-- {
                t.commands[j].Rollback()
            }
            return err
        }
    }
    return nil
}
\`\`\`

**Продакшен паттерн: Система очереди задач**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Job представляет команду с метаданными
type Job interface {
    Execute() error
    GetID() string
    GetPriority() int
}

// EmailJob отправляет email
type EmailJob struct {
    ID        string
    Priority  int
    To        string
    Subject   string
    Body      string
}

func (j *EmailJob) Execute() error {
    fmt.Printf("Sending email to %s: %s\n", j.To, j.Subject)
    time.Sleep(100 * time.Millisecond)  // имитация отправки
    return nil
}

func (j *EmailJob) GetID() string      { return j.ID }
func (j *EmailJob) GetPriority() int   { return j.Priority }

// ReportJob генерирует отчёт
type ReportJob struct {
    ID       string
    Priority int
    Type     string
    Period   string
}

func (j *ReportJob) Execute() error {
    fmt.Printf("Generating %s report for %s\n", j.Type, j.Period)
    time.Sleep(200 * time.Millisecond)  // имитация генерации
    return nil
}

func (j *ReportJob) GetID() string      { return j.ID }
func (j *ReportJob) GetPriority() int   { return j.Priority }

// JobQueue управляет и выполняет задачи
type JobQueue struct {
    jobs      []Job
    history   []Job
    running   bool
    mu        sync.Mutex
    workers   int
    jobChan   chan Job
}

func NewJobQueue(workers int) *JobQueue {
    return &JobQueue{
        jobs:    make([]Job, 0),
        history: make([]Job, 0),
        workers: workers,
        jobChan: make(chan Job, 100),
    }
}

func (q *JobQueue) Enqueue(job Job) {
    q.mu.Lock()
    defer q.mu.Unlock()
    q.jobs = append(q.jobs, job)
    // Сортировка по приоритету (высший первым)
    for i := len(q.jobs) - 1; i > 0; i-- {
        if q.jobs[i].GetPriority() > q.jobs[i-1].GetPriority() {
            q.jobs[i], q.jobs[i-1] = q.jobs[i-1], q.jobs[i]
        }
    }
}

func (q *JobQueue) Start() {
    q.running = true
    // Запускаем рабочие горутины
    for i := 0; i < q.workers; i++ {
        go q.worker(i)
    }
    // Диспетчер задач
    go func() {
        for q.running {
            q.mu.Lock()
            if len(q.jobs) > 0 {
                job := q.jobs[0]
                q.jobs = q.jobs[1:]
                q.mu.Unlock()
                q.jobChan <- job
            } else {
                q.mu.Unlock()
                time.Sleep(100 * time.Millisecond)
            }
        }
    }()
}

func (q *JobQueue) worker(id int) {
    for job := range q.jobChan {
        fmt.Printf("Worker %d processing job %s\n", id, job.GetID())
        if err := job.Execute(); err != nil {
            fmt.Printf("Job %s failed: %v\n", job.GetID(), err)
        } else {
            q.mu.Lock()
            q.history = append(q.history, job)
            q.mu.Unlock()
        }
    }
}

func (q *JobQueue) Stop() {
    q.running = false
    close(q.jobChan)
}

func (q *JobQueue) GetHistory() []Job {
    q.mu.Lock()
    defer q.mu.Unlock()
    return append([]Job{}, q.history...)
}
\`\`\`

**Распространённые ошибки**

1. **Команды со слишком большой логикой** — Команды должны делегировать получателям, а не реализовывать бизнес-логику
2. **Забывают сохранить состояние для отмены** — Команды должны захватывать достаточно состояния для отмены операции
3. **Не обрабатывают nil команду** — Всегда проверяйте, установлена ли команда перед выполнением
4. **Изменяемое состояние команды** — Команды должны быть неизменяемыми после создания для безопасной отмены/повтора
5. **Неограниченная история** — Рассмотрите ограничение размера истории или очистку`,
			solutionCode: `package patterns

// Command интерфейс определяет операции, которые должны реализовать все команды
type Command interface {	// все команды реализуют этот интерфейс
	Execute() string	// выполнить действие команды
	Undo() string	// отменить действие команды
}

// Light — получатель, выполняющий реальные операции
type Light struct {	// класс получателя
	isOn bool	// внутреннее состояние
}

// On включает свет
func (l *Light) On() string {	// метод получателя
	l.isOn = true	// обновляем состояние
	return "Light is ON"	// возвращаем подтверждение
}

// Off выключает свет
func (l *Light) Off() string {	// метод получателя
	l.isOn = false	// обновляем состояние
	return "Light is OFF"	// возвращаем подтверждение
}

// LightOnCommand инкапсулирует действие "включить"
type LightOnCommand struct {	// конкретная команда
	light *Light	// ссылка на получателя
}

// Execute включает свет
func (c *LightOnCommand) Execute() string {	// реализует интерфейс Command
	return c.light.On()	// делегируем получателю
}

// Undo отменяет действие (выключает свет)
func (c *LightOnCommand) Undo() string {	// реализует интерфейс Command
	return c.light.Off()	// обратное действие
}

// LightOffCommand инкапсулирует действие "выключить"
type LightOffCommand struct {	// конкретная команда
	light *Light	// ссылка на получателя
}

// Execute выключает свет
func (c *LightOffCommand) Execute() string {	// реализует интерфейс Command
	return c.light.Off()	// делегируем получателю
}

// Undo отменяет действие (включает свет)
func (c *LightOffCommand) Undo() string {	// реализует интерфейс Command
	return c.light.On()	// обратное действие
}

// RemoteControl — инициатор, запускающий команды
type RemoteControl struct {	// класс инициатора
	command Command	// текущая команда для выполнения
	history []Command	// стек выполненных команд для отмены
}

// NewRemoteControl создаёт новый пульт управления
func NewRemoteControl() *RemoteControl {	// фабричная функция
	return &RemoteControl{history: make([]Command, 0)}	// инициализируем пустую историю
}

// SetCommand устанавливает текущую команду
func (r *RemoteControl) SetCommand(cmd Command) {	// настройка инициатора
	r.command = cmd	// сохраняем ссылку на команду
}

// PressButton выполняет текущую команду
func (r *RemoteControl) PressButton() string {	// запуск выполнения
	if r.command == nil {	// проверяем, установлена ли команда
		return "No command set"	// сообщение об ошибке если нет команды
	}
	result := r.command.Execute()	// выполняем команду
	r.history = append(r.history, r.command)	// добавляем в историю для отмены
	return result	// возвращаем результат
}

// PressUndo отменяет последнюю выполненную команду
func (r *RemoteControl) PressUndo() string {	// функциональность отмены
	if len(r.history) == 0 {	// проверяем, есть ли команды в истории
		return "No commands to undo"	// ошибка если нечего отменять
	}
	lastCmd := r.history[len(r.history)-1]	// получаем последнюю выполненную команду
	r.history = r.history[:len(r.history)-1]	// удаляем из истории
	return lastCmd.Undo()	// вызываем undo на команде
}`
		},
		uz: {
			title: 'Command (Buyruq) Pattern',
			description: `Go tilida Command patternini amalga oshiring — so'rovni ob'ekt sifatida inkapsulyatsiya qiling, parametrlash, navbat qo'yish, loglash va bekor qilish operatsiyalarini yoqing.

Command patterni so'rovni so'rov haqidagi barcha ma'lumotlarni o'z ichiga olgan mustaqil ob'ektga aylantiradi. Bu so'rovlarni metod argumentlari sifatida uzatish, bajarishni kechiktirish yoki navbatga qo'yish va bekor qilinadigan operatsiyalarni qo'llab-quvvatlash imkonini beradi. Pattern operatsiyani chaqiradigan ob'ektni uni qanday bajarishni biladigan ob'ektdan ajratadi.

**Siz amalga oshirasiz:**

1. **Command interfeysi** — Execute() va Undo() metodlari bilan umumiy interfeys
2. **Light struct** — Haqiqiy operatsiyalarni bajaradigan qabul qiluvchi
3. **LightOnCommand/LightOffCommand** — Operatsiyalarni inkapsulyatsiya qiluvchi konkret buyruqlar
4. **RemoteControl** — Tarix bilan buyruqlarni saqlovchi va bajaruvchi chaqiruvchi

**Foydalanish namunasi:**

\`\`\`go
light := &Light{}	// qabul qiluvchi yaratamiz
remote := NewRemoteControl()	// chaqiruvchi yaratamiz

onCmd := &LightOnCommand{light: light}	// qabul qiluvchi bilan buyruq yaratamiz
remote.SetCommand(onCmd)	// chaqiruvchiga buyruq o'rnatamiz
result1 := remote.PressButton()	// buyruqni bajaramiz
// result1: "Light is ON"

remote.SetCommand(&LightOffCommand{light: light})	// boshqa buyruq o'rnatamiz
result2 := remote.PressButton()	// yangi buyruqni bajaramiz
// result2: "Light is OFF"

result3 := remote.PressUndo()	// oxirgi buyruqni bekor qilamiz
// result3: "Light is ON"

result4 := remote.PressUndo()	// oldingi buyruqni bekor qilamiz
// result4: "Light is OFF"
\`\`\``,
			hint1: `Light — On() va Off() metodlari bilan qabul qiluvchi, ular isOn holatini o'rnatadi va status satrlarini qaytaradi. LightOnCommand.Execute light.On() ni, uning Undo si light.Off() ni chaqiradi (teskari harakat). LightOffCommand teskari.`,
			hint2: `RemoteControl — chaqiruvchi. SetCommand buyruq havolasini saqlaydi. PressButton nil ni tekshiradi, buyruqni bajaradi, tarixga qo'shadi va natijani qaytaradi. PressUndo tarix bo'shligini tekshiradi, tarixdan oxirgi buyruqni oladi va uning Undo metodini chaqiradi.`,
			whyItMatters: `**Command Pattern nima uchun kerak**

Command siz chaqiruvchi to'g'ridan-to'g'ri qabul qiluvchilar metodlarini chaqiradi:

\`\`\`go
// Muammo: Bekor qilish, navbatga qo'yish yoki loglash yo'li yo'q
type RemoteControl struct {
    light *Light
}

func (r *RemoteControl) TurnOn() string {
    return r.light.On()  // to'g'ridan-to'g'ri Light ga bog'langan
    // Qanday bekor qilamiz? Bir nechta operatsiyalarni navbatga qanday qo'yamiz?
    // Turli qurilmalarni qanday qo'llab-quvvatlaymiz?
}
\`\`\`

Command bilan operatsiyalar birinchi darajali ob'ektlarga aylanadi:

\`\`\`go
// Yechim: Operatsiyalar ob'ektlar sifatida inkapsulyatsiya qilingan
type Command interface {
    Execute() string
    Undo() string
}

func (r *RemoteControl) PressButton() string {
    result := r.command.Execute()  // har qanday buyruq bilan ishlaydi
    r.history = append(r.history, r.command)  // tarixni kuzatish mumkin
    return result
}
// Buyruqlarni navbatga qo'yish, loglash, bekor qilish, seriyalash mumkin
\`\`\`

**Go da Command ning real dunyo misollari**

1. **Undo/Redo stekli matn muharriri**:
\`\`\`go
type TextCommand interface {
    Execute() string
    Undo() string
}

type InsertTextCommand struct {
    document *Document
    position int
    text     string
}

func (c *InsertTextCommand) Execute() string {
    c.document.InsertAt(c.position, c.text)
    return fmt.Sprintf("Inserted '%s' at position %d", c.text, c.position)
}

func (c *InsertTextCommand) Undo() string {
    c.document.DeleteAt(c.position, len(c.text))
    return fmt.Sprintf("Removed '%s' from position %d", c.text, c.position)
}

type TextEditor struct {
    undoStack []TextCommand
    redoStack []TextCommand
}

func (e *TextEditor) Execute(cmd TextCommand) {
    cmd.Execute()
    e.undoStack = append(e.undoStack, cmd)
    e.redoStack = nil  // yangi buyruqda redo stekini tozalaymiz
}

func (e *TextEditor) Undo() {
    if len(e.undoStack) == 0 { return }
    cmd := e.undoStack[len(e.undoStack)-1]
    e.undoStack = e.undoStack[:len(e.undoStack)-1]
    cmd.Undo()
    e.redoStack = append(e.redoStack, cmd)
}
\`\`\`

2. **Orqaga qaytarish bilan ma'lumotlar bazasi tranzaksiyasi**:
\`\`\`go
type DBCommand interface {
    Execute() error
    Rollback() error
}

type InsertCommand struct {
    db    *Database
    table string
    data  map[string]interface{}
    id    int  // orqaga qaytarish uchun execute dan keyin saqlanadi
}

func (c *InsertCommand) Execute() error {
    id, err := c.db.Insert(c.table, c.data)
    c.id = id
    return err
}

func (c *InsertCommand) Rollback() error {
    return c.db.Delete(c.table, c.id)
}

type Transaction struct {
    commands []DBCommand
}

func (t *Transaction) Commit() error {
    for i, cmd := range t.commands {
        if err := cmd.Execute(); err != nil {
            // Barcha bajarilgan buyruqlarni orqaga qaytaramiz
            for j := i - 1; j >= 0; j-- {
                t.commands[j].Rollback()
            }
            return err
        }
    }
    return nil
}
\`\`\`

**Prodakshen pattern: Vazifalar navbati tizimi**

\`\`\`go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Job metadata bilan buyruqni ifodalaydi
type Job interface {
    Execute() error
    GetID() string
    GetPriority() int
}

// EmailJob email yuboradi
type EmailJob struct {
    ID        string
    Priority  int
    To        string
    Subject   string
    Body      string
}

func (j *EmailJob) Execute() error {
    fmt.Printf("Sending email to %s: %s\n", j.To, j.Subject)
    time.Sleep(100 * time.Millisecond)  // yuborishni simulyatsiya qilish
    return nil
}

func (j *EmailJob) GetID() string      { return j.ID }
func (j *EmailJob) GetPriority() int   { return j.Priority }

// ReportJob hisobot yaratadi
type ReportJob struct {
    ID       string
    Priority int
    Type     string
    Period   string
}

func (j *ReportJob) Execute() error {
    fmt.Printf("Generating %s report for %s\n", j.Type, j.Period)
    time.Sleep(200 * time.Millisecond)  // yaratishni simulyatsiya qilish
    return nil
}

func (j *ReportJob) GetID() string      { return j.ID }
func (j *ReportJob) GetPriority() int   { return j.Priority }

// JobQueue vazifalarni boshqaradi va bajaradi
type JobQueue struct {
    jobs      []Job
    history   []Job
    running   bool
    mu        sync.Mutex
    workers   int
    jobChan   chan Job
}

func NewJobQueue(workers int) *JobQueue {
    return &JobQueue{
        jobs:    make([]Job, 0),
        history: make([]Job, 0),
        workers: workers,
        jobChan: make(chan Job, 100),
    }
}

func (q *JobQueue) Enqueue(job Job) {
    q.mu.Lock()
    defer q.mu.Unlock()
    q.jobs = append(q.jobs, job)
    // Prioritet bo'yicha saralash (yuqori birinchi)
    for i := len(q.jobs) - 1; i > 0; i-- {
        if q.jobs[i].GetPriority() > q.jobs[i-1].GetPriority() {
            q.jobs[i], q.jobs[i-1] = q.jobs[i-1], q.jobs[i]
        }
    }
}

func (q *JobQueue) Start() {
    q.running = true
    // Ishchi gorutinlarni ishga tushiramiz
    for i := 0; i < q.workers; i++ {
        go q.worker(i)
    }
    // Vazifa dispetcheri
    go func() {
        for q.running {
            q.mu.Lock()
            if len(q.jobs) > 0 {
                job := q.jobs[0]
                q.jobs = q.jobs[1:]
                q.mu.Unlock()
                q.jobChan <- job
            } else {
                q.mu.Unlock()
                time.Sleep(100 * time.Millisecond)
            }
        }
    }()
}

func (q *JobQueue) worker(id int) {
    for job := range q.jobChan {
        fmt.Printf("Worker %d processing job %s\n", id, job.GetID())
        if err := job.Execute(); err != nil {
            fmt.Printf("Job %s failed: %v\n", job.GetID(), err)
        } else {
            q.mu.Lock()
            q.history = append(q.history, job)
            q.mu.Unlock()
        }
    }
}

func (q *JobQueue) Stop() {
    q.running = false
    close(q.jobChan)
}

func (q *JobQueue) GetHistory() []Job {
    q.mu.Lock()
    defer q.mu.Unlock()
    return append([]Job{}, q.history...)
}
\`\`\`

**Oldini olish kerak bo'lgan keng tarqalgan xatolar**

1. **Juda ko'p mantiqqa ega buyruqlar** — Buyruqlar qabul qiluvchilarga delegatsiya qilishi kerak, biznes mantiqni amalga oshirmasligi kerak
2. **Bekor qilish uchun holatni saqlashni unutish** — Buyruqlar operatsiyani bekor qilish uchun yetarli holatni yozib olishi kerak
3. **Nil buyruqni qayta ishlamaslik** — Bajarishdan oldin doimo buyruq o'rnatilganligini tekshiring
4. **O'zgaruvchan buyruq holati** — Xavfsiz bekor qilish/qayta bajarish uchun buyruqlar yaratilgandan keyin o'zgarmas bo'lishi kerak
5. **Cheksiz tarix** — Tarix hajmini cheklash yoki tarixni tozalashni ko'rib chiqing`,
			solutionCode: `package patterns

// Command interfeysi barcha buyruqlar amalga oshirishi kerak bo'lgan operatsiyalarni aniqlaydi
type Command interface {	// barcha buyruqlar bu interfeysni amalga oshiradi
	Execute() string	// buyruq harakatini bajarish
	Undo() string	// buyruq harakatini bekor qilish
}

// Light — haqiqiy operatsiyalarni bajaradigan qabul qiluvchi
type Light struct {	// qabul qiluvchi sinfi
	isOn bool	// ichki holat
}

// On chiroqni yoqadi
func (l *Light) On() string {	// qabul qiluvchi metodi
	l.isOn = true	// holatni yangilaymiz
	return "Light is ON"	// tasdiqlashni qaytaramiz
}

// Off chiroqni o'chiradi
func (l *Light) Off() string {	// qabul qiluvchi metodi
	l.isOn = false	// holatni yangilaymiz
	return "Light is OFF"	// tasdiqlashni qaytaramiz
}

// LightOnCommand "yoqish" harakatini inkapsulyatsiya qiladi
type LightOnCommand struct {	// konkret buyruq
	light *Light	// qabul qiluvchiga havola
}

// Execute chiroqni yoqadi
func (c *LightOnCommand) Execute() string {	// Command interfeysini amalga oshiradi
	return c.light.On()	// qabul qiluvchiga delegatsiya qilamiz
}

// Undo harakatni bekor qiladi (chiroqni o'chiradi)
func (c *LightOnCommand) Undo() string {	// Command interfeysini amalga oshiradi
	return c.light.Off()	// teskari harakat
}

// LightOffCommand "o'chirish" harakatini inkapsulyatsiya qiladi
type LightOffCommand struct {	// konkret buyruq
	light *Light	// qabul qiluvchiga havola
}

// Execute chiroqni o'chiradi
func (c *LightOffCommand) Execute() string {	// Command interfeysini amalga oshiradi
	return c.light.Off()	// qabul qiluvchiga delegatsiya qilamiz
}

// Undo harakatni bekor qiladi (chiroqni yoqadi)
func (c *LightOffCommand) Undo() string {	// Command interfeysini amalga oshiradi
	return c.light.On()	// teskari harakat
}

// RemoteControl — buyruqlarni ishga tushiruvchi chaqiruvchi
type RemoteControl struct {	// chaqiruvchi sinfi
	command Command	// bajarish uchun joriy buyruq
	history []Command	// bekor qilish uchun bajarilgan buyruqlar steki
}

// NewRemoteControl yangi masofadan boshqarish pulti yaratadi
func NewRemoteControl() *RemoteControl {	// fabrika funksiyasi
	return &RemoteControl{history: make([]Command, 0)}	// bo'sh tarixni initsializatsiya qilamiz
}

// SetCommand joriy buyruqni o'rnatadi
func (r *RemoteControl) SetCommand(cmd Command) {	// chaqiruvchini sozlash
	r.command = cmd	// buyruq havolasini saqlaymiz
}

// PressButton joriy buyruqni bajaradi
func (r *RemoteControl) PressButton() string {	// bajarishni ishga tushirish
	if r.command == nil {	// buyruq o'rnatilganligini tekshiramiz
		return "No command set"	// buyruq yo'q bo'lsa xato xabari
	}
	result := r.command.Execute()	// buyruqni bajaramiz
	r.history = append(r.history, r.command)	// bekor qilish uchun tarixga qo'shamiz
	return result	// natijani qaytaramiz
}

// PressUndo oxirgi bajarilgan buyruqni bekor qiladi
func (r *RemoteControl) PressUndo() string {	// bekor qilish funksionalligi
	if len(r.history) == 0 {	// tarixda buyruqlar borligini tekshiramiz
		return "No commands to undo"	// bekor qilish uchun hech narsa yo'q bo'lsa xato
	}
	lastCmd := r.history[len(r.history)-1]	// oxirgi bajarilgan buyruqni olamiz
	r.history = r.history[:len(r.history)-1]	// tarixdan olib tashlaymiz
	return lastCmd.Undo()	// buyruqda undo ni chaqiramiz
}`
		}
	}
};

export default task;
