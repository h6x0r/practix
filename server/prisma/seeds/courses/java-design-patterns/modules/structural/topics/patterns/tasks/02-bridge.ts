import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-bridge',
	title: 'Bridge Pattern',
	difficulty: 'hard',
	tags: ['java', 'design-patterns', 'structural', 'bridge'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Bridge pattern in Java - decouple abstraction from implementation.

**You will implement:**

1. **Device interface** - Implementation interface
2. **TV, Radio** - Concrete implementations
3. **Remote abstract class** - Abstraction
4. **BasicRemote** - Refined abstraction

**Example Usage:**

\`\`\`java
Device tv = new TV();	// concrete implementation
Remote remote = new BasicRemote(tv);	// abstraction with implementation
String power = remote.togglePower();	// "Device is ON"
String volUp = remote.volumeUp();	// "Volume: 40"
String volDown = remote.volumeDown();	// "Volume: 30"

Device radio = new Radio();	// different implementation
Remote radioRemote = new BasicRemote(radio);	// same abstraction, different impl
radioRemote.togglePower();	// works the same way
\`\`\``,
	initialCode: `interface Device {
    boolean isEnabled();
    void enable();
    void disable();
    int getVolume();
    void setVolume(int volume);
}

class TV implements Device {
    private boolean on = false;
    private int volume = 30;

    @Override
    public boolean isEnabled() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void enable() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void disable() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public int getVolume() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void setVolume(int volume) { throw new UnsupportedOperationException("TODO"); }
}

class Radio implements Device {
    private boolean on = false;
    private int volume = 20;

    @Override
    public boolean isEnabled() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void enable() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void disable() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public int getVolume() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public void setVolume(int volume) { throw new UnsupportedOperationException("TODO"); }
}

abstract class Remote {
    protected Device device;

    public Remote(Device device) {
    }

    public abstract String togglePower();
    public abstract String volumeUp();
    public abstract String volumeDown();
}

class BasicRemote extends Remote {
    public BasicRemote(Device device) {
    }

    @Override
    public String togglePower() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public String volumeUp() { throw new UnsupportedOperationException("TODO"); }
    @Override
    public String volumeDown() { throw new UnsupportedOperationException("TODO"); }
}`,
	solutionCode: `interface Device {	// Implementation interface - defines device operations
    boolean isEnabled();	// check if device is on
    void enable();	// turn device on
    void disable();	// turn device off
    int getVolume();	// get current volume
    void setVolume(int volume);	// set volume level
}

class TV implements Device {	// Concrete Implementation - TV device
    private boolean on = false;	// power state
    private int volume = 30;	// default volume

    @Override
    public boolean isEnabled() { return on; }	// return power state
    @Override
    public void enable() { on = true; }	// turn on
    @Override
    public void disable() { on = false; }	// turn off
    @Override
    public int getVolume() { return volume; }	// get volume
    @Override
    public void setVolume(int volume) {	// set volume with bounds
        this.volume = Math.max(0, Math.min(100, volume));	// clamp between 0-100
    }
}

class Radio implements Device {	// Concrete Implementation - Radio device
    private boolean on = false;	// power state
    private int volume = 20;	// different default volume

    @Override
    public boolean isEnabled() { return on; }	// return power state
    @Override
    public void enable() { on = true; }	// turn on
    @Override
    public void disable() { on = false; }	// turn off
    @Override
    public int getVolume() { return volume; }	// get volume
    @Override
    public void setVolume(int volume) {	// set volume with bounds
        this.volume = Math.max(0, Math.min(100, volume));	// clamp between 0-100
    }
}

abstract class Remote {	// Abstraction - remote control operations
    protected Device device;	// bridge to implementation

    public Remote(Device device) {	// inject implementation
        this.device = device;	// store device reference
    }

    public abstract String togglePower();	// toggle device power
    public abstract String volumeUp();	// increase volume
    public abstract String volumeDown();	// decrease volume
}

class BasicRemote extends Remote {	// Refined Abstraction - basic remote
    public BasicRemote(Device device) {	// constructor
        super(device);	// pass to parent
    }

    @Override
    public String togglePower() {	// toggle power implementation
        if (device.isEnabled()) {	// check current state
            device.disable();	// turn off if on
            return "Device is OFF";	// return status
        }
        device.enable();	// turn on if off
        return "Device is ON";	// return status
    }

    @Override
    public String volumeUp() {	// increase volume
        device.setVolume(device.getVolume() + 10);	// add 10 to volume
        return "Volume: " + device.getVolume();	// return new volume
    }

    @Override
    public String volumeDown() {	// decrease volume
        device.setVolume(device.getVolume() - 10);	// subtract 10 from volume
        return "Volume: " + device.getVolume();	// return new volume
    }
}`,
	hint1: `**Device Implementations:**

TV and Radio implement the Device interface with simple state management:

\`\`\`java
class TV implements Device {
    private boolean on = false;	// power state
    private int volume = 30;	// TV starts at volume 30

    @Override
    public boolean isEnabled() { return on; }	// simple getter

    @Override
    public void enable() { on = true; }	// set power on

    @Override
    public void disable() { on = false; }	// set power off

    @Override
    public int getVolume() { return volume; }	// return current volume

    @Override
    public void setVolume(int volume) {
        this.volume = Math.max(0, Math.min(100, volume));	// clamp 0-100
    }
}
\`\`\`

Radio follows the same pattern with different default volume (20).`,
	hint2: `**BasicRemote Implementation:**

The remote delegates all operations to the device through the bridge:

\`\`\`java
class BasicRemote extends Remote {
    public BasicRemote(Device device) {
        super(device);	// store device reference
    }

    @Override
    public String togglePower() {
        if (device.isEnabled()) {	// check if device is on
            device.disable();	// turn off
            return "Device is OFF";
        }
        device.enable();	// turn on
        return "Device is ON";
    }

    @Override
    public String volumeUp() {
        device.setVolume(device.getVolume() + 10);	// increase by 10
        return "Volume: " + device.getVolume();	// return new volume
    }

    @Override
    public String volumeDown() {
        device.setVolume(device.getVolume() - 10);	// decrease by 10
        return "Volume: " + device.getVolume();	// return new volume
    }
}
\`\`\``,
	whyItMatters: `## Why Bridge Exists

Bridge separates abstraction (what you do) from implementation (how it's done). This allows both to evolve independently without affecting each other.

**Problem - Explosion of Classes:**

\`\`\`java
// ❌ Bad: Inheritance-based approach causes class explosion
class TVBasicRemote extends TV { ... }
class TVAdvancedRemote extends TV { ... }
class RadioBasicRemote extends Radio { ... }
class RadioAdvancedRemote extends Radio { ... }
// Adding new device or remote multiplies classes!
// 3 devices × 3 remotes = 9 classes
\`\`\`

**Solution - Bridge Composition:**

\`\`\`java
// ✅ Good: Composition-based Bridge pattern
interface Device { ... }	// implementation interface
class TV implements Device { ... }	// implementation 1
class Radio implements Device { ... }	// implementation 2

abstract class Remote {	// abstraction
    protected Device device;	// bridge to implementation
}
class BasicRemote extends Remote { ... }	// abstraction variant 1
class AdvancedRemote extends Remote { ... }	// abstraction variant 2

// 3 devices + 3 remotes = 6 classes
// Any remote works with any device!
\`\`\`

---

## Real-World Examples

1. **JDBC** - DriverManager (abstraction) bridges to database drivers (implementations)
2. **GUI Frameworks** - Window abstraction bridges to OS-specific rendering
3. **Logging** - SLF4J (abstraction) bridges to Log4j, Logback (implementations)
4. **Drawing** - Shape (abstraction) bridges to rendering engine (implementation)
5. **Messaging** - MessageSender bridges to Email, SMS, Push implementations

---

## Production Pattern: Notification System

\`\`\`java
// Implementation Interface - how messages are sent
interface MessageSender {	// defines sending mechanism
    void sendMessage(String recipient, String subject, String body);	// send message
    String getSenderType();	// identify sender type
}

// Concrete Implementations
class EmailSender implements MessageSender {	// email implementation
    private final String smtpHost;	// SMTP server
    private final String smtpPort;	// SMTP port

    public EmailSender(String host, String port) {	// constructor with config
        this.smtpHost = host;	// store host
        this.smtpPort = port;	// store port
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Email sending implementation via SMTP
        System.out.printf("Email to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Email"; }	// sender identifier
}

class SMSSender implements MessageSender {	// SMS implementation
    private final String apiKey;	// SMS gateway API key
    private final String gatewayUrl;	// SMS gateway URL

    public SMSSender(String apiKey, String gatewayUrl) {	// constructor
        this.apiKey = apiKey;	// store API key
        this.gatewayUrl = gatewayUrl;	// store gateway URL
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // SMS sending implementation via API
        System.out.printf("SMS to %s: %s%n", recipient, body);	// SMS has no subject
    }

    @Override
    public String getSenderType() { return "SMS"; }	// sender identifier
}

class PushNotificationSender implements MessageSender {	// push notification
    private final String firebaseKey;	// Firebase Cloud Messaging key

    public PushNotificationSender(String firebaseKey) {	// constructor
        this.firebaseKey = firebaseKey;	// store key
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Push notification via Firebase
        System.out.printf("Push to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Push"; }	// sender identifier
}

// Abstraction - types of notifications
abstract class Notification {	// notification abstraction
    protected MessageSender sender;	// bridge to implementation

    public Notification(MessageSender sender) {	// inject implementation
        this.sender = sender;	// store reference
    }

    public abstract void send(String recipient, String message);	// send notification
    public abstract String getNotificationType();	// notification type
}

// Refined Abstractions - different notification types
class AlertNotification extends Notification {	// urgent alerts
    public AlertNotification(MessageSender sender) {	// constructor
        super(sender);	// pass to parent
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "🚨 ALERT: Immediate Action Required";	// urgent subject
        String body = "[URGENT] " + message;	// mark as urgent
        sender.sendMessage(recipient, subject, body);	// delegate to sender
    }

    @Override
    public String getNotificationType() { return "Alert"; }
}

class ReminderNotification extends Notification {	// reminder notifications
    public ReminderNotification(MessageSender sender) {	// constructor
        super(sender);	// pass to parent
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "📝 Reminder";	// friendly subject
        String body = "Don't forget: " + message;	// reminder format
        sender.sendMessage(recipient, subject, body);	// delegate to sender
    }

    @Override
    public String getNotificationType() { return "Reminder"; }
}

class MarketingNotification extends Notification {	// marketing messages
    private final String unsubscribeLink;	// required for marketing

    public MarketingNotification(MessageSender sender, String unsubscribeLink) {
        super(sender);	// pass to parent
        this.unsubscribeLink = unsubscribeLink;	// store link
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "✨ Special Offer Just For You!";	// marketing subject
        String body = message + "\n\nUnsubscribe: " + unsubscribeLink;	// add unsubscribe
        sender.sendMessage(recipient, subject, body);	// delegate to sender
    }

    @Override
    public String getNotificationType() { return "Marketing"; }
}

// Usage - any notification type with any sender
class NotificationService {	// uses bridge pattern
    public void sendUrgentAlert(String userId, String message) {
        // Send alert via multiple channels
        MessageSender email = new EmailSender("smtp.example.com", "587");
        MessageSender sms = new SMSSender("api-key", "https://sms.api.com");

        Notification emailAlert = new AlertNotification(email);	// alert via email
        Notification smsAlert = new AlertNotification(sms);	// alert via SMS

        emailAlert.send(userId + "@email.com", message);	// send email alert
        smsAlert.send("+1234567890", message);	// send SMS alert
    }

    public void sendMarketingCampaign(List<String> userIds, String offer) {
        MessageSender push = new PushNotificationSender("firebase-key");
        Notification marketing = new MarketingNotification(push, "https://unsubscribe.link");

        for (String userId : userIds) {	// send to all users
            marketing.send(userId, offer);	// each via push notification
        }
    }
}
\`\`\`

---

## Bridge vs Adapter

\`\`\`java
// Adapter: Makes incompatible interfaces work together
// - Applied AFTER classes are designed
// - Wraps existing interface

// Bridge: Separates abstraction from implementation
// - Applied BEFORE classes are designed
// - Both sides can vary independently
\`\`\`

---

## Common Mistakes to Avoid

1. **Tight coupling** - Abstraction shouldn't know concrete implementations
2. **Breaking the bridge** - Don't bypass the implementation interface
3. **Wrong pattern choice** - Use Adapter for compatibility, Bridge for flexibility
4. **Over-engineering** - Don't use Bridge when simple inheritance suffices`,
	order: 1,
	testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Test1 {
    @Test
    void tvStartsDisabled() {
        Device tv = new TV();
        assertFalse(tv.isEnabled(), "TV should start disabled");
    }
}

class Test2 {
    @Test
    void tvEnableAndDisable() {
        Device tv = new TV();
        tv.enable();
        assertTrue(tv.isEnabled(), "TV should be enabled after enable()");
        tv.disable();
        assertFalse(tv.isEnabled(), "TV should be disabled after disable()");
    }
}

class Test3 {
    @Test
    void tvDefaultVolume() {
        Device tv = new TV();
        assertEquals(30, tv.getVolume(), "TV should start with volume 30");
    }
}

class Test4 {
    @Test
    void radioDefaultVolume() {
        Device radio = new Radio();
        assertEquals(20, radio.getVolume(), "Radio should start with volume 20");
    }
}

class Test5 {
    @Test
    void deviceVolumeClamping() {
        Device tv = new TV();
        tv.setVolume(150);
        assertEquals(100, tv.getVolume(), "Volume should be clamped to 100");
        tv.setVolume(-10);
        assertEquals(0, tv.getVolume(), "Volume should be clamped to 0");
    }
}

class Test6 {
    @Test
    void basicRemoteTogglePowerOn() {
        Device tv = new TV();
        Remote remote = new BasicRemote(tv);
        String result = remote.togglePower();
        assertEquals("Device is ON", result, "Should turn device ON");
        assertTrue(tv.isEnabled(), "Device should be enabled");
    }
}

class Test7 {
    @Test
    void basicRemoteTogglePowerOff() {
        Device tv = new TV();
        tv.enable();
        Remote remote = new BasicRemote(tv);
        String result = remote.togglePower();
        assertEquals("Device is OFF", result, "Should turn device OFF");
        assertFalse(tv.isEnabled(), "Device should be disabled");
    }
}

class Test8 {
    @Test
    void basicRemoteVolumeUp() {
        Device tv = new TV();
        Remote remote = new BasicRemote(tv);
        String result = remote.volumeUp();
        assertEquals("Volume: 40", result, "Volume should increase to 40");
    }
}

class Test9 {
    @Test
    void basicRemoteVolumeDown() {
        Device tv = new TV();
        Remote remote = new BasicRemote(tv);
        String result = remote.volumeDown();
        assertEquals("Volume: 20", result, "Volume should decrease to 20");
    }
}

class Test10 {
    @Test
    void remoteWorksSameWithDifferentDevices() {
        Device radio = new Radio();
        Remote remote = new BasicRemote(radio);
        String result = remote.volumeUp();
        assertEquals("Volume: 30", result, "Radio volume should increase from 20 to 30");
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Bridge (Мост)',
			description: `Реализуйте паттерн Bridge на Java — отделите абстракцию от реализации.

**Вы реализуете:**

1. **Интерфейс Device** - Интерфейс реализации
2. **TV, Radio** - Конкретные реализации
3. **Абстрактный класс Remote** - Абстракция
4. **BasicRemote** - Уточнённая абстракция

**Пример использования:**

\`\`\`java
Device tv = new TV();	// конкретная реализация
Remote remote = new BasicRemote(tv);	// абстракция с реализацией
String power = remote.togglePower();	// "Device is ON"
String volUp = remote.volumeUp();	// "Volume: 40"
String volDown = remote.volumeDown();	// "Volume: 30"

Device radio = new Radio();	// другая реализация
Remote radioRemote = new BasicRemote(radio);	// та же абстракция, другая реализация
radioRemote.togglePower();	// работает так же
\`\`\``,
			hint1: `**Реализации Device:**

TV и Radio реализуют интерфейс Device с простым управлением состоянием:

\`\`\`java
class TV implements Device {
    private boolean on = false;	// состояние питания
    private int volume = 30;	// TV начинает с громкости 30

    @Override
    public boolean isEnabled() { return on; }	// простой геттер

    @Override
    public void enable() { on = true; }	// включить питание

    @Override
    public void disable() { on = false; }	// выключить питание

    @Override
    public int getVolume() { return volume; }	// вернуть текущую громкость

    @Override
    public void setVolume(int volume) {
        this.volume = Math.max(0, Math.min(100, volume));	// ограничить 0-100
    }
}
\`\`\`

Radio следует той же схеме с другой громкостью по умолчанию (20).`,
			hint2: `**Реализация BasicRemote:**

Пульт делегирует все операции устройству через мост:

\`\`\`java
class BasicRemote extends Remote {
    public BasicRemote(Device device) {
        super(device);	// сохраняем ссылку на устройство
    }

    @Override
    public String togglePower() {
        if (device.isEnabled()) {	// проверяем включено ли устройство
            device.disable();	// выключаем
            return "Device is OFF";
        }
        device.enable();	// включаем
        return "Device is ON";
    }

    @Override
    public String volumeUp() {
        device.setVolume(device.getVolume() + 10);	// увеличиваем на 10
        return "Volume: " + device.getVolume();	// возвращаем новую громкость
    }

    @Override
    public String volumeDown() {
        device.setVolume(device.getVolume() - 10);	// уменьшаем на 10
        return "Volume: " + device.getVolume();	// возвращаем новую громкость
    }
}
\`\`\``,
			whyItMatters: `## Зачем нужен Bridge

Bridge разделяет абстракцию (что вы делаете) от реализации (как это делается). Это позволяет им развиваться независимо друг от друга.

**Проблема - Взрыв классов:**

\`\`\`java
// ❌ Плохо: Подход на наследовании вызывает взрыв классов
class TVBasicRemote extends TV { ... }
class TVAdvancedRemote extends TV { ... }
class RadioBasicRemote extends Radio { ... }
class RadioAdvancedRemote extends Radio { ... }
// Добавление нового устройства или пульта умножает классы!
// 3 устройства × 3 пульта = 9 классов
\`\`\`

**Решение - Композиция через Bridge:**

\`\`\`java
// ✅ Хорошо: Паттерн Bridge на композиции
interface Device { ... }	// интерфейс реализации
class TV implements Device { ... }	// реализация 1
class Radio implements Device { ... }	// реализация 2

abstract class Remote {	// абстракция
    protected Device device;	// мост к реализации
}
class BasicRemote extends Remote { ... }	// вариант абстракции 1
class AdvancedRemote extends Remote { ... }	// вариант абстракции 2

// 3 устройства + 3 пульта = 6 классов
// Любой пульт работает с любым устройством!
\`\`\`

---

## Примеры из реального мира

1. **JDBC** - DriverManager (абстракция) мостит к драйверам БД (реализации)
2. **GUI фреймворки** - Абстракция окна мостит к OS-специфичному рендерингу
3. **Логирование** - SLF4J (абстракция) мостит к Log4j, Logback (реализации)
4. **Рисование** - Shape (абстракция) мостит к движку рендеринга (реализация)
5. **Сообщения** - MessageSender мостит к Email, SMS, Push реализациям

---

## Production паттерн: Система уведомлений

\`\`\`java
// Интерфейс реализации - как отправляются сообщения
interface MessageSender {	// определяет механизм отправки
    void sendMessage(String recipient, String subject, String body);	// отправить сообщение
    String getSenderType();	// идентифицировать тип отправителя
}

// Конкретные реализации
class EmailSender implements MessageSender {	// email реализация
    private final String smtpHost;	// SMTP сервер
    private final String smtpPort;	// SMTP порт

    public EmailSender(String host, String port) {	// конструктор с конфигурацией
        this.smtpHost = host;	// сохраняем хост
        this.smtpPort = port;	// сохраняем порт
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Реализация отправки email через SMTP
        System.out.printf("Email to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Email"; }	// идентификатор отправителя
}

class SMSSender implements MessageSender {	// SMS реализация
    private final String apiKey;	// API ключ SMS шлюза
    private final String gatewayUrl;	// URL SMS шлюза

    public SMSSender(String apiKey, String gatewayUrl) {	// конструктор
        this.apiKey = apiKey;	// сохраняем API ключ
        this.gatewayUrl = gatewayUrl;	// сохраняем URL шлюза
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Реализация отправки SMS через API
        System.out.printf("SMS to %s: %s%n", recipient, body);	// SMS без темы
    }

    @Override
    public String getSenderType() { return "SMS"; }	// идентификатор отправителя
}

class PushNotificationSender implements MessageSender {	// push уведомление
    private final String firebaseKey;	// ключ Firebase Cloud Messaging

    public PushNotificationSender(String firebaseKey) {	// конструктор
        this.firebaseKey = firebaseKey;	// сохраняем ключ
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Push уведомление через Firebase
        System.out.printf("Push to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Push"; }	// идентификатор отправителя
}

// Абстракция - типы уведомлений
abstract class Notification {	// абстракция уведомления
    protected MessageSender sender;	// мост к реализации

    public Notification(MessageSender sender) {	// внедряем реализацию
        this.sender = sender;	// сохраняем ссылку
    }

    public abstract void send(String recipient, String message);	// отправить уведомление
    public abstract String getNotificationType();	// тип уведомления
}

// Уточнённые абстракции - разные типы уведомлений
class AlertNotification extends Notification {	// срочные оповещения
    public AlertNotification(MessageSender sender) {	// конструктор
        super(sender);	// передаём родителю
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "🚨 ALERT: Immediate Action Required";	// срочная тема
        String body = "[URGENT] " + message;	// помечаем как срочное
        sender.sendMessage(recipient, subject, body);	// делегируем отправителю
    }

    @Override
    public String getNotificationType() { return "Alert"; }
}

class ReminderNotification extends Notification {	// уведомления-напоминания
    public ReminderNotification(MessageSender sender) {	// конструктор
        super(sender);	// передаём родителю
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "📝 Reminder";	// дружелюбная тема
        String body = "Don't forget: " + message;	// формат напоминания
        sender.sendMessage(recipient, subject, body);	// делегируем отправителю
    }

    @Override
    public String getNotificationType() { return "Reminder"; }
}

class MarketingNotification extends Notification {	// маркетинговые сообщения
    private final String unsubscribeLink;	// обязательно для маркетинга

    public MarketingNotification(MessageSender sender, String unsubscribeLink) {
        super(sender);	// передаём родителю
        this.unsubscribeLink = unsubscribeLink;	// сохраняем ссылку
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "✨ Special Offer Just For You!";	// маркетинговая тема
        String body = message + "\n\nUnsubscribe: " + unsubscribeLink;	// добавляем отписку
        sender.sendMessage(recipient, subject, body);	// делегируем отправителю
    }

    @Override
    public String getNotificationType() { return "Marketing"; }
}

// Использование - любой тип уведомления с любым отправителем
class NotificationService {	// использует паттерн bridge
    public void sendUrgentAlert(String userId, String message) {
        // Отправляем оповещение через несколько каналов
        MessageSender email = new EmailSender("smtp.example.com", "587");
        MessageSender sms = new SMSSender("api-key", "https://sms.api.com");

        Notification emailAlert = new AlertNotification(email);	// оповещение через email
        Notification smsAlert = new AlertNotification(sms);	// оповещение через SMS

        emailAlert.send(userId + "@email.com", message);	// отправляем email оповещение
        smsAlert.send("+1234567890", message);	// отправляем SMS оповещение
    }

    public void sendMarketingCampaign(List<String> userIds, String offer) {
        MessageSender push = new PushNotificationSender("firebase-key");
        Notification marketing = new MarketingNotification(push, "https://unsubscribe.link");

        for (String userId : userIds) {	// отправляем всем пользователям
            marketing.send(userId, offer);	// каждому через push уведомление
        }
    }
}
\`\`\`

---

## Bridge vs Adapter

\`\`\`java
// Adapter: Делает несовместимые интерфейсы совместимыми
// - Применяется ПОСЛЕ проектирования классов
// - Оборачивает существующий интерфейс

// Bridge: Разделяет абстракцию и реализацию
// - Применяется ДО проектирования классов
// - Обе стороны могут меняться независимо
\`\`\`

---

## Частые ошибки, которых следует избегать

1. **Жёсткая связанность** - Абстракция не должна знать о конкретных реализациях
2. **Нарушение моста** - Не обходите интерфейс реализации
3. **Неправильный выбор паттерна** - Используйте Adapter для совместимости, Bridge для гибкости
4. **Избыточное проектирование** - Не используйте Bridge когда достаточно простого наследования`
		},
		uz: {
			title: 'Bridge (Ko\'prik) Pattern',
			description: `Java da Bridge patternini amalga oshiring — abstraktsiyani realizatsiyadan ajrating.

**Siz amalga oshirasiz:**

1. **Device interfeysi** - Realizatsiya interfeysi
2. **TV, Radio** - Konkret realizatsiyalar
3. **Remote abstrakt klassi** - Abstraktsiya
4. **BasicRemote** - Yaxshilangan abstraktsiya

**Foydalanish misoli:**

\`\`\`java
Device tv = new TV();	// konkret realizatsiya
Remote remote = new BasicRemote(tv);	// realizatsiyali abstraktsiya
String power = remote.togglePower();	// "Device is ON"
String volUp = remote.volumeUp();	// "Volume: 40"
String volDown = remote.volumeDown();	// "Volume: 30"

Device radio = new Radio();	// boshqa realizatsiya
Remote radioRemote = new BasicRemote(radio);	// bir xil abstraktsiya, boshqa realizatsiya
radioRemote.togglePower();	// xuddi shunday ishlaydi
\`\`\``,
			hint1: `**Device realizatsiyalari:**

TV va Radio Device interfeysini oddiy holat boshqaruvi bilan amalga oshiradi:

\`\`\`java
class TV implements Device {
    private boolean on = false;	// quvvat holati
    private int volume = 30;	// TV 30 ovoz balandligidan boshlanadi

    @Override
    public boolean isEnabled() { return on; }	// oddiy getter

    @Override
    public void enable() { on = true; }	// quvvatni yoqish

    @Override
    public void disable() { on = false; }	// quvvatni o'chirish

    @Override
    public int getVolume() { return volume; }	// joriy ovoz balandligini qaytarish

    @Override
    public void setVolume(int volume) {
        this.volume = Math.max(0, Math.min(100, volume));	// 0-100 oralig'ida cheklash
    }
}
\`\`\`

Radio boshqa standart ovoz balandligi (20) bilan xuddi shunday sxemaga amal qiladi.`,
			hint2: `**BasicRemote realizatsiyasi:**

Pult barcha operatsiyalarni ko'prik orqali qurilmaga delegatsiya qiladi:

\`\`\`java
class BasicRemote extends Remote {
    public BasicRemote(Device device) {
        super(device);	// qurilma havolasini saqlaymiz
    }

    @Override
    public String togglePower() {
        if (device.isEnabled()) {	// qurilma yoqilganmi tekshiramiz
            device.disable();	// o'chiramiz
            return "Device is OFF";
        }
        device.enable();	// yoqamiz
        return "Device is ON";
    }

    @Override
    public String volumeUp() {
        device.setVolume(device.getVolume() + 10);	// 10 ga oshiramiz
        return "Volume: " + device.getVolume();	// yangi ovoz balandligini qaytaramiz
    }

    @Override
    public String volumeDown() {
        device.setVolume(device.getVolume() - 10);	// 10 ga kamaytiramiz
        return "Volume: " + device.getVolume();	// yangi ovoz balandligini qaytaramiz
    }
}
\`\`\``,
			whyItMatters: `## Bridge nima uchun kerak

Bridge abstraktsiyani (nima qilasiz) realizatsiyadan (qanday qilinadi) ajratadi. Bu ularga bir-biriga ta'sir qilmasdan mustaqil rivojlanish imkonini beradi.

**Muammo - Klasslar portlashi:**

\`\`\`java
// ❌ Yomon: Merosga asoslangan yondashuv klasslar portlashiga olib keladi
class TVBasicRemote extends TV { ... }
class TVAdvancedRemote extends TV { ... }
class RadioBasicRemote extends Radio { ... }
class RadioAdvancedRemote extends Radio { ... }
// Yangi qurilma yoki pult qo'shish klasslarni ko'paytiradi!
// 3 qurilma × 3 pult = 9 klass
\`\`\`

**Yechim - Bridge kompozitsiyasi:**

\`\`\`java
// ✅ Yaxshi: Kompozitsiyaga asoslangan Bridge pattern
interface Device { ... }	// realizatsiya interfeysi
class TV implements Device { ... }	// realizatsiya 1
class Radio implements Device { ... }	// realizatsiya 2

abstract class Remote {	// abstraktsiya
    protected Device device;	// realizatsiyaga ko'prik
}
class BasicRemote extends Remote { ... }	// abstraktsiya varianti 1
class AdvancedRemote extends Remote { ... }	// abstraktsiya varianti 2

// 3 qurilma + 3 pult = 6 klass
// Har qanday pult har qanday qurilma bilan ishlaydi!
\`\`\`

---

## Haqiqiy dunyo misollari

1. **JDBC** - DriverManager (abstraktsiya) DB drayverlarga (realizatsiyalar) ko'prik
2. **GUI freymvorklari** - Oyna abstraktsiyasi OS-ga xos renderingga ko'prik
3. **Loglash** - SLF4J (abstraktsiya) Log4j, Logback ga (realizatsiyalar) ko'prik
4. **Chizish** - Shape (abstraktsiya) rendering dvigateliga (realizatsiya) ko'prik
5. **Xabarlar** - MessageSender Email, SMS, Push realizatsiyalariga ko'prik

---

## Production pattern: Bildirishnomalar tizimi

\`\`\`java
// Realizatsiya interfeysi - xabarlar qanday yuboriladi
interface MessageSender {	// yuborish mexanizmini belgilaydi
    void sendMessage(String recipient, String subject, String body);	// xabar yuborish
    String getSenderType();	// yuboruvchi turini aniqlash
}

// Konkret realizatsiyalar
class EmailSender implements MessageSender {	// email realizatsiyasi
    private final String smtpHost;	// SMTP server
    private final String smtpPort;	// SMTP port

    public EmailSender(String host, String port) {	// konfiguratsiya bilan konstruktor
        this.smtpHost = host;	// hostni saqlaymiz
        this.smtpPort = port;	// portni saqlaymiz
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // SMTP orqali email yuborish realizatsiyasi
        System.out.printf("Email to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Email"; }	// yuboruvchi identifikatori
}

class SMSSender implements MessageSender {	// SMS realizatsiyasi
    private final String apiKey;	// SMS gateway API kaliti
    private final String gatewayUrl;	// SMS gateway URL

    public SMSSender(String apiKey, String gatewayUrl) {	// konstruktor
        this.apiKey = apiKey;	// API kalitini saqlaymiz
        this.gatewayUrl = gatewayUrl;	// gateway URL ni saqlaymiz
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // API orqali SMS yuborish realizatsiyasi
        System.out.printf("SMS to %s: %s%n", recipient, body);	// SMS da mavzu yo'q
    }

    @Override
    public String getSenderType() { return "SMS"; }	// yuboruvchi identifikatori
}

class PushNotificationSender implements MessageSender {	// push bildirishnoma
    private final String firebaseKey;	// Firebase Cloud Messaging kaliti

    public PushNotificationSender(String firebaseKey) {	// konstruktor
        this.firebaseKey = firebaseKey;	// kalitni saqlaymiz
    }

    @Override
    public void sendMessage(String recipient, String subject, String body) {
        // Firebase orqali push bildirishnoma
        System.out.printf("Push to %s: [%s] %s%n", recipient, subject, body);
    }

    @Override
    public String getSenderType() { return "Push"; }	// yuboruvchi identifikatori
}

// Abstraktsiya - bildirishnoma turlari
abstract class Notification {	// bildirishnoma abstraktsiyasi
    protected MessageSender sender;	// realizatsiyaga ko'prik

    public Notification(MessageSender sender) {	// realizatsiyani kiritamiz
        this.sender = sender;	// havolani saqlaymiz
    }

    public abstract void send(String recipient, String message);	// bildirishnoma yuborish
    public abstract String getNotificationType();	// bildirishnoma turi
}

// Yaxshilangan abstraktsiyalar - turli bildirishnoma turlari
class AlertNotification extends Notification {	// shoshilinch ogohlantirishlar
    public AlertNotification(MessageSender sender) {	// konstruktor
        super(sender);	// ota-onaga uzatamiz
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "🚨 ALERT: Immediate Action Required";	// shoshilinch mavzu
        String body = "[URGENT] " + message;	// shoshilinch deb belgilaymiz
        sender.sendMessage(recipient, subject, body);	// yuboruvchiga delegatsiya
    }

    @Override
    public String getNotificationType() { return "Alert"; }
}

class ReminderNotification extends Notification {	// eslatma bildirishnomalari
    public ReminderNotification(MessageSender sender) {	// konstruktor
        super(sender);	// ota-onaga uzatamiz
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "📝 Reminder";	// do'stona mavzu
        String body = "Don't forget: " + message;	// eslatma formati
        sender.sendMessage(recipient, subject, body);	// yuboruvchiga delegatsiya
    }

    @Override
    public String getNotificationType() { return "Reminder"; }
}

class MarketingNotification extends Notification {	// marketing xabarlari
    private final String unsubscribeLink;	// marketing uchun majburiy

    public MarketingNotification(MessageSender sender, String unsubscribeLink) {
        super(sender);	// ota-onaga uzatamiz
        this.unsubscribeLink = unsubscribeLink;	// havolani saqlaymiz
    }

    @Override
    public void send(String recipient, String message) {
        String subject = "✨ Special Offer Just For You!";	// marketing mavzusi
        String body = message + "\n\nUnsubscribe: " + unsubscribeLink;	// obunadan chiqish qo'shamiz
        sender.sendMessage(recipient, subject, body);	// yuboruvchiga delegatsiya
    }

    @Override
    public String getNotificationType() { return "Marketing"; }
}

// Foydalanish - har qanday bildirishnoma turi har qanday yuboruvchi bilan
class NotificationService {	// bridge pattern dan foydalanadi
    public void sendUrgentAlert(String userId, String message) {
        // Bir nechta kanal orqali ogohlantirish yuboramiz
        MessageSender email = new EmailSender("smtp.example.com", "587");
        MessageSender sms = new SMSSender("api-key", "https://sms.api.com");

        Notification emailAlert = new AlertNotification(email);	// email orqali ogohlantirish
        Notification smsAlert = new AlertNotification(sms);	// SMS orqali ogohlantirish

        emailAlert.send(userId + "@email.com", message);	// email ogohlantirish yuboramiz
        smsAlert.send("+1234567890", message);	// SMS ogohlantirish yuboramiz
    }

    public void sendMarketingCampaign(List<String> userIds, String offer) {
        MessageSender push = new PushNotificationSender("firebase-key");
        Notification marketing = new MarketingNotification(push, "https://unsubscribe.link");

        for (String userId : userIds) {	// barcha foydalanuvchilarga yuboramiz
            marketing.send(userId, offer);	// har biriga push bildirishnoma orqali
        }
    }
}
\`\`\`

---

## Bridge vs Adapter

\`\`\`java
// Adapter: Mos kelmaydigan interfeyslarni birga ishlashga majburlaydi
// - Klasslar loyihalashdan SO'NG qo'llaniladi
// - Mavjud interfeysni o'raydi

// Bridge: Abstraktsiya va realizatsiyani ajratadi
// - Klasslar loyihalashdan OLDIN qo'llaniladi
// - Ikkala tomon mustaqil o'zgarishi mumkin
\`\`\`

---

## Qochish kerak bo'lgan keng tarqalgan xatolar

1. **Qattiq bog'liqlik** - Abstraktsiya konkret realizatsiyalarni bilmasligi kerak
2. **Ko'prikni buzish** - Realizatsiya interfeysini chetlab o'tmang
3. **Noto'g'ri pattern tanlash** - Moslik uchun Adapter, moslashuvchanlik uchun Bridge ishlating
4. **Ortiqcha loyihalash** - Oddiy meros yetarli bo'lganda Bridge ishlatmang`
		}
	}
};

export default task;
