import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-isp-advanced',
	title: 'Interface Segregation Principle - Advanced',
	difficulty: 'hard',
	tags: ['go', 'solid', 'isp', 'interfaces', 'advanced'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply ISP to a complex multimedia player system with varying capabilities.

**Current Problem:**

A MediaPlayer interface with all possible methods, forcing implementations to have many unsupported operations.

**Your task:**

Create a granular interface system:

1. **Player interface** - Basic playback (Play, Pause, Stop)
2. **VideoPlayer interface** - Extends Player, adds video controls
3. **AudioPlayer interface** - Extends Player, adds audio controls
4. **StreamablePlayer interface** - Adds streaming capability
5. **MP3Player** - Audio only
6. **VideoFile** - Video and audio
7. **LiveStream** - All capabilities

**Key Concepts:**
- **Minimal interfaces**: Each interface has minimal methods
- **Interface composition**: Build complex interfaces from simple ones
- **Client-specific interfaces**: Different clients see different views

**Real-World Impact:**
- Implement different player types easily
- Each player has appropriate controls
- Type safety prevents unsupported operations
- Clear capability documentation`,
	initialCode: `package principles

import "fmt"

type MediaPlayer interface {
}

type MP3Player struct {
	Volume int
}

func (mp *MP3Player) Play() error {
	return nil
}

func (mp *MP3Player) Pause() error { return nil }
func (mp *MP3Player) Stop() error { return nil }
func (mp *MP3Player) SetVolume(level int) error {
	return nil
}

func (mp *MP3Player) AdjustBrightness(level int) error {
	return fmt.Errorf("audio player doesn't support brightness")
}

func (mp *MP3Player) SetSubtitles(enabled bool) error {
	return fmt.Errorf("audio player doesn't support subtitles")
}

func (mp *MP3Player) SetQuality(quality string) error {
	return nil  // empty implementation
}

func (mp *MP3Player) ToggleFullscreen() error {
	return fmt.Errorf("audio player doesn't support fullscreen")
}

type Player interface {
}

type VolumeControl interface {
}

type VideoControl interface {
}

type SubtitleSupport interface {
}

type QualityControl interface {
}

type AudioPlayer interface {
}

type VideoPlayer interface {
}

type StreamablePlayer interface {
}

type MP3PlayerRefactored struct {
	Volume int
}

func (mp *MP3PlayerRefactored) Play() error {
}

func (mp *MP3PlayerRefactored) Pause() error {
}

func (mp *MP3PlayerRefactored) Stop() error {
}

func (mp *MP3PlayerRefactored) SetVolume(level int) error {
}

type VideoFilePlayer struct {
	Volume     int
	Brightness int
	Fullscreen bool
	Subtitles  bool
}

func (vf *VideoFilePlayer) Play() error {
}

func (vf *VideoFilePlayer) Pause() error {
}

func (vf *VideoFilePlayer) Stop() error {
}

func (vf *VideoFilePlayer) SetVolume(level int) error {
}

func (vf *VideoFilePlayer) AdjustBrightness(level int) error {
}

func (vf *VideoFilePlayer) ToggleFullscreen() error {
}

func (vf *VideoFilePlayer) SetSubtitles(enabled bool) error {
}

type LiveStreamPlayer struct {
	Volume     int
	Brightness int
	Fullscreen bool
	Subtitles  bool
	Quality    string
}

func (ls *LiveStreamPlayer) Play() error {
}

func (ls *LiveStreamPlayer) Pause() error {
}

func (ls *LiveStreamPlayer) Stop() error {
}

func (ls *LiveStreamPlayer) SetVolume(level int) error {
}

func (ls *LiveStreamPlayer) AdjustBrightness(level int) error {
}

func (ls *LiveStreamPlayer) ToggleFullscreen() error {
}

func (ls *LiveStreamPlayer) SetSubtitles(enabled bool) error {
}

func (ls *LiveStreamPlayer) SetQuality(quality string) error {
}

func PlayMedia(player Player) {
}

func AdjustAudio(player VolumeControl, level int) {
}

func ConfigureVideo(player VideoControl, brightness int, fullscreen bool) {
}

func SetStreamQuality(player QualityControl, quality string) {
}`,
	solutionCode: `package principles

import "fmt"

// Player - minimal interface for basic playback
// ISP compliant: only essential playback methods
type Player interface {
	Play() error
	Pause() error
	Stop() error
}

// VolumeControl - separate interface for volume
// ISP compliant: only volume-related methods
type VolumeControl interface {
	SetVolume(level int) error
}

// VideoControl - separate interface for video-specific controls
// ISP compliant: only video-related methods
type VideoControl interface {
	AdjustBrightness(level int) error
	ToggleFullscreen() error
}

// SubtitleSupport - separate interface for subtitles
// ISP compliant: only subtitle-related methods
type SubtitleSupport interface {
	SetSubtitles(enabled bool) error
}

// QualityControl - separate interface for quality adjustment
// ISP compliant: only quality-related methods
type QualityControl interface {
	SetQuality(quality string) error
}

// AudioPlayer - composed interface for audio players
// Combines Player + VolumeControl
type AudioPlayer interface {
	Player
	VolumeControl
}

// VideoPlayer - composed interface for video players
// Combines Player + VolumeControl + VideoControl + SubtitleSupport
type VideoPlayer interface {
	Player
	VolumeControl
	VideoControl
	SubtitleSupport
}

// StreamablePlayer - composed interface for streaming players
// Combines VideoPlayer + QualityControl
type StreamablePlayer interface {
	VideoPlayer
	QualityControl
}

// MP3PlayerRefactored implements only AudioPlayer
// ISP compliant: no forced video methods
type MP3PlayerRefactored struct {
	Volume int
}

func (mp *MP3PlayerRefactored) Play() error {
	fmt.Println("Playing MP3 audio")
	return nil
}

func (mp *MP3PlayerRefactored) Pause() error {
	fmt.Println("Pausing MP3")
	return nil
}

func (mp *MP3PlayerRefactored) Stop() error {
	fmt.Println("Stopping MP3")
	return nil
}

func (mp *MP3PlayerRefactored) SetVolume(level int) error {
	mp.Volume = level
	fmt.Printf("MP3 volume set to %d\\n", level)
	return nil
}

// VideoFilePlayer implements VideoPlayer
// ISP compliant: implements all video-related interfaces
type VideoFilePlayer struct {
	Volume     int
	Brightness int
	Fullscreen bool
	Subtitles  bool
}

func (vf *VideoFilePlayer) Play() error {
	fmt.Println("Playing video file")
	return nil
}

func (vf *VideoFilePlayer) Pause() error {
	fmt.Println("Pausing video")
	return nil
}

func (vf *VideoFilePlayer) Stop() error {
	fmt.Println("Stopping video")
	return nil
}

func (vf *VideoFilePlayer) SetVolume(level int) error {
	vf.Volume = level
	fmt.Printf("Video volume set to %d\\n", level)
	return nil
}

func (vf *VideoFilePlayer) AdjustBrightness(level int) error {
	vf.Brightness = level
	fmt.Printf("Brightness adjusted to %d\\n", level)
	return nil
}

func (vf *VideoFilePlayer) ToggleFullscreen() error {
	vf.Fullscreen = !vf.Fullscreen
	fmt.Printf("Fullscreen: %v\\n", vf.Fullscreen)
	return nil
}

func (vf *VideoFilePlayer) SetSubtitles(enabled bool) error {
	vf.Subtitles = enabled
	fmt.Printf("Subtitles: %v\\n", enabled)
	return nil
}

// LiveStreamPlayer implements StreamablePlayer
// ISP compliant: implements all interfaces including quality control
type LiveStreamPlayer struct {
	Volume     int
	Brightness int
	Fullscreen bool
	Subtitles  bool
	Quality    string
}

func (ls *LiveStreamPlayer) Play() error {
	fmt.Println("Streaming live content")
	return nil
}

func (ls *LiveStreamPlayer) Pause() error {
	fmt.Println("Buffering stream")
	return nil
}

func (ls *LiveStreamPlayer) Stop() error {
	fmt.Println("Ending stream")
	return nil
}

func (ls *LiveStreamPlayer) SetVolume(level int) error {
	ls.Volume = level
	fmt.Printf("Stream volume set to %d\\n", level)
	return nil
}

func (ls *LiveStreamPlayer) AdjustBrightness(level int) error {
	ls.Brightness = level
	fmt.Printf("Stream brightness adjusted to %d\\n", level)
	return nil
}

func (ls *LiveStreamPlayer) ToggleFullscreen() error {
	ls.Fullscreen = !ls.Fullscreen
	fmt.Printf("Stream fullscreen: %v\\n", ls.Fullscreen)
	return nil
}

func (ls *LiveStreamPlayer) SetSubtitles(enabled bool) error {
	ls.Subtitles = enabled
	fmt.Printf("Stream subtitles: %v\\n", enabled)
	return nil
}

func (ls *LiveStreamPlayer) SetQuality(quality string) error {
	ls.Quality = quality
	fmt.Printf("Stream quality set to %s\\n", quality)
	return nil
}

// PlayMedia works with any Player
// ISP compliant: depends only on playback capability
func PlayMedia(player Player) {
	player.Play()
}

// AdjustAudio works with any VolumeControl
// ISP compliant: depends only on volume capability
func AdjustAudio(player VolumeControl, level int) {
	player.SetVolume(level)
}

// ConfigureVideo works with any VideoControl
// ISP compliant: depends only on video controls
func ConfigureVideo(player VideoControl, brightness int, fullscreen bool) {
	player.AdjustBrightness(brightness)
	if fullscreen {
		player.ToggleFullscreen()
	}
}

// SetStreamQuality works with any QualityControl
// ISP compliant: depends only on quality capability
func SetStreamQuality(player QualityControl, quality string) {
	player.SetQuality(quality)
}`,
	hint1: `For each player type, implement the methods by printing appropriate messages and updating struct fields. MP3Player needs Play/Pause/Stop/SetVolume. VideoFilePlayer adds AdjustBrightness/ToggleFullscreen/SetSubtitles. LiveStreamPlayer adds SetQuality.`,
	hint2: `For control functions: PlayMedia calls player.Play(). AdjustAudio calls player.SetVolume(level). ConfigureVideo calls player.AdjustBrightness(brightness) and conditionally calls player.ToggleFullscreen(). SetStreamQuality calls player.SetQuality(quality).`,
	testCode: `package principles

import "testing"

// Test1: MP3Player implements Play
func Test1(t *testing.T) {
	mp := &MP3PlayerRefactored{}
	if err := mp.Play(); err != nil {
		t.Errorf("MP3Player.Play() error: %v", err)
	}
}

// Test2: MP3Player implements SetVolume
func Test2(t *testing.T) {
	mp := &MP3PlayerRefactored{}
	if err := mp.SetVolume(50); err != nil {
		t.Errorf("MP3Player.SetVolume() error: %v", err)
	}
	if mp.Volume != 50 {
		t.Errorf("Volume = %d, want 50", mp.Volume)
	}
}

// Test3: VideoFilePlayer implements all video methods
func Test3(t *testing.T) {
	vf := &VideoFilePlayer{}
	if err := vf.Play(); err != nil {
		t.Errorf("VideoFilePlayer.Play() error: %v", err)
	}
	if err := vf.AdjustBrightness(80); err != nil {
		t.Errorf("AdjustBrightness error: %v", err)
	}
}

// Test4: VideoFilePlayer implements ToggleFullscreen
func Test4(t *testing.T) {
	vf := &VideoFilePlayer{}
	vf.ToggleFullscreen()
	if !vf.Fullscreen {
		t.Error("Fullscreen should be true after toggle")
	}
}

// Test5: LiveStreamPlayer implements SetQuality
func Test5(t *testing.T) {
	ls := &LiveStreamPlayer{}
	if err := ls.SetQuality("1080p"); err != nil {
		t.Errorf("SetQuality error: %v", err)
	}
	if ls.Quality != "1080p" {
		t.Errorf("Quality = %s, want 1080p", ls.Quality)
	}
}

// Test6: PlayMedia works with any Player
func Test6(t *testing.T) {
	PlayMedia(&MP3PlayerRefactored{})
	PlayMedia(&VideoFilePlayer{})
	PlayMedia(&LiveStreamPlayer{})
}

// Test7: AdjustAudio works with VolumeControl
func Test7(t *testing.T) {
	mp := &MP3PlayerRefactored{}
	AdjustAudio(mp, 75)
	if mp.Volume != 75 {
		t.Errorf("Volume = %d, want 75", mp.Volume)
	}
}

// Test8: ConfigureVideo works with VideoControl
func Test8(t *testing.T) {
	vf := &VideoFilePlayer{}
	ConfigureVideo(vf, 100, true)
	if vf.Brightness != 100 {
		t.Errorf("Brightness = %d, want 100", vf.Brightness)
	}
}

// Test9: SetStreamQuality works with QualityControl
func Test9(t *testing.T) {
	ls := &LiveStreamPlayer{}
	SetStreamQuality(ls, "720p")
	if ls.Quality != "720p" {
		t.Errorf("Quality = %s, want 720p", ls.Quality)
	}
}

// Test10: Interface composition works correctly
func Test10(t *testing.T) {
	var ap AudioPlayer = &MP3PlayerRefactored{}
	_ = ap
	var vp VideoPlayer = &VideoFilePlayer{}
	_ = vp
	var sp StreamablePlayer = &LiveStreamPlayer{}
	_ = sp
}
`,
	whyItMatters: `Advanced ISP shows how to build flexible systems with granular interfaces.`,
	order: 7,
	translations: {
		ru: {
			title: 'Принцип разделения интерфейсов - Продвинутый',
			description: `Примените ISP к сложной системе мультимедийного плеера с различными возможностями.`,
			hint1: `Для каждого типа плеера реализуйте методы, которые выводят соответствующие сообщения и обновляют поля структуры. MP3Player нужны Play/Pause/Stop/SetVolume.`,
			hint2: `Для функций управления: PlayMedia вызывает player.Play(). AdjustAudio вызывает player.SetVolume(level). ConfigureVideo вызывает player.AdjustBrightness и условно player.ToggleFullscreen().`,
			whyItMatters: `Продвинутый ISP показывает как строить гибкие системы с детализированными интерфейсами.`,
			solutionCode: `package principles

import "fmt"

type Player interface {
	Play() error
	Pause() error
	Stop() error
}

type VolumeControl interface {
	SetVolume(level int) error
}

type VideoControl interface {
	AdjustBrightness(level int) error
	ToggleFullscreen() error
}

type SubtitleSupport interface {
	SetSubtitles(enabled bool) error
}

type QualityControl interface {
	SetQuality(quality string) error
}

type AudioPlayer interface {
	Player
	VolumeControl
}

type VideoPlayer interface {
	Player
	VolumeControl
	VideoControl
	SubtitleSupport
}

type StreamablePlayer interface {
	VideoPlayer
	QualityControl
}

type MP3PlayerRefactored struct {
	Volume int
}

func (mp *MP3PlayerRefactored) Play() error {
	fmt.Println("Воспроизведение MP3 аудио")
	return nil
}

func (mp *MP3PlayerRefactored) Pause() error {
	fmt.Println("Пауза MP3")
	return nil
}

func (mp *MP3PlayerRefactored) Stop() error {
	fmt.Println("Остановка MP3")
	return nil
}

func (mp *MP3PlayerRefactored) SetVolume(level int) error {
	mp.Volume = level
	fmt.Printf("Громкость MP3 установлена на %d\\n", level)
	return nil
}

func PlayMedia(player Player) {
	player.Play()
}

func AdjustAudio(player VolumeControl, level int) {
	player.SetVolume(level)
}

func ConfigureVideo(player VideoControl, brightness int, fullscreen bool) {
	player.AdjustBrightness(brightness)
	if fullscreen {
		player.ToggleFullscreen()
	}
}

func SetStreamQuality(player QualityControl, quality string) {
	player.SetQuality(quality)
}`
		},
		uz: {
			title: 'Interfeys ajratish printsipi - Kengaytirilgan',
			description: `Turli imkoniyatlarga ega murakkab multimedia pleyer tizimiga ISP ni qo'llang.`,
			hint1: `Har bir pleyer turi uchun tegishli xabarlarni chiqaruvchi va struktura maydonlarini yangilovchi metodlarni amalga oshiring. MP3Player ga Play/Pause/Stop/SetVolume kerak.`,
			hint2: `Boshqaruv funktsiyalari uchun: PlayMedia player.Play() ni chaqiradi. AdjustAudio player.SetVolume(level) ni chaqiradi. ConfigureVideo player.AdjustBrightness va shartli player.ToggleFullscreen() ni chaqiradi.`,
			whyItMatters: `Kengaytirilgan ISP granular interfeyslar bilan moslashuvchan tizimlarni qanday qurishni ko'rsatadi.`,
			solutionCode: `package principles

import "fmt"

type Player interface {
	Play() error
	Pause() error
	Stop() error
}

type VolumeControl interface {
	SetVolume(level int) error
}

type VideoControl interface {
	AdjustBrightness(level int) error
	ToggleFullscreen() error
}

type SubtitleSupport interface {
	SetSubtitles(enabled bool) error
}

type QualityControl interface {
	SetQuality(quality string) error
}

type AudioPlayer interface {
	Player
	VolumeControl
}

type VideoPlayer interface {
	Player
	VolumeControl
	VideoControl
	SubtitleSupport
}

type StreamablePlayer interface {
	VideoPlayer
	QualityControl
}

type MP3PlayerRefactored struct {
	Volume int
}

func (mp *MP3PlayerRefactored) Play() error {
	fmt.Println("MP3 audio ijro etilmoqda")
	return nil
}

func (mp *MP3PlayerRefactored) Pause() error {
	fmt.Println("MP3 to'xtatildi")
	return nil
}

func (mp *MP3PlayerRefactored) Stop() error {
	fmt.Println("MP3 to'xtatildi")
	return nil
}

func (mp *MP3PlayerRefactored) SetVolume(level int) error {
	mp.Volume = level
	fmt.Printf("MP3 ovoz balandligi %d ga o'rnatildi\\n", level)
	return nil
}

func PlayMedia(player Player) {
	player.Play()
}

func AdjustAudio(player VolumeControl, level int) {
	player.SetVolume(level)
}

func ConfigureVideo(player VideoControl, brightness int, fullscreen bool) {
	player.AdjustBrightness(brightness)
	if fullscreen {
		player.ToggleFullscreen()
	}
}

func SetStreamQuality(player QualityControl, quality string) {
	player.SetQuality(quality)
}`
		}
	}
};

export default task;
