//go:build tinygo

// Pico WH インターホン音検出ファームウェア
//
// MAX9814マイクモジュールからADC入力をサンプリングし、
// detect パッケージでインターホンの「ピンポン」を検出する。
//
// サンプリング方式:
//
//	256サンプル連続収集 → フレーム処理 → 256サンプル連続収集 → ...
//	オーバーラップなし。フレーム内のサンプル連続性を保証するため。
//
// Wi-Fi接続時はドアベル検出をntfyサーバーへ通知する。
// 資格情報は -ldflags "-X main.wifiSSID=... -X main.wifiPass=..." で注入。
package main

import (
	"machine"
	"machine/usb"
	"time"

	"pico-doorbell-detector/detect"

	"github.com/soypat/cyw43439"
)

// USBシリアル番号をフラッシュチップ固有IDから設定する。
// encoding/hexはfmtを引き込みUSBスタックを壊すため、手動で16進変換する。
func init() {
	const hextable = "0123456789abcdef"
	id := machine.DeviceID()
	buf := make([]byte, len(id)*2)
	for i, b := range id {
		buf[i*2] = hextable[b>>4]     // 上位4bit
		buf[i*2+1] = hextable[b&0x0f] // 下位4bit
	}
	usb.Serial = string(buf)
}

const (
	sampleInterval = 125 // サンプリング間隔(μs) ≈ 8kHz

	// ntfy通知メッセージ
	msgDoorbellType1 = "エントランスからのインターホンが鳴りました"
	msgDoorbellType2 = "玄関からのインターホンが鳴りました"
)

// measureSampleRate はADCサンプリングの実効レートを計測する。
// メインループと同じ条件（adc.Get + Sleep）で1024サンプル分の時間を計る。
func measureSampleRate(adc machine.ADC) float64 {
	const calibrationSamples = 1024
	start := time.Now()
	for i := 0; i < calibrationSamples; i++ {
		adc.Get()
		time.Sleep(sampleInterval * time.Microsecond)
	}
	elapsedUs := time.Since(start).Microseconds()
	return float64(calibrationSamples) * 1_000_000.0 / float64(elapsedUs)
}

func main() {
	// ---- ハードウェア初期化 ----

	dev := cyw43439.NewPicoWDevice()
	cfg := cyw43439.DefaultWifiConfig()
	if err := dev.Init(cfg); err != nil {
		println("Device init failed:", err.Error())
	}

	// セットアップ中はLED点灯
	dev.GPIOSet(0, true)

	// シリアルモニター接続待ち
	time.Sleep(5 * time.Second)

	println("mode:", mode, "-> ntfy topic:", ntfyTopic())

	// ---- ネットワーク初期化 ----

	if wifiSSID == "" {
		println("FATAL: No WiFi credentials. Build with -ldflags \"-X main.wifiSSID=... -X main.wifiPass=...\"")
		for {
			time.Sleep(time.Hour)
		}
	}

	stack, rstack, err := setupNetwork(dev)
	if err != nil {
		println("FATAL: Network setup failed:", err.Error())
		for {
			time.Sleep(time.Hour)
		}
	}
	println("Network ready")

	// ---- ADC初期化 ----

	machine.InitADC()
	adc := machine.ADC{Pin: machine.ADC0}
	adc.Configure(machine.ADCConfig{})

	// サンプルレート実測 → Detector生成
	measuredRate := measureSampleRate(adc)
	println("Measured sample rate:", int(measuredRate), "Hz")

	detector := detect.NewDetector(measuredRate)

	println("=== doorbell detector started ===")

	// セットアップ完了: 5回点滅→消灯
	for i := 0; i < 10; i++ {
		dev.GPIOSet(0, i%2 == 0)
		time.Sleep(200 * time.Millisecond)
	}
	dev.GPIOSet(0, false)

	var frameBuf [detect.FrameSize]float64

	for {
		// 256サンプルを連続収集
		for i := 0; i < detect.FrameSize; i++ {
			frameBuf[i] = float64(adc.Get())
			time.Sleep(sampleInterval * time.Microsecond)
		}

		event, _ := detector.Feed(frameBuf[:])

		switch event {
		case detect.EventType2DingDetected:
			println("DING detected")

		case detect.EventType2Detected:
			println(">>> DOORBELL TYPE 2 DETECTED <<<")
			if err := sendNtfy(stack, rstack, msgDoorbellType2); err != nil {
				println("ntfy failed:", err.Error())
			}
			// LED高速点滅（3秒間）
			for i := 0; i < 30; i++ {
				dev.GPIOSet(0, i%2 == 0)
				time.Sleep(100 * time.Millisecond)
			}
			dev.GPIOSet(0, false)

		case detect.EventType1Detected:
			println(">>> DOORBELL TYPE 1 DETECTED <<<")
			if err := sendNtfy(stack, rstack, msgDoorbellType1); err != nil {
				println("ntfy failed:", err.Error())
			}
			// LED高速点滅（3秒間）
			for i := 0; i < 30; i++ {
				dev.GPIOSet(0, i%2 == 0)
				time.Sleep(100 * time.Millisecond)
			}
			dev.GPIOSet(0, false)

		case detect.EventType2Timeout:
			println("timeout, back to IDLE")
		}
	}
}
