package detect

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

const testSampleRate = 8000.0

// ---- ヘルパー関数 ----

// generateSineFrame は指定周波数の正弦波フレーム（FrameSize サンプル）を生成する。
func generateSineFrame(freq, sampleRate, amplitude float64) []float64 {
	frame := make([]float64, FrameSize)
	for i := range frame {
		frame[i] = amplitude * math.Sin(2*math.Pi*freq*float64(i)/sampleRate)
	}
	return frame
}

// generateMultiSineFrame は複数周波数を重ね合わせたフレームを生成する。
func generateMultiSineFrame(freqs, amps []float64, sampleRate float64) []float64 {
	frame := make([]float64, FrameSize)
	for j, freq := range freqs {
		for i := range frame {
			frame[i] += amps[j] * math.Sin(2*math.Pi*freq*float64(i)/sampleRate)
		}
	}
	return frame
}

// silenceFrame は無音フレームを返す。
func silenceFrame() []float64 {
	return make([]float64, FrameSize)
}

// readTestWAV はテスト用にWAVファイルを読み込み、サンプルレートと正規化サンプル列を返す。
func readTestWAV(t *testing.T, path string) (int, []float64) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read WAV: %v", err)
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		t.Fatal("invalid WAV file")
	}

	numChannels := int(binary.LittleEndian.Uint16(data[22:24]))
	sampleRate := int(binary.LittleEndian.Uint32(data[24:28]))
	bitsPerSample := int(binary.LittleEndian.Uint16(data[34:36]))
	if numChannels != 1 || bitsPerSample != 16 {
		t.Fatalf("only mono 16-bit PCM supported (got %dch %dbit)", numChannels, bitsPerSample)
	}

	offset := 12
	for offset < len(data)-8 {
		id := string(data[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		if id == "data" {
			raw := data[offset+8:]
			n := min(size/2, len(raw)/2)
			samples := make([]float64, n)
			for i := range n {
				s := int16(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
				samples[i] = float64(s) / 32768.0
			}
			return sampleRate, samples
		}
		offset += 8 + size
	}

	t.Fatal("data chunk not found")
	return 0, nil
}

// feedAllFrames は全サンプルを FrameSize ごとに Feed し、EventNone 以外のイベントを収集する。
func feedAllFrames(d *Detector, samples []float64) []Event {
	var events []Event
	for i := 0; i+FrameSize <= len(samples); i += FrameSize {
		frame := make([]float64, FrameSize)
		copy(frame, samples[i:i+FrameSize])
		event, _ := d.Feed(frame)
		if event != EventNone {
			events = append(events, event)
		}
	}
	return events
}

// ---- A. RemoveDC ----

func TestRemoveDC_ConstantOffset(t *testing.T) {
	frame := make([]float64, FrameSize)
	for i := range frame {
		frame[i] = 0.5
	}
	RemoveDC(frame)
	for i, v := range frame {
		if math.Abs(v) > 1e-12 {
			t.Fatalf("frame[%d] = %v, want 0.0", i, v)
		}
	}
}

func TestRemoveDC_SineWave(t *testing.T) {
	dc := 0.3
	frame := generateSineFrame(850, testSampleRate, 0.5)
	for i := range frame {
		frame[i] += dc
	}
	RemoveDC(frame)

	var sum float64
	for _, v := range frame {
		sum += v
	}
	mean := sum / float64(len(frame))
	if math.Abs(mean) > 1e-12 {
		t.Fatalf("mean after RemoveDC = %v, want ~0.0", mean)
	}
}

// ---- B. AnalyzeFrame ----

func TestAnalyzeFrame_Silence(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	frame := silenceFrame()
	res := AnalyzeFrame(frame, &coeffs)

	if res.Ding {
		t.Error("Ding=true on silence")
	}
	if res.Dong {
		t.Error("Dong=true on silence")
	}
	if res.RMS > noiseFloor {
		t.Errorf("RMS = %v, want < %v", res.RMS, noiseFloor)
	}
}

func TestAnalyzeFrame_PureDing(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	// 850Hz（基本）+ 2550Hz（第3倍音）、1700Hz（第2倍音）は含めない
	frame := generateMultiSineFrame(
		[]float64{freqDingBase, freqDingHarm3},
		[]float64{0.5, 0.08},
		testSampleRate,
	)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if !res.Ding {
		t.Errorf("Ding=false, want true (R850=%.4f, RHarm=%.4f, RHarm2=%.4f)",
			res.R850, res.RHarm, res.RHarm2)
	}
	if res.Dong {
		t.Error("Dong=true, want false")
	}
}

func TestAnalyzeFrame_PureDong(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	frame := generateSineFrame(freqDongBase, testSampleRate, 0.5)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if !res.Dong {
		t.Errorf("Dong=false, want true (R662=%.4f)", res.R662)
	}
	if res.Ding {
		t.Error("Ding=true, want false")
	}
}

func TestAnalyzeFrame_DingWithoutHarmonic(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	// 850Hzのみ、第3倍音（2550Hz）なし
	frame := generateSineFrame(freqDingBase, testSampleRate, 0.5)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if res.Ding {
		t.Errorf("Ding=true without harmonic (RHarm=%.4f)", res.RHarm)
	}
}

func TestAnalyzeFrame_UnrelatedFrequency(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	frame := generateSineFrame(440, testSampleRate, 0.5)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if res.Ding {
		t.Error("Ding=true on 440Hz")
	}
	if res.Dong {
		t.Error("Dong=true on 440Hz")
	}
}

// ---- C. Type 2 Detector 状態遷移 ----

// makeDingFrame はDetector.Feedに渡すDingフレームを生成する。
// Feed内部でRemoveDCが呼ばれるため、DC除去は不要。
func makeDingFrame() []float64 {
	return generateMultiSineFrame(
		[]float64{freqDingBase, freqDingHarm3},
		[]float64{0.5, 0.08},
		testSampleRate,
	)
}

// makeDongFrame はDetector.Feedに渡すDongフレームを生成する。
func makeDongFrame() []float64 {
	return generateSineFrame(freqDongBase, testSampleRate, 0.5)
}

func TestDetector_Type2FullSequence(t *testing.T) {
	d := NewDetector(testSampleRate)

	// Ding フレームを WindowM 回 Feed → EventType2DingDetected
	var dingDetected bool
	for range WindowM {
		event, _ := d.Feed(makeDingFrame())
		if event == EventType2DingDetected {
			dingDetected = true
		}
	}
	if !dingDetected {
		t.Fatal("expected EventType2DingDetected")
	}

	// 無音で Type2DongMinFrames まで待つ
	for range Type2DongMinFrames {
		event, _ := d.Feed(silenceFrame())
		if event == EventType2Detected {
			t.Fatal("EventType2Detected fired before dong")
		}
	}

	// Dong フレームを WindowM 回 Feed → EventType2Detected
	var doorbellDetected bool
	for range WindowM {
		event, _ := d.Feed(makeDongFrame())
		if event == EventType2Detected {
			doorbellDetected = true
		}
	}
	if !doorbellDetected {
		t.Fatal("expected EventType2Detected")
	}
}

func TestDetector_Type2Timeout(t *testing.T) {
	d := NewDetector(testSampleRate)

	// Ding 検出
	for range WindowM {
		d.Feed(makeDingFrame())
	}

	// Type2DongMaxFrames + 1 フレーム無音 → タイムアウト
	var timedOut bool
	for range Type2DongMaxFrames + 1 {
		event, _ := d.Feed(silenceFrame())
		if event == EventType2Timeout {
			timedOut = true
			break
		}
	}
	if !timedOut {
		t.Fatal("expected EventType2Timeout")
	}

	// タイムアウト後にIDLEに戻り、再度Dingが検出できること
	var dingDetected bool
	for range WindowM {
		event, _ := d.Feed(makeDingFrame())
		if event == EventType2DingDetected {
			dingDetected = true
		}
	}
	if !dingDetected {
		t.Fatal("expected EventType2DingDetected after timeout")
	}
}

func TestDetector_Type2DongTooEarly(t *testing.T) {
	d := NewDetector(testSampleRate)

	// Ding 検出
	for range WindowM {
		d.Feed(makeDingFrame())
	}

	// Type2DongMinFrames 未満で Dong を送る
	for range Type2DongMinFrames - 1 {
		event, _ := d.Feed(makeDongFrame())
		if event == EventType2Detected {
			t.Fatal("EventType2Detected fired before Type2DongMinFrames")
		}
	}
}

func TestDetector_Type2ResetAfterDetection(t *testing.T) {
	d := NewDetector(testSampleRate)

	// 1回目の検出
	for range WindowM {
		d.Feed(makeDingFrame())
	}
	for range Type2DongMinFrames {
		d.Feed(silenceFrame())
	}
	var first bool
	for range WindowM {
		event, _ := d.Feed(makeDongFrame())
		if event == EventType2Detected {
			first = true
		}
	}
	if !first {
		t.Fatal("expected first EventType2Detected")
	}

	// 2回目の検出
	var dingDetected bool
	for range WindowM {
		event, _ := d.Feed(makeDingFrame())
		if event == EventType2DingDetected {
			dingDetected = true
		}
	}
	if !dingDetected {
		t.Fatal("expected EventType2DingDetected after reset")
	}

	for range Type2DongMinFrames {
		d.Feed(silenceFrame())
	}
	var second bool
	for range WindowM {
		event, _ := d.Feed(makeDongFrame())
		if event == EventType2Detected {
			second = true
		}
	}
	if !second {
		t.Fatal("expected second EventType2Detected")
	}
}

// ---- D. Type 2 WAV 統合テスト ----

func TestIntegration_DoorbellType2WAV(t *testing.T) {
	sr, samples := readTestWAV(t, "../testdata/doorbell_type2.wav")
	d := NewDetector(float64(sr))
	events := feedAllFrames(d, samples)

	var hasDing, hasDoorbell bool
	for _, e := range events {
		if e == EventType2DingDetected {
			hasDing = true
		}
		if e == EventType2Detected {
			hasDoorbell = true
		}
	}
	if !hasDing {
		t.Error("doorbell_type2.wav: expected EventType2DingDetected")
	}
	if !hasDoorbell {
		t.Error("doorbell_type2.wav: expected EventType2Detected")
	}
}

func TestIntegration_VoiceWAV(t *testing.T) {
	sr, samples := readTestWAV(t, "../testdata/voice.wav")
	d := NewDetector(float64(sr))
	events := feedAllFrames(d, samples)

	for _, e := range events {
		if e == EventType2Detected {
			t.Error("voice.wav: unexpected EventType2Detected")
		}
	}
}

// ---- E. Type 1 フレーム解析 ----

func TestAnalyzeFrame_HighTone(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	// 850Hzのみ（倍音なし）→ HighTone=true, Ding=false
	frame := generateSineFrame(freqDingBase, testSampleRate, 0.5)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if !res.HighTone {
		t.Errorf("HighTone=false, want true (R850=%.4f, R662=%.4f)", res.R850, res.R662)
	}
	if res.LowTone {
		t.Error("LowTone=true, want false")
	}
	if res.Ding {
		t.Error("Ding=true, want false (no harmonic)")
	}
}

func TestAnalyzeFrame_LowTone(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	frame := generateSineFrame(freqDongBase, testSampleRate, 0.5)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if !res.LowTone {
		t.Errorf("LowTone=false, want true (R662=%.4f, R850=%.4f)", res.R662, res.R850)
	}
	if res.HighTone {
		t.Error("HighTone=true, want false")
	}
}

func TestAnalyzeFrame_BothFrequencies_NeitherTone(t *testing.T) {
	coeffs := NewCoefficients(testSampleRate)
	// 850Hzと662Hzを同振幅で重ねる → どちらも優位でない
	frame := generateMultiSineFrame(
		[]float64{freqDingBase, freqDongBase},
		[]float64{0.5, 0.5},
		testSampleRate,
	)
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &coeffs)

	if res.HighTone {
		t.Errorf("HighTone=true, want false (R850=%.4f, R662=%.4f)", res.R850, res.R662)
	}
	if res.LowTone {
		t.Errorf("LowTone=true, want false (R850=%.4f, R662=%.4f)", res.R850, res.R662)
	}
}

// ---- F. Type 1 Detector 状態遷移 ----

// makeHighToneFrame はType 1の高音トーン（850Hz、倍音なし）フレームを生成する。
func makeHighToneFrame() []float64 {
	return generateSineFrame(freqDingBase, testSampleRate, 0.5)
}

// makeLowToneFrame はType 1の低音トーン（662Hz）フレームを生成する。
func makeLowToneFrame() []float64 {
	return generateSineFrame(freqDongBase, testSampleRate, 0.5)
}

func TestDetector_Type1TrillDetection(t *testing.T) {
	d := NewDetector(testSampleRate)

	feedN := func(makeFrame func() []float64, n int) Event {
		var lastEvent Event
		for range n {
			event, _ := d.Feed(makeFrame())
			if event != EventNone {
				lastEvent = event
			}
		}
		return lastEvent
	}

	// HIGH(5) → LOW(5) → HIGH(5) → LOW(5) → HIGH(5) → LOW(5)
	feedN(makeHighToneFrame, 5)
	feedN(makeLowToneFrame, 5)
	feedN(makeHighToneFrame, 5)
	feedN(makeLowToneFrame, 5)
	feedN(makeHighToneFrame, 5)
	feedN(makeLowToneFrame, 5)

	// 6回目の交替で検出
	var detected bool
	for range 5 {
		event, _ := d.Feed(makeHighToneFrame())
		if event == EventType1Detected {
			detected = true
		}
	}
	if !detected {
		t.Fatal("expected EventType1Detected")
	}
}

func TestDetector_Type1NotEnoughAlternations(t *testing.T) {
	d := NewDetector(testSampleRate)

	// 5回の交替（6未満）→ 検出なし
	feedN := func(makeFrame func() []float64, n int) {
		for range n {
			event, _ := d.Feed(makeFrame())
			if event == EventType1Detected {
				t.Fatal("unexpected EventType1Detected with insufficient alternations")
			}
		}
	}

	feedN(makeHighToneFrame, 5)
	feedN(makeLowToneFrame, 5)
	feedN(makeHighToneFrame, 5)
	feedN(makeLowToneFrame, 5)
	feedN(makeHighToneFrame, 5)
	// 5回の交替のみ → ギャップ挿入でリセット
	feedN(silenceFrame, 5)
}

func TestDetector_Type1GapResets(t *testing.T) {
	d := NewDetector(testSampleRate)

	// 2交替 → 4フレームのギャップ（> trillMaxGapFrames=3）→ リセット
	for range 5 {
		d.Feed(makeHighToneFrame())
	}
	for range 5 {
		d.Feed(makeLowToneFrame())
	}
	for range 5 {
		d.Feed(makeHighToneFrame())
	}

	for range 4 {
		d.Feed(silenceFrame())
	}

	// リセット後に3交替しても合計6に届かない
	for range 5 {
		d.Feed(makeLowToneFrame())
	}
	for range 5 {
		d.Feed(makeHighToneFrame())
	}
	for range 5 {
		d.Feed(makeLowToneFrame())
	}
	for range 5 {
		event, _ := d.Feed(makeHighToneFrame())
		if event == EventType1Detected {
			t.Fatal("unexpected EventType1Detected after gap reset")
		}
	}
}

func TestDetector_Type1ShortToneIgnored(t *testing.T) {
	d := NewDetector(testSampleRate)

	// 各トーンが1フレームのみ → trillMinToneFrames=2を満たさない
	for i := 0; i < 20; i++ {
		var event Event
		if i%2 == 0 {
			event, _ = d.Feed(makeHighToneFrame())
		} else {
			event, _ = d.Feed(makeLowToneFrame())
		}
		if event == EventType1Detected {
			t.Fatal("unexpected EventType1Detected with single-frame tones")
		}
	}
}

func TestDetector_Type1DoesNotTriggerType2(t *testing.T) {
	d := NewDetector(testSampleRate)

	for cycle := 0; cycle < 4; cycle++ {
		for range 5 {
			event, _ := d.Feed(makeHighToneFrame())
			if event == EventType2Detected {
				t.Fatal("unexpected EventType2Detected on Type 1 pattern")
			}
		}
		for range 5 {
			event, _ := d.Feed(makeLowToneFrame())
			if event == EventType2Detected {
				t.Fatal("unexpected EventType2Detected on Type 1 pattern")
			}
		}
	}
}

func TestDetector_Type1ResetAfterDetection(t *testing.T) {
	d := NewDetector(testSampleRate)

	trillSequence := func() bool {
		for range 5 {
			d.Feed(makeHighToneFrame())
		}
		for range 5 {
			d.Feed(makeLowToneFrame())
		}
		for range 5 {
			d.Feed(makeHighToneFrame())
		}
		for range 5 {
			d.Feed(makeLowToneFrame())
		}
		for range 5 {
			d.Feed(makeHighToneFrame())
		}
		for range 5 {
			d.Feed(makeLowToneFrame())
		}
		for range 5 {
			event, _ := d.Feed(makeHighToneFrame())
			if event == EventType1Detected {
				return true
			}
		}
		return false
	}

	if !trillSequence() {
		t.Fatal("expected first EventType1Detected")
	}
	if !trillSequence() {
		t.Fatal("expected second EventType1Detected")
	}
}

// ---- G. Type 1 WAV 統合テスト ----

func TestIntegration_DoorbellType1WAV(t *testing.T) {
	sr, samples := readTestWAV(t, "../testdata/doorbell_type1.wav")
	d := NewDetector(float64(sr))
	events := feedAllFrames(d, samples)

	var hasType1 bool
	for _, e := range events {
		if e == EventType1Detected {
			hasType1 = true
		}
		if e == EventType2Detected {
			t.Error("doorbell_type1.wav: unexpected EventType2Detected")
		}
	}
	if !hasType1 {
		t.Error("doorbell_type1.wav: expected EventType1Detected")
	}
}

func TestIntegration_DoorbellType2WAV_NoType1(t *testing.T) {
	sr, samples := readTestWAV(t, "../testdata/doorbell_type2.wav")
	d := NewDetector(float64(sr))
	events := feedAllFrames(d, samples)

	for _, e := range events {
		if e == EventType1Detected {
			t.Error("doorbell_type2.wav: unexpected EventType1Detected")
		}
	}
}

func TestIntegration_VoiceWAV_NoType1(t *testing.T) {
	sr, samples := readTestWAV(t, "../testdata/voice.wav")
	d := NewDetector(float64(sr))
	events := feedAllFrames(d, samples)

	for _, e := range events {
		if e == EventType1Detected {
			t.Error("voice.wav: unexpected EventType1Detected")
		}
	}
}
