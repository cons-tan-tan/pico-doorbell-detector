// Package detect はインターホンのチャイム音パターンを検出する。
//
// Goertzel周波数解析と比率ベース判定により、音量・距離に依存しない検出を行う。
//
//   - Type 2（ピーンポーン）: dingdong.go
//   - Type 1（ピンポンピンポン）: trill.go
//
// math のみに依存し、TinyGo/標準Go 両対応。
package detect

import "math"

// ---- 周波数定数 ----

const (
	freqDingBase  = 850.0  // Ding基本周波数
	freqDingHarm2 = 1700.0 // 第2倍音（電子音では欠落）
	freqDingHarm3 = 2550.0 // 第3倍音（電子音では存在）
	freqDongBase  = 662.0  // Dong基本周波数
)

// ---- 音声処理パラメータ ----

// FrameSize は1フレームのサンプル数（8kHz時32ms、分解能31.25Hz）。
const FrameSize = 256

const noiseFloor = 0.001 // 無音判定（唯一の絶対閾値）

// ---- Goertzel係数 ----

// Coefficients はサンプルレート別の事前計算済みGoertzel係数。
type Coefficients struct {
	DingBase  float64
	DingHarm2 float64
	DingHarm3 float64
	DongBase  float64
}

// NewCoefficients は指定サンプルレートのGoertzel係数を計算する。
func NewCoefficients(sampleRate float64) Coefficients {
	return Coefficients{
		DingBase:  goertzelCoeff(freqDingBase, sampleRate),
		DingHarm2: goertzelCoeff(freqDingHarm2, sampleRate),
		DingHarm3: goertzelCoeff(freqDingHarm3, sampleRate),
		DongBase:  goertzelCoeff(freqDongBase, sampleRate),
	}
}

func goertzelCoeff(targetFreq, sampleRate float64) float64 {
	k := float64(FrameSize) * targetFreq / sampleRate
	w := 2.0 * math.Pi * k / float64(FrameSize)
	return 2.0 * math.Cos(w)
}

// ---- コア関数 ----

// 事前計算済み係数による周波数強度の算出。
func goertzelMagnitude(samples []float64, coeff float64) float64 {
	var s1, s2 float64
	for _, x := range samples {
		s0 := x + coeff*s1 - s2
		s2 = s1
		s1 = s0
	}
	power := s1*s1 + s2*s2 - coeff*s1*s2
	if power < 0 {
		return 0
	}
	return math.Sqrt(power)
}

// RemoveDC はフレームのDCバイアスを除去する（in-place）。
func RemoveDC(frame []float64) {
	var sum float64
	for _, s := range frame {
		sum += s
	}
	mean := sum / float64(len(frame))
	for i := range frame {
		frame[i] -= mean
	}
}

// ---- フレーム解析 ----

// FrameResult は1フレームの解析結果。
type FrameResult struct {
	Ding     bool    // Ding判定（Type 2用：850Hz＋倍音）
	Dong     bool    // Dong判定（Type 2用：662Hz比率）
	HighTone bool    // 850Hz優位（Type 1用：倍音不要）
	LowTone  bool    // 662Hz優位（Type 1用）
	RMS      float64 // フレームのRMS
	R850     float64 // m850 / (RMS * N)
	R662     float64 // m662 / (RMS * N)
	RHarm    float64 // m2550 / m850（第3倍音比率）
	RHarm2   float64 // m1700 / m850（第2倍音比率）
}

// AnalyzeFrame はDC除去済みフレームの周波数解析と判定を行う。
func AnalyzeFrame(frame []float64, c *Coefficients) FrameResult {
	var res FrameResult

	// RMS計算
	var sum float64
	for _, s := range frame {
		sum += s * s
	}
	rms := math.Sqrt(sum / float64(len(frame)))
	res.RMS = rms

	if rms < noiseFloor {
		return res
	}

	n := float64(len(frame))

	// Goertzelを各周波数に対して実行（計4回）
	m850 := goertzelMagnitude(frame, c.DingBase)
	m2550 := goertzelMagnitude(frame, c.DingHarm3)
	m1700 := goertzelMagnitude(frame, c.DingHarm2)
	m662 := goertzelMagnitude(frame, c.DongBase)

	// 比率の計算
	res.R850 = m850 / (rms * n)
	res.R662 = m662 / (rms * n)
	if m850 > 0 {
		res.RHarm = m2550 / m850
		res.RHarm2 = m1700 / m850
	}

	// Ding判定: 850Hz比率 + 第3倍音の存在 + 第2倍音の不在
	if res.R850 > dingRatioThreshold && m850 > 0 {
		if res.RHarm > harmonicRatioThreshold && m2550 > m1700*1.5 {
			res.Ding = true
		}
	}

	// Dong判定: 662Hz比率
	if res.R662 > dongRatioThreshold {
		res.Dong = true
	}

	// Type 1 トーン判定: 周波数の優位性
	if res.R850 > trillToneRatioMin && res.R850 > res.R662*trillToneDominance {
		res.HighTone = true
	}
	if res.R662 > trillToneRatioMin && res.R662 > res.R850*trillToneDominance {
		res.LowTone = true
	}

	return res
}

// ---- 状態遷移イベント ----

// Event はDetectorの状態遷移イベント。
type Event int

const (
	EventNone              Event = iota
	EventType2DingDetected       // Ding確定（3-of-5）（Type 2）
	EventType2Detected           // Ding+Dong確定 → Type 2 インターホン検出
	EventType2Timeout            // Dong待機タイムアウト → IDLEに戻る（Type 2）
	EventType1Detected           // Type 1 トリルパターン検出
)

// ---- Detector ----

// Detector はフレームごとの検出判定を行うステートマシン。
type Detector struct {
	coeffs Coefficients

	// Type 2（dingdong.go）
	dingWindow      [WindowM]bool
	dongWindow      [WindowM]bool
	frameIdx        int
	state           int
	framesSinceDing int

	// Type 1（trill.go）
	trillCurrentTone    int
	trillToneFrameCount int
	trillAlternations   int
	trillGapFrames      int
}

// NewDetector は指定サンプルレートのDetectorを生成する。
func NewDetector(sampleRate float64) *Detector {
	return &Detector{
		coeffs: NewCoefficients(sampleRate),
	}
}

// Feed はDC除去→解析→状態遷移を実行する。入力frameはin-placeで変更される。
func (d *Detector) Feed(frame []float64) (Event, FrameResult) {
	RemoveDC(frame)
	res := AnalyzeFrame(frame, &d.coeffs)

	// Type 1 トリル検出（Type 2とは独立に処理）
	if evt := d.feedTrill(&res); evt != EventNone {
		d.reset()
		return evt, res
	}

	// Type 2 Ding-Dong検出
	if evt := d.feedDingDong(&res); evt != EventNone {
		return evt, res
	}

	return EventNone, res
}

func (d *Detector) reset() {
	d.resetDingDong()
	d.resetTrill()
}
