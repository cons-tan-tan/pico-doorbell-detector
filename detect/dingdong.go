package detect

// Type 2: 850Hz持続→662Hz持続の2段階パターン（ピーンポーン）
//
//	[IDLE] → Ding検出(3-of-5) → [DING_DETECTED] → Dong検出(3-of-5) → [DETECTED]
//	                                    |
//	                              タイムアウト(1.5s) → [IDLE]

// ---- スライディングウィンドウ ----

const (
	WindowN = 3 // 最低検出回数
	WindowM = 5 // ウィンドウ幅
)

// ---- Dong検出の時間制約（フレーム数）----
// 1フレーム = 256サンプル ≈ 32ms（8kHz時）

const (
	Type2DongMinFrames = 10 // 0.3s / 0.032s ≈ 10フレーム
	Type2DongMaxFrames = 47 // 1.5s / 0.032s ≈ 47フレーム
)

// ---- 比率ベース閾値（音量非依存）----

const (
	dingRatioThreshold     = 0.25 // 850/RMS
	dongRatioThreshold     = 0.20 // 662/RMS
	harmonicRatioThreshold = 0.03 // 2550/850
)

// ---- 検出状態 ----

const (
	stateIdle         = 0
	stateDingDetected = 1
)

// Type 2のスライディングウィンドウ＋状態遷移処理。
func (d *Detector) feedDingDong(res *FrameResult) Event {
	d.dingWindow[d.frameIdx%WindowM] = res.Ding
	d.dongWindow[d.frameIdx%WindowM] = res.Dong
	d.frameIdx++

	if d.frameIdx < WindowM {
		return EventNone
	}

	dingCount := countWindow(&d.dingWindow)
	dongCount := countWindow(&d.dongWindow)

	switch d.state {
	case stateIdle:
		if dingCount >= WindowN {
			d.state = stateDingDetected
			d.framesSinceDing = 0
			return EventType2DingDetected
		}

	case stateDingDetected:
		d.framesSinceDing++

		if d.framesSinceDing > Type2DongMaxFrames {
			d.state = stateIdle
			return EventType2Timeout
		}
		if d.framesSinceDing >= Type2DongMinFrames && dongCount >= WindowN {
			d.reset()
			return EventType2Detected
		}
	}

	return EventNone
}

func (d *Detector) resetDingDong() {
	d.state = stateIdle
	d.framesSinceDing = 0
	d.frameIdx = 0
	for i := range d.dingWindow {
		d.dingWindow[i] = false
		d.dongWindow[i] = false
	}
}

func countWindow(window *[WindowM]bool) int {
	count := 0
	for _, v := range window {
		if v {
			count++
		}
	}
	return count
}
