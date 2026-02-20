package detect

// Type 2: 850Hz持続→662Hz持続の2段階パターン（ピーンポーン）
//
//	[IDLE] → Ding連続8フレーム → [DING_DETECTED] → Dong検出(3-of-5) → [DETECTED]
//	                                    |
//	                              タイムアウト(1.5s) → [IDLE]

// ---- Dong検出のスライディングウィンドウ ----

const (
	WindowN = 3 // 最低検出回数
	WindowM = 5 // ウィンドウ幅
)

// ---- Ding検出の連続フレーム閾値 ----
//
// Type 1のHIGH区間は4-5フレーム、Type 2のDing区間は15-16フレーム。
// 8フレーム以上の連続でType 2のDingと判定する。

const DingStreakThreshold = 8

// ---- Dong検出の時間制約（フレーム数）----
// 1フレーム = 256サンプル ≈ 32ms（8kHz時）

const (
	Type2DongMinFrames = 10 // 0.3s / 0.032s ≈ 10フレーム
	Type2DongMaxFrames = 47 // 1.5s / 0.032s ≈ 47フレーム
)

// ---- 比率ベース閾値（音量非依存）----

const (
	dingRatioThreshold = 0.25 // 850/RMS
	dongRatioThreshold = 0.20 // 662/RMS
)

// ---- 検出状態 ----

const (
	stateIdle         = 0
	stateDingDetected = 1
)

// Type 2の連続Dingストリーク＋Dongスライディングウィンドウ＋状態遷移処理。
func (d *Detector) feedDingDong(res *FrameResult) Event {
	// Dong用スライディングウィンドウは常に更新
	d.dongWindow[d.dongFrameIdx%WindowM] = res.Dong
	d.dongFrameIdx++

	switch d.state {
	case stateIdle:
		// Ding連続カウント
		if res.Ding {
			d.dingStreak++
		} else {
			d.dingStreak = 0
		}
		if d.dingStreak >= DingStreakThreshold {
			d.state = stateDingDetected
			d.framesSinceDing = 0
			return EventType2DingDetected
		}

	case stateDingDetected:
		d.framesSinceDing++

		if d.framesSinceDing > Type2DongMaxFrames {
			d.state = stateIdle
			d.dingStreak = 0
			return EventType2Timeout
		}

		dongCount := countWindow(&d.dongWindow)
		if d.dongFrameIdx >= WindowM && d.framesSinceDing >= Type2DongMinFrames && dongCount >= WindowN {
			d.reset()
			return EventType2Detected
		}
	}

	return EventNone
}

func (d *Detector) resetDingDong() {
	d.state = stateIdle
	d.framesSinceDing = 0
	d.dingStreak = 0
	d.dongFrameIdx = 0
	for i := range d.dongWindow {
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
