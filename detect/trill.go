package detect

// Type 1: 850Hz⇔662Hzの高速交互トリルパターン（ピンポンピンポン）
//
// フレームごとにトーン分類(HIGH/LOW) → 交替回数カウント → 6回で検出

// ---- トリル検出パラメータ ----

const (
	trillToneRatioMin    = 0.30 // HIGH/LOW判定の最小比率閾値
	trillToneDominance   = 1.5  // 周波数優位性の倍率（R850 > R662*1.5 → HIGH）
	trillMinToneFrames   = 2    // トーン切替認識の最小フレーム数
	trillMinAlternations = 6    // 検出に必要な最小交替回数（3サイクル）
	trillMaxGapFrames    = 3    // パターン内の最大許容ギャップ
)

// ---- トーン種別 ----

const (
	toneNone = 0
	toneHigh = 1 // 850Hz優位
	toneLow  = 2 // 662Hz優位
)

// Type 1トリルパターンの状態遷移処理。
func (d *Detector) feedTrill(res *FrameResult) Event {
	tone := toneNone
	if res.HighTone {
		tone = toneHigh
	} else if res.LowTone {
		tone = toneLow
	}

	if tone == toneNone {
		d.trillGapFrames++
		if d.trillGapFrames > trillMaxGapFrames {
			d.resetTrill()
		}
		return EventNone
	}

	d.trillGapFrames = 0

	if tone == d.trillCurrentTone {
		d.trillToneFrameCount++
		return EventNone
	}

	// トーンが切り替わった
	if d.trillCurrentTone != toneNone && d.trillToneFrameCount >= trillMinToneFrames {
		d.trillAlternations++
	}

	d.trillCurrentTone = tone
	d.trillToneFrameCount = 1

	if d.trillAlternations >= trillMinAlternations {
		return EventType1Detected
	}

	return EventNone
}

func (d *Detector) resetTrill() {
	d.trillCurrentTone = toneNone
	d.trillToneFrameCount = 0
	d.trillAlternations = 0
	d.trillGapFrames = 0
}
