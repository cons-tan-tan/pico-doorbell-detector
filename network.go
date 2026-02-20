package main

import (
	"errors"
	"net/netip"
	"strconv"
	"time"

	"github.com/soypat/cyw43439"
	"github.com/soypat/lneto/http/httpraw"
	"github.com/soypat/lneto/tcp"
	"github.com/soypat/lneto/x/xnet"
)

// keepAlive はウォッチドッグを更新して自動リセットを防止する。main.go から注入。
var keepAlive func()

// hardReset はデバイスを強制再起動する。main.go から注入。
var hardReset func()

// ビルド時に -ldflags で注入する変数
var wifiSSID string
var wifiPass string
var mode string // "dev" or "prod"（デフォルト: "dev"）

// 静的ネットワーク設定
var (
	staticIP  = netip.AddrFrom4([4]byte{192, 168, 1, 101})
	gatewayIP = netip.AddrFrom4([4]byte{192, 168, 0, 1})
	dnsServer = netip.AddrFrom4([4]byte{192, 168, 0, 1})
	subnet    = netip.MustParsePrefix("192.168.0.0/20")
)

// ntfy通知設定
const ntfyHost = "ntfy.constantan.dev"
const ntfyPort = 80

var ntfyAddr netip.AddrPort

const tcpBufSize = 2030 // MTU - ethhdr - iphdr - tcphdr

// ntfyTopic はmodeに応じたntfyトピックを返す。
func ntfyTopic() string {
	if mode == "prod" {
		return "/doorbell"
	}
	return "/debug"
}

// setupNetwork はWi-Fi接続→ネットワークスタック構築→静的IP設定→ゲートウェイ解決を行う。
// Wi-Fi接続は成功するまで無制限にリトライする。
func setupNetwork(dev *cyw43439.Device) (*xnet.StackAsync, xnet.StackRetrying, error) {
	println("Connecting to WiFi:", wifiSSID)
	for attempt := 1; ; attempt++ {
		keepAlive()
		err := dev.JoinWPA2(wifiSSID, wifiPass)
		if err == nil {
			break
		}
		println("WiFi join failed (attempt", attempt, "):", err.Error())
		for i := 0; i < 5; i++ {
			time.Sleep(time.Second)
			keepAlive()
		}
	}
	println("WiFi connected")

	mac, err := dev.HardwareAddr6()
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}

	var stack xnet.StackAsync
	err = stack.Reset(xnet.StackConfig{
		Hostname:        "pico-doorbell",
		MaxTCPConns:     2,
		RandSeed:        time.Now().UnixNano(),
		StaticAddress:   staticIP,
		DNSServer:       dnsServer,
		HardwareAddress: mac,
		MTU:             cyw43439.MTU,
	})
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}
	stack.SetSubnet(subnet)

	dev.RecvEthHandle(func(pkt []byte) error {
		return stack.Demux(pkt, 0)
	})

	// バックグラウンドパケット処理ループ（goroutine、-scheduler=tasks必須）
	sendbuf := make([]byte, cyw43439.MaxFrameSize)
	go func() {
		for {
			dev.PollOne()
			n, _ := stack.Encapsulate(sendbuf, -1, 0)
			if n > 0 {
				dev.SendEth(sendbuf[:n])
			} else {
				time.Sleep(5 * time.Millisecond)
			}
		}
	}()

	const pollTime = 50 * time.Millisecond
	rstack := stack.StackRetrying(pollTime)

	println("Static IP:", staticIP.String())

	// ゲートウェイのMACアドレス解決
	if err := resolveGateway(&stack, rstack); err != nil {
		return nil, xnet.StackRetrying{}, err
	}

	return &stack, rstack, nil
}

// resolveGateway はゲートウェイのMACアドレスをARP解決して設定する。
func resolveGateway(stack *xnet.StackAsync, rstack xnet.StackRetrying) error {
	gatewayHW, err := rstack.DoResolveHardwareAddress6(gatewayIP, 500*time.Millisecond, 4)
	if err != nil {
		return err
	}
	stack.SetGateway6(gatewayHW)
	println("Gateway resolved:", gatewayIP.String())
	return nil
}

// resolveNtfy はntfyサーバーのDNS解決を行い ntfyAddr を更新する。
func resolveNtfy(rstack xnet.StackRetrying) error {
	addrs, err := rstack.DoLookupIP(ntfyHost, 3*time.Second, 3)
	if err != nil {
		return err
	}
	if len(addrs) == 0 {
		return errors.New("DNS: no addresses returned")
	}
	ntfyAddr = netip.AddrPortFrom(addrs[0], ntfyPort)
	println("ntfy server:", ntfyAddr.String())
	return nil
}

// recoverNetwork はWiFi接続確認→再接続→ゲートウェイARP再解決→DNS再解決を行う。
func recoverNetwork(dev *cyw43439.Device, stack *xnet.StackAsync, rstack xnet.StackRetrying) error {
	if !dev.IsLinkUp() {
		println("WiFi disconnected, reconnecting...")
		for attempt := 1; attempt <= 3; attempt++ {
			keepAlive()
			err := dev.JoinWPA2(wifiSSID, wifiPass)
			if err == nil {
				println("WiFi reconnected")
				break
			}
			println("WiFi rejoin failed (attempt", attempt, "):", err.Error())
			if attempt == 3 {
				return errors.New("WiFi reconnect failed after 3 attempts")
			}
			for i := 0; i < 5; i++ {
				time.Sleep(time.Second)
				keepAlive()
			}
		}
	}

	// ゲートウェイMACを再解決
	if err := resolveGateway(stack, rstack); err != nil {
		return err
	}

	// ntfyサーバーのDNSを再解決
	if err := resolveNtfy(rstack); err != nil {
		return err
	}

	return nil
}

// notifyWithRecovery は通知送信を試み、失敗時はネットワーク復旧→リトライを行う。
// 復旧不能な場合は hardReset() でデバイスを再起動する。
func notifyWithRecovery(dev *cyw43439.Device, stack *xnet.StackAsync, rstack xnet.StackRetrying, message string) {
	if err := sendNtfy(stack, rstack, message); err != nil {
		println("ntfy failed:", err.Error())
		println("attempting network recovery...")

		if err := recoverNetwork(dev, stack, rstack); err != nil {
			println("recovery failed:", err.Error(), "-> resetting")
			hardReset()
		}

		// リカバリ後にリトライ
		if err := sendNtfy(stack, rstack, message); err != nil {
			println("ntfy retry failed:", err.Error(), "-> resetting")
			hardReset()
		}
	}
}

// sendNtfy はntfyサーバーにHTTP POSTで通知を送信する。
func sendNtfy(stack *xnet.StackAsync, rstack xnet.StackRetrying, message string) error {
	defer stack.DiscardResolveHardwareAddress6(ntfyAddr.Addr())

	var conn tcp.Conn
	if err := conn.Configure(tcp.ConnConfig{
		RxBuf:             make([]byte, tcpBufSize),
		TxBuf:             make([]byte, tcpBufSize),
		TxPacketQueueSize: 3,
	}); err != nil {
		return err
	}

	lport := uint16(stack.Prand32()>>17) + 1024
	if err := rstack.DoDialTCP(&conn, lport, ntfyAddr, 5*time.Second, 3); err != nil {
		return err
	}
	defer closeConn(&conn)

	// HTTPリクエスト構築
	body := []byte(message)
	var hdr httpraw.Header
	hdr.SetMethod("POST")
	hdr.SetRequestURI(ntfyTopic())
	hdr.SetProtocol("HTTP/1.1")
	hdr.Set("Host", ntfyHost)
	hdr.Set("Content-Type", "text/plain")
	hdr.Set("Content-Length", strconv.Itoa(len(body)))
	hdr.Set("Connection", "close")

	reqbytes, err := hdr.AppendRequest(nil)
	if err != nil {
		return err
	}

	if _, err := conn.Write(reqbytes); err != nil {
		return err
	}
	if _, err := conn.Write(body); err != nil {
		return err
	}

	// レスポンス待ち（ポーリング）
	var rxBuf [512]byte
	var n int
	for i := 0; i < 10; i++ {
		time.Sleep(100 * time.Millisecond)
		n, _ = conn.Read(rxBuf[:])
		if n > 0 {
			break
		}
	}

	if n == 0 {
		return errors.New("no response from ntfy server")
	}

	resp := string(rxBuf[:n])
	println("ntfy response:", resp)

	// HTTPステータスチェック（"HTTP/1.x 2xx"形式を確認）
	if n < 12 || rxBuf[9] < '2' || rxBuf[9] > '2' {
		return errors.New("ntfy returned non-2xx: " + resp[:min(n, 40)])
	}

	return nil
}

func closeConn(conn *tcp.Conn) {
	conn.Close()
	for i := 0; i < 50 && !conn.State().IsClosed(); i++ {
		time.Sleep(100 * time.Millisecond)
	}
	conn.Abort()
}
