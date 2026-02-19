package main

import (
	"net/netip"
	"strconv"
	"time"

	"github.com/soypat/cyw43439"
	"github.com/soypat/lneto/http/httpraw"
	"github.com/soypat/lneto/tcp"
	"github.com/soypat/lneto/x/xnet"
)

// ビルド時に -ldflags で注入する変数
var wifiSSID string
var wifiPass string
var mode string // "dev" or "prod"（デフォルト: "dev"）

// ntfy通知設定
var ntfyAddr = netip.AddrPortFrom(netip.AddrFrom4([4]byte{192, 168, 2, 241}), 80)

const tcpBufSize = 2030 // MTU - ethhdr - iphdr - tcphdr

// ntfyTopic はmodeに応じたntfyトピックを返す。
func ntfyTopic() string {
	if mode == "prod" {
		return "/doorbell"
	}
	return "/debug"
}

var requestedIP = [4]byte{192, 168, 2, 99}

// setupNetwork はWi-Fi接続→ネットワークスタック構築→DHCP→ゲートウェイ解決を行う。
// Wi-Fi接続は成功するまで無制限にリトライする。
func setupNetwork(dev *cyw43439.Device) (*xnet.StackAsync, xnet.StackRetrying, error) {
	println("Connecting to WiFi:", wifiSSID)
	for attempt := 1; ; attempt++ {
		err := dev.JoinWPA2(wifiSSID, wifiPass)
		if err == nil {
			break
		}
		println("WiFi join failed (attempt", attempt, "):", err.Error())
		time.Sleep(5 * time.Second)
	}
	println("WiFi connected")

	mac, err := dev.HardwareAddr6()
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}

	var stack xnet.StackAsync
	err = stack.Reset(xnet.StackConfig{
		Hostname:        "pico-doorbell",
		MaxTCPConns:     1,
		RandSeed:        time.Now().UnixNano(),
		HardwareAddress: mac,
		MTU:             cyw43439.MTU,
	})
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}

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

	// DHCP
	const pollTime = 50 * time.Millisecond
	rstack := stack.StackRetrying(pollTime)
	dhcpResults, err := rstack.DoDHCPv4(requestedIP, 3*time.Second, 3)
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}
	if err := stack.AssimilateDHCPResults(dhcpResults); err != nil {
		return nil, xnet.StackRetrying{}, err
	}
	println("DHCP complete:", dhcpResults.AssignedAddr.String())

	// ゲートウェイのMACアドレス解決
	gatewayHW, err := rstack.DoResolveHardwareAddress6(dhcpResults.Router, 500*time.Millisecond, 4)
	if err != nil {
		return nil, xnet.StackRetrying{}, err
	}
	stack.SetGateway6(gatewayHW)

	return &stack, rstack, nil
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
	hdr.Set("Host", ntfyAddr.Addr().String())
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

	// レスポンス待ち
	time.Sleep(500 * time.Millisecond)
	var rxBuf [512]byte
	n, _ := conn.Read(rxBuf[:])
	if n > 0 {
		println("ntfy response:", string(rxBuf[:n]))
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
