param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$SerialNumber
)
$ErrorActionPreference = 'Stop'

if (-not $SerialNumber) {
    $boards = @(Get-PnpDevice -PresentOnly | Where-Object {
        $_.InstanceId -match '^USB\\VID_2E8A&PID_000[AF]\\'
    })
    if ($boards.Count -ne 1) { throw "Connect exactly one Pico (found $($boards.Count))." }
    $instance = $boards[0].InstanceId
    $SerialNumber = $instance.Split('\')[-1]

    if ($instance -like 'USB\VID_2E8A&PID_000A\*') {
        $ports = @(Get-CimInstance Win32_SerialPort | Where-Object {
            $_.PNPDeviceID -match '^USB\\VID_2E8A&PID_000A(?:&MI_\d+)?\\'
        })
        if ($ports.Count -ne 1) {
            throw "No unique Pico COM port (found $($ports.Count)). Attach the normal-mode Pico to WSL or use BOOTSEL for the first flash."
        }
        $serial = [System.IO.Ports.SerialPort]::new($ports[0].DeviceID, 115200)
        $serial.DtrEnable = $true
        $serial.WriteTimeout = 2000
        try {
            $serial.Open()
            $serial.Write("`nBOOTSEL`n")
            Start-Sleep -Milliseconds 200
        } finally {
            $serial.Dispose()
        }
    }
}

$deadline = [DateTime]::UtcNow.AddSeconds(20)
do {
    # The ROM disk exposes the same chip ID as the firmware's USB serial port.
    $targets = @(Get-Disk | Where-Object {
        $_.BusType -eq 'USB' -and $_.FriendlyName -eq 'RPI RP2350' -and $_.SerialNumber -eq $SerialNumber
    } | Get-Partition -ErrorAction SilentlyContinue | Get-Volume -ErrorAction SilentlyContinue |
        Where-Object { $_.FileSystemLabel -eq 'RP2350' -and $_.DriveLetter })
    if ($targets.Count -eq 1) {
        Copy-Item -LiteralPath $Source -Destination ($targets[0].DriveLetter + ':\')
        Write-Host 'Firmware copied. The Pico will reboot into the application.'
        return
    }
    if ($targets.Count -gt 1) { throw 'Multiple volumes match the Pico chip ID.' }

    $state = (& usbipd.exe state | Out-String | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0) { throw 'Unable to query usbipd.' }
    $attached = @($state.Devices | Where-Object {
        $_.InstanceId -eq "USB\VID_2E8A&PID_000F\$SerialNumber" -and $_.ClientIPAddress
    })
    if ($attached.Count -gt 1) { throw 'Multiple BOOTSEL devices are attached to WSL.' }
    if ($attached.Count -eq 1) {
        & usbipd.exe detach --busid $attached[0].BusId
        if ($LASTEXITCODE -ne 0) { throw 'Unable to release the BOOTSEL device from WSL.' }
    }
    Start-Sleep -Milliseconds 300
} while ([DateTime]::UtcNow -lt $deadline)
throw 'The selected Pico did not mount in BOOTSEL mode. Older firmware may need the button once.'
