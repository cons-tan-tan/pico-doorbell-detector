# Two RP2350 drives have the same label. Only the selected chip may receive a copy.
$ErrorActionPreference = 'Stop'
function Get-Disk {
    [pscustomobject]@{ BusType='USB'; FriendlyName='RPI RP2350'; SerialNumber='0000000000000001'; DriveLetter='X' }
    [pscustomobject]@{ BusType='USB'; FriendlyName='RPI RP2350'; SerialNumber='000000000000000A'; DriveLetter='Y' }
}
function Get-Partition { process { $_ } }
function Get-Volume { process { [pscustomobject]@{ FileSystemLabel='RP2350'; DriveLetter=$_.DriveLetter } } }
function Copy-Item($LiteralPath, $Destination) { $Destination }
function usbipd.exe { throw 'Unexpected USB operation in volume selection test.' }

$copies = @(& "$PSScriptRoot/flash.ps1" -Source test.uf2 -SerialNumber '000000000000000a' 6>$null)
if ($copies.Count -ne 1 -or $copies[0] -ne 'Y:\') { throw "Incorrect flash destination: $copies" }
