param(
    [string]$Source = "$PSScriptRoot\..\.local\dolphin-native-source",
    [string]$Build = "$PSScriptRoot\..\.local\dolphin-native-build",
    [string]$ToolchainSetup = "",
    [string]$CMake = "cmake",
    [int]$Jobs = 8,
    [string]$Deploy = "$PSScriptRoot\..\.local\native-dolphin"
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$revision = '409881c34357d0f105fd473167d15ab0bd9c1628'
if (-not (Test-Path -LiteralPath $Source)) {
    git clone --depth 1 --branch 2609a https://github.com/dolphin-emu/dolphin.git $Source
    if ($LASTEXITCODE -ne 0) { throw 'Dolphin clone failed.' }
}
$actual = git -C $Source rev-parse HEAD
if ($actual -ne $revision) { throw "Unexpected Dolphin revision: $actual" }
git -C $Source submodule update --init --recursive --depth 1
if ($LASTEXITCODE -ne 0) { throw 'Dolphin submodule setup failed.' }
& "$projectRoot\.venv\Scripts\python.exe" "$PSScriptRoot\prepare_native_source.py" $Source
if ($LASTEXITCODE -ne 0) { throw 'Native bridge patch failed.' }

if ($ToolchainSetup) {
    # Import a developer environment only into this process, without a global PATH change.
    $environmentLines = & cmd /d /c "call `"$ToolchainSetup`" >nul && set"
    foreach ($entry in $environmentLines) {
        if ($entry -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}
& $CMake -S $Source -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF `
    -DUSE_MGBA=OFF -DUSE_RETRO_ACHIEVEMENTS=OFF -DENABLE_ANALYTICS=OFF `
    -DENABLE_AUTOUPDATE=OFF -DENABLE_CLI_TOOL=OFF -DENCODE_FRAMEDUMPS=OFF -DDISTRIBUTOR=SMG-RL
if ($LASTEXITCODE -ne 0) { throw 'Dolphin configure failed.' }
& $CMake --build $Build --target dolphin-emu --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Dolphin build failed.' }
if ((Test-Path -LiteralPath $Deploy) -and
    -not (Test-Path -LiteralPath "$Deploy\.smg-rl-native-build")) {
    throw "Refusing to deploy into an unowned directory: $Deploy"
}
New-Item -ItemType Directory -Path $Deploy -Force | Out-Null
Set-Content -LiteralPath "$Deploy\.smg-rl-native-build" -Value $revision -Encoding UTF8
Copy-Item -Path "$Build\Binaries\*" -Destination $Deploy -Recurse -Force
$manifest = @{
    upstream_revision = $revision
    upstream_license = 'GPL-2.0-or-later'
    bridge_license = 'MIT'
    protocol_version = 3
    frontier_capability = 1
    frontier_scratch_slots = @(8, 9)
    executable_sha256 = (Get-FileHash -LiteralPath "$Deploy\Dolphin.exe" -Algorithm SHA256).Hash
    patch_script_sha256 = (Get-FileHash -LiteralPath "$PSScriptRoot\prepare_native_source.py" -Algorithm SHA256).Hash
    bridge_header_sha256 = (Get-FileHash -LiteralPath "$projectRoot\native\SMGRLBridge.h" -Algorithm SHA256).Hash
    bridge_host_sha256 = (Get-FileHash -LiteralPath "$projectRoot\native\SMGRLHost.cpp" -Algorithm SHA256).Hash
    image_preprocessor_sha256 = (Get-FileHash -LiteralPath "$projectRoot\native\SMGRLImage.h" -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath "$Deploy\native-build.json" -Encoding UTF8
Write-Output "Native build deployed to $Deploy. Validate both game configurations before training."
