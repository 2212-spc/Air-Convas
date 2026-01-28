# ===================================================================
# Anaconda/Conda 自动激活配置脚本
# 用途：让 Cursor 的 PowerShell 终端自动激活 conda base 环境
# ===================================================================

Write-Host "开始配置 Anaconda 环境..." -ForegroundColor Green

# 1. 修改 PowerShell 执行策略（需要管理员权限或当前用户权限）
Write-Host "`n步骤 1: 修改 PowerShell 执行策略..." -ForegroundColor Yellow
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✓ 执行策略已设置为 RemoteSigned" -ForegroundColor Green
} catch {
    Write-Host "✗ 无法修改执行策略，请以管理员身份运行此脚本" -ForegroundColor Red
    Write-Host "或手动运行: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
}

# 2. 初始化 Conda 到 PowerShell
Write-Host "`n步骤 2: 初始化 Conda..." -ForegroundColor Yellow
$condaPath = "D:\Anaconda\anaconda3"
$condaExe = Join-Path $condaPath "Scripts\conda.exe"

if (Test-Path $condaExe) {
    Write-Host "找到 Conda: $condaExe" -ForegroundColor Cyan
    
    # 初始化 conda
    & $condaExe "shell.powershell" "hook" | Out-String | Invoke-Expression
    
    # 运行 conda init
    & $condaExe init powershell
    Write-Host "✓ Conda 已初始化到 PowerShell" -ForegroundColor Green
} else {
    Write-Host "✗ 未找到 Conda，请确认路径: $condaPath" -ForegroundColor Red
    exit 1
}

# 3. 创建或修改 PowerShell Profile
Write-Host "`n步骤 3: 配置 PowerShell Profile..." -ForegroundColor Yellow

# 获取 PowerShell Profile 路径
$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath

# 创建 Profile 目录（如果不存在）
if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "✓ 创建了 Profile 目录: $profileDir" -ForegroundColor Green
}

# 准备要添加的 Conda 初始化代码
$condaInitCode = @"

# ==================== Anaconda/Conda 自动激活配置 ====================
# 由 setup_conda_env.ps1 自动生成
# 生成时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# Conda 安装路径
`$Env:CONDA_ROOT = "$condaPath"

# 初始化 Conda
if (Test-Path "`$Env:CONDA_ROOT\Scripts\conda.exe") {
    # 方法1: 通过 conda hook 初始化
    (& "`$Env:CONDA_ROOT\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
    
    # 自动激活 base 环境
    conda activate base
    
    Write-Host "✓ Conda base 环境已激活" -ForegroundColor Green
} else {
    Write-Host "⚠ 警告: 未找到 Conda，请检查路径配置" -ForegroundColor Yellow
}
# ====================================================================

"@

# 检查 Profile 是否已存在
if (Test-Path $profilePath) {
    $currentContent = Get-Content -Path $profilePath -Raw -ErrorAction SilentlyContinue
    
    # 检查是否已经包含 Conda 配置
    if ($currentContent -match "Anaconda/Conda 自动激活配置") {
        Write-Host "⚠ Profile 中已存在 Conda 配置，正在更新..." -ForegroundColor Yellow
        
        # 移除旧的配置（如果存在）
        $newContent = $currentContent -replace "(?ms)# ==================== Anaconda/Conda 自动激活配置 ====================.*?# ====================================================================", ""
        Set-Content -Path $profilePath -Value $newContent -Force
    }
    
    # 追加新配置
    Add-Content -Path $profilePath -Value $condaInitCode
    Write-Host "✓ 已更新 Profile: $profilePath" -ForegroundColor Green
} else {
    # 创建新的 Profile
    Set-Content -Path $profilePath -Value $condaInitCode
    Write-Host "✓ 已创建新 Profile: $profilePath" -ForegroundColor Green
}

# 4. 测试配置
Write-Host "`n步骤 4: 测试配置..." -ForegroundColor Yellow
try {
    # 重新加载 Profile
    . $profilePath
    Write-Host "✓ Profile 已重新加载" -ForegroundColor Green
    
    # 检查 conda 是否可用
    $condaVersion = conda --version 2>$null
    if ($condaVersion) {
        Write-Host "✓ Conda 版本: $condaVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ 测试时出现警告，但配置可能已成功" -ForegroundColor Yellow
}

# 5. 完成
Write-Host "`n" + "="*70 -ForegroundColor Cyan
Write-Host "配置完成！" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "`n请执行以下操作之一：" -ForegroundColor Yellow
Write-Host "  1. 关闭并重新打开终端窗口" -ForegroundColor White
Write-Host "  2. 或运行: . `$PROFILE (重新加载配置)" -ForegroundColor White
Write-Host "`n之后你的终端提示符前面应该会显示 (base)" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

# 显示 Profile 位置
Write-Host "`nPowerShell Profile 位置: $profilePath" -ForegroundColor Cyan
Write-Host "你可以手动编辑此文件来自定义配置`n" -ForegroundColor Gray

