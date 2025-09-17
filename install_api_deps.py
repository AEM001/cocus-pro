#!/usr/bin/env python3
"""
API依赖安装脚本
自动安装阿里云百炼API调用所需的依赖包

使用方法：
  python install_api_deps.py
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并打印输出"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        
        if result.stdout:
            print(f"[STDOUT] {result.stdout}")
        if result.stderr:
            print(f"[STDERR] {result.stderr}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 命令执行失败: {e}")
        if e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"[INFO] Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] 需要Python 3.8或更高版本")
        return False
    
    return True

def install_package(package, description=""):
    """安装单个包"""
    print(f"\n{'='*50}")
    print(f"📦 安装 {package}")
    if description:
        print(f"   {description}")
    print("="*50)
    
    # 检查包是否已安装
    try:
        __import__(package.split('==')[0].replace('-', '_'))
        print(f"[✓] {package} 已安装")
        return True
    except ImportError:
        pass
    
    # 安装包
    cmd = f"{sys.executable} -m pip install {package}"
    return run_command(cmd, f"安装 {package}")

def verify_installation():
    """验证安装结果"""
    print(f"\n{'='*50}")
    print("🔍 验证安装结果")
    print("="*50)
    
    packages_to_check = [
        ("dashscope", "阿里云DashScope SDK"),
        ("openai", "OpenAI兼容SDK"),
        ("pillow", "图像处理"),
        ("numpy", "数值计算"),
        ("requests", "HTTP请求")
    ]
    
    all_good = True
    
    for package, desc in packages_to_check:
        try:
            if package == "pillow":
                import PIL
                print(f"[✓] {package} ({desc}) - 可用")
            else:
                __import__(package)
                print(f"[✓] {package} ({desc}) - 可用")
        except ImportError:
            print(f"[✗] {package} ({desc}) - 不可用")
            all_good = False
    
    return all_good

def main():
    print("="*60)
    print("🚀 阿里云百炼API依赖安装脚本")
    print("="*60)
    
    # 检查Python版本
    if not check_python_version():
        return 1
    
    # 需要安装的包列表
    packages = [
        ("dashscope", "阿里云DashScope SDK - 推荐的API调用方式"),
        ("openai>=1.0.0", "OpenAI SDK - 兼容模式API调用"),
        ("pillow>=8.0.0", "图像处理库"),
        ("numpy>=1.20.0", "数值计算库"),
        ("requests>=2.25.0", "HTTP请求库")
    ]
    
    # 逐个安装
    failed_packages = []
    
    for package, desc in packages:
        if not install_package(package, desc):
            failed_packages.append(package)
    
    # 验证安装
    print(f"\n{'='*50}")
    print("📋 安装总结")
    print("="*50)
    
    if failed_packages:
        print(f"[⚠️] 部分包安装失败:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n[INFO] 请手动安装失败的包:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
    else:
        print("[✅] 所有包安装成功!")
    
    # 验证安装结果
    if verify_installation():
        print(f"\n[✅] 环境配置完成! 可以开始使用API功能")
        
        # 提供使用示例
        print(f"\n{'='*50}")
        print("🎯 使用示例")
        print("="*50)
        print("1. 设置API密钥:")
        print("   export DASHSCOPE_API_KEY='your_api_key_here'")
        print("\n2. 运行目标检测:")
        print("   python auxiliary/scripts/detect_target_api.py --name f --target 'find the scorpionfish'")
        print("\n3. 运行完整流程:")
        print("   python run_api_pipeline.py --name f")
        print("\n4. 运行SAM精修 (使用API):")
        print("   python clean_sam_sculpt.py --name f --use-api")
        
        return 0
    else:
        print(f"\n[❌] 部分依赖验证失败，请检查安装结果")
        return 1

if __name__ == "__main__":
    exit(main())