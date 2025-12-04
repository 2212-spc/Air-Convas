#!/usr/bin/env python3
"""测试所有导入和基本功能"""

import sys
import traceback

def test_imports():
    """测试所有模块的导入"""
    tests = [
        ("config", lambda: __import__('config')),
        ("core.hand_detector", lambda: __import__('core.hand_detector', fromlist=['HandDetector'])),
        ("core.gesture_recognizer", lambda: __import__('core.gesture_recognizer', fromlist=['GestureRecognizer'])),
        ("core.coordinate_mapper", lambda: __import__('core.coordinate_mapper', fromlist=['CoordinateMapper'])),
        ("modules.canvas", lambda: __import__('modules.canvas', fromlist=['Canvas'])),
        ("modules.virtual_pen", lambda: __import__('modules.virtual_pen', fromlist=['VirtualPen'])),
        ("modules.eraser", lambda: __import__('modules.eraser', fromlist=['Eraser'])),
        ("modules.ppt_controller", lambda: __import__('modules.ppt_controller', fromlist=['PPTController'])),
        ("modules.shape_recognizer", lambda: __import__('modules.shape_recognizer', fromlist=['ShapeRecognizer'])),
        ("utils.smoothing", lambda: __import__('utils.smoothing', fromlist=['EmaSmoother'])),
    ]
    
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, import_func in tests:
        try:
            import_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}")
            print(f"  错误: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能")
    print("=" * 60)
    
    try:
        import config
        import numpy as np
        from core.hand_detector import HandDetector
        from modules.canvas import Canvas
        
        # 测试 Canvas
        print("✓ 创建 Canvas...")
        canvas = Canvas(1280, 720)
        
        # 测试 HandDetector
        print("✓ 创建 HandDetector...")
        detector = HandDetector(max_num_hands=1)
        
        # 测试绘制
        print("✓ 测试画布绘制...")
        canvas.draw_line((100, 100), (200, 200), (0, 255, 255), 3)
        
        # 测试清除
        print("✓ 测试画布清除...")
        canvas.clear()
        
        print("=" * 60)
        print("所有基本功能测试通过！")
        print("=" * 60)
        
        detector.close()
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports() and test_basic_functionality()
    sys.exit(0 if success else 1)


