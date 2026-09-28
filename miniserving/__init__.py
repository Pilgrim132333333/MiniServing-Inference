try:
    from miniserving._miniserving import add
except ImportError:
    # C++ 扩展未编译时忽略，纯 Python 功能正常使用
    add = None
__all__ = ["add"]
