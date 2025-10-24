def sin(freq, time):
    """生成正弦波信号"""
    import numpy as np
    
    return np.sin(2 * np.pi * freq * time)