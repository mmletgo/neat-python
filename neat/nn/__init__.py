from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork

# 尝试导入 Cython 优化的前向传播网络
try:
    from neat._cython.fast_network import FastFeedForwardNetwork
except ImportError:
    # Cython 版本不可用时，使用纯 Python 版本作为别名
    FastFeedForwardNetwork = FeedForwardNetwork
