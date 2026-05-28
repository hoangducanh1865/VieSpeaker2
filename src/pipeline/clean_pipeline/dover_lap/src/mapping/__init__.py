from .greedy import GreedyMap
try:
    from .hungarian import HungarianMap
except ImportError:
    HungarianMap = None
