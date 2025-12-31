# HillCLimb uses the default profilers
from ...tools.profiler import ProfilerBase
from ...tools.profiler import TensorboardProfiler
from ...tools.profiler import WandBProfiler

InitDebugProfiler = ProfilerBase
InitDebugTensorboardProfiler = TensorboardProfiler
InitDebugWandBProfiler = WandBProfiler
