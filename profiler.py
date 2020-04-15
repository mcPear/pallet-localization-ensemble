import cProfile
import os

def profiled(expr, globalz, localz):
    cProfile.runctx(
            expr,
            globalz,
            localz,
            '/home/maciej/repos/pallet-recogntion-gpu/profiler_result.pstats'
        )
    os.system('python3 /usr/local/lib/python3.6/dist-packages/gprof2dot.py -f pstats /home/maciej/repos/pallet-recogntion-gpu/profiler_result.pstats | dot -Tpng -o /home/maciej/repos/pallet-recogntion-gpu/profiler_result.png')