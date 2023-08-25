from multiprocessing import Process

from simulation_constants import pospischilsPYr

from act import simulator

if __name__ == "__main__":
    constants = pospischilsPYr

    p = Process(target=simulator.run, args=[constants])
    p.start()
    p.join()
    p.terminate()
