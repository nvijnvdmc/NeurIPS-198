import sys
import time

from pygame import mixer  # Load the popular external library

from p02_traj_factory import generate_trajectories

if __name__ == '__main__':
    # ================== default arguments ===================
    # available arguments:
    #
    #           "i1000_Acrobot.json"
    #           "i2000_CartPole.json"
    #           "i3000_MountainCar.json"
    #           "i4000_Taxi.json"
    #           "i5000_FrozenLake.json"
    #           "i6000_Breakout.json"
    #           "i7000_PongNoFrameskip.json"
    #

    filename = "i1000_Acrobot.json"

    # ================== experimental setup ==================

    generate_trajectories(filename=filename)

    print("End of trajectory generation.")
