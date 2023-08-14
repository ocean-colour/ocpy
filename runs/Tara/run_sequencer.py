""" Run Sequencer on Tara data """
import numpy as np


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Unnormalized
    if flg & (2**0):
        pass



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Unnormalied
    else:
        flg = sys.argv[1]

    main(flg)
