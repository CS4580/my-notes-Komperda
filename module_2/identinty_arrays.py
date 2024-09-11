
"""Practice some some numpy array identities
"""

import numpy as np

def main():
    """Driven Function
    """
    # Create a 2D 3 by 3 identity matrix 
    identity_3x3 = np.eye(3,3)
    print(identity_3x3)

    # 2D diagnoal, with given entries
    diagnoal_2D = np.diag([2,3,4,5])
    print(diagnoal_2D)

    # Create a 5x3 2D array of unsigned int filled with zeros
    array_5x3 = np.zeros((5,3), dtype=np.uint)
    print(f'array_5x3 {array_5x3}')

    # Create a 5x3 2D array of unsigned int filled with ones
    array_5x3_1 = np.ones((5,3), dtype=np.uint)
    print(f'array_5x3 {array_5x3_1}')

    # Create a 5x3 2D array of  filled with given values
    array_5x3_full = np.full((5,3), 10)
    print(f'array_5x3 {array_5x3_full}')

    # Create a 5x3 2D array of filled with random values
    array_5x3_random = np.random.random((5,3))
    print(f'array_5x3_random {array_5x3_random}')

if __name__ == '__main__':
    main()