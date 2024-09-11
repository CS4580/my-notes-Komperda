

"""Array Indexing
"""

import numpy as np

def main():
    """Driven Function
    """
    arr_1d = np.arange(10)
    # last element
    print(arr_1d[-1])

    arr_2d = np.array([[21,22,23,24],
              [31,32,33,34],
              [41,42,43,44]])
    print(arr_2d[2,2])
    print(f'Full Row {arr_2d[0]}')
    #Slicing
    print(f'Full Row {arr_2d[0]}')


if __name__ == '__main__':
    main()