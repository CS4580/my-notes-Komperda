
"""Range Arrays
"""
import numpy as np

def main():
    """Practicing range arrays
    """
    '''
    numbers = range(10)
    for num in range(1,11,2):
        print(num)
    '''

    # generate 1D arrays from 0 to 8
    array = np.arange(9)
    print(array)

    # negative numbers
    array2 = np.arange(-4,4)
    print(array2)

    # array of values from 0 to 5 , in steps of 0.1
    array3 = np.arange(0.1,0.3,0.01)
    print(array3)


if __name__ == '__main__':
    main()