
"""Array operations
"""
import numpy as np

def main():
    """Driven Function
    """
    numbers = [2,4,6,8,10]
    print(numbers)
    for item in range(len(numbers)):
        numbers[item] += 3
    print(numbers)

    # convert list to Numpy array
    numbers_arr = np.array(numbers)
    print(f'Before array {numbers_arr}')
    numbers_arr += 3
    print(f'After array {numbers_arr}')


if __name__ == '__main__':
    main()