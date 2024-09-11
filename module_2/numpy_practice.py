
"""1D arrays
"""
import numpy as np

def main():
    """Driven Function
    """
    # create an array
    array = np.array([-2,1,-5,10])
    print(array,type(array))

    numbers = [-2,1,-5 ,10]
    print(numbers, type(numbers))
    # convert list into array
    new_array =np.array(numbers)
    print(new_array, type(new_array))

    #2D arrays
    matrix = np.array([[-1,0,4],[-3,6,9]])
    print(matrix, type(matrix))

    #3D arrays
    '''array3d = np.array([[[[-1,2,3],
                         [3,5,7]],
                         [4,6,8],
                         [3,2,5]]
                        ])'''
    #print(array3d)

    # use the dtype optional argument
    new_array =np.array(numbers, dtype=np.short)
    print(new_array, new_array.dtype)
    numbers = [2,1,5 ,10]
    new_array2 =np.array(numbers, dtype=np.ushort)
    print(new_array2,new_array2.dtype)

    # Floats
    numbers = [2,1,5 ,10]
    new_array2 =np.array(numbers, dtype=float)
    print(new_array2,new_array2.dtype)

if __name__ == '__main__':
    main()