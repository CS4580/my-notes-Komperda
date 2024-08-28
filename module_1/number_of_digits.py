"""Library to calculate number of digits for 
    different algorithms
"""
import math

def factorial_length(number):
    """Count the number of digits in a factorial number

    Args:
        number (integer): integer value to calculate factorial

    Returns:
        integer: number of digits for factorial of input
    """
    
    value = math.factorial(number)

    length = len(str(value)) # cast value to string then take len of the string
    return length

def main():
    """Driven Function
    """
    number = 100
    digits = factorial_length(number)
    print(f'You have {digits} digits in factorial({number})')

    number = 1000
    digits = factorial_length(number)
    print(f'You have {digits} digits in factorial({number})')

if __name__ == '__main__':
    main()