import sys # unused import
import os

def calculate_divide(a, b):
    # Potential ZeroDivisionError if b is 0
    return a / b

def main():
    print("Starting calculations")
    # Will raise ZeroDivisionError
    res = calculate_divide(10, 0)
    print(res)

if __name__ == '__main__':
    main()
