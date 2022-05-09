import numpy as np

from modules.dataTransform import normalize
from modules.dataLoad import load

def main():
    attrs, labels = load('./Train.txt')
    attrs = normalize(attrs)

if __name__ == '__main__':
    main()

