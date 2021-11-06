import numpy as np
from PIL import Image
import sys

if not len(sys.argv) == 3:
    print('give 2 arguments, input file path and output directory path')
    print('example: python extract_data.py ./datasets/1634481396404/1634481396404.npz ./extracted/')
    exit()

input_file = sys.argv[1]
output_file = sys.argv[2]
print(input_file, output_file)
data = np.load(input_file)
data = data['arr_0']
for i in range(data.shape[0]):
    image = data[i]
    image = Image.fromarray(image)
    image.save(output_file+str(i)+'.png')
    
    
