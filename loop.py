from tqdm import tqdm
import os
import time
if __name__ == '__main__':
    os.mkdir(os.getcwd() + "/we/")
    i = 0
    for index in tqdm(range(0, 100), desc ="Loop"):
        time.sleep(0.01)
    print("Running")