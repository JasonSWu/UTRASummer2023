import numpy as np
import pandas as pd
import pickle as pkl

def srg(img, seed, epsilon): #Seeded region growing, for detecting land vs ocean
  x_list, y_list = [seed[0]], [seed[1]]
  width, height = img.shape
  I = [img[seed]]
  calc_mean, calc_stdev = True, True
  mean = np.mean(I)
  stdev = np.std(I)
  prev_mean = mean
  prev_stdev = stdev
  RT = (mean ** (1/2))
  q = []
  q.append(seed)
  checked = set([seed])
  dirs = [[1,0], [0,-1], [0,1], [-1,0]]
  while len(q) > 0:
    x, y = q.pop()
    for dx, dy in dirs:
        x1 = x + dx
        y1 = y + dy
        neighbor = (x1, y1)
        if neighbor not in checked and x1 >= 0 and x1 < width and y1 >= 0 and y1 < height:
            checked.add(neighbor)
            if abs(img[neighbor] - mean) < RT + epsilon:
                q.append(neighbor)
                x_list.append(neighbor[0])
                y_list.append(neighbor[1])
                I.append(img[neighbor])
                prev_mean = mean
                prev_stdev = stdev
                if calc_mean:
                    mean = np.mean(I)
                    if (mean - prev_mean) / prev_mean < 0.01:
                        calc_mean = False
                if calc_stdev:
                    stdev = np.std(I)
                    if (stdev - prev_stdev) / prev_stdev < 0.01:
                       calc_stdev = False
                RT = (mean/stdev) if stdev != 0 else mean
  return np.array(x_list), np.array(y_list)

file_name = input()
data = pd.read_csv(file_name).to_numpy()
land_mask = srg(data, (int(input()), int(input())), 1e-3)
with open("LandMask.pkl", "wb") as f:
   pkl.dump(land_mask, f)
data[land_mask] = np.nan
data = pd.DataFrame(data)
data.to_csv(f"LandMasked/{file_name.split('/')[-1]}")