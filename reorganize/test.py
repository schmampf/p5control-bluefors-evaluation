import pickle
import matplotlib.pyplot as plt

path = r"A:\Documents\Git\ProjektPraktikum\Data\dcapalbo\exp_data\Amplitude Study (19.3GHz, Antenna).pickle"

with open(path, "rb") as f:
    data = pickle.load(f)

print()
print(data.keys())
set = "evaluated"
print()
print(data[set])
# print()
# print(data["evaluated"])

# plt.imshow(data["mapped"]["y_axis"])
