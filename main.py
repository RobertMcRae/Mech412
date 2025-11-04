import os
import json
import numpy as np

from load_data_sc import load_data
from system_ID import SystemID

data = load_data() #Data struct: [csv files, entries, time/input/output]

data1 = data[0,:,:]  #IO_data_1_0.csv
data2 = data[1,:,:]  #IO_data_1_1.csv
data3 = data[2,:,:]  #IO_data_1_2.csv
data4 = data[3,:,:]  #IO_data_1_3.csv

print("Starting SystemID")
system1 = SystemID(name = "System 1",
                  u = data1[:,1],
                  y = data1[:,2],
                  T = data1[:,0],
                  normalize = True)
system1.run()
system2 = SystemID(name = "System 2",
                  u = data2[:,1],
                  y = data2[:,2],
                  T = data2[:,0],
                  normalize = True)
system2.run()
system3 = SystemID(name = "System 3",
                  u = data3[:,1],
                  y = data3[:,2],
                  T = data3[:,0],
                  normalize = True)
system3.run()
system4 = SystemID(name = "System 4",
                  u = data4[:,1],
                  y = data4[:,2],
                  T = data4[:,0],
                  normalize = True)
system4.run()
system123 = SystemID(name = "System 1,2,3 Combined",
                    u = np.concatenate([data1[:,1], data2[:,1], data3[:,1]]),
                    y = np.concatenate([data1[:,2], data2[:,2], data3[:,2]]),
                    T = np.concatenate([data1[:,0],
                                        data2[:,0] + data1[-1,0],
                                        data3[:,0] + data1[-1,0] + data2[-1,0]]),
                    normalize = True)
system123.run()
print("SystemID complete.")

cwd = os.getcwd()
with open('Assignments/Project/results/system_id.json', 'w') as f: 
    json.dump(SystemID._system_id_catalog, f, indent=4)

fig = SystemID.plot()
fig.savefig('Assignments/Project/results/system_id_plots.pdf')
