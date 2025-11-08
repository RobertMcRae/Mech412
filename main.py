import os
import json
import numpy as np

from load_data_sc import load_data
from system_ID import SystemID
from model_uncertainty import ModelUncertainty

data = load_data() #Data struct: [csv files, entries, time/input/output]

data1 = data[0,:,:]  #IO_data_1_0.csv
data2 = data[1,:,:]  #IO_data_1_1.csv
data3 = data[2,:,:]  #IO_data_1_2.csv
data4 = data[3,:,:]  #IO_data_1_3.csv

step_size = data1[1,0] - data1[0,0]

print("Starting SystemID")
system1 = SystemID(name = "System 1",
                  u = data1[:,1],
                  y = data1[:,2],
                  T = data1[:,0],
                  u_test = data4[:,1],
                  y_test = data4[:,2],
                  t_test = data4[:,0],
                  normalize = True)
system1.run()
system2 = SystemID(name = "System 2",
                  u = data2[:,1],
                  y = data2[:,2],
                  T = data2[:,0] + data1[-1,0],
                  u_test = data4[:,1],
                  y_test = data4[:,2],
                  t_test = data4[:,0],
                  normalize = True)
system2.run()
system3 = SystemID(name = "System 3",
                  u = data3[:,1],
                  y = data3[:,2],
                  T = data3[:,0] + data1[-1,0] + data2[-1,0],
                  u_test = data4[:,1],
                  y_test = data4[:,2],
                    t_test = data4[:,0],
                  normalize = True)
system3.run()
system4 = SystemID(name = "System 4",
                  u = data4[:,1],
                  y = data4[:,2],
                  T = data4[:,0] + data1[-1,0] + data2[-1,0] + data3[-1,0],
                  u_test = np.concatenate([data1[:,1], data2[:,1], data3[:,1]]),
                  y_test = np.concatenate([data1[:,2], data2[:,2], data3[:,2]]),
                  t_test = np.concatenate([data1[:,0],
                                           data2[:,0] + data1[-1,0],
                                           data3[:,0] + data1[-1,0] + data2[-1,0]]),
                  normalize = True)
system4.run()
system123 = SystemID(name = "System 1,2,3 Combined",
                    u = np.concatenate([data1[:,1], data2[:,1], data3[:,1]]),
                    y = np.concatenate([data1[:,2], data2[:,2], data3[:,2]]),
                    T = np.concatenate([data1[:,0],
                                        data2[:,0] + data1[-1,0] + step_size,
                                        data3[:,0] + data1[-1,0] + data2[-1,0] + 2*step_size]),
                    u_test = data4[:,1],
                    y_test = data4[:,2],
                    t_test = data4[:,0],
                    normalize = True)
system123.run()
print("SystemID complete.")

cwd = os.getcwd()
with open('Assignments/Project/results/system_id.json', 'w') as f: 
    json.dump(SystemID._system_id_catalog, f, indent=4)

#Save all plots
for key, value in SystemID.plot().items():
    value.savefig(f'Assignments/Project/results/{key}.pdf')

forms = [(1,1), (2,1), (1,2), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]
results = SystemID._best_model.analyze_different_u_y_forms(forms)
with open('Assignments/Project/results/best_model_u_y_forms.json', 'w') as f:
    json.dump(results, f, indent=4)

SystemID._best_model.plot_u_y_forms(results).savefig('Assignments/Project/results/best_model_u_y_forms.pdf')

models = ModelUncertainty(SystemID.system_id_collection)
fig = models.plot_residuals()
fig.savefig('Assignments/Project/results/model_uncertainty_residuals.pdf')
