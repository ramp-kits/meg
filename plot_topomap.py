import matplotlib.pyplot as plt
import mne

# this is the script to be loaded from the starting kit. If you want to use it
# standalone you need to provide the X data and import necessary python
# libraries

data_path = mne.datasets.sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
info = mne.read_evokeds(fname)[0].pick('grad').info
fig, axes = plt.subplots(1, 3, figsize=(8, 5))
X_plot = X.iloc[:, :-1]

info_temp = info.copy()
info_temp['bads'] = []
mne.viz.plot_topomap(X_plot.iloc[0].values, info_temp, axes=axes[0],
                     show=False)
mne.viz.plot_topomap(X_plot.iloc[1].values, info_temp, axes=axes[1],
                     show=False)
mne.viz.plot_topomap(X_plot.iloc[2].values, info_temp, axes=axes[2],
                     show=False)
plt.tight_layout()
