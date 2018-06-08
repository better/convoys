from matplotlib import pyplot
import pickle
import convoys.plotting
import convoys.utils


f = open('marriage.pickle', 'rb')
print('loading data')
df = pickle.load(f)
print('converting to arrays')
_, (G, B, T) = convoys.utils.get_arrays(df, groups='state', created='born', converted='married')
print('plotting')
convoys.plotting.plot_cohorts(G, B, T, model='generalized-gamma')
pyplot.savefig('marriage.png')
