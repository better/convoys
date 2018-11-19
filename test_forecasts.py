import datetime
import pytz
import pandas as pd
from convoys.multi import GeneralizedGamma, Weibull
from convoys.utils import get_arrays
from convoys.plotting import plot_cohorts
import matplotlib.pyplot as plt
from collections import namedtuple

df = pd.read_csv('funded.csv')
df['now'] = datetime.datetime.now(pytz.utc)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['funded_at'] = pd.to_datetime(df['funded_at'])
df = df[df['timestamp'] >= '2017-07-01']

_, months, (G, B, T) = get_arrays(df, unit='days', created='timestamp', converted='funded_at', groups='month')

# plt.figure(figsize=[20,7])
# plot_cohorts(G, B, T, model='kaplan-meier', ci=0.5, groups=months)
# plt.legend()
# plt.show()

ci = False
model = Weibull(ci=bool(ci)) # try with both Weibull and Gamma
model.fit(G, B, T)

accounts_by_month = df.groupby('month').size().to_dict()
model_params = pd.DataFrame(model.base_model.params['map']).to_dict('records')
# print(accounts_by_month)


Forecast = namedtuple('Forecast', ['month', 'ndays', 'num_accounts', 'expected', 'high', 'low', 'model_params'])
forecasts = []
for i, month in enumerate(months):
    num_accounts = accounts_by_month[month]
    for ndays in [30, 60, 90, 100, 120, 150, 180, 210, 240, 270, 300, 330, 365]:
    # for ndays in [100]:

        if ci:
            mean, low, high = (num_accounts * z for z in model.cdf(i, ndays, ci=ci))
            if ndays == 100:
                print(month, ndays, num_accounts, mean, low, high)
            forecasts.append(Forecast(month, ndays, num_accounts, mean, high, low, model_params[i]))

        else:
            mean = num_accounts * model.cdf(i, ndays)
            forecasts.append(Forecast(month, ndays, num_accounts, mean, None, None, model_params[i]))
            # print(month, ndays, num_accounts, mean)
forecasts = pd.DataFrame(forecasts)
forecasts['conv_rate'] = forecasts['expected']/ forecasts['num_accounts']
print(forecasts[forecasts['month']>='2018-06'].pivot(index='ndays', columns='month', values='conv_rate'))
