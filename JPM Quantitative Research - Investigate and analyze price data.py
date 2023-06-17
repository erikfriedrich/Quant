# FIRST DRAFT, THIS IS NOT FINISHED - WILL BE PUT INTO A JUPYTER NOTEBOOK SOON


# import everything we'll need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


# read in the csv-file containing the data
data = pd.read_csv("/Users/erikfriedrich/Downloads/Nat_Gas.csv")

data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%f/%y')
data['Dates'] = data['Dates'].dt.date
data = data.set_index('Dates')

start_date = data.index.min()
end_date = data.index.max()
daily_range = pd.date_range(start=start_date, end=end_date, freq="D")
daily_data = pd.DataFrame(index=daily_range)

merged_df = daily_data.merge(data, how="left", left_index=True, right_index=True)
merged_df["Prices"] = merged_df['Prices'].interpolate(method='linear', limit_direction='backward')
merged_df = merged_df.reset_index()
merged_df.columns = ["ds", "y"]

m = Prophet()
m.fit(merged_df)
future = m.make_future_dataframe(periods=365)
prediction = m.predict(future)
#m.plot(prediction)

#plt.title("Prediction of Natural Gas Prices 365 into the future")
#plt.xlabel("Date")
#plt.ylabel("Nat Gas Price")
#plt.show()
print(prediction)
