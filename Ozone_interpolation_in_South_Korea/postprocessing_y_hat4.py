import pandas as pd
import numpy as np

months = ["2021-01","2021-02","2021-03","2021-04","2021-05","2021-06","2021-07","2021-08","2021-09","2021-10","2021-11","2021-12"]
df = pd.read_csv('C:/Users/ATMOS/Desktop/data_fin/y_hat/y_hat_Ulsan.csv')

# y_true
# y_imputed
# O3_abs

for month in months:
    df1 = df[df["month"] == month]
    o3_values1 = df1["y_true"].dropna()
    mean1 = np.mean(o3_values1)
    percentiles1 = np.percentile(o3_values1, [10, 25, 50, 75, 90, 95, 99])
    q751, q251 = np.percentile(o3_values1, [75 ,25])
    iqr1 = q751 - q251
    low_fence1 = q251 - 1.5 * iqr1
    high_fence1 = q751 + 1.5 * iqr1
    
    print("")
    print(month)
    print("")
    print(mean1)
    print(low_fence1)
    print(percentiles1[0])
    print(percentiles1[1])
    print(percentiles1[2])
    print(percentiles1[3])
    print(percentiles1[4])
    print(percentiles1[5])
    print(percentiles1[6])
    print(high_fence1)
    print("")

    
