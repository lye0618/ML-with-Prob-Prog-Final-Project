def preprocess():
  data = pd.read_csv("/data_fund.csv")
  fun = data.loc[~data['Indicator Name'].isin(['Common Shares Outstanding','Share Price'])].reset_index(drop=True)
  fun['publish date']=pd.to_datetime(fun['publish date'])
  fun['yyyymm'] = fun['publish date'].map(lambda x: 100*x.year+x.month) 
  fun = pd.pivot_table(fun, values='Indicator Value', index=['Ticker', 'yyyymm'],columns=['Indicator Name'], aggfunc=np.mean).reset_index()

  to_remove = ['Ticker','yyyymm','Avg. Basic Shares Outstanding','Avg. Diluted Shares Outstanding','Total Assets']
  fun_cols = list(fun.columns)
  for elem in to_remove:
    fun_cols.remove(elem)

  fun[fun_cols] = fun[fun_cols].div(fun['Total Assets'], axis=0).reset_index(drop=True)

  features = fun[fun_cols+['yyyymm','Ticker']]
  features = features.dropna().reset_index(drop=True)
  features = features.fillna(0).reset_index(drop=True)

  zscore = lambda x: (x - x.mean()) / x.std()
  final = features[['Ticker','yyyymm']]
  for item in fun_cols:
    temp = features.groupby([features.yyyymm])[item].transform(zscore)
    final = pd.concat([final,temp], axis=1).reset_index(drop=True)

  X_train = final[fun_cols].values
  X_data = torch.tensor(X_train, dtype=torch.float)
  return X_data