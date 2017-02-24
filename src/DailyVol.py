# def DailyVolDF(df, ar):
#     # ar = [NumDays2008,NumDays2009,NumDays2010,NumDays2011,NumDays2012,NumDays2013]
#     # df is the input, which can be file1...file9
#     DailyVols = []
#     for i in range(ar[0]):
#         DailyVols.append(DailyVol(DFList[i], ar[0]))
#     for i in range(ar[1]):
#         DailyVols.append(DailyVol(DFList[i + ar[0]], ar[1]))
#     for i in range(ar[2]):
#         DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1]], ar[2]))
#     for i in range(ar[3]):
#         DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2]], ar[3]))
#     for i in range(ar[4]):
#         DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2] + ar[3]], ar[4]))
#     for i in range(ar[5]):
#         DailyVols.append(DailyVol(DFList[i + ar[0] + ar[1] + ar[2] + ar[3] + ar[4]], ar[5]))
#
#     d = {'Date': df.Date.unique(), 'Volatility_Daily': DailyVols}
#     NewDailyVolDF = pd.DataFrame(d)
#
#     return NewDailyVolDF
