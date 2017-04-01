import pandas as pd


def forecaster_classifier(df,drop=False,**kwargs):
	"""
	:param df: columns including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
	:param Delta: Delta value which is a candidate of the optimized Delta
	:param kwargs: is a fxn with params
	:return: the df with value label
	"""
	for vs,kvs,name in zip(['p','q'],['vol_past','ret_past'],['vol_name','ret_name']):
		if vs in kwargs['params'].keys():
			v = int(kwargs['params'][vs])
			df['E'+vs] = df[kvs].rolling(v).mean().copy()
			kwargs['params'][name] = 'E'+vs

	df.dropna(inplace=True,axis=0)

	df['label'] = df.apply(kwargs['fxn'],**kwargs['params'],axis=1)

	#clean up
	if drop:
		for name in ['Ep','Eq']: 
			try: 
				df.drop(name,axis=1,inplace=True)
			except:
				print((10*"# "+"Selected method does not require rolling averages"))

	return df

def volonly(x,**kwargs):
	"""
	:param x: row including Date, V (seperating training and test samples), ret_past, vol_past, vol_now, vol_future
	:param kwargs: dictionary of delta and vol_name value which is a candidate of the optimized Delta
	:return: integer 1 or -1
	"""
	delta = kwargs['delta']
	name = kwargs['vol_name']
	upside = abs(x['vol_now']- x[name]*(1+delta))
	downside = abs(x['vol_now'] - x[name]*(1-delta))
	return 1 if upside < downside else -1


def volandret(x,**kwargs):
	"""
	:param x: row including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
	:param params: dictionary of delta and vol_name and ret_name which is a candidate of the optimized Delta
	:return: integer 1 or -1
	"""

	delta = kwargs['delta']
	v_name = kwargs['vol_name']
	r_name = kwargs['ret_name']

	upside1 = abs(x['vol_now'] - (x[v_name]+ x[r_name])*(1+delta))
	upside2 = abs(x['vol_now'] - (x[v_name]+ x[r_name])*(1-delta))

	upside = min(upside1,upside2)


	downside1 = abs(x['vol_now'] - (x[v_name]*(1+delta)+ x[r_name]*(1-delta)))
	downside2 = abs(x['vol_now'] - (x[v_name]*(1-delta)+ x[r_name]*(1+delta)))

	downside = min(downside1,downside2)
	return 1 if upside < downside else -1








