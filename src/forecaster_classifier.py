import pandas as pd


def forecaster_classifier(df,fdict={}):
    
    """
    :param df: columns including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
    :param Delta: Delta value which is a candidate of the optimized Delta
    :param fdict: is a fxn with params
    :return: the df with value label
    """
    for vs,kvs in zip(['p','q'],['vol_past','ret_past']):
    	if vs in fdict['params'].keys():
    		v = int(fdict['params'][vs])
    		df['E'+vs] = df[kvs].rolling(v).mean().copy()


    df['label'] = df.apply(fdict['fxn'],args=(fdict['params']),axis=1)
    
    return df


 def volonly(x,params={'delta':0.2}):
 	 """
    :param x: row including Date, V (seperating training and test samples), ret_past, vol_past, vol_now, vol_future
    :param params: dictionary of delta and vol_name value which is a candidate of the optimized Delta
    :return: integer 1 or -1
    """
 	delta = params['delta']
 	name = params['vol_name']
 	upside = abs(x['vol_now']- x[name]*(1+delta))
 	downside = abs(x['vol_now'] - x[name]*(1-delta))
 	return 1 if upside < downside else -1


 def volandret(x,params={'delta':0.2}):
 	"""
    :param x: row including Date, V(seperating training and test samples), ret_past, vol_past, vol_now, vol_future
    :param params: dictionary of delta and vol_name and ret_name which is a candidate of the optimized Delta
    :return: integer 1 or -1
    """
 	delta = params['delta']
 	v_name = params['vol_name']
 	r_name = params['ret_name']
 	upside1 = abs(x['vol_now'] - (x[v_name]+ x[ret_name])*(1+delta))
	upside2 = abs(x['vol_now'] - (x[v_name]+ x[ret_name])*(1-delta))

	upside = min(upside1,upside2)
	

	downside1 = abs(x['vol_now'] - (x[v_name]*(1+delta)+ x[ret_name]*(1-delta)))
	downside2 = abs(x['vol_now'] - (x[v_name]*(1-delta)+ x[ret_name]*(1+delta)))

	downside = min(downside1,downside2)
	return 1 if upside < downside else -1








