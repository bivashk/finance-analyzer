"""Chronological expense forecasting with a last-month baseline and honest coverage limits."""
from datetime import date
from sklearn.linear_model import LinearRegression


def forecast(monthly, today=None):
    today = today or date.today()
    complete = [r for r in monthly if r['month'] < today.strftime('%Y-%m')]
    if len(complete)<8:
        return {'available':False,'reason':'At least 8 consecutive past months are required; the current month is excluded.'}
    indices = [int(r['month'][:4])*12+int(r['month'][5:]) for r in complete]
    if any(b-a != 1 for a,b in zip(indices,indices[1:])):
        return {'available':False,'reason':'Missing months: confirm data completeness before forecasting.'}
    y = [r['expense_paise']/100 for r in complete]
    x = [[i] for i in range(len(y))]
    errors = {'last_month':[], 'linear_trend':[]}
    for cutoff in range(max(5,len(y)-6),len(y)):
        trend = max(0,float(LinearRegression().fit(x[:cutoff],y[:cutoff]).predict([x[cutoff]])[0]))
        errors['last_month'].append(abs(y[cutoff]-y[cutoff-1]))
        errors['linear_trend'].append(abs(y[cutoff]-trend))
    mae = {name:sum(values)/len(values) for name,values in errors.items()}
    # Keep the baseline unless the trend improves held-out MAE.
    selected = min(mae,key=mae.get)
    predicted = y[-1] if selected=='last_month' else max(0,float(LinearRegression().fit(x,y).predict([[len(y)]])[0]))
    width = max(errors[selected])
    next_index = indices[-1]+1
    return {'available':True,'month':f'{(next_index-1)//12:04d}-{(next_index-1)%12+1:02d}',
            'forecast_inr':round(predicted,2),'selected_model':selected,
            'backtest_mae_inr':{k:round(v,2) for k,v in mae.items()},
            'backtest_windows':len(errors[selected]),
            'illustrative_range_inr':[round(max(0,predicted-width),2),round(predicted+width,2)],
            'limitation':'Range uses the largest recent backtest error, not a calibrated prediction interval. Recorded months may still be incomplete.'}
