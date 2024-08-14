from random import random

def ema(l,K=1):
	_ema = [l[0]]
	for e in l[1:]:
		_ema += [_ema[-1] * (1 - 1/K) + e*(1/K)]
	return _ema

##############################################

def direct(x):
	return x

def diff(x):
	return [0]+[a-b for a,b in zip(x[1:], x[:-1])]

def diff_ema(x):
	ema26 = ema(x,26)
	ema12 = ema(x,12)
	return [ema26[i]-ema12[i] for i in range(len(x))]

def macd(x, e):
	e > 0.0
	ema12 = ema(x, e*12)
	ema26 = ema(x, e*26)
	__macd = [a-b for a,b in zip(ema12,ema26)]
	ema9 = ema(__macd, 9*e)
	return [a-b for a,b in zip(__macd, ema9)]

def chiffre(x, __chiffre):
	ret = []
	for _x in x:
		ch = (int(_x)%__chiffre)/__chiffre
		ret += [2*(0.5 - (ch if ch <= 0.5 else 1-ch))]
	return ret

def rsi(prices, n):
	deltas = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
	
	gains = [delta if delta > 0 else 0 for delta in deltas]
	losses = [-delta if delta < 0 else 0 for delta in deltas]

	avg_gain = sum(gains[:n]) / n
	avg_loss = sum(losses[:n]) / n

	rsi = [0] * len(prices)

	for i in range(n, len(prices)):
		if i == n:
			avg_gain = sum(gains[:n]) / n
			avg_loss = sum(losses[:n]) / n
		else:
			avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
			avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n

		if avg_loss == 0:
			rs = float('inf')
			rsi_val = 100
		else:
			rs = avg_gain / avg_loss
			rsi_val = 100 - (100 / (1 + rs))

		rsi[i] = rsi_val

	return rsi

def stoch_rsi(prices, n=14, stoch_n=14):
    rsi_values = rsi(prices, n)
    stoch_rsi = [0] * len(rsi_values)

    for i in range(stoch_n, len(rsi_values)):
        rsi_window = rsi_values[i-stoch_n:i+1]
        rsi_min = min(rsi_window)
        rsi_max = max(rsi_window)
        if rsi_max - rsi_min == 0:
            stoch_rsi[i] = 0
        else:
            stoch_rsi[i] = (rsi_values[i] - rsi_min) / (rsi_max - rsi_min)

    return stoch_rsi