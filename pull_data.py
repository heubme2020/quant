import os
import ssl
# import time
import certifi
import argparse
import json
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from time import perf_counter
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
args = parser.parse_args()


TOKEN ='6263d89930d3ba4b1329d603814270ad'

URLS = dict(
    profile='https://financialmodelingprep.com/api/v3/profile/',
    financial='https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=' ,
    stock_screener='https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&exchange=',
    income='https://financialmodelingprep.com/api/v3/income-statement/',
    balance='https://financialmodelingprep.com/api/v3/balance-sheet-statement/',
    cashflow='https://financialmodelingprep.com/api/v3/cash-flow-statement/',
    dividend='https://financialmodelingprep.com/api/v3/enterprise-values/',
    price='https://financialmodelingprep.com/api/v3/historical-price-full/',
)

# dividend_data

EXCHANGE = dict(
    shenzhen="SHZ", 
    shanghai="SHH",
    hongkong="HKSE",
    newyork="NYSE",
    nasdaq="NASDAQ",
    brussels="EURONEXT",
    london="LSE",
    tokyo="JPX",
    india="NSE",
    korea="KSC",
    taiwan="Tai",
)


def load_json(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)



def remove_unused(profile):
    unused = [
        "phone", "country", 
        "fullTimeEmployees", 
        "exchangeShortName",
        "isActivelyTrading",
        "price", "beta", "volAvg", "mktCap",
        "lastDiv", "range","changes", "cik",
        "isin", "cusip", "industry", "description",
        "ceo", "sector", "address", "state",
        "zip", "dcfDiff", "dcf", "image", 
        "defaultImage", "isEtf", "isAdr","isFund",
    ]
    for name in unused:
        profile.pop(name)
    return profile


def load_symbol(exchange):
    save_dir = args.save_dir 
    save_path = os.path.join(save_dir, "symbols.npy")

    if os.path.exists(save_path):
        return np.load(save_path)        

    url = URLS.get("financial") + TOKEN
    symbols = load_json(url)
    url = URLS.get("stock_screener") + exchange + '&apikey=' + TOKEN
    stocks = load_json(url)
    symbols_of = [s["symbol"] for s in stocks]
    symbols = np.intersect1d(symbols, symbols_of)
    np.save(save_path, symbols)
    return symbols


def load_stock(symbols):
    save_dir = args.save_dir 
    save_path = os.path.join(save_dir, "stocks.npy")
    if os.path.exists(save_path):
        return np.load(save_path, allow_pickle=True)        
        
    stocks = []
    for symbol in tqdm(symbols):
        url = URLS.get("profile") + symbol + "?apikey=" + TOKEN
        profile = load_json(url)[0]
        profile = remove_unused(profile)

        if len(profile) == 0:
            print(f"No data, {symbol} !!!")
            continue

        stocks.append(profile)  

    np.save(save_path, stocks)
    return stocks  



def load_financial(stocks, fal="income", min_quarter=12, limit=480):
    data = []
    for stock in tqdm(stocks):
        symbol = stock['symbol']
        url = URLS.get(fal)+symbol+f'?period=quarter&limit={limit}&apikey='+TOKEN
        record = load_json(url)
        if len(record) < min_quarter:
            print(f"num_quarter {len(record)} is not enough !!!")
            continue
        data.append(record)
    return data


def load_price(stocks):
    data = []
    channel = ["open", "close", "adjClose", "low", "high", "volume"]
    
    for stock in tqdm(stocks):
        symbol = stock['symbol']    
        ipoDate = stock['ipoDate'] 
        t1 = pd.Timestamp(ipoDate) if ipoDate else pd.Timestamp("19000101")
        t2 = t1.today()
        t1_str = t1.strftime("%Y-%m-%d")
        t2_str = t2.strftime("%Y-%m-%d")
        url = URLS.get("price") + symbol + '?from=' + t1_str +'&to='+ t2_str +'&apikey='+ TOKEN
        records = load_json(url)

        ds = []
        for record in records['historical'][:10]:
            time = pd.to_datetime(record["date"])
            data = np.array([record[k] for k in channel], dtype=np.float32)
            # time channel
            v = xr.DataArray(
                name="price",
                data=data[None],
                dims=["time", "channel"],
                coords=dict(
                    time=[time],
                    channel=channel,
                )
            )
            ds.append(v)
        ds = xr.concat(ds, "time")
        # ds = ds.sortby("time")
        from IPython import embed; embed()
        data.append(record)

    return data
    

start = perf_counter()
print("Load symbol ...")
symbols = load_symbol(EXCHANGE.get("shanghai"))
print(f"Num symbols: {len(symbols)}")
load_time = perf_counter() - start
print(f"Load symbol take, {load_time:.3f} secs")


print("Load stock ...")
stocks = load_stock(symbols[:1])
print(f"Num stocks: {len(stocks)}")
load_time = perf_counter() - start
print(f"Load stock take, {load_time:.3f} secs")

# incomes = load_financial(stocks, fal="income")
# balances = load_financial(stocks, fal="balance")
# cashflows = load_financial(stocks, fal="cashflow")
# dividends = load_financial(stocks, fal="dividend")
prices = load_price(stocks)






