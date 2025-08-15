# Original Authors: Irene Garousi-NejadDavid Tarboton
# https://www.hydroshare.org/resource/8d46451858c34b5a8461faab79a53012/
# Modified by: Stanley Akor 14-Oct-2023


'''
Script to download snotel data for a specific set of variables
'''

import sys
import urllib3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.options.mode.chained_assignment = None

def getData(SiteName, SiteID, StateAbb, StartDate, EndDate):
    url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
    url2 = f'{SiteID}:{StateAbb}:SNTL%7Cid=%22%22%7Cname/'
    url3 = f'{StartDate}-10-01,{EndDate}-09-30/'
    url4 = 'TAVG::value,TMIN::value,TMAX::value,PREC::value,PRCP::value,WTEQ::value,SNWD::value?fitToScreen=false'
    url = url1+url2+url3+url4
    
    print(f'Start retrieving data for {SiteName}, {SiteID}')
    
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    data = response.data.decode('utf-8')
    i=0
    for line in data.split("\n"):
        if line.startswith("#"):
            i=i+1
    data = data.split("\n")[i:]
    
    df = pd.DataFrame.from_dict(data)
    df = df[0].str.split(',', expand=True)
    df.rename(columns={0:df[0][0], 
                       1:df[1][0], 
                       2:df[2][0],
                       3:df[3][0],
                       4:df[4][0],
                       5:df[5][0],
                       6:df[6][0],
                       7:df[7][0]}, inplace=True)
    
    df.drop(0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df[f'{SiteName} ({SiteID}) Air Temperature Average (degF)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Air Temperature Average (degF)'])
    df[f'{SiteName} ({SiteID}) Air Temperature Minimum (degF)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Air Temperature Minimum (degF)'])
    df[f'{SiteName} ({SiteID}) Air Temperature Maximum (degF)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Air Temperature Maximum (degF)'])
    df[f'{SiteName} ({SiteID}) Precipitation Accumulation (in) Start of Day Values'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Precipitation Accumulation (in) Start of Day Values'])
    df[f'{SiteName} ({SiteID}) Precipitation Increment (in)'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Precipitation Increment (in)'])
    df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Snow Water Equivalent (in) Start of Day Values'])
    df[f'{SiteName} ({SiteID}) Snow Depth (in) Start of Day Values'] = pd.to_numeric(df[f'{SiteName} ({SiteID}) Snow Depth (in) Start of Day Values'])
    df.to_csv(f'./df_{SiteID}.csv', index=False)

if __name__ == "__main__":
    SiteName = sys.argv[1]
    SiteID = sys.argv[2]
    StateAbb = sys.argv[3]
    StartDate = sys.argv[4]
    EndDate = sys.argv[5]
    
    getData(SiteName, SiteID, StateAbb, StartDate, EndDate)   

