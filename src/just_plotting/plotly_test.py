#!/usr/bin/env python

import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import time

username = 'jondeaton' # Plotly Username
api_key = '1FUWee9d7Y0DSRpncEei' # Plotly API Key
plotly.tools.set_credentials_file(username=username, api_key=api_key)

plotly_filename = 'My Plot'

data = Data([Scatter(x=[0,0], y=[0,0])])
plot_url = py.plot(data, filename=plotly_filename)
print("Plotly URL: %s" % plot_url)

for i in range(10):
    x = np.arange(i)
    y = pow(x, 2)
    data = Data([Scatter(x=x, y=y)])
    # Take 2: extend the traces on the plot with the data in the order supplied.
    plot_url = py.plot(data, filename=plotly_filename, auto_open=False, fileopt='extend')
    time.sleep(10)
