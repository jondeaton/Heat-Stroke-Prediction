# Get this figure: fig = py.get_figure("https://plot.ly/~streaming-demos/6/")
# Get this figure's data: data = py.get_figure("https://plot.ly/~streaming-demos/6/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="streamSimpleSensor", fileopt="extend")

# Get figure documentation: https://plot.ly/python/get-requests/
# Add data documentation: https://plot.ly/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('jondeaton', '1FUWee9d7Y0DSRpncEei')
trace1 = {
  "line": {"color": "rgba(31, 119, 180, 0.31)"}, 
  "marker": {"color": "rgba(31, 119, 180, 0.96)"}, 
  "mode": "lines markers", 
  "stream": {
    "maxpoints": 100, 
    "token": "your_stream_token_here"
  }, 
  "type": "scatter"
}


file = "/Users/jonpdeaton/Google Drive/school/present/BIOE 141A_B/BIOE 141b/Heller Lab/2_28_17/all_data_2017.02.28-16.04.58_.csv"
df = pd.read_csv(file)

data = Data([trace1])
layout = {"title": "Heat Stroke Sensor Data"}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)