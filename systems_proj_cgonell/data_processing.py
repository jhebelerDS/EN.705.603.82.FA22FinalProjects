import sage_data_client
import pandas as pd

# query and load data into pandas data frame
w02C_traffic = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "traffic.state.*",
        "vsn": "W02C"
    }
)
w02C_envcount = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "env.count.*",
        "vsn": "W02C"
    }
)
w07A_traffic = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "traffic.state.*",
        "vsn": "W07A"
    }
)
w07A_envcount = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "env.count.*",
        "vsn": "W07A"
    }
)
w07B_traffic = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "traffic.state.*",
        "vsn": "W07B"
    }
)
w07B_envcount = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "env.count.*",
        "vsn": "W07B"
    }
)
w026_traffic = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "traffic.state.*",
        "vsn": "W026"
    }
)
w026_envcount = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "env.count.*",
        "vsn": "W026"
    }
)
w079_traffic = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "traffic.state.*",
        "vsn": "W079"
    }
)
w079_envcount = sage_data_client.query(
    start="-4912h",
    filter={
        "name": "env.count.*",
        "vsn": "W079"
    }
)

nodes_count = [w02C_envcount, w07A_envcount, w07B_envcount, w026_envcount, w079_envcount]
count_data = pd.concat(nodes_count)

nodes_traffic = [w02C_traffic, w07A_traffic, w07B_traffic, w026_traffic, w079_traffic]
traffic_data = pd.concat(nodes_traffic)

node_data = [count_data, traffic_data]
data = pd.concat(node_data)
data['month'] = data['timestamp'].dt.month
data['day'] = data['timestamp'].dt.day
data['year'] = data['timestamp'].dt.year
data['hour'] = data['timestamp'].dt.hour
data['min'] = data['timestamp'].dt.minute
data['sec'] = data['timestamp'].dt.second
data = data.rename(columns={"meta.vsn": "nodes"})

data.to_csv('api_data.csv', index=False)



